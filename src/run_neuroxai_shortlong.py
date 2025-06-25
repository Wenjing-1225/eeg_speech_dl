#!/usr/bin/env python
# run_neuroxai_eegnet.py  ——  FastICA → NeuroXAI 选通道 → EEGNet 分类

import argparse, json, warnings, random, time
from pathlib import Path

import numpy as np, torch, torch.nn as nn
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.decomposition import FastICA
from sklearn.model_selection import GroupKFold
from tqdm import tqdm, trange

from neuroxai.explanation import BrainExplainer, GlobalBrainExplainer

# ============= 全局超参 =============
SEED      = 0
FS        = 256
WIN_S     = 2.0;   WIN  = int(WIN_S * FS)
STEP_S    = 0.5;   STEP = int(STEP_S * FS)
EPOCHS    = 150
BATCH     = 128
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data/Short_Long_words"
FILES = sorted(f for f in DATA.glob("*.mat") if "_8s" not in f.name)

# ---------- 自动确定通道 ----------
first_mat = loadmat(FILES[0], simplify_cells=True)
key       = next(k for k in first_mat if k.endswith("last_beep"))
n_total   = first_mat[key][0][0].shape[0]

DROP_FIXED = {0, 9, 32, 63}
drop_id    = DROP_FIXED & set(range(n_total))
keep_idx   = [i for i in range(n_total) if i not in drop_id]

if "ch_names" in first_mat:
    orig_names = [str(s).strip() for s in first_mat["ch_names"]][:n_total]
else:                                   # 若无名称就占位
    orig_names = [f"Ch{i}" for i in range(n_total)]

CHAN_NAMES = [orig_names[i] for i in keep_idx]
N_CH       = len(CHAN_NAMES)
print(f"可用通道数 = {N_CH}")
# -----------------------------------

# ---------- 预处理 4-40 Hz + 60 Hz Notch ----------
bp_b, bp_a = butter(4, [4, 40], fs=FS, btype="band")
nt_b, nt_a = iirnotch(60, 30, fs=FS)

def bandpass_notch(sig):                     # (N_CH_full, T)
    sig = sig[keep_idx]                     # (N_CH, T)
    sig = filtfilt(nt_b, nt_a, sig, axis=1)
    sig = filtfilt(bp_b, bp_a, sig, axis=1)
    sig -= sig.mean(1, keepdims=True)
    sig /= sig.std (1, keepdims=True) + 1e-6
    return sig.astype(np.float32)

def slide(sig):
    out=[]
    for st in range(0, sig.shape[1]-WIN+1, STEP):
        out.append(sig[:, st:st+WIN])
    return np.stack(out)                    # (n_win,C,T)

# ---------- 读 & 预处理所有 trial ----------
def load_trials():
    trials, labels = [], []
    for f in FILES:
        m   = loadmat(f, simplify_cells=True)
        key = next(k for k in m if k.endswith("last_beep"))
        for cls, tset in enumerate(m[key]):
            for tr in tset:
                trials.append(bandpass_notch(tr)); labels.append(cls)
    trials, labels = map(np.asarray,(trials,labels))
    # 类别平衡
    i0,i1 = np.where(labels==0)[0], np.where(labels==1)[0]
    n = min(len(i0),len(i1)); keep = np.sort(np.hstack([i0[:n], i1[:n]]))
    return trials[keep], labels[keep]

# ---------- 一次性 FastICA 去伪迹 ----------
def apply_global_ica(trials, n_comp=None):
    """
    n_comp : int | None
        - None (默认) = 保持原始通道维数
        - int  = 降到固定分量数（<= N_CH）
    """
    print("Fit & apply FastICA …")
    C, T = trials[0].shape
    Xcat = np.concatenate(trials, axis=1).T        # (all_time , C)

    if n_comp is None:                 # 保留全部维度
        n_comp = C                     # ＝N_CH；满足新版本 API

    ica = FastICA(
        n_components=n_comp,
        whiten='unit-variance',
        max_iter=300,
        random_state=SEED
    )

    _ = ica.fit_transform(Xcat)        # 拟合混合矩阵

    transformed = []
    for sig in tqdm(trials, desc="ICA transform"):
        transformed.append(ica.transform(sig.T).T.astype(np.float32))
    return np.asarray(transformed)     # (N_trial, C, T)

# ---------- EEGNet 动态 fc ----------
# ---------- EEGNet (auto‐shape) ----------
class EEGNet(nn.Module):
    def __init__(self, C:int, T:int, n_cls:int = 2):
        super().__init__()
        # ----- block-1 -----
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.bn1   = nn.BatchNorm2d(8)

        # ----- block-2 -----
        self.conv2 = nn.Conv2d(8, 16, (C, 1), groups=8, bias=False)
        self.bn2   = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 4))
        self.drop2 = nn.Dropout(0.25)

        # ----- block-3 -----
        self.conv3 = nn.Conv2d(16, 16, (1, 16), padding=(0, 8), bias=False)
        self.bn3   = nn.BatchNorm2d(16)
        self.pool3 = nn.AvgPool2d((1, 8))
        self.drop3 = nn.Dropout(0.25)

        # ----- 自动计算展平维度 -----
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)      # (B,1,C,T)
            feat  = self._forward_features(dummy)
            in_dim = feat.shape[1]               # = 640 in 你当前设置
        self.fc = nn.Linear(in_dim, n_cls)

    # 把卷积流程抽成私有函数
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x); x = self.drop2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x); x = self.drop3(x)
        return x.flatten(1)                      # (B, feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        return self.fc(x)

# ---------- 训练 / 评估 ----------
def train_eegnet(X, y, C, epochs=EPOCHS, lr=1e-3):
    net  = EEGNet(C, WIN).to(DEVICE)
    opt  = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)
    loss = nn.CrossEntropyLoss()
    net.train()
    for _ in trange(epochs, leave=False):
        idx = torch.randperm(len(X), device=DEVICE)
        for beg in range(0,len(idx),BATCH):
            sl = idx[beg:beg+BATCH]
            opt.zero_grad()
            l  = loss(net(X[sl]), y[sl]); l.backward(); opt.step()
    return net

def eval_eegnet(net, X, y_np, g_np):
    net.eval(); pred=[]
    with torch.no_grad():
        for beg in range(0,len(X),BATCH):
            pred.append(net(X[beg:beg+BATCH]).argmax(1).cpu())
    pred = torch.cat(pred).numpy(); vote={}
    for p,i in zip(pred,g_np): vote.setdefault(i,[]).append(p)
    trial_pred={i:max(set(v), key=v.count) for i,v in vote.items()}
    return np.mean([trial_pred[i]==int(y_np[np.where(g_np==i)[0][0]])
                    for i in trial_pred])

# ---------- NeuroXAI ----------
# ---------- NeuroXAI 计算权重 ----------
def neuroxai_importance(baseline, trials, labels, n_samples):
    """
    baseline : 已训练好的全通道 EEGNet
    trials   : (n_trial, C, T_full)
    """
    def clf(batch_np):
        """NeuroXAI 要求的 classifier_fn —— 先对齐到 WIN 再推理"""
        # batch_np : (B, C, T_full)
        C, T_full = batch_np.shape[1], batch_np.shape[2]

        # ① 裁剪 / 补零到 WIN (=512 sample)
        if T_full > WIN:                        # 裁中间一段
            st = (T_full - WIN) // 2
            batch_np = batch_np[:, :, st:st+WIN]
        elif T_full < WIN:                      # 尾部补 0
            pad = np.zeros((batch_np.shape[0], C, WIN-T_full),
                           dtype=batch_np.dtype)
            batch_np = np.concatenate([batch_np, pad], axis=2)

        # ② (B,1,C,T) → baseline
        tensor = torch.tensor(batch_np[:, None, :, :],
                              dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            out = baseline(tensor)              # (B,2)
        return torch.softmax(out, dim=1).cpu().numpy()

    # ======== 调用 NeuroXAI ========
    brain = BrainExplainer(kernel_width=25, class_names=['short','long'])
    gexp  = GlobalBrainExplainer(brain)
    gexp.explain_instance(trials, labels, clf,
                          num_samples=n_samples)

    imp = [gexp.explain_global_channel_importance().get(i, 0.0)
           for i in range(N_CH)]
    return np.asarray(imp, dtype=np.float32)

# ---------- 主入口 ----------
def main(k_list, n_samples):
    print("Torch device:", DEVICE)
    t0=time.time()
    trials_raw, labels = load_trials()
    trials = apply_global_ica(trials_raw)           # 加 ICA
    print(f"加载 & ICA 用时 {time.time()-t0:.1f}s")

    # ---- 滑窗 ----
    print("Sliding windows …")
    X_win, Y, G, gid = [], [], [], 0
    for sig, lab in tqdm(zip(trials,labels), total=len(trials)):
        seg = slide(sig)
        X_win.append(seg)
        Y.extend([lab]*len(seg)); G.extend([gid]*len(seg)); gid += 1
    X_all = torch.tensor(np.concatenate(X_win)[:,None,:,:], device=DEVICE)
    Y_all = torch.tensor(Y, device=DEVICE)
    Y_np, G_np = np.asarray(Y), np.asarray(G)

    # ---- baseline for NeuroXAI ----
    print("Train baseline EEGNet …")
    base_net = train_eegnet(X_all, Y_all, N_CH, epochs=EPOCHS//2)

    print("Compute NeuroXAI channel importance …")
    imp = neuroxai_importance(base_net, trials, labels, n_samples)
    order = np.argsort(-imp)

    gkf = GroupKFold(10)
    results={}
    for K in k_list:
        sel = order[:K]; sel_names=[CHAN_NAMES[i] for i in sel]
        print(f"\n==> Top-{K} channels: {sel_names}")

        X = X_all[:,:,sel,:]
        acc=[]
        for tr,te in gkf.split(X, Y_np, groups=G_np):
            net=train_eegnet(X[tr], Y_all[tr], K, epochs=EPOCHS//2)
            acc.append(eval_eegnet(net, X[te], Y_np, G_np[te]))
        m,s = float(np.mean(acc)), float(np.std(acc))
        print(f"Top-{K} 10-fold acc: {m:.3f} ± {s:.3f}")
        results[K]=(m,s)

    # ----- 保存 -----
    out=ROOT/"results/neuroxai_eegnet_curve.json"
    json.dump({
        "k_list":k_list,
        "order":order.tolist(),
        "chan_names":[CHAN_NAMES[i] for i in order],
        "accuracy":{str(k):[results[k][0],results[k][1]] for k in k_list}
    }, open(out,"w"), indent=2)
    print("✔ 结果已保存到", out)

# ---------- CLI ----------
if __name__=="__main__":
    warnings.filterwarnings("ignore")
    pa=argparse.ArgumentParser()
    pa.add_argument("--k", type=int, nargs='+', default=[4,8,16,32,60],
                    help="想评估的 Top-K 通道数列表")
    pa.add_argument("--n_samples", type=int, default=1000,
                    help="NeuroXAI 随机扰动样本数")
    args=pa.parse_args()
    main(args.k, args.n_samples)