#!/usr/bin/env python
# run_neuroxai_eegnet.py  ——  NeuroXAI 选通道 + EEGNet 分类 (自动适配通道数)

import argparse, json, warnings, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import GroupKFold
from tqdm import trange, tqdm

from neuroxai.explanation import BrainExplainer, GlobalBrainExplainer

# ============= 基本超参 =============
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
else:
    orig_names = [f"Ch{i}" for i in range(n_total)]

CHAN_NAMES = [orig_names[i] for i in keep_idx]
N_CH       = len(CHAN_NAMES)
print(f"可用通道数 = {N_CH}")

# ---------- 预处理 4-40 Hz + 60 Hz Notch ----------
bp_b, bp_a = butter(4, [4,40], fs=FS, btype="band")
nt_b, nt_a = iirnotch(60, 30, fs=FS)

def preprocess(sig):                         # sig:(n_total,T)
    sig = sig[keep_idx]                     # (N_CH,T)
    sig = filtfilt(nt_b, nt_a, sig, axis=1)
    sig = filtfilt(bp_b, bp_a, sig, axis=1)
    sig -= sig.mean(1, keepdims=True); sig /= sig.std(1, keepdims=True) + 1e-6
    return sig.astype(np.float32)

def slide(sig):
    out=[]
    for st in range(0, sig.shape[1]-WIN+1, STEP):
        out.append(sig[:, st:st+WIN])
    return np.stack(out)                    # (n_win,C,T)

# ---------- 读取 & 平衡 trial ----------
def load_trials():
    trials, labels = [], []
    for f in FILES:
        m   = loadmat(f, simplify_cells=True)
        key = next(k for k in m if k.endswith("last_beep"))
        for cls, tset in enumerate(m[key]):
            for tr in tset:
                trials.append(preprocess(tr)); labels.append(cls)
    trials, labels = map(np.asarray,(trials,labels))
    # 平衡两类
    i0,i1 = np.where(labels==0)[0], np.where(labels==1)[0]
    n = min(len(i0),len(i1)); keep = np.sort(np.hstack([i0[:n], i1[:n]]))
    return trials[keep], labels[keep]

# ---------- EEGNet ----------
class EEGNet(nn.Module):
    def __init__(self, C, T, n_cls=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1,8,(1,64),padding=(0,32),bias=False)
        self.bn1   = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,16,(C,1),groups=8,bias=False)
        self.bn2   = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1,4)); self.drop2 = nn.Dropout(.25)
        self.conv3 = nn.Conv2d(16,16,(1,16),padding=(0,8),bias=False)
        self.bn3   = nn.BatchNorm2d(16)
        self.pool3 = nn.AvgPool2d((1,8)); self.drop3 = nn.Dropout(.25)
        out_len = ((T+64-1)//1 - 63)//4
        out_len = ((out_len+16-1)//1 -15)//8
        self.fc = nn.Linear(16*out_len, n_cls)
    def forward(self,x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x); x = self.drop2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x); x = self.drop3(x)
        return self.fc(x.flatten(1))

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
    pred_trial={i:max(set(v), key=v.count) for i,v in vote.items()}
    return np.mean([pred_trial[i]==int(y_np[np.where(g_np==i)[0][0]])
                    for i in pred_trial])

# ---------- NeuroXAI 计算权重 ----------
def neuroxai_importance(baseline, trials, labels, n_samples):
    def clf(batch):
        batch = torch.tensor(batch[:,None,:,:], dtype=torch.float32, device=DEVICE)
        with torch.no_grad(): out = baseline(batch)
        return torch.softmax(out,1).cpu().numpy()

    brain=BrainExplainer(25,['short','long'])
    gexp = GlobalBrainExplainer(brain)
    gexp.explain_instance(trials, labels, clf, num_samples=n_samples)
    imp = [gexp.explain_global_channel_importance().get(i,0.0) for i in range(N_CH)]
    return np.asarray(imp, dtype=np.float32)

# ---------- 主入口 ----------
def main(k_list, n_samples):
    print("Torch device:", DEVICE)
    trials, labels = load_trials()
    print("Sliding windows …")
    X_win, Y, G, gid = [], [], [], 0
    for sig, lab in tqdm(zip(trials,labels), total=len(trials)):
        seg = slide(sig)
        X_win.append(seg)
        Y.extend([lab]*len(seg)); G.extend([gid]*len(seg)); gid += 1
    X_all = np.concatenate(X_win)             # (N,C,T)
    X_all = torch.tensor(X_all[:,None,:,:], device=DEVICE)
    Y_all = torch.tensor(Y, device=DEVICE)
    Y_np, G_np = np.asarray(Y), np.asarray(G)

    # ---- 训练全通道基线，供 NeuroXAI ----
    print("Train baseline EEGNet …")
    base_net = train_eegnet(X_all, Y_all, N_CH, epochs=EPOCHS//2)

    print("Compute NeuroXAI channel importance …")
    imp = neuroxai_importance(base_net, trials, labels, n_samples)
    order = np.argsort(-imp)                 # importance 降序

    gkf = GroupKFold(10)
    results={}
    for K in k_list:
        sel = order[:K]
        sel_names=[CHAN_NAMES[i] for i in sel]
        print(f"\n==> Top-{K} channels: {sel_names}")

        X = X_all[:,:,sel,:]                 # 选通道
        acc=[]
        for tr,te in gkf.split(X, Y_np, groups=G_np):
            net=train_eegnet(X[tr], Y_all[tr], K, epochs=EPOCHS//2)
            acc.append(eval_eegnet(net, X[te], Y_np, G_np[te]))
        acc_mean, acc_std = float(np.mean(acc)), float(np.std(acc))
        results[K]=(acc_mean, acc_std)
        print(f"Top-{K} 10-fold acc: {acc_mean:.3f} ± {acc_std:.3f}")

    # ----- 保存 -----
    out=ROOT/"results/neuroxai_eegnet_curve.json"
    json.dump({
        "k_list":k_list,
        "order":order.tolist(),
        "chan_names":[CHAN_NAMES[i] for i in order],
        "accuracy":{str(k):[results[k][0],results[k][1]] for k in k_list}
    }, open(out,"w"), indent=2)
    print("Saved:", out)

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