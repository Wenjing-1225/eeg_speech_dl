#!/usr/bin/env python
# run_neuroxai_shortlong.py —— CAN-Lite + log-STFT + NeuroXAI（自动适配通道数）

import argparse, json, warnings, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.model_selection import GroupKFold
from tqdm import trange, tqdm

from neuroxai.explanation import BrainExplainer, GlobalBrainExplainer

# ========= 基本超参 =========
SEED   = 0
FS     = 256
WIN_S  = 3.0;   WIN  = int(WIN_S * FS)
STEP_S = 0.25;  STEP = int(STEP_S * FS)

N_FFT, HOP = 64, 32                 # 33 频点, 2 Hz 分辨率
EPOCHS      = 160                  # 交叉验证内 → EPOCHS//2
BATCH       = 256
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data/Short_Long_words"
FILES = sorted(f for f in DATA.glob("*.mat") if "_8s" not in f.name)

# ========= 根据首个 .mat 文件自动确定通道 =========
first_mat = loadmat(FILES[0], simplify_cells=True)
key       = next(k for k in first_mat if k.endswith("last_beep"))
n_total   = first_mat[key][0][0].shape[0]

DROP_FIXED = {0, 9, 32, 63}                 # 想排除的原始序号
drop_id    = DROP_FIXED & set(range(n_total))
keep_idx   = [i for i in range(n_total) if i not in drop_id]

if "ch_names" in first_mat:
    orig_names = [str(s).strip() for s in first_mat["ch_names"]][:n_total]
else:                                        # 没有名字就占位
    orig_names = [f"Ch{i}" for i in range(n_total)]

CHAN_NAMES = [orig_names[i] for i in keep_idx]
N_CH       = len(CHAN_NAMES)
print(f"可用通道数 = {N_CH}")

# ========= 预处理：去趋势 + 2-32 Hz 带通 =========
b_bp, a_bp = butter(4, [2, 32], fs=FS, btype="band")

def preprocess(sig):                          # sig:(n_total,T)
    sig = sig[keep_idx]                       # 保留通道
    sig = filtfilt(b_bp, a_bp, sig, axis=1)
    sig -= sig.mean(1, keepdims=True)
    sig /= sig.std (1, keepdims=True) + 1e-6
    return sig.astype(np.float32)             # (N_CH,T)

# ========= STFT & Augment =========
_window = torch.hann_window(N_FFT, device=DEVICE)
F_BINS  = N_FFT // 2 + 1                      # 33

def stft_tensor(x_np: np.ndarray) -> torch.Tensor:
    x = torch.tensor(x_np, device=DEVICE)                 # (C,T)
    s = torch.stft(x, N_FFT, hop_length=HOP, window=_window,
                   return_complex=False)                  # (C,F,T',2)
    mag = torch.sqrt(s.pow(2).sum(-1) + 1e-6)
    return torch.log1p(mag)                               # (C,F,T')

def augment_spec(spec: torch.Tensor) -> torch.Tensor:
    """
    SpecAugment : 频率 / 时间随机遮挡
    既可处理 3-维 (C,F,T) ，也可处理 4-维 (B,C,F,T)
    """
    if spec.dim() == 3:                       # 单样本
        spec = spec.unsqueeze(0)              # 变成 (1,C,F,T)

    B, C, F, T = spec.shape
    for i in range(B):
        # ---- 频率 Mask ----
        f0 = np.random.randint(0, max(1, F-4))
        f_w = np.random.randint(2, 5)
        spec[i, :, f0:f0+f_w, :] = 0

        # ---- 时间 Mask ----
        t0 = np.random.randint(0, max(1, T-8))
        t_w = np.random.randint(4, 9)
        spec[i, :, :, t0:t0+t_w] = 0

    return spec if spec.size(0) > 1 else spec.squeeze(0)

def slide_stft(sig):
    wins=[]
    for st in range(0, sig.shape[1]-WIN+1, STEP):
        wins.append(stft_tensor(sig[:, st:st+WIN]).cpu())
    return torch.stack(wins)                                # (n,C,F,T')

# ========= 读数据 =========
def load_trials():
    trials, labels = [], []
    for f in FILES:
        mat  = loadmat(f, simplify_cells=True)
        key  = next(k for k in mat if k.endswith("last_beep"))
        for cls, tset in enumerate(mat[key]):
            for tr in tset:
                trials.append(preprocess(tr)); labels.append(cls)
    trials, labels = map(np.asarray, (trials, labels))
    # 类别平衡
    i0, i1 = np.where(labels==0)[0], np.where(labels==1)[0]
    n = min(len(i0), len(i1)); keep = np.sort(np.hstack([i0[:n], i1[:n]]))
    return trials[keep], labels[keep]

# ========= CAN-Lite =========
class CANLite(nn.Module):
    def __init__(self, heads:int=8, cls:int=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (1,5), padding=(0,2)), nn.BatchNorm2d(32), nn.ELU(),
            nn.Conv2d(32,64,(1,5), padding=(0,2)), nn.BatchNorm2d(64), nn.ELU(),
            nn.AvgPool2d((1,2)), nn.Dropout(0.3)          # F ➜ F/2
        )
        self.att  = nn.MultiheadAttention(64, heads, batch_first=True, dropout=0.2)
        self.fc   = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, cls))

    def forward(self, x):                   # x:(B,C,F,T)
        b,c,f,t = x.shape
        x = x.view(b,1,c,f*t)
        x = self.conv(x)                    # (B,64,C,F/2*T)
        x = x.mean(2)                       # (B,64,seq)
        x = x.permute(0,2,1)                # (B,seq,64)
        x,_ = self.att(x, x, x)
        return self.fc(x.mean(1))

# ========= 训练 / 评估 =========
def train_can(X, y_cpu, epochs=EPOCHS, lr=3e-3):
    net   = CANLite().to(DEVICE)
    opt   = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 40)
    crit  = nn.CrossEntropyLoss()
    net.train()
    for _ in trange(epochs, leave=False):
        idx = torch.randperm(len(X))
        for beg in range(0, len(idx), BATCH):
            sl = idx[beg:beg+BATCH]
            xb = augment_spec(X[sl].clone()).to(DEVICE, non_blocking=True)
            yb = y_cpu[sl].to(DEVICE)
            opt.zero_grad(); loss = crit(net(xb), yb); loss.backward(); opt.step()
        sched.step()
    return net

def eval_can(net, X, y_np, g_np):
    net.eval(); pred=[]
    with torch.no_grad():
        for beg in range(0, len(X), BATCH):
            pred.append(net(X[beg:beg+BATCH].to(DEVICE)).argmax(1).cpu())
    pred = torch.cat(pred).numpy(); vote={}
    for p,i in zip(pred, g_np): vote.setdefault(i,[]).append(p)
    pred_trial = {i:max(set(v), key=v.count) for i,v in vote.items()}
    return np.mean([pred_trial[i]==int(y_np[np.where(g_np==i)[0][0]])
                    for i in pred_trial])

# ========= NeuroXAI =========
def channel_importance(baseline, trials, labels, n_samples=1000):
    keep = np.random.choice(len(trials), int(0.4*len(trials)), replace=False)
    trials_s, labels_s = trials[keep], labels[keep]

    def clf(batch_np):
        batch = torch.tensor(batch_np, dtype=torch.float32, device=DEVICE)
        with torch.no_grad(): out = baseline(batch)
        return torch.softmax(out, 1).cpu().numpy()

    brain = BrainExplainer(25, ['short','long'])
    gexp  = GlobalBrainExplainer(brain)
    gexp.explain_instance(trials_s, labels_s, clf,
                          num_samples=min(n_samples, 1000))

    imp = [gexp.explain_global_channel_importance().get(i, 0.0)
           for i in range(N_CH)]
    return np.asarray(imp, dtype=np.float32)

# ========= 主流程 =========
def main(k_top, n_samples):
    print(f"Torch device: {DEVICE}   n_fft={N_FFT}  channels={N_CH}")
    print("① 读取 trial 并做 STFT …")
    trials, labels = load_trials()

    X_list, Y, G, gid = [], [], [], 0
    for sig, lab in tqdm(zip(trials, labels), total=len(trials)):
        w = slide_stft(sig); X_list.append(w)
        Y.extend([lab]*len(w)); G.extend([gid]*len(w)); gid += 1
    X     = torch.cat(X_list)          # (B,C,F,T')
    Y_t   = torch.tensor(Y)
    Y_np  = np.asarray(Y); G_np = np.asarray(G)

    # —— Baseline CAN 10-fold
    print("② Baseline CAN 10-fold …")
    gkf = GroupKFold(10); acc_base=[]
    for tr, te in gkf.split(X, Y_np, groups=G_np):
        net = train_can(X[tr], Y_t[tr], epochs=EPOCHS//2)
        acc_base.append(eval_can(net, X[te], Y_np, G_np[te]))
    print(f"Baseline (all {N_CH} ch): {np.mean(acc_base):.3f} ± {np.std(acc_base):.3f}")

    # —— NeuroXAI 选择电极
    print("③ 计算 NeuroXAI 权重 …")
    pre_net = train_can(X, Y_t, epochs=40)          # 轻量预训练
    imp     = channel_importance(pre_net, trials, labels, n_samples)
    sel_idx = np.argsort(-imp)[:k_top]
    sel_names = [CHAN_NAMES[i] for i in sel_idx]
    print(f"Top-{k_top} 电极:", sel_names)

    rand_idx = np.random.choice(N_CH, k_top, replace=False)

    def cv(mask_idx):
        Xi = X[:,:,mask_idx,:]
        acc=[]
        for tr, te in gkf.split(Xi, Y_np, groups=G_np):
            net = train_can(Xi[tr], Y_t[tr], epochs=EPOCHS//2)
            acc.append(eval_can(net, Xi[te], Y_np, G_np[te]))
        return np.mean(acc), np.std(acc)

    print("④ NeuroXAI-K 10-fold …")
    acc_neu = cv(sel_idx)
    print("⑤ Random-K 10-fold …")
    acc_rnd = cv(rand_idx)

    print(f"NeuroXAI-{k_top}: {acc_neu[0]:.3f} ± {acc_neu[1]:.3f}")
    print(f"Random-{k_top}:  {acc_rnd[0]:.3f} ± {acc_rnd[1]:.3f}")

    out = ROOT / f"results/CAN_neuroxai_vs_random_{k_top}.json"
    out.parent.mkdir(exist_ok=True)
    json.dump({
        "k": k_top,
        "names_neuro": sel_names,
        "acc_base":  [float(np.mean(acc_base)), float(np.std(acc_base))],
        "acc_neuro": list(map(float, acc_neu)),
        "acc_rand":  list(map(float, acc_rnd))
    }, open(out, "w"), indent=2)
    print("✔ 结果已保存到", out)

# ========= CLI =========
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pa = argparse.ArgumentParser()
    pa.add_argument("--k", type=int, default=12,
                    help="保留 Top-K 电极 (默认 12)")
    pa.add_argument("--n_samples", type=int, default=1000,
                    help="NeuroXAI 随机扰动样本数")
    args = pa.parse_args()
    main(args.k, args.n_samples)