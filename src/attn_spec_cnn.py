#!/usr/bin/env python
# attn_spec_cnn_v2.py  ·  SE-Attention-3D-CNN + STFT 4-40 Hz
# -----------------------------------------------------------
# 1. 64-ch → 去 EOG → 60-ch
# 2. 2-comp CSP 计算 |w_max|+|w_min| 排序通道
# 3. 对每个 trial 滑窗 (2 s, step 0.5 s)  →  log-STFT (4-40 Hz, 39 bins)
# 4. reshape → (N,1,Freq,Chan,Time)  输入 3-D CNN
# 5. Spec3DCNN：Conv(1×3×3) → depthwise Conv(K,3,3) + SE Block
# 6. 10-fold GroupKFold 试次投票 → #Channels-Accuracy 曲线
# -----------------------------------------------------------

import re, math, numpy as np, torch, torch.nn as nn
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, stft
from mne.decoding import CSP
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ========== 运行环境 ==========
SEED   = 0
np.random.seed(SEED);  torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻  device = {DEVICE}")

# ========== 数据与通道 ==========
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data" / "Short_Long_words"
MAT_FILES = sorted(f for f in DATA_DIR.glob("*.mat") if "_8s" not in f.name)

# !! 按真实顺序补齐 64 个名称
ALL_CH_NAMES = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FCz','FC2','FC6',
    'T7','C3','Cz','C4','T8','CP5','CP1','CP2','CP6','P7','P3','Pz',
    'P4','P8','O1','Oz','O2','PO7','PO3','PO4','PO8','AF7','AF3','AF4',
    'AF8','FT7','FT8','TP7','TP8','CP3','CP4','C1','C2','P1','P2','CPz',
    'POz','FC3','FC4','F1','F2','F5','F6','C5','C6','P5','P6','O9',
    'O10','T9','T10','Iz'
]
assert len(ALL_CH_NAMES) == 64, "请补全/确认 64-ch 名称顺序！"

EOG_IDX     = {0, 9, 32, 63}                 # ← 依实际调整
KEEP_CH_ID  = [i for i in range(64) if i not in EOG_IDX]  # 60-ch
CHAN_NAMES  = [ALL_CH_NAMES[i] for i in KEEP_CH_ID]

# ========== 信号处理参数 ==========
FS       = 256
WIN_S    = 2.0 ;  WIN  = int(WIN_S * FS)
STEP_S   = 0.5 ;  STEP = int(STEP_S * FS)
K_LIST   = [4, 8, 16, 32, 60]
EPOCHS   = 120
BATCH    = 128

# ========= 预处理滤波器 (4-40 Hz & 60 Hz notch) =========
bp_b, bp_a = butter(4, [4, 40], fs=FS, btype='band')
nt_b, nt_a = iirnotch(60, 30, fs=FS)

def preprocess(sig):
    sig = sig.astype(np.float32, copy=False)
    sig = filtfilt(nt_b, nt_a, sig, axis=1)
    sig = filtfilt(bp_b, bp_a, sig, axis=1)
    sig -= sig.mean(axis=1, keepdims=True)
    sig /= sig.std(axis=1, keepdims=True) + 1e-6
    return sig

def slide(sig_k):
    return np.stack([sig_k[:, st:st+WIN]
                     for st in range(0, sig_k.shape[1]-WIN+1, STEP)])

def to_spec(win):
    """(K,T) → (K, 39, T')  4-40 Hz log-STFT"""
    spec = []
    for ch in win:
        f, t, Z = stft(ch, fs=FS, window='hann', nperseg=128, noverlap=64)
        Z = np.abs(Z)
        spec.append(np.log1p(Z[2:41]))   # 4–40 Hz => bins 2-40 (共39)
    return np.stack(spec)                # (K,39,T')

# ========= 模型 =========
class SE_Block(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, c//r, bias=False), nn.ReLU(),
            nn.Linear(c//r, c, bias=False), nn.Sigmoid()
        )
    def forward(self, x):                # x:(B,C,1,1,1)
        w = self.fc(x.flatten(1)).view(x.size(0), x.size(1), 1, 1, 1)
        return w

class Spec3DCNN(nn.Module):
    def __init__(self, K_sel, F_bins, T_bins):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, (1,3,3), padding=(0,1,1))
        self.bn1   = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, (K_sel,3,3),
                               groups=32, padding=(0,1,1))
        self.bn2   = nn.BatchNorm3d(64)
        self.se    = SE_Block(64)

        self.pool  = nn.AdaptiveAvgPool3d((1,1,1))
        self.drop  = nn.Dropout(.4)
        self.fc    = nn.Linear(64, 2)

    def forward(self, x):                # x:(B,1,F,K,T')
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        w = self.se(self.pool(x))
        x = x * w
        x = self.drop(self.pool(x).flatten(1))
        return self.fc(x)                # logits

# ========= 主循环 =========
curve = {k: [] for k in K_LIST}

for matf in MAT_FILES:
    mdict = loadmat(matf, simplify_cells=True)
    key   = next(k for k in mdict if re.search(r"last_beep$", k, re.I))
    raw   = mdict[key]                                      # (2, trials)

    trials, labels = [], []
    for cls, trial_set in enumerate(raw):
        for ep in trial_set:
            trials.append(preprocess(ep[KEEP_CH_ID]))
            labels.append(cls)
    trials = np.asarray(trials)
    labels = np.asarray(labels)

    # 平衡
    i0, i1 = np.where(labels==0)[0], np.where(labels==1)[0]
    m = min(len(i0), len(i1))
    keep = np.sort(np.hstack([i0[:m], i1[:m]]))
    trials, labels = trials[keep], labels[keep]

    # 粗排通道
    csp   = CSP(2, reg='ledoit_wolf').fit(trials.astype(np.float64), labels)
    score = np.abs(csp.filters_[0]) + np.abs(csp.filters_[-1])
    order = np.argsort(score)[::-1]

    for K in K_LIST:
        sel = order[:K] if K != 60 else order
        print(f"[{matf.stem}] Top-{K:<2}: " +
              ', '.join(CHAN_NAMES[i] for i in sel))

        # ---------- 构造窗口样本 ----------
        X_win, y_win, g_id = [], [], []
        gid = 0
        for sig, lab in zip(trials[:, sel], labels):
            for w in slide(sig):                       # (K,T)
                X_win.append(to_spec(w))               # (K,39,T')
                y_win.append(lab); g_id.append(gid)
            gid += 1

        X_win = np.asarray(X_win)                      # (N,K,39,T')
        K_sel, F_bins, T_bins = X_win.shape[1:]
        X_win = X_win.transpose(0, 2, 1, 3)            # (N,39,K,T')
        X_win = X_win[:, None]                         # (N,1,39,K,T')
        y_win = np.asarray(y_win); g_id = np.asarray(g_id)

        # ---------- 10-Fold GroupKFold ----------
        acc_fold=[]; gkf=GroupKFold(10)
        for tr, te in gkf.split(X_win, y_win, groups=g_id):
            Xtr = torch.tensor(X_win[tr], dtype=torch.float32, device=DEVICE)
            ytr = torch.tensor(y_win[tr], dtype=torch.long,  device=DEVICE)
            Xte = torch.tensor(X_win[te], dtype=torch.float32, device=DEVICE)
            yte = y_win[te]; gte = g_id[te]

            net = Spec3DCNN(K_sel, F_bins, T_bins).to(DEVICE)
            opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4)
            lossf = nn.CrossEntropyLoss()

            net.train()
            for epoch in range(EPOCHS):
                perm = torch.randperm(len(Xtr), device=DEVICE)
                for beg in range(0, len(perm), BATCH):
                    idx = perm[beg:beg+BATCH]
                    logits = net(Xtr[idx])
                    loss   = lossf(logits, ytr[idx])
                    opt.zero_grad(); loss.backward(); opt.step()

            # -------- 评估 --------
            net.eval(); pred=[]
            with torch.no_grad():
                for beg in range(0, len(Xte), BATCH):
                    sl = slice(beg, beg+BATCH)
                    pred.append(net(Xte[sl]).argmax(1).cpu())
            pred = torch.cat(pred).numpy()

            # trial 投票
            vote = {}
            for p, gid_ in zip(pred, gte):
                vote.setdefault(gid_, []).append(p)
            acc_fold.append(np.mean([
                max(set(v), key=v.count) ==
                yte[np.where(gte == gid_)[0][0]]
                for gid_, v in vote.items()
            ]))
        curve[K].append(np.mean(acc_fold))

# ========= 结果 =========
print("\n#Channels |  acc_mean ± std")
print("---------------------------")
for k in K_LIST:
    arr = np.asarray(curve[k])
    print(f"{k:>9} |  {arr.mean():.3f} ± {arr.std():.3f}")