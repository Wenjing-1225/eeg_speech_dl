#!/usr/bin/env python
# attn_spec_cnn_fixed.py
# ---------------------------------------------------------------
# 1. 64-ch → 去掉 EOG → 60-ch
# 2. 2-comp CSP 粗排分数             （|w_max|+|w_min|）
# 3. SE-Attention-CNN 细调通道权重   （训练时可观测注意力）
# 4. 对 4-40 Hz 片段做 STFT → log-power 频谱图
# 5. GroupKFold(10) 逐 trial 投票精度曲线
# ---------------------------------------------------------------

import re, math, numpy as np, torch, torch.nn as nn
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, stft
from mne.decoding import CSP
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ========= 运行环境 & 随机种子 =========
SEED = 0
np.random.seed(SEED);  torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 device = {DEVICE}")

# ========= 数据路径 =========
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "Short_Long_words"
MAT_FILE = sorted(f for f in DATA_DIR.glob("*.mat") if "_8s" not in f.name)

# ========= 通道名字 (64) =========
ALL_CH_NAMES = [
    # ↓请替换成你真实的 64 通道顺序
    'Fp1','Fp2','F7','F3','Fz','F4','F8','FC5',
    'FC1','FCz','FC2','FC6','T7','C3','Cz','C4',
    'T8','CP5','CP1','CP2','CP6','P7','P3','Pz',
    'P4','P8','O1','Oz','O2','PO7','PO3','PO4',
    'PO8','AF7','AF3','AF4','AF8','FT7','FT8','TP7',
    'TP8','CP3','CP4','C1','C2','P1','P2','CPz',
    'POz','FC3','FC4','F1','F2','F5','F6','C5',
    'C6','P5','P6','O9','O10','T9','T10','Iz'
]
assert len(ALL_CH_NAMES) == 64, "ALL_CH_NAMES 必须正好 64 个！"

# → 要剔除的 4 个 EOG / reference 索引
EOG_IDX = {0, 9, 32, 63}
KEEP_CH_ID = [i for i in range(64) if i not in EOG_IDX]
CHAN_NAMES = [ALL_CH_NAMES[i] for i in KEEP_CH_ID]
print("保留通道数 =", len(KEEP_CH_ID))                       # 应为 60

# ========= 信号处理参数 =========
FS       = 256
WIN_S    = 2.0;   WIN  = int(WIN_S * FS)      # 2-s 片段
STEP_S   = 0.5;   STEP = int(STEP_S * FS)
K_LIST   = [4, 8, 16, 32, 60]
EPOCHS   = 120
BATCH    = 128

# ========= 预处理滤波器 =========
bp_b, bp_a = butter(4, [4, 40], fs=FS, btype='band')
nt_b, nt_a = iirnotch(60, 30, fs=FS)

def preprocess(raw):
    """
    raw : (C,T) float64/float32
    return: (C,T) float32, 已带通、陷波、标准化
    """
    x = raw.astype(np.float32, copy=False)
    x = filtfilt(nt_b, nt_a, x, axis=1)
    x = filtfilt(bp_b, bp_a, x, axis=1)
    x -= x.mean(axis=1, keepdims=True)
    x /= x.std (axis=1, keepdims=True) + 1e-6
    return x

def slide(sig_k):
    """(K,T) → (n_win, K, T_win)"""
    wins = [sig_k[:, st:st+WIN]
            for st in range(0, sig_k.shape[1]-WIN+1, STEP)]
    return np.stack(wins)

def to_spec(win):
    """(K,T_win) → (K,F,T')  log-power STFT"""
    spec = []
    for ch in win:
        f, t, Z = stft(ch, fs=FS, window='hann',
                       nperseg=128, noverlap=64)
        spec.append(np.log1p(np.abs(Z)))
    return np.stack(spec)                      # (K, F, T')

# ========= 注意力 CNN =========
class SE_Block(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(c, c//r, bias=False), nn.ReLU(),
            nn.Linear(c//r, c, bias=False), nn.Sigmoid())
    def forward(self, x):
        y = self.gap(x).flatten(1)             # (B,C)
        w = self.fc(y).view(x.size(0), x.size(1), 1, 1)
        return x * w, w                        # 返回注意力权重

class SpecCNN(nn.Module):
    def __init__(self, img_h, img_w):          # h = K*F , w = T'
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.se1   = SE_Block(16)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(.25)

        h2 = img_h // 2
        w2 = img_w // 2
        self.fc = nn.Linear(16*h2*w2, 2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))   # (B,16,H,W)
        x, att = self.se1(x)
        x = self.pool1(x); x = self.drop1(x)
        return self.fc(x.flatten(1)), att         # att→(B,16,1,1)

# ========= 主流程 =========
curve = {k: [] for k in K_LIST}

for matf in MAT_FILE:
    mdict = loadmat(matf, simplify_cells=True)
    key   = next(k for k in mdict if re.search(r"last_beep$", k, re.I))
    raw   = mdict[key]                             # (2, trials)

    trials, labels = [], []
    for cls, trial_set in enumerate(raw):
        for ep in trial_set:
            sig = preprocess(ep[KEEP_CH_ID])       # (60,T)
            trials.append(sig); labels.append(cls)
    trials  = np.asarray(trials)
    labels  = np.asarray(labels)

    # 类别均衡
    i0, i1 = np.where(labels==0)[0], np.where(labels==1)[0]
    m = min(len(i0), len(i1))
    keep = np.sort(np.hstack([i0[:m], i1[:m]]))
    trials, labels = trials[keep], labels[keep]

    # 粗排 CSP
    csp = CSP(2, reg='ledoit_wolf').fit(trials.astype(np.float64), labels)
    score = np.abs(csp.filters_[0]) + np.abs(csp.filters_[-1])
    order = np.argsort(score)[::-1]

    for K in K_LIST:
        sel = order[:K] if K != 60 else order        # 60 时用全部
        print(f"[{matf.stem}] Top-{K:<2}：",
              ', '.join(CHAN_NAMES[i] for i in sel))

        # -------- 构造窗口时频图 --------
        X_win, y_win, g_id = [], [], []
        gid = 0
        for sig, lab in zip(trials[:, sel], labels):
            for w in slide(sig):                    # (K,T) → windows
                X_win.append(to_spec(w))            # (K,F,T')
                y_win.append(lab)
                g_id.append(gid)
            gid += 1

        X_win = np.asarray(X_win)                   # (N, K, F, T')
        K_sel, F_bins, T_bins = X_win.shape[1], X_win.shape[2], X_win.shape[3]

        # reshape → (N,1,H,W) 把 (K,F) 合并为“高”
        X_win = X_win.reshape(len(X_win), 1, K_sel*F_bins, T_bins)
        y_win = np.asarray(y_win); g_id = np.asarray(g_id)

        # -------- GroupKFold (10) --------
        acc_fold = []
        gkf = GroupKFold(10)
        for tr, te in gkf.split(X_win, y_win, groups=g_id):
            Xtr = torch.tensor(X_win[tr], dtype=torch.float32, device=DEVICE)
            ytr = torch.tensor(y_win[tr], dtype=torch.long, device=DEVICE)

            Xte = torch.tensor(X_win[te], dtype=torch.float32, device=DEVICE)
            yte = y_win[te]  # ←★ 加回这一行
            gte = g_id[te]  # ←★ 加回这一行

            net = SpecCNN(img_h=K_sel * F_bins, img_w=T_bins).to(DEVICE)
            opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4)
            lossf = nn.CrossEntropyLoss()

            # ---------- 训练 ----------
            net.train()
            for ep in range(EPOCHS):
                perm = torch.randperm(len(Xtr), device=DEVICE)
                for beg in range(0, len(perm), BATCH):
                    idx = perm[beg:beg + BATCH]
                    logit, _ = net(Xtr[idx])
                    loss = lossf(logit, ytr[idx])
                    opt.zero_grad();
                    loss.backward();
                    opt.step()

            # ---------- 推断 ----------
            net.eval();
            preds = []
            with torch.no_grad():
                for beg in range(0, len(Xte), BATCH):
                    sl = slice(beg, beg + BATCH)
                    logits, _ = net(Xte[sl])
                    preds.append(logits.argmax(1).cpu())
            preds = torch.cat(preds).numpy()

            # ---------- trial-level 投票 ----------
            vote = {}
            for p, gid_ in zip(preds, gte):
                vote.setdefault(gid_, []).append(p)

            true_trial = {gid_: yte[np.where(gte == gid_)[0][0]]
                          for gid_ in vote}  # ← 用 yte / gte
            acc_fold.append(
                np.mean([max(set(v), key=v.count) == true_trial[gid_]
                         for gid_, v in vote.items()])
            )

        acc_fold_mean = np.mean(acc_fold)
        curve[K].append(acc_fold_mean)  # 别忘记录

# ========= 输出 =========
print("\n#Channels |  acc_mean ± std")
print("---------------------------")
for k in K_LIST:
    arr = np.asarray(curve[k])
    print(f"{k:>9} |  {arr.mean():.3f} ± {arr.std():.3f}")