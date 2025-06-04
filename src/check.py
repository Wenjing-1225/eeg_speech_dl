#!/usr/bin/env python
# csp_pair_logvar_sliding.py  —  CSP 选 N 对通道 + log(var) + 滑窗增强 + 多数表决
# -------------------------------------------------------------------------------
import numpy as np, torch, torch.nn as nn
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from pathlib import Path
from mne.decoding import CSP
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ----------------- 全局配置 -----------------
SEED           = 0
PAIR_LIST      = [1, 3, 5, 7, 9, 11, 13, 15]   # 不同“通道对”数量
FS, WIN        = 256, 5 * 256                  # 5 s 窗
EOG            = [0, 9, 32, 63]                # 去除 4 个眼动通道
EPOCHS         = 300                           # 充分训练
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(SEED); torch.manual_seed(SEED)

# ---------- 滑窗设置 ----------
SHIFT_MODE     = "step"            # "step" → 等步长；"list" → 固定列表
SHIFT_STEP     = 0.5               # 秒；仅在 step 模式下生效
MAX_SHIFT      = 3.0               # 秒；仅在 step 模式下生效
SHIFT_LIST     = [0, 0.75, 1.5, 2.25]    # 仅在 list 模式下生效
MAX_SHIFT_SMP  = int(MAX_SHIFT * FS)     # 用于切片

# ---------- 数据路径 ----------
ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data/Short_Long_words"
FILES = sorted(f for f in DATA.glob("*.mat") if "_8s" not in f.name)

# ----------------- 滤波器 -----------------
bp_b, bp_a = butter(5, [8, 70], fs=FS, btype="band")
nt_b, nt_a = iirnotch(60, 30, fs=FS)
def band_notch(x):
    x = filtfilt(bp_b, bp_a, x, axis=1)
    return filtfilt(nt_b, nt_a, x, axis=1).astype(np.float32)

# ------------- 两通道 log-variance 特征 -------------
def pair_logvar(chA, chB):
    # 输出 shape (2,)
    return np.log(np.var([chA, chB], axis=1) + 1e-12).astype(np.float32)

# ------------- 轻量 DNN — 始终输入 2 维 -------------
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 40), nn.ReLU(), nn.BatchNorm1d(40), nn.Dropout(0.25),
            nn.Linear(40, 40), nn.ReLU(), nn.BatchNorm1d(40), nn.Dropout(0.25),
            nn.Linear(40, 40), nn.Tanh(), nn.BatchNorm1d(40), nn.Dropout(0.25),
            nn.Linear(40, 40), nn.ReLU(), nn.BatchNorm1d(40), nn.Dropout(0.25),
            nn.Linear(40, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# ------------ 通道对选择 ------------
def pick_pairs(filters, n_pair):
    w_max, w_min = filters[0], filters[-1]
    idx_max = np.argsort(np.abs(w_max))[::-1][:n_pair]
    idx_min = np.argsort(np.abs(w_min))[::-1][:n_pair]
    return idx_max, idx_min

# ------------ 生成滑窗偏移 ------------
def gen_shifts(trial_len):
    if SHIFT_MODE == "step":
        secs = np.arange(0, MAX_SHIFT + 1e-3, SHIFT_STEP)
        return [int(s * FS) for s in secs if int(s * FS) + WIN <= trial_len]
    else:  # "list"
        return [int(s * FS) for s in SHIFT_LIST if int(s * FS) + WIN <= trial_len]

# ================= 主流程 =================
results = {p: [] for p in PAIR_LIST}

for matf in FILES:
    mat = loadmat(matf, simplify_cells=True)
    key = next(k for k in mat if k.endswith("last_beep"))
    raw = mat[key]                                      # shape (2, trials)

    # -------- Trial 级读取与预处理 --------
    X_trials, y_trials = [], []
    for cls, trials in enumerate(raw):
        for tr in trials:
            # 为保证足够长度，多截 MAX_SHIFT_SMP
            sig = band_notch(np.delete(tr[:, : WIN + MAX_SHIFT_SMP], EOG, 0))
            X_trials.append(sig);  y_trials.append(cls)
    X_trials, y_trials = np.asarray(X_trials), np.asarray(y_trials)

    # -------- 类平衡 --------
    idx0, idx1 = np.where(y_trials == 0)[0], np.where(y_trials == 1)[0]
    m = min(len(idx0), len(idx1))
    keep = np.sort(np.concatenate([idx0[:m], idx1[:m]]))
    X_trials, y_trials = X_trials[keep], y_trials[keep]

    # -------- 2-comp CSP --------
    csp = CSP(n_components=2, reg="ledoit_wolf").fit(X_trials.astype(np.float64),
                                                     y_trials)

    for P in PAIR_LIST:
        idx_max, idx_min = pick_pairs(csp.filters_, P)

        feats, labs, gids = [], [], []
        for tid, (trial, lab) in enumerate(zip(X_trials, y_trials)):
            for shift in gen_shifts(trial.shape[1]):
                seg = trial[:, shift:shift + WIN]
                for i in range(P):
                    feats.append(pair_logvar(seg[idx_max[i]], seg[idx_min[i]]))
                    labs.append(lab);  gids.append(tid)

        feats = np.asarray(feats)        # shape (N, 2)
        labs  = np.asarray(labs,  np.float32)
        gids  = np.asarray(gids)         # trial id

        # -------- 10-折 GroupKFold --------
        accs=[];  gkf = GroupKFold(10)
        for tr, te in gkf.split(feats, labs, groups=gids):
            scaler = StandardScaler().fit(feats[tr])
            Xtr = torch.tensor(scaler.transform(feats[tr]),
                               dtype=torch.float32, device=DEVICE)
            Xte = torch.tensor(scaler.transform(feats[te]),
                               dtype=torch.float32, device=DEVICE)
            ytr = torch.tensor(labs[tr], dtype=torch.float32, device=DEVICE)
            gte = gids[te]

            net = DNN().to(DEVICE)
            opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=3e-4)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=15, min_lr=1e-4
            )

            for ep in range(EPOCHS):
                net.train()
                perm = torch.randperm(len(Xtr), device=DEVICE)
                running = 0.0
                for beg in range(0, len(perm), 256):
                    sl = perm[beg:beg + 256]
                    opt.zero_grad()
                    loss = nn.BCEWithLogitsLoss()(net(Xtr[sl]), ytr[sl])
                    loss.backward(); opt.step()
                    running += loss.item() * len(sl)
                sched.step(running / len(Xtr))

            net.eval()
            with torch.no_grad():
                prob = torch.sigmoid(net(Xte)).cpu().numpy().flatten()
            pred_vec = (prob >= 0.5).astype(int)

            # -------- Trial-level 多数表决 --------
            tv = {}
            for p, t in zip(pred_vec, gte):
                tv.setdefault(t, []).append(p)
            preds = {t: int(np.mean(v) > 0.5) for t, v in tv.items()}
            truth = {t: int(labs[np.where(gids == t)[0][0]]) for t in preds}
            accs.append(np.mean([preds[t] == truth[t] for t in preds]))

        results[P].append(np.mean(accs))

# -------------------- 汇总结果 --------------------
print(f"\nPairs | acc_mean ± std  (SHIFT_MODE: {SHIFT_MODE})")
print("----------------------------------------------")
for p in PAIR_LIST:
    arr = np.asarray(results[p])
    print(f"{p:>5d} | {arr.mean():.3f} ± {arr.std():.3f}")