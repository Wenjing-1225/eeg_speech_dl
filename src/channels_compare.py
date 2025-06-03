#!/usr/bin/env python
# channels_compare_fbcsp.py
# ─────────────────────────────────────────────────────────────
#  Panachakel & Ramakrishnan 2019 imagined short-vs-long 复现
#    • 60-channel baseline (DWT-12 ×60)
#    • 前 K 对 CSP 滤波器 + DWT-24
#    • 4-band FB-CSP + TangentSpace + DWT-24  （论文配置）
#    • 可选 SFFS 物理通道挑选
# ─────────────────────────────────────────────────────────────

import os, warnings, numpy as np, pywt, torch, torch.nn as nn
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from mne.decoding import CSP
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation  import Covariances
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────── 开关 ────────────────
USE_FBCSP = True          # 论文配置：FB-CSP + TS + DWT
USE_SFFS  = False         # 若 True 请实现 run_sffs()
SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("💻  device =", DEVICE)

# ─────────────── 常量 ────────────────
BASE      = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE / "data/Short_Long_words"
EOG_CH    = [0, 9, 32, 63]                 # 1,10,33,64
FS, WIN   = 256, 5*256                     # 5 s
# K_LIST    = [0, 1, 3, 5, 7, 9, 11, 13, 15] # 0=60-ch baseline
K_LIST = [0] + list(range(1, 31, 1))
FBANDS    = [(8,12), (12,16), (16,20), (20,24)]
PAIR_DIM_DWT = 24                          # 12 × 2

# ─────────────── 滤波器 ───────────────
bp_b, bp_a = butter(4, [8, 70], fs=FS, btype='band')
nb_b, nb_a = iirnotch(60, 30, fs=FS)

def preprocess(sig60):
    sig = filtfilt(bp_b, bp_a,  sig60, axis=1)
    sig = filtfilt(nb_b, nb_a, sig,    axis=1)
    return sig

def bandpass(data, lo, hi):
    b, a = butter(4, [lo, hi], fs=FS, btype='band')
    return filtfilt(b, a, data, axis=-1)

# ─────────────── DWT-12 ───────────────
def dwt12(x):
    coeffs = pywt.wavedec(x, "db4", level=4)[:4]   # A4,D4,D3,D2
    feat = []
    for arr in coeffs:
        rms  = np.sqrt((arr**2).mean())
        var  = arr.var()
        p    = (arr**2)/(arr**2).sum()
        ent  = -(p*np.log(p+1e-12)).sum()
        feat.extend([rms, var, ent])
    return np.asarray(feat, np.float32)            # (12,)

# ─────────────── DNN ────────────────
class DNN(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 40), nn.ReLU(), nn.BatchNorm1d(40), nn.Dropout(.10),
            nn.Linear(40, 40),   nn.ReLU(), nn.BatchNorm1d(40), nn.Dropout(.30),
            nn.Linear(40, 40),   nn.Tanh(), nn.BatchNorm1d(40), nn.Dropout(.30),
            nn.Linear(40, 40),   nn.ReLU(), nn.BatchNorm1d(40), nn.Dropout(.30),
            nn.Linear(40, 1))
    def forward(self, x): return self.net(x).squeeze(1)

# ─────────────── (占位) SFFS ───────────────
def run_sffs(X60, y, k=15):
    """示例：直接返回前 k 个通道；如需真正 SFFS 请自行替换"""
    return list(range(k))

# ─────────────── 主流程 ───────────────
subj_files = sorted([f for f in DATA_DIR.glob("*.mat") if "_8s" not in f.name])
results = {k: [] for k in K_LIST}

for fmat in subj_files:
    mat = loadmat(fmat, simplify_cells=True)
    key = next(k for k in mat if k.endswith("last_beep"))
    raw = mat[key]                                   # (2, trials)

    # ——预处理——
    X, y = [], []
    for cls, row in enumerate(raw):
        for ep in row:
            sig = preprocess(np.delete(ep[:, :WIN], EOG_CH, 0))  # 60×1280
            X.append(sig); y.append(cls)
    X = np.stack(X); y = np.asarray(y, float)

    # ——类均衡——
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    if n0 != n1:
        n_min = min(n0, n1)
        keep0 = np.random.choice(np.where(y == 0)[0], n_min, False)
        keep1 = np.random.choice(np.where(y == 1)[0], n_min, False)
        keep  = np.sort(np.hstack([keep0, keep1]))
        X, y  = X[keep], y[keep]

    # ——SFFS（可选）——
    if USE_SFFS:
        sel = run_sffs(X, y, k=15)   # 返回 ≤15 个通道索引
        X   = X[:, sel]

    # ========== 逐 K 计算 ==========
    for K in K_LIST:
        feats_trial = []             # per-trial 特征列表

        if K == 0:                   # ——60-ch baseline——
            pair_dim, pair_num = 12, X.shape[1]
            for ep in X:
                feats_trial.append(np.stack([dwt12(ch) for ch in ep]))

        else:                        # ——先取 CSP K 对——
            csp = CSP(2*K, reg='ledoit_wolf', transform_into='csp_space').fit(X, y)
            Wmax, Wmin = csp.filters_[:K], csp.filters_[-K:]

            if USE_FBCSP:            # ---- FB-CSP + TS + DWT ----
                ts_bands = []
                for lo, hi in FBANDS:
                    Xfb = bandpass(X, lo, hi)
                    csp_fb = CSP(2*K, reg='ledoit_wolf',
                                  transform_into='csp_space').fit(Xfb, y)
                    covs   = Covariances(estimator='oas').transform(
                             np.tensordot(csp_fb.filters_[:K], Xfb, axes=(1,1))
                             .transpose(1,0,2))
                    ts_bands.append(TangentSpace().fit_transform(covs))

                ts_dim   = ts_bands[0].shape[1]
                pair_dim = ts_dim*len(FBANDS) + 24*K
                pair_num = 1

                for t, ep in enumerate(X):
                    dwt_pairs = []
                    for i in range(K):
                        a, b = Wmax[i]@ep, Wmin[i]@ep
                        dwt_pairs.append(np.hstack([dwt12(a), dwt12(b)]))
                    feats_trial.append(
                        np.hstack([band[t] for band in ts_bands] + dwt_pairs))

            else:                    # ---- 仅 CSP + DWT-24 ----
                pair_dim, pair_num = 24, K
                for ep in X:
                    pairs=[]
                    for i in range(K):
                        a,b = Wmax[i]@ep, Wmin[i]@ep
                        vec = np.hstack([dwt12(a), dwt12(b)])
                        pairs.append((vec-vec.mean())/(vec.std()+1e-6))
                    feats_trial.append(np.stack(pairs))

        feats_trial = np.asarray(feats_trial)         # (n, pair_num, pair_dim)
        if feats_trial.ndim == 2:                     # baseline 情况
            feats_trial = feats_trial[:, None, :]
            pair_num    = 1
            pair_dim    = feats_trial.shape[-1]

        # --------------- 5-fold CV ---------------
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        acc_fold = []
        for tr, te in cv.split(np.arange(len(feats_trial)), y):
            Xtr = torch.tensor(feats_trial[tr].reshape(-1, pair_dim),
                               dtype=torch.float32, device=DEVICE)
            ytr = torch.tensor(np.repeat(y[tr], pair_num),
                               dtype=torch.float32, device=DEVICE)

            Xte = torch.tensor(feats_trial[te].reshape(-1, pair_dim),
                               dtype=torch.float32, device=DEVICE)
            split_te = [pair_num]*len(te)

            net  = DNN(pair_dim).to(DEVICE)
            opt  = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
            loss = nn.BCEWithLogitsLoss()

            for _ in range(50):
                net.train()
                perm = torch.randperm(len(Xtr), device=DEVICE)
                for beg in range(0, len(perm), 512):
                    idx = perm[beg:beg+512]
                    opt.zero_grad()
                    loss(net(Xtr[idx]), ytr[idx]).backward()
                    opt.step()

            net.eval()
            with torch.no_grad():
                prob = torch.sigmoid(net(Xte)).cpu().numpy()

            pred, cur = [], 0
            for n in split_te:
                pred.append(int(prob[cur:cur+n].mean() > 0.5)); cur += n
            acc_fold.append(accuracy_score(y[te], pred))

        results[K].append(np.mean(acc_fold))
    print("✔", fmat.name)

# ─────────────── 汇总输出 ───────────────
print("\nElectrodes |  acc_mean ± std")
print("-----------------------------")
for k in K_LIST:
    tag = "60 (all)" if k == 0 else str(k*2).rjust(2)
    m, s = np.mean(results[k]), np.std(results[k])
    print(f"{tag:>10} |  {m:.3f} ± {s:.3f}")