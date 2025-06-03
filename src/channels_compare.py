#!/usr/bin/env python
# channels_compare_fbcsp.py   2025-06-03
# ────────────────────────────────────────────────────────────────
# 复现 Panachakel & Ramakrishnan 2019（short-vs-long）
# 并比较：
#   • 60 EEG baseline
#   • 前 K 对 CSP 滤波器 + DWT-24
#   • Filter-Bank CSP (4 子带) + TangentSpace + DWT-24
#   可选 SFFS 物理通道搜索
# ────────────────────────────────────────────────────────────────

import os, warnings, numpy as np, pywt, torch, torch.nn as nn
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from mne.decoding import CSP
from pyriemann.tangentspace import TangentSpace        # ← 0.8 以后
from pyriemann.estimation  import Covariances
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ────────── 全局开关 ──────────
USE_FBCSP = True          # True = 论文配置（FB-CSP+TS+DWT），False = 仅 CSP+DWT
USE_SFFS  = False         # True 时会调用占位的 sffs_dummy()

SEED  = 0
torch.manual_seed(SEED); np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("💻  device =", DEVICE)

# ────────── 路径 & 常量 ───────
DATA_DIR  = Path(__file__).resolve().parent.parent / "data/Short_Long_words"
EOG_CH    = [0, 9, 32, 63]        # 1,10,33,64
FS, WIN   = 256, 5*256
K_LIST    = [0, 1, 3, 5, 7, 9, 11, 15]   # 0 = baseline-60
FBANDS    = [(8,12), (12,16), (16,20), (20,24)]
PAIR_DIM  = 24                 # 12×2  (DWT + 左右 CSP)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ────────── 前处理滤波 ────────
bp_b, bp_a  = butter(4, [8,70], fs=FS, btype='bandpass')
nb_b, nb_a  = iirnotch(60, 30, fs=FS)

def preprocess(sig60):
    sig = filtfilt(bp_b, bp_a,  sig60, axis=1)
    sig = filtfilt(nb_b, nb_a, sig,   axis=1)
    return sig

def bandpass(data, lo, hi):
    b, a = butter(4, [lo,hi], fs=FS, btype='band')
    return filtfilt(b, a, data, axis=-1)

# ────────── DWT-12 特征 ───────
def dwt12(x):
    coeffs = pywt.wavedec(x, "db4", level=4)[:4]   # A4 D4 D3 D2
    out = []
    for arr in coeffs:
        rms  = np.sqrt((arr**2).mean())
        var  = arr.var()
        p    = (arr**2)/(arr**2).sum()
        ent  = -(p*np.log(p+1e-12)).sum()
        out.extend([rms, var, ent])
    return np.asarray(out, np.float32)             # (12,)

# ────────── DNN 40-40-40-40 ───
class DNN(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 40), nn.ReLU(), nn.BatchNorm1d(40), nn.Dropout(.10),
            nn.Linear(40, 40),   nn.ReLU(), nn.BatchNorm1d(40), nn.Dropout(.30),
            nn.Linear(40, 40),   nn.Tanh(), nn.BatchNorm1d(40), nn.Dropout(.30),
            nn.Linear(40, 40),   nn.ReLU(), nn.BatchNorm1d(40), nn.Dropout(.30),
            nn.Linear(40, 1)
        )
    def forward(self, x): return self.net(x).squeeze(1)

# ────────── (占位) SFFS ───────
def sffs_dummy(X60, y):
    """返回前 15 个物理通道索引，仅作占位示例"""
    return list(range(15))

# ────────── 主流程 ────────────
subj_files = sorted([f for f in DATA_DIR.glob("*.mat") if "_8s" not in f.name])
results = {k: [] for k in K_LIST}

for fmat in subj_files:
    mat = loadmat(fmat, simplify_cells=True)
    key = [k for k in mat if k.endswith("last_beep")][0]
    raw = mat[key]                                       # (2,n_trial)

    # ---- 预处理、截 5 s、去 EOG ----
    X_trials, y_trials = [], []
    for cls, row in enumerate(raw):
        for ep in row:
            ep = preprocess(np.delete(ep[:, :WIN], EOG_CH, 0))   # 60×1280
            X_trials.append(ep); y_trials.append(cls)
    X_trials = np.stack(X_trials); y_trials = np.asarray(y_trials, np.float32)

    # ---- 类均衡（可选剔除坏试次） ----
    n0, n1 = (y_trials==0).sum(), (y_trials==1).sum()
    if n0 != n1:
        n_min = min(n0, n1)
        idx0  = np.random.choice(np.where(y_trials==0)[0], n_min, replace=False)
        idx1  = np.random.choice(np.where(y_trials==1)[0], n_min, replace=False)
        keep  = np.sort(np.hstack([idx0, idx1]))
        X_trials, y_trials = X_trials[keep], y_trials[keep]

    # ---- SFFS：提前选物理通道（如果启用） ----
    if USE_SFFS:
        pick_idx = sffs_dummy(X_trials, y_trials)        # user-defined
        X_trials = X_trials[:, pick_idx]                 # 维度变 (n,len(idx),T)

    # ==============================================================
    #                 针对每个 K 构造特征 + 5-fold CV
    # ==============================================================

    for K in K_LIST:
        feats_trial = []                # 每个 trial → (pair_num , feat_dim)
        if K == 0:
            # ===== baseline：60 EEG × DWT-12 =====
            pair_dim, pair_num = 12, X_trials.shape[1]
            for ep in X_trials:
                feats_trial.append(np.stack([dwt12(ch) for ch in ep]))

        else:
            # ====== 先求 CSP 滤波器 ======
            csp = CSP(n_components=2*K, reg='ledoit_wolf',
                      transform_into='csp_space').fit(X_trials, y_trials)
            Wmax, Wmin = csp.filters_[:K], csp.filters_[-K:]

            if USE_FBCSP:
                # ——4 子带 Tangent-Space + DWT-24 (同论文)——
                # ── FB-CSP + TangentSpace + DWT ───────────────────
                ts_bands = []
                for lo, hi in FBANDS:
                    X_fb = bandpass(X_trials, lo, hi)
                    csp = CSP(n_components=2 * K, reg='ledoit_wolf',
                              transform_into='csp_space').fit(X_fb, y_trials)
                    Wmax, Wmin = csp.filters_[:K], csp.filters_[-K:]
                    covs = Covariances(estimator='oas').transform(
                        np.tensordot(Wmax, X_fb, axes=(1, 1)).transpose(1, 0, 2))
                    ts_bands.append(TangentSpace().fit_transform(covs).astype(np.float32))

                pair_dim = ts_bands[0].shape[1] + 24 * K  # ← 乘 K!!
                pair_num = 1  # 每 trial 合成 1 向量
                for t, ep in enumerate(X_trials):
                    dwt_pairs = []
                    for i in range(K):
                        a, b = Wmax[i] @ ep, Wmin[i] @ ep
                        dwt_pairs.append(np.hstack([dwt12(a), dwt12(b)]))
                    feats_trial.append(np.hstack([band[t] for band in ts_bands] + dwt_pairs))

            else:
                # ——只用 DWT-24，pair_num = K——
                pair_dim, pair_num = 24, K
                for ep in X_trials:
                    pairs=[]
                    for i in range(K):
                        a,b = Wmax[i]@ep, Wmin[i]@ep
                        vec = np.hstack([dwt12(a), dwt12(b)])
                        pairs.append((vec-vec.mean())/(vec.std()+1e-6))
                    feats_trial.append(np.stack(pairs))

        # -- 转 ndarray，安全检查 --
        feats_trial = np.asarray(feats_trial)

        # ---- 自适应补维度 ----
        if feats_trial.ndim == 2:  # shape = (n_trial, pair_dim)
            pair_num = 1
            pair_dim = feats_trial.shape[1]
            feats_trial = feats_trial[:, None, :]  # → (n_trial, 1, pair_dim)
        else:  # shape = (n_trial, pair_num, pair_dim)
            pair_dim = feats_trial.shape[2]  # 保持原定义

        # ============ 5-fold CV ============
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        fold_acc=[]
        for tr, te in cv.split(np.arange(len(feats_trial)), y_trials):
            # flatten train
            X_tr = feats_trial[tr].reshape(len(tr)*pair_num, pair_dim)
            y_tr = np.repeat(y_trials[tr], pair_num)
            X_tr = torch.tensor(X_tr, device=DEVICE)
            y_tr = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)

            # flatten test & 记录分段
            X_te = feats_trial[te].reshape(len(te)*pair_num, pair_dim)
            X_te = torch.tensor(X_te, device=DEVICE)
            split_te = [pair_num]*len(te)

            net = DNN(pair_dim).to(DEVICE)
            opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
            lossf = nn.BCEWithLogitsLoss()

            for _ in range(50):
                net.train()
                perm = torch.randperm(len(X_tr), device=DEVICE)
                for beg in range(0, len(perm), 512):
                    idx = perm[beg:beg+512]
                    opt.zero_grad()
                    lossf(net(X_tr[idx]), y_tr[idx]).backward()
                    opt.step()

            # ---- test ----
            net.eval()
            with torch.no_grad():
                prob = torch.sigmoid(net(X_te)).cpu().numpy()
            pred, cur = [], 0
            for n in split_te:
                pred.append(int(prob[cur:cur+n].mean() > .5)); cur += n
            fold_acc.append(accuracy_score(y_trials[te], pred))

        results[K].append(np.mean(fold_acc))
    print("✔", fmat.name)

# ────────── 打印总表 ──────────
print("\nElectrodes |  acc_mean ± std")
print("-----------------------------")
for k in K_LIST:
    tag = "60 (all)" if k==0 else f"{k*2}  "
    print(f"{tag:>10} |  {np.mean(results[k]):.3f} ± {np.std(results[k]):.3f}")