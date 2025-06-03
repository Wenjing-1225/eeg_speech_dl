#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
channels_compare_paper_flat.py  · 2025-06-03
────────────────────────────────────────────
复现 Panachakel & Ramakrishnan 2019 (“short vs long words”)
▪ 8-70 Hz Butterworth + 60 Hz notch
▪ 整段 5 s 想象期（无滑窗）
▪ CSP 选 1,3,5,7,9,11,13,15 电极对
▪ 特征：db4-DWT（4 层 ×3 统计）→ 12 维 / 电极 ×2 = 24 维 / pair
▪ DNN：40-40-40-40，全连接，Dropout 10/30/30/30 %
▪ 5-fold CV，50 epoch，Adam 1 e-3
"""

import os, numpy as np, pywt, torch, torch.nn as nn
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from mne.decoding import CSP
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ─────────── 基本设置 ───────────
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("💻 device =", DEVICE,
      "| CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

DATA_DIR = Path(__file__).resolve().parent.parent / "data/Short_Long_words"
EOG_CH   = [0, 9, 32, 63]                # 1,10,33,64
FS, WIN  = 256, 5 * 256                  # 5 s 想象期
K_LIST   = [1, 3, 5, 7, 9, 11, 13, 15]   # Figure 1
PAIR_DIM = 24                            # 12 × 2

# ─────────── 滤波器 ───────────
bp_b, bp_a       = butter(4, [8, 70], btype="bandpass", fs=FS)
notch_b, notch_a = iirnotch(60, 30, fs=FS)          # Q≈30

def filt(sig):                                      # sig: (60,T)
    sig = filtfilt(bp_b,    bp_a,   sig, axis=1)
    sig = filtfilt(notch_b, notch_a, sig, axis=1)
    return sig

# ─────────── db4-DWT 12 维 ───────────
def dwt12(x):
    # 只保 A4 + D4 + D3 + D2 共 4 组 → 4 × 3 = 12
    coeffs = pywt.wavedec(x, "db4", level=4)[:4]
    feat = []
    for arr in coeffs:
        rms  = np.sqrt((arr**2).mean())
        var  = arr.var()
        p    = (arr**2) / (arr**2).sum()
        ent  = -(p * np.log(p + 1e-12)).sum()
        feat.extend([rms, var, ent])
    return np.asarray(feat, np.float32)             # shape = (12,)

# ─────────── DNN 40-40-40-40 ───────────
class DNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 40), nn.ReLU(),  nn.BatchNorm1d(40), nn.Dropout(.10),
            nn.Linear(40, 40),     nn.ReLU(),  nn.BatchNorm1d(40), nn.Dropout(.30),
            nn.Linear(40, 40),     nn.Tanh(),  nn.BatchNorm1d(40), nn.Dropout(.30),
            nn.Linear(40, 40),     nn.ReLU(),  nn.BatchNorm1d(40), nn.Dropout(.30),
            nn.Linear(40, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# ─────────── 主流程 ───────────
subj_files = sorted([f for f in DATA_DIR.glob("*.mat") if "_8s" not in f.name])
results = {k: [] for k in K_LIST}

for fmat in subj_files:
    mat = loadmat(fmat, simplify_cells=True)
    key = [k for k in mat if k.endswith("last_beep")][0]
    sig = mat[key]                                    # (2 类, trials)

    # ── 预处理（截 5 s → 去 EOG → 滤波） ──
    trials, labels = [], []
    for cls, row in enumerate(sig):
        for ep in row:
            ep = np.delete(ep[:, :WIN], EOG_CH, 0)    # 60 × 1280
            trials.append(filt(ep))
            labels.append(cls)
    trials = np.stack(trials)                         # (n_tr,60,1280)
    labels = np.asarray(labels, np.float32)

    # ── 类均衡 ──
    n0, n1 = np.sum(labels == 0), np.sum(labels == 1)
    if n0 != n1:
        n_min = min(n0, n1)
        idx0  = np.random.choice(np.where(labels == 0)[0], n_min, replace=False)
        idx1  = np.random.choice(np.where(labels == 1)[0], n_min, replace=False)
        keep  = np.sort(np.hstack([idx0, idx1]))
        trials, labels = trials[keep], labels[keep]

    # ── 针对不同 K 训练 ──
    for K in K_LIST:
        # ① CSP
        csp = CSP(n_components=2*K, reg="ledoit_wolf", transform_into="csp_space")
        csp.fit(trials, labels)
        Wmax, Wmin = csp.filters_[:K], csp.filters_[-K:]

        # ② trial-级特征
        trial_feats, trial_labs = [], []
        for ep, lab in zip(trials, labels):
            pairs = []
            for i in range(K):
                a, b = Wmax[i] @ ep, Wmin[i] @ ep
                vec = np.hstack([dwt12(a), dwt12(b)])         # 24 维
                vec = (vec - vec.mean()) / (vec.std() + 1e-6)
                pairs.append(vec)
            trial_feats.append(np.stack(pairs))               # (K,24)
            trial_labs.append(lab)

        # ③ 5-fold CV
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        fold_acc = []
        for tr_idx, te_idx in cv.split(np.arange(len(trial_feats)), trial_labs):
            # ——训练集展平——
            X_tr = np.concatenate([trial_feats[i] for i in tr_idx], axis=0)
            y_tr = np.concatenate([[trial_labs[i]] * len(trial_feats[i]) for i in tr_idx])
            X_tr = torch.tensor(X_tr, device=DEVICE)
            y_tr = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)

            # ——测试集展平，记录 each-trial pair 数——
            X_te_list, len_list, y_te_trial = [], [], []
            for i in te_idx:
                X_te_list.append(trial_feats[i])
                len_list.append(len(trial_feats[i]))
                y_te_trial.append(trial_labs[i])
            X_te = torch.tensor(np.concatenate(X_te_list), device=DEVICE)

            net  = DNN(in_dim=PAIR_DIM).to(DEVICE)
            opt  = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            loss = nn.BCEWithLogitsLoss()

            # ——训练——
            for epoch in range(50):
                net.train()
                perm = torch.randperm(len(X_tr), device=DEVICE)
                for beg in range(0, len(perm), 256):
                    batch = perm[beg:beg+256]
                    opt.zero_grad()
                    loss(net(X_tr[batch]), y_tr[batch]).backward()
                    opt.step()

            # ——测试：pair → trial 投票——
            net.eval()
            with torch.no_grad():
                prob_flat = torch.sigmoid(net(X_te)).cpu().numpy()

            pred_trial, cursor = [], 0
            for L in len_list:
                pred_trial.append(int(prob_flat[cursor:cursor+L].mean() > 0.5))
                cursor += L
            fold_acc.append(accuracy_score(y_te_trial, pred_trial))

        results[K].append(np.mean(fold_acc))
    print(f"✔ {fmat.name}")

# ─────────── 结果 ───────────
print("\nElectrodes |  acc_mean ± std")
print("-----------------------------")
for k in K_LIST:
    print(f"{k:>9} |  {np.mean(results[k]):.3f} ± {np.std(results[k]):.3f}")