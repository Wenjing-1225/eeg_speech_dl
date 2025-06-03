#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
channels_compare_fbcsp.py  · 2025-06-03
────────────────────────────────────────
• 全 60 通道 (K=0) vs 少通道 CSP (K=1..15)
• 8-70 Hz Butterworth + 60 Hz notch
• db4-DWT (4层) ×3 统计
• DNN: 40-80-80-40, Dropout 10/30/40/40 %
• 10-fold, 100 epoch, StepLR 35→γ0.3
"""

import os, numpy as np, pywt, torch, torch.nn as nn
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from mne.decoding import CSP
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ────────── 全局常量 ──────────
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("💻 device =", DEVICE, "| CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))

DATA_DIR  = Path(__file__).resolve().parent.parent / "data/Short_Long_words"
EOG_CH    = [0, 9, 32, 63]
FS, WIN   = 256, 5*256
K_LIST    = [0, 1, 3, 5, 7, 9, 11, 13, 15]              # 0=全 60
PAIR_DIM0 = 12                                          # 全通道单电极 dim
PAIR_DIMK = 24                                          # CSP pair dim
EPOCHS    = 100
BATCH     = 256
PATIENCE  = 20

# ────────── 滤波 ──────────
bp_b, bp_a = butter(4, [8, 70], btype="bandpass", fs=FS)
notch_b, notch_a = iirnotch(60, 30, fs=FS)

def filt(sig):
    sig = filtfilt(bp_b, bp_a, sig, axis=1)
    sig = filtfilt(notch_b, notch_a, sig, axis=1)
    return sig

# ────────── db4-DWT 12 维 ──────────
def dwt12(x):
    coeffs = pywt.wavedec(x, "db4", level=4)[:4]        # A4+D4+D3+D2
    feat = []
    for arr in coeffs:
        rms = np.sqrt((arr**2).mean())
        var = arr.var()
        p   = (arr**2)/(arr**2).sum()
        ent = -(p*np.log(p+1e-12)).sum()
        feat += [rms, var, ent]
    return np.asarray(feat, np.float32)

# ────────── 改进版 MLP ──────────
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 40), nn.ReLU(),  nn.BatchNorm1d(40), nn.Dropout(.10),
            nn.Linear(40, 80),     nn.ReLU(),  nn.BatchNorm1d(80), nn.Dropout(.30),
            nn.Linear(80, 80),     nn.Tanh(),  nn.BatchNorm1d(80), nn.Dropout(.40),
            nn.Linear(80, 40),     nn.ReLU(),  nn.BatchNorm1d(40), nn.Dropout(.40),
            nn.Linear(40, 1)
        )
    def forward(self, x): return self.net(x).squeeze(1)

# ────────── 主流程 ──────────
subj_files = sorted([f for f in DATA_DIR.glob("*.mat") if "_256Hz" in f.name])
results = {k: [] for k in K_LIST}

for fmat in subj_files:
    mat = loadmat(fmat, simplify_cells=True)
    key = [k for k in mat if k.endswith("last_beep")][0]
    sig = mat[key]                                        # (2类, trials)

    # —— 预处理 —— #
    trials, labels = [], []
    for cls, row in enumerate(sig):
        for ep in row:
            ep = np.delete(ep[:, :WIN], EOG_CH, 0)        # 60 × 1280
            trials.append(filt(ep))
            labels.append(cls)
    trials, labels = np.stack(trials), np.asarray(labels, np.float32)

    # —— 平衡两类 —— #
    n0, n1 = np.sum(labels==0), np.sum(labels==1)
    if n0 != n1:
        n_min = min(n0, n1)
        idx0  = np.random.choice(np.where(labels==0)[0], n_min, replace=False)
        idx1  = np.random.choice(np.where(labels==1)[0], n_min, replace=False)
        keep  = np.sort(np.hstack([idx0, idx1]))
        trials, labels = trials[keep], labels[keep]

    # —— 遍历 K —— #
    for K in K_LIST:
        # ① 特征生成
        trial_feats, trial_labs = [], []
        if K == 0:                                       # 全 60 电极
            for ep, lab in zip(trials, labels):
                ch_feats = [dwt12(ep[ch]) for ch in range(60)]  # 60×12
                trial_feats.append(np.stack(ch_feats))
                trial_labs.append(lab)
            pair_dim  = PAIR_DIM0
        else:                                            # CSP K 对
            csp = CSP(n_components=2*K, reg="ledoit_wolf", transform_into="csp_space")
            csp.fit(trials, labels)
            Wmax, Wmin = csp.filters_[:K], csp.filters_[-K:]
            for ep, lab in zip(trials, labels):
                pairs=[]
                for i in range(K):
                    a, b = Wmax[i]@ep, Wmin[i]@ep
                    pairs.append(np.hstack([dwt12(a), dwt12(b)]))
                trial_feats.append(np.stack(pairs))
                trial_labs.append(lab)
            pair_dim  = PAIR_DIMK

        # ② 10-fold CV (trial 级) —— #
        cv = StratifiedKFold(10, shuffle=True, random_state=SEED)
        fold_scores = []
        for tr_idx, te_idx in cv.split(np.arange(len(trial_feats)), trial_labs):
            # —— 训练集展平 —— #
            Xtr = np.concatenate([trial_feats[i] for i in tr_idx], axis=0)
            ytr = np.concatenate([[trial_labs[i]]*len(trial_feats[i]) for i in tr_idx])
            Xtr = torch.tensor(Xtr, device=DEVICE)
            ytr = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)

            # —— 测试集展平 & trial meta —— #
            Xte_list, len_list, yte_trial = [], [], []
            for i in te_idx:
                Xte_list.append(trial_feats[i])
                len_list.append(len(trial_feats[i]))
                yte_trial.append(trial_labs[i])
            Xte = torch.tensor(np.concatenate(Xte_list), device=DEVICE)

            net = MLP(in_dim=pair_dim).to(DEVICE)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.StepLR(opt, step_size=35, gamma=0.3)
            lossf = nn.BCEWithLogitsLoss()

            best, stale = 0, 0
            for epc in range(EPOCHS):
                net.train()
                perm = torch.randperm(len(Xtr), device=DEVICE)
                for beg in range(0, len(perm), BATCH):
                    idx = perm[beg:beg+BATCH]
                    opt.zero_grad(); lossf(net(Xtr[idx]), ytr[idx]).backward(); opt.step()
                sch.step()

                # —— early-stop：验证 on 训练集分块 —— #
                net.eval()
                with torch.no_grad(): pr = torch.sigmoid(net(Xtr)).cpu()
                acc_now = (pr > .5).eq(ytr.cpu()).float().mean().item()
                best, stale = (acc_now, 0) if acc_now > best else (best, stale+1)
                if stale >= PATIENCE: break

            # —— 测试 —— #
            net.eval()
            with torch.no_grad(): prob_flat = torch.sigmoid(net(Xte)).cpu().numpy()

            pred_trial, cur = [], 0
            for L in len_list:
                pred_trial.append(int(prob_flat[cur:cur+L].mean() > .5))
                cur += L
            fold_scores.append(accuracy_score(yte_trial, pred_trial))

        results[K].append(np.mean(fold_scores))
    print("✔", fmat.name)

# ────────── 汇总输出 ──────────
print("\nChannels |  acc_mean ± std")
print("---------------------------")
for k in K_LIST:
    ch = 60 if k == 0 else 2*k
    print(f"{ch:>8} |  {np.mean(results[k]):.3f} ± {np.std(results[k]):.3f}")