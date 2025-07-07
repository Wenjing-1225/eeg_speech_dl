#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_vmd_ebt_short.py  ——  Short-words (3-class: in / out / up)

Pipeline
1. 预处理 (+ 可选 FastICA)
2. 滑窗 2 s（步长 0.5 s）
3. 每窗:  VMD(K=8) → 18 统计特征/Mode/Channel
4. Kruskal-Wallis 选 Top-10 特征
5. Ensemble Bagged Tree 分类 (10-fold GroupKFold, trial-level 投票)
"""
import argparse, json, random, time, warnings
from pathlib import Path

import numpy as np
import torch  # 仅用于统一随机种子
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import kruskal, kurtosis, skew, entropy
from sklearn.decomposition import FastICA
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from vmdpy import VMD
except ImportError:
    raise ImportError("请先安装 vmdpy：  pip install vmdpy")

# --------- 全局超参 ---------
SEED = 0
FS = 256
WIN_S, STEP_S = 2.0, .5
WIN, STEP = int(WIN_S * FS), int(STEP_S * FS)

K_VMD       = 8      # VMD mode 数
TOP_K_FEAT  = 10     # KW 选特征数
N_EST       = 100    # EBT 树数
THR_BASE    = 0.30   # (可保留，未用)
N_CLASS     = 3

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data/Short_words"
FILES = sorted(DATA.glob("*.mat"))

# ---------- 通道信息 ----------
first = loadmat(FILES[0], simplify_cells=True)
k0    = next(k for k in first if k.endswith("last_beep"))
n_tot = first[k0][0][0].shape[0]

DROP_FIXED = {0, 9, 32, 63}
keep_idx   = [i for i in range(n_tot) if i not in DROP_FIXED]

orig_names = [f"Ch{i}" for i in range(n_tot)]
if "ch_names" in first:
    orig_names = [str(s).strip() for s in first["ch_names"]][:n_tot]

CHAN_NAMES = [orig_names[i] for i in keep_idx]
N_CH       = len(CHAN_NAMES)
print(f"可用通道 = {N_CH}")

# ---------- 滤波 ----------
bp_b, bp_a = butter(4, [4, 40], fs=FS, btype="band")
nt_b, nt_a = iirnotch(60, 30, fs=FS)

def preprocess(sig):
    sig = sig[keep_idx]
    sig = filtfilt(nt_b, nt_a, sig, axis=1)
    sig = filtfilt(bp_b, bp_a, sig, axis=1)
    sig -= sig.mean(1, keepdims=True)
    sig /= sig.std(1, keepdims=True) + 1e-6
    return sig.astype(np.float32)

def slide(sig, tid):
    wins, gids = [], []
    for st in range(0, sig.shape[1]-WIN+1, STEP):
        wins.append(sig[:, st:st+WIN]); gids.append(tid)
    return wins, gids

# ---------- FastICA (可选) ----------
def fast_ica_all(trials):
    C,_ = trials[0].shape
    X = np.concatenate(trials, 1).T
    ica = FastICA(n_components=C, whiten='unit-variance',
                  random_state=SEED, max_iter=300)
    _ = ica.fit_transform(X)
    return np.asarray([ica.transform(s.T).T.astype(np.float32) for s in trials])

# ---------- VMD ----------
def vmd_decompose(x1d, K=K_VMD, alpha=2000, tau=0, DC=0, init=1, tol=1e-7):
    u, _, _ = VMD(x1d, alpha, tau, K, DC, init, tol)
    return u.astype(np.float32)           # (K, T)

# ---------- 18 维统计特征 ----------
def feat_vector(x):
    diff = np.diff(x)
    th   = 0.02
    features = [
        np.var(x),                           # 1 VAR
        kurtosis(x, fisher=False),           # 2 Kurtosis
        np.sqrt(np.mean(x**2)),              # 3 RMS
        np.mean(np.abs(x)),                  # 4 MAV
        np.mean(np.abs(diff)),               # 5 AAC
        np.sum(np.diff(np.sign(diff)) != 0), # 6 SSC
        np.sqrt(np.mean(diff**2)),           # 7 DASDV
        np.std(x),                           # 8 σ
        ((x[:-1]*x[1:]) < 0).sum(),          # 9 ZC
        entropy(np.abs(x) + 1e-12),          #10 SE
        np.median(x),                        #11 Median
        np.sum(np.abs(x)),                   #12 IEEG
        np.sum(x**2),                        #13 SSI
        skew(x),                             #14 Skewness
        np.mean(np.abs(x[len(x)//2:])),      #15 MMAV2
        np.sum(np.abs(diff) > th),           #16 WA
        np.mean(np.abs(x[len(x)//4:])),      #17 MMAV1
        np.sum(np.abs(diff))                 #18 WL
    ]
    return np.asarray(features, dtype=np.float32)

def window_features(win_sig):
    feats = []
    for ch in win_sig:
        modes = vmd_decompose(ch)            # (K, T)
        for m in modes:
            feats.extend(feat_vector(m))
    return np.asarray(feats, dtype=np.float32)  # (C*K*18,)

# ---------- KW 评分 ----------
def kw_score(X, y):
    classes = np.unique(y)
    pvals = np.array([kruskal(*[X[y==c, i] for c in classes]).pvalue
                      for i in range(X.shape[1])])
    scores = -np.log(pvals + 1e-300)
    return scores, pvals

# ---------- 分类器 ----------
def build_clf():
    base = DecisionTreeClassifier(max_depth=None, random_state=SEED)
    return BaggingClassifier(
        base_estimator=base,
        n_estimators=N_EST,
        max_samples=1.0,
        bootstrap=True,
        n_jobs=-1,
        random_state=SEED
    )

# ---------- 主 ----------
def main(use_ica=True):
    all_res = {}
    gkf = GroupKFold(10)

    for subj_i, matf in enumerate(FILES, 1):
        print(f"\n=== Subject {subj_i}/{len(FILES)} ({matf.name}) ===")
        t0 = time.time()

        m   = loadmat(matf, simplify_cells=True)
        key = next(k for k in m if k.endswith("last_beep"))
        trials = [preprocess(tr) for cls in m[key] for tr in cls]
        labels = [cls for cls, tset in enumerate(m[key]) for _ in tset]
        trials, labels = np.asarray(trials), np.asarray(labels, dtype=int)

        if use_ica:
            trials = fast_ica_all(trials)
        print(f"Loaded trials = {len(trials)} | {time.time()-t0:.1f}s")

        # ---- 滑窗 + 特征 ----
        feats, Yn, Gn = [], [], []
        for tid, (sig, lab) in enumerate(zip(trials, labels)):
            wins, gids = slide(sig, tid)
            for w, gid in zip(wins, gids):
                feats.append(window_features(w))
                Yn.append(lab)
                Gn.append(gid)

        X_feat = np.ascontiguousarray(np.stack(feats))  # (N_win, F)
        Yn = np.asarray(Yn, dtype=int)
        Gn = np.asarray(Gn, dtype=int)
        print(f"Feature matrix shape = {X_feat.shape}")

        # ---- CV ----
        fold_acc = []
        for tr, te in gkf.split(X_feat, Yn, groups=Gn):
            sel = SelectKBest(kw_score, k=TOP_K_FEAT)
            X_tr = sel.fit_transform(X_feat[tr], Yn[tr])
            X_te = sel.transform(X_feat[te])

            clf = build_clf()
            clf.fit(X_tr, Yn[tr])
            y_pred = clf.predict(X_te)

            # trial-level 投票
            vote = {}
            for p, gid in zip(y_pred, Gn[te]):
                vote.setdefault(gid, []).append(p)
            trial_pred = {gid:max(v, key=v.count) for gid,v in vote.items()}
            acc = np.mean([trial_pred[gid]==labels[gid] for gid in trial_pred])
            fold_acc.append(acc)

        acc_m, acc_s = float(np.mean(fold_acc)), float(np.std(fold_acc))
        print(f"VMD+EBT = {acc_m:.3f} ± {acc_s:.3f}")

        all_res[f"sub{subj_i:02d}"] = {"vmd_ebt":[acc_m, acc_s]}

    out = ROOT / "results/vmd_ebt_short.json"
    json.dump(all_res, open(out, "w"), indent=2, ensure_ascii=False)
    print("\n✔ 结果写入", out)

# ---------- CLI ----------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_ica", action="store_true",
                        help="禁用 FastICA（默认开启）")
    args, _ = parser.parse_known_args()
    main(use_ica=not args.no_ica)