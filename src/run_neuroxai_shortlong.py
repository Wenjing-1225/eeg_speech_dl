#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_neuroxai_eegnet.py
======================
Pipeline: 60-ch Baseline → (可选) NeuroXAI 选通道 → Top-K 评估

修正 & 增强
-----------
A1  基于 GroupKFold(10) 按 trial-ID 分段，baseline 不再“虚高”
A3  额外评估 FBCSP-30 及 NeuroXAI Top-16，快速定位瓶颈
负步长 bug  已彻底修掉（Torch 不再报 ValueError）
"""
print('ss')
import argparse, json, random, time, warnings
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from mne.decoding import CSP
from neuroxai.explanation import BrainExplainer, GlobalBrainExplainer
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.decomposition import FastICA
from sklearn.model_selection import GroupKFold

# ---------------- 全局超参 ----------------
SEED = 0
FS = 256
WIN_S, STEP_S = 2.0, .5                 # 2-s 窗，0.5-s 步
WIN, STEP = int(WIN_S * FS), int(STEP_S * FS)
EPOCH_BASE = 100                        # baseline 训练 epoch
EPOCH_CV   = 60                         # 通道选择后 epoch
BATCH      = 128
THR_BASE   = .60                        # baseline 低于此阈值 → 跳通道选择
FBCSP_BANDS = [(4 + i * 4, 8 + i * 4) for i in range(8)]   # 4-40 Hz
CANDIDATE   = 30                        # FBCSP 候选数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reproducibility
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/Short_Long_words"
FILES = sorted(f for f in DATA.glob("*.mat") if "_8s" not in f.name)

# ---------- 通道信息 ----------
first = loadmat(FILES[0], simplify_cells=True)
k0 = next(k for k in first if k.endswith("last_beep"))
n_tot = first[k0][0][0].shape[0]          # 64

DROP_FIXED = {0, 9, 32, 63}
keep_idx = [i for i in range(n_tot) if i not in DROP_FIXED]

orig_names = [f"Ch{i}" for i in range(n_tot)]
if "ch_names" in first:
    orig_names = [str(s).strip() for s in first["ch_names"]][:n_tot]

CHAN_NAMES = [orig_names[i] for i in keep_idx]
N_CH = len(CHAN_NAMES)
print(f"可用通道 = {N_CH} | device = {DEVICE}")

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

def slide(sig, trial_id):
    wins, gids = [], []
    for st in range(0, sig.shape[1] - WIN + 1, STEP):
        wins.append(sig[:, st:st + WIN])
        gids.append(trial_id)
    return wins, gids

# ---------- FastICA ----------
def fast_ica_all(trials, n_comp=None):
    C, _ = trials[0].shape
    X = np.concatenate(trials, 1).T
    n_comp = C if n_comp is None else n_comp
    ica = FastICA(n_components=n_comp, whiten='unit-variance',
                  random_state=SEED, max_iter=300)
    _ = ica.fit_transform(X)
    return np.asarray([ica.transform(s.T).T.astype(np.float32) for s in trials])

# ---------- FBCSP 排序 ----------
def fbcsp_rank(trials, labels):
    score = np.zeros(N_CH)
    for low, high in FBCSP_BANDS:
        b, a = butter(4, [low, high], fs=FS, btype='band')
        fb   = filtfilt(b, a, trials, axis=2)
        csp  = CSP(n_components=2, reg='ledoit_wolf', log=False)
        csp.fit(fb.astype(np.float64), labels)
        w = csp.filters_
        score += np.abs(w[0]) + np.abs(w[-1])
    return np.argsort(score)[::-1]

# ---------- EEGNet ----------
class EEGNet(nn.Module):
    def __init__(self, C, n_cls=2, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 16, (C, 1), groups=8, bias=False),        nn.BatchNorm2d(16),
            nn.ReLU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(16, 16, (1, 16), padding=(0, 8), bias=False), nn.BatchNorm2d(16),
            nn.ReLU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
            nn.Conv2d(16, 16, 1, bias=False), nn.ReLU()
        )
        self.gap  = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(16, n_cls)

    def forward(self, x):
        x = self.block(x)
        x = self.gap(x).flatten(1)
        return self.head(x)

def train_net(X, y, epochs, lr=1e-3, dropout=0.25):
    net = EEGNet(X.shape[2], dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, 50, 0.5)
    cri = nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        idx = torch.randperm(len(X), device=DEVICE)
        for beg in range(0, len(idx), BATCH):
            sl = idx[beg:beg + BATCH]
            opt.zero_grad()
            loss = cri(net(X[sl]), y[sl]); loss.backward(); opt.step()
        sch.step()
    return net

def eval_net(net, X, y_np, g_np):
    net.eval(); preds = []
    with torch.no_grad():
        for beg in range(0, len(X), BATCH):
            preds.append(net(X[beg:beg + BATCH]).argmax(1).cpu())
    preds = np.concatenate(preds)
    vote = {}
    for p, i in zip(preds, g_np):
        vote.setdefault(i, []).append(p)
    trial_pred = {i: max(set(v), key=v.count) for i, v in vote.items()}
    return np.mean([trial_pred[i] == int(y_np[g_np == i][0]) for i in trial_pred])

# ---------- NeuroXAI importance ----------
def neuroxai_imp(base, trials, labels, cand_idx, n_samples):
    def clf(batch):
        t = torch.tensor(batch[:, None, :, :], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            out = base(t)
        return torch.softmax(out, 1).cpu().numpy()

    brain = BrainExplainer(25, ['short', 'long'])
    gexp  = GlobalBrainExplainer(brain)
    gexp.explain_instance(
        x=trials[:, :, cand_idx],
        y=labels,
        classifier_fn=clf,
        num_samples=n_samples,
        replacement_method='mean'
    )
    imp = np.zeros(N_CH)
    imp[cand_idx] = [gexp.explain_global_channel_importance().get(i, 0.0)
                     for i in range(len(cand_idx))]
    return imp

# ---------- 主流程 ----------
def main(k_list, n_samples, use_ica=True):
    all_res, gkf = {}, GroupKFold(10)

    for subj_i, matf in enumerate(FILES, 1):
        print(f"\n=== Subject {subj_i}/{len(FILES)}  ({matf.name}) ===")
        t0 = time.time()

        m   = loadmat(matf, simplify_cells=True)
        key = next(k for k in m if k.endswith("last_beep"))
        trials = [preprocess(tr) for cls in m[key] for tr in cls]
        labels = [cls for cls, tset in enumerate(m[key]) for _ in tset]
        trials, labels = np.asarray(trials), np.asarray(labels, dtype=int)
        if use_ica:
            trials = fast_ica_all(trials)
        print(f"Loaded trials = {len(trials)} | {time.time()-t0:.1f}s")

        Xw, Yn, Gn = [], [], []
        for tid, (sig, lab) in enumerate(zip(trials, labels)):
            wins, gids = slide(sig, tid)
            Xw.extend(wins); Yn.extend([lab]*len(wins)); Gn.extend(gids)

        Xw = np.ascontiguousarray(np.stack(Xw))          # 🔧确保连续
        Yn, Gn = np.asarray(Yn, dtype=int), np.asarray(Gn, dtype=int)
        X_t = torch.tensor(Xw[:, None, :, :], device=DEVICE)
        Y_t = torch.tensor(Yn, device=DEVICE)

        # ---------- Baseline ----------
        print("Baseline 60-ch …")
        cv_acc = []
        for tr, te in gkf.split(Xw, Yn, groups=Gn):
            net = train_net(X_t[tr], Y_t[tr], EPOCH_BASE)
            cv_acc.append(eval_net(net, X_t[te], Yn[te], Gn[te]))
        acc0, std0 = float(np.mean(cv_acc)), float(np.std(cv_acc))
        print(f"Baseline = {acc0:.3f} ± {std0:.3f}")
        sub_res = {"baseline": [acc0, std0]}

        if acc0 < THR_BASE:
            print("↳ baseline < threshold, skip channel selection")
            all_res[f"sub{subj_i:02d}"] = sub_res; continue

        # ---------- FBCSP ----------
        cand_idx = fbcsp_rank(trials, labels)[:CANDIDATE]
        print("FBCSP-30:", [CHAN_NAMES[i] for i in cand_idx])

        def cv_acc_subset(sel_idx, epochs):
            # sel_idx 可能来自花式索引，stride 不一定正；先做一次“安全复制”
            sel_idx = np.asarray(sel_idx, dtype=np.int64).copy()  # <-- 关键：copy()
            # 如果你担心顺序问题，也可以显式升序：
            # sel_idx.sort()

            idx_tensor = torch.from_numpy(sel_idx).to(DEVICE)

            # 使用 index_select 保证 Xi 连续
            Xi = X_t.index_select(2, idx_tensor)  # (n_win, 1, K, T)

            scores = []
            for tr, te in gkf.split(Xi.cpu().numpy(), Yn, groups=Gn):
                net_tmp = train_net(Xi[tr], Y_t[tr], epochs)
                scores.append(eval_net(net_tmp, Xi[te], Yn[te], Gn[te]))
            return float(np.mean(scores)), float(np.std(scores))

        acc30, std30 = cv_acc_subset(cand_idx, EPOCH_BASE)
        print(f"FBCSP-30 = {acc30:.3f} ± {std30:.3f}")
        sub_res["fbcsp30"] = [acc30, std30]

        # ---------- NeuroXAI ----------
        print("NeuroXAI …")
        imp   = neuroxai_imp(net, trials, labels, cand_idx, n_samples)
        order = np.argsort(-imp)

        acc16, std16 = cv_acc_subset(order[:16], EPOCH_CV)
        print(f"Top-16   = {acc16:.3f} ± {std16:.3f}")
        sub_res["top16"] = [acc16, std16]

        for K in k_list:
            sel = order[:K]
            accK, stdK = cv_acc_subset(sel, EPOCH_CV)
            sub_res[str(K)] = [accK, stdK]
            print(f"Top-{K:<2}  = {accK:.3f} ± {stdK:.3f}")

        all_res[f"sub{subj_i:02d}"] = sub_res

    out = ROOT / "results/subject_dep_neuroxai_eegnet.json"
    json.dump(all_res, open(out, "w"), indent=2)
    print("\n✔ 结果写入", out)

# ---------- CLI ----------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pa = argparse.ArgumentParser()
    pa.add_argument("--k", type=int, nargs="+", default=[4, 8, 16, 24,32])
    pa.add_argument("--n_samples", type=int, default=800)
    pa.add_argument("--no_ica", action="store_true")
    pa.add_argument("--json", default=None, help=argparse.SUPPRESS)  # 忽略旧参数
    args, _ = pa.parse_known_args()
    main(args.k, args.n_samples, use_ica=not args.no_ica)