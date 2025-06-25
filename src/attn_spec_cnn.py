#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_neuroxai_dwt.py —— 被试内: 60-ch DWT-MLP baseline → (可选) FBCSP 预筛 + NeuroXAI 通道选择
Author: ChatGPT-integrated refactor, 2025-06
"""

import argparse, json, random, time, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pywt, torch
from mne.decoding import CSP
from neuroxai.explanation import BrainExplainer, GlobalBrainExplainer
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.decomposition import FastICA
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

# ========= 超参 =========
SEED          = 0
FS            = 256
WIN_S, STEP_S = 2.0, .5
WIN,  STEP    = int(WIN_S*FS), int(STEP_S*FS)
THR_BASE      = .60        # baseline < 0.60 就跳过通道选择
FBCSP_BANDS   = [(4+4*i, 8+4*i) for i in range(8)]   # 4–40 Hz 8 个带
CANDIDATE     = 30          # 送入 NeuroXAI 的预筛通道数
N_SAMPLES_XAI = 800         # NeuroXAI 扰动次数
DEVICE        = "cpu"       # 这里只用 CPU；若需 GPU 可修改

random.seed(SEED); np.random.seed(SEED)

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data/Short_Long_words"
FILES = sorted(p for p in DATA.glob("*.mat") if "_8s" not in p.name)

# ========= 通道信息 =========
first_mat = loadmat(FILES[0], simplify_cells=True)
key0      = next(k for k in first_mat if k.endswith("last_beep"))
n_tot     = first_mat[key0][0][0].shape[0]

DROP_FIXED = {0,9,32,63}
keep_idx   = [i for i in range(n_tot) if i not in DROP_FIXED]

orig_names = [f"Ch{i}" for i in range(n_tot)]
if "ch_names" in first_mat:
    orig_names = [str(s).strip() for s in first_mat["ch_names"]][:n_tot]

CHAN_NAMES = [orig_names[i] for i in keep_idx]
N_CH       = len(CHAN_NAMES)
print(f"✓ 可用通道 = {N_CH}")

# ========= 预处理 =========
bp_b, bp_a = butter(4, [4,40], fs=FS, btype="band")
nt_b, nt_a = iirnotch(60, 30, fs=FS)
def preprocess(sig):
    sig = sig[keep_idx]
    sig = filtfilt(nt_b, nt_a, sig, axis=1)
    sig = filtfilt(bp_b, bp_a, sig, axis=1)
    sig -= sig.mean(1, keepdims=True)
    sig /= sig.std (1, keepdims=True) + 1e-6
    return sig.astype(np.float32)

# ========= FastICA（可选） =========
def fast_ica_all(trials, n_comp=None):
    C,T = trials[0].shape
    Xcat = np.concatenate(trials,1).T           # (samples,C)
    if n_comp is None: n_comp = C
    ica = FastICA(n_components=n_comp, whiten='unit-variance',
                  random_state=SEED, max_iter=300)
    _ = ica.fit_transform(Xcat)
    return np.asarray([ica.transform(tr.T).T.astype(np.float32) for tr in trials])

# ========= DWT 特征 =========
def extract_features(trial_sig, wavelet='db4', level=4, channels=None):
    if channels is None: channels = range(trial_sig.shape[0])
    feats=[]
    for ch in channels:
        coeffs = pywt.wavedec(trial_sig[ch], wavelet, level=level)
        for detail in coeffs[1:]:
            power = detail**2
            p = power / (power.sum()+1e-12)
            entropy = -np.sum(p * np.log2(p+1e-12))
            feats.extend([np.sqrt(power.mean()), power.var(), entropy])
    return np.asarray(feats, dtype=np.float32)

# ========= 滑窗（若想做窗口级特征，可用） =========
def slide(sig):
    out=[];
    for st in range(0, sig.shape[1]-WIN+1, STEP):
        out.append(sig[:, st:st+WIN])
    return out  # 返回 list，各窗口长度 WIN

# ========= FBCSP 排序 =========
def fbcsp_rank(trials, labels):
    score=np.zeros(N_CH)
    for low,high in FBCSP_BANDS:
        b,a = butter(4,[low,high],fs=FS,btype='band')
        fb  = filtfilt(b,a,trials,axis=2)
        csp = CSP(n_components=2, reg='ledoit_wolf', log=False)
        csp.fit(fb.astype(np.float64), labels)
        w   = csp.filters_
        score += np.abs(w[0]) + np.abs(w[-1])
    return np.argsort(score)[::-1]

# ========= 交叉验证函数 =========
def trial_cv_MLP(X_feat, y, groups, n_fold=10):
    gkf=GroupKFold(n_fold)
    accs=[]
    pipe=Pipeline([
        ('scaler',StandardScaler()),
        ('mlp',MLPClassifier(hidden_layer_sizes=(64,),max_iter=500,
                             random_state=SEED))
    ])
    for tr,te in gkf.split(X_feat,y,groups):
        pipe.fit(X_feat[tr], y[tr])
        y_pred = pipe.predict(X_feat[te])
        test_grp = groups[te]
        correct=0; total=0
        for gid in np.unique(test_grp):
            idx = np.where(test_grp==gid)[0]
            pred = Counter(y_pred[idx]).most_common(1)[0][0]
            if pred == y[te][idx][0]: correct+=1
            total+=1
        accs.append(correct/total)
    return np.mean(accs), np.std(accs), pipe  # 返回均值、方差、最后一折模型

# ========= NeuroXAI 通道重要度 =========
def neuroxai_importance(X_raw, y, clf_pipe, n_samples=800):
    def clf_fn(batch):  # batch:(B,C,T)
        feats = np.vstack([extract_features(b) for b in batch])
        return clf_pipe.predict_proba(feats)
    brain = BrainExplainer(25, ['0','1'])
    gexp  = GlobalBrainExplainer(brain)
    gexp.explain_instance(X_raw, y, clf_fn, num_samples=n_samples)
    imp = gexp.explain_global_channel_importance()
    # 转成长度 60 的 ndarray
    scores = np.zeros(N_CH)
    for k,v in imp.items():
        scores[int(k)] = v
    return scores

# ========= 主流程 =========
def main(k_list, use_ica=True):
    subj_results={}
    for subj_i, matf in enumerate(FILES,1):
        print(f"\n=== Subject {subj_i}/{len(FILES)}  ({matf.name}) ===")
        # ---------- 读入 ----------
        mat = loadmat(matf, simplify_cells=True)
        k = next(kk for kk in mat if kk.endswith("last_beep"))
        trials = [preprocess(tr) for cls in mat[k] for tr in cls]
        labels = [cls         for cls,tset in enumerate(mat[k]) for _ in tset]
        trials, labels = np.asarray(trials), np.asarray(labels,dtype=int)

        if use_ica:
            trials = fast_ica_all(trials)
        print(f"Trials={len(trials)}")

        # ---------- 提取 60ch 特征 ----------
        X_feat = np.vstack([extract_features(t) for t in trials])
        groups = np.arange(len(trials))          # 每个trial一组

        # ---------- baseline ----------
        mean_b, std_b, pipe_full = trial_cv_MLP(X_feat, labels, groups)
        print(f"Baseline(60ch) = {mean_b:.3f} ± {std_b:.3f}")
        subj_results['baseline'] = [mean_b, std_b]

        if mean_b < THR_BASE:
            print("Baseline 低于阈值，跳过通道选择。")
            subj_results['skip'] = True
            subj_results[f"Top{max(k_list)}"] = [mean_b, std_b]
            subj_results[f"channels"] = CHAN_NAMES
            continue

        # ---------- FBCSP 预筛 → NeuroXAI ----------
        cand_idx  = fbcsp_rank(trials, labels)[:CANDIDATE]
        print("FBCSP 候选30:", [CHAN_NAMES[i] for i in cand_idx])

        imp_scores = neuroxai_importance(trials, labels, pipe_full, N_SAMPLES_XAI)
        rank_all   = np.argsort(-imp_scores)
        # 只保留在候选集里的顺序
        rank_filtered = [ch for ch in rank_all if ch in cand_idx]

        for K in k_list:
            sel = rank_filtered[:K]
            Xk  = np.vstack([extract_features(tr, channels=sel) for tr in trials])
            m,s,_ = trial_cv_MLP(Xk, labels, groups)
            subj_results[f"Top{K}"] = [m,s]
            print(f"  • Top-{K:<2}: {m:.3f} ± {s:.3f}  ← { [CHAN_NAMES[i] for i in sel] }")

        subj_results['channels'] = CHAN_NAMES
        # 保存每个被试结果
        out = ROOT/f"results/sub{subj_i:02d}_dwt_neuroxai.json"
        Path(out).parent.mkdir(exist_ok=True)
        json.dump(subj_results, open(out,'w'), indent=2)
        print("  ↳ 结果写入", out)
    print("\n✓ All subjects done.")

# ========= CLI =========
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pa = argparse.ArgumentParser()
    pa.add_argument("--k", type=int, nargs='+', default=[4,8,16,32],
                    help="要评估的 Top-K 通道数")
    pa.add_argument("--no_ica", action='store_true', help="禁用 FastICA")
    args = pa.parse_args()
    main(args.k, use_ica=not args.no_ica)