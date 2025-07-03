#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_neuroxai_multimodal.py
==========================
Short/Long-words (2-class) – 通道选择实验

Pipeline
1. 60-ch baseline EEGNet           (时域)
2. 多模态 surrogate  (时域 + PSD + EWT)  ➜ Kernel-SHAP 得 imp[i]
3. Top-K 时域 EEGNet (用 imp 排序)     – 10-fold CV
结果 → results/neuroxai_multimodal.json
"""
# --------------------------------------------------
import json, random, time, warnings
from pathlib import Path

import numpy as np
import shap
import torch, torch.nn as nn
from mne.decoding import CSP                                  # 仍保留，可选
from neuroxai.explanation import BrainExplainer, GlobalBrainExplainer
from PyEMD import EMD                                          # pip install EMD-signal
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, welch
from sklearn.model_selection import GroupKFold
# --------------------------------------------------
# 全局超参
SEED = 0
FS = 256
WIN_S, STEP_S = 2.0, .5
WIN, STEP     = int(WIN_S*FS), int(STEP_S*FS)
BATCH         = 128
EPOCH_BASE    = 100
EPOCH_CV      = 60
SHAP_SAMP     = 256
THR_BASE      = .60
DROP_FIXED    = {0,9,32,63}
N_EWT         = 5              # ★ EWT / EMD 分量数
USE_PSD       = True           # ★ 频谱分支开关
USE_EWT       = True           # ★ IMF  分支开关
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
# --------------------------------------------------
ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / 'data/Short_Long_words'
FILES = sorted(f for f in DATA.glob('*.mat') if '_8s' not in f.name)

# ---------- 通道 & 滤波 ----------
first  = loadmat(FILES[0], simplify_cells=True)
k0     = next(k for k in first if k.endswith('last_beep'))
n_tot  = first[k0][0][0].shape[0]

keep_idx = [i for i in range(n_tot) if i not in DROP_FIXED]
N_CH     = len(keep_idx)
T_LEN    = WIN

bp_b, bp_a = butter(4, [4,40], fs=FS, btype='band')
nt_b, nt_a = iirnotch(60, 30, fs=FS)

emd = EMD()                                # ★ 一次实例化

# ---------- 预处理（时域 / PSD / IMF） ----------
def preprocess(sig):
    sig = sig[keep_idx]
    sig = filtfilt(nt_b, nt_a, sig, axis=1)
    sig = filtfilt(bp_b, bp_a, sig, axis=1)
    sig -= sig.mean(1, keepdims=True)
    sig /= sig.std (1, keepdims=True) + 1e-6
    sig_t = sig.astype(np.float32)                          # (C,T)

    psd = None; imf = None
    if USE_PSD:
        f, Pxx = welch(sig_t, fs=FS, nperseg=WIN, axis=1)
        psd = np.log(Pxx + 1e-12).astype(np.float32)        # (C,F)

    if USE_EWT:
        imfs = []
        for ch in sig_t:
            comps = emd(ch)[:N_EWT]                         # (≤N,T)
            if comps.shape[0] < N_EWT:                      # 不足补 0
                comps = np.pad(comps, ((0,N_EWT-comps.shape[0]),(0,0)))
            imfs.append(comps)
        imf = np.array(imfs, np.float32)                   # (C,N_EWT,T)

    return sig_t, psd, imf

# ---------- 滑窗 ----------
def slide(sig, tid):
    wins, gids = [], []
    for st in range(0, sig.shape[1]-WIN+1, STEP):
        wins.append(sig[:, st:st+WIN]); gids.append(tid)
    return wins, gids
# ---------- 多模态 surrogate CNN ----------
class MultiModalNet(nn.Module):
    def __init__(self, n_ch=N_CH, n_cls=2):
        super().__init__()
        self.use_psd = USE_PSD
        self.use_ewt = USE_EWT

        # 时域分支
        self.time = nn.Sequential(
            nn.Conv2d(1,16,(1,64),padding=(0,32)), nn.ReLU(),
            nn.Conv2d(16,32,(n_ch,1),groups=16),   nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())      # out = 32

        feat_dim = 32
        if self.use_psd:
            self.psd = nn.Sequential(nn.Flatten(),
                                     nn.Linear(n_ch*129,64), nn.ReLU())
            feat_dim += 64
        if self.use_ewt:
            self.ewt = nn.Sequential(
                nn.Conv3d(1,8,(1,1,64),padding=(0,0,32)), nn.ReLU(),
                nn.AdaptiveAvgPool3d((1,1,1)), nn.Flatten())
            feat_dim += 8

        self.head = nn.Linear(feat_dim, n_cls)

    def forward(self, x_t, x_psd=None, x_ewt=None):
        feats = [self.time(x_t)]
        if self.use_psd and x_psd is not None:
            feats.append(self.psd(x_psd))
        if self.use_ewt and x_ewt is not None:
            feats.append(self.ewt(x_ewt))
        return self.head(torch.cat(feats, 1))

# ---------- 时域 EEGNet（分类评估） ----------
class EEGNet(nn.Module):
    def __init__(self, C, n_cls=2, dropout=.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1,8,(1,64),padding=(0,32),bias=False), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8,16,(C,1),groups=8,bias=False),       nn.BatchNorm2d(16),
            nn.ReLU(), nn.AvgPool2d((1,4)), nn.Dropout(dropout),
            nn.Conv2d(16,16,(1,16),padding=(0,8),bias=False),nn.BatchNorm2d(16),
            nn.ReLU(), nn.AvgPool2d((1,8)), nn.Dropout(dropout),
            nn.Conv2d(16,16,1,bias=False), nn.ReLU())
        self.gap  = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(16,n_cls)
    def forward(self,x): return self.head(self.gap(self.block(x)).flatten(1))

# ---------- DataLoader ----------
def make_loader(idxs, X_t, X_psd, X_ewt, Y, G, shuffle):
    xb = [X_t[idxs]]
    if USE_PSD: xb.append(X_psd[idxs])
    if USE_EWT: xb.append(X_ewt[idxs])
    y  = torch.tensor(Y[idxs], dtype=torch.long)
    g  = torch.tensor(G[idxs], dtype=torch.long)
    ds = torch.utils.data.TensorDataset(*xb, y, g)
    return torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

# ---------- 训练 & 评估 ----------
def train_clf(net, loader, epochs, lr=1e-3):
    opt = torch.optim.Adam(net.parameters(), lr)
    cri = nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for batch in loader:
            *xs, yb, _ = batch           # 最后一个是 gid
            xs = [v.to(DEVICE) for v in xs]; yb = yb.to(DEVICE)
            opt.zero_grad()
            out = net(*xs) if isinstance(net,MultiModalNet) else net(xs[0])
            cri(out, yb).backward(); opt.step()

def eval_trial(net, loader):
    net.eval(); vote={}
    with torch.no_grad():
        for batch in loader:
            *xs, yb, gid = batch
            xs = [v.to(DEVICE) for v in xs]
            pred = (net(*xs) if isinstance(net,MultiModalNet) else net(xs[0])
                    ).argmax(1).cpu().numpy()
            for p,g in zip(pred,gid):
                vote.setdefault(int(g),[]).append(int(p))
    return np.mean([max(set(v),key=v.count)==int(k) for k,v in vote.items()])

# ---------- 主流程 ----------
def main(k_top=[4,8,16,24,32], n_samples=800):

    res_all, gkf = {}, GroupKFold(10)

    for subj_i,matf in enumerate(FILES,1):
        print(f"\n=== Subject {subj_i}/{len(FILES)}  ({matf.name}) ===")

        # -------- 读入整个 subject trials --------
        m   = loadmat(matf,simplify_cells=True)
        key = next(k for k in m if k.endswith('last_beep'))
        trials_t, trials_p, trials_e, labels = [],[],[],[]
        for cls, trials in enumerate(m[key]):
            for tr in trials:
                t,p,e = preprocess(tr)
                trials_t.append(t); trials_p.append(p); trials_e.append(e)
                labels.append(cls)
        trials_t = np.asarray(trials_t)
        labels   = np.asarray(labels,dtype=int)

        # -------- 划窗 --------
        X_t, X_psd, X_ewt, Yn, Gn = [],[],[],[],[]
        for tid,(t,p,e,l) in enumerate(zip(trials_t,trials_p,trials_e,labels)):
            wins,gids = slide(t,tid)
            X_t.extend(wins); Yn.extend([l]*len(wins)); Gn.extend(gids)
            if USE_PSD: X_psd.extend([p]*len(wins))
            if USE_EWT: X_ewt.extend([e]*len(wins))

        X_t   = torch.tensor(np.stack(X_t)[:,None,:,:], device=DEVICE)
        if USE_PSD:
            X_psd = torch.tensor(np.stack(X_psd), device=DEVICE)      # (n,C,F)
        else: X_psd=None
        if USE_EWT:
            X_ewt = torch.tensor(np.stack(X_ewt)[:,None,:,:,:], device=DEVICE) # (n,1,C,N,T)
        else: X_ewt=None
        Yn = np.asarray(Yn); Gn = np.asarray(Gn)

        # -------- Baseline (时域) --------
        base_scores=[]
        for tr,te in gkf.split(X_t.cpu().numpy(),Yn,groups=Gn):
            dl_tr = make_loader(tr,X_t,X_psd,X_ewt,Yn,Gn,True)
            dl_te = make_loader(te,X_t,X_psd,X_ewt,Yn,Gn,False)
            net   = EEGNet(N_CH).to(DEVICE)
            train_clf(net, dl_tr, EPOCH_BASE)
            base_scores.append(eval_trial(net, dl_te))
        b_avg, b_std = float(np.mean(base_scores)), float(np.std(base_scores))
        print(f"Baseline = {b_avg:.3f} ± {b_std:.3f}")
        res_sub = {"baseline":[b_avg,b_std]}

        if b_avg < THR_BASE: res_all[f"sub{subj_i:02d}"]=res_sub; continue

        # -------- surrogate 多模态训练 --------
        sur_scores=[]
        for tr,te in gkf.split(X_t.cpu().numpy(),Yn,groups=Gn):
            dl_tr = make_loader(tr,X_t,X_psd,X_ewt,Yn,Gn,True)
            sur   = MultiModalNet(N_CH).to(DEVICE)
            train_clf(sur, dl_tr, EPOCH_BASE)
            sur_scores.append(1)     # 这里只训练一次取 shap，用不到分数

        # -------- Kernel-SHAP (只用时域) --------
        samp = np.random.choice(len(X_t), SHAP_SAMP, replace=False)
        back = X_t[samp[:32]].cpu().numpy().reshape(32,-1)
        expl = X_t[samp].cpu().numpy().reshape(SHAP_SAMP,-1)

        def pred(arr2d):
            x = torch.tensor(arr2d.reshape(-1,1,N_CH,T_LEN), device=DEVICE)
            with torch.no_grad(): out = sur.time(x)
            return torch.softmax(sur.head(out),1).cpu().numpy()

        shap_exp = shap.KernelExplainer(pred, back)
        sv = shap_exp.shap_values(expl, nsamples=128)        # list[n_cls]

        sv_arr = np.stack(sv,0).reshape(len(sv),SHAP_SAMP,N_CH,T_LEN)
        imp = np.mean(np.abs(sv_arr), axis=(0,1,3))
        imp = imp/imp.max()
        order = np.argsort(-imp)

        # -------- Top-K 评估 --------
        def eval_topK(K):
            idx = torch.tensor(order[:K], dtype=torch.long, device=DEVICE)
            Xt_sel = X_t.index_select(2, idx)
            scores=[]
            for tr,te in gkf.split(Xt_sel.cpu().numpy(),Yn,groups=Gn):
                dl_tr = make_loader(tr,Xt_sel,None,None,Yn,Gn,True)
                dl_te = make_loader(te,Xt_sel,None,None,Yn,Gn,False)
                clf   = EEGNet(K).to(DEVICE)
                train_clf(clf, dl_tr, EPOCH_CV)
                scores.append(eval_trial(clf, dl_te))
            return float(np.mean(scores)), float(np.std(scores))

        for K in k_top:
            m,s = eval_topK(K)
            res_sub[str(K)] = [m,s]
            print(f"Top-{K:<2}= {m:.3f} ± {s:.3f}")

        res_all[f"sub{subj_i:02d}"]=res_sub

    out = ROOT/'results/neuroxai_multimodal.json'
    json.dump(res_all,open(out,'w'),indent=2)
    print("\n✔ 结果写入", out)

# ---------- CLI ----------
if __name__=='__main__':
    warnings.filterwarnings('ignore')
    main()