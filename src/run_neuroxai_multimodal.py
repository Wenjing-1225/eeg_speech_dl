#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_neuroxai_multimodal.py
==========================
Short / Long words —— 通道选择对比

Pipeline
1. 60-ch baseline EEGNet（时域）
2. 多模态 surrogate  (时域 + PSD + EWT)  ➜ Kernel-SHAP 得 imp[i]
3. 仅时域 Top-K EEGNet（按 imp 排序）  → 10-fold CV

输出: results/neuroxai_multimodal.json
"""
# --------------------------------------------------
import json, random, time, warnings
from pathlib import Path

import numpy as np
import shap
import torch
import torch.nn as nn
from PyEMD import EMD                  # pip install EMD-signal
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, welch
from sklearn.model_selection import GroupKFold
# --------------------------------------------------
# 全局配置
SEED          = 0
FS            = 256
WIN_S, STEP_S = 2.0, .5
WIN, STEP     = int(WIN_S*FS), int(STEP_S*FS)
BATCH         = 128
EPOCH_BASE    = 100
EPOCH_CV      = 60
SHAP_SAMP     = 256
THR_BASE      = .60
DROP_FIXED    = {0,9,32,63}

N_EWT         = 5      # 用多少个 IMF
USE_PSD       = True
USE_EWT       = True

N_CLASS       = 2      # ★★ 此处决定整个脚本的类别数 ★★
CLASS_NAMES   = ['short', 'long']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
# --------------------------------------------------
ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / 'data/Short_Long_words'
FILES = sorted(f for f in DATA.glob('*.mat') if '_8s' not in f.name)

# ---------- 通道 & 滤波 ----------
first   = loadmat(FILES[0], simplify_cells=True)
k0      = next(k for k in first if k.endswith('last_beep'))
n_tot   = first[k0][0][0].shape[0]
keep_idx = [i for i in range(n_tot) if i not in DROP_FIXED]
N_CH     = len(keep_idx)
T_LEN    = WIN

bp_b, bp_a = butter(4, [4, 40], fs=FS, btype='band')
nt_b, nt_a = iirnotch(60, 30, fs=FS)
emd         = EMD()                                  # IMF 分解器

# ---------- 预处理 ----------
def preprocess(sig):
    sig = sig[keep_idx]
    sig = filtfilt(nt_b, nt_a, sig, axis=1)
    sig = filtfilt(bp_b, bp_a, sig, axis=1)
    sig -= sig.mean(1, keepdims=True)
    sig /= sig.std(1, keepdims=True) + 1e-6
    sig_t = sig.astype(np.float32)                   # (C,T)

    # PSD 分支
    psd = None
    if USE_PSD:
        f, Pxx = welch(sig_t, fs=FS, nperseg=WIN, axis=1)
        psd = np.log(Pxx + 1e-12).astype(np.float32) # (C,F=129)

    # EWT / EMD 分支
    imf = None
    if USE_EWT:
        imfs = []
        for ch in sig_t:
            comps = emd(ch)[:N_EWT]                  # (≤N_EWT, T)
            if comps.shape[0] < N_EWT:               # 不足补 0
                comps = np.pad(comps, ((0, N_EWT-comps.shape[0]), (0, 0)))
            imfs.append(comps)
        imf = np.array(imfs, np.float32)             # (C, N_EWT, T)

    return sig_t, psd, imf

# ---------- 滑窗 ----------
def slide(sig, tid):
    wins, gids = [], []
    for st in range(0, sig.shape[1] - WIN + 1, STEP):
        wins.append(sig[:, st:st+WIN]); gids.append(tid)
    return wins, gids

# ---------- 多模态 surrogate CNN ----------
class MultiModalNet(nn.Module):
    def __init__(self, n_ch=N_CH, n_cls=N_CLASS):
        super().__init__()
        self.use_psd, self.use_ewt = USE_PSD, USE_EWT
        # 时域
        self.time = nn.Sequential(
            nn.Conv2d(1,16,(1,64),padding=(0,32)), nn.ReLU(),
            nn.Conv2d(16,32,(n_ch,1),groups=16),   nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())  # 32
        feat_dim = 32
        # PSD
        if self.use_psd:
            self.psd = nn.Sequential(nn.Flatten(),
                                     nn.Linear(n_ch*129,64), nn.ReLU())
            feat_dim += 64
        # IMF
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
        return self.head(torch.cat(feats,1))

# ---------- 时域 EEGNet ----------
class EEGNet(nn.Module):
    def __init__(self, C, n_cls=N_CLASS, drop=.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1,8,(1,64),padding=(0,32),bias=False), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8,16,(C,1),groups=8,bias=False),       nn.BatchNorm2d(16),
            nn.ReLU(), nn.AvgPool2d((1,4)), nn.Dropout(drop),
            nn.Conv2d(16,16,(1,16),padding=(0,8),bias=False),nn.BatchNorm2d(16),
            nn.ReLU(), nn.AvgPool2d((1,8)), nn.Dropout(drop),
            nn.Conv2d(16,16,1,bias=False), nn.ReLU())
        self.head = nn.Linear(16,n_cls)
        self.gap  = nn.AdaptiveAvgPool2d((1,1))
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
            *xs, yb, _ = batch
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
    return np.mean([max(set(v),key=v.count)==k for k,v in vote.items()])

# ---------- 主 ----------
def main(k_list=[4,8,16,24,32]):
    res_all, gkf = {}, GroupKFold(10)

    for subj_i, matf in enumerate(FILES, 1):
        print(f"\n=== Subject {subj_i}/{len(FILES)} ({matf.name}) ===")

        # ----- 读取 trial -----
        m   = loadmat(matf, simplify_cells=True)
        key = next(k for k in m if k.endswith('last_beep'))
        t_list, p_list, e_list, y_list = [],[],[],[]
        for cls, trials in enumerate(m[key]):
            for tr in trials:
                t, p, e = preprocess(tr)
                t_list.append(t); p_list.append(p); e_list.append(e)
                y_list.append(cls)
        y_list = np.asarray(y_list, dtype=int)

        # ----- 划窗 -----
        X_t, X_psd, X_ewt, Yn, Gn = [],[],[],[],[]
        for tid,(t,p,e,lbl) in enumerate(zip(t_list,p_list,e_list,y_list)):
            wins, gids = slide(t, tid)
            X_t.extend(wins); Yn.extend([lbl]*len(wins)); Gn.extend(gids)
            if USE_PSD: X_psd.extend([p]*len(wins))
            if USE_EWT: X_ewt.extend([e]*len(wins))

        X_t   = torch.tensor(np.stack(X_t)[:,None,:,:], device=DEVICE)
        if USE_PSD: X_psd = torch.tensor(np.stack(X_psd), device=DEVICE)
        else:      X_psd = None
        if USE_EWT: X_ewt = torch.tensor(np.stack(X_ewt)[:,None,:,:,:], device=DEVICE)
        else:       X_ewt = None
        Yn = np.asarray(Yn); Gn = np.asarray(Gn)

        # ----- ① Baseline -----
        base=[]
        for tr,te in gkf.split(X_t.cpu().numpy(),Yn,groups=Gn):
            dl_tr = make_loader(tr,X_t,X_psd,X_ewt,Yn,Gn,True)
            dl_te = make_loader(te,X_t,X_psd,X_ewt,Yn,Gn,False)
            net   = EEGNet(N_CH).to(DEVICE)
            train_clf(net, dl_tr, EPOCH_BASE)
            base.append(eval_trial(net, dl_te))
        b_avg,b_std = float(np.mean(base)), float(np.std(base))
        print(f"Baseline = {b_avg:.3f} ± {b_std:.3f}")
        sub_res = {'baseline':[b_avg,b_std]}
        if b_avg < THR_BASE: res_all[f'sub{subj_i:02d}']=sub_res; continue

        # ----- ② surrogate 训练 -----
        sur = MultiModalNet(N_CH).to(DEVICE)
        full_dl = make_loader(np.arange(len(X_t)), X_t, X_psd, X_ewt, Yn, Gn, True)
        train_clf(sur, full_dl, EPOCH_BASE)

        # ----- Kernel-SHAP (时域分支) -----
        samp = np.random.choice(len(X_t), SHAP_SAMP, replace=False)
        back = X_t[samp[:32]].cpu().numpy().reshape(32,-1)
        expl = X_t[samp].cpu().numpy().reshape(-1,  N_CH*T_LEN)

        def predict(arr2d):
            x = torch.tensor(arr2d.reshape(-1,1,N_CH,T_LEN), device=DEVICE)
            with torch.no_grad():
                feat = sur.time(x)
                out  = sur.head(feat)
            return torch.softmax(out,1).cpu().numpy()

        shap_exp = shap.KernelExplainer(predict, back)
        sv = shap_exp.shap_values(expl, nsamples=128)      # list[n_cls=2]
        sv_arr = np.stack(sv,0).reshape(len(sv),len(expl),N_CH,T_LEN)
        imp = np.mean(np.abs(sv_arr), axis=(0,1,3))
        imp = imp/imp.max()
        order = np.argsort(-imp)

        # ----- ③ Top-K 评估 -----
        def eval_top(K):
            idx = torch.tensor(order[:K],dtype=torch.long,device=DEVICE)
            Xt_sel = X_t.index_select(2, idx)
            scores=[]
            for tr,te in gkf.split(Xt_sel.cpu().numpy(),Yn,groups=Gn):
                dl_tr = make_loader(tr,Xt_sel,None,None,Yn,Gn,True)
                dl_te = make_loader(te,Xt_sel,None,None,Yn,Gn,False)
                clf   = EEGNet(K).to(DEVICE)
                train_clf(clf, dl_tr, EPOCH_CV)
                scores.append(eval_trial(clf, dl_te))
            return float(np.mean(scores)), float(np.std(scores))

        for K in k_list:
            m,s = eval_top(K)
            sub_res[str(K)] = [m,s]
            print(f"Top-{K:<2} = {m:.3f} ± {s:.3f}")

        res_all[f'sub{subj_i:02d}'] = sub_res

    out = ROOT/'results/neuroxai_multimodal.json'
    json.dump(res_all, open(out,'w'), indent=2)
    print('\n✔ 结果写入', out)

# ---------- CLI ----------
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()