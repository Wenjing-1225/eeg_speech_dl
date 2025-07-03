#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_neuroxai_multimodal.py
--------------------------
Short-/Long-words 二分类

Pipeline
1. 60-ch baseline (时域 EEGNet)
2. 多模态 surrogate (时域 + PSD + IMF)  →  Kernel-SHAP 通道重要度 imp[i]
3. 仅时域 EEGNet   Top-K(4/8/16/24/32)  → 10-fold CV
结果保存：results/neuroxai_multimodal.json
"""
# ==================================================
import json, random, warnings
from pathlib import Path

import numpy as np
import shap
import torch, torch.nn as nn
from PyEMD import EMD                       # pip install EMD-signal
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, welch
from sklearn.model_selection import GroupKFold
# ---------------- 全局参数 ----------------
SEED, FS = 0, 256
WIN_S, STEP_S = 2.0, .5
WIN, STEP     = int(WIN_S*FS), int(STEP_S*FS)
BATCH = 128;   EPOCH_BASE = 100;  EPOCH_CV = 60
SHAP_SAMP = 256;  THR_BASE = .60
DROP_FIXED = {0,9,32,63}
USE_PSD = True;  USE_EWT = True;  N_EWT = 5          # IMF 个数
N_CLASS = 2;  CLASS_NAMES = ['short','long']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / 'data/Short_Long_words'
FILES = sorted(f for f in DATA.glob('*.mat') if '_8s' not in f.name)

# ---------- 通道 & 滤波 ----------
first  = loadmat(FILES[0], simplify_cells=True)
k0     = next(k for k in first if k.endswith('last_beep'))
n_tot  = first[k0][0][0].shape[0]
keep_idx = [i for i in range(n_tot) if i not in DROP_FIXED]
N_CH   = len(keep_idx)
T_LEN  = WIN                                        # 单窗点数

bp_b, bp_a = butter(4, [4,40], fs=FS, btype='band')
nt_b, nt_a = iirnotch(60, 30, fs=FS)
emd = EMD()

# Welch 得到的频点数（与 nperseg=WIN 对应）
_, Pxx_demo = welch(np.zeros(WIN), fs=FS, nperseg=256)  # 129 bins
PSD_BINS = Pxx_demo.shape[-1]                     # == 257

# ---------- 数据预处理 ----------
def preprocess(sig):
    sig = sig[keep_idx]
    sig = filtfilt(nt_b, nt_a, sig, axis=1)
    sig = filtfilt(bp_b, bp_a, sig, axis=1)
    sig -= sig.mean(1, keepdims=True); sig /= sig.std(1, keepdims=True)+1e-6
    sig_t = sig.astype(np.float32)                                 # (C,T)

    psd = None; imf = None
    if USE_PSD:
        -        _, Pxx = welch(sig_t, fs=FS, nperseg=WIN, axis=1)
        +        _, Pxx = welch(sig_t, fs=FS, nperseg=256, axis=1)
        # (C,F)
        psd = np.log(Pxx + 1e-12).astype(np.float32)             # (C,PSD_BINS)

    if USE_EWT:
        imfs=[]
        for ch in sig_t:
            comps = emd(ch)[:N_EWT]                                # (≤N_EWT,T)
            if comps.shape[0] < N_EWT:
                comps = np.pad(comps, ((0,N_EWT-comps.shape[0]),(0,0)))
            imfs.append(comps)
        imf = np.asarray(imfs, np.float32)                         # (C,N_EWT,T)
    return sig_t, psd, imf

def slide(sig, tid):
    wins,gids=[],[]
    for st in range(0, sig.shape[1]-WIN+1, STEP):
        wins.append(sig[:,st:st+WIN]); gids.append(tid)
    return wins,gids
# ---------- 网络 ----------
class MultiModalNet(nn.Module):
    def __init__(self, n_ch=N_CH, psd_bins=PSD_BINS, n_cls=2):
        super().__init__()
        self.use_psd, self.use_ewt = USE_PSD, USE_EWT
        # 时域
        self.time = nn.Sequential(
            nn.Conv2d(1,16,(1,64),padding=(0,32)), nn.ReLU(),
            nn.Conv2d(16,32,(n_ch,1),groups=16),   nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
        feat_dim = 32
        # PSD
        if self.use_psd:
            self.psd = nn.Sequential(nn.Flatten(),
                                     nn.Linear(n_ch*psd_bins,64), nn.ReLU())
            feat_dim += 64
        # IMF
        if self.use_ewt:
            self.ewt = nn.Sequential(
                nn.Conv3d(1,8,(1,1,64),padding=(0,0,32)), nn.ReLU(),
                nn.AdaptiveAvgPool3d((1,1,1)), nn.Flatten())
            feat_dim += 8
        self.head = nn.Linear(feat_dim, n_cls)

    def forward(self, x_t, x_psd=None, x_ewt=None):
        feats=[self.time(x_t)]
        if self.use_psd and x_psd is not None: feats.append(self.psd(x_psd))
        if self.use_ewt and x_ewt is not None: feats.append(self.ewt(x_ewt))
        return self.head(torch.cat(feats,1))

class EEGNet(nn.Module):
    def __init__(self,C,n_cls=N_CLASS,drop=.25):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(1,8,(1,64),padding=(0,32),bias=False), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8,16,(C,1),groups=8,bias=False),       nn.BatchNorm2d(16),
            nn.ReLU(), nn.AvgPool2d((1,4)), nn.Dropout(drop),
            nn.Conv2d(16,16,(1,16),padding=(0,8),bias=False),nn.BatchNorm2d(16),
            nn.ReLU(), nn.AvgPool2d((1,8)), nn.Dropout(drop),
            nn.Conv2d(16,16,1,bias=False), nn.ReLU())
        self.gap  = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(16, n_cls)
    def forward(self,x): return self.head(self.gap(self.block(x)).flatten(1))
# ---------- DataLoader ----------
def make_loader(idxs,Xt,Xp,Xe,Y,G,shuffle):
    xb=[Xt[idxs]]
    if USE_PSD: xb.append(Xp[idxs])
    if USE_EWT: xb.append(Xe[idxs])
    y  = torch.tensor(Y[idxs],dtype=torch.long)
    g  = torch.tensor(G[idxs],dtype=torch.long)
    ds = torch.utils.data.TensorDataset(*xb,y,g)
    return torch.utils.data.DataLoader(ds,batch_size=BATCH,shuffle=shuffle)
# ---------- 训练 & 评估 ----------
def train(net,loader,epochs,lr=1e-3):
    opt=torch.optim.Adam(net.parameters(),lr); cri=nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for batch in loader:
            *xs,y,_ = batch
            xs=[v.to(DEVICE) for v in xs]; y=y.to(DEVICE)
            opt.zero_grad()
            out = net(*xs) if isinstance(net,MultiModalNet) else net(xs[0])
            cri(out,y).backward(); opt.step()

def trial_acc(net,loader,true_lbl):
    net.eval(); vote={}
    with torch.no_grad():
        for batch in loader:
            *xs,_,gid=batch
            xs=[v.to(DEVICE) for v in xs]
            pr = (net(*xs) if isinstance(net,MultiModalNet) else net(xs[0])
                 ).argmax(1).cpu().numpy()
            for p,g in zip(pr,gid): vote.setdefault(int(g),[]).append(int(p))
    return np.mean([max(set(v),key=v.count)==true_lbl[k] for k,v in vote.items()])
# ---------- 主 ----------
def main(k_top=[4,8,16,24,32]):
    res_all, gkf = {}, GroupKFold(10)

    for si,matf in enumerate(FILES,1):
        print(f'\n=== Subject {si}/{len(FILES)} ({matf.name}) ===')
        m = loadmat(matf,simplify_cells=True)
        key = next(k for k in m if k.endswith('last_beep'))
        t_list,p_list,e_list,lbls=[],[],[],[]
        for cls,trs in enumerate(m[key]):
            for tr in trs:
                t,p,e = preprocess(tr)
                t_list.append(t); p_list.append(p); e_list.append(e)
                lbls.append(cls)
        lbls=np.asarray(lbls,dtype=int); true_dict=dict(enumerate(lbls))

        Xt,Xp,Xe,Yn,Gn=[],[],[],[],[]
        for tid,(t,p,e,l) in enumerate(zip(t_list,p_list,e_list,lbls)):
            wins,gids = slide(t,tid)
            Xt.extend(wins); Yn.extend([l]*len(wins)); Gn.extend(gids)
            if USE_PSD: Xp.extend([p]*len(wins))
            if USE_EWT: Xe.extend([e]*len(wins))
        Xt=torch.tensor(np.stack(Xt)[:,None,:,:],device=DEVICE)
        Xp=torch.tensor(np.stack(Xp),device=DEVICE) if USE_PSD else None
        Xe=torch.tensor(np.stack(Xe)[:,None,:,:,:],device=DEVICE) if USE_EWT else None
        Yn=np.asarray(Yn); Gn=np.asarray(Gn)

        # baseline
        base=[]
        for tr,te in gkf.split(Xt.cpu().numpy(),Yn,groups=Gn):
            dl_tr=make_loader(tr,Xt,Xp,Xe,Yn,Gn,True)
            dl_te=make_loader(te,Xt,Xp,Xe,Yn,Gn,False)
            net=EEGNet(N_CH).to(DEVICE); train(net,dl_tr,EPOCH_BASE)
            base.append(trial_acc(net,dl_te,true_dict))
        b_m,b_s=float(np.mean(base)),float(np.std(base))
        print(f"Baseline = {b_m:.3f} ± {b_s:.3f}")
        sub={'baseline':[b_m,b_s]}
        if b_m < THR_BASE: res_all[f'sub{si:02d}']=sub; continue

        # surrogate
        sur = MultiModalNet(N_CH, psd_bins=PSD_BINS, n_cls=2).to(DEVICE)
        full_dl = make_loader(np.arange(len(Xt)),Xt,Xp,Xe,Yn,Gn,True)
        train(sur,full_dl,EPOCH_BASE)

        # SHAP
        samp = np.random.choice(len(Xt), SHAP_SAMP, False)
        back = Xt[samp[:32]].cpu().numpy().reshape(32,-1)
        expl = Xt[samp].cpu().numpy().reshape(-1,N_CH*T_LEN)
        def pred(arr):
            x = torch.tensor(arr.reshape(-1,1,N_CH,T_LEN),device=DEVICE)
            with torch.no_grad():
                feat = sur.time(x); out = sur.head(feat)
            return torch.softmax(out,1).cpu().numpy()
        shap_exp = shap.KernelExplainer(pred, back)
        sv = shap_exp.shap_values(expl, nsamples=128)
        sv = np.stack(sv,0).reshape(N_CLASS,len(expl),N_CH,T_LEN)
        imp = np.mean(np.abs(sv),axis=(0,1,3)); imp/=imp.max()
        order = np.argsort(-imp)

        # Top-K
        def eval_K(K):
            idx = torch.tensor(order[:K],dtype=torch.long,device=DEVICE)
            Xt_sel = Xt.index_select(2, idx)
            scores=[]
            for tr,te in gkf.split(Xt_sel.cpu().numpy(),Yn,groups=Gn):
                dl_tr=make_loader(tr,Xt_sel,None,None,Yn,Gn,True)
                dl_te=make_loader(te,Xt_sel,None,None,Yn,Gn,False)
                clf = EEGNet(K).to(DEVICE); train(clf,dl_tr,EPOCH_CV)
                scores.append(trial_acc(clf,dl_te,true_dict))
            return float(np.mean(scores)),float(np.std(scores))
        for K in k_top:
            m,s=eval_K(K); sub[str(K)]=[m,s]
            print(f"Top-{K:<2}= {m:.3f} ± {s:.3f}")

        res_all[f'sub{si:02d}']=sub

    out=ROOT/'results/neuroxai_multimodal.json'
    json.dump(res_all,open(out,'w'),indent=2)
    print('\n✔ 结果写入',out)

# ---------- CLI ----------
if __name__=='__main__':
    warnings.filterwarnings('ignore')
    main()