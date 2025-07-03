#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_neuroxai_multimodal.py
——————————————
Short/Long 二分类 — 通道选择对比
1. 60-ch baseline (时域 EEGNet)
2. 多模态 surrogate (时域+PSD+IMF) → Kernel-SHAP imp[i]
3. Top-K 时域 EEGNet（按 imp 排序）10-fold CV
输出：results/neuroxai_multimodal.json
"""
# --------------------------------------------------
import json, random, warnings
from pathlib import Path
import numpy as np, shap, torch, torch.nn as nn
from PyEMD import EMD
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, welch
from sklearn.model_selection import GroupKFold

# ---------- 全局参数 ----------
SEED, FS = 0, 256
WIN_S, STEP_S = 2.0, .5
WIN, STEP     = int(WIN_S*FS), int(STEP_S*FS)
BATCH = 128; EPOCH_BASE = 100; EPOCH_CV = 60
SHAP_SAMP = 256; THR_BASE = .60
DROP_FIXED = {0,9,32,63}
N_EWT = 5; USE_PSD = True; USE_EWT = True
N_CLASS = 2; CLASS_NAMES = ['short', 'long']
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
N_CH, T_LEN = len(keep_idx), WIN
bp_b, bp_a  = butter(4,[4,40],fs=FS,btype='band')
nt_b, nt_a  = iirnotch(60,30,fs=FS)
emd = EMD()

# ---------- 预处理 ----------
def preprocess(sig):
    sig = sig[keep_idx]
    sig = filtfilt(nt_b,nt_a,sig,axis=1)
    sig = filtfilt(bp_b,bp_a,sig,axis=1)
    sig -= sig.mean(1,keepdims=True); sig /= sig.std(1,keepdims=True)+1e-6
    sig_t = sig.astype(np.float32)

    psd = imf = None
    if USE_PSD:
        _, Pxx = welch(sig_t, fs=FS, nperseg=WIN, axis=1)
        psd = np.log(Pxx+1e-12).astype(np.float32)          # (C, F=129)
    if USE_EWT:
        imfs=[]
        for ch in sig_t:
            comp = emd(ch)[:N_EWT]
            if comp.shape[0] < N_EWT:
                comp = np.pad(comp, ((0,N_EWT-comp.shape[0]),(0,0)))
            imfs.append(comp)
        imf = np.asarray(imfs, np.float32)                  # (C,N_EWT,T)
    return sig_t, psd, imf

# ---------- 滑窗 ----------
def slide(sig, tid):
    wins,gids=[],[]
    for st in range(0,sig.shape[1]-WIN+1,STEP):
        wins.append(sig[:,st:st+WIN]); gids.append(tid)
    return wins,gids

# ---------- 网络 ----------
class MultiModalNet(nn.Module):
    def __init__(self,n_ch=N_CH,n_cls=N_CLASS):
        super().__init__()
        self.use_psd, self.use_ewt = USE_PSD, USE_EWT
        self.time = nn.Sequential(
            nn.Conv2d(1,16,(1,64),padding=(0,32)), nn.ReLU(),
            nn.Conv2d(16,32,(n_ch,1),groups=16),   nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
        feat = 32
        if USE_PSD:
            self.psd = nn.Sequential(nn.Flatten(),
                                     nn.Linear(n_ch*129,64), nn.ReLU())
            feat += 64
        if USE_EWT:
            self.ewt = nn.Sequential(
                nn.Conv3d(1,8,(1,1,64),padding=(0,0,32)), nn.ReLU(),
                nn.AdaptiveAvgPool3d((1,1,1)), nn.Flatten())
            feat += 8
        self.head = nn.Linear(feat, n_cls)
    def forward(self, x_t, x_psd=None, x_ewt=None):
        outs=[self.time(x_t)]
        if self.use_psd and x_psd is not None: outs.append(self.psd(x_psd))
        if self.use_ewt and x_ewt is not None: outs.append(self.ewt(x_ewt))
        return self.head(torch.cat(outs,1))

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

# ---------- Loader & 训练 ----------
def make_loader(idxs,Xt,Xp,Xe,Y,G,shuffle):
    xb=[Xt[idxs]]
    if USE_PSD: xb.append(Xp[idxs])
    if USE_EWT: xb.append(Xe[idxs])
    y  = torch.tensor(Y[idxs], dtype=torch.long)
    g  = torch.tensor(G[idxs], dtype=torch.long)
    ds = torch.utils.data.TensorDataset(*xb,y,g)
    return torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

def train(net,loader,epochs,lr=1e-3):
    opt=torch.optim.Adam(net.parameters(),lr); cri=nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for batch in loader:
            *xs, y, _ = batch
            xs=[v.to(DEVICE) for v in xs]; y=y.to(DEVICE)
            opt.zero_grad()
            out = net(*xs) if isinstance(net,MultiModalNet) else net(xs[0])
            cri(out,y).backward(); opt.step()

def trial_acc(net,loader,true_labels):
    net.eval(); vote={}
    with torch.no_grad():
        for batch in loader:
            *xs, _, gid = batch
            xs=[v.to(DEVICE) for v in xs]
            pred=(net(*xs) if isinstance(net,MultiModalNet) else net(xs[0])
                 ).argmax(1).cpu().numpy()
            for p,g in zip(pred,gid):
                vote.setdefault(int(g),[]).append(int(p))
    correct=sum(max(set(v),key=v.count)==true_labels[k] for k,v in vote.items())
    return correct/len(vote)
# ---------- 主 ----------
def main(k_list=[4,8,16,24,32]):
    all_res, gkf = {}, GroupKFold(10)

    for si,matf in enumerate(FILES,1):
        print(f"\n=== Subject {si}/{len(FILES)} ({matf.name}) ===")
        m = loadmat(matf,simplify_cells=True)
        key=next(k for k in m if k.endswith('last_beep'))
        trials_t,trials_p,trials_e,labels=[],[],[],[]
        for cls,trs in enumerate(m[key]):
            for tr in trs:
                t,p,e=preprocess(tr)
                trials_t.append(t); trials_p.append(p); trials_e.append(e)
                labels.append(cls)
        labels=np.asarray(labels,dtype=int)

        # build trial-id→label dict
        true_label = dict(enumerate(labels))

        # 划窗
        Xt,Xp,Xe,Yn,Gn=[],[],[],[],[]
        for tid,(t,p,e,lbl) in enumerate(zip(trials_t,trials_p,trials_e,labels)):
            wins,gids=slide(t,tid)
            Xt.extend(wins); Yn.extend([lbl]*len(wins)); Gn.extend(gids)
            if USE_PSD: Xp.extend([p]*len(wins))
            if USE_EWT: Xe.extend([e]*len(wins))
        Xt=torch.tensor(np.stack(Xt)[:,None,:,:],device=DEVICE)
        if USE_PSD: Xp=torch.tensor(np.stack(Xp),device=DEVICE)
        else: Xp=None
        if USE_EWT: Xe=torch.tensor(np.stack(Xe)[:,None,:,:,:],device=DEVICE)
        else: Xe=None
        Yn=np.asarray(Yn); Gn=np.asarray(Gn)

        # baseline
        base=[]
        for tr,te in gkf.split(Xt.cpu().numpy(),Yn,groups=Gn):
            dl_tr=make_loader(tr,Xt,Xp,Xe,Yn,Gn,True)
            dl_te=make_loader(te,Xt,Xp,Xe,Yn,Gn,False)
            net=EEGNet(N_CH).to(DEVICE)
            train(net,dl_tr,EPOCH_BASE)
            base.append(trial_acc(net,dl_te,true_label))
        b_m,b_s=float(np.mean(base)),float(np.std(base))
        print(f"Baseline = {b_m:.3f} ± {b_s:.3f}")
        sub={'baseline':[b_m,b_s]}
        if b_m<THR_BASE: all_res[f'sub{si:02d}']=sub; continue

        # surrogate
        sur=MultiModalNet(N_CH).to(DEVICE)
        full_dl=make_loader(np.arange(len(Xt)),Xt,Xp,Xe,Yn,Gn,True)
        train(sur,full_dl,EPOCH_BASE)

        # SHAP (时域)
        samp=np.random.choice(len(Xt),SHAP_SAMP,False)
        back=Xt[samp[:32]].cpu().numpy().reshape(32,-1)
        expl=Xt[samp].cpu().numpy().reshape(-1,N_CH*T_LEN)
        def pred(a):
            x=torch.tensor(a.reshape(-1,1,N_CH,T_LEN),device=DEVICE)
            with torch.no_grad():
                feat=sur.time(x); out=sur.head(feat)
            return torch.softmax(out,1).cpu().numpy()
        shap_exp=shap.KernelExplainer(pred,back)
        sv=shap_exp.shap_values(expl,nsamples=128)          # list[2]
        sv=np.stack(sv,0).reshape(N_CLASS,len(expl),N_CH,T_LEN)
        imp=np.mean(np.abs(sv),axis=(0,1,3)); imp/=imp.max()
        order=np.argsort(-imp)

        # Top-K
        def eval_K(K):
            idx=torch.tensor(order[:K],dtype=torch.long,device=DEVICE)
            Xt_sel=Xt.index_select(2,idx)
            scores=[]
            for tr,te in gkf.split(Xt_sel.cpu().numpy(),Yn,groups=Gn):
                dl_tr=make_loader(tr,Xt_sel,None,None,Yn,Gn,True)
                dl_te=make_loader(te,Xt_sel,None,None,Yn,Gn,False)
                clf=EEGNet(K).to(DEVICE)
                train(clf,dl_tr,EPOCH_CV)
                scores.append(trial_acc(clf,dl_te,true_label))
            return float(np.mean(scores)),float(np.std(scores))
        for K in k_list:
            m,s=eval_K(K)
            sub[str(K)]=[m,s]
            print(f"Top-{K:<2}= {m:.3f} ± {s:.3f}")
        all_res[f'sub{si:02d}']=sub

    out=ROOT/'results/neuroxai_multimodal.json'
    json.dump(all_res,open(out,'w'),indent=2)
    print('\n✔ 结果写入',out)

# ---------- CLI ----------
if __name__=='__main__':
    warnings.filterwarnings('ignore')
    main()