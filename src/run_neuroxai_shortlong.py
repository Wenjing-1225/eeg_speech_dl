#!/usr/bin/env python
# run_neuroxai_eegnet.py —— 被试内：baseline→(可选)通道选择→自适应 EEGNet

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
from tqdm import tqdm, trange

# ============= 全局超参 =============
SEED          = 0
FS            = 256
WIN_S, STEP_S = 2.0, .5
WIN,  STEP    = int(WIN_S*FS), int(STEP_S*FS)
EPOCH_BASE    = 100          # 60-ch baseline
EPOCH_CV      = 60           # CV & NeuroXAI
BATCH         = 128
THR_BASE      = .60          # baseline 低于此阈值则跳过通道选择
FBCSP_BANDS   = [(4+i*4, 8+i*4) for i in range(8)]   # 4–40 Hz
CANDIDATE     = 30           # 送给 NeuroXAI 的预筛数
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reproducibility
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data/Short_Long_words"
FILES = sorted(f for f in DATA.glob("*.mat") if "_8s" not in f.name)

# ---------- 通道信息 ----------
first = loadmat(FILES[0], simplify_cells=True)
k0    = next(k for k in first if k.endswith("last_beep"))
n_tot = first[k0][0][0].shape[0]                    # 64

DROP_FIXED = {0,9,32,63}
keep_idx   = [i for i in range(n_tot) if i not in DROP_FIXED]

orig_names = [f"Ch{i}" for i in range(n_tot)]
if "ch_names" in first:
    orig_names = [str(s).strip() for s in first["ch_names"]][:n_tot]

CHAN_NAMES = [orig_names[i] for i in keep_idx]
N_CH       = len(CHAN_NAMES)
print(f"可用通道数 = {N_CH}  |  Torch device = {DEVICE}")

# ---------- 滤波 ----------
bp_b, bp_a = butter(4, [4,40],  fs=FS, btype="band")
nt_b, nt_a = iirnotch(60, 30,   fs=FS)

def preprocess(sig):
    sig = sig[keep_idx]                                 # (C,T)
    sig = filtfilt(nt_b, nt_a, sig, axis=1)
    sig = filtfilt(bp_b, bp_a, sig, axis=1)
    sig -= sig.mean(1, keepdims=True)
    sig /= sig.std (1, keepdims=True) + 1e-6
    return sig.astype(np.float32)

def slide(sig):
    wins = []
    for st in range(0, sig.shape[1]-WIN+1, STEP):
        wins.append(sig[:, st:st+WIN])
    return np.stack(wins)                              # (n,C,T)

# ---------- FastICA（所有 trial 一次 Fit） ----------
def fast_ica_all(trials, n_comp=None):
    C,T = trials[0].shape
    X   = np.concatenate(trials,1).T                  # (S,C)
    n_comp = C if n_comp is None else n_comp
    ica = FastICA(n_components=n_comp, whiten='unit-variance',
                  random_state=SEED, max_iter=300)
    _   = ica.fit_transform(X)
    return np.asarray([ica.transform(s.T).T.astype(np.float32) for s in trials])

# ---------- FBCSP 排序 ----------
def fbcsp_rank(trials, labels):
    score = np.zeros(N_CH)
    for low,high in FBCSP_BANDS:
        b,a = butter(4, [low,high], fs=FS, btype='band')
        fb  = filtfilt(b,a,trials,axis=2)
        csp = CSP(n_components=2, reg='ledoit_wolf', log=False)
        csp.fit(fb.astype(np.float64), labels)
        w   = csp.filters_                            # (2,C)
        score += np.abs(w[0]) + np.abs(w[-1])
    return np.argsort(score)[::-1]                    # 60→0

# ---------- 自适应 EEGNet ----------
class EEGNet(nn.Module):
    def __init__(self, C:int, n_cls:int=2):
        super().__init__()
        self.block = nn.Sequential(                   # ①–③ 与原同
            nn.Conv2d(1,8,(1,64),padding=(0,32),bias=False), nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8,16,(C,1),groups=8,bias=False), nn.BatchNorm2d(16),
            nn.ReLU(), nn.AvgPool2d((1,4)), nn.Dropout(.25),
            nn.Conv2d(16,16,(1,16),padding=(0,8),bias=False), nn.BatchNorm2d(16),
            nn.ReLU(), nn.AvgPool2d((1,8)), nn.Dropout(.25),
            nn.Conv2d(16,16,1,bias=False), nn.ReLU()   # ④ 1×1 conv
        )
        self.gap  = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(16, n_cls)

    def forward(self,x):
        x = self.block(x)           # (B,16,?,?)
        x = self.gap(x).flatten(1)  # (B,16)
        return self.head(x)

def train_net(X, y, epochs):
    net = EEGNet(X.shape[2]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, 50, 0.5)
    cri = nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        idx = torch.randperm(len(X), device=DEVICE)
        for beg in range(0,len(idx),BATCH):
            sl = idx[beg:beg+BATCH]
            opt.zero_grad()
            loss = cri(net(X[sl]), y[sl]); loss.backward(); opt.step()
        sch.step()
    return net

def eval_net(net,X,y_np,g_np):
    net.eval(); preds=[]
    with torch.no_grad():
        for beg in range(0,len(X),BATCH):
            preds.append(net(X[beg:beg+BATCH]).argmax(1).cpu())
    preds=np.concatenate(preds); vote={}
    for p,i in zip(preds,g_np): vote.setdefault(i,[]).append(p)
    trial_pred={i:max(set(v),key=v.count) for i,v in vote.items()}
    return np.mean([trial_pred[i]==int(y_np[g_np==i][0]) for i in trial_pred])

# ---------- NeuroXAI importance ----------
def neuroxai_imp(base,trials,labels,cand_idx,n_samples):
    def clf(b):
        t = torch.tensor(b[:,None,:,:],dtype=torch.float32,device=DEVICE)
        with torch.no_grad(): out = base(t)
        return torch.softmax(out,1).cpu().numpy()
    brain=BrainExplainer(25,['short','long'])
    gexp = GlobalBrainExplainer(brain)
    gexp.explain_instance(trials[:,:,cand_idx],labels,clf,n_samples=n_samples)
    imp = np.zeros(N_CH)
    imp[cand_idx] = [gexp.explain_global_channel_importance().get(i,0.)
                     for i in range(len(cand_idx))]
    return imp

# ---------- 主流程 ----------
def main(k_list,n_samples, use_ica=True):
    all_res={}
    for subj_i,matf in enumerate(FILES,1):
        print(f"\n=== Subject {subj_i}/{len(FILES)}  ({matf.name}) ===")
        t0=time.time()

        # 读取并预处理
        m   = loadmat(matf,simplify_cells=True)
        key = next(k for k in m if k.endswith("last_beep"))
        trials=[preprocess(tr) for cls in m[key] for tr in cls]
        labels=[cls for cls,tset in enumerate(m[key]) for _ in tset]
        trials=np.asarray(trials); labels=np.asarray(labels,dtype=int)

        if use_ica:
            trials=fast_ica_all(trials)
        print(f"Loaded  trials={len(trials)}  用时 {time.time()-t0:.1f}s")

        # baseline 60-ch
        Xw,Y,G,gid=[],[],[],0
        for sig,lab in zip(trials,labels):
            seg=slide(sig); Xw.append(seg)
            Y.extend([lab]*len(seg)); G.extend([gid]*len(seg)); gid+=1
        X = torch.tensor(np.concatenate(Xw)[:,None,:,:],device=DEVICE)
        Yt= torch.tensor(Y,device=DEVICE)
        Yn,Gn=np.asarray(Y),np.asarray(G)

        print("Train 60-ch baseline …")
        base = train_net(X,Yt,EPOCH_BASE)
        acc0 = eval_net(base,X,Yn,Gn)
        print(f"Baseline acc = {acc0:.3f}")

        sub_res={"baseline":acc0}
        if acc0 < THR_BASE:
            print(f"  ↳ baseline < {THR_BASE}, 跳过通道选择")
            all_res[f"sub{subj_i}"]=sub_res; continue

        # FBCSP 预筛
        cand_idx = fbcsp_rank(trials,labels)[:CANDIDATE]
        print("FBCSP 30 候选:",[CHAN_NAMES[i] for i in cand_idx])

        # NeuroXAI
        print("NeuroXAI importance …")
        imp = neuroxai_imp(base,trials,labels,cand_idx,n_samples)
        order=np.argsort(-imp)

        gkf=GroupKFold(10)
        for K in k_list:
            sel=order[:K]
            Xi=X[:,:,sel,:]
            acc=[]
            for tr,te in gkf.split(Xi,Yn,groups=Gn):
                net=train_net(Xi[tr],Yt[tr],EPOCH_CV)
                acc.append(eval_net(net,Xi[te],Yn,Gn[te]))
            sub_res[str(K)]=(float(np.mean(acc)),float(np.std(acc)))
            print(f"  Top-{K:<2}: {sub_res[str(K)][0]:.3f} ± {sub_res[str(K)][1]:.3f}")
        all_res[f"sub{subj_i}"]=sub_res

    # 保存
    out = ROOT/"results/subject_dep_neuroxai_eegnet.json"
    json.dump(all_res, open(out,"w"), indent=2)
    print("\n✔ 完成，结果写入", out)

# ---------- CLI ----------
if __name__=="__main__":
    warnings.filterwarnings("ignore")
    pa = argparse.ArgumentParser()
    pa.add_argument("--k", type=int, nargs='+', default=[4,8,16,32],
                    help="评估的 Top-K 列表")
    pa.add_argument("--n_samples", type=int, default=800,
                    help="NeuroXAI 随机扰动样本数")
    pa.add_argument("--no_ica", action="store_true",
                    help="不给 trial 做 ICA")
    args = pa.parse_args()
    main(args.k, args.n_samples, use_ica=not args.no_ica)