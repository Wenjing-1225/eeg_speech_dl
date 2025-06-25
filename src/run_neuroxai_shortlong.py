#!/usr/bin/env python
# run_neuroxai_eegnet.py —— 被试内：FBCSP 预筛 + NeuroXAI 通道选择 + 自适应 EEGNet

import argparse, json, warnings, random, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from mne.decoding import CSP
from sklearn.decomposition import FastICA
from sklearn.model_selection import GroupKFold
from tqdm import tqdm, trange
from neuroxai.explanation import BrainExplainer, GlobalBrainExplainer

# ============= 超参 =============
SEED      = 0
FS        = 256
WIN_S     = 2.0;   WIN  = int(WIN_S * FS)      # 512
STEP_S    = 0.5;   STEP = int(STEP_S * FS)     # 256
EPOCHS    = 150    # 单 fold 内再次训练时减半
BATCH     = 128
FBCSP_BANDS = [(4+i*4, 8+i*4) for i in range(8)]  # 4–40 Hz 共 8 band
CANDIDATE = 30                                    # NeuroXAI 之前先筛到 30
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data/Short_Long_words"
FILES = sorted(f for f in DATA.glob("*.mat") if "_8s" not in f.name)

# ---------- 通道信息 ----------
first_mat = loadmat(FILES[0], simplify_cells=True)
key       = next(k for k in first_mat if k.endswith("last_beep"))
n_total   = first_mat[key][0][0].shape[0]                 # 64

DROP_FIXED = {0, 9, 32, 63}
keep_idx   = [i for i in range(n_total) if i not in DROP_FIXED]

orig_names = [f"Ch{i}" for i in range(n_total)]
if "ch_names" in first_mat:
    orig_names = [str(s).strip() for s in first_mat["ch_names"]][:n_total]

CHAN_NAMES = [orig_names[i] for i in keep_idx]
N_CH       = len(CHAN_NAMES)
print(f"可用通道数 = {N_CH}")

# ---------- 基本滤波 ----------
b_bp, a_bp = butter(4, [4,40], fs=FS, btype="band")
nt_b, nt_a = iirnotch(60, 30, fs=FS)
def bandpass_notch(sig):
    sig = sig[keep_idx]                       # (C,T)
    sig = filtfilt(nt_b, nt_a, sig, axis=1)
    sig = filtfilt(b_bp, a_bp, sig, axis=1)
    sig -= sig.mean(1, keepdims=True)
    sig /= sig.std (1, keepdims=True)+1e-6
    return sig.astype(np.float32)

def slide(sig):
    seg=[]
    for st in range(0, sig.shape[1]-WIN+1, STEP):
        seg.append(sig[:, st:st+WIN])
    return np.stack(seg)                      # (n_win,C,T)

# ---------- FastICA ----------
def fast_ica_all(trials, n_comp=None):
    C,T = trials[0].shape
    Xcat = np.concatenate(trials,1).T         # (samples, C)
    n_comp = C if n_comp is None else n_comp
    ica = FastICA(n_components=n_comp,
                  whiten='unit-variance',
                  max_iter=300,
                  random_state=SEED)
    _ = ica.fit_transform(Xcat)
    out=[ica.transform(sig.T).T.astype(np.float32) for sig in trials]
    return np.asarray(out)

# ---------- FBCSP 排序 ----------
# ---------- FBCSP 排序（修正版） ----------
def fbcsp_rank(trials, labels):
    """
    trials : ndarray (n_trial, C, T)
    labels : ndarray (n_trial,)
    返回电极重要度降序索引 (len = N_CH)
    """
    score = np.zeros(N_CH)

    for low, high in FBCSP_BANDS:
        b, a = butter(4, [low, high], fs=FS, btype="band")
        fb   = filtfilt(b, a, trials, axis=2)              # (N, C, T) ✓

        # ★ 保持 (N, C, T) 形状直接喂给 CSP
        csp  = CSP(n_components=2, reg='ledoit_wolf', log=False)
        csp.fit(fb.astype(np.float64), labels)

        w = csp.filters_                                   # (2, C=60)
        score += np.abs(w[0]) + np.abs(w[-1])              # (60,)

    return np.argsort(score)[::-1]                         # 长度 60                             # 长度 N_CH

# ---------- 自适应 EEGNet ----------
class EEGNet(nn.Module):
    def __init__(self, C:int, T:int, cls:int=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, (1,64), padding=(0,32), bias=False)
        self.bn1   = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,16,(C,1), groups=8, bias=False)
        self.bn2   = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1,4)); self.drop2 = nn.Dropout(.25)
        self.conv3 = nn.Conv2d(16,16,(1,16), padding=(0,8), bias=False)
        self.bn3   = nn.BatchNorm2d(16)
        self.pool3 = nn.AvgPool2d((1,8)); self.drop3 = nn.Dropout(.25)
        # 1×1 conv + GAP
        self.conv4 = nn.Conv2d(16, 16, 1, bias=False)
        self.gap   = nn.AdaptiveAvgPool2d((1,1))
        self.head  = nn.Linear(16, cls)

    def features(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x))); x = self.pool2(x); x = self.drop2(x)
        x = torch.relu(self.bn3(self.conv3(x))); x = self.pool3(x); x = self.drop3(x)
        x = torch.relu(self.conv4(x))
        return x                                 # (B,16,1,t')

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)               # (B,16)
        return self.head(x)

def train_eegnet(X, y, C, epochs, lr=1e-3):
    net  = EEGNet(C, WIN).to(DEVICE)
    opt  = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        idx = torch.randperm(len(X), device=DEVICE)
        for beg in range(0,len(idx),BATCH):
            sl = idx[beg:beg+BATCH]
            opt.zero_grad()
            loss = crit(net(X[sl]), y[sl]); loss.backward(); opt.step()
    return net

def eval_eegnet(net, X, y_np, g_np):
    net.eval(); preds=[]
    with torch.no_grad():
        for beg in range(0,len(X),BATCH):
            preds.append(net(X[beg:beg+BATCH]).argmax(1).cpu())
    preds=torch.cat(preds).numpy(); vote={}
    for p,i in zip(preds,g_np): vote.setdefault(i,[]).append(p)
    pred_trial={i:max(set(v),key=v.count) for i,v in vote.items()}
    return np.mean([pred_trial[i]==int(y_np[np.where(g_np==i)[0][0]])
                    for i in pred_trial])

# ---------- NeuroXAI ----------
def neuroxai_imp(base, trials, labels, cand_idx, n_samples):
    def clf(batch):
        batch=torch.tensor(batch[:,None,:,:],dtype=torch.float32,device=DEVICE)
        with torch.no_grad(): out = base(batch)
        return torch.softmax(out,1).cpu().numpy()
    brain = BrainExplainer(25,['short','long'])
    gexp  = GlobalBrainExplainer(brain)
    gexp.explain_instance(trials[:,:,cand_idx], labels, clf,
                          num_samples=n_samples)
    imp=np.zeros(N_CH)
    for i,c in enumerate(cand_idx):
        imp[c]=gexp.explain_global_channel_importance().get(i,0.0)
    return imp

# ---------- 主流程 ----------
def main(k_list, n_samples):
    print("Torch device:", DEVICE)
    overall={}
    for subj_idx,matf in enumerate(FILES,1):
        print(f"\n=== Subject {subj_idx}/{len(FILES)} ({matf.name}) ===")
        t0=time.time()
        # ---- 读 & ICA ----
        m=loadmat(matf,simplify_cells=True); key=next(k for k in m if k.endswith("last_beep"))
        trials=[bandpass_notch(tr) for cls in m[key] for tr in cls]
        labels=[cls     for cls,tset in enumerate(m[key]) for _ in tset]
        trials=np.asarray(trials); labels=np.asarray(labels,dtype=int)
        trials=fast_ica_all(trials)              # ICA
        print(f"Loaded & ICA: {(time.time()-t0):.1f}s  trials={len(trials)}")

        # ---- FBCSP 排序 ----
        cand_order=fbcsp_rank(trials,labels)[:CANDIDATE]
        cand_names=[CHAN_NAMES[i] for i in cand_order]
        print("FBCSP Top-30 候选:", cand_names)

        # ---- 滑窗 ----
        Xw,Y,G,gid=[],[],[],0
        for sig,lab in zip(trials,labels):
            seg=slide(sig); Xw.append(seg)
            Y.extend([lab]*len(seg)); G.extend([gid]*len(seg)); gid+=1
        X=torch.tensor(np.concatenate(Xw)[:,None,:,:],device=DEVICE)
        Yt=torch.tensor(Y,device=DEVICE)
        Yn,Gn=np.asarray(Y),np.asarray(G)

        # ---- baseline (全 60 ch) ----
        base=train_eegnet(X, Yt, N_CH, epochs=EPOCHS//3)

        # ---- NeuroXAI on candidate ----
        imp=neuroxai_imp(base, trials, labels, cand_order, n_samples)
        order=np.argsort(-imp)

        gkf=GroupKFold(10); subject_res={}
        for K in k_list:
            sel=order[:K]; sel_names=[CHAN_NAMES[i] for i in sel]
            Xi=X[:,:,sel,:]
            acc=[]
            for tr,te in gkf.split(Xi,Yn,groups=Gn):
                net=train_eegnet(Xi[tr],Yt[tr],K,epochs=EPOCHS//2)
                acc.append(eval_eegnet(net, Xi[te], Yn, Gn[te]))
            m,s=float(np.mean(acc)),float(np.std(acc))
            print(f"  Top-{K:<2}  acc={m:.3f} ± {s:.3f}  {sel_names}")
            subject_res[K]=(m,s)
        overall[f"subj{subj_idx}"]=subject_res

    # ---------- 保存 ----------
    out=ROOT/"results/subject_dep_neuroxai_eegnet.json"
    json.dump(overall, open(out,"w"), indent=2)
    print("\n✔ 所有被试完成，结果写入", out)

# ---------- CLI ----------
if __name__=="__main__":
    warnings.filterwarnings("ignore")
    pa=argparse.ArgumentParser()
    pa.add_argument("--k", type=int, nargs='+', default=[4,8,16,32],
                    help="评估的 Top-K 通道数列表")
    pa.add_argument("--n_samples", type=int, default=800,
                    help="NeuroXAI 随机扰动样本数")
    args=pa.parse_args()
    main(args.k, args.n_samples)