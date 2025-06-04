#!/usr/bin/env python
# csp_rank_eegnet_curve.py  ---  accuracy vs #channels
# ----------------------------------------------------
import numpy as np, torch, torch.nn as nn
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from pathlib import Path
from mne.decoding import CSP
from sklearn.model_selection import GroupKFold
import warnings; warnings.filterwarnings("ignore")

# -------- Config --------
SEED      = 0
FS        = 256
WIN_S     = 2.0;   WIN  = int(WIN_S*FS)
STEP_S    = 0.5;   STEP = int(STEP_S*FS)
EPOCHS    = 150
BATCH     = 128
K_LIST    = [4, 8, 16, 32, 60]     # 通道数比较
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(SEED); torch.manual_seed(SEED)

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data/Short_Long_words"
FILES = sorted(f for f in DATA.glob("*.mat") if "_8s" not in f.name)

# -------- Pre-filters 4–40 + Notch --------
bp_b, bp_a = butter(4, [4,40], fs=FS, btype='band')
nt_b, nt_a = iirnotch(60, 30, fs=FS)
def preprocess(sig):
    sig = filtfilt(nt_b, nt_a, sig, axis=1)
    sig = filtfilt(bp_b, bp_a, sig, axis=1)
    sig -= sig.mean(axis=1, keepdims=True)
    sig /= sig.std(axis=1, keepdims=True) + 1e-6
    return sig.astype(np.float32)

def slide(sig):
    wins=[]
    for st in range(0, sig.shape[1]-WIN+1, STEP):
        wins.append(sig[:, st:st+WIN])
    return np.stack(wins)                  # (n_win,C,T)

# -------- EEGNet (通道自适应) --------
class EEGNet(nn.Module):
    def __init__(self, C, T, n_cls=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1,8,(1,64),padding=(0,32),bias=False)
        self.bn1   = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,16,(C,1),groups=8,bias=False)
        self.bn2   = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1,4)); self.drop2 = nn.Dropout(.25)
        self.conv3 = nn.Conv2d(16,16,(1,16),padding=(0,8),bias=False)
        self.bn3   = nn.BatchNorm2d(16)
        self.pool3 = nn.AvgPool2d((1,8)); self.drop3 = nn.Dropout(.25)
        out_len = ((T+64-1)//1 - 63)//4   # conv1+pad -> pool2
        out_len = ((out_len+16-1)//1 -15)//8
        self.fc = nn.Linear(16*out_len, n_cls)
    def forward(self,x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x); x = self.drop2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x); x = self.drop3(x)
        x = x.flatten(1)
        return self.fc(x)

# -------- 主循环 --------
curve = {k:[] for k in K_LIST}

for matf in FILES:
    mat  = loadmat(matf, simplify_cells=True)
    key  = next(k for k in mat if k.endswith("last_beep"))
    raw  = mat[key]                          # (2, trials)

    # --- 60-ch trials ---
    trials=[]; labels=[]
    for cls, tset in enumerate(raw):
        for tr in tset:
            sig = preprocess(np.delete(tr, [0,9,32,63], axis=0))  # 60×L
            trials.append(sig); labels.append(cls)
    trials, labels = np.asarray(trials), np.asarray(labels)

    # --- 平衡 ---
    i0,i1=np.where(labels==0)[0],np.where(labels==1)[0]
    n=min(len(i0),len(i1)); keep=np.sort(np.hstack([i0[:n],i1[:n]]))
    trials, labels = trials[keep], labels[keep]

    # --- 计算 CSP 分数 ---
    csp=CSP(2, reg='ledoit_wolf').fit(trials.astype(np.float64), labels)
    wmax,wmin = csp.filters_[0], csp.filters_[-1]
    score = np.abs(wmax)+np.abs(wmin)
    order = np.argsort(score)[::-1]          # 由高到低

    # --- 遍历 K ---
    for K in K_LIST:
        sel = order[:K]                      # 选前 K 通道

        # ---- 构造滑窗样本 ----
        X, y, g = [], [], []
        gid=0
        for sig, lab in zip(trials, labels):
            wins = slide(sig[sel])           # (n_win,K,WIN)
            X.append(wins)
            y.extend([lab]*len(wins))
            g.extend([gid]*len(wins))
            gid += 1
        X=np.concatenate(X)                 # (N,K,WIN)
        X=X[:,None,:,:]                     # => (N,1,K,T)
        y=np.asarray(y); g=np.asarray(g)

        # ---- 10-折 CV ----
        acc_fold=[]; gkf=GroupKFold(10)
        for tr,te in gkf.split(X,y,groups=g):
            Xtr=torch.tensor(X[tr],device=DEVICE)
            ytr=torch.tensor(y[tr],device=DEVICE)
            Xte=torch.tensor(X[te],device=DEVICE)
            yte=y[te]; gte=g[te]

            net=EEGNet(C=K,T=WIN).to(DEVICE)
            opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
            lossf=nn.CrossEntropyLoss()

            net.train()
            for ep in range(EPOCHS):
                perm=torch.randperm(len(Xtr),device=DEVICE)
                for beg in range(0,len(perm),BATCH):
                    sl=perm[beg:beg+BATCH]
                    opt.zero_grad()
                    loss=lossf(net(Xtr[sl]), ytr[sl])
                    loss.backward(); opt.step()

            net.eval();  pred=[]
            with torch.no_grad():
                for beg in range(0,len(Xte),BATCH):
                    sl=slice(beg,beg+BATCH)
                    pred.append(net(Xte[sl]).argmax(1).cpu())
            pred=torch.cat(pred).numpy()

            # trial-level多数表决
            vote={}
            for p,id in zip(pred,gte): vote.setdefault(id,[]).append(p)
            pred_trial={id:np.bincount(v).argmax() for id,v in vote.items()}
            true_trial={id:int(y[np.where(g==id)[0][0]]) for id in pred_trial}
            acc_fold.append(np.mean([pred_trial[t]==true_trial[t] for t in pred_trial]))
        curve[K].append(np.mean(acc_fold))

# -------- 输出结果 --------
print("\n#Channels | acc_mean ± std")
print("---------------------------")
for k in K_LIST:
    arr=np.asarray(curve[k])
    print(f"{k:>9} | {arr.mean():.3f} ± {arr.std():.3f}")