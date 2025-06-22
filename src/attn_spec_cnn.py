#!/usr/bin/env python
# attn_spec_cnn.py  ·  SE-CNN + Spectrogram + channel-attention ranking
# -------------------------------------------------------------------
import os, random, warnings, numpy as np, torch, torch.nn as nn
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, stft
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ============ 0. 绝对确定性 ============ #
SEED=0
os.environ["PYTHONHASHSEED"]=str(SEED)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============ 1. 全局参数 ============== #
FS        = 256
WIN_S     = 5        # 秒
WIN       = FS*WIN_S
N_PERSEG  = 128
N_OVERLAP = 64
F_BINS    = N_PERSEG//2 + 1          # 65
STEP_BS   = 256
EPOCHS    = 120
EARLY_PAT = 15
K_LIST    = [0, 4, 8, 16, 32, 60]    # 0/60 = baseline
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
print("💻  device:", DEVICE)

ROOT   = Path(__file__).resolve().parent.parent
DATA   = ROOT / "data/Short_Long_words"
FILES  = sorted(f for f in DATA.glob("*.mat") if "_8s" not in f.name)
EOG_CH = [0,9,32,63]

# ============ 2. 预处理&时-频 ============ #
bp_b, bp_a = butter(4, [8,70], fs=FS, btype='band')
nt_b, nt_a = iirnotch(60, 30, fs=FS)

def filt(sig):
    sig = filtfilt(bp_b, bp_a,  sig, axis=1)
    sig = filtfilt(nt_b, nt_a, sig, axis=1)
    sig -= sig.mean(axis=1, keepdims=True)
    sig /= sig.std(axis=1, keepdims=True)+1e-6
    return sig.astype(np.float32)

def spec_power(ch_sig):
    """单通道→ log-power spectrogram  shape=(F,T)"""
    f, t, Z = stft(ch_sig, fs=FS, nperseg=N_PERSEG, noverlap=N_OVERLAP,
                   boundary=None, padded=False)
    P = np.log(np.abs(Z)**2 + 1e-12)
    return P.astype(np.float32)              # (F_BINS, T_LEN)

# ============ 3.  注意力 CNN ============ #
class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        mid = max(ch//r, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),         # (B,C,1,1,1)
            nn.Flatten(),
            nn.Linear(ch, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, ch, bias=False),  nn.Sigmoid())
    def forward(self, x):
        w = self.fc(x).view(x.size(0), -1, 1, 1, 1)
        return x * w, w                    # 返回权重

class SpecAttnCNN(nn.Module):
    def __init__(self, C, F=F_BINS, T_step=WIN):
        super().__init__()
        # Conv3d: treat (channel, freq, time) = 3D “图像”
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(C,3,5),
                               padding=(0,1,2), bias=False)
        self.bn1   = nn.BatchNorm3d(16)
        self.se1   = SEBlock(16)
        self.pool1 = nn.AdaptiveAvgPool3d((16, 1, 1))  # 融合 freq/time
        self.drop1 = nn.Dropout(.3)
        self.fc    = nn.Linear(16, 2)

        self.attn_collector = []           # 保存 SE 权重

    def forward(self, x, collect=False):
        # x : (B,1,C,F,T)
        x = torch.relu(self.bn1(self.conv1(x)))   # (B,16,C?,F',T')
        x, w = self.se1(x)                        # SE 权重 (B,16,1,1,1)
        if collect:
            self.attn_collector.append(w.detach().mean(0))  # (16,1,1,1)
        x = self.pool1(x)                          # (B,16,1,1,1)
        x = self.drop1(x)
        return self.fc(x.flatten(1))

# ============ 4. 训练一个 SE-CNN 得到通道重要度 ============ #
def train_collect_attention(trials, labels):
    """返回 attention importance (length = 60)"""
    X = []
    for sig in trials:                # sig=(60,1280)
        spec = np.stack([spec_power(ch) for ch in sig])  # (60,F,T)
        X.append(spec)
    X = torch.tensor(np.asarray(X)[:,None,:,:,:], device=DEVICE)  # (N,1,60,F,T)
    y = torch.tensor(labels, device=DEVICE)

    net = SpecAttnCNN(C=60).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    sched= ReduceLROnPlateau(opt, mode='max', factor=0.3,
                             patience=5, verbose=False)
    lossf= nn.CrossEntropyLoss()
    best, wait = 0, 0
    for ep in range(EPOCHS):
        net.train()
        perm=torch.randperm(len(X), device=DEVICE)
        for beg in range(0,len(perm),STEP_BS):
            idx=perm[beg:beg+STEP_BS]
            opt.zero_grad(); lossf(net(X[idx]), y[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            out = net(X, collect=True)
            acc = (out.argmax(-1)==y).float().mean().item()
        sched.step(acc)
        if acc>best+1e-3: best,wait=acc,0
        else: wait+=1
        if wait>=EARLY_PAT: break

    # ---- 收集到的 SE 权重求均值 ----
    alpha = torch.stack(net.attn_collector).mean(0).squeeze().cpu().numpy()  # (16,)
    # conv1 输出通道=16，对应输入 60 通道的 group-kernel；这里简单平均映射回 60
    importance = np.repeat(alpha, 4)[:60]
    return np.argsort(importance)[::-1]     # 从大到小排序索引

# ============ 5. 评估 Top-K (10-fold DWT-DNN) ========= #
def eval_topk(sel_idx, trials, labels):
    vote = len(sel_idx)
    feats = np.stack([[spec_power(sig[ch]) for ch in sel_idx] for sig in trials])
    # reshape → (N,1,K,F,T) 作为 mini-CNN 输入
    feats = feats[:,None,:,:,:]
    cv = StratifiedKFold(10, shuffle=True, random_state=SEED)
    acc_all=[]
    for tr, te in cv.split(np.arange(len(feats)), labels):
        Xtr = torch.tensor(feats[tr], device=DEVICE)
        ytr = torch.tensor(labels[tr], device=DEVICE)
        Xte = torch.tensor(feats[te], device=DEVICE)
        yte = labels[te]

        net = SpecAttnCNN(C=len(sel_idx)).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        lossf= nn.CrossEntropyLoss()
        best, wait=0,0
        for ep in range(EPOCHS):
            net.train()
            perm=torch.randperm(len(Xtr), device=DEVICE)
            for beg in range(0,len(perm),STEP_BS):
                idx=perm[beg:beg+STEP_BS]
                opt.zero_grad(); lossf(net(Xtr[idx]), ytr[idx]).backward(); opt.step()
            # 简易 early-stop
            if ep%10==0:
                net.eval(); with torch.no_grad():
                    acc=(net(Xtr).argmax(-1)==ytr).float().mean().item()
                if acc>best: best,wait=acc,0
                else: wait+=1
                if wait>=EARLY_PAT: break

        net.eval(); pred=[]
        with torch.no_grad():
            for beg in range(0,len(Xte),STEP_BS):
                sl=slice(beg,beg+STEP_BS)
                pred.append(net(Xte[sl]).argmax(-1).cpu())
        pred=torch.cat(pred).numpy()
        acc_all.append(accuracy_score(yte, pred))
    return np.mean(acc_all)

# ============ 6. 主流程 ================ #
curve = {k:[] for k in K_LIST}

for fmat in FILES:
    mat = loadmat(fmat, simplify_cells=True)
    key = next(k for k in mat if k.endswith("last_beep"))
    raw = mat[key]

    TRS, LAB = [], []
    for cls, trials in enumerate(raw):
        for ep in trials:
            TRS.append(filt(np.delete(ep[:,:WIN], EOG_CH, 0))); LAB.append(cls)
    TRS, LAB = np.asarray(TRS), np.asarray(LAB)

    # 类均衡
    n=min((LAB==0).sum(),(LAB==1).sum())
    idx=np.hstack([np.where(LAB==0)[0][:n], np.where(LAB==1)[0][:n]])
    TRS, LAB = TRS[idx], LAB[idx]

    ranked_idx = train_collect_attention(TRS, LAB)

    for k in K_LIST:
        sel = list(range(60)) if k==0 or k==60 else ranked_idx[:k]
        acc = eval_topk(sel, TRS, LAB)
        curve[k].append(acc)
    print("✔", fmat.name)

# ============ 7. 输出 ================ #
print("\n#Channels |  acc_mean ± std")
print("-----------------------------")
for k in K_LIST:
    arr=np.asarray(curve[k]); print(f"{k:>9} | {arr.mean():.3f} ± {arr.std():.3f}")