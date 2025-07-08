#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_hilbert_spectro_quality.py
==============================
Hilbert-Spectrum  + 15-layer CNN  + 信号质量关联分析
--------------------------------------------------
• 数据集   : data/Short_words/*.mat     (与 Agarwal & Kumar 论文同格式)
• 任务     : 5-class imagined-word
• 输出     : 10-fold CV 准确率、df_metrics.csv、corr_metrics.csv
"""

# ========== 依赖 ==========
from pathlib import Path
import random, warnings, io, math, itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.signal import spectrogram as spcg   # 只用于信号质量函数

from PyEMD import EMD                          # pip install EMD-signal
import matplotlib.pyplot as plt

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold

# ---------- 全局参数 ----------
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

FS          = 256          # 采样率 (Hz)
WIN_S       = 2.0          # 滑窗长度  (s)
STEP_S      = 0.5          # 滑窗步长  (s)
WIN, STEP   = int(WIN_S*FS), int(STEP_S*FS)
DROP_FIXED  = {0, 9, 32, 63}
N_CLASS     = 3            # ← 五分类
BATCH       = 64
EPOCH       = 120          # 论文用 400，这里先 120 方便跑通
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REDUCE_DATA = False        # True 时只取少量 trial 快速调试

ROOT  = Path(__file__).resolve().parents[1]
DATA  = ROOT / 'data/Short_words'
FILES = sorted(DATA.glob('*.mat'))

assert FILES, f'✘ 未找到 .mat 文件, 请确认路径: {DATA}'

# ---------- I/O 滤波器 ----------
bp_b, bp_a = butter(4, [4, 40], fs=FS, btype='band')
nt_b, nt_a = butter(2, [48, 52], fs=FS, btype='bandstop')

# ---------- 通道信息 ----------
first = loadmat(FILES[0], simplify_cells=True)
key0  = next(k for k in first if k.endswith('last_beep'))
n_tot = first[key0][0][0].shape[0]
keep_idx = [i for i in range(n_tot) if i not in DROP_FIXED]
N_CH = len(keep_idx)

# ---------- 预处理 ----------
def preprocess(sig):
    """bandstop → bandpass → z-score,  sig:(C,T)"""
    sig = sig[keep_idx]
    sig = filtfilt(nt_b, nt_a, sig, axis=1)
    sig = filtfilt(bp_b, bp_a, sig, axis=1)
    sig -= sig.mean(1, keepdims=True)
    sig /= sig.std(1, keepdims=True) + 1e-6
    return sig.astype(np.float32)

# ---------- 滑窗 ----------
def slide(sig, label, trial_id, subj):
    wins=[]
    for st in range(0, sig.shape[1]-WIN+1, STEP):
        wins.append(dict(win=sig[:, st:st+WIN], label=label,
                         trial=trial_id, subj=subj, st=st))
    return wins

# ---------- 信号质量指标 ----------
def extract_quality(win):
    # 低频漂移 (0.1-1 Hz)
    b,a = butter(2, [0.1,1], fs=FS, btype='band')
    drift = filtfilt(b,a,win,axis=1)
    drift_std = float(drift.std())

    # 伪迹比例 (>100 µV)
    art_ratio = float(((win>100)|(win<-100)).sum()/win.size)

    # 各频段功率
    f,Pxx = welch(win, fs=FS, nperseg=256, axis=1)
    def bp(lo,hi): idx=(f>=lo)&(f<=hi); return float(Pxx[:,idx].mean())
    pows = dict(delta=bp(.5,4), theta=bp(4,8), alpha=bp(8,13),
                beta=bp(13,30), gamma=bp(30,50))
    return dict(drift=drift_std, artifact=art_ratio, **pows)

# ---------- Hilbert-Spectrum 图像 ----------
def hs_image(win, imf_k=3):
    """
    EEG window (C,T) → torch tensor (3,H,W)   H≈656, W≈875
    步骤: EMD→取前k个IMF→Hilbert→能量→伪彩色
    """
    emd = EMD(); ims=[]
    for ch in win:
        imfs = emd(ch)[:imf_k]        # (k,T)
        energy = np.abs(hilbert(imfs, axis=1))**2
        ims.append(energy)
    hs = np.mean(np.stack(ims,0),0)   # (k,T)

    hs = (hs-hs.min())/(hs.max()-hs.min()+1e-9)
    fig = plt.figure(figsize=(8.75,6.56), dpi=100)
    plt.axis('off'); plt.imshow(hs,aspect='auto',cmap='viridis',origin='lower')
    buf = io.BytesIO(); plt.savefig(buf,format='png',
                                    bbox_inches='tight',pad_inches=0); plt.close(fig)
    buf.seek(0)
    img = plt.imread(buf)[:,:,:3]     # H×W×3
    img = np.transpose(img,(2,0,1))   # →3×H×W
    img = (img - img.mean())/img.std()
    return torch.tensor(img, dtype=torch.float32)

# ---------- Dataset ----------
class EEGDataset(Dataset):
    def __init__(self, idxs, cache=dict()):
        self.idxs=idxs; self.cache=cache
    def __len__(self): return len(self.idxs)
    def __getitem__(self,i):
        rec = all_windows[self.idxs[i]]
        key = rec['idx_cache']        # 唯一索引
        if key not in self.cache:
            self.cache[key]=hs_image(rec['win'])
        x = self.cache[key]
        y = rec['label']
        return x, y, self.idxs[i]

# ---------- 15-层 CNN ----------
class PaperCNN(nn.Module):
    def __init__(self, n_cls=N_CLASS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),  nn.BatchNorm2d(8),  nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8,16,3,padding=1),    nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),   nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(32, n_cls)
    def forward(self,x):
        x = self.features(x).flatten(1)
        return self.fc(x)

# ---------- 读取全部 trial ----------
all_windows=[]
for si,matf in enumerate(FILES):
    m = loadmat(matf, simplify_cells=True)
    key = next(k for k in m if k.endswith('last_beep'))
    trials = [preprocess(tr) for cls in m[key] for tr in cls]
    labels = [cls for cls, t in enumerate(m[key]) for _ in t]
    if REDUCE_DATA:
        idx_keep = np.random.choice(len(trials),10,False)
    else:
        idx_keep = range(len(trials))
    for tid in idx_keep:
        # 记下全局唯一编号，便于缓存
        slices = slide(trials[tid], labels[tid], tid, si)
        for rec in slices:
            rec['idx_cache'] = f"S{si}_{tid}_{rec['st']}"
        all_windows.extend(slices)

print(f"✓ Total windows = {len(all_windows)}")

# ---------- K-fold 训练 ----------
groups = [rec['trial'] for rec in all_windows]
labels = [rec['label'] for rec in all_windows]
gkf = GroupKFold(10)

metrics=[]; fold_acc=[]
for fold,(tr,te) in enumerate(gkf.split(np.arange(len(all_windows)), labels, groups)):
    print(f"\nFold {fold}")
    ds_tr,ds_te=EEGDataset(tr),EEGDataset(te)
    dl_tr=DataLoader(ds_tr,batch_size=BATCH,shuffle=True)
    dl_te=DataLoader(ds_te,batch_size=BATCH,shuffle=False)

    net=PaperCNN().to(DEVICE)
    opt=torch.optim.Adam(net.parameters(),1e-3, weight_decay=1e-4)
    cri=nn.CrossEntropyLoss()

    net.train()
    for ep in range(EPOCH):
        for xb,yb,_ in dl_tr:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad(); loss=cri(net(xb),yb); loss.backward(); opt.step()

    # ---- 测试 & 记录质量指标 ----
    net.eval(); correct=total=0
    with torch.no_grad():
        for xb,yb,idxs in dl_te:
            pred=net(xb.to(DEVICE)).argmax(1).cpu()
            correct+=(pred==yb).sum().item(); total+=len(yb)
            for j,idx0 in enumerate(idxs):
                rec=all_windows[idx0]
                qual=extract_quality(rec['win'])
                metrics.append(dict(fold=fold, idx=int(idx0),
                                    correct=int(pred[j]==yb[j]), **qual))
    acc=correct/total; fold_acc.append(acc)
    print(f"  acc={acc:.3f}")

print(f"\nMean 10-fold acc = {np.mean(fold_acc):.3f}")

# ---------- 结果保存 ----------
df=pd.DataFrame(metrics); df.to_csv('df_metrics.csv',index=False)
corr=df.drop(columns=['fold','idx']).corr(method='spearman')
corr.to_csv('corr_metrics.csv')
print("\nSpearman ρ 与分类正确性的相关：")
print(corr['correct'].sort_values(ascending=False))