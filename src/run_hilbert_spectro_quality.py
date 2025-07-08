#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_hilbert_spectro_quality.py
------------------------------
Hilbert Spectrum  + 15-layer CNN  + 信号质量分析
"""

# ===== 依赖 =====
from pathlib import Path
import random, io, warnings
import numpy as np, pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert, welch
from PyEMD import EMD                        # pip install EMD-signal
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold

# ===== 全局参数 =====
SEED   = 0
FS     = 256           # 采样率
WIN_S  = 2.0           # 窗长 (s)
STEP_S = 0.5           # 步长 (s)
WIN, STEP = int(WIN_S * FS), int(STEP_S * FS)
DROP_FIXED   = {0, 9, 32, 63}
N_CLASS      = 3        # ← 3 类或 5 类
BATCH        = 64
EPOCH        = 120
NUM_WORKERS  = 0        # macOS 先设 0，跑通再调大
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REDUCE_DATA  = False    # True→只抽少量 trial 便于快速测试

# ============ 主逻辑 ============ #
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    ROOT  = Path(__file__).resolve().parents[1]
    DATA  = ROOT / "data/Short_words"
    FILES = sorted(DATA.glob("*.mat"))
    assert FILES, f"✘ 未找到 .mat 文件, 路径: {DATA}"

    # I/O 滤波器
    bp_b, bp_a = butter(4, [4, 40], fs=FS, btype="band")
    nt_b, nt_a = butter(2, [48, 52], fs=FS, btype="bandstop")

    # 通道索引
    first  = loadmat(FILES[0], simplify_cells=True)
    key0   = next(k for k in first if k.endswith("last_beep"))
    n_tot  = first[key0][0][0].shape[0]
    keep_i = [i for i in range(n_tot) if i not in DROP_FIXED]

    # ---------- 工具函数 ----------
    def preprocess(sig):
        sig = sig[keep_i]
        sig = filtfilt(nt_b, nt_a, sig, axis=1)
        sig = filtfilt(bp_b, bp_a, sig, axis=1)
        sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True)+1e-6)
        return sig.astype(np.float32)

    def slide(sig, label, trial_id, subj):
        out = []
        for st in range(0, sig.shape[1]-WIN+1, STEP):
            out.append(dict(win=sig[:, st:st+WIN], label=label,
                            trial=trial_id, subj=subj, st=st))
        return out

    def extract_quality(win):
        b,a = butter(2, [0.1,1], fs=FS, btype="band")
        drift = filtfilt(b,a,win,axis=1).std()
        art   = ((win>100)|(win<-100)).mean()
        f,Pxx = welch(win,fs=FS,nperseg=256,axis=1)
        def bp(lo,hi): idx=(f>=lo)&(f<=hi); return Pxx[:,idx].mean()
        return dict(drift=float(drift), artifact=float(art),
                    delta=float(bp(.5,4)), theta=float(bp(4,8)),
                    alpha=float(bp(8,13)), beta=float(bp(13,30)),
                    gamma=float(bp(30,50)))

    def hs_image(win, imf_k=3):
        emd = EMD(); ims=[]
        for ch in win:
            imfs  = emd(ch)[:imf_k]
            energy= np.abs(hilbert(imfs,axis=1))**2
            ims.append(energy)
        hs = np.mean(np.stack(ims,0),0)
        hs = (hs-hs.min())/(hs.max()-hs.min()+1e-9)
        fig = plt.figure(figsize=(8.75,6.56),dpi=100)
        plt.axis("off"); plt.imshow(hs,aspect="auto",cmap="viridis",origin="lower")
        buf = io.BytesIO()
        plt.savefig(buf,format="png",bbox_inches="tight",pad_inches=0); plt.close(fig)
        buf.seek(0); img = plt.imread(buf)[:,:,:3]
        img = (img - img.mean())/img.std()
        return torch.tensor(img.transpose(2,0,1), dtype=torch.float32)

    # ---------- Dataset ----------
    class EEGDataset(Dataset):
        def __init__(self, idxs, cache=dict()):
            self.idxs, self.cache = idxs, cache
        def __len__(self): return len(self.idxs)
        def __getitem__(self, i):
            rec = all_windows[self.idxs[i]]
            key = rec["idx_cache"]
            if key not in self.cache:
                self.cache[key] = hs_image(rec["win"])
            return self.cache[key], rec["label"], self.idxs[i]

    # ---------- CNN ----------
    class PaperCNN(nn.Module):
        def __init__(self, n_cls=N_CLASS):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3,8,3,padding=1),  nn.BatchNorm2d(8),  nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1))
            )
            self.fc = nn.Linear(32,n_cls)
        def forward(self,x): return self.fc(self.features(x).flatten(1))

    # ---------- 读取全部 trial ----------
    print("🔄  Loading .mat files ...")
    all_windows=[]
    for si,matf in tqdm(enumerate(FILES), total=len(FILES)):
        m   = loadmat(matf, simplify_cells=True)
        key = next(k for k in m if k.endswith("last_beep"))
        trials = [preprocess(tr) for cls in m[key] for tr in cls]
        labels = [cls for cls,tlist in enumerate(m[key]) for _ in tlist]
        if REDUCE_DATA: idx_keep = np.random.choice(len(trials),10,False)
        else:            idx_keep = range(len(trials))
        for tid in idx_keep:
            for rec in slide(trials[tid], labels[tid], tid, si):
                rec["idx_cache"] = f"S{si}_{tid}_{rec['st']}"
                all_windows.append(rec)
    print(f"✓ Total trials : {len(all_windows)}\n")

    # ---------- K-fold ----------
    groups=[r["trial"] for r in all_windows]
    labels=[r["label"] for r in all_windows]
    gkf   = GroupKFold(10)

    metrics, fold_acc = [], []
    for fold,(tr,te) in enumerate(gkf.split(np.arange(len(all_windows)),labels,groups)):
        print(f"=== Fold {fold} ===")
        ds_tr,ds_te = EEGDataset(tr),EEGDataset(te)
        dl_tr=DataLoader(ds_tr,BATCH,shuffle=True,
                         num_workers=NUM_WORKERS,pin_memory=False)
        dl_te=DataLoader(ds_te,BATCH,shuffle=False,
                         num_workers=NUM_WORKERS,pin_memory=False)

        net = PaperCNN().to(DEVICE)
        opt = torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
        cri = nn.CrossEntropyLoss()

        net.train()
        for _ in range(EPOCH):
            for xb,yb,_ in dl_tr:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); cri(net(xb),yb).backward(); opt.step()

        # ---- 测试 & 记录 ----
        net.eval(); correct=total=0
        with torch.no_grad():
            for xb,yb,idxs in dl_te:
                pred = net(xb.to(DEVICE)).argmax(1).cpu()
                correct += (pred==yb).sum().item(); total += len(yb)
                for j,idx0 in enumerate(idxs):
                    q = extract_quality(all_windows[idx0]["win"])
                    metrics.append(dict(fold=fold, idx=int(idx0),
                                        correct=int(pred[j]==yb[j]), **q))
        acc = correct/total; fold_acc.append(acc)
        print(f"    acc = {acc:.3f}")

    print(f"\nMean 10-fold acc = {np.mean(fold_acc):.3f}")

    # ---------- 保存 ----------
    df = pd.DataFrame(metrics); df.to_csv("df_metrics.csv",index=False)
    corr = df.drop(columns=["fold","idx"]).corr("spearman")
    corr.to_csv("corr_metrics.csv")
    print("\nSpearman ρ wrt correctness:")
    print(corr["correct"].sort_values(ascending=False))


# ========= macOS / Windows 必须 main-guard =========
if __name__ == "__main__":
    main()