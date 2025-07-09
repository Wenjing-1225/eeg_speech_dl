#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_hilbert_spectro_quality.py
------------------------------
一次生成 Hilbert-Spectrum 缓存 + 15-layer CNN 训练 / 10-fold CV
—————————————————————————————————————————————————————————
• 数据     : data/Short_words/*.mat   (Agarwal & Kumar 同格式)
• 输出     : df_metrics.csv, corr_metrics.csv, 10-fold 准确率
• 运行步骤 : 第一次 ⇒ 自动生成 cached_hs/*.pt   以后 ⇒ 直接训练
"""

import io, random, warnings, time
from pathlib import Path

import numpy as np, pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert, welch
from PyEMD import EMD                          # pip install EMD-signal
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold

# ---------------- 全局参数 ----------------
SEED = 0
FS   = 256
WIN_S, STEP_S = 2.0, 0.5
WIN, STEP = int(WIN_S*FS), int(STEP_S*FS)
DROP_FIXED = {0, 9, 32, 63}
N_CLASS    = 3        # ← 改成 5 即可
BATCH      = 64
EPOCH      = 120
NUM_WORKERS_TRAIN = 4   # Linux/GPU 建议 >=4，Mac/Win 请改 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 路径设置：始终相对脚本所在目录的上一级 -------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]     # ← 项目根
DATA         = PROJECT_ROOT / "data" / "Short_words"   # data/Short_words/
CACHE        = PROJECT_ROOT / "cached_hs"
CACHE.mkdir(exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
warnings.filterwarnings("ignore", category=UserWarning)  # matplotlib/Torch 杂项

print(">>> Torch device:", DEVICE)

# ---------------- 预处理工具 ----------------
bp_b, bp_a = butter(4, [4, 40], fs=FS, btype="band")
nt_b, nt_a = butter(2, [48, 52], fs=FS, btype="bandstop")

first = loadmat(next(DATA.glob("*.mat")), simplify_cells=True)
key0  = next(k for k in first if k.endswith("last_beep"))
n_tot = first[key0][0][0].shape[0]
keep_i = [i for i in range(n_tot) if i not in DROP_FIXED]

def preprocess(sig: np.ndarray) -> np.ndarray:
    sig = sig[keep_i]
    sig = filtfilt(nt_b, nt_a, sig, axis=1)
    sig = filtfilt(bp_b, bp_a, sig, axis=1)
    sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True)+1e-6)
    return sig.astype(np.float32)

def slide(sig, label, tid, subj):
    for st in range(0, sig.shape[1]-WIN+1, STEP):
        yield dict(win=sig[:,st:st+WIN], label=label,
                   trial=tid, subj=subj, st=st)

# ---------------- 生成 Hilbert-Spectrum 图像 ----------------
def hs_image(win, imf_k=3) -> torch.Tensor:
    emd, ims = EMD(), []
    for ch in win:
        imfs = emd(ch)[:imf_k]
        energy = np.abs(hilbert(imfs, axis=1))**2
        ims.append(energy)
    hs = np.mean(np.stack(ims,0),0)
    hs = (hs - hs.min()) / (hs.max() - hs.min() + 1e-9)

    fig = plt.figure(figsize=(8.75,6.56), dpi=100)
    plt.axis("off"); plt.imshow(hs, aspect="auto", cmap="viridis", origin="lower")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0); plt.close(fig)
    buf.seek(0)
    img = plt.imread(buf)[:,:,:3]          # H×W×3
    img = (img - img.mean()) / img.std()
    return torch.tensor(img.transpose(2,0,1), dtype=torch.float32)

def build_cache():
    """若有缺失 .pt 文件则自动补齐"""
    need = False
    for si, mfile in enumerate(sorted(DATA.glob("*.mat"))):
        m   = loadmat(mfile, simplify_cells=True)
        key = next(k for k in m if k.endswith("last_beep"))
        trials = [preprocess(tr) for cls in m[key] for tr in cls]
        for tid, tri in enumerate(trials):
            for st in range(0, tri.shape[1]-WIN+1, STEP):
                pt = CACHE / f"S{si}_{tid}_{st}.pt"
                if not pt.exists():
                    need = True; break
        if need: break
    if not need:
        print("✓ HS cache 已存在，跳过生成"); return

    print("🔄 正在生成 Hilbert-Spectrum 缓存（一次性） ...")
    t0=time.time()
    for si, mfile in enumerate(sorted(DATA.glob("*.mat"))):
        m   = loadmat(mfile, simplify_cells=True)
        key = next(k for k in m if k.endswith("last_beep"))
        trials = [preprocess(tr) for cls in m[key] for tr in cls]
        for tid, tri in enumerate(trials):
            for st in range(0, tri.shape[1]-WIN+1, STEP):
                pt = CACHE / f"S{si}_{tid}_{st}.pt"
                if pt.exists(): continue
                torch.save(hs_image(tri[:,st:st+WIN]), pt)
    print(f"✅ 缓存生成完成，用时 {time.time()-t0:.1f}s\n")

# ---------------- Dataset ----------------
class HSDataset(Dataset):
    def __init__(self, idxs, meta):
        self.idxs, self.meta = idxs, meta
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        rec = self.meta[self.idxs[i]]
        x = torch.load(CACHE / f"S{rec['subj']}_{rec['trial']}_{rec['st']}.pt")
        return x, rec['label']

# ---------------- CNN ----------------
class PaperCNN(nn.Module):
    def __init__(self, n_cls=N_CLASS):
        super().__init__()
        self.fea = nn.Sequential(
            nn.Conv2d(3,8,3,padding=1),  nn.BatchNorm2d(8),  nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(32, n_cls)
    def forward(self,x): return self.fc(self.fea(x).flatten(1))

# ---------------- 训练 + 10-fold CV ----------------
def main():
    build_cache()

    # ---- 构造 meta 信息 ----
    meta=[]
    for si, mfile in enumerate(sorted(DATA.glob("*.mat"))):
        m   = loadmat(mfile, simplify_cells=True)
        key = next(k for k in m if k.endswith("last_beep"))
        trials=[preprocess(tr) for cls in m[key] for tr in cls]
        labels=[cls for cls,t in enumerate(m[key]) for _ in t]
        for tid,(tri,y) in enumerate(zip(trials,labels)):
            for rec in slide(tri, y, tid, si):
                meta.append(rec)

    print("Total windows:", len(meta))
    groups=[r['trial'] for r in meta]
    labels=[r['label'] for r in meta]
    gkf=GroupKFold(10)

    fold_acc=[]
    for fold,(tr,te) in enumerate(gkf.split(np.arange(len(meta)), labels, groups)):
        print(f"\n=== Fold {fold} ===")
        dl_tr = DataLoader(HSDataset(tr, meta), BATCH, True,
                           num_workers=NUM_WORKERS_TRAIN, pin_memory=False)
        dl_te = DataLoader(HSDataset(te, meta), BATCH, False,
                           num_workers=NUM_WORKERS_TRAIN, pin_memory=False)

        net = PaperCNN().to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        cri = nn.CrossEntropyLoss()

        net.train()
        for _ in range(EPOCH):
            for xb,yb in dl_tr:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); cri(net(xb), yb).backward(); opt.step()

        net.eval(); correct=total=0
        with torch.no_grad():
            for xb,yb in dl_te:
                pred = net(xb.to(DEVICE)).argmax(1).cpu()
                correct += (pred==yb).sum().item(); total += len(yb)
        acc=correct/total; fold_acc.append(acc)
        print(f"Fold acc = {acc:.3f}")

    print("\nMean 10-fold acc =", np.mean(fold_acc))

if __name__ == "__main__":
    main()