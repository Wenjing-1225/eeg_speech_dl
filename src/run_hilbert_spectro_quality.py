# -*- coding: utf-8 -*-
"""
EEG Pre‑processing → Hilbert‑Huang Transform → CNN classification
===============================================================
Author : ChatGPT (OpenAI o3)
Updated: 2025‑07‑09

Pipeline
--------
1. **滤波**：1‑40 Hz 双向 Butterworth + 50 Hz notch
2. **CAR**：通道均值参考
3. **Hilbert 包络归一化**
4. **EMD**：能量排名选前 *k* 个 IMF
5. **Hilbert Spectrum** 热力图缓存到 `cached_hs/k{k}/<subject>/<label>/trial_N.png`
6. **PyTorch CNN** 读取光谱图做三分类

用法示例
^^^^^^^^
```bash
# 生成 Hilbert Spectrum（选前 3 个 IMF）
python src/run_hilbert_spectro_quality.py --make-cache --k 3

# 训练 CNN
python src/run_hilbert_spectro_quality.py --train --k 3 --epochs 50
```

依赖
----
```
numpy<2, scipy, matplotlib, mne, PyEMD, pillow, torch, torchvision
```
"""
from __future__ import annotations
import argparse, multiprocessing as mp, os, re, math, json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert
import mne  # noqa: F401  # 备用：高级滤波/去伪迹
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from functools import partial

# ---------------------------------------------------------------------------
# 全局路径 & 常量
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data" / "Short_words"
CACHE_DIR: Path = PROJECT_ROOT / "cached_hs"
CACHE_DIR.mkdir(exist_ok=True)

# >>> 新增：自动从文件名中解析被试 ID（支持 sub_6b 这类带字母后缀）
SUBJECTS: List[str] = sorted(
    {re.match(r"(sub_[A-Za-z0-9]+)", p.name).group(1)
     for p in DATA_DIR.glob("sub_*_256Hz.mat")}
)
EEG_SAMPLING_RATE = 256  # Hz
RAW_EXT = ".mat"

# ---------------------------------------------------------------------------
# 滤波工具
# ---------------------------------------------------------------------------

def _bandpass_sos(l_freq: float, h_freq: float, sfreq: float, order: int = 4):
    return butter(order, [l_freq, h_freq], btype="band", fs=sfreq, output="sos")


def bandpass_filter(x: np.ndarray, l_freq: float = 1.0, h_freq: float = 40.0,
                    sfreq: float = EEG_SAMPLING_RATE, order: int = 4) -> np.ndarray:
    return sosfiltfilt(_bandpass_sos(l_freq, h_freq, sfreq, order), x, axis=-1)


def notch_filter(x: np.ndarray, band: Tuple[float, float] = (49, 51),
                 sfreq: float = EEG_SAMPLING_RATE, order: int = 2) -> np.ndarray:
    sos = butter(order, band, btype="bandstop", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def car_reference(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)

# ---------------------------------------------------------------------------
# Hilbert–Huang 工具
# ---------------------------------------------------------------------------

def emd_decompose(sig: np.ndarray, max_imf: int | None = None) -> List[np.ndarray]:
    # 这里再 import 一次没关系，Python 会从缓存里取，开销极小
    from PyEMD.EMD import EMD as _EMD
    emd = _EMD()                # ← 现在 100% 是类，可实例化
    imfs = emd.emd(sig)
    return imfs[:max_imf] if max_imf else imfs


def energy_rank(imfs: List[np.ndarray]) -> List[int]:
    return np.argsort([np.sum(i ** 2) for i in imfs])[::-1].tolist()


def compute_hilbert_spectrum(imfs: List[np.ndarray], sfreq: float,
                              t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    E_list, F_list = [], []
    for imf in imfs:
        analytic = hilbert(imf)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        freq = np.concatenate((np.diff(phase), [0])) * sfreq / (2 * np.pi)
        E_list.append(amp)
        F_list.append(freq)
    return t, np.vstack(F_list), np.vstack(E_list)


def save_hilbert_spectrum_png(t: np.ndarray, F: np.ndarray, E: np.ndarray,
                              out_png: Path, dpi: int = 150):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    E = np.where(F <= 0, 0, E)
    T = np.tile(t, (F.shape[0], 1))
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(T, F, E, shading = 'auto',norm = LogNorm(vmin=E[E > 0].min(), vmax=E.max()))
    plt.colorbar(label="Amplitude")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()

# ---------------------------------------------------------------------------
# 数据读入
# ---------------------------------------------------------------------------

def _assemble_from_object_array(raw: np.ndarray) -> Dict[str, np.ndarray]:
    """raw: object array shaped (classes, trials) with each element (ch,samp)"""
    signals, labels = [], []
    for cls_idx in range(raw.shape[0]):
        for trial in raw[cls_idx]:
            signals.append(trial.astype(np.float32))
            labels.append(cls_idx)
    return {"signals": np.stack(signals), "labels": np.array(labels, int)}


def load_subject_mat(subject: str, var_pattern: str = r"eeg_data.*256Hz") -> Dict[str, np.ndarray]:
    # >>> 改为 glob 搜索，以免严格文件名不匹配
    cand = list(DATA_DIR.glob(f"{subject}*_256Hz.mat"))
    if not cand:
        raise FileNotFoundError(f"No .mat file for {subject} in {DATA_DIR}")
    mat_path = cand[0]                     # 若同一 subject 有多文件可自行再加逻辑
    mat = loadmat(mat_path)

    # 优先检测明确字段
    if mat.get("signals") is not None and mat.get("labels") is not None:
        return {"signals": mat["signals"].astype(np.float32),
                "labels": mat["labels"].flatten().astype(int)}

    # 否则用正则匹配对象数组 (classes × trials)
    data_key = next((k for k in mat if re.match(var_pattern, k, flags=re.I)), None)
    if data_key is None:
        raise KeyError(f"No EEG variable matching '{var_pattern}' found in {mat_path.name}")
    return _assemble_from_object_array(mat[data_key])

# ---------------------------------------------------------------------------
# 光谱缓存
# ---------------------------------------------------------------------------

def process_trial(subject: str, trial_idx: int, k_imf: int | None = None,
                  l_freq: float = 1.0, h_freq: float = 40.0) -> Path:
    subj = load_subject_mat(subject)
    sig_raw = subj['signals'][trial_idx]      # (ch, samples)
    label = subj['labels'][trial_idx]

    # 预处理
    sig = bandpass_filter(sig_raw, l_freq, h_freq)
    sig = notch_filter(sig)
    sig = car_reference(sig)

    # Hilbert 包络归一化
    analytic = hilbert(sig, axis=1)
    env = np.abs(analytic)
    sig_norm = sig / (env + 1e-12)

    # 多通道取平均，也可改为挑选特定通道
    sig_mean = sig_norm.mean(axis=0)

    imfs = emd_decompose(sig_mean)
    if k_imf is not None:
        imfs = [imfs[i] for i in energy_rank(imfs)[:k_imf]]

    n = sig_mean.size
    t = np.arange(n) / EEG_SAMPLING_RATE
    t, F, E = compute_hilbert_spectrum(imfs, EEG_SAMPLING_RATE, t)

    out_png = CACHE_DIR / f"k{k_imf or 'all'}" / subject / str(label) / f"trial_{trial_idx}.png"
    if not out_png.exists():
        save_hilbert_spectrum_png(t, F, E, out_png)
    return out_png



def build_cache(k_imf: int | None = None, workers: int = 0):
    jobs: list[tuple[str, int]] = []
    for subj in SUBJECTS:
        try:
            subj_data = load_subject_mat(subj)
        except FileNotFoundError as e:     # 理论上不会再触发，但稳妥起见
            print(e); continue
        for tri in range(len(subj_data['signals'])):
            jobs.append((subj, tri))

    if workers <= 1:
        for s, t_idx in jobs:
            process_trial(s, t_idx, k_imf=k_imf)
    else:
        with mp.Pool(processes=workers) as pool:
            # >>> 用 partial 固定 k_imf，避免 lambda
            pool.starmap(partial(process_trial, k_imf=k_imf), jobs)

# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class HilbertSpectrumDataset(Dataset):
    def __init__(self, k_imf: int | None = None, train: bool = True,
                 val_split: float = 0.2, seed: int = 42):
        self.root = CACHE_DIR / f"k{k_imf or 'all'}"
        # transforms
        tf = transforms.Compose([
            transforms.Resize((656, 875)),  # 保持与论文一致
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        self.samples: List[Tuple[Path, int]] = []
        for subj_dir in self.root.iterdir():
            if not subj_dir.is_dir():  # ← 跳过 .DS_Store 等非目录
                continue
            for label_dir in subj_dir.iterdir():
                if not label_dir.is_dir():  # ← 再次过滤
                    continue
                label = int(label_dir.name)
                for img in label_dir.glob("*.png"):
                    self.samples.append((img, label))
        # 固定随机切分
        rng = np.random.default_rng(seed)
        rng.shuffle(self.samples)
        split = int(len(self.samples)*(1-val_split))
        self.samples = self.samples[:split] if train else self.samples[split:]
        self.transform = tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        x = Image.open(p).convert('RGB')
        x = self.transform(x)
        return x, y

# ---------------------------------------------------------------------------
# 简易 CNN
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#   CNN-3  ⇢ 8-16-32  + BN + ReLU + MaxPool
#   输入默认 3×656×875，可用 transforms.Resize((656, 875))
# ---------------------------------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            # ① 3×3 Conv, 8ch
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(8, eps=1e-5),        # epsilon 1e-5≈0.00001
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # ② 3×3 Conv, 16ch
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # ③ 3×3 Conv, 32ch
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32, eps=1e-5),
            nn.ReLU(inplace=True),
        )

        # 原图 656×875 经过两次 2×2 pooling → 164×218
        # 第三层后不再池化，直接全局平均
        self.global_pool = nn.AdaptiveAvgPool2d(1)   # 输出 32×1×1
        self.classifier  = nn.Linear(32, num_classes)

        # weight init ≈ ‘glorot’ (aka Xavier uniform)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)

# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------

def train_model(k_imf: int | None = None, batch_size: int = 32, epochs: int = 50,
                lr: float = 1e-3, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    train_ds = HilbertSpectrumDataset(k_imf=k_imf, train=True)
    val_ds = HilbertSpectrumDataset(k_imf=k_imf, train=False)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ConvNet().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for ep in range(1, epochs+1):
        model.train(); running = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); out = model(x); loss = crit(out, y)
            loss.backward(); opt.step()
            running += loss.item()*x.size(0)
        train_loss = running/len(train_ds)

        # 验证
        model.eval(); correct = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred==y).sum().item()
        acc = correct/len(val_ds)
        best_acc = max(best_acc, acc)
        print(f"[Ep {ep:03d}] loss {train_loss:.4f}  val {acc*100:.2f}%  best {best_acc*100:.2f}%")

    model_out = PROJECT_ROOT / f"cnn_k{k_imf or 'all'}.pt"
    torch.save(model.state_dict(), model_out)
    print(f"Model saved → {model_out}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("EEG‑HHT‑CNN pipeline")
    p.add_argument('--make-cache', action='store_true', help='Generate Hilbert spectra')
    p.add_argument('--train', action='store_true', help='Train CNN')
    p.add_argument('--k', type=int, default=None, help='Top‑k IMFs (energy)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()
    if args.make_cache:
        print("Generating Hilbert spectrum cache …")
        build_cache(k_imf=args.k, workers=max(mp.cpu_count()-1, 1))
    if args.train:
        print("Training CNN …")
        train_model(k_imf=args.k, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)
    if not (args.make_cache or args.train):
        print("Nothing to do – use --make-cache and/or --train. See --help")


if __name__ == '__main__':
    main()
