#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_graphshap_eegnet_short.py
-----------------------------
Short-words 三分类（in / out / up）

Pipeline
1) Surrogate 1-D CNN 训练
2) Kernel-SHAP 求全局通道重要度 imp[i]
3) 以 imp_i·imp_j 生成图，GCN 十折交叉验证
"""
import argparse, json, random, time, warnings
from pathlib import Path

import numpy as np
import shap
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from mne.decoding import CSP            # 只留下，后面可删
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import GroupKFold

# ---------- 全局超参 ----------
SEED = 0
FS = 256
WIN_S, STEP_S = 2.0, .5
WIN, STEP = int(WIN_S * FS), int(STEP_S * FS)
BATCH = 128
EPOCH_SUR = 60
EPOCH_GCN = 80
SHAP_SAMP = 256
DROP_FIXED = {0, 9, 32, 63}
N_CLASS = 3

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/Short_words"
FILES = sorted(DATA.glob("*.mat"))

# ---------- 通道、滤波 ----------
first = loadmat(FILES[0], simplify_cells=True)
k0 = next(k for k in first if k.endswith("last_beep"))
n_tot = first[k0][0][0].shape[0]

keep_idx = [i for i in range(n_tot) if i not in DROP_FIXED]
N_CH = len(keep_idx)

bp_b, bp_a = butter(4, [4, 40], fs=FS, btype='band')
nt_b, nt_a = iirnotch(60, 30, fs=FS)

def preprocess(sig):
    sig = sig[keep_idx]
    sig = filtfilt(nt_b, nt_a, sig, axis=1)
    sig = filtfilt(bp_b, bp_a, sig, axis=1)
    sig -= sig.mean(1, keepdims=True)
    sig /= sig.std(1, keepdims=True) + 1e-6
    return sig.astype(np.float32)

def slide(sig, tid):
    wins, gids = [], []
    for st in range(0, sig.shape[1] - WIN + 1, STEP):
        wins.append(sig[:, st:st + WIN])
        gids.append(tid)
    return wins, gids

# ---------- 1-D surrogate CNN ----------
class SurrogateNet(nn.Module):
    def __init__(self, n_ch=N_CH, n_cls=N_CLASS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32)), nn.ReLU(),
            nn.Conv2d(16, 32, (n_ch, 1), groups=16),    nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(32, n_cls)
        )
    def forward(self, x):
        return self.net(x)

# ---------- GCN ----------
class GCNClassifier(nn.Module):
    def __init__(self, in_feat=WIN, hidden=64, n_cls=N_CLASS):
        super().__init__()
        self.gc1 = GCNConv(in_feat, hidden)
        self.gc2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, n_cls)
    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = torch.relu(self.gc1(x, ei))
        x = torch.relu(self.gc2(x, ei))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)

# ---------- Graph utils ----------
def build_edge_index(imp, thr=0.0):
    src, dst = [], []
    for i in range(N_CH):
        for j in range(N_CH):
            if imp[i] * imp[j] > thr:
                src.append(i); dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)

def windows_to_graph_tensors(wins, imp, edge_index):
    # wins: list 1 个 window, shape (C,T)
    xs, bs = [], []
    for bid, w in enumerate(wins):
        xs.append(torch.tensor(w.T))  # 节点=C，特征=T
        bs.append(torch.full((N_CH,), bid, dtype=torch.long))
    x = torch.cat(xs, 0)
    ei = edge_index.repeat(1, len(wins)) + \
         torch.arange(len(wins)).repeat_interleave(edge_index.shape[1]) * N_CH
    batch = torch.cat(bs, 0)
    return x.float().to(DEVICE), ei.to(DEVICE), batch.to(DEVICE)

# ---------- 主流程 ----------
def main():
    gkf = GroupKFold(10)
    results = {}

    for subj_i, matf in enumerate(FILES, 1):
        print(f"\n=== Subject {subj_i}/{len(FILES)} ({matf.name}) ===")
        m = loadmat(matf, simplify_cells=True)
        key = next(k for k in m if k.endswith("last_beep"))
        trials = [preprocess(tr) for cls in m[key] for tr in cls]
        labels = [cls for cls, tset in enumerate(m[key]) for _ in tset]
        trials, labels = np.asarray(trials), np.asarray(labels, dtype=int)

        # -- 划窗 --
        Xw, Yn, Gn = [], [], []
        for tid, (sig, lab) in enumerate(zip(trials, labels)):
            wins, gids = slide(sig, tid)
            Xw.extend(wins); Yn.extend([lab] * len(wins)); Gn.extend(gids)
        Xw = np.stack(Xw)                            # (n_win, C, T)
        X_t = torch.tensor(Xw[:, None, :, :], device=DEVICE)  # (n,1,C,T)
        Y_t = torch.tensor(Yn, device=DEVICE)

        # -- ① 训练 surrogate CNN --
        net_s = SurrogateNet().to(DEVICE)
        opt_s = torch.optim.Adam(net_s.parameters(), 1e-3)
        cri   = nn.CrossEntropyLoss()
        net_s.train()
        for _ in range(EPOCH_SUR):
            idx = torch.randperm(len(X_t), device=DEVICE)
            for beg in range(0, len(idx), BATCH):
                sl = idx[beg:beg + BATCH]
                opt_s.zero_grad()
                loss = cri(net_s(X_t[sl]), Y_t[sl])
                loss.backward(); opt_s.step()

        # -- ② Kernel-SHAP 全局通道重要度 --
        samp_idx = np.random.choice(len(Xw),
                                    size=min(SHAP_SAMP, len(Xw)),
                                    replace=False)
        background = Xw[samp_idx[:32]]              # (b,C,T)

        # >>> 关键：包装 predict_fn，自动补 channel 维 <<<
        def predict_fn(arr):
            t = torch.tensor(arr, device=DEVICE).unsqueeze(1)  # (n,1,C,T)
            with torch.no_grad():
                out = net_s(t)
            return torch.softmax(out, 1).cpu().numpy()

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_vals = explainer.shap_values(Xw[samp_idx], nsamples=128)
        # shap_vals: list(n_cls)[n, C, T] → 取平均绝对值
        imp = np.mean(np.abs(shap_vals), axis=(0, 2))   # (C,)
        imp /= imp.max() + 1e-6

        # -- ③ Graph-GCN 十折 CV --
        edge_index = build_edge_index(imp, thr=0.05)

        from torch_geometric.data import Data, DataLoader
        def make_loader(indices, shuffle):
            graphs = []
            for w_idx in indices:
                win = Xw[w_idx]
                lab = Yn[w_idx]
                x, ei, batch = windows_to_graph_tensors([win], imp, edge_index)
                graphs.append(Data(x=x, edge_index=ei, y=torch.tensor([lab])))
            return DataLoader(graphs, batch_size=BATCH, shuffle=shuffle)

        accs = []
        for tr, te in gkf.split(Xw, Yn, groups=Gn):
            dl_tr = make_loader(tr, True)
            dl_te = make_loader(te, False)

            gcn = GCNClassifier(WIN).to(DEVICE)
            opt_g = torch.optim.Adam(gcn.parameters(), 1e-3)
            gcn.train()
            for _ in range(EPOCH_GCN):
                for batch in dl_tr:
                    batch = batch.to(DEVICE)
                    opt_g.zero_grad()
                    loss = cri(gcn(batch), batch.y)
                    loss.backward(); opt_g.step()

            # 验证
            gcn.eval(); correct, total = 0, 0
            for batch in dl_te:
                batch = batch.to(DEVICE)
                pred = gcn(batch).argmax(1)
                correct += (pred == batch.y).sum().item()
                total   += batch.y.numel()
            accs.append(correct / total)

        results[f"sub{subj_i:02d}"] = float(np.mean(accs))
        print(f"Subject {subj_i}: acc = {results[f'sub{subj_i:02d}']:.3f}")

    out = ROOT / "results/graphshap_short.json"
    json.dump(results, open(out, "w"), indent=2)
    print("\n✔ 结果写入", out)

# ---------- CLI ----------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()