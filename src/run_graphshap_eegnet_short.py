#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_graphshap_eegnet_short.py
=============================
Short-words (3-class: in / out / up)

Pipeline
--------
1.  Surrogate 1-D CNN  (60 ch)            – 训练并保存权重
2.  Kernel-SHAP on surrogate              – 全局通道重要度 imp[i]
3.  Graph-GCN using A_ij = imp_i * imp_j  – 真正 10-fold CV 准确率

结果写入 results/graphshap_short.json
"""
import argparse, json, random, time, warnings
from pathlib import Path

import numpy as np
import shap
import torch, torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from mne.decoding import CSP
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import GroupKFold

# ============ 全局超参 ============
SEED = 0
FS   = 256
WIN_S, STEP_S = 2.0, .5
WIN,  STEP    = int(WIN_S*FS), int(STEP_S*FS)
BATCH = 128
EPOCH_SUR = 60         # surrogate 训练 epoch
EPOCH_GCN = 80         # GCN 训练 epoch
SHAP_SAMP = 256        # Kernel-SHAP 采样窗口数
DROP_FIXED = {0,9,32,63}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data/Short_words"
FILES = sorted(DATA.glob("*.mat"))

# ---------- 通道与滤波 ----------
first = loadmat(FILES[0], simplify_cells=True)
k0    = next(k for k in first if k.endswith("last_beep"))
n_tot = first[k0][0][0].shape[0]

keep_idx   = [i for i in range(n_tot) if i not in DROP_FIXED]
N_CH       = len(keep_idx)
bp_b, bp_a = butter(4,[4,40], fs=FS, btype='band')
nt_b, nt_a = iirnotch(60,30,fs=FS)

def preprocess(sig):
    sig = sig[keep_idx]
    sig = filtfilt(nt_b,nt_a,sig,axis=1)
    sig = filtfilt(bp_b,bp_a,sig,axis=1)
    sig -= sig.mean(1,keepdims=True)
    sig /= sig.std (1,keepdims=True)+1e-6
    return sig.astype(np.float32)

def slide(sig,tid):
    wins,gids=[],[]
    for st in range(0,sig.shape[1]-WIN+1,STEP):
        wins.append(sig[:,st:st+WIN]); gids.append(tid)
    return wins,gids

# ---------- surrogate 1-D CNN ----------
class SurrogateNet(nn.Module):
    def __init__(self,n_ch=N_CH,n_cls=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,(1,64),padding=(0,32)), nn.ReLU(),
            nn.Conv2d(16,32,(n_ch,1),groups=16),   nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
            nn.Linear(32,n_cls)
        )
    def forward(self,x): return self.net(x)

# ---------- GCN 真正分类器 ----------
class GCNClassifier(nn.Module):
    def __init__(self,in_feat=WIN,hidden=64,n_cls=3):
        super().__init__()
        self.gc1 = GCNConv(in_feat, hidden)
        self.gc2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, n_cls)
    def forward(self,data):
        x,edge_index = data.x, data.edge_index
        x = torch.relu(self.gc1(x,edge_index))
        x = torch.relu(self.gc2(x,edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)

# ---------- Utils ----------
def build_edge_index(imp, thr=0.0):
    # 完全图但可按阈值稀疏
    src, dst = [], []
    for i in range(N_CH):
        for j in range(N_CH):
            if imp[i]*imp[j] > thr:
                src.append(i); dst.append(j)
    return torch.tensor([src,dst], dtype=torch.long)

def windows_to_graph_tensors(windows, imp, edge_index):
    # windows: (n_win, C, T)
    xs, ei, bs = [], [], []
    for bid,(w) in enumerate(windows):
        xs.append(torch.tensor(w.T))  # (T,C) → 节点=C, feat=T
        bs.append(torch.full((N_CH,), bid, dtype=torch.long))
    x   = torch.cat(xs,0)            # (N_CH*B,  T)
    ei  = edge_index.repeat(1,len(windows)) + \
          torch.arange(len(windows)).repeat_interleave(edge_index.shape[1])*N_CH
    batch = torch.cat(bs,0)
    return x.float().to(DEVICE), ei.to(DEVICE), batch.to(DEVICE)

# ---------- 主流程 ----------
def main():
    gkf = GroupKFold(10)
    results = {}

    for subj_i,matf in enumerate(FILES,1):
        m   = loadmat(matf,simplify_cells=True)
        key = next(k for k in m if k.endswith("last_beep"))
        trials=[preprocess(tr) for cls in m[key] for tr in cls]
        labels=[cls for cls,tset in enumerate(m[key]) for _ in tset]
        trials,labels = np.asarray(trials),np.asarray(labels,dtype=int)

        # -- 划窗 --
        Xw,Yn,Gn=[],[],[]
        for tid,(sig,lab) in enumerate(zip(trials,labels)):
            wins,gids=slide(sig,tid)
            Xw.extend(wins); Yn.extend([lab]*len(wins)); Gn.extend(gids)
        Xw = np.stack(Xw)
        X_t = torch.tensor(Xw[:,None,:,:],device=DEVICE)
        Y_t = torch.tensor(Yn,device=DEVICE)

        # -- ➊ 训练 surrogate --
        net_s = SurrogateNet().to(DEVICE)
        opt = torch.optim.Adam(net_s.parameters(),1e-3)
        cri = nn.CrossEntropyLoss()
        net_s.train()
        for ep in range(EPOCH_SUR):
            idx = torch.randperm(len(X_t),device=DEVICE)
            for beg in range(0,len(idx),BATCH):
                sl = idx[beg:beg+BATCH]
                opt.zero_grad()
                loss = cri(net_s(X_t[sl]), Y_t[sl]); loss.backward(); opt.step()

        # -- ➋ Kernel-SHAP (全局重要度) --
        # 采样部分窗口避免太慢
        samp_idx = np.random.choice(len(Xw), size=min(SHAP_SAMP,len(Xw)), replace=False)
        background = Xw[samp_idx[:32]]             # shap 建议小背景
        explainer  = shap.KernelExplainer(
            lambda x: torch.softmax(net_s(torch.tensor(x,device=DEVICE)),1).cpu().numpy(),
            background)
        shap_vals  = explainer.shap_values(Xw[samp_idx], nsamples=128)
        # shap_vals → list(n_cls)[n_samples,C,T] ，求均值再取绝对值
        imp = np.mean(np.abs(shap_vals), axis=(0,2))  # (C,)
        imp /= imp.max() + 1e-6

        # -- ➌ 用 imp 构图训练 GCN --
        edge_index = build_edge_index(imp, thr=0.05)
        scores=[]
        for tr,te in gkf.split(Xw, Yn, groups=Gn):
            # 组装 PyG Data
            from torch_geometric.data import Data, DataLoader
            def make_loader(indices, shuffle):
                X_sel = Xw[indices]
                Y_sel = Yn[indices]
                graphs=[]
                for i,(win,lbl) in enumerate(zip(X_sel,Y_sel)):
                    x,e,b = windows_to_graph_tensors([win], imp, edge_index)
                    graphs.append(Data(x=x, edge_index=e, y=torch.tensor([lbl])))
                return DataLoader(graphs,batch_size=BATCH,shuffle=shuffle)
            dl_train = make_loader(tr, True)
            dl_test  = make_loader(te, False)

            gcn = GCNClassifier(WIN).to(DEVICE)
            opt = torch.optim.Adam(gcn.parameters(),1e-3)
            cri = nn.CrossEntropyLoss()

            for ep in range(EPOCH_GCN):
                gcn.train()
                for batch in dl_train:
                    batch = batch.to(DEVICE)
                    opt.zero_grad()
                    out = gcn(batch)
                    loss = cri(out, batch.y)
                    loss.backward(); opt.step()

            # 验证
            gcn.eval(); correct,n_tot = 0,0
            for batch in dl_test:
                batch = batch.to(DEVICE)
                pred = gcn(batch).argmax(1)
                correct += (pred==batch.y).sum().item()
                n_tot   += batch.y.numel()
            scores.append(correct/n_tot)

        results[f"sub{subj_i:02d}"] = float(np.mean(scores))
        print(f"Subject {subj_i}: acc = {results[f'sub{subj_i:02d}']:.3f}")

    out = ROOT/"results/graphshap_short.json"
    json.dump(results, open(out,"w"), indent=2)
    print("✔ 结果写入", out)

# ---------- CLI ----------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()