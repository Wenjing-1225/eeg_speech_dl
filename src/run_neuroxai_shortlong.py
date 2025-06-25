#!/usr/bin/env python
# run_neuroxai_shortlong.py —— Filter-Bank EEGNet + NeuroXAI (动态 60-ch)

import argparse, json, warnings
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import GroupKFold
from tqdm import trange

from eegnet_model import EEGNet
from neuroxai.explanation import BrainExplainer, GlobalBrainExplainer

# ---------------- 基本参数 ----------------
SEED = 0
FS = 256
WIN_S = 3.0; WIN = int(WIN_S * FS)
STEP_S = 0.25; STEP = int(STEP_S * FS)
BANDS  = [(4, 7), (8, 13), (14, 30)]
EPOCHS = 400
BATCH  = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(SEED); torch.manual_seed(SEED)

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data/Short_Long_words"
FILES = sorted(f for f in DATA.glob("*.mat") if "_8s" not in f.name)

# ---------- 动态生成 60-通道名表 ----------
_first = next(iter(FILES))
_raw   = loadmat(_first, simplify_cells=True)
_key   = next(k for k in _raw if k.endswith("last_beep"))
_sig0  = _raw[_key][0][0]              # (C,T)
n_total = _sig0.shape[0]

if "ch_names" in _raw:
    ORIG_NAMES = [str(ch).strip() for ch in _raw["ch_names"]][:n_total]
else:                                  # 若无通道名就占位
    ORIG_NAMES = [f"Ch{i}" for i in range(n_total)]

DROP_FIXED = {0, 9, 32, 63} & set(range(n_total))
keep_idx = [i for i in range(n_total) if i not in DROP_FIXED]
while len(keep_idx) > 60:              # 继续删最前面，直到 60
    keep_idx.pop(0)

CHAN_NAMES = [ORIG_NAMES[i] for i in keep_idx]
assert len(CHAN_NAMES) == 60, f"仍非 60，当前 {len(CHAN_NAMES)}"

DROP_ID = set(range(n_total)) - set(keep_idx)
N_CH, N_BAND, C_ALL = 60, len(BANDS), 60 * len(BANDS)

# ---------------- 预处理 ----------------
nt_b, nt_a = iirnotch(60, 30, fs=FS)
def preprocess_fb(sig):
    sig = np.delete(sig, list(DROP_ID), axis=0)      # → (60,T)
    bank = []
    for lo, hi in BANDS:
        b, a = butter(4, [lo, hi], fs=FS, btype='band')
        tmp  = filtfilt(b, a, sig, axis=1)
        tmp -= tmp.mean(1, keepdims=True)
        tmp /= tmp.std(1, keepdims=True) + 1e-6
        bank.append(tmp)
    return np.concatenate(bank, 0).astype(np.float32)  # (180,T)

def slide(sig):
    out=[]
    for st in range(0, sig.shape[1]-WIN+1, STEP):
        out.append(sig[:, st:st+WIN])
    return np.stack(out)

# ---------------- 数据加载 ----------------
def load_trials():
    trials, labels = [], []
    for f in FILES:
        mat = loadmat(f, simplify_cells=True)
        key = next(k for k in mat if k.endswith("last_beep"))
        for cls, tset in enumerate(mat[key]):
            for tr in tset:
                trials.append(preprocess_fb(tr)); labels.append(cls)
    trials, labels = map(np.asarray, (trials, labels))
    i0, i1 = np.where(labels==0)[0], np.where(labels==1)[0]
    n = min(len(i0), len(i1))
    keep = np.sort(np.hstack([i0[:n], i1[:n]]))
    return trials[keep], labels[keep]

# ---------------- 训练 & 评估 ----------------
def train_eegnet(X, y, C, lr=1e-3, epochs=EPOCHS):
    net = EEGNet(C, WIN).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss = nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=DEVICE)
        for beg in range(0, len(perm), BATCH):
            sl = perm[beg:beg+BATCH]
            xb = X[sl] * (1 + 0.01 * torch.randn_like(X[sl]))
            opt.zero_grad(); l = loss(net(xb), y[sl]); l.backward(); opt.step()
        sched.step()
    return net

def evaluate(net, X, y, g):
    net.eval(); pr=[]
    with torch.no_grad():
        for beg in range(0,len(X),BATCH):
            pr.append(net(X[beg:beg+BATCH]).argmax(1).cpu())
    pr = np.concatenate(pr); vote={}
    for p,i in zip(pr,g): vote.setdefault(i,[]).append(p)
    pred = {i:max(set(v), key=v.count) for i,v in vote.items()}
    truth = {i:int(y[np.where(g==i)[0][0]]) for i in pred}
    return np.mean([pred[t]==truth[t] for t in pred])

# ---------------- NeuroXAI ----------------
def channel_importance(baseline, X_trials, y_trials, n_samples):
    def clf(batch):
        C,T = batch.shape[1], batch.shape[2]
        if T>WIN: st=(T-WIN)//2; batch=batch[:,:,st:st+WIN]
        elif T<WIN:
            pad = np.zeros((batch.shape[0],C,WIN-T), batch.dtype)
            batch = np.concatenate([batch,pad],2)
        tensor = torch.tensor(batch[:,None,:,:], device=DEVICE)
        return torch.softmax(baseline(tensor),1).cpu().numpy()

    brain  = BrainExplainer(kernel_width=25,class_names=['short','long'])
    g_exp  = GlobalBrainExplainer(brain)
    g_exp.explain_instance(X_trials,y_trials,clf,n_samples)

    raw = np.array([g_exp.explain_global_channel_importance().get(i,0.0)
                    for i in range(C_ALL)], dtype=np.float32)
    return raw.reshape(N_BAND, N_CH).mean(0)

# ---------------- 主入口 ----------------
def main(k_top,n_samples):
    print("① 读数据 …"); trials, labels = load_trials()

    Xw,Y,G, gid = [],[],[],0
    for sig,lab in zip(trials,labels):
        w = slide(sig)
        Xw.append(w); Y.extend([lab]*len(w)); G.extend([gid]*len(w)); gid+=1
    Xw = np.concatenate(Xw)
    Y  = np.asarray(Y); G = np.asarray(G)
    Xt = torch.tensor(Xw[:,None,:,:], device=DEVICE)
    Yt = torch.tensor(Y, device=DEVICE)

    ckpt = ROOT/"results/eegnet_fb_60.pt"
    base = EEGNet(C_ALL,WIN).to(DEVICE)
    if ckpt.exists():
        try: base.load_state_dict(torch.load(ckpt,map_location=DEVICE),strict=True)
        except RuntimeError: ckpt.unlink()
    if not ckpt.exists():
        print("⏳ 训练 180-ch 基线…"); base=train_eegnet(Xt,Yt,C_ALL)
        ckpt.parent.mkdir(exist_ok=True); torch.save(base.state_dict(), ckpt)

    # 10-fold Baseline
    gkf=GroupKFold(10); acc_b=[]
    for tr,te in gkf.split(Xt,Yt,groups=G):
        acc_b.append(evaluate(train_eegnet(Xt[tr],Yt[tr],C_ALL,EPOCHS//2),
                              Xt[te],Y[te],G[te]))
    print(f"② Baseline 60-ch: {np.mean(acc_b):.3f} ± {np.std(acc_b):.3f}")

    # NeuroXAI
    print("③ 计算 NeuroXAI 权重 …")
    imp = channel_importance(base, trials, labels, n_samples)
    sel = np.argsort(-imp)[:k_top]; names = [CHAN_NAMES[i] for i in sel]
    print("Top-{} 电极: {}".format(k_top, names))

    exp = lambda idx: np.concatenate([idx,idx+N_CH,idx+2*N_CH])
    sel_idx, rnd_idx = exp(sel), exp(np.random.choice(N_CH,k_top,False))

    def cv_acc(idx):
        Xs = torch.tensor(Xw[:,idx][:,None,:,:],device=DEVICE)
        a=[]
        for tr,te in gkf.split(Xs,Yt,groups=G):
            a.append(evaluate(train_eegnet(Xs[tr],Yt[tr],len(idx),EPOCHS//2),
                              Xs[te],Y[te],G[te]))
        return np.mean(a), np.std(a)

    print("④ NeuroXAI-{} …".format(k_top)); acc_n = cv_acc(sel_idx)
    print("⑤ Random-{} …".format(k_top));   acc_r = cv_acc(rnd_idx)

    print(f"✔ NeuroXAI-{k_top}: {acc_n[0]:.3f} ± {acc_n[1]:.3f}")
    print(f"✔ Random-{k_top}:  {acc_r[0]:.3f} ± {acc_r[1]:.3f}")

    out = ROOT/f"results/FB_eegnet_vs_random_top{k_top}.json"
    json.dump({"k":k_top,"names_neuro":names,
               "acc_base":[*map(float,acc_b)],
               "acc_neuro":[*map(float,acc_n)],
               "acc_rand":[*map(float,acc_r)]},
              open(out,"w"), indent=2)
    print("✔ 结果存到", out)

# ---------------- CLI ----------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    p=argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--n_samples", type=int, default=3000)
    a=p.parse_args(); main(a.k, a.n_samples)