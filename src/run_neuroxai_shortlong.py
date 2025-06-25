#!/usr/bin/env python
# run_neuroxai_shortlong.py —— NeuroXAI + Convolutional-Attention Network (CAN)

import argparse, json, warnings, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.model_selection import GroupKFold
from tqdm import trange

from neuroxai.explanation import BrainExplainer, GlobalBrainExplainer
print("Torch device:", torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
print("cuda.is_available =", torch.cuda.is_available())
# ---------------- 基本参数 ----------------
SEED = 0
FS   = 256
WIN_S = 3.0;  WIN  = int(WIN_S * FS)        # 3-s 窗口
STEP_S = 0.25; STEP = int(STEP_S * FS)
BANDS  = (4, 30)                            # 做 STFT 后再让卷积自己选频段
N_FFT  = 128; HOP = 64                     # → F_bin = 65
EPOCHS = 120
BATCH  = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data/Short_Long_words"
FILES = sorted(f for f in DATA.glob("*.mat") if "_8s" not in f.name)

# ---------- 动态生成 60-通道名表 ----------
_first = next(iter(FILES))
_raw   = loadmat(_first, simplify_cells=True)
_key   = next(k for k in _raw if k.endswith("last_beep"))
sig0   = _raw[_key][0][0]
n_total = sig0.shape[0]

if "ch_names" in _raw:
    ORIG_NAMES = [str(ch).strip() for ch in _raw["ch_names"]][:n_total]
else:
    ORIG_NAMES = [f"Ch{i}" for i in range(n_total)]

DROP_FIXED = {0, 9, 32, 63} & set(range(n_total))
idx_keep = [i for i in range(n_total) if i not in DROP_FIXED]
while len(idx_keep) > 60:
    idx_keep.pop(0)

CHAN_NAMES = [ORIG_NAMES[i] for i in idx_keep]
DROP_ID = set(range(n_total)) - set(idx_keep)
assert len(CHAN_NAMES) == 60, f"Still not 60 → {len(CHAN_NAMES)}"

N_CH   = 60
F_BINS = N_FFT // 2 + 1                   # STFT 输出频点 65
C_IN   = 1                                # STFT 给 CAN 的“输入通道”=1

# ---------------- 时域预滤（4-30 Hz） ----------------
b_bp, a_bp = butter(4, [4, 30], fs=FS, btype="band")

def preprocess(sig):
    sig = np.delete(sig, list(DROP_ID), axis=0)       # (60,T)
    sig = filtfilt(b_bp, a_bp, sig, axis=1)
    sig -= sig.mean(1, keepdims=True)
    sig /= sig.std (1, keepdims=True) + 1e-6
    return sig.astype(np.float32)

# ---------------- 切窗 & STFT ----------------
_window = torch.hann_window(N_FFT, device=DEVICE)

def stft_tensor(x_np: np.ndarray) -> torch.Tensor:
    """x_np:(C,T) → torch:(C,F,T ')"""
    x = torch.tensor(x_np, device=DEVICE)                # (C,T)
    spec = torch.stft(x, N_FFT, hop_length=HOP,
                      window=_window, return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)        # magnitude
    return spec                                          # (C,F,T')

def slide_stft(sig):
    """sig:(C,T) → (n_win,C,F,T')"""
    wins=[]
    for st in range(0, sig.shape[1]-WIN+1, STEP):
        seg = sig[:, st:st+WIN]
        wins.append(stft_tensor(seg).cpu())              # 移到 CPU 存
    return torch.stack(wins)                             # (n,C,F,T')

# ---------------- 数据加载 ----------------
def load_trials():
    trials, labels = [], []
    for f in FILES:
        mat = loadmat(f, simplify_cells=True)
        key = next(k for k in mat if k.endswith("last_beep"))
        for cls, tset in enumerate(mat[key]):
            for tr in tset:
                trials.append(preprocess(tr))
                labels.append(cls)
    trials, labels = map(np.asarray, (trials, labels))
    i0, i1 = np.where(labels==0)[0], np.where(labels==1)[0]
    n = min(len(i0), len(i1))
    keep = np.sort(np.hstack([i0[:n], i1[:n]]))
    return trials[keep], labels[keep]

# ---------------- CAN 模型 ----------------
class CAN(nn.Module):
    def __init__(self, C_sel:int, F:int, T:int, n_cls:int=2, head:int=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C_IN, 32, (1,3), padding=(0,1)), nn.ELU(),
            nn.Conv2d(32, 64, (1,3), padding=(0,1)), nn.ELU(),
            nn.AvgPool2d((1,2))                           # ↓F/2
        )
        F2 = F//2
        self.att = nn.MultiheadAttention(64, head, batch_first=True)
        self.fc = nn.Linear(64, n_cls)

    def forward(self, x):             # x:(B,C,F,T)
        b,c,f,t = x.shape
        x = x.view(b,1,c,f,t).reshape(b,1,c,f*t)    # 先把电极展成“宽度”
        x = self.conv(x)                            # (B,64,c,F2*T)
        x = x.mean(2)  # (B,64,F2*T)
        x = x.permute(0, 2, 1)  # (B,seq,64)
        x,_ = self.att(x,x,x)
        x = x.mean(1)
        return self.fc(x)

# ---------------- 训练 & 评估 ----------------
def train_can(X, y, C_sel, epochs=EPOCHS, lr=3e-3):
    _, _, F, Tt = X.shape
    net = CAN(C_sel, F, Tt, n_cls=2).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)
    loss = nn.CrossEntropyLoss()
    net.train()
    for ep in trange(epochs, leave=False):
        perm = torch.randperm(len(X), device=DEVICE)
        for beg in range(0, len(perm), BATCH):
            sl = perm[beg:beg+BATCH]
            xb = X[sl].to(DEVICE)
            opt.zero_grad(); l = loss(net(xb), y[sl]); l.backward(); opt.step()
    return net

def eval_can(net, X, y, g):
    net.eval(); pr=[]
    with torch.no_grad():
        for beg in range(0,len(X),BATCH):
            pr.append(net(X[beg:beg+BATCH].to(DEVICE)).argmax(1).cpu())
    pr = torch.cat(pr)
    vote={}
    for p,i in zip(pr,g): vote.setdefault(i,[]).append(int(p))
    pred = {i:max(set(v), key=v.count) for i,v in vote.items()}
    return np.mean([pred[i]==int(y[np.where(g==i)[0][0]]) for i in pred])

# ---------------- NeuroXAI 权重 ----------------
def ch_importance(baseline,x_trial,y_trial,n_sample):
    def clf(b):
        _,C,F,T = b.shape
        b = b.reshape(-1,C,F,T)
        with torch.no_grad(): z = baseline(b.to(DEVICE))
        return torch.softmax(z,1).cpu().numpy()

    brain = BrainExplainer(25,['short','long'])
    gexp  = GlobalBrainExplainer(brain)
    gexp.explain_instance(x_trial, y_trial, clf, n_sample)
    imp = [gexp.explain_global_channel_importance().get(i,0.0)
           for i in range(N_CH)]
    return np.asarray(imp,dtype=np.float32)

# ---------------- 主入口 ----------------
def main(k_top,n_samples):
    print("① 读取 trial 并做 STFT …")
    trials, labels = load_trials()
    xs,ys,gs,gid=[],[],[],0
    for sig,lab in zip(trials,labels):
        w = slide_stft(sig)           # (n,C,F,T')
        xs.append(w); ys.extend([lab]*len(w)); gs.extend([gid]*len(w)); gid+=1
    X = torch.cat(xs)
    Y = torch.tensor(ys, device=DEVICE)
    G = np.asarray(gs)

    # -------- 基线 CAN-60 --------
    print("② Baseline-60 CAN 10-fold …")
    gkf=GroupKFold(10); acc_b=[]
    for tr,te in gkf.split(X,Y,groups=G):
        net=train_can(X[tr],Y[tr],N_CH,epochs=EPOCHS//2)
        acc_b.append(eval_can(net,X[te],Y, G[te]))
    print(f"Baseline-60: {np.mean(acc_b):.3f} ± {np.std(acc_b):.3f}")

    # -------- NeuroXAI --------
    print("③ NeuroXAI 计算权重 …")
    base_net = train_can(X, Y, N_CH, epochs=60)      # 预训练作解释器
    imp = ch_importance(base_net, trials, labels, n_samples)
    sel = np.argsort(-imp)[:k_top]; sel_names=[CHAN_NAMES[i] for i in sel]
    print("Top-{} 电极: {}".format(k_top,sel_names))

    rand_sel = np.random.choice(N_CH,k_top,False)

    def build_idx(sel_elec):        # 把电极子集映射到 X 的维度
        mask = torch.zeros(N_CH,dtype=torch.bool)
        mask[sel_elec]=True
        return mask

    idx_neu = build_idx(sel); idx_rnd = build_idx(rand_sel)

    # -------- CV 对比 --------
    def cv(mask):
        Xi = X[:,:,mask,:]                           # (B,K,F,T')
        a=[]
        for tr,te in gkf.split(Xi,Y,groups=G):
            net=train_can(Xi[tr],Y[tr],mask.sum(),epochs=EPOCHS//2)
            a.append(eval_can(net,Xi[te],Y,G[te]))
        return np.mean(a),np.std(a)

    print("④ NeuroXAI-K …"); acc_neu=cv(idx_neu)
    print("⑤ Random-K  … "); acc_ran=cv(idx_rnd)

    print(f"NeuroXAI-{k_top}: {acc_neu[0]:.3f} ± {acc_neu[1]:.3f}")
    print(f"Random-{k_top}:  {acc_ran[0]:.3f} ± {acc_ran[1]:.3f}")

    out = ROOT/f"results/can_neuroxai_vs_random_{k_top}.json"
    json.dump({"k":k_top,"names_neuro":sel_names,
               "acc_base":list(map(float,acc_b)),
               "acc_neuro":list(map(float,acc_neu)),
               "acc_rand":list(map(float,acc_ran))},
              open(out,"w"),indent=2)
    print("结果已保存到",out)

# ---------------- CLI ----------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    p=argparse.ArgumentParser()
    p.add_argument("--k",type=int,default=12, help="保留 Top-K 电极")
    p.add_argument("--n_samples",type=int,default=3000, help="NeuroXAI 样本数")
    a=p.parse_args(); main(a.k,a.n_samples)