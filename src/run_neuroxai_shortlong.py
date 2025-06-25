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

# ---------------- 基本参数 ----------------
SEED = 0
FS = 256
WIN_S = 3.0; WIN = int(WIN_S * FS)
STEP_S = 0.25; STEP = int(STEP_S * FS)
N_FFT, HOP = 128, 64               # STFT
EPOCHS = 120                       # 交叉验证折内用 EPOCHS//2
BATCH  = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

ORIG_NAMES = (
    [str(ch).strip() for ch in _raw["ch_names"]][:n_total]
    if "ch_names" in _raw else [f"Ch{i}" for i in range(n_total)]
)
DROP_FIXED = {0, 9, 32, 63} & set(range(n_total))
keep_idx = [i for i in range(n_total) if i not in DROP_FIXED]
while len(keep_idx) > 60:
    keep_idx.pop(0)

CHAN_NAMES = [ORIG_NAMES[i] for i in keep_idx]
assert len(CHAN_NAMES) == 60, f"still not 60, got {len(CHAN_NAMES)}"
DROP_ID = set(range(n_total)) - set(keep_idx)
N_CH = 60
F_BINS = N_FFT // 2 + 1            # 65

# ---------------- 预处理 ----------------
b_bp, a_bp = butter(4, [4, 30], fs=FS, btype="band")

def preprocess(sig):
    sig = np.delete(sig, list(DROP_ID), axis=0)             # (60,T)
    sig = filtfilt(b_bp, a_bp, sig, axis=1)
    sig -= sig.mean(1, keepdims=True)
    sig /= sig.std (1, keepdims=True) + 1e-6
    return sig.astype(np.float32)

# ---------------- 窗口 + STFT ----------------
_window = torch.hann_window(N_FFT, device=DEVICE)

def stft_tensor(x_np: np.ndarray) -> torch.Tensor:
    x = torch.tensor(x_np, device=DEVICE)                   # (C,T)
    s = torch.stft(x, N_FFT, hop_length=HOP, window=_window,
                   return_complex=False)                    # (C,F,T',2)
    s = torch.sqrt(s.pow(2).sum(-1) + 1e-6)                 # magnitude
    return s                                                # (C,F,T')

def slide_stft(sig):
    wins=[]
    for st in range(0, sig.shape[1]-WIN+1, STEP):
        wins.append(stft_tensor(sig[:, st:st+WIN]).cpu())
    return torch.stack(wins)                                # (n,C,F,T')

# ---------------- 数据读取 ----------------
def load_trials():
    trials, labels = [], []
    for f in FILES:
        mat = loadmat(f, simplify_cells=True)
        key = next(k for k in mat if k.endswith("last_beep"))
        for cls, tset in enumerate(mat[key]):
            for tr in tset:
                trials.append(preprocess(tr)); labels.append(cls)
    trials, labels = map(np.asarray, (trials, labels))
    i0,i1 = np.where(labels==0)[0], np.where(labels==1)[0]
    n = min(len(i0), len(i1)); keep = np.sort(np.hstack([i0[:n], i1[:n]]))
    return trials[keep], labels[keep]

# ---------------- CAN 模型 ----------------
class CAN(nn.Module):
    def __init__(self, F:int, T:int, n_cls:int=2, heads:int=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,(1,3),padding=(0,1)), nn.ELU(),
            nn.Conv2d(32,64,(1,3),padding=(0,1)), nn.ELU(),
            nn.AvgPool2d((1,2))                 # F → F/2
        )
        self.att = nn.MultiheadAttention(64, heads, batch_first=True)
        self.fc  = nn.Linear(64, n_cls)

    def forward(self, x):            # x:(B,C,F,T)
        b,c,f,t = x.shape
        x = x.view(b,1,c,f*t)        # (B,1,C, F*T)
        x = self.conv(x)             # (B,64,C,F/2*T)
        x = x.mean(2)                # 跨电极平均 → (B,64,F/2*T)
        x = x.permute(0,2,1)         # (B,seq,64)
        x,_ = self.att(x,x,x)
        return self.fc(x.mean(1))    # (B,n_cls)

# ---------------- 训练 / 评估 ----------------
def train_can(X, y_cpu, epochs=EPOCHS, lr=3e-3):
    _,_,F,T = X.shape
    net = CAN(F, T, n_cls=2).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    net.train()
    for _ in trange(epochs, leave=False):
        idx = torch.randperm(len(X))
        for beg in range(0, len(idx), BATCH):
            sl = idx[beg:beg+BATCH]
            xb = X[sl].to(DEVICE, non_blocking=True)
            yb = y_cpu[sl].to(DEVICE)
            opt.zero_grad(); loss = crit(net(xb), yb); loss.backward(); opt.step()
    return net

def eval_can(net, X, y_np, g_np):
    net.eval(); preds=[]
    with torch.no_grad():
        for beg in range(0,len(X),BATCH):
            preds.append(net(X[beg:beg+BATCH].to(DEVICE)).argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    vote={}
    for p,i in zip(preds,g_np): vote.setdefault(i,[]).append(p)
    pred = {i:max(set(v), key=v.count) for i,v in vote.items()}
    return np.mean([pred[i]==int(y_np[np.where(g_np==i)[0][0]]) for i in pred])

# ---------------- NeuroXAI 权重 ----------------
def channel_importance(baseline, trials, labels, n_samples):
    def clf(batch):                              # batch:(B,C,F,T)
        with torch.no_grad():
            out = baseline(batch.to(DEVICE))
        return torch.softmax(out,1).cpu().numpy()

    brain = BrainExplainer(25, ['short','long'])
    gexp  = GlobalBrainExplainer(brain)
    gexp.explain_instance(trials, labels, clf, n_samples)
    imp = [gexp.explain_global_channel_importance().get(i,0.0)
           for i in range(N_CH)]
    return np.asarray(imp, dtype=np.float32)

# ---------------- 主流程 ----------------
def main(k_top,n_samples):
    print("① 读取 trial 并做 STFT …")
    trials, labels = load_trials()

    X_list, Y, G, gid = [], [], [], 0
    for sig, lab in zip(trials, labels):
        w = slide_stft(sig); X_list.append(w)
        Y.extend([lab]*len(w)); G.extend([gid]*len(w)); gid += 1
    X = torch.cat(X_list)              # (B,C,F,T')
    Y_t = torch.tensor(Y)              # CPU tensor
    Y_np, G_np = np.asarray(Y), np.asarray(G)

    # -------- Baseline 10-fold --------
    print("② Baseline-60 CAN 10-fold …")
    gkf = GroupKFold(10); acc_b=[]
    for tr,te in gkf.split(X, Y_np, groups=G_np):
        net = train_can(X[tr], Y_t[tr], epochs=EPOCHS//2)
        acc_b.append(eval_can(net, X[te], Y_np, G_np[te]))
    print(f"Baseline-60: {np.mean(acc_b):.3f} ± {np.std(acc_b):.3f}")

    # -------- NeuroXAI 计算权重 --------
    print("③ NeuroXAI 权重 …")
    pre_net = train_can(X, Y_t, epochs=60)          # 轻训作解释器
    imp = channel_importance(pre_net, trials, labels, n_samples)
    sel = np.argsort(-imp)[:k_top]; sel_names=[CHAN_NAMES[i] for i in sel]
    print("Top-{} 电极: {}".format(k_top, sel_names))

    mask_neu = torch.zeros(N_CH, dtype=torch.bool); mask_neu[sel]=True
    mask_rnd = torch.zeros_like(mask_neu); mask_rnd[np.random.choice(N_CH,k_top,False)] = True

    def cv(mask):
        Xi = X[:,:,mask,:]
        a=[]
        for tr,te in gkf.split(Xi, Y_np, groups=G_np):
            net=train_can(Xi[tr],Y_t[tr],epochs=EPOCHS//2)
            a.append(eval_can(net,Xi[te],Y_np,G_np[te]))
        return np.mean(a), np.std(a)

    print("④ NeuroXAI-K …"); acc_neu=cv(mask_neu)
    print("⑤ Random-K  … "); acc_ran=cv(mask_rnd)

    print(f"NeuroXAI-{k_top}: {acc_neu[0]:.3f} ± {acc_neu[1]:.3f}")
    print(f"Random-{k_top}:  {acc_ran[0]:.3f} ± {acc_ran[1]:.3f}")

    out = ROOT/f"results/can_neuroxai_vs_random_{k_top}.json"
    json.dump({"k":k_top,"names_neuro":sel_names,
               "acc_base":list(map(float,acc_b)),
               "acc_neuro":list(map(float,acc_neu)),
               "acc_rand":list(map(float,acc_ran))},
              open(out,"w"), indent=2)
    print("✔ 结果已保存到", out)

# ---------------- CLI ----------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    p=argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=12, help="保留 Top-K 电极")
    p.add_argument("--n_samples", type=int, default=3000, help="NeuroXAI 样本数")
    a=p.parse_args(); main(a.k, a.n_samples)