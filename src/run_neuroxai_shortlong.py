#!/usr/bin/env python
# run_neuroxai_shortlong.py —— Filter-Bank EEGNet + NeuroXAI (60 ch, 3 bands)

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
FS   = 256
WIN_S  = 3.0;  WIN  = int(WIN_S * FS)     # 3-s 窗口
STEP_S = 0.25; STEP = int(STEP_S * FS)
BANDS  = [(4, 7), (8, 13), (14, 30)]     # θ / α / β
EPOCHS = 400
BATCH  = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(SEED); torch.manual_seed(SEED)

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data/Short_Long_words"
FILES = sorted(f for f in DATA.glob("*.mat") if "_8s" not in f.name)

# ---------- 构造 60-通道名表（删 0,9,32,63） ----------
ORIG_64 = [
    'Fp1','Fpz','Fp2','AF7','AF3','AFz','AF4','AF8',
    'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8',
    'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8',
    'P7','P5','P3','P1','Pz','P2','P4','P6','P8',
    'PO7','PO3','POz','PO4','PO8','O1','Oz','O2'
]  # ← 共 64

DROP_ID = {0, 9, 32, 63}
CHAN_NAMES = [name for i, name in enumerate(ORIG_64) if i not in DROP_ID]
assert len(CHAN_NAMES) == 60, f"Expect 60 names, got {len(CHAN_NAMES)}"

N_CH   = 60
N_BAND = len(BANDS)
C_ALL  = N_CH * N_BAND                # 180

# ---------------- 预处理：Filter-Bank ----------------
nt_b, nt_a = iirnotch(60, 30, fs=FS)

def preprocess_fb(sig):
    """输入 (64,T) → 输出 (60*3,T)"""
    sig = np.delete(sig, list(DROP_ID), axis=0)   # 先物理删除 4 通道
    bank = []
    for low, high in BANDS:
        b, a = butter(4, [low, high], fs=FS, btype='band')
        tmp  = filtfilt(b, a, sig, axis=1)
        tmp -= tmp.mean(axis=1, keepdims=True)
        tmp /= tmp.std(axis=1, keepdims=True) + 1e-6
        bank.append(tmp)
    return np.concatenate(bank, axis=0).astype(np.float32)      # (180,T)

def slide(sig):
    wins=[]
    for st in range(0, sig.shape[1]-WIN+1, STEP):
        wins.append(sig[:, st:st+WIN])
    return np.stack(wins)

# ---------------- 读数据 ----------------
def load_trials():
    trials, labels = [], []
    for f in FILES:
        mat = loadmat(f, simplify_cells=True)
        key = next(k for k in mat if k.endswith("last_beep"))
        for cls, tset in enumerate(mat[key]):
            for tr in tset:
                trials.append(preprocess_fb(tr))
                labels.append(cls)
    trials, labels = map(np.asarray, (trials, labels))
    # 类别均衡
    i0, i1 = np.where(labels==0)[0], np.where(labels==1)[0]
    n = min(len(i0), len(i1))
    keep = np.sort(np.hstack([i0[:n], i1[:n]]))
    return trials[keep], labels[keep]

# ---------------- 训练 & 评估 ----------------
def train_eegnet(X, y, C, lr=1e-3, epochs=EPOCHS):
    net = EEGNet(C, WIN).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss()
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=DEVICE)
        for beg in range(0, len(perm), BATCH):
            sl = perm[beg:beg+BATCH]
            Xb = X[sl] * (1 + 0.01 * torch.randn_like(X[sl]))  # 1 % 噪声增广
            opt.zero_grad()
            loss = lossf(net(Xb), y[sl]); loss.backward(); opt.step()
        sched.step()
    return net

def evaluate(net, X, y, g):
    net.eval(); preds=[]
    with torch.no_grad():
        for beg in range(0,len(X),BATCH):
            preds.append(net(X[beg:beg+BATCH]).argmax(1).cpu())
    preds = np.concatenate(preds); vote={}
    for p,i in zip(preds,g): vote.setdefault(i,[]).append(p)
    pred_trial={i:max(set(v), key=v.count) for i,v in vote.items()}
    true_trial={i:int(y[np.where(g==i)[0][0]]) for i in pred_trial}
    return np.mean([pred_trial[k]==true_trial[k] for k in pred_trial])

# ---------------- NeuroXAI 权重 ----------------
def get_channel_imp(baseline, X_trials, y_trials, n_samples):
    def clf_fn(batch):
        C,T = batch.shape[1], batch.shape[2]
        if T > WIN:
            st = (T-WIN)//2; batch = batch[:,:,st:st+WIN]
        elif T < WIN:
            pad = np.zeros((batch.shape[0], C, WIN-T), batch.dtype)
            batch = np.concatenate([batch, pad], 2)
        tensor = torch.tensor(batch[:,None,:,:], device=DEVICE)
        return torch.softmax(baseline(tensor), 1).cpu().numpy()

    brain  = BrainExplainer(kernel_width=25, class_names=['short','long'])
    global_exp = GlobalBrainExplainer(brain)
    global_exp.explain_instance(X_trials, y_trials, clf_fn, n_samples)

    imp_raw = np.array(
        [global_exp.explain_global_channel_importance().get(i, 0.0)
         for i in range(C_ALL)],
        dtype=np.float32
    )
    return np.mean(imp_raw.reshape(N_BAND, N_CH), axis=0)  # (60,)

# ---------------- 主流程 ----------------
def main(k_top, n_samples):
    print("① 载入 & 切窗 …")
    trials, labels = load_trials()

    X_win, y_win, g_win = [], [], []
    gid = 0
    for sig, lab in zip(trials, labels):
        w = slide(sig)
        X_win.append(w);   y_win.extend([lab]*len(w))
        g_win.extend([gid]*len(w)); gid += 1
    X_win = np.concatenate(X_win)
    y_arr = np.asarray(y_win); g_arr = np.asarray(g_win)
    X_t   = torch.tensor(X_win[:,None,:,:], device=DEVICE)
    y_t   = torch.tensor(y_arr, device=DEVICE)

    # -------- 训练 / 载入 180-ch 基线 --------
    ckpt = ROOT / "results/eegnet_fb_60.pt"
    baseline = EEGNet(C_ALL, WIN).to(DEVICE)
    need_retrain = True
    if ckpt.exists():
        try:
            baseline.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=True)
            need_retrain = False
            print("✔ 已加载匹配 ckpt")
        except RuntimeError as e:
            print("⚠ ckpt 不匹配，将重训\n", str(e).split('\n')[0])

    if need_retrain:
        print("⏳ 训练 180-通道基线 …")
        baseline = train_eegnet(X_t, y_t, C_ALL)
        ckpt.parent.mkdir(exist_ok=True); torch.save(baseline.state_dict(), ckpt)

    # -------- Baseline 10-fold --------
    print("② Baseline-60 10-fold …")
    gkf = GroupKFold(10); acc_base=[]
    for tr, te in gkf.split(X_t, y_t, groups=g_arr):
        net = train_eegnet(X_t[tr], y_t[tr], C_ALL, epochs=EPOCHS//2)
        acc_base.append(evaluate(net, X_t[te], y_arr[te], g_arr[te]))
    print(f"✔ 60-ch: {np.mean(acc_base):.3f} ± {np.std(acc_base):.3f}")

    # -------- NeuroXAI 权重 --------
    print("③ 计算 NeuroXAI 权重 …")
    imp = get_channel_imp(baseline, trials, labels, n_samples)
    sel_base = np.argsort(-imp)[:k_top]
    sel_names = [CHAN_NAMES[i] for i in sel_base]
    print(f"✔ Top-{k_top}: {sel_names}")

    expand = lambda idx: np.concatenate([idx, idx+N_CH, idx+2*N_CH])
    sel_idx  = expand(sel_base)
    rand_idx = expand(np.random.choice(N_CH, k_top, replace=False))

    # -------- NeuroXAI-K --------
    print("④ NeuroXAI-Top-K …")
    X_sel = torch.tensor(X_win[:, sel_idx][:,None,:,:], device=DEVICE)
    acc_neuro=[]
    for tr, te in gkf.split(X_sel, y_t, groups=g_arr):
        net = train_eegnet(X_sel[tr], y_t[tr], len(sel_idx), epochs=EPOCHS//2)
        acc_neuro.append(evaluate(net, X_sel[te], y_arr[te], g_arr[te]))
    print(f"✔ NeuroXAI-{k_top}: {np.mean(acc_neuro):.3f} ± {np.std(acc_neuro):.3f}")

    # -------- Random-K --------
    print("⑤ Random-Top-K …")
    X_rand = torch.tensor(X_win[:, rand_idx][:,None,:,:], device=DEVICE)
    acc_rand=[]
    for tr, te in gkf.split(X_rand, y_t, groups=g_arr):
        net = train_eegnet(X_rand[tr], y_t[tr], len(rand_idx), epochs=EPOCHS//2)
        acc_rand.append(evaluate(net, X_rand[te], y_arr[te], g_arr[te]))
    print(f"✔ Random-{k_top}: {np.mean(acc_rand):.3f} ± {np.std(acc_rand):.3f}")

    # -------- 保存 --------
    out = ROOT / f"results/FB_eegnet_vs_random_top{k_top}.json"
    json.dump({
        "k": k_top,
        "names_neuro": sel_names,
        "acc_base":  [float(np.mean(acc_base)),  float(np.std(acc_base))],
        "acc_neuro": [float(np.mean(acc_neuro)), float(np.std(acc_neuro))],
        "acc_rand":  [float(np.mean(acc_rand)),  float(np.std(acc_rand))]
    }, open(out, "w"), indent=2)
    print("✔ 结果已存到", out)

# ---------------- CLI ----------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16,
                        help="保留 Top-K 电极 (默认 16)")
    parser.add_argument("--n_samples", type=int, default=3000,
                        help="NeuroXAI 扰动样本数")
    args = parser.parse_args()
    main(args.k, args.n_samples)