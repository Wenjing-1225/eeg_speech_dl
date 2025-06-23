#!/usr/bin/env python
# run_neuroxai_shortlong.py
# -------------------------------------------------------------------
# 1) 训练 / 载入 60 通道 EEGNet 基线
# 2) 用 LIME 版 NeuroXAI 计算通道权重
# 3) 比较 Baseline-60 / NeuroXAI-K / Random-K 三条准确率
# -------------------------------------------------------------------
import argparse, json, warnings
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import GroupKFold
from tqdm import trange

from eegnet_model import EEGNet
from neuroxai.explanation import BrainExplainer, GlobalBrainExplainer

# ---------------- 全局超参 ----------------
SEED   = 0
FS     = 256
WIN_S  = 2.0; WIN  = int(WIN_S * FS)   # 512
STEP_S = 0.5; STEP = int(STEP_S * FS)
EPOCHS = 150
BATCH  = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(SEED); torch.manual_seed(SEED)

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data/Short_Long_words"
FILES = sorted(f for f in DATA.glob("*.mat") if "_8s" not in f.name)

# 64-channel 名称（已去掉 0,9,32,63）
CHAN_NAMES = [
    'Fp1','Fpz','Fp2','AF7','AF3','AFz','AF4','AF8',
    'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8',
    'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8',
    'P7','P5','P3','P1','Pz','P2','P4','P6','P8',
    'PO7','PO3','POz','PO4','PO8','O1','Oz','O2'
]

# ---------------- 滤波 + 归一化 ----------------
bp_b, bp_a = butter(4, [4, 40], fs=FS, btype='band')
nt_b, nt_a = iirnotch(60, 30, fs=FS)
def preprocess(sig):
    sig = filtfilt(nt_b, nt_a, sig, axis=1)
    sig = filtfilt(bp_b, bp_a, sig, axis=1)
    sig -= sig.mean(axis=1, keepdims=True)
    sig /= sig.std(axis=1, keepdims=True) + 1e-6
    return sig.astype(np.float32)

def slide(sig):
    wins = []
    for st in range(0, sig.shape[1] - WIN + 1, STEP):
        wins.append(sig[:, st:st + WIN])
    return np.stack(wins)                # (n_win, C, T)

# ---------------- 数据加载 ----------------
def load_trials():
    trials, labels = [], []
    for f in FILES:
        mat  = loadmat(f, simplify_cells=True)
        key  = next(k for k in mat if k.endswith("last_beep"))
        raw  = mat[key]
        for cls, tset in enumerate(raw):
            for tr in tset:
                sig = preprocess(np.delete(tr, [0,9,32,63], axis=0))
                trials.append(sig); labels.append(cls)
    trials, labels = np.asarray(trials), np.asarray(labels)
    i0, i1 = np.where(labels==0)[0], np.where(labels==1)[0]
    n = min(len(i0), len(i1))
    keep = np.sort(np.hstack([i0[:n], i1[:n]]))
    return trials[keep], labels[keep]

# ---------------- 训练 & 评估 ----------------
def train_eegnet(X, y, C, lr=1e-3, epochs=EPOCHS):
    net = EEGNet(C=C, T=WIN).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    net.train()
    for ep in trange(epochs, desc=f"train {C}ch", leave=False):
        perm = torch.randperm(len(X), device=DEVICE)
        for beg in range(0, len(perm), BATCH):
            sl = perm[beg:beg+BATCH]
            opt.zero_grad()
            loss = lossf(net(X[sl]), y[sl])
            loss.backward(); opt.step()
    return net

def evaluate(net, X, y, g):
    net.eval(); preds=[]
    with torch.no_grad():
        for beg in range(0, len(X), BATCH):
            preds.append(net(X[beg:beg+BATCH]).argmax(1).cpu())
    preds = np.concatenate(preds)
    vote = {}
    for p, gid in zip(preds, g): vote.setdefault(gid, []).append(p)
    pred_trial={gid:max(set(v), key=v.count) for gid, v in vote.items()}
    true_trial={gid:int(y[np.where(g==gid)[0][0]]) for gid in pred_trial}
    return np.mean([pred_trial[t]==true_trial[t] for t in pred_trial])

# ---------------- NeuroXAI 权重 ----------------
def get_channel_importance(baseline, X_trials, y_trials,
                           num_samples=1000, replacement='mean'):

    # 裁/补为 WIN 长度
    def classifier_fn(batch):
        C, T_full = batch.shape[1], batch.shape[2]
        if T_full > WIN:
            st = (T_full - WIN)//2
            batch = batch[:, :, st:st+WIN]
        elif T_full < WIN:
            pad = np.zeros((batch.shape[0], C, WIN-T_full), dtype=batch.dtype)
            batch = np.concatenate([batch, pad], axis=2)
        tensor = torch.tensor(batch[:,None,:,:],
                              dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            prob = torch.softmax(baseline(tensor), dim=1)
        return prob.cpu().numpy()

    brain_exp  = BrainExplainer(kernel_width=25, class_names=['short','long'])
    global_exp = GlobalBrainExplainer(brain_exp)
    global_exp.explain_instance(x=X_trials, y=y_trials,
                                classifier_fn=classifier_fn,
                                num_samples=num_samples,
                                replacement_method=replacement)
    imp_dict = global_exp.explain_global_channel_importance()
    C = len(CHAN_NAMES)
    return np.array([imp_dict.get(i, 0.0) for i in range(C)], dtype=np.float32)

# ---------------- 主流程 ----------------
def main(k_top, n_samples):
    print("① 读取并窗口化数据 …")
    trials, labels = load_trials()

    X_win, y_win, g_win = [], [], []
    gid = 0
    for sig, lab in zip(trials, labels):
        wins = slide(sig)
        X_win.append(wins)
        y_win.extend([lab]*len(wins))
        g_win.extend([gid]*len(wins)); gid += 1
    X_win = np.concatenate(X_win)
    y_arr = np.asarray(y_win); g_arr = np.asarray(g_win)
    X_torch = torch.tensor(X_win[:,None,:,:], device=DEVICE)
    y_torch = torch.tensor(y_arr, device=DEVICE)

    # -------- 训练 / 载入 60-通道基线 --------
    ckpt = ROOT/"results/eegnet_60ch.pt"
    if ckpt.exists():
        baseline = EEGNet(60, WIN).to(DEVICE)
        baseline.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print("✔ 已加载 60-通道基线")
    else:
        print("⏳ 训练 60-通道基线 …")
        baseline = train_eegnet(X_torch, y_torch, 60)
        ckpt.parent.mkdir(exist_ok=True); torch.save(baseline.state_dict(), ckpt)

    # -------- Baseline-60 CV --------
    print("② Baseline-60 10-fold 评估 …")
    acc_base=[]
    gkf = GroupKFold(10)
    for _, te in gkf.split(X_torch, y_torch, groups=g_arr):
        acc_base.append(evaluate(baseline, X_torch[te], y_arr[te], g_arr[te]))
    print(f"✔ 60-通道准确率: {np.mean(acc_base):.3f} ± {np.std(acc_base):.3f}")

    # -------- NeuroXAI 通道 --------
    print("③ 计算 NeuroXAI 权重 …")
    imp = get_channel_importance(baseline, trials, labels,
                                 num_samples=n_samples, replacement='mean')
    idx_sorted = np.argsort(-imp)
    sel_idx = idx_sorted[:k_top]
    sel_names = [CHAN_NAMES[i] for i in sel_idx]
    print(f"✔ NeuroXAI Top-{k_top}: {sel_names}")

    # -------- NeuroXAI-K CV --------
    print("④ 重训 EEGNet (NeuroXAI-Top-K) …")
    X_sel = X_win[:, sel_idx, :]
    X_sel_t = torch.tensor(X_sel[:,None,:,:], device=DEVICE)
    acc_neuro=[]
    for tr, te in gkf.split(X_sel_t, y_torch, groups=g_arr):
        net_k = train_eegnet(X_sel_t[tr], y_torch[tr], k_top, lr=3e-3)  # 通道少→lr↑
        acc_neuro.append(evaluate(net_k, X_sel_t[te], y_arr[te], g_arr[te]))
    print(f"✔ NeuroXAI-{k_top} 准确率: {np.mean(acc_neuro):.3f} ± {np.std(acc_neuro):.3f}")

    # -------- Random-K CV --------
    print("⑤ 随机选 Top-K …")
    np.random.seed(42)
    rand_idx = np.random.choice(60, k_top, replace=False)
    X_rand = X_win[:, rand_idx, :]
    X_rand_t = torch.tensor(X_rand[:,None,:,:], device=DEVICE)
    acc_rand=[]
    for tr, te in gkf.split(X_rand_t, y_torch, groups=g_arr):
        net_r = train_eegnet(X_rand_t[tr], y_torch[tr], k_top, lr=3e-3)
        acc_rand.append(evaluate(net_r, X_rand_t[te], y_arr[te], g_arr[te]))
    print(f"✔ Random-{k_top} 准确率: {np.mean(acc_rand):.3f} ± {np.std(acc_rand):.3f}")

    # -------- 保存 --------
    out = ROOT/f"results/neuroxai_vs_random_top{k_top}.json"
    out.parent.mkdir(exist_ok=True)
    json.dump({
        "k": k_top,
        "neuroxai_idx": sel_idx.tolist(),
        "neuroxai_names": sel_names,
        "acc_base": [float(np.mean(acc_base)), float(np.std(acc_base))],
        "acc_neuro": [float(np.mean(acc_neuro)), float(np.std(acc_neuro))],
        "acc_rand": [float(np.mean(acc_rand)), float(np.std(acc_rand))]
    }, open(out, "w"), indent=2)
    print("✔ 结果已存到", out)

# ---------------- CLI ----------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16,
                        help="保留 Top-K 通道 (默认 16)")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="NeuroXAI 随机扰动样本数 (默认 1000)")
    args = parser.parse_args()
    main(args.k, args.n_samples)