#!/usr/bin/env python
# run_neuroxai_shortlong.py —— Filter-Bank EEGNet + NeuroXAI
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
WIN_S = 3.0; WIN  = int(WIN_S * FS)   # 3-s 窗口
STEP_S = 0.25; STEP = int(STEP_S * FS)
BANDS  = [(4, 7), (8, 13), (14, 30)]  # θ / α / β
EPOCHS = 400
BATCH  = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(SEED); torch.manual_seed(SEED)

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data/Short_Long_words"
FILES = sorted(f for f in DATA.glob("*.mat") if "_8s" not in f.name)

# 60-通道名称 (删 0,9,32,63)
CHAN_NAMES = [...]
N_CH  = len(CHAN_NAMES)          # 60
N_BAND = len(BANDS)              # 3
C_ALL = N_CH * N_BAND            # 180

# ---------------- 预处理：Filter-Bank ----------------
nt_b, nt_a = iirnotch(60, 30, fs=FS)
def preprocess_fb(sig):
    """输入 (C,T)-> 输出 (3C,T)"""
    band_sig = []
    for low, high in BANDS:
        b, a = butter(4, [low, high], fs=FS, btype='band')
        tmp  = filtfilt(b, a, sig, axis=1)
        tmp -= tmp.mean(axis=1, keepdims=True)
        tmp /= tmp.std(axis=1, keepdims=True) + 1e-6
        band_sig.append(tmp)
    return np.concatenate(band_sig, axis=0).astype(np.float32)

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
                sig = preprocess_fb(np.delete(tr, [0,9,32,63], axis=0))
                trials.append(sig); labels.append(cls)
    trials, labels = np.asarray(trials), np.asarray(labels)
    i0,i1 = np.where(labels==0)[0], np.where(labels==1)[0]
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
            Xb = X[sl]*(1+0.01*torch.randn_like(X[sl]))
            opt.zero_grad(); loss = lossf(net(Xb), y[sl])
            loss.backward(); opt.step()
        sched.step()
    return net

def evaluate(net, X, y, g):
    net.eval(); preds=[]
    with torch.no_grad():
        for beg in range(0,len(X),BATCH):
            preds.append(net(X[beg:beg+BATCH]).argmax(1).cpu())
    preds=np.concatenate(preds); vote={}
    for p,i in zip(preds,g): vote.setdefault(i,[]).append(p)
    pred_trial={i:max(set(v), key=v.count) for i,v in vote.items()}
    true_trial={i:int(y[np.where(g==i)[0][0]]) for i in pred_trial}
    return np.mean([pred_trial[k]==true_trial[k] for k in pred_trial])

# ---------------- NeuroXAI 权重 ----------------
def get_channel_imp(baseline, X_trials, y_trials, n_samples):
    def clf_fn(batch):
        C,T = batch.shape[1], batch.shape[2]
        if T>WIN: st=(T-WIN)//2; batch=batch[:,:,st:st+WIN]
        elif T<WIN:
            pad=np.zeros((batch.shape[0],C,WIN-T),dtype=batch.dtype)
            batch=np.concatenate([batch,pad],2)
        tensor=torch.tensor(batch[:,None,:,:],device=DEVICE)
        return torch.softmax(baseline(tensor),1).cpu().numpy()

    brain = BrainExplainer(kernel_width=25,class_names=['short','long'])
    global_exp=GlobalBrainExplainer(brain)
    global_exp.explain_instance(X_trials,y_trials,clf_fn,n_samples)
    imp_raw = np.array(
        [global_exp.explain_global_channel_importance().get(i,0.0)
         for i in range(C_ALL)],
        dtype=np.float32
    )
    # 把 3 个频带对同一电极的权重做平均 → 60 维
    imp = np.mean(imp_raw.reshape(N_BAND, N_CH), axis=0)
    return imp

# ---------------- 主流程 ----------------
def main(k_top, n_samples):
    print("① 读取并窗口化 …")
    trials, labels = load_trials()

    X_win,y_win,g_win, gid = [],[],[],0
    for sig,lab in zip(trials,labels):
        wins=slide(sig)
        X_win.append(wins); y_win.extend([lab]*len(wins))
        g_win.extend([gid]*len(wins)); gid+=1
    X_win=np.concatenate(X_win)
    y_arr=np.asarray(y_win); g_arr=np.asarray(g_win)
    X_t  = torch.tensor(X_win[:,None,:,:],device=DEVICE)
    y_t  = torch.tensor(y_arr, device=DEVICE)

    # -------- 训练 / 载入 180-ch 基线 --------
    ckpt = ROOT/"results/eegnet_fb_60.pt"
    baseline = EEGNet(C_ALL, WIN).to(DEVICE)
    if ckpt.exists():
        baseline.load_state_dict(torch.load(ckpt,map_location=DEVICE))
        print("✔ 已加载 ckpt")
    else:
        print("⏳ 训练基线 …")
        baseline = train_eegnet(X_t, y_t, C_ALL)
        ckpt.parent.mkdir(exist_ok=True)
        torch.save(baseline.state_dict(), ckpt)

    # -------- Baseline CV --------
    print("② Baseline-60 CV …")
    gkf=GroupKFold(10); acc_base=[]
    for tr,te in gkf.split(X_t,y_t,groups=g_arr):
        net=train_eegnet(X_t[tr],y_t[tr],C_ALL,epochs=EPOCHS//2)
        acc_base.append(evaluate(net,X_t[te],y_arr[te],g_arr[te]))
    print(f"✔ 60-ch: {np.mean(acc_base):.3f} ± {np.std(acc_base):.3f}")

    # -------- NeuroXAI 权重 --------
    print("③ 计算 NeuroXAI 权重 …")
    imp = get_channel_imp(baseline,trials,labels,n_samples)
    sel_base = np.argsort(-imp)[:k_top]        # 选电极索引 (0-59)
    sel_names= [CHAN_NAMES[i] for i in sel_base]
    print("✔ Top-{}: {}".format(k_top, sel_names))

    # 把每个电极在 3 个频带的通道索引全部收集
    def expand(idx): return np.concatenate([idx, idx+N_CH, idx+2*N_CH])
    sel_idx = expand(sel_base)
    rand_base = np.random.choice(N_CH, k_top, replace=False)
    rand_idx  = expand(rand_base)

    # -------- NeuroXAI-K CV --------
    print("④ NeuroXAI-Top-K CV …")
    X_sel=torch.tensor(X_win[:,sel_idx][:,None,:,:],device=DEVICE)
    acc_neuro=[]
    for tr,te in gkf.split(X_sel,y_t,groups=g_arr):
        net=train_eegnet(X_sel[tr],y_t[tr], len(sel_idx), epochs=EPOCHS//2)
        acc_neuro.append(evaluate(net,X_sel[te],y_arr[te],g_arr[te]))
    print(f"✔ NeuroXAI-{k_top}: {np.mean(acc_neuro):.3f} ± {np.std(acc_neuro):.3f}")

    # -------- Random-K CV --------
    print("⑤ Random-Top-K CV …")
    X_rand=torch.tensor(X_win[:,rand_idx][:,None,:,:],device=DEVICE)
    acc_rand=[]
    for tr,te in gkf.split(X_rand,y_t,groups=g_arr):
        net=train_eegnet(X_rand[tr],y_t[tr],len(rand_idx),epochs=EPOCHS//2)
        acc_rand.append(evaluate(net,X_rand[te],y_arr[te],g_arr[te]))
    print(f"✔ Random-{k_top}: {np.mean(acc_rand):.3f} ± {np.std(acc_rand):.3f}")

    # -------- 保存 --------
    out = ROOT/f"results/FB_eegnet_vs_random_top{k_top}.json"
    json.dump({
        "k":k_top,
        "names_neuro":sel_names,
        "acc_base":[float(np.mean(acc_base)),float(np.std(acc_base))],
        "acc_neuro":[float(np.mean(acc_neuro)),float(np.std(acc_neuro))],
        "acc_rand":[float(np.mean(acc_rand)),float(np.std(acc_rand))]
    }, open(out,"w"), indent=2)
    print("✔ Results saved to", out)


# ---------------- CLI ----------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=16,
                   help="保留 Top-K 电极 (默认 16)")
    p.add_argument("--n_samples", type=int, default=3000,
                   help="NeuroXAI 扰动样本数")
    args = p.parse_args(); main(args.k, args.n_samples)