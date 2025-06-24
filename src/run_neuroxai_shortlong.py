#!/usr/bin/env python
# run_neuroxai_shortlong.py  ——  EEGNet + 提升 Baseline 准确率的完整流程
import argparse, json, warnings
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import GroupKFold
from tqdm import trange

from eegnet_model import EEGNet                        # ← 切回 EEGNet
from neuroxai.explanation import BrainExplainer, GlobalBrainExplainer

# ---------------- 超参 ----------------
SEED = 0
FS = 256
WIN_S = 2.0; WIN = int(WIN_S * FS)          # 512
STEP_S = 0.25; STEP = int(STEP_S * FS)      # 滑窗更密
EPOCHS = 400                                # 训练更久
BATCH  = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(SEED); torch.manual_seed(SEED)

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data/Short_Long_words"
FILES = sorted(f for f in DATA.glob("*.mat") if "_8s" not in f.name)

CHAN_NAMES = [  # 省略，保持不变 …
 'Fp1','Fpz','Fp2','AF7','AF3','AFz','AF4','AF8','F7','F5','F3','F1','Fz','F2',
 'F4','F6','F8','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5',
 'C3','C1','Cz','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPz','CP2','CP4',
 'CP6','TP8','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO3','POz',
 'PO4','PO8','O1','Oz','O2'
]

# ---------------- 预处理 ----------------
bp_b, bp_a = butter(4, [8, 30], fs=FS, btype='band')   # 聚焦 8-30 Hz
nt_b, nt_a = iirnotch(60, 30, fs=FS)
def preprocess(sig):
    sig = filtfilt(nt_b, nt_a, sig, axis=1)
    sig = filtfilt(bp_b, bp_a, sig, axis=1)
    sig -= sig.mean(axis=1, keepdims=True)
    sig /= sig.std(axis=1, keepdims=True) + 1e-6
    return sig.astype(np.float32)

def slide(sig):
    wins=[]
    for st in range(0, sig.shape[1]-WIN+1, STEP):
        wins.append(sig[:, st:st+WIN])
    return np.stack(wins)

# ---------------- 数据加载 ----------------
def load_trials():
    trials, labels = [], []
    for f in FILES:
        mat = loadmat(f, simplify_cells=True)
        key = next(k for k in mat if k.endswith("last_beep"))
        for cls, tset in enumerate(mat[key]):
            for tr in tset:
                sig = preprocess(np.delete(tr, [0,9,32,63], axis=0))
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
            Xb = X[sl] * (1 + 0.01 * torch.randn_like(X[sl]))   # 1 % 噪声增广
            opt.zero_grad()
            loss = lossf(net(Xb), y[sl]); loss.backward(); opt.step()
        sched.step()
    return net

def evaluate(net, X, y, g):
    net.eval(); pred=[]
    with torch.no_grad():
        for beg in range(0,len(X),BATCH):
            pred.append(net(X[beg:beg+BATCH]).argmax(1).cpu())
    pred=np.concatenate(pred)
    vote={}
    for p,id in zip(pred,g): vote.setdefault(id,[]).append(p)
    pred_trial={id:max(set(v), key=v.count) for id,v in vote.items()}
    true_trial={id:int(y[np.where(g==id)[0][0]]) for id in pred_trial}
    return np.mean([pred_trial[k]==true_trial[k] for k in pred_trial])

# ---------------- NeuroXAI（保持不变） ----------------
def get_channel_importance(baseline, X_trials, y_trials, num_samples):
    def classifier_fn(batch):
        C,T = batch.shape[1], batch.shape[2]
        if T>WIN:
            st=(T-WIN)//2; batch=batch[:,:,st:st+WIN]
        elif T<WIN:
            pad=np.zeros((batch.shape[0],C,WIN-T),dtype=batch.dtype)
            batch=np.concatenate([batch,pad],axis=2)
        tensor=torch.tensor(batch[:,None,:,:],device=DEVICE)
        with torch.no_grad():
            prob=torch.softmax(baseline(tensor),1)
        return prob.cpu().numpy()

    brain = BrainExplainer(kernel_width=25,class_names=['short','long'])
    global_exp = GlobalBrainExplainer(brain)
    global_exp.explain_instance(X_trials,y_trials,classifier_fn,
                                num_samples=num_samples)
    imp_d = global_exp.explain_global_channel_importance()
    return np.array([imp_d.get(i,0.0) for i in range(len(CHAN_NAMES))],
                    dtype=np.float32)

# ---------------- 主流程 ----------------
def main(k_top, n_samples):
    print("① 读取并窗口化数据 …")
    trials, labels = load_trials()

    X_win,y_win,g_win=[],[],[]
    gid=0
    for sig,lab in zip(trials,labels):
        wins=slide(sig)
        X_win.append(wins)
        y_win.extend([lab]*len(wins))
        g_win.extend([gid]*len(wins)); gid+=1
    X_win=np.concatenate(X_win)
    y_arr=np.asarray(y_win); g_arr=np.asarray(g_win)
    X_t=torch.tensor(X_win[:,None,:,:],device=DEVICE)
    y_t=torch.tensor(y_arr,device=DEVICE)

    # -------- 训练 / 载入 EEGNet-60 --------
    ckpt = ROOT/"results/eegnet_60ch.pt"
    retrain = not ckpt.exists()
    baseline = EEGNet(60, WIN).to(DEVICE)
    if not retrain:
        try:
            baseline.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            print("✔ 已加载 EEGNet-60ch ckpt")
        except RuntimeError:
            retrain = True
    if retrain:
        print("⏳ 训练 EEGNet-60ch 基线 …")
        baseline = train_eegnet(X_t, y_t, 60)
        ckpt.parent.mkdir(exist_ok=True)
        torch.save(baseline.state_dict(), ckpt)
        print("✔ 新 ckpt 保存:", ckpt)

    # -------- Baseline-60 CV --------
    print("② Baseline-60 10-fold 评估 …")
    gkf = GroupKFold(10); acc_base=[]
    for tr,te in gkf.split(X_t,y_t,groups=g_arr):
        net=train_eegnet(X_t[tr],y_t[tr],60,epochs=EPOCHS//2)
        acc_base.append(evaluate(net,X_t[te],y_arr[te],g_arr[te]))
    print(f"✔ 60-ch: {np.mean(acc_base):.3f} ± {np.std(acc_base):.3f}")

    # -------- NeuroXAI --------
    print("③ 计算 NeuroXAI 权重 …")
    imp=get_channel_importance(baseline,trials,labels,n_samples)
    sel_idx=np.argsort(-imp)[:k_top]
    sel_names=[CHAN_NAMES[i] for i in sel_idx]
    print(f"✔ Top-{k_top} 通道:", sel_names)

    # -------- NeuroXAI-K CV --------
    print("④ NeuroXAI-Top-K 10-fold …")
    X_sel=torch.tensor(X_win[:,sel_idx][:,None,:,:],device=DEVICE)
    acc_neuro=[]
    for tr,te in gkf.split(X_sel,y_t,groups=g_arr):
        net=train_eegnet(X_sel[tr],y_t[tr],k_top,epochs=EPOCHS//2)
        acc_neuro.append(evaluate(net,X_sel[te],y_arr[te],g_arr[te]))
    print(f"✔ NeuroXAI-{k_top}: {np.mean(acc_neuro):.3f} ± {np.std(acc_neuro):.3f}")

    # -------- Random-K CV --------
    print("⑤ Random-Top-K 10-fold …")
    np.random.seed(42)
    rand_idx = np.random.choice(60, k_top, replace=False)
    X_rand = torch.tensor(X_win[:, rand_idx][:, None, :, :], device=DEVICE)

    acc_rand = []
    for tr, te in gkf.split(X_rand, y_t, groups=g_arr):
        net_r = train_eegnet(X_rand[tr], y_t[tr], k_top, epochs=EPOCHS // 2)
        acc_rand.append(evaluate(net_r, X_rand[te], y_arr[te], g_arr[te]))
    print(f"✔ Random-{k_top}: {np.mean(acc_rand):.3f} ± {np.std(acc_rand):.3f}")

    # -------- 保存结果 --------
    out = ROOT / f"results/eegnet_neuroxai_vs_random_top{k_top}.json"
    out.parent.mkdir(exist_ok=True)
    json.dump(
        {
            "k": k_top,
            "idx_neuro": sel_idx.tolist(),
            "names_neuro": sel_names,
            "acc_base": [float(np.mean(acc_base)), float(np.std(acc_base))],
            "acc_neuro": [float(np.mean(acc_neuro)), float(np.std(acc_neuro))],
            "acc_rand": [float(np.mean(acc_rand)), float(np.std(acc_rand))]
        },
        open(out, "w"),
        indent=2
    )
    print("✔ 结果已存到", out)


# ---------------- CLI ----------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16,
                        help="保留 Top-K 通道 (默认 16)")
    parser.add_argument("--n_samples", type=int, default=3000,
                        help="NeuroXAI 随机扰动样本数 (默认 3000)")
    args = parser.parse_args()
    main(args.k, args.n_samples)