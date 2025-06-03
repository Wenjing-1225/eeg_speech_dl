import numpy as np, pywt, torch, torch.nn as nn
from pathlib import Path
from scipy.io import loadmat
from mne.decoding import CSP
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ---------- 路径 & 全局常量 ----------
DATA_DIR = Path(__file__).resolve().parent.parent / "data/Short_Long_words"
EOG      = [0, 9, 32, 63]
FS, WIN  = 256, 5 * 256
SW_START = [0, FS, 2 * FS]                # 3×2 s
KEEP_SET = [0, 3, 6, 9, 12]               # 0 = 全 60 通道，其余 = K 对
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 小波 15 维特征 ----------
def dwt_feats(sig):
    feats = []
    for arr in pywt.wavedec(sig, "db4", level=4):          # A4+D4+D3+D2+D1
        rms  = np.sqrt((arr**2).mean())
        var  = arr.var()
        p    = (arr**2)/(arr**2).sum()
        ent  = -(p*np.log(p + 1e-12)).sum()
        feats += [rms, var, ent]
    return np.asarray(feats, np.float32)                  # 15-D

# ---------- BigNet，输入维度可变 ----------
class BigNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 60), nn.ReLU(), nn.BatchNorm1d(60), nn.Dropout(.15),
            nn.Linear(60, 60),     nn.ReLU(), nn.BatchNorm1d(60), nn.Dropout(.15),
            nn.Linear(60, 60),     nn.ReLU(), nn.BatchNorm1d(60), nn.Dropout(.15),
            nn.Linear(60, 60),     nn.ReLU(), nn.BatchNorm1d(60), nn.Dropout(.15),
            nn.Linear(60, 1)                                        # logits
        )
    def forward(self, x): return self.mlp(x).squeeze(1)

# ---------- 读取全部 .mat ----------
subj_files = sorted([f for f in DATA_DIR.glob("*.mat") if "_8s" not in f.name])
results = {k: [] for k in KEEP_SET}

for f_mat in subj_files:
    mat = loadmat(f_mat, simplify_cells=True)
    key = [k for k in mat if k.endswith("last_beep")][0]
    sig = mat[key]                                          # (2 类, trials)

    # ---- 3×2 s 滑窗 / trial ----
    trials, labels = [], []
    for cls, row in enumerate(sig):
        for ep in row:
            ep60 = np.delete(ep[:, :WIN], EOG, 0)           # 60×1280
            for s in SW_START:
                trials.append(ep60[:, s:s+2*FS])            # 60×512
                labels.append(cls)
    trials = np.stack(trials); labels = np.asarray(labels, np.float32)

    for K in KEEP_SET:
        if K == 0:                                          # === 全 60 通道 ===
            pair_dim  = 15
            pair_num  = 60
            feats = []
            for ep in trials:
                feats.append(np.stack([dwt_feats(ep[ch]) for ch in range(60)]))
        else:                                               # === CSP K 对 ===
            csp  = CSP(n_components=2*K, reg="ledoit_wolf", transform_into="csp_space")
            csp.fit(trials, labels)
            Wmax, Wmin = csp.filters_[:K], csp.filters_[-K:]
            pair_dim, pair_num = 30, K
            feats=[]
            for ep in trials:
                pairs=[]
                for i in range(K):
                    a, b = Wmax[i] @ ep, Wmin[i] @ ep
                    vec = np.hstack([dwt_feats(a), dwt_feats(b)])  # 30-D
                    vec = (vec-vec.mean())/(vec.std()+1e-6)
                    pairs.append(vec)
                feats.append(np.stack(pairs))
        X = torch.tensor(np.asarray(feats), device=DEVICE)
        y = torch.tensor(labels, dtype=torch.float32, device=DEVICE)

        # ---- 5-fold CV，保持与原脚本一致 ----
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        acc_fold=[]
        for tr, te in cv.split(X.cpu(), y.cpu()):
            net = BigNet(pair_dim).to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=3e-2,
                                    betas=(0.9,0.95))
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 120)
            lossf = nn.BCEWithLogitsLoss()
            best, patience = 0, 0
            for epoch in range(120):
                net.train()
                for b in torch.randperm(len(tr), device=DEVICE).split(64):
                    xb = X[tr][b].reshape(-1, pair_dim)
                    yb = y[tr][b].unsqueeze(1).repeat(1, pair_num).flatten()
                    opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
                sch.step()
                net.eval()
                prob = torch.sigmoid(net(X[tr].reshape(-1, pair_dim)))
                acc_now = (prob.reshape(-1,pair_num).mean(1) > .5).eq(y[tr].cpu()).float().mean()
                if acc_now > best: best, patience = acc_now, 0
                else: patience += 1
                if patience == 18: break
            net.eval()
            prob = torch.sigmoid(net(X[te].reshape(-1, pair_dim)))
            pred = (prob.reshape(-1, pair_num).mean(1) > .5).cpu()
            acc_fold.append(accuracy_score(y[te].cpu(), pred))
        results[K].append(np.mean(acc_fold))
    print(f"done {f_mat.name}")

# ---------- 输出 ----------
print("\nChannels |  acc_mean ± std")
print("---------------------------")
for k in KEEP_SET:
    ch = 60 if k == 0 else 2*k
    print(f"{ch:>8} |  {np.mean(results[k]):.3f} ± {np.std(results[k]):.3f}")