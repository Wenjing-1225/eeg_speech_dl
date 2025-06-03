import numpy as np, pywt, torch, torch.nn as nn
from pathlib import Path
from scipy.io import loadmat
from mne.decoding import CSP
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ---------------- 常量 ----------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data/Short_Long_words"
EOG      = [0, 9, 32, 63]
FS, WIN  = 256, 5*256                     # 5 s → 1280
SW_START = [0, FS, 2*FS]                 # 0–2, 1–3, 2–4 s
KEEP_SET = [9, 10, 12]                   # 网格
FEAT_CH  = 15                            # db4×4 级 × 3
PAIR_DIM = 30                            # 15×2
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# -------------- 小波特征 --------------
def dwt_feats(x):
    out = []
    for arr in pywt.wavedec(x, "db4", level=4):
        rms, var = np.sqrt((arr**2).mean()), arr.var()
        p = (arr**2)/(arr**2).sum(); ent = -(p*np.log(p+1e-12)).sum()
        out += [rms, var, ent]
    return np.asarray(out, np.float32)          # 15 维

# -------------- 大网络 --------------
class BigNet(nn.Module):
    def __init__(self, dim=PAIR_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 60), nn.ReLU(), nn.BatchNorm1d(60), nn.Dropout(.15),
            nn.Linear(60, 60), nn.ReLU(), nn.BatchNorm1d(60), nn.Dropout(.15),
            nn.Linear(60, 60), nn.ReLU(), nn.BatchNorm1d(60), nn.Dropout(.15),
            nn.Linear(60, 60), nn.ReLU(), nn.BatchNorm1d(60), nn.Dropout(.15),
            nn.Linear(60, 1)                                       # logits
        )
    def forward(self, x): return self.mlp(x).squeeze(1)

# ============ 主循环：每受试者 5-fold + 网格 =============
all_best = []
subj_files = sorted([f for f in DATA_DIR.glob("*.mat") if "_8s" not in f.name])

for f_mat in subj_files:
    mat  = loadmat(f_mat, simplify_cells=True)
    key  = [k for k in mat if k.endswith("last_beep")][0]
    sig  = mat[key]                                         # (2 类, trials)

    # ---- 预切窗 & 去 EOG ----
    trials, labels = [], []
    for cls, row in enumerate(sig):
        for ep in row:                                      # 64×≤1280
            ep = np.delete(ep[:, :WIN], EOG, 0)             # 60×1280
            for s in SW_START:
                trials.append(ep[:, s:s+2*FS])              # 60×512
                labels.append(cls)
    trials, labels = np.stack(trials), np.asarray(labels, np.float32)

    best_subj = 0.0
    # ---- 对每个 N_KEEP 网格一次 ----
    for K in KEEP_SET:
        # 1) CSP
        csp = CSP(n_components=2*K, reg="ledoit_wolf", transform_into="csp_space")
        csp.fit(trials, labels)
        Wmax, Wmin = csp.filters_[:K], csp.filters_[-K:]

        # 2) 特征 K×30
        X, y = [], []
        for ep, lab in zip(trials, labels):
            pair=[]
            for i in range(K):
                a, b = Wmax[i]@ep, Wmin[i]@ep
                vec = np.hstack([dwt_feats(a), dwt_feats(b)])
                vec = (vec-vec.mean())/(vec.std()+1e-6)     # Z-score
                pair.append(vec)
            X.append(np.stack(pair)); y.append(lab)
        X = torch.tensor(np.asarray(X), device=DEVICE)
        y = torch.tensor(np.asarray(y), device=DEVICE)

        # 3) 5-fold CV
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        scores=[]
        for tr, te in cv.split(X.cpu(), y.cpu()):
            net = BigNet().to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=3e-2,
                                    betas=(0.9, 0.95))
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 120)
            lossf = nn.BCEWithLogitsLoss()
            best, patience = 0, 0
            for epoch in range(120):
                net.train()
                for b in torch.randperm(len(tr), device=DEVICE).split(64):
                    xb = X[tr][b].reshape(-1,PAIR_DIM)
                    yb = y[tr][b].unsqueeze(1).repeat(1, K).flatten()
                    opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
                sch.step()
                # early-stop
                net.eval()
                pr = torch.sigmoid(net(X[tr].reshape(-1,PAIR_DIM)))
                acc_now = (pr.reshape(-1,K).mean(1) > .5).eq(y[tr].cpu()).float().mean()
                if acc_now > best: best, patience = acc_now, 0
                else: patience += 1
                if patience == 18: break
            # --- 测试折 ---
            net.eval()
            pr = torch.sigmoid(net(X[te].reshape(-1,PAIR_DIM)))
            pred = pr.reshape(-1,K).mean(1) > .5
            scores.append(accuracy_score(y[te].cpu(), pred.cpu()))
        best_subj = max(best_subj, np.mean(scores))         # 取最佳 K
    all_best.append(best_subj)
    print(f"{f_mat.name:<35}  best_acc={best_subj:.3f}")

print("\n=== Overall ===")
print(f"mean ± sd : {np.mean(all_best):.3f} ± {np.std(all_best):.3f}")