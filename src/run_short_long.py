"""
Paper-aligned baseline for Panachakel et al., *J. Neural Eng.* 15 (2018)

✔ 每位受试者单独训练          ✔ 10-fold CV（与论文一致）
✔ 排除 S10 / S14              ✔ 整段 5 s（无滑窗）
✔ 9 CSP × 2 通道              ✔ 每通道 12-D (db4 4-level, RMS/Var/Ent)
✔ 通道对 = 24-D               ✔ 4-layer (40-40-40-40) 网络，Dropout 10/30/30/0
"""

import numpy as np, pywt, torch, torch.nn as nn
from pathlib import Path
from scipy.io import loadmat
from mne.decoding import CSP
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ---------------- 常量 ----------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data/Short_Long_words"
EXCLUDE  = ("sub_10", "sub_14")            # 两名 trial 较少/质量差的受试者
EOG      = [0, 9, 32, 63]                  # 1,10,33,64 通道
FS, WIN  = 256, 5*256                      # 整段 5 s → 1280 样本
N_KEEP   = 9                               # 论文固定 9×2
PAIR_DIM = 24                              # 12×2
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# -------------- 小波 12-D 特征 --------------
def dwt_feats(sig):
    """db4 4 level → 4 detail bands × (RMS,Var,Ent) = 12-D"""
    coeffs = pywt.wavedec(sig, "db4", level=4)[1:]   # D4-D1 (ignore A4)
    feats=[]
    for arr in coeffs:
        rms  = np.sqrt((arr**2).mean())
        var  = arr.var()
        p    = (arr**2)/(arr**2).sum(); ent = -(p*np.log(p+1e-12)).sum()
        feats += [rms, var, ent]
    return np.asarray(feats, np.float32)             # 12 维

# -------------- 40-40-40-40 网络 --------------
class PaperNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PAIR_DIM, 40), nn.ReLU(),
            nn.BatchNorm1d(40), nn.Dropout(.10),

            nn.Linear(40, 40), nn.ReLU(),
            nn.BatchNorm1d(40), nn.Dropout(.30),

            nn.Linear(40, 40), nn.Tanh(),
            nn.BatchNorm1d(40), nn.Dropout(.30),

            nn.Linear(40, 40), nn.ReLU(),
            nn.BatchNorm1d(40),

            nn.Linear(40, 1)      # logits
        )
    def forward(self, x): return self.net(x).squeeze(1)

# ---------------- 主循环 ----------------
all_acc = []
files = sorted([f for f in DATA_DIR.glob("*.mat")
                if "_8s" not in f.name and not f.name.startswith(EXCLUDE)])

for f_mat in files:
    mat  = loadmat(f_mat, simplify_cells=True)
    key  = [k for k in mat if k.endswith("last_beep")][0]
    sig  = mat[key]                                        # (2 类, trials)

    # ---- 准备整段 5 s 60 通道信号 ----
    trials, labels = [], []
    for cls, row in enumerate(sig):                        # 0 = coop, 1 = in
        for ep in row:
            ep = np.delete(ep[:, :WIN], EOG, 0)            # 60×1280
            trials.append(ep)
            labels.append(cls)
    trials = np.stack(trials)
    labels = np.asarray(labels, np.float32)

    # ---- 9×2 CSP 通道 ----
    csp = CSP(n_components=2*N_KEEP, reg="ledoit_wolf", transform_into="csp_space")
    csp.fit(trials, labels)
    Wmax, Wmin = csp.filters_[:N_KEEP], csp.filters_[-N_KEEP:]

    # ---- 提 9 对 × 24-D 特征 ----
    X, y = [], []
    for ep, lab in zip(trials, labels):
        pair_feats=[]
        for i in range(N_KEEP):
            a, b = Wmax[i] @ ep, Wmin[i] @ ep              # 1-D (1280)
            vec = np.hstack([dwt_feats(a), dwt_feats(b)])  # 24-D
            vec = (vec - vec.mean()) / (vec.std()+1e-6)    # Z-score
            pair_feats.append(vec)
        X.append(np.stack(pair_feats))
        y.append(lab)
    X = torch.tensor(np.asarray(X), device=DEVICE)         # (T,9,24)
    y = torch.tensor(np.asarray(y), dtype=torch.float32, device=DEVICE)

    # ---- 10-fold CV ----
    cv = StratifiedKFold(10, shuffle=True, random_state=0)
    scores=[]
    for tr, te in cv.split(X.cpu(), y.cpu()):
        net   = PaperNet().to(DEVICE)
        opt   = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()

        for epoch in range(60):                            # 60 epoch 足够收敛
            net.train()
            for idx in torch.randperm(len(tr), device=DEVICE).split(64):
                xb = X[tr][idx].reshape(-1, PAIR_DIM)      # 9× 展平
                yb = y[tr][idx].unsqueeze(1).repeat(1, N_KEEP).flatten()
                opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()

        # ---- 测试 ----
        net.eval()
        with torch.no_grad():
            prob = torch.sigmoid(net(X[te].reshape(-1, PAIR_DIM)))
        pred = (prob.reshape(-1, N_KEEP).mean(1) > .5).cpu()
        scores.append(accuracy_score(y[te].cpu(), pred))
    acc_subj = np.mean(scores)
    all_acc.append(acc_subj)
    print(f"{f_mat.name:<35}  acc={acc_subj:.3f}")

print("\n=== Overall (paper-aligned) ===")
print(f"mean ± sd : {np.mean(all_acc):.3f} ± {np.std(all_acc):.3f}")
