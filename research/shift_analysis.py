"""Where does the domain shift live, and which normalisation removes it?

(1) variance decomposition of per-tile mean band values into between-state, between-site (within state) and
    within-site components;
(2) pooled dead-vs-live separability (ROC AUC of a single pixel feature with one global threshold) under raw,
    tile-normalised, site-normalised and local-contrast features, overall and per region.

    python research/shift_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
info = pd.read_csv(ROOT / "datasets" / "tile_info224_v3_aux.csv")
st = pd.read_csv(ROOT / "datasets" / "packed" / "tile_stats.csv")
x = np.load(ROOT / "datasets" / "packed" / "x.npy", mmap_mode="r")
y = np.load(ROOT / "datasets" / "packed" / "y.npy", mmap_mode="r")
WEST = set("AR AZ CA CO IA ID KS LA MN MO MT ND NE NM NV OK OR SD TX UT WA WY".split())
site_of = info.ImageRawPath.values
state_of = info.State.values
out = {}

# ---- (1) variance decomposition ------------------------------------------------------------------------------
d = info[["ImageRawPath", "State"]].copy()
for c in ["mean_R", "mean_G", "mean_B", "mean_N", "mean_ndvi"]:
    d[c] = st[c].values
d = d.dropna()
dec = {}
for c in ["mean_R", "mean_G", "mean_B", "mean_N", "mean_ndvi"]:
    v = d[c]
    tot = v.var()
    st_mean = d.groupby("State")[c].transform("mean")
    site_mean = d.groupby("ImageRawPath")[c].transform("mean")
    dec[c] = {"between_state": float(st_mean.var() / tot),
              "between_site_within_state": float(((site_mean - st_mean).var()) / tot),
              "within_site": float(((v - site_mean).var()) / tot)}
out["variance_decomposition_of_tile_means"] = dec

# ---- per-site pooled statistics (valid pixels, all tiles) ------------------------------------------------------
def feats(t):
    f = t.astype(np.float32)
    ndvi = (f[3] - f[0]) / (f[3] + f[0] + 1e-6)
    bright = f[:3].mean(0)
    return np.stack([ndvi, f[0], f[1], f[2], f[3], bright])  # 6 features
names = ["ndvi", "red", "green", "blue", "nir", "brightness"]

s1, s2, n = {}, {}, {}
for i in range(len(info)):
    v = y[i] != 255
    if not v.any():
        continue
    F = feats(x[i])[:, v]
    k = site_of[i]
    s1[k] = s1.get(k, 0) + F.sum(1)
    s2[k] = s2.get(k, 0) + (F ** 2).sum(1)
    n[k] = n.get(k, 0) + v.sum()
smean = {k: s1[k] / n[k] for k in n}
sstd = {k: np.sqrt(np.maximum(s2[k] / n[k] - smean[k] ** 2, 1e-8)) for k in n}

# ---- (2) pooled AUC under normalisation schemes -----------------------------------------------------------
def auc(pos, neg):
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort() + 1.0
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

schemes = ["raw", "tile_z", "site_z", "local_contrast_31", "site_z_local_31"]
samples = {s: {"dead": [], "live": [], "region": []} for s in schemes}
pos_tiles = np.flatnonzero((st.n_dead.values > 0))
for i in pos_tiles:
    v = y[i] != 255
    dead = y[i] == 1
    live = v & ~dead
    if dead.sum() < 5 or live.sum() < 50:
        continue
    F = feats(x[i])
    k = site_of[i]
    vals = {}
    vals["raw"] = F
    tm = F[:, v].mean(1, keepdims=True)[:, :, None] if False else F[:, v].mean(1)[:, None, None]
    ts = F[:, v].std(1)[:, None, None] + 1e-3
    vals["tile_z"] = (F - tm) / ts
    sz = (F - smean[k][:, None, None]) / (sstd[k][:, None, None] + 1e-3)
    vals["site_z"] = sz
    # local contrast: deviation from the mean of the surrounding 31x31 window (valid pixels only)
    w = v.astype(np.float32)
    den = uniform_filter(w, 31, mode="nearest") + 1e-6
    loc = np.stack([uniform_filter(F[c] * w, 31, mode="nearest") / den for c in range(F.shape[0])])
    vals["local_contrast_31"] = F - loc
    vals["site_z_local_31"] = sz - ((loc - smean[k][:, None, None]) / (sstd[k][:, None, None] + 1e-3))
    nd, nl = min(dead.sum(), 150), min(live.sum(), 150)
    di = rng.choice(np.flatnonzero(dead.ravel()), nd, replace=False)
    li = rng.choice(np.flatnonzero(live.ravel()), nl, replace=False)
    reg = 1 if state_of[i] in WEST else 0
    for s in schemes:
        flat = vals[s].reshape(vals[s].shape[0], -1)
        samples[s]["dead"].append(flat[:, di].T)
        samples[s]["live"].append(flat[:, li].T)
        samples[s]["region"].append(np.full(nd + nl, reg))

res = {}
for s in schemes:
    D = np.concatenate(samples[s]["dead"])
    L = np.concatenate(samples[s]["live"])
    R = np.concatenate(samples[s]["region"])
    rd = R[: 0]  # placeholder
    nd_total = len(D)
    # regions of dead / live separately
    reg_d = np.concatenate([np.full(len(a), r[0]) for a, r in zip(samples[s]["dead"], samples[s]["region"])])
    reg_l = np.concatenate([np.full(len(a), r[0]) for a, r in zip(samples[s]["live"], samples[s]["region"])])
    res[s] = {}
    for j, nme in enumerate(names):
        # dead trees are darker / less green: lower feature value -> score = -feature
        res[s][nme] = {
            "pooled": auc(-D[:, j], -L[:, j]),
            "west": auc(-D[reg_d == 1, j], -L[reg_l == 1, j]),
            "east": auc(-D[reg_d == 0, j], -L[reg_l == 0, j]),
        }
out["pooled_auc_dead_vs_live"] = res
out["n_dead_samples"] = int(nd_total)
Path(ROOT / "research" / "reports").mkdir(exist_ok=True)
(ROOT / "research" / "reports" / "shift_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

print("variance decomposition (share of variance of per-tile means):")
for c, v in dec.items():
    print(f"  {c:10s} between-state {v['between_state']:.2f}  between-site {v['between_site_within_state']:.2f}  within-site {v['within_site']:.2f}")
print("\npooled AUC (dead darker than live), feature x scheme:")
print("scheme".ljust(20) + "".join(n_.ljust(12) for n_ in names))
for s in schemes:
    print(s.ljust(20) + "".join(f"{res[s][n_]['pooled']:.3f}".ljust(12) for n_ in names))
print("\nWest / East AUC for NDVI and brightness:")
for s in schemes:
    print(s.ljust(20), "ndvi W/E %.3f / %.3f" % (res[s]["ndvi"]["west"], res[s]["ndvi"]["east"]),
          " bright W/E %.3f / %.3f" % (res[s]["brightness"]["west"], res[s]["brightness"]["east"]))
