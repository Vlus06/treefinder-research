"""Which input representation separates dead trees from live canopy?

(1) Quantitative ranking: for ~20 derived channels, the ROC AUC of "dead vs live pixel"
      - pooled      : one global threshold over all sampled tiles (what a pixel classifier sees across sites)
      - within-tile : AUC computed inside each tile, then averaged (what remains after removing site/illumination shifts)
    AUC is reported as max(auc, 1-auc); the sign says whether dead trees are higher (+) or lower (-) than live canopy.
(2) Visual galleries with the dead-tree outline drawn on top (tiles chosen to be diverse, not cherry-picked by model).

    python research/feature_gallery.py            -> research/reports/feature_auc.csv, gallery_*.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

x = np.load("datasets/packed/x.npy", mmap_mode="r")
y = np.load("datasets/packed/y.npy", mmap_mode="r")
info = pd.read_csv("datasets/tile_info224_v3_aux.csv")
rng = np.random.default_rng(1)
EPS = 1e-6


def features(t):
    R, G, B, N = [t[i].astype(np.float32) / 255.0 for i in range(4)]
    s = R + G + B + EPS
    mx, mn = np.maximum(np.maximum(R, G), B), np.minimum(np.minimum(R, G), B)
    nir_loc = ndimage.uniform_filter(N, 7)
    f = {
        "R": R, "G": G, "B": B, "NIR": N,
        "brightness (R+G+B)/3": s / 3,
        "NDVI (N-R)/(N+R)": (N - R) / (N + R + EPS),
        "GNDVI (N-G)/(N+G)": (N - G) / (N + G + EPS),
        "SAVI 1.5(N-R)/(N+R+.5)": 1.5 * (N - R) / (N + R + 0.5),
        "NIR/R ratio": N / (R + 0.02),
        "NIR - R": N - R,
        "ExG (2G-R-B)/sum": (2 * G - R - B) / s,
        "VARI (G-R)/(G+R-B)": np.clip((G - R) / (G + R - B + 0.05), -2, 2),
        "red fraction R/sum": R / s,
        "blue fraction B/sum": B / s,
        "saturation (HSV)": (mx - mn) / (mx + EPS),
        "NIR texture (7x7 std)": np.sqrt(np.maximum(ndimage.uniform_filter(N * N, 7) - nir_loc ** 2, 0)),
        "NDVI minus local mean (31x31)": None,
    }
    nd = f["NDVI (N-R)/(N+R)"]
    f["NDVI minus local mean (31x31)"] = nd - ndimage.uniform_filter(nd, 31)
    return f


def auc(pos, neg):
    v = np.concatenate([pos, neg])
    r = v.argsort().argsort() + 1.0
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


cand = np.flatnonzero((info.LabelSize > 40).values)
tiles = np.sort(rng.choice(cand, 900, replace=False))
pool_d, pool_l, within = {}, {}, {}
for i in tiles:
    t = np.asarray(x[i])
    lab = np.asarray(y[i])
    dead = lab == 1
    live = (lab == 0)
    if dead.sum() < 20 or live.sum() < 300:
        continue
    di = rng.choice(np.flatnonzero(dead.ravel()), min(150, dead.sum()), replace=False)
    li = rng.choice(np.flatnonzero(live.ravel()), 300, replace=False)
    for k, f in features(t).items():
        fl = f.ravel()
        pool_d.setdefault(k, []).append(fl[di]); pool_l.setdefault(k, []).append(fl[li])
        within.setdefault(k, []).append(auc(fl[di], fl[li]))
rows = []
for k in pool_d:
    a = auc(np.concatenate(pool_d[k]), np.concatenate(pool_l[k]))
    w = float(np.mean(within[k]))
    rows.append({"channel": k, "pooled_AUC": max(a, 1 - a), "pooled_sign": "+" if a >= 0.5 else "-",
                 "within_tile_AUC": max(w, 1 - w), "within_sign": "+" if w >= 0.5 else "-",
                 "within_tile_AUC_sd": float(np.std(within[k]))})
df = pd.DataFrame(rows).sort_values("within_tile_AUC", ascending=False)
df.to_csv("research/reports/feature_auc.csv", index=False)
print(df.round(3).to_string(index=False))
best_name = df.iloc[0].channel

# -------------------------------------------------------------------------------------------------------------- galleries
def stretch(a, valid):
    lo, hi = np.percentile(a[valid], (2, 98))
    return np.clip((a - lo) / (hi - lo + EPS), 0, 1)


def outline(mask):
    return mask & ~ndimage.binary_erosion(mask, iterations=1) if mask.any() else mask


def panel(idx_list, fname, title):
    fig, ax = plt.subplots(len(idx_list), 5, figsize=(16, 3.2 * len(idx_list)))
    for r, i in enumerate(idx_list):
        t = np.asarray(x[i]); lab = np.asarray(y[i]); valid = lab != 255; dead = lab == 1
        R, G, B, N = [t[c].astype(np.float32) for c in range(4)]
        rgb = np.dstack([stretch(R, valid), stretch(G, valid), stretch(B, valid)]); rgb[~valid] = .5
        fc = np.dstack([stretch(N, valid), stretch(R, valid), stretch(G, valid)]); fc[~valid] = .5
        feats = features(t)
        ed = outline(ndimage.binary_dilation(dead, iterations=1))
        panels = [(rgb, "RGB"), (fc, "false colour (NIR,R,G)"),
                  (np.where(valid, feats["NDVI (N-R)/(N+R)"], np.nan), "NDVI"),
                  (np.where(valid, feats["blue fraction B/sum"], np.nan), "blue fraction B/(R+G+B)"),
                  (np.where(valid, feats[best_name], np.nan), best_name)]
        for c, (img, ttl) in enumerate(panels):
            a = ax[r, c]
            if img.ndim == 3:
                a.imshow(img)
            else:
                a.imshow(img, cmap="RdYlGn" if "NDVI" in ttl or "SAVI" in ttl else "gray")
            ov = np.zeros(dead.shape + (4,)); ov[ed] = [1, 1, 0, 1] if c in (0, 1) else [1, 0, 1, 1]
            a.imshow(ov)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(ttl, fontsize=10)
        ax[r, 0].set_ylabel("%s %s\n%d dead px" % (info.State[i], info.FileName[i], dead.sum()), fontsize=8)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=95)
    plt.close()


dens = info[(info.LabelSize > 0) & (info.NoDataSize < 6000)].copy()
dens["d"] = dens.LabelSize / (224 * 224 - dens.NoDataSize)
pick = {}
pick["sparse"] = dens[dens.d < dens.d.quantile(.25)].sample(5, random_state=2).index.tolist()
pick["medium"] = dens[(dens.d > dens.d.quantile(.45)) & (dens.d < dens.d.quantile(.65))].sample(5, random_state=2).index.tolist()
pick["dense"] = dens[dens.d > dens.d.quantile(.85)].sample(5, random_state=2).index.tolist()
for k, v in pick.items():
    panel(v, f"research/reports/gallery_{k}.png", f"Dead-tree density: {k}  (outline = annotated dead trees; yellow on colour panels, magenta on index panels)")
print("best channel by within-tile AUC:", best_name)
