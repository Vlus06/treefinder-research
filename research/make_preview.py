"""Render a few TreeFinder tiles the way they are meant to be seen (RGB, false colour NIR-R-G, NDVI, label overlay)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

x = np.load("datasets/packed/x.npy", mmap_mode="r")
y = np.load("datasets/packed/y.npy", mmap_mode="r")
info = pd.read_csv("datasets/tile_info224_v3_aux.csv")
rng = np.random.default_rng(3)
pos = info[(info.LabelSize > 300) & (info.NoDataSize < 3000)]
rows = []
for st in ["CO", "FL", "OR", "NY"]:
    sub = pos[pos.State == st]
    if len(sub):
        rows.append(int(sub.index[rng.integers(len(sub))]))


def stretch(a, valid):
    lo, hi = np.percentile(a[valid], (2, 98))
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


fig, ax = plt.subplots(len(rows), 4, figsize=(13, 3.3 * len(rows)))
for r, i in enumerate(rows):
    t = np.asarray(x[i]).astype(np.float32)
    lab = np.asarray(y[i])
    valid = lab != 255
    R, G, B, N = t
    rgb = np.dstack([stretch(R, valid), stretch(G, valid), stretch(B, valid)])
    fc = np.dstack([stretch(N, valid), stretch(R, valid), stretch(G, valid)])
    ndvi = (N - R) / (N + R + 1e-6)
    rgb[~valid] = 0.5
    fc[~valid] = 0.5
    ov = rgb.copy()
    ov[lab == 1] = [1, 0, 0]
    for c, (img, ttl) in enumerate([(rgb, "RGB"), (fc, "false colour (NIR,R,G)"), (None, "NDVI"), (ov, "dead trees (red)")]):
        a = ax[r, c]
        if img is None:
            a.imshow(np.where(valid, ndvi, np.nan), cmap="RdYlGn", vmin=-0.1, vmax=0.8)
        else:
            a.imshow(img)
        a.set_xticks([]); a.set_yticks([])
        if r == 0:
            a.set_title(ttl, fontsize=11)
    ax[r, 0].set_ylabel("%s  %s\n%d dead px" % (info.State[i], info.FileName[i], (lab == 1).sum()), fontsize=9)
plt.tight_layout()
plt.savefig("research/reports/tile_preview.png", dpi=110)
print("saved", rows)
