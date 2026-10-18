"""Ground truth vs prediction galleries from the probability maps dumped by air/error_objects.py.

    python research/pred_gallery.py --dump DIR_with_npz --run SUBSTR --out research/reports/pred_gallery [--n 6]

For each of the three tile groups stored in the dump (miss / falsealarm / good) it draws, per tile:
  RGB + ground-truth outline (yellow) | false colour + prediction outline (cyan) | probability map | error map
  (green = true positive, red = missed dead pixel, blue = false alarm, grey = no data)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--run", required=True, help="substring of the npz file name, e.g. air6001_convnext_upernet")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=6)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

x = np.load("datasets/packed/x.npy", mmap_mode="r")
y = np.load("datasets/packed/y.npy", mmap_mode="r")
info = pd.read_csv("datasets/tile_info224_v3_aux.csv")
row = {f: i for i, f in enumerate(info.FileName)}
EPS = 1e-6


def stretch(v, valid):
    lo, hi = np.percentile(v[valid], (2, 98))
    return np.clip((v - lo) / (hi - lo + EPS), 0, 1)


def outline(m):
    return m & ~ndimage.binary_erosion(m) if m.any() else m


for f in sorted(glob.glob(os.path.join(a.dump, "*.npz"))):
    if a.run not in os.path.basename(f):
        continue
    z = np.load(f, allow_pickle=True)
    thr = float(z["thr"])
    tag = os.path.basename(f)[:-4]
    for kind in ("miss", "falsealarm", "good"):
        names = z[f"{kind}_names"][: a.n]
        prob = z[f"{kind}_prob"][: a.n].astype(np.float32) / 255.0
        if len(names) == 0:
            continue
        fig, ax = plt.subplots(len(names), 4, figsize=(13.5, 3.3 * len(names)))
        ax = np.atleast_2d(ax)
        for r, (nm, p) in enumerate(zip(names, prob)):
            i = row[str(nm)]
            t = np.asarray(x[i]).astype(np.float32)
            lab = np.asarray(y[i])
            valid = lab != 255
            gt = lab == 1
            pred = (p >= thr) & valid
            R, G, B, N = t
            rgb = np.dstack([stretch(R, valid), stretch(G, valid), stretch(B, valid)]); rgb[~valid] = .5
            fc = np.dstack([stretch(N, valid), stretch(R, valid), stretch(G, valid)]); fc[~valid] = .5
            err = np.full(gt.shape + (3,), 1.0)
            err[~valid] = .6
            err[pred & gt] = [0.1, 0.7, 0.1]
            err[~pred & gt] = [0.9, 0.1, 0.1]
            err[pred & ~gt] = [0.15, 0.3, 0.95]
            def ov(base, m, col):
                o = np.zeros(m.shape + (4,)); o[outline(m)] = col; return o
            ax[r, 0].imshow(rgb); ax[r, 0].imshow(ov(rgb, gt, [1, 1, 0, 1]))
            ax[r, 1].imshow(fc); ax[r, 1].imshow(ov(fc, gt, [1, 1, 0, 1])); ax[r, 1].imshow(ov(fc, pred, [0, 1, 1, 1]))
            ax[r, 2].imshow(np.where(valid, p, np.nan), cmap="magma", vmin=0, vmax=1); ax[r, 2].imshow(ov(rgb, gt, [0, 1, 0, 1]))
            ax[r, 3].imshow(err)
            tp, fn, fp = (pred & gt).sum(), (~pred & gt).sum(), (pred & ~gt).sum()
            ax[r, 0].set_ylabel("%s %s\nGT %d | TP %d FN %d FP %d" % (info.State[i], nm, gt.sum(), tp, fn, fp), fontsize=8)
            for c in range(4):
                ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        for c, t_ in enumerate(["RGB + ground truth (yellow)", "false colour + GT (yellow) + prediction (cyan)",
                                "predicted probability (green = GT)", "errors: green TP, red missed, blue false alarm"]):
            ax[0, c].set_title(t_, fontsize=9)
        title = {"miss": "tiles with a large dead patch (>256 px) that was missed", "falsealarm": "tiles with most isolated false alarms",
                 "good": "best-segmented tiles"}[kind]
        fig.suptitle(f"{tag}   threshold {thr:.2f}   {title}", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(a.out, f"{tag}.{kind}.png"), dpi=90)
        plt.close()
    print("done", tag)
