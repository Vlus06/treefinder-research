"""Does the augmentation (statistics bank + consistency) lose more on test tiles whose source image is heavily represented in training?

Hypothesis behind "why no gain on the random split": a test tile of a source image that also supplies many training tiles can profit from
the acquisition conditions (colour, brightness, labelling style) of that image; an augmentation that removes this cue should therefore cost
more on such tiles than on tiles of images with few training tiles.

    python research/leak_gradient.py --stats tile_stats_random.csv --manifest research/splits/official_random_seed2025.csv \
        --info datasets/tile_info224_v3_aux.csv --plain tile_dino_plain tile3_dino_plain --method tile_dino_method tile3_dino_method
Inputs: per-tile tp/fp/fn of finished runs (air/tile_stats.py; columns '<exp>_tp', '<exp>_fp', '<exp>_fn', exp = airNNNN_<model>-<tag>).
Output: pooled dead-tree F1 of the plain and the method runs (mean over seeds) per group of test tiles, and the difference.
"""
import argparse
import re

import numpy as np
import pandas as pd


def f1(tp, fp, fn):
    return 200.0 * tp / max(1.0, 2 * tp + fp + fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--manifest", required=True, help="split manifest with columns FileName, split")
    ap.add_argument("--info", required=True)
    ap.add_argument("--plain", nargs="+", required=True, help="tags of the plain runs")
    ap.add_argument("--method", nargs="+", required=True, help="tags of the method runs")
    ap.add_argument("--bins", type=int, default=3)
    a = ap.parse_args()
    s = pd.read_csv(a.stats)
    info = pd.read_csv(a.info)[["FileName", "ImageRawID"]]
    man = pd.read_csv(a.manifest)
    man = man.merge(info, on="FileName", how="left")
    n_train = man[man.split == "train"].groupby("ImageRawID").size().rename("n_train_same_image")
    n_all = man.groupby("ImageRawID").size().rename("n_tiles_image")
    s["FileName"] = s["file"].astype(str).str.replace(r".*[\\/]", "", regex=True)
    s = s.merge(info, on="FileName", how="left").merge(n_train, on="ImageRawID", how="left").merge(n_all, on="ImageRawID", how="left")
    s["n_train_same_image"] = s["n_train_same_image"].fillna(0)
    s["frac_train"] = s["n_train_same_image"] / s["n_tiles_image"].clip(lower=1)
    cols = {}
    for c in s.columns:
        m = re.match(r"(air\d+_.+)_tp$", c)
        if m:
            cols[m.group(1)] = m.group(1)
    runs = {"plain": [e for e in cols if any(e.endswith("-" + t) for t in a.plain)],
            "method": [e for e in cols if any(e.endswith("-" + t) for t in a.method)]}
    print("runs:", {k: len(v) for k, v in runs.items()})
    s["grp"] = pd.qcut(s["n_train_same_image"].rank(method="first"), a.bins,
                       labels=[f"G{i + 1}" for i in range(a.bins)])
    rows = []
    for g, d in [("all", s)] + [(str(k), v) for k, v in s.groupby("grp", observed=True)]:
        r = {"group": g, "tiles": len(d), "pos_px": int(d.n_pos.sum()), "train_tiles_same_image(median)": float(d.n_train_same_image.median())}
        for k, es in runs.items():
            r[k] = float(np.mean([f1(d[e + "_tp"].sum(), d[e + "_fp"].sum(), d[e + "_fn"].sum()) for e in es]))
        r["method-plain"] = r["method"] - r["plain"]
        rows.append(r)
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
