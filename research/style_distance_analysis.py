"""Natural spectral shift: does the benefit of the statistics bank + consistency loss grow with the distance between a test tile's style and
the source bank?  Test tiles are split into quartiles by the distance of their 8 band statistics (mean / std of R, G, B, NIR) to the nearest
training tile (standardised space, as in StatBank.nearest); pooled dead-tree F1 per quartile is compared between groups of runs.

    python research/style_distance_analysis.py --config configs_hpc/res_state_CO.yaml --split_seed 2025 --tile_stats tile_stats_CO.csv \
        --group baseline=exp1,exp2,exp3 --group kaleido=exp4,exp5,exp6 --out out.json
Inputs: per-tile tp/fp/fn from air/tile_stats.py and datasets/tile_band_stats.csv (no labels are used for the style statistics).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from air.dataset import AirDataset      # noqa: E402

COLS = [f"m{i}" for i in range(4)] + [f"s{i}" for i in range(4)]


def f1(tp, fp, fn):
    return 200.0 * tp / max(1, 2 * tp + fp + fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split_seed", type=int, default=2025)
    ap.add_argument("--tile_stats", required=True)
    ap.add_argument("--group", action="append", required=True, help="name=exp1,exp2,... (experiment names as columns of tile_stats)")
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    cfg["experiment"]["seed"] = a.split_seed
    cfg["experiment"]["num_workers"] = 0
    train = [Path(str(p)).name for p in AirDataset(cfg, "train", "none").base.paths]
    stats = pd.read_csv(Path(cfg["data"]["root_dir"]) / "tile_band_stats.csv")
    if not (ROOT / cfg["data"]["root_dir"]).exists() and not Path(cfg["data"]["root_dir"]).exists():
        stats = pd.read_csv(ROOT / cfg["data"]["root_dir"] / "tile_band_stats.csv")
    bank = stats[stats.FileName.isin(set(train))][COLS].to_numpy(np.float64)
    sd = bank.std(0) + 1e-9
    ts = pd.read_csv(a.tile_stats)
    ts["FileName"] = ts["file"].astype(str).str.replace(r".*[\/]", "", regex=True)
    ts = ts.merge(stats[["FileName"] + COLS], on="FileName", how="inner")
    x = ts[COLS].to_numpy(np.float64)
    bs = bank / sd
    d = np.empty(len(x))
    for i in range(0, len(x), 256):
        blk = x[i:i + 256] / sd
        d[i:i + 256] = np.sqrt(((blk[:, None, :] - bs[None, :, :]) ** 2).sum(-1)).min(1)
    ts["dist"] = d
    ts["bin"] = pd.qcut(ts["dist"].rank(method="first"), a.bins, labels=False)
    res = {"n_test_tiles": int(len(ts)), "bins": [], "groups": {}}
    for b in range(a.bins):
        sub = ts[ts.bin == b]
        res["bins"].append({"bin": b, "dist_min": float(sub.dist.min()), "dist_max": float(sub.dist.max()), "tiles": int(len(sub)), "dead_pixels": int(sub.n_pos.sum())})
    for g in a.group:
        name, exps = g.split("=", 1)
        per = {}
        for e in exps.split(","):
            vals = [f1(ts[ts.bin == b][f"{e}_tp"].sum(), ts[ts.bin == b][f"{e}_fp"].sum(), ts[ts.bin == b][f"{e}_fn"].sum()) for b in range(a.bins)]
            vals.append(f1(ts[f"{e}_tp"].sum(), ts[f"{e}_fp"].sum(), ts[f"{e}_fn"].sum()))
            per[e] = vals
        arr = np.array(list(per.values()))
        res["groups"][name] = {"per_run": per, "mean": arr.mean(0).tolist(), "std": (arr.std(0, ddof=1) if len(arr) > 1 else np.zeros(arr.shape[1])).tolist()}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("bins:", [(r["tiles"], round(r["dist_min"], 2), round(r["dist_max"], 2)) for r in res["bins"]])
    for name, v in res["groups"].items():
        print(name, " ".join("%.1f" % m for m in v["mean"]), "(last = all tiles)")


if __name__ == "__main__":
    main()
