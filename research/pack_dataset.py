"""Pack TreeFinder GeoTIFF tiles into two memory-mappable arrays and collect per-tile statistics.

Outputs (datasets/packed/):
  x.npy      uint8 [N, 4, 224, 224]  R, G, B, NIR (row order = tile_info224_v3_aux.csv)
  y.npy      uint8 [N, 224, 224]     0 background, 1 dead tree, 255 no-data
  tile_stats.csv                     per-tile band moments and dead-tree component sizes

Run: python research/pack_dataset.py [--workers 8]
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
TILES = ROOT / "datasets" / "tiles224_v3"
OUT = ROOT / "datasets" / "packed"


def read(name: str):
    with rasterio.open(TILES / name) as src:
        arr = src.read()
    return arr[:4].astype(np.uint8), arr[4].astype(np.uint8)


def components(mask: np.ndarray) -> list[int]:
    from scipy import ndimage
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return []
    return np.bincount(labeled.ravel())[1:].tolist()


def stats(args):
    index, name = args
    x, y = read(name)
    valid = y != 255
    dead = y == 1
    live = valid & ~dead
    xf = x.astype(np.float32)
    ndvi = (xf[3] - xf[0]) / (xf[3] + xf[0] + 1e-6)
    row = {"index": index, "n_valid": int(valid.sum()), "n_dead": int(dead.sum())}
    for b, band in enumerate("RGBN"):
        v = xf[b][valid]
        row[f"mean_{band}"] = float(v.mean()) if v.size else np.nan
        row[f"std_{band}"] = float(v.std()) if v.size else np.nan
        row[f"dead_mean_{band}"] = float(xf[b][dead].mean()) if dead.any() else np.nan
        row[f"live_mean_{band}"] = float(xf[b][live].mean()) if live.any() else np.nan
    row["mean_ndvi"] = float(ndvi[valid].mean()) if valid.any() else np.nan
    row["dead_ndvi"] = float(ndvi[dead].mean()) if dead.any() else np.nan
    row["live_ndvi"] = float(ndvi[live].mean()) if live.any() else np.nan
    comps = components(dead)
    row["n_components"] = len(comps)
    row["component_sizes"] = json.dumps(comps)
    return row, x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    info = pd.read_csv(ROOT / "datasets" / "tile_info224_v3_aux.csv")
    n = len(info)
    OUT.mkdir(parents=True, exist_ok=True)
    xs = np.lib.format.open_memmap(OUT / "x.npy", mode="w+", dtype=np.uint8, shape=(n, 4, 224, 224))
    ys = np.lib.format.open_memmap(OUT / "y.npy", mode="w+", dtype=np.uint8, shape=(n, 224, 224))
    rows = []
    with Pool(args.workers) as pool:
        for k, (row, x, y) in enumerate(pool.imap(stats, enumerate(info.FileName), chunksize=32)):
            xs[row["index"]] = x
            ys[row["index"]] = y
            rows.append(row)
            if k % 2000 == 0:
                print(k, n, flush=True)
    xs.flush()
    ys.flush()
    out = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    out.insert(1, "FileName", info.FileName.values)
    out.to_csv(OUT / "tile_stats.csv", index=False)
    print("done", n)


if __name__ == "__main__":
    main()
