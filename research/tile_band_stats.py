"""Per-tile band statistics (mean / std of R,G,B,NIR on valid pixels, scale 0-1) for every TreeFinder tile -> the 'statistics bank'.

    python research/tile_band_stats.py --info datasets/tile_info224_v3_aux.csv --tiles datasets/tiles224_v3 --out datasets/tile_band_stats.csv
No labels are read.
"""
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import rasterio

TILES = None
BG_ONLY = False


def one(f):
    with rasterio.open(f"{TILES}/{f}") as r:
        arr = r.read([1, 2, 3, 4, 5]) if BG_ONLY else r.read([1, 2, 3, 4])
    a = arr[:4].astype(np.float32) / 255.0
    valid = a.sum(0) > 0
    if valid.sum() < 50:
        return None
    if BG_ONLY:                                   # statistics of the non-dead pixels only (label band == 0)
        bg = valid & (arr[4] == 0)
        if bg.sum() >= 50:
            valid = bg
    v = a[:, valid]
    return [f] + v.mean(1).tolist() + v.std(1).tolist()


def main():
    global TILES, BG_ONLY
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", required=True)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--bg_only", action="store_true", help="statistics on non-dead pixels only (reads the label band)")
    a = ap.parse_args()
    TILES = a.tiles
    BG_ONLY = a.bg_only
    files = pd.read_csv(a.info).FileName.tolist()
    with Pool(a.procs) as p:
        rows = [r for r in p.map(one, files, chunksize=64) if r is not None]
    pd.DataFrame(rows, columns=["FileName"] + [f"m{i}" for i in range(4)] + [f"s{i}" for i in range(4)]).to_csv(a.out, index=False)
    print(len(rows), "tiles ->", a.out)


if __name__ == "__main__":
    main()
