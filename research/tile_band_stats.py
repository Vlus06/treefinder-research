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


def one(f):
    with rasterio.open(f"{TILES}/{f}") as r:
        a = r.read([1, 2, 3, 4]).astype(np.float32) / 255.0
    valid = a.sum(0) > 0
    if valid.sum() < 50:
        return None
    v = a[:, valid]
    return [f] + v.mean(1).tolist() + v.std(1).tolist()


def main():
    global TILES
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", required=True)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    TILES = a.tiles
    files = pd.read_csv(a.info).FileName.tolist()
    with Pool(a.procs) as p:
        rows = [r for r in p.map(one, files, chunksize=64) if r is not None]
    pd.DataFrame(rows, columns=["FileName"] + [f"m{i}" for i in range(4)] + [f"s{i}" for i in range(4)]).to_csv(a.out, index=False)
    print(len(rows), "tiles ->", a.out)


if __name__ == "__main__":
    main()
