"""Georeference table + 8-neighbour lookup for every TreeFinder tile (tiles are contiguous 224 px = 134.4 m blocks of one source image).

    python research/tile_geo.py --info datasets/tile_info224_v3_aux.csv --tiles datasets/tiles224_v3 --out datasets/tile_geo.csv
Columns: FileName, crs, x0, y0 (upper-left, metres), lon, lat (tile centre), n_nb and nb_<dr>_<dc> (neighbour file or empty).
"""
import argparse

import pandas as pd
import rasterio
from rasterio.warp import transform

S = 224 * 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", required=True)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.info)
    rows = []
    for f in d.FileName:
        with rasterio.open(f"{a.tiles}/{f}") as r:
            t, crs = r.transform, str(r.crs)
        rows.append((f, crs, t.c, t.f))
    g = pd.DataFrame(rows, columns=["FileName", "crs", "x0", "y0"]).merge(d[["FileName", "ImageRawID", "State"]], on="FileName")
    lon, lat = [], []
    for crs, sub in g.groupby("crs"):
        lo, la = transform(crs, "EPSG:4326", (sub.x0 + S / 2).tolist(), (sub.y0 - S / 2).tolist())
        lon += list(zip(sub.index, lo)); lat += list(zip(sub.index, la))
    g["lon"] = pd.Series(dict(lon)); g["lat"] = pd.Series(dict(lat))
    g["col"] = ((g.x0 - g.groupby("ImageRawID").x0.transform("min")) / S).round().astype(int)
    g["row"] = ((g.groupby("ImageRawID").y0.transform("max") - g.y0) / S).round().astype(int)
    key = {(r.ImageRawID, r.row, r.col): r.FileName for r in g.itertuples()}
    n = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            g[f"nb_{dr}_{dc}"] = [key.get((r.ImageRawID, r.row + dr, r.col + dc), "") for r in g.itertuples()]
    g["n_nb"] = (g[[c for c in g.columns if c.startswith("nb_")]] != "").sum(1)
    g.to_csv(a.out, index=False)
    print(len(g), "tiles; mean neighbours %.2f; share with all 8: %.1f%%; with none: %.1f%%" % (g.n_nb.mean(), 100 * (g.n_nb == 8).mean(), 100 * (g.n_nb == 0).mean()))


if __name__ == "__main__":
    main()
