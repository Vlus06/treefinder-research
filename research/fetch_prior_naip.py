"""Pilot: for TreeFinder tiles, find which NAIP year each tile comes from and cut the same footprint from the other years
(Microsoft Planetary Computer STAC, public data). Output: one npz per tile (uint8 [4,224,224] per year) + matches.csv.

    python research/fetch_prior_naip.py --geo datasets/tile_geo.csv --tiles datasets/tiles224_v3 --n 200 --out datasets/prior_pilot
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/naip"
S = 224 * 0.6


def token():
    return requests.get(SAS, timeout=60).json()["token"]


def crop(href, crs, l, b, r, t):
    with rasterio.open(href) as src:
        bl = transform_bounds(crs, src.crs, l, b, r, t)
        w = from_bounds(*bl, transform=src.transform)
        if w.col_off < 0 or w.row_off < 0 or w.col_off + w.width > src.width or w.row_off + w.height > src.height:
            return None
        return src.read(indexes=[1, 2, 3, 4], window=w, out_shape=(4, 224, 224), resampling=Resampling.bilinear)


def gray_corr(a, b):
    ga, gb = a[:3].astype(np.float32).mean(0).ravel(), b[:3].astype(np.float32).mean(0).ravel()
    ok = (ga > 0) & (gb > 0)
    return float(np.corrcoef(ga[ok], gb[ok])[0, 1]) if ok.sum() > 500 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", required=True)
    ap.add_argument("--info", default="datasets/tile_info224_v3_aux.csv")
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    g = pd.read_csv(a.geo).merge(pd.read_csv(a.info)[["FileName", "LabelSize"]], on="FileName")
    rng = np.random.RandomState(a.seed)
    pos = g[g.LabelSize > 1000].sample(a.n // 2, random_state=a.seed)
    rnd = g.drop(pos.index).sample(a.n - len(pos), random_state=a.seed)
    sel = pd.concat([pos, rnd])
    state = {"tok": token(), "t": time.time()}

    def one(r):
        if time.time() - state["t"] > 1500:
            state["tok"], state["t"] = token(), time.time()
        tok = state["tok"]
        try:
            with rasterio.open(f"{a.tiles}/{r.FileName}") as src:
                tile, crs = src.read([1, 2, 3, 4]), src.crs
            tile = tile.astype(np.uint8) if tile.dtype != np.uint8 else tile
            j = requests.post(STAC, json={"collections": ["naip"], "intersects": {"type": "Point", "coordinates": [r.lon, r.lat]}, "limit": 50}, timeout=60).json()
            crops, gsd = {}, {}
            for it in j.get("features", []):
                yr = int(it["properties"]["naip:year"])
                try:
                    c = crop(it["assets"]["image"]["href"] + "?" + tok, crs, r.x0, r.y0 - S, r.x0 + S, r.y0)
                except Exception:
                    c = None
                if c is None:
                    continue
                crops.setdefault(yr, c)
                gsd.setdefault(yr, it["properties"].get("gsd"))
            corr = {y: gray_corr(tile, c) for y, c in crops.items()}
            best = max(corr, key=lambda y: -1 if np.isnan(corr[y]) else corr[y]) if corr else None
            np.savez_compressed(out / f"{r.FileName[:-4]}.npz", tile=tile, **{f"y{y}": c for y, c in crops.items()})
            return dict(FileName=r.FileName, State=r.State, years=sorted(crops), gsd=gsd, best=best, corr=round(corr[best], 3) if best else None,
                        corrs={y: round(v, 3) for y, v in corr.items()})
        except Exception as e:
            return dict(FileName=r.FileName, State=r.State, error=repr(e)[:150])

    from concurrent.futures import ThreadPoolExecutor
    rows = []
    with ThreadPoolExecutor(a.threads) as ex:
        for k, row in enumerate(ex.map(one, list(sel.itertuples()))):
            rows.append(row)
            if k % 10 == 0:
                print(k, row, flush=True)
                pd.DataFrame(rows).to_csv(out / "matches.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "matches.csv", index=False)
    print("done", len(rows))


if __name__ == "__main__":
    main()
