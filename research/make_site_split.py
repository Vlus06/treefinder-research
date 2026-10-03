"""Create a reproducible, site-disjoint TreeFinder split and audit tile leakage.

Run: python research/make_site_split.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "tile_info224_v3_aux.csv"
OUT = ROOT / "research" / "splits" / "site_seed2025.csv"
AUDIT = ROOT / "research" / "reports" / "split_audit.json"
SEED = 2025


def counts(frame: pd.DataFrame) -> dict:
    return {
        "tiles": int(len(frame)),
        "sites": int(frame.ImageRawPath.nunique()),
        "positive_tiles": int((frame.LabelSize > 0).sum()),
        "positive_pixels": int(frame.LabelSize.sum()),
        "valid_pixels": int((224 * 224 * len(frame) - frame.NoDataSize.sum())),
    }


def main() -> None:
    df = pd.read_csv(DATA)
    assert df.FileName.is_unique
    assert df.ImageRawPath.notna().all()
    assert (df.LabelSize >= 0).all()
    assert (df.NoDataSize >= 0).all()
    assert ((df.LabelSize + df.NoDataSize) <= 224 * 224).all()

    # Stratify sites by state and whether a dead tree is present. Splitting at the
    # source image level prevents neighboring 224px tiles crossing partitions.
    sites = df.groupby("ImageRawPath", as_index=False).agg(
        State=("State", "first"), positive_pixels=("LabelSize", "sum")
    )
    sites["has_positive"] = sites.positive_pixels.gt(0)
    rng = np.random.default_rng(SEED)
    assignment = {}
    for _, group in sites.groupby(["State", "has_positive"], sort=True):
        names = group.ImageRawPath.to_numpy().copy()
        rng.shuffle(names)
        n = len(names)
        n_test = round(0.2 * n)
        n_val = round(0.1 * n)
        if n >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        if n_test + n_val >= n:
            n_val = max(0, n - n_test - 1)
        for path in names[:n_test]:
            assignment[path] = "test"
        for path in names[n_test:n_test + n_val]:
            assignment[path] = "val"
        for path in names[n_test + n_val:]:
            assignment[path] = "train"

    df["split"] = df.ImageRawPath.map(assignment)
    assert df.split.notna().all()
    split_sites = {s: set(df.loc[df.split == s, "ImageRawPath"]) for s in ["train", "val", "test"]}
    assert not (split_sites["train"] & split_sites["val"])
    assert not (split_sites["train"] & split_sites["test"])
    assert not (split_sites["val"] & split_sites["test"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # Reproduce the public random_split.py's default tile-level shuffling.
    old_rng = np.random.RandomState(SEED)
    tiles = df.FileName.unique().tolist()
    old_rng.shuffle(tiles)
    n_test = int(0.2 * len(tiles))
    n_trainval = int(0.8 * len(tiles))
    tile_partition = pd.DataFrame({"FileName": tiles})
    tile_partition["split"] = "unused"
    tile_partition.loc[:n_test - 1, "split"] = "test"
    tile_partition.loc[n_test:n_test + n_trainval - 1, "split"] = "trainval"
    old = df[["FileName", "ImageRawPath"]].merge(tile_partition, on="FileName", validate="one_to_one")
    old_site_sets = {s: set(old.loc[old.split == s, "ImageRawPath"]) for s in ["trainval", "test"]}
    overlap = old_site_sets["trainval"] & old_site_sets["test"]

    report = {
        "seed": SEED,
        "dataset": counts(df),
        "site_disjoint": {s: counts(df[df.split == s]) for s in ["train", "val", "test"]},
        "original_tile_random_test_site_overlap": {
            "overlapping_sites": len(overlap),
            "total_test_sites": len(old_site_sets["test"]),
            "fraction_of_test_sites": len(overlap) / len(old_site_sets["test"]),
            "test_tiles_from_overlapping_sites": int(old[(old.split == "test") & old.ImageRawPath.isin(overlap)].shape[0]),
            "total_test_tiles": n_test,
        },
    }
    AUDIT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
