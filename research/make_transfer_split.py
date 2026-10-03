"""Prepare site-disjoint East/West source and target pools for active transfer."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "tile_info224_v3_aux.csv"
WEST = set("AR AZ CA CO IA ID KS LA MN MO MT ND NE NM NV OK OR SD TX UT WA WY".split())
EAST = set("AL CT DE FL GA IL IN KY MA MD ME MI MS NC NH NJ NY OH PA RI SC TN VA VT WV WI".split())


def main() -> None:
    df = pd.read_csv(SOURCE)
    assert set(df.State) == WEST | EAST
    sites = df.groupby("ImageRawPath", as_index=False).agg(
        State=("State", "first"), positive_pixels=("LabelSize", "sum")
    )
    sites["has_positive"] = sites.positive_pixels.gt(0)
    for direction, source_states in (("west_to_east", WEST), ("east_to_west", EAST)):
        rng = np.random.default_rng(2025)
        assignment = {}
        for _, group in sites.groupby(["State", "has_positive"], sort=True):
            names = group.ImageRawPath.to_numpy().copy()
            rng.shuffle(names)
            n = len(names)
            if group.State.iloc[0] in source_states:
                n_val = min(max(1, round(0.1 * n)), n - 1) if n > 1 else 0
                for name in names[:n_val]:
                    assignment[name] = "source_val"
                for name in names[n_val:]:
                    assignment[name] = "source_train"
            else:
                n_test = min(max(1, round(0.4 * n)), n - 1) if n > 1 else 0
                for name in names[:n_test]:
                    assignment[name] = "target_test"
                for name in names[n_test:]:
                    assignment[name] = "target_pool"
        out = df.copy()
        out["split"] = out.ImageRawPath.map(assignment)
        assert out.split.notna().all()
        assert out.groupby("ImageRawPath").split.nunique().eq(1).all()
        target = ROOT / "research" / "splits" / f"{direction}_seed2025.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(target, index=False)
        print(direction, out.groupby("split").agg(tiles=("FileName", "size"), sites=("ImageRawPath", "nunique")).to_dict("index"))


if __name__ == "__main__":
    main()
