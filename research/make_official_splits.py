"""Materialise the paper's official splits (random, W-E, E-W, CO, Climate, Climate-hard, Forest, Forest-hard) as
tile manifests aligned with datasets/tile_info224_v3_aux.csv, by instantiating the authors' own dataset classes.

    python research/make_official_splits.py --seeds 2025 2026 2027
Writes research/splits/official_<scenario>_seed<S>.csv with columns FileName, split in {train,val,test,unused}.
"""
import argparse
import copy
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

SCENARIOS = {
    "random": "res_random.yaml",
    "w2e": "res_state_WE.yaml",        # train west, test east
    "e2w": "res_state_EW.yaml",
    "co": "res_state_CO.yaml",
    "climate": "res_climate1.yaml",    # train Cfa, test the rest
    "climate_hard": "res_climate3.yaml",  # train Dfa (humid continental) only
    "forest": "res_treetype1.yaml",    # train top-5 tree types
    "forest_hard": "res_treetype3.yaml",  # train maple only
}


def make_ds(cfg, split):
    m = cfg["data"]["split"]["method"]
    if m == "random":
        from data_loader.random_split import RandomSplitDataset as DS
    elif m == "by_state":
        from data_loader.by_state_split import StateSplitDataset as DS
    elif m == "by_climate":
        from data_loader.by_climate_split import ClimateSplitDataset as DS
    else:
        from data_loader.by_tree_split import TreeSplitDataset as DS
    return DS(cfg, split=split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027])
    a = ap.parse_args()
    aux = pd.read_csv(ROOT / "datasets" / "tile_info224_v3_aux.csv")
    out_dir = ROOT / "research" / "splits"
    for name, cfgfile in SCENARIOS.items():
        base = yaml.safe_load(open(ROOT / "configs" / cfgfile))
        for seed in a.seeds:
            cfg = copy.deepcopy(base)
            cfg["experiment"]["seed"] = seed
            lab = pd.Series("unused", index=aux.FileName)
            for split in ("train", "val", "test"):
                files = [os.path.basename(p) for p in make_ds(cfg, split).paths]
                lab.loc[files] = split
            df = pd.DataFrame({"FileName": aux.FileName.values, "split": lab.loc[aux.FileName].values})
            df.to_csv(out_dir / f"official_{name}_seed{seed}.csv", index=False)
            print(name, seed, df.split.value_counts().to_dict())


if __name__ == "__main__":
    main()
