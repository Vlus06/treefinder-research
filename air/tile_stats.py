"""Per-tile tp/fp/fn (threshold 0.5) on the test split for finished runs and their probability-averaged ensemble.

    python air/tile_stats.py --config configs_hpc/res_random.yaml --split_seed 2025 --out tile_stats.csv \
        --exps air6007_convnext_upernet air6008_treemortseg-none air8028_dinov2_b-base --ckpt_dirs DIR1 DIR2
Used to decompose the pooled F1 (mega tiles vs normal tiles) and to compute oracle gains (perfect gate, etc.).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from air.dataset import AirDataset                      # noqa: E402
from air.distill import load_teacher                    # noqa: E402
from air.main import AirInput, build                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs_hpc/res_random.yaml")
    ap.add_argument("--split_seed", type=int, default=2025)
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--ckpt_dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(ROOT / a.config))
    cfg["experiment"]["seed"] = a.split_seed
    cfg["experiment"]["num_workers"] = a.workers
    ds = AirDataset(cfg, "test", "none")
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=a.workers)
    names = [Path(str(p)).name for p in ds.base.paths]
    models = {e: load_teacher(e, a.ckpt_dirs, build, lambda n: AirInput(n, norm="none"), device) for e in a.exps}
    rows, i0 = [], 0
    with torch.no_grad():
        for b in loader:
            x = b["image"].to(device)
            y = b["label"].to(device)
            valid = ~b["no_data_mask"].to(device)
            probs = {}
            for e, m in models.items():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    o = m(x)
                probs[e] = (o[0] if isinstance(o, (tuple, list)) else o).float().sigmoid().squeeze(1)
            probs["ens"] = sum(probs.values()) / len(models)
            pos, neg = (y == 1) & valid, (y != 1) & valid
            for k in range(x.shape[0]):
                r = {"idx": i0 + k, "file": str(names[i0 + k]) if names else "", "n_pos": int(pos[k].sum()), "n_valid": int(valid[k].sum())}
                for e, p in probs.items():
                    pr = p[k] > 0.5
                    r[f"{e}_tp"] = int((pr & pos[k]).sum())
                    r[f"{e}_fp"] = int((pr & neg[k]).sum())
                    r[f"{e}_fn"] = int((~pr & pos[k]).sum())
                rows.append(r)
            i0 += x.shape[0]
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("wrote", a.out, len(rows))


if __name__ == "__main__":
    main()
