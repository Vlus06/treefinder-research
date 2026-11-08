"""Robustness to controlled spectral-style shifts (no retraining): each finished run is evaluated on the test split after the
four bands of every test tile are perturbed (brightness, gamma, colour cast, contrast, NIR gain; NDVI recomputed). Reports the pooled
dead-tree F1 per shift, averaged over the seeds of each group of runs.

    python air/style_shift_eval.py --logs LOGS --ckpts DIR... --configs_dir configs_hpc --config_name res_state_CO \
        --group plain=co_lr001_p_ck --group kaleido=co_fz001 --models dinov2_b --out out.json
"""
import argparse
import ast
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from air.dataset import AirDataset                      # noqa: E402
from air.main import AirInput, build                    # noqa: E402
from air.spectral_aug import _finish, _valid            # noqa: E402

SHIFTS = ["identity", "bright0.70", "bright0.85", "bright1.15", "bright1.30", "gamma0.70", "gamma1.40", "castRB+", "castRB-",
          "contrast0.60", "contrast1.40", "nir0.70", "nir1.30"]


def shift(x, name):
    if name == "identity":
        return x
    valid = _valid(x)
    v = x[:, :4].clone()
    if name.startswith("bright"):
        v = v * float(name[6:])
    elif name.startswith("gamma"):
        v = v.clamp(0, 1) ** float(name[5:])
    elif name == "castRB+":
        v[:, 0] *= 1.2
        v[:, 2] *= 0.8
    elif name == "castRB-":
        v[:, 0] *= 0.8
        v[:, 2] *= 1.2
    elif name.startswith("contrast"):
        n = valid.sum((2, 3), keepdim=True).clamp_min(1)
        mu = (v * valid).sum((2, 3), keepdim=True) / n
        v = (v - mu) * float(name[8:]) + mu
    elif name.startswith("nir"):
        v[:, 3] *= float(name[3:])
    return _finish(v, valid)


@torch.no_grad()
def evaluate(model, loader, device):
    cnt = {s: [0, 0, 0] for s in SHIFTS}
    for b in loader:
        x0 = b["image"].to(device)
        y = b["label"].to(device)
        valid = ~b["no_data_mask"].to(device)
        pos, neg = (y == 1) & valid, (y != 1) & valid
        for s in SHIFTS:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                o = model(shift(x0, s))
            o = (o[0] if isinstance(o, (tuple, list)) else o).float().squeeze(1)
            p = o > 0
            c = cnt[s]
            c[0] += int((p & pos).sum()); c[1] += int((p & neg).sum()); c[2] += int((~p & pos).sum())
    return {s: 2 * c[0] / max(1, 2 * c[0] + c[1] + c[2]) for s, c in cnt.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--configs_dir", default="configs_hpc")
    ap.add_argument("--config_name", required=True)
    ap.add_argument("--group", action="append", required=True, help="name=tag (one group per tag)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[2025, 2024, 2023])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(ROOT / a.configs_dir / f"{a.config_name}.yaml"))
    cfg["experiment"]["num_workers"] = a.workers
    loader = DataLoader(AirDataset(cfg, "test", "none"), batch_size=32, shuffle=False, num_workers=a.workers)
    tags = dict(g.split("=", 1) for g in a.group)
    runs = defaultdict(dict)
    for log in sorted(glob.glob(os.path.join(a.logs, "air*.log"))):
        t = open(log, errors="ignore").read()
        m = re.search(r"args=(\{.*\})", t)
        if not m or "Combined metrics" not in t:
            continue
        g = ast.literal_eval(m.group(1))
        if Path(g["config"]).stem != a.config_name or g["model"] not in a.models:
            continue
        for name, tag in tags.items():
            if g.get("tag", "") == tag and g["seed"] in a.seeds:
                exp = f"air{str(g['exp_id']).zfill(4)}_{g['model']}-{g['tag']}"
                ck = next((Path(d) / exp / f"{exp}_best.pth" for d in a.ckpts if (Path(d) / exp / f"{exp}_best.pth").exists()), None)
                if ck is not None:
                    runs[name][g["seed"]] = (g, ck)
    res = {}
    for name, rs in runs.items():
        per_seed = {}
        for seed, (g, ck) in sorted(rs.items()):
            net = AirInput(build(g["model"], 5), norm=g.get("norm", "none")).to(device)
            sd = {k[4:] if k.startswith("net.") else k: v for k, v in torch.load(ck, map_location=device).items()}
            (net if any(k.startswith("net.") for k in sd) else net.net).load_state_dict(sd)
            net.eval()
            per_seed[seed] = evaluate(net, loader, device)
            print(name, seed, " ".join("%s=%.1f" % (k, 100 * v) for k, v in per_seed[seed].items()), flush=True)
            del net
            torch.cuda.empty_cache()
        res[name] = {"seeds": sorted(per_seed), "per_seed": per_seed}
        res[name]["mean"] = {s: sum(per_seed[k][s] for k in per_seed) / len(per_seed) for s in SHIFTS}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
