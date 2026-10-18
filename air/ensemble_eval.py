"""Probability-averaging ensembles of already-trained air/ checkpoints (no retraining).

For every (split type, seed) it finds the canonical plain runs of the requested model families in the training logs,
averages their sigmoid outputs on the validation and test splits and reports, for every member and every ensemble:
  F1 at the benchmark threshold 0.5, F1 at the threshold picked on validation, AP, precision / recall.

    python air/ensemble_eval.py --logs LOGS --ckpts DIR... --configs_dir configs_hpc --out out.json
"""
import argparse
import ast
import glob
import itertools
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from air.dataset import AirDataset                         # noqa: E402
from air.eval_extra import BINS, average_precision, best_thr, metrics   # noqa: E402
from air.main import AirInput, build                        # noqa: E402

FAMILIES = {"cnx": ("convnext_upernet", "none", ""), "dino": ("dinov2_b", "none", "base"), "tms": ("treemortseg", "none", "")}


def discover(logs):
    runs = {}
    for f in glob.glob(os.path.join(logs, "air*.log")):
        t = open(f, errors="ignore").read()
        a = re.search(r"args=(\{.*\})", t)
        if not a or "Combined metrics" not in t:
            continue
        g = ast.literal_eval(a.group(1))
        if g.get("split_seed") or g.get("norm", "none") != "none" or g.get("photo") or g.get("ema", 0) or g.get("tta") or \
                g.get("sampler", "random") != "random" or g.get("up", 1.0) != 1.0 or g.get("sched", "exp") != "exp" or g.get("loss", "bce_dice") != "bce_dice":
            continue
        split = "site" if "site" in g["config"] else "tile"
        tag = g.get("tag", "") or ""
        key = (split, g["model"], g.get("aux", "none") if g["model"] == "treemortseg" else "none", tag, g["seed"])
        tagn = g["model"] + (f"-{g['aux']}" if g["model"] == "treemortseg" else "") + (f"-{tag}" if tag else "")
        runs[key] = (f"air{str(g['exp_id']).zfill(4)}_{tagn}", g)
    return runs


def load(exp, args, ckpts, cfg_dir, device):
    ck = next(Path(d) / exp / f"{exp}_best.pth" for d in ckpts if (Path(d) / exp / f"{exp}_best.pth").exists())
    model = AirInput(build(args["model"], 5), norm=args.get("norm", "none")).to(device)
    sd = {k[4:] if k.startswith("net.") else k: v for k, v in torch.load(ck, map_location=device).items()}
    (model if any(k.startswith("net.") for k in sd) else model.net).load_state_dict(sd)
    return model.eval()


@torch.no_grad()
def run_split(models, loader, device, tta):
    names = list(models)
    combos = [c for r in (1, 2, 3) for c in itertools.combinations(names, r)]
    hist = {c: torch.zeros(2, BINS, dtype=torch.long, device=device) for c in combos}
    for b in loader:
        x = b["image"].to(device)
        y = b["label"].to(device)
        valid = ~b["no_data_mask"].to(device)
        probs = {}
        for n in names:
            acc = 0
            views = [(None, None)] + ([((3,), (3,)), ((2,), (2,)), ((2, 3), (2, 3))] if tta else [])
            for fl, back in views:
                xi = x if fl is None else x.flip(fl)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    o = models[n](xi)
                o = (o[0] if isinstance(o, (tuple, list)) else o).float().sigmoid().squeeze(1)
                acc = acc + (o if back is None else o.flip([d - 1 for d in back]))
            probs[n] = acc / len(views)
        pos, neg = (y == 1) & valid, (y != 1) & valid
        for c in combos:
            p = sum(probs[n] for n in c) / len(c)
            bin_ = (p * BINS).long().clamp(0, BINS - 1)
            hist[c][0] += torch.bincount(bin_[pos], minlength=BINS)
            hist[c][1] += torch.bincount(bin_[neg], minlength=BINS)
    return {c: h.cpu().numpy() for c, h in hist.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--configs_dir", default="configs_hpc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", nargs="+", default=["cnx", "dino", "tms"])
    ap.add_argument("--splits", nargs="+", default=["tile", "site"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[2025, 2024, 2023])
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = discover(a.logs)
    out = {}
    for split in a.splits:
        for seed in a.seeds:
            members = {}
            for fam in a.families:
                m, aux, tag = FAMILIES[fam]
                key = (split, m, aux, tag, seed)
                if key in runs:
                    members[fam] = runs[key]
            if len(members) < 2:
                print("skip", split, seed, "members:", list(members))
                continue
            cfgname = "res_random_site.yaml" if split == "site" else "res_random.yaml"
            cfg = yaml.safe_load(open(ROOT / a.configs_dir / cfgname))
            cfg["experiment"]["seed"] = seed
            cfg["experiment"]["num_workers"] = a.workers
            loaders = {s: DataLoader(AirDataset(cfg, s, "none"), batch_size=32, shuffle=False, num_workers=a.workers) for s in ("val", "test")}
            models = {fam: load(e, g, a.ckpts, a.configs_dir, device) for fam, (e, g) in members.items()}
            hv = run_split(models, loaders["val"], device, a.tta)
            ht = run_split(models, loaders["test"], device, a.tta)
            for c in ht:
                thr, f1v = best_thr(hv[c])
                out[f"{split}|{seed}|{'+'.join(c)}"] = {
                    "split": split, "seed": seed, "members": list(c), "val_thr": thr, "val_f1": f1v, "val_ap": average_precision(hv[c]),
                    "ap": average_precision(ht[c]), "at_0.5": metrics(ht[c], 0.5), "at_val_thr": metrics(ht[c], thr)}
                r = out[f"{split}|{seed}|{'+'.join(c)}"]
                print(f"{split} seed={seed} {'+'.join(c):14s} F1@0.5={100*r['at_0.5']['f1']:.1f} F1@val={100*r['at_val_thr']['f1']:.1f} AP={100*r['ap']:.1f}", flush=True)
            Path(a.out).write_text(json.dumps(out))
            del models
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
