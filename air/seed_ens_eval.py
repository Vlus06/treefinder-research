"""Seed ensemble and flip test-time augmentation of FINISHED runs (no retraining, no target-domain information).

For every (config, tag) group with several training seeds, the sigmoid outputs of the models of the different seeds are averaged
(ensemble) with and without 4-view flip TTA, and the pooled dead-tree F1 / precision / recall at 0.5 is reported next to the mean of
the single-model scores. Cost note: an ensemble of K models costs K times the inference of a single model.

    python air/seed_ens_eval.py --logs LOGS --ckpts DIR... --configs_dir configs_hpc --config_name res_state_CO \
        --tags co_dino_plain co_dino_method co_tms_fixed_w25 --models dinov2_b treemortseg --out out.json
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

FLIPS = [None, (3,), (2,), (2, 3)]


def first(o):
    return (o[0] if isinstance(o, (tuple, list)) else o).float().sigmoid()


@torch.no_grad()
def evaluate(models, loader, device):
    names = list(models)
    cnt = {f"single:{n}": [0, 0, 0] for n in names}
    cnt.update({"ens": [0, 0, 0], "ens_tta": [0, 0, 0], "single_tta_mean": [0, 0, 0]})
    st = {n: [0, 0, 0] for n in names}
    for b in loader:
        x = b["image"].to(device)
        y = b["label"].to(device)
        valid = ~b["no_data_mask"].to(device)
        pos, neg = (y == 1) & valid, (y != 1) & valid
        probs, probs_tta = {}, {}
        for n, m in models.items():
            acc = 0
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                for fl in FLIPS:
                    xi = x if fl is None else x.flip(fl)
                    o = first(m(xi)).squeeze(1)
                    o = o if fl is None else o.flip([d - 1 for d in fl])
                    if fl is None:
                        probs[n] = o
                    acc = acc + o
            probs_tta[n] = acc / len(FLIPS)

        def add(key, p):
            pr = p > 0.5
            c = cnt[key]
            c[0] += int((pr & pos).sum()); c[1] += int((pr & neg).sum()); c[2] += int((~pr & pos).sum())

        for n in names:
            add(f"single:{n}", probs[n])
            pr = probs_tta[n] > 0.5
            st[n][0] += int((pr & pos).sum()); st[n][1] += int((pr & neg).sum()); st[n][2] += int((~pr & pos).sum())
        add("ens", sum(probs.values()) / len(names))
        add("ens_tta", sum(probs_tta.values()) / len(names))
    out = {}
    for k, (tp, fp, fn) in cnt.items():
        if k == "single_tta_mean":
            continue
        out[k] = {"f1": 2 * tp / max(1, 2 * tp + fp + fn), "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}
    out["single_tta"] = {n: 2 * v[0] / max(1, 2 * v[0] + v[1] + v[2]) for n, v in st.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--configs_dir", default="configs_hpc")
    ap.add_argument("--config_name", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[2025, 2024, 2023])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(ROOT / a.configs_dir / f"{a.config_name}.yaml"))
    cfg["experiment"]["num_workers"] = a.workers
    split_seed = None
    loader = DataLoader(AirDataset(cfg, "test", "none"), batch_size=32, shuffle=False, num_workers=a.workers)
    groups = defaultdict(dict)
    for log in sorted(glob.glob(os.path.join(a.logs, "air*.log"))):
        t = open(log, errors="ignore").read()
        m = re.search(r"args=(\{.*\})", t)
        if not m or "Combined metrics" not in t:
            continue
        g = ast.literal_eval(m.group(1))
        if Path(g["config"]).stem != a.config_name or g.get("tag", "") not in a.tags or g["model"] not in a.models:
            continue
        tag = g["model"] + (f"-{g['aux']}" if g["model"] == "treemortseg" else "") + (f"-{g['tag']}" if g.get("tag") else "")
        exp = f"air{str(g['exp_id']).zfill(4)}_{tag}"
        ck = next((Path(d) / exp / f"{exp}_best.pth" for d in a.ckpts if (Path(d) / exp / f"{exp}_best.pth").exists()), None)
        if ck is not None and g["seed"] in a.seeds:
            groups[(g["tag"], g["model"])][g["seed"]] = (exp, ck, g)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for (tag, model), runs in groups.items():
        key = f"{a.config_name}|{tag}"
        if key in res or len(runs) < 2:
            continue
        models = {}
        for seed, (exp, ck, g) in sorted(runs.items()):
            net = AirInput(build(g["model"], 5), norm=g.get("norm", "none")).to(device)
            sd = {k[4:] if k.startswith("net.") else k: v for k, v in torch.load(ck, map_location=device).items()}
            (net if any(k.startswith("net.") for k in sd) else net.net).load_state_dict(sd)
            models[f"s{seed}"] = net.eval()
        r = evaluate(models, loader, device)
        singles = [v["f1"] for k, v in r.items() if k.startswith("single:")]
        r["single_mean"] = sum(singles) / len(singles)
        r["single_tta_mean"] = sum(r["single_tta"].values()) / len(r["single_tta"])
        res[key] = {"n_models": len(models), **r}
        print(key, "single %.1f | single+TTA %.1f | ens %.1f | ens+TTA %.1f" % (
            100 * r["single_mean"], 100 * r["single_tta_mean"], 100 * r["ens"]["f1"], 100 * r["ens_tta"]["f1"]), flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        del models
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
