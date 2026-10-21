"""AdaBN-style test-time adaptation of finished runs: re-estimate every BatchNorm layer's statistics on the (unlabeled) test-domain
images, then evaluate with the benchmark metric.  Labels are never used for the adaptation.

    python air/adabn_eval.py --logs LOGS --ckpts DIR... --configs_dir configs_hpc --config_name res_state_CO --out out.json
Reports, per finished run of that config: F1 / precision / recall at 0.5 before and after adaptation.
"""
import argparse
import ast
import glob
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from air.dataset import AirDataset                      # noqa: E402
from air.main import AirInput, build                    # noqa: E402


@torch.no_grad()
def pooled(model, loader, device):
    tp = fp = fn = 0
    model.eval()
    for b in loader:
        x = b["image"].to(device)
        y = b["label"].to(device)
        valid = ~b["no_data_mask"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            o = model(x)
        o = (o[0] if isinstance(o, (tuple, list)) else o).float().squeeze(1)
        p = o.sigmoid() > 0.5
        pos, neg = (y == 1) & valid, (y != 1) & valid
        tp += int((p & pos).sum()); fp += int((p & neg).sum()); fn += int((~p & pos).sum())
    return {"f1": 2 * tp / max(1, 2 * tp + fp + fn), "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}


@torch.no_grad()
def adapt_bn(model, loader, device):
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                                 # cumulative average over all batches
    model.eval()
    for m in bns:
        m.train()
    for b in loader:
        model(b["image"].to(device))
    model.eval()
    return len(bns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--configs_dir", default="configs_hpc")
    ap.add_argument("--config_name", default="res_state_CO")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    cfg = yaml.safe_load(open(ROOT / a.configs_dir / f"{a.config_name}.yaml"))
    cfg["experiment"]["num_workers"] = a.workers
    loader = DataLoader(AirDataset(cfg, "test", "none"), batch_size=64, shuffle=False, num_workers=a.workers)
    for log in sorted(glob.glob(os.path.join(a.logs, "air*.log"))):
        t = open(log, errors="ignore").read()
        m = re.search(r"args=(\{.*\})", t)
        if not m or "Combined metrics" not in t:
            continue
        g = ast.literal_eval(m.group(1))
        if Path(g["config"]).stem != a.config_name or g.get("ctx") or g.get("gate") or g.get("up", 1.0) != 1.0:
            continue
        tag = g["model"] + (f"-{g['aux']}" if g["model"] == "treemortseg" else "") + (f"-{g['tag']}" if g.get("tag") else "")
        exp = f"air{str(g['exp_id']).zfill(4)}_{tag}"
        if exp in res:
            continue
        ck = next((Path(d) / exp / f"{exp}_best.pth" for d in a.ckpts if (Path(d) / exp / f"{exp}_best.pth").exists()), None)
        if ck is None:
            print("no checkpoint", exp, flush=True)
            continue
        model = AirInput(build(g["model"], 5), norm=g.get("norm", "none")).to(device)
        sd = {k[4:] if k.startswith("net.") else k: v for k, v in torch.load(ck, map_location=device).items()}
        (model if any(k.startswith("net.") for k in sd) else model.net).load_state_dict(sd)
        before = pooled(model, loader, device)
        n = adapt_bn(model, loader, device)
        after = pooled(model, loader, device)
        res[exp] = {"model": g["model"], "aux": g.get("aux"), "seed": g["seed"], "tag": g.get("tag", ""), "bn_layers": n, "before": before, "after": after}
        print(exp, "F1 %.1f -> %.1f (BN layers: %d)" % (100 * before["f1"], 100 * after["f1"], n), flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
