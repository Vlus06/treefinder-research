"""Test-time style normalisation of FINISHED runs (no retraining, no labels used for the adaptation).

For every finished run of a given config (e.g. res_state_CO) the test tiles are evaluated with
  none        plain forward
  nearest     every tile is restyled with the band statistics of its nearest source-bank entry (full)
  nearest_h   same, half-way (lam = 0.5)
  avg4 / avg8 mean of the logits over the original and 3 / 7 random bank restyles (lam ~ U(0,1))
Metric: pooled dead-tree F1 / precision / recall at 0.5 over valid pixels (as in exps/evaluate.py).

    python air/tts_eval.py --logs LOGS --ckpts DIR... --configs_dir configs_hpc --config_name res_state_CO --out out.json
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
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from air.dataset import AirDataset                      # noqa: E402
from air.main import AirInput, build                    # noqa: E402
from air.spectral_aug import StatBank                   # noqa: E402

VARIANTS = ["none", "nearest", "nearest_h", "avg4", "avg8"]


def first(o):
    return (o[0] if isinstance(o, (tuple, list)) else o).float()


@torch.no_grad()
def evaluate(model, loader, bank, device):
    cnt = {v: [0, 0, 0] for v in VARIANTS}
    model.eval()
    for b in loader:
        x = b["image"].to(device)
        y = b["label"].to(device)
        valid = ~b["no_data_mask"].to(device)
        pos, neg = (y == 1) & valid, (y != 1) & valid
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            l0 = first(model(x))
            ln = first(model(bank.nearest(x, 1.0)))
            lh = first(model(bank.nearest(x, 0.5)))
            ls = [l0] + [first(model(bank(x))) for _ in range(7)]
        logits = {"none": l0, "nearest": ln, "nearest_h": lh,
                  "avg4": torch.stack(ls[:4]).mean(0), "avg8": torch.stack(ls).mean(0)}
        for k, l in logits.items():
            p = l.squeeze(1) > 0
            cnt[k][0] += int((p & pos).sum()); cnt[k][1] += int((p & neg).sum()); cnt[k][2] += int((~p & pos).sum())
    out = {}
    for k, (tp, fp, fn) in cnt.items():
        out[k] = {"f1": 2 * tp / max(1, 2 * tp + fp + fn), "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--configs_dir", default="configs_hpc")
    ap.add_argument("--config_name", default="res_state_CO")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tags", nargs="*", default=None, help="only runs whose tag is in this list")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    cfg = yaml.safe_load(open(ROOT / a.configs_dir / f"{a.config_name}.yaml"))
    cfg["experiment"]["num_workers"] = a.workers
    bank = StatBank(os.path.join(cfg["data"]["root_dir"], "tile_band_stats.csv"),
                    [os.path.basename(p) for p in AirDataset(cfg, "train", "none").base.paths], device)
    loader = DataLoader(AirDataset(cfg, "test", "none"), batch_size=64, shuffle=False, num_workers=a.workers)
    for log in sorted(glob.glob(os.path.join(a.logs, "air*.log"))):
        t = open(log, errors="ignore").read()
        m = re.search(r"args=(\{.*\})", t)
        if not m or "Combined metrics" not in t:
            continue
        g = ast.literal_eval(m.group(1))
        if Path(g["config"]).stem != a.config_name or g["model"] not in ("convnext_upernet",) or g.get("ctx") or g.get("gate") or g.get("rel") \
                or g.get("up", 1.0) != 1.0 or g.get("max_samples") or g.get("tag", "") == "smoke":
            continue
        if a.tags is not None and g.get("tag", "") not in a.tags:
            continue
        exp = f"air{str(g['exp_id']).zfill(4)}_{g['model']}" + (f"-{g['tag']}" if g.get("tag") else "")
        if exp in res:
            continue
        ck = next((Path(d) / exp / f"{exp}_best.pth" for d in a.ckpts if (Path(d) / exp / f"{exp}_best.pth").exists()), None)
        if ck is None:
            continue
        model = AirInput(build(g["model"], 5), norm=g.get("norm", "none")).to(device)
        sd = {k[4:] if k.startswith("net.") else k: v for k, v in torch.load(ck, map_location=device).items()}
        (model if any(k.startswith("net.") for k in sd) else model.net).load_state_dict(sd)
        r = evaluate(model, loader, bank, device)
        res[exp] = {"tag": g.get("tag", ""), "seed": g["seed"], "res": r}
        print(exp, g["seed"], " ".join("%s=%.1f" % (k, 100 * v["f1"]) for k, v in r.items()), flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
