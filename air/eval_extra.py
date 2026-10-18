"""Re-evaluate saved air/ checkpoints with threshold-free and per-site metrics.

For every finished run (parsed from its log) this computes, on the *test* split of the run's own protocol:
  AP            pixel-level average precision (ranking quality, independent of the 0.5 threshold)
  F1@0.5        as in the benchmark
  F1@val        F1 at the threshold that maximises F1 on the *validation* split (no test information)
  F1 oracle     best achievable threshold on test (diagnostic only; never a result)
and writes per-site tp/fp/fn (at the validation threshold) so that paired, site-level statistics can be computed.

    python air/eval_extra.py --logs /path/to/logs --out /path/to/out_dir [--only SUBSTR]
"""
import argparse
import ast
import glob
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

from air.dataset import AirDataset            # noqa: E402
from air.main import AirInput, build           # noqa: E402

BINS = 200


@torch.no_grad()
def site_hists(model, loader, device, n_sites):
    """[n_sites, 2, BINS] histograms of scores for positive / negative valid pixels, per source image (site)."""
    out = torch.zeros(n_sites, 2, BINS, dtype=torch.long, device=device)
    model.eval()
    for b in loader:
        x = b["image"].to(device)
        y = b["label"].to(device)
        valid = ~b["no_data_mask"].to(device)
        site = b["site"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            o = model(x)
        o = o[0] if isinstance(o, (tuple, list)) else o
        p = o.float().sigmoid().squeeze(1)
        bin_ = (p * BINS).long().clamp(0, BINS - 1)
        sid = site.view(-1, 1, 1).expand_as(bin_)
        for c, m in enumerate(((y == 1) & valid, (y != 1) & valid)):
            idx = (sid[m] * BINS + bin_[m])
            out[:, c] += torch.bincount(idx, minlength=n_sites * BINS).view(n_sites, BINS)
    return out.cpu().numpy()


def metrics(h, thr):
    cut = int(round(thr * BINS))
    tp, fn, fp = h[0, cut:].sum(), h[0, :cut].sum(), h[1, cut:].sum()
    return {"precision": float(tp / max(1, tp + fp)), "recall": float(tp / max(1, tp + fn)),
            "f1": float(2 * tp / max(1, 2 * tp + fp + fn)), "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def best_thr(h):
    f = [metrics(h, c / BINS)["f1"] for c in range(2, BINS - 1)]
    k = int(np.argmax(f))
    return (k + 2) / BINS, f[k]


def average_precision(h):
    pos, neg = h[0][::-1].cumsum(), h[1][::-1].cumsum()
    rec, prec = pos / max(1, h[0].sum()), pos / np.maximum(1, pos + neg)
    return float((prec * np.diff(np.concatenate([[0], rec]))).sum())


def parse_run(log_path):
    t = open(log_path, errors="ignore").read()
    m = re.search(r"args=(\{.*\})", t)
    if not m or "Combined metrics" not in t:
        return None
    return ast.literal_eval(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True, help="directories that contain <exp_name>/<exp_name>_best.pth")
    ap.add_argument("--configs_dir", default="configs_hpc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=0, help="debug only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for log in sorted(glob.glob(os.path.join(a.logs, "air*.log"))):
        args = parse_run(log)
        if args is None:
            continue
        tag = args["model"] + (f"-{args['aux']}" if args["model"] == "treemortseg" else "") + (f"-{args['tag']}" if args.get("tag") else "")
        exp_name = f"air{str(args['exp_id']).zfill(4)}_{tag}"
        if a.only and a.only not in exp_name:
            continue
        done = Path(a.out) / f"{exp_name}.json"
        if done.exists():
            continue
        ck = next((Path(d) / exp_name / f"{exp_name}_best.pth" for d in a.ckpts if (Path(d) / exp_name / f"{exp_name}_best.pth").exists()), None)
        if ck is None:
            print("no checkpoint", exp_name, flush=True)
            continue
        cfg = yaml.safe_load(open(ROOT / a.configs_dir / Path(args["config"]).name))
        cfg["experiment"]["seed"] = args.get("split_seed") or args["seed"]
        cfg["experiment"]["num_workers"] = a.workers
        loaders = {}
        for s in ("val", "test"):
            ds = AirDataset(cfg, s, "none")
            if a.max_samples:
                ds.base.paths, ds.site_ids = ds.base.paths[:a.max_samples], ds.site_ids[:a.max_samples]
            loaders[s] = (ds, DataLoader(ds, batch_size=64, shuffle=False, num_workers=a.workers))
        n_sites = 1 + max(max(ds.site_ids) for ds, _ in loaders.values())
        core = build(args["model"], cfg["model"].get("in_channels", 5), gate=args.get("gate", 0) > 0)
        model = AirInput(core, norm=args.get("norm", "none"), tta=False).to(device)
        sd = torch.load(ck, map_location=device)
        sd = {k[len("net."):] if k.startswith("net.") else k: v for k, v in sd.items()}   # strip MaskOnly prefix
        if any(k.startswith("net.") for k in sd):          # checkpoint saved from an AirInput-wrapped model
            model.load_state_dict(sd)
        else:                                              # older runs: raw network
            model.net.load_state_dict(sd)
        hv = site_hists(model, loaders["val"][1], device, n_sites).sum(0)
        ht = site_hists(model, loaders["test"][1], device, n_sites)
        thr_val, f1_val = best_thr(hv)
        tot = ht.sum(0)
        thr_or, f1_or = best_thr(tot)
        sites = np.flatnonzero(ht.sum((1, 2)) > 0)
        per_site = [{"site": int(s), **{k: v for k, v in metrics(ht[s], thr_val).items() if k in ("tp", "fp", "fn")}} for s in sites]
        res = {"exp_name": exp_name, "args": args, "val_thr": thr_val, "val_f1": f1_val,
               "ap": average_precision(tot), "val_ap": average_precision(hv), "at_0.5": metrics(tot, 0.5), "at_val_thr": metrics(tot, thr_val),
               "oracle": {"thr": thr_or, **metrics(tot, thr_or)}, "per_site": per_site}
        done.write_text(json.dumps(res))
        np.save(Path(a.out) / f"{exp_name}.site_hist.npy", ht.astype(np.int32))           # [site, {pos,neg}, BINS]
        np.save(Path(a.out) / f"{exp_name}.val_hist.npy", hv.astype(np.int32))
        print(exp_name, "AP=%.3f F1@.5=%.3f F1@val=%.3f oracle=%.3f" % (res["ap"], res["at_0.5"]["f1"], res["at_val_thr"]["f1"], f1_or), flush=True)


if __name__ == "__main__":
    main()
