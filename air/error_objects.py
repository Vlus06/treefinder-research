"""Object-level error anatomy of saved air/ checkpoints (test split of the run's own protocol).

For every run it thresholds the prediction at the *validation* threshold stored by eval_extra.py and reports

  ground-truth crowns (connected components, 8-connectivity)
    - by size bin: how many, share of dead pixels, share *detected* (>= 30 % of the crown predicted)
    - where the missed dead pixels are:  whole crowns that were missed  vs  the unpredicted rim of crowns that were found
  predicted blobs
    - share of false-positive pixels in blobs that touch no ground truth (isolated)  vs  blobs that overlap a crown
    - share of false-positive pixels within 3 px of a ground-truth crown (boundary spill)
  context
    - mean NDVI of missed vs detected crowns, share of crowns touching the tile border / no-data

    python air/error_objects.py --evals DIR_with_eval_extra_json --ckpts DIR... --configs_dir configs_hpc --out out.json --only SUBSTR
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from air.dataset import AirDataset            # noqa: E402
from air.main import AirInput, build           # noqa: E402

S8 = np.ones((3, 3), bool)
BINS = [(1, 16), (17, 64), (65, 256), (257, 10 ** 9)]
LABEL = ["<=16px", "17-64px", "65-256px", ">256px"]


def analyse(pred_all, gt_all, nod_all, ndvi_all, names=None, comp_rec=None, tile_rec=None):
    acc = {k: np.zeros(len(BINS)) for k in ("n", "px", "det", "px_det_total", "px_found", "ndvi_sum_miss", "n_miss", "ndvi_sum_det", "n_det")}
    fn_whole = fn_rim = 0
    fp_iso = fp_ovl = fp_near = fp_total = 0
    border = near_nod = ncomp = 0
    for ti, (pred, gt, nod, ndvi) in enumerate(zip(pred_all, gt_all, nod_all, ndvi_all)):
        fp_i0, fp_o0 = fp_iso, fp_ovl
        lg, ng = ndimage.label(gt, S8)
        lp, npred = ndimage.label(pred, S8)
        near_gt = ndimage.binary_dilation(gt, S8, iterations=3) if gt.any() else np.zeros_like(gt)
        nod_d = ndimage.binary_dilation(nod, S8, iterations=3) if nod.any() else nod
        if ng:
            area = np.bincount(lg.ravel(), minlength=ng + 1)[1:]
            cov = np.bincount(lg[pred & gt].ravel(), minlength=ng + 1)[1:]
            ndv = ndimage.mean(ndvi, lg, np.arange(1, ng + 1))
            sl = ndimage.find_objects(lg)
            for i in range(ng):
                b = next(j for j, (lo, hi) in enumerate(BINS) if lo <= area[i] <= hi)
                det = cov[i] / area[i] >= 0.3
                if comp_rec is not None:
                    comp_rec.append((names[ti], int(area[i]), float(cov[i] / area[i]), float(ndv[i])))
                acc["n"][b] += 1
                acc["px"][b] += area[i]
                acc["det"][b] += det
                if det:
                    acc["ndvi_sum_det"][b] += ndv[i]; acc["n_det"][b] += 1
                    fn_rim += area[i] - cov[i]
                else:
                    acc["ndvi_sum_miss"][b] += ndv[i]; acc["n_miss"][b] += 1
                    fn_whole += area[i] - cov[i]
                ys, xs = sl[i]
                border += int(ys.start == 0 or xs.start == 0 or ys.stop == gt.shape[0] or xs.stop == gt.shape[1])
                near_nod += int(nod_d[lg == i + 1].any()) if nod.any() else 0
                ncomp += 1
        if npred:
            parea = np.bincount(lp.ravel(), minlength=npred + 1)[1:]
            pin = np.bincount(lp[gt].ravel(), minlength=npred + 1)[1:]
            fpp = parea - pin
            iso = pin == 0
            fp_iso += fpp[iso].sum(); fp_ovl += fpp[~iso].sum(); fp_total += fpp.sum()
            fp_near += (pred & ~gt & near_gt).sum()
        if tile_rec is not None:
            tile_rec.append((names[ti], int(gt.sum()), int((pred & gt).sum()), int(fp_iso - fp_i0), int(fp_ovl - fp_o0)))
    return acc, dict(fn_whole=fn_whole, fn_rim=fn_rim, fp_iso=fp_iso, fp_ovl=fp_ovl, fp_near=fp_near, fp_total=fp_total,
                     border=border, near_nod=near_nod, ncomp=ncomp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--configs_dir", default="configs_hpc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="+", required=True, help="substrings of exp names (eval_extra json) to analyse")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--records", default="", help="prefix for per-component / per-tile CSVs")
    ap.add_argument("--dump", default="", help="directory for probability maps of selected tiles (miss / false alarm / good)")
    ap.add_argument("--n_dump", type=int, default=60)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for jf in sorted(Path(a.evals).glob("*.json")):
        ev = json.loads(jf.read_text())
        name = ev["exp_name"]
        if not any(s in name for s in a.only):
            continue
        args, thr = ev["args"], ev["val_thr"]
        ck = next((Path(d) / name / f"{name}_best.pth" for d in a.ckpts if (Path(d) / name / f"{name}_best.pth").exists()), None)
        if ck is None:
            continue
        cfg = yaml.safe_load(open(ROOT / a.configs_dir / Path(args["config"]).name))
        cfg["experiment"]["seed"] = args.get("split_seed") or args["seed"]
        cfg["experiment"]["num_workers"] = a.workers
        ds = AirDataset(cfg, "test", "none")
        if a.max_samples:
            ds.base.paths, ds.site_ids = ds.base.paths[:a.max_samples], ds.site_ids[:a.max_samples]
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=a.workers)
        model = AirInput(build(args["model"], 5), norm=args.get("norm", "none")).to(device)
        sd = {k[4:] if k.startswith("net.") else k: v for k, v in torch.load(ck, map_location=device).items()}
        (model if any(k.startswith("net.") for k in sd) else model.net).load_state_dict(sd)
        model.eval()
        P, G, N, V, PR = [], [], [], [], []
        with torch.no_grad():
            for b in loader:
                x = b["image"].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    o = model(x)
                o = o[0] if isinstance(o, (tuple, list)) else o
                valid = ~b["no_data_mask"]
                pr = o.float().sigmoid().squeeze(1).cpu()
                PR.append((pr * 255).round().to(torch.uint8).numpy())
                P.append(((pr >= thr) & valid).numpy())
                G.append(((b["label"] == 1) & valid).numpy()); N.append(b["no_data_mask"].numpy()); V.append(b["image"][:, 4].numpy())
        names = [os.path.basename(p) for p in ds.base.paths]
        comp_rec, tile_rec = [], []
        acc, oth = analyse(np.concatenate(P), np.concatenate(G), np.concatenate(N), np.concatenate(V), names, comp_rec, tile_rec)
        if a.records:
            import pandas as pd
            pd.DataFrame(comp_rec, columns=["tile", "area", "covered", "ndvi"]).assign(exp=name).to_csv(f"{a.records}.{name}.components.csv", index=False)
            pd.DataFrame(tile_rec, columns=["tile", "gt_px", "tp_px", "fp_isolated_px", "fp_overlap_px"]).assign(exp=name).to_csv(f"{a.records}.{name}.tiles.csv", index=False)
        if a.dump:
            os.makedirs(a.dump, exist_ok=True)
            tr = pd.DataFrame(tile_rec, columns=["tile", "gt_px", "tp_px", "fp_isolated_px", "fp_overlap_px"])
            cr = pd.DataFrame(comp_rec, columns=["tile", "area", "covered", "ndvi"])
            big_miss = cr[(cr.area > 256) & (cr.covered < 0.3)].groupby("tile").area.sum().sort_values(ascending=False)
            fa = tr.sort_values("fp_isolated_px", ascending=False)
            good = tr[(tr.gt_px > 100)].assign(f1=lambda d: 2 * d.tp_px / (2 * d.tp_px + d.fp_isolated_px + d.fp_overlap_px + (d.gt_px - d.tp_px)).clip(lower=1)).sort_values("f1", ascending=False)
            sel = {"miss": big_miss.index[:a.n_dump].tolist(), "falsealarm": fa.tile[:a.n_dump].tolist(), "good": good.tile[:a.n_dump].tolist()}
            idx = {n: k for k, n in enumerate(names)}
            PRa = np.concatenate(PR)
            out = {"thr": np.float32(thr)}
            for kind, lst in sel.items():
                ii = [idx[n] for n in lst]
                out[f"{kind}_names"] = np.array(lst)
                out[f"{kind}_prob"] = PRa[ii]
            np.savez_compressed(os.path.join(a.dump, f"{name}.npz"), **out)
        results[name] = {"args": args, "thr": thr, "acc": {k: v.tolist() for k, v in acc.items()}, "other": {k: float(v) for k, v in oth.items()}}
        print("done", name, flush=True)
        Path(a.out).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
