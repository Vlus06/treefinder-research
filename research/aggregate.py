"""Aggregate research/reports/runs/*/metrics.json into a table (mean +- std over seeds) with site-bootstrap CIs.

    python research/aggregate.py [--glob 'w2e_*'] [--csv out.csv]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parent / "reports" / "runs"


def site_bootstrap(csv: Path, n=1000, seed=0):
    s = pd.read_csv(csv)
    rng = np.random.default_rng(seed)
    tp, fp, fn = s.tp.values, s.fp.values, s.fn.values
    f = []
    for _ in range(n):
        i = rng.integers(0, len(s), len(s))
        f.append(2 * tp[i].sum() / max(1, 2 * tp[i].sum() + fp[i].sum() + fn[i].sum()))
    return np.percentile(f, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    rows = []
    for d in sorted(RUNS.glob(a.glob)):
        m = d / "metrics.json"
        if not m.exists():
            continue
        r = json.loads(m.read_text())
        name = d.name
        mt = re.match(r"(site|tile|w2e|e2w)_(.+?)_s(\d+)_?(.*)", name)
        protocol, model, seed, tag = mt.groups()
        t = r["test"]
        ta = r.get("test_adabn")
        lo, hi = site_bootstrap(d / "test_by_site.csv")
        rows.append({"protocol": protocol, "model": model, "tag": tag, "seed": int(seed), "params_M": r["params_M"],
                     "val_f1": r["val_f1"], "f1@0.5": t["at_0.5"]["f1"], "f1@val": t["at_val_thr"]["f1"],
                     "P@val": t["at_val_thr"]["precision"], "R@val": t["at_val_thr"]["recall"],
                     "IoU@val": t["at_val_thr"]["iou"], "AP": t["ap"], "f1_oracle": t["oracle_thr"]["f1"],
                     "f1_adabn": ta["at_val_thr"]["f1"] if ta else np.nan, "obj_f1": t.get("objects_at_val_thr", {}).get("obj_f1", np.nan),
                     "ci_lo": lo, "ci_hi": hi, "min": r["minutes"]})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no runs")
        return
    g = df.groupby(["protocol", "model", "tag"])
    summary = g.agg(n=("seed", "size"), params_M=("params_M", "first"),
                    f1_val=("f1@val", "mean"), f1_val_sd=("f1@val", "std"), f1_05=("f1@0.5", "mean"),
                    P=("P@val", "mean"), R=("R@val", "mean"), IoU=("IoU@val", "mean"),
                    f1_adabn=("f1_adabn", "mean"), obj_f1=("obj_f1", "mean"), AP=("AP", "mean"), AP_sd=("AP", "std"), oracle=("f1_oracle", "mean"), minutes=("min", "mean")).round(3)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(summary.to_string())
    if a.csv:
        df.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()
