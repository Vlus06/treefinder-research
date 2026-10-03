"""Site-cluster bootstrap confidence interval for positive-class pixel F1.

Run: python research/bootstrap_sites.py research/reports/site_unet/seed2025/test_by_site.csv
"""

import argparse

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("per_site_csv")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    table = pd.read_csv(args.per_site_csv)
    values = table[["tp", "fp", "fn"]].to_numpy(dtype=np.int64)
    n = len(values)
    rng = np.random.default_rng(args.seed)
    scores = np.empty(args.replicates, dtype=float)
    for iteration in range(args.replicates):
        tp, fp, fn = values[rng.integers(0, n, size=n)].sum(axis=0)
        scores[iteration] = 2 * tp / max(1, 2 * tp + fp + fn)
    tp, fp, fn = values.sum(axis=0)
    point = 2 * tp / max(1, 2 * tp + fp + fn)
    low, high = np.quantile(scores, [0.025, 0.975])
    print(f"site-bootstrap F1: {point:.4f} (95% CI {low:.4f}-{high:.4f}; {n} sites)")


if __name__ == "__main__":
    main()
