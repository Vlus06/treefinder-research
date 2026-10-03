"""Inspect active-transfer selection yield from a Kaggle results ZIP.

Run: python research/analyze_selection.py /path/to/treefinder_results.zip
"""

import argparse
import json
import zipfile

import pandas as pd

from train_baseline import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_zip")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--mode", default="source_replay_tiles4", help="Result suffix, e.g. source_replay_tiles4")
    args = parser.parse_args()
    metadata = pd.read_csv(ROOT / "datasets" / "tile_info224_v3_aux.csv")
    with zipfile.ZipFile(args.results_zip) as archive:
        for direction in ("west_to_east", "east_to_west"):
            base = f"research/reports/active_transfer/{direction}"
            new_prefix = f"{base}/seed{args.seed}"
            new_result = f"results_{args.mode}.json"
            prefix = new_prefix if f"{new_prefix}/{new_result}" in archive.namelist() else base
            result_name = new_result if prefix == new_prefix else "results.json"
            result = json.loads(archive.read(f"{prefix}/{result_name}"))
            score_rows = json.loads(archive.read(f"{prefix}/pool_scores.json"))
            scores = pd.DataFrame(score_rows).set_index("site")
            print("\n", direction, "pool sites", len(scores), "\n")
            summary = []
            for row in result["results"]:
                sites = row.get("selected_sites", [])
                labeled = metadata[metadata.ImageRawPath.isin(sites)]
                summary.append({
                    "strategy": row["strategy"],
                    "budget_sites": row["budget_sites"],
                    "labeled_tiles": row.get("labeled_tiles", 0),
                    "labeled_hectares": row.get("labeled_valid_hectares"),
                    "positive_sites": int(labeled.groupby("ImageRawPath").LabelSize.sum().gt(0).sum()) if sites else 0,
                    "positive_tiles": int(labeled.LabelSize.gt(0).sum()),
                    "positive_pixels": int(labeled.LabelSize.sum()),
                    "mean_uncertainty": float(scores.loc[sites, "uncertainty"].mean()) if sites else None,
                    "test_f1": row["test"]["f1"],
                    "calibrated_test_f1": row.get("test_at_source_validation_threshold", {}).get("f1"),
                    "test_precision": row["test"]["precision"],
                    "test_recall": row["test"]["recall"],
                })
            table = pd.DataFrame(summary)
            print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
