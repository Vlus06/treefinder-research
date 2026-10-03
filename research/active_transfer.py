"""Site-level active transfer without looking at target-pool labels during selection.

Example: python research/active_transfer.py --direction west_to_east --budgets 5 10 20
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_baseline import ROOT, TileDataset, evaluate, masked_loss
from data_loader.utils import load_tile
from models.unet import UNet


def make_loader(frame: pd.DataFrame, batch_size: int, augment: bool = False, shuffle: bool = False):
    return DataLoader(
        TileDataset(frame, augment=augment),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def train_epochs(model, frame, device, epochs, batch_size, lr, validation=None, checkpoint=None):
    loader = make_loader(frame, batch_size, augment=True, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses = []
    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for images, labels, valid, _ in loader:
            images, labels, valid = images.to(device), labels.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(images), labels, valid)
            loss.backward()
            optimizer.step()
            total += loss.item()
        losses.append(total / len(loader))
        print(f"epoch {epoch + 1}/{epochs}: loss={losses[-1]:.5f}", flush=True)
        if validation is not None:
            val, _ = evaluate(model, validation, device)
            print(f"source validation loss={val['loss']:.5f}", flush=True)
            if val["loss"] < best_val:
                best_val = val["loss"]
                torch.save(model.state_dict(), checkpoint)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return losses


def score_pool(model, frame, device, max_tiles_per_site=4):
    """Compute label-free spectral features and candidate-focused entropy per site."""
    rows = []
    model.eval()
    with torch.no_grad():
        for site, group in frame.groupby("ImageRawPath", sort=True):
            # Deterministic tile subset keeps acquisition cost bounded per site.
            subset = group.sort_values("FileName").iloc[:max_tiles_per_site]
            tile_features, tile_uncertainty = [], []
            for tile in subset.itertuples():
                image, _, no_data = load_tile(str(ROOT / "datasets" / "tiles224_v3" / tile.FileName))
                valid = ~no_data
                if not valid.any():
                    continue
                values = image[:, valid]
                tile_features.append(np.concatenate([values.mean(1), values.std(1)]))
                tensor = torch.from_numpy(image[None]).float().to(device)
                p = model(tensor).sigmoid().cpu().numpy()[0, 0][valid]
                entropy = -(p * np.log(p + 1e-7) + (1 - p) * np.log(1 - p + 1e-7))
                k = max(1, round(0.01 * len(entropy)))
                tile_uncertainty.append(float(np.partition(entropy, -k)[-k:].mean()))
            if not tile_features:
                continue
            rows.append({
                "site": site,
                "uncertainty": float(np.mean(tile_uncertainty)),
                "feature": np.mean(tile_features, axis=0).tolist(),
                "candidate_tiles": len(group),
            })
    return rows


def select_sites(rows, strategy, max_budget):
    sites = [row["site"] for row in rows]
    if strategy == "random":
        return np.random.default_rng(2025).permutation(sites)[:max_budget].tolist()
    uncertainty = np.asarray([row["uncertainty"] for row in rows], dtype=float)
    if strategy == "uncertainty":
        return [sites[i] for i in np.argsort(-uncertainty)[:max_budget]]

    features = np.asarray([row["feature"] for row in rows], dtype=float)
    features = (features - features.mean(0)) / (features.std(0) + 1e-6)
    # Scores are normalized once, before any labels are revealed.
    u = (uncertainty - uncertainty.min()) / (np.ptp(uncertainty) + 1e-6)
    chosen = []
    center = features.mean(0)
    distance = np.linalg.norm(features - center, axis=1)
    for _ in range(min(max_budget, len(sites))):
        d = (distance - distance.min()) / (np.ptp(distance) + 1e-6)
        score = d if strategy == "diversity" else 0.5 * d + 0.5 * u
        if chosen:
            score[chosen] = -np.inf
        index = int(np.argmax(score))
        chosen.append(index)
        distance = np.minimum(distance, np.linalg.norm(features - features[index], axis=1))
    return [sites[i] for i in chosen]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["west_to_east", "east_to_west"], default="west_to_east")
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--strategies", nargs="+", choices=["random", "uncertainty", "diversity", "hybrid"], default=["random", "uncertainty", "diversity", "hybrid"])
    parser.add_argument("--source-epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-score-tiles", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert min(args.budgets) > 0
    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(ROOT / "research" / "splits" / f"{args.direction}_seed2025.csv")
    source_train = frame[frame.split == "source_train"]
    source_val = frame[frame.split == "source_val"]
    target_pool = frame[frame.split == "target_pool"]
    target_test = frame[frame.split == "target_test"]
    if args.smoke:
        source_train = source_train.sample(n=min(64, len(source_train)), random_state=2025)
        source_val = source_val.sample(n=min(16, len(source_val)), random_state=2025)
        pool_sites = sorted(target_pool.ImageRawPath.unique())[:max(args.budgets) + 2]
        target_pool = target_pool[target_pool.ImageRawPath.isin(pool_sites)]
        target_test = target_test.sample(n=min(32, len(target_test)), random_state=2025)

    output = ROOT / "research" / "reports" / ("smoke_active" if args.smoke else "active_transfer") / args.direction
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / "source_model.pt"
    base = UNet(5, 1, [16, 32, 64]).to(device)
    source_val_loader = make_loader(source_val, args.batch_size)
    if source_path.exists() and not args.smoke:
        base.load_state_dict(torch.load(source_path, map_location=device, weights_only=True))
    else:
        train_epochs(base, source_train, device, 1 if args.smoke else args.source_epochs, args.batch_size, 1e-3, source_val_loader, source_path)
    source_val_metrics, _ = evaluate(base, source_val_loader, device)
    test_loader = make_loader(target_test, args.batch_size)
    source_metrics, _ = evaluate(base, test_loader, device)
    print("zero_shot", json.dumps(source_metrics), flush=True)

    score_file = output / "pool_scores.json"
    if score_file.exists() and not args.smoke:
        rows = json.loads(score_file.read_text(encoding="utf-8"))
    else:
        rows = score_pool(base, target_pool, device, args.max_score_tiles)
        score_file.write_text(json.dumps(rows), encoding="utf-8")
    if len(rows) < max(args.budgets):
        raise ValueError(f"Only {len(rows)} scorable sites for maximum budget {max(args.budgets)}")

    results = [{"strategy": "zero_shot", "budget_sites": 0, "test": source_metrics}]
    for strategy in args.strategies:
        order = select_sites(rows, strategy, max(args.budgets))
        for budget in sorted(set(args.budgets)):
            chosen = order[:budget]
            labeled = target_pool[target_pool.ImageRawPath.isin(chosen)]
            model = UNet(5, 1, [16, 32, 64]).to(device)
            model.load_state_dict(torch.load(source_path, map_location=device, weights_only=True))
            losses = train_epochs(model, labeled, device, 1 if args.smoke else args.finetune_epochs, args.batch_size, 1e-4)
            metrics, _ = evaluate(model, test_loader, device)
            result = {
                "strategy": strategy, "budget_sites": budget,
                "labeled_tiles": len(labeled), "selected_sites": chosen,
                "finetune_loss": losses, "test": metrics,
            }
            results.append(result)
            (output / "results.json").write_text(json.dumps({
                "smoke_only": args.smoke, "direction": args.direction,
                "source_train_tiles": len(source_train), "source_val_tiles": len(source_val),
                "source_val_metrics": source_val_metrics,
                "target_pool_tiles": len(target_pool), "target_test_tiles": len(target_test),
                "results": results,
            }, indent=2), encoding="utf-8")
            print(strategy, budget, json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
