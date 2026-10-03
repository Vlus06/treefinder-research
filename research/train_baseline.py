"""Small, reproducible U-Net baseline for the site-disjoint split.

Run a pipeline check: python research/train_baseline.py --smoke
Run a full baseline:  python research/train_baseline.py --epochs 15
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_loader.utils import load_tile  # noqa: E402
from models.unet import UNet  # noqa: E402


class TileDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, augment: bool):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image, label, no_data = load_tile(str(ROOT / "datasets" / "tiles224_v3" / row.FileName))
        if self.augment:
            if random.random() < 0.5:
                image, label, no_data = image[:, :, ::-1], label[:, ::-1], no_data[:, ::-1]
            if random.random() < 0.5:
                image, label, no_data = image[:, ::-1, :], label[::-1, :], no_data[::-1, :]
            k = random.randrange(4)
            image, label, no_data = np.rot90(image, k, (1, 2)), np.rot90(label, k), np.rot90(no_data, k)
        return (
            torch.from_numpy(np.ascontiguousarray(image)).float(),
            torch.from_numpy(np.ascontiguousarray(label[None])).float(),
            torch.from_numpy(np.ascontiguousarray(~no_data[None])).float(),
            row.ImageRawPath,
        )


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1)
    probs = logits.sigmoid() * valid
    targets = labels * valid
    dice = 1 - (2 * (probs * targets).sum() + 1) / (probs.sum() + targets.sum() + 1)
    return bce + dice


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    loss_sum = 0.0
    totals = {"tp": 0, "fp": 0, "fn": 0}
    by_site = {}
    with torch.no_grad():
        for images, labels, valid, paths in loader:
            images, labels, valid = images.to(device), labels.to(device), valid.to(device)
            logits = model(images)
            loss_sum += masked_loss(logits, labels, valid).item()
            pred = (logits.sigmoid() >= threshold) & valid.bool()
            truth = labels.bool() & valid.bool()
            for i, path in enumerate(paths):
                row = by_site.setdefault(path, {"tp": 0, "fp": 0, "fn": 0})
                row["tp"] += int((pred[i] & truth[i]).sum().item())
                row["fp"] += int((pred[i] & ~truth[i] & valid[i].bool()).sum().item())
                row["fn"] += int((~pred[i] & truth[i]).sum().item())
    for row in by_site.values():
        for key in totals:
            totals[key] += row[key]
    tp, fp, fn = (totals[key] for key in ("tp", "fp", "fn"))
    metrics = {
        "loss": loss_sum / len(loader),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "iou": tp / max(1, tp + fp + fn),
        **totals,
    }
    return metrics, by_site


def calibrate_threshold(model, loader, device, bins=100):
    """Choose a threshold on labeled validation data, never on target test data."""
    model.eval()
    positive = np.zeros(bins, dtype=np.int64)
    negative = np.zeros(bins, dtype=np.int64)
    with torch.no_grad():
        for images, labels, valid, _ in loader:
            scores = model(images.to(device)).sigmoid()
            truth = labels.to(device).bool()
            mask = valid.to(device).bool()
            positive += torch.histc(scores[mask & truth], bins=bins, min=0, max=1).cpu().numpy().astype(np.int64)
            negative += torch.histc(scores[mask & ~truth], bins=bins, min=0, max=1).cpu().numpy().astype(np.int64)
    tp = np.cumsum(positive[::-1])
    fp = np.cumsum(negative[::-1])
    fn = positive.sum() - tp
    f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    # Restrict away from 0 and 1 to avoid degenerate thresholds.
    candidate = np.arange(4, bins - 5)
    best = int(candidate[np.argmax(f1[candidate])])
    return {"threshold": (bins - 1 - best) / bins, "validation_f1": float(f1[best])}


def smoke_subset(frame: pd.DataFrame, split: str, seed: int) -> pd.DataFrame:
    # This subset only checks whether training and evaluation run end to end.
    rng = np.random.default_rng(seed)
    per_class = 64 if split == "train" else 16
    return pd.concat([
        group.sample(n=min(per_class, len(group)), random_state=int(rng.integers(2**31)))
        for _, group in frame.groupby(frame.LabelSize.gt(0))
    ]).sort_values("FileName")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Small pipeline check; metrics are not research results")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    assert args.epochs > 0 and args.batch_size > 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = pd.read_csv(ROOT / "research" / "splits" / f"site_seed{args.seed}.csv")
    loaders = {}
    for split in ("train", "val", "test"):
        frame = manifest[manifest.split == split]
        if args.smoke:
            frame = smoke_subset(frame, split, args.seed)
        loaders[split] = DataLoader(
            TileDataset(frame, augment=(split == "train")),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )

    model = UNet(in_channels=5, num_classes=1, features=[16, 32, 64]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    output = ROOT / "research" / "reports" / ("smoke_unet" if args.smoke else "site_unet") / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
        model.train()
        train_loss = 0.0
        for images, labels, valid, _ in loaders["train"]:
            images, labels, valid = images.to(device), labels.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(images), labels, valid)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        val, _ = evaluate(model, loaders["val"], device)
        row = {"epoch": epoch, "train_loss": train_loss / len(loaders["train"]), "val": val}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["loss"] < best:
            best = val["loss"]
            torch.save(model.state_dict(), output / "best.pt")

    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True))
    test, by_site = evaluate(model, loaders["test"], device)
    calibrated = calibrate_threshold(model, loaders["val"], device)
    calibrated_test, _ = evaluate(model, loaders["test"], device, calibrated["threshold"])
    report = {
        "smoke_only": args.smoke,
        "seed": args.seed,
        "epochs": 1 if args.smoke else args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": str(device),
        "model": "UNet features=[16,32,64], RGB+NIR+NDVI",
        "split": f"site_seed{args.seed}",
        "counts": {s: len(loader.dataset) for s, loader in loaders.items()},
        "history": history,
        "test": test,
        "source_validation_threshold": calibrated,
        "test_at_source_validation_threshold": calibrated_test,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame.from_dict(by_site, orient="index").rename_axis("ImageRawPath").to_csv(output / "test_by_site.csv")
    print(json.dumps({"output": str(output), "test": test, "smoke_only": args.smoke}), flush=True)


if __name__ == "__main__":
    main()
