"""Unified TreeFinder trainer for site-disjoint and cross-region protocols.

Reads the arrays written by research/pack_dataset.py (x.npy, y.npy) and a split manifest from
research/splits. All preprocessing and augmentation run on the GPU, so one run is a few minutes.

    python research/trainer.py --protocol site --model unet_r34 --epochs 30
    python research/trainer.py --protocol w2e --model segformer_b2 --norm tile --photo-aug --adabn

Protocols
  site : train/val/test, site-disjoint random split          (manifest site_seed{S}.csv)
  w2e  : source_train/source_val -> target_test (West->East) (manifest west_to_east_seed{S}.csv)
  e2w  : source_train/source_val -> target_test (East->West) (manifest east_to_west_seed{S}.csv)
Cross-region models never see target_test labels; target_pool images (without labels) are only used
by --adabn. Checkpoints are selected on *source* validation data.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BINS = 200


# ----------------------------------------------------------------------------- data
class Arrays:
    def __init__(self, data_dir: Path, in_memory: bool):
        mm = None if in_memory else "r"
        self.x = np.load(data_dir / "x.npy", mmap_mode=mm)
        self.y = np.load(data_dir / "y.npy", mmap_mode=mm)

    def batch(self, idx: np.ndarray, device):
        idx = np.sort(idx)
        x = torch.from_numpy(np.ascontiguousarray(self.x[idx])).to(device, non_blocking=True)
        y = torch.from_numpy(np.ascontiguousarray(self.y[idx])).to(device, non_blocking=True)
        return x, y


class Prep:
    """uint8 RGBN + label -> float 5-channel input (RGB, NIR, NDVI), validity mask and target."""

    def __init__(self, norm: str, mean: torch.Tensor | None, std: torch.Tensor | None):
        self.norm, self.mean, self.std = norm, mean, std

    @staticmethod
    def make5(x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        ndvi = (x[:, 3] - x[:, 0]) / (x[:, 3] + x[:, 0] + 1e-6)
        return torch.cat([x, ((ndvi.clamp(-1, 1) + 1) / 2).unsqueeze(1)], 1)

    def __call__(self, x_u8, y_u8, augment=False, photo=False):
        valid = (y_u8 != 255).unsqueeze(1).float()
        target = (y_u8 == 1).unsqueeze(1).float()
        x = x_u8.float() / 255.0
        if augment:
            b = x.shape[0]
            if photo:  # per-tile, per-band gain / bias / gamma: simulates acquisition-year shifts
                g = torch.empty(b, 4, 1, 1, device=x.device).uniform_(0.75, 1.25)
                o = torch.empty(b, 4, 1, 1, device=x.device).uniform_(-0.08, 0.08)
                gm = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.8, 1.25)
                x = ((x * g + o).clamp(0, 1)) ** gm
            fh = torch.rand(b, device=x.device) < 0.5
            fv = torch.rand(b, device=x.device) < 0.5
            x = torch.where(fh[:, None, None, None], x.flip(3), x)
            target = torch.where(fh[:, None, None, None], target.flip(3), target)
            valid = torch.where(fh[:, None, None, None], valid.flip(3), valid)
            x = torch.where(fv[:, None, None, None], x.flip(2), x)
            target = torch.where(fv[:, None, None, None], target.flip(2), target)
            valid = torch.where(fv[:, None, None, None], valid.flip(2), valid)
            k = int(torch.randint(0, 4, (1,)))
            x, target, valid = (torch.rot90(t, k, (2, 3)) for t in (x, target, valid))
        ndvi = (x[:, 3] - x[:, 0]) / (x[:, 3] + x[:, 0] + 1e-6)
        x = torch.cat([x, ((ndvi.clamp(-1, 1) + 1) / 2).unsqueeze(1)], 1)
        if self.norm == "tile":  # per-tile z-score over valid pixels (radiometric invariance)
            w = valid.expand_as(x)
            n = w.sum((2, 3), keepdim=True).clamp_min(1)
            mu = (x * w).sum((2, 3), keepdim=True) / n
            var = (((x - mu) ** 2) * w).sum((2, 3), keepdim=True) / n
            x = (x - mu) / (var.sqrt() + 1e-3) * valid
        elif self.norm == "both":  # absolute (global) and relative (per-tile) colour side by side
            w = valid.expand_as(x)
            n = w.sum((2, 3), keepdim=True).clamp_min(1)
            mu = (x * w).sum((2, 3), keepdim=True) / n
            var = (((x - mu) ** 2) * w).sum((2, 3), keepdim=True) / n
            rel = (x - mu) / (var.sqrt() + 1e-3) * valid
            x = torch.cat([(x - self.mean) / self.std, rel], 1)
        else:
            x = (x - self.mean) / self.std
        return x, target, valid


# ----------------------------------------------------------------------------- models
def adapt_first_conv(conv: nn.Conv2d, in_ch: int) -> nn.Conv2d:
    new = nn.Conv2d(in_ch, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight[:, :3].copy_(conv.weight)
        new.weight[:, 3:].copy_(conv.weight.mean(1, keepdim=True).repeat(1, in_ch - 3, 1, 1))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class SegFormerWrap(nn.Module):
    def __init__(self, name: str, in_ch: int, classes: int = 1):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation
        self.net = SegformerForSemanticSegmentation.from_pretrained(name, num_labels=classes, ignore_mismatched_sizes=True)
        pe = self.net.segformer.encoder.patch_embeddings[0]
        pe.proj = adapt_first_conv(pe.proj, in_ch)

    def forward(self, x):
        return F.interpolate(self.net(pixel_values=x).logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(name: str, in_ch: int = 5, classes: int = 1) -> nn.Module:
    if name.startswith("official_"):  # the benchmark authors' own model definitions (models/)
        import yaml
        from models import get_model
        if name == "official_mask2former_rgb":  # pre-274e112 behaviour: RGB only
            from models.mask2former_rgb import build_mask2former as build_rgb
            return build_rgb(yaml.safe_load(open(ROOT / "configs" / "res_state_WE.yaml"))["model"])
        cfg = yaml.safe_load(open(ROOT / "configs" / "res_state_WE.yaml"))["model"]
        cfg["name"] = {"official_unet": "unet", "official_deeplab": "deeplabv3", "official_vit": "vitseg",
                       "official_segformer": "segformer", "official_mask2former": "mask2former",
                       # what scripts/run_*.sh actually run: RGB-only SegFormer / ViT (x[:, :3])
                       "official_segformer3": "segformer_large3", "official_vit3": "vitseg3",
                       "official_dofa": "dofa"}[name]
        return get_model(cfg)
    import segmentation_models_pytorch as smp
    if name == "unet_small":  # the repository's small from-scratch baseline
        from models.unet import UNet
        return UNet(in_ch, classes, [16, 32, 64])
    if name == "unet_r34":
        return smp.Unet("resnet34", encoder_weights="imagenet", in_channels=in_ch, classes=classes)
    if name == "unetpp_r34":
        return smp.UnetPlusPlus("resnet34", encoder_weights="imagenet", in_channels=in_ch, classes=classes)
    if name == "fpn_r34":
        return smp.FPN("resnet34", encoder_weights="imagenet", in_channels=in_ch, classes=classes)
    if name == "dlv3p_r50":
        return smp.DeepLabV3Plus("resnet50", encoder_weights="imagenet", in_channels=in_ch, classes=classes)
    if name == "segformer_b2":
        return SegFormerWrap("nvidia/segformer-b2-finetuned-ade-512-512", in_ch, classes)
    if name == "segformer_b0":
        return SegFormerWrap("nvidia/segformer-b0-finetuned-ade-512-512", in_ch, classes)
    raise ValueError(name)


class Upscaled(nn.Module):
    """Run the network at `factor`x input resolution; dead crowns are only ~8-16 px wide at 0.6 m."""

    def __init__(self, net: nn.Module, factor: int):
        super().__init__()
        self.net, self.factor = net, factor

    def forward(self, x):
        if self.factor == 1:
            return self.net(x)
        h, w = x.shape[-2:]
        out = self.net(F.interpolate(x, scale_factor=self.factor, mode="bilinear", align_corners=False))
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)


# ----------------------------------------------------------------------------- loss & metrics
def seg_loss(logits, target, valid, kind: str):
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum() / valid.sum().clamp_min(1)
    p = logits.sigmoid() * valid
    t = target * valid
    tp, fp, fn = (p * t).sum(), (p * (1 - t)).sum(), ((1 - p) * t).sum()
    if kind == "bce_dice":
        return bce + 1 - (2 * tp + 1) / (2 * tp + fp + fn + 1)
    if kind == "bce_ftversky":  # recall-weighted focal Tversky
        tv = (tp + 1) / (tp + 0.3 * fp + 0.7 * fn + 1)
        return bce + (1 - tv) ** 0.75
    raise ValueError(kind)


def structure_targets(target, valid, d=6):
    """GPU structure targets from the binary mask: truncated inside/outside distance (Chebyshev, signed, in [-1,1])
    and a one-pixel boundary map. Auxiliary supervision in the spirit of TreeMort-3T-UNet / TreeMortSeg."""
    inside = target.clone()
    outside = target.clone()
    dist = target.clone()
    for _ in range(d):
        inside = -F.max_pool2d(-inside, 3, 1, 1)
        outside = F.max_pool2d(outside, 3, 1, 1)
        dist = dist + inside
        dist = dist - (1 - outside)
    sd = (dist / d).clamp(-1, 1)
    boundary = (F.max_pool2d(target, 3, 1, 1) - (-F.max_pool2d(-target, 3, 1, 1))).clamp(0, 1)
    return sd, boundary


def aux_loss(out, target, valid):
    sd, bd = structure_targets(target, valid)
    reg = (F.smooth_l1_loss(torch.tanh(out[:, 1:2]), sd, reduction="none") * valid).sum() / valid.sum().clamp_min(1)
    bce = (F.binary_cross_entropy_with_logits(out[:, 2:3], bd, reduction="none", pos_weight=torch.tensor(5.0, device=out.device)) * valid).sum() / valid.sum().clamp_min(1)
    return 2.0 * reg + 0.5 * bce


def hist_scores(prob, target, valid, n_tiles):
    """Per-tile score histograms of positive and negative valid pixels, [n_tiles, 2, BINS]."""
    b = (prob * BINS).long().clamp(0, BINS - 1)
    out = torch.zeros(n_tiles, 2, BINS, device=prob.device, dtype=torch.long)
    tile = torch.arange(n_tiles, device=prob.device).view(-1, 1, 1, 1).expand_as(b)
    pos = (target > 0.5) & (valid > 0.5)
    neg = (target <= 0.5) & (valid > 0.5)
    for c, m in enumerate((pos, neg)):
        out[:, c] = torch.bincount((tile[m] * BINS + b[m]), minlength=n_tiles * BINS).view(n_tiles, BINS)
    return out


def metrics_from_hist(h: np.ndarray, thr: float):
    """h: [..., 2, BINS] summed over tiles. Predict positive when score >= thr."""
    cut = int(round(thr * BINS))
    tp = h[0, cut:].sum()
    fn = h[0, :cut].sum()
    fp = h[1, cut:].sum()
    return {
        "precision": float(tp / max(1, tp + fp)), "recall": float(tp / max(1, tp + fn)),
        "f1": float(2 * tp / max(1, 2 * tp + fp + fn)), "iou": float(tp / max(1, tp + fp + fn)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
    }


def official_metrics(h: np.ndarray, thr: float = 0.5):
    """Metrics exactly as exps/evaluate.py: per-class P/R/F1/IoU from pooled counts, 'All' = mean of both classes."""
    cut = int(round(thr * BINS))
    tp1, fn1, fp1, tn1 = h[0, cut:].sum(), h[0, :cut].sum(), h[1, cut:].sum(), h[1, :cut].sum()
    def prf(tp, fp, fn):
        p, r = tp / (tp + fp + 1e-6), tp / (tp + fn + 1e-6)
        return p, r, 2 * p * r / (p + r + 1e-6), tp / (tp + fp + fn + 1e-6)
    d, b = prf(tp1, fp1, fn1), prf(tn1, fn1, fp1)
    return {"dead_precision": d[0], "dead_recall": d[1], "dead_f1": d[2], "dead_iou": d[3],
            "all_precision": (d[0] + b[0]) / 2, "all_recall": (d[1] + b[1]) / 2, "all_f1": (d[2] + b[2]) / 2,
            "all_iou": (d[3] + b[3]) / 2, "accuracy": float((tp1 + tn1) / max(1, tp1 + tn1 + fp1 + fn1))}


def best_threshold(h: np.ndarray):
    f1s = [metrics_from_hist(h, c / BINS)["f1"] for c in range(2, BINS - 1)]
    k = int(np.argmax(f1s))
    return (k + 2) / BINS, f1s[k]


def average_precision(h: np.ndarray):
    """Pixel-level average precision from score histograms (threshold-free ranking quality)."""
    pos = h[0][::-1].cumsum()
    neg = h[1][::-1].cumsum()
    tot = max(1, h[0].sum())
    recall = pos / tot
    precision = pos / np.maximum(1, pos + neg)
    d = np.diff(np.concatenate([[0], recall]))
    return float((precision * d).sum())


@torch.no_grad()
def predict_hist(model, arrays, idx, prep, device, bs, tta=False):
    model.eval()
    hists = []
    for s in range(0, len(idx), bs):
        xb, yb = arrays.batch(idx[s:s + bs], device)
        x, t, v = prep(xb, yb)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            p = model(x)[:, :1].float().sigmoid()
            if tta:
                p = p + model(x.flip(3))[:, :1].float().sigmoid().flip(3)
                p = p + model(x.flip(2))[:, :1].float().sigmoid().flip(2)
                p = p / 3
        hists.append(hist_scores(p, t, v, len(x)).cpu())
    return torch.cat(hists).numpy()  # [tiles(sorted idx order), 2, BINS]


@torch.no_grad()
def val_loss(model, arrays, idx, prep, device, bs, kind):
    model.eval()
    tot, n = 0.0, 0
    idx = np.sort(idx)
    for s in range(0, len(idx), bs):
        xb, yb = arrays.batch(idx[s:s + bs], device)
        x, t, v = prep(xb, yb)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x)
        tot += seg_loss(out.float()[:, :1], t, v, kind).item()
        n += 1
    return tot / max(1, n)


@torch.no_grad()
def adabn(model, arrays, idx, prep, device, bs):
    """Label-free adaptation: re-estimate BatchNorm statistics on unlabeled target images."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return False
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for s in range(0, len(idx), bs):
        xb, yb = arrays.batch(idx[s:s + bs], device)
        x, _, _ = prep(xb, yb)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            model(x)
    return True


@torch.no_grad()
def object_metrics(model, arrays, idx, prep, device, bs, thr, min_cover=0.3):
    """Object-level detection scores. A reference crown is detected when >= min_cover of its pixels are predicted
    dead; a predicted blob is correct when >= min_cover of its pixels are reference dead. Pixel F1 is overly harsh for
    scattered crowns whose boundaries are ambiguous."""
    from scipy import ndimage
    model.eval()
    st = np.ones((3, 3))
    gt_n = gt_hit = pr_n = pr_ok = 0
    idx = np.sort(idx)
    for s0 in range(0, len(idx), bs):
        xb, yb = arrays.batch(idx[s0:s0 + bs], device)
        x, t, v = prep(xb, yb)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            p = model(x)[:, :1].float().sigmoid()
        pred = ((p >= thr) & (v > 0.5)).squeeze(1).cpu().numpy()
        gt = ((t > 0.5) & (v > 0.5)).squeeze(1).cpu().numpy()
        for g, q in zip(gt, pred):
            lg, ng = ndimage.label(g, st)
            lq, nq = ndimage.label(q, st)
            if ng:
                area = np.bincount(lg.ravel(), minlength=ng + 1)[1:]
                cov = np.bincount(lg[q].ravel(), minlength=ng + 1)[1:]
                gt_n += ng
                gt_hit += int((cov / area >= min_cover).sum())
            if nq:
                area = np.bincount(lq.ravel(), minlength=nq + 1)[1:]
                cov = np.bincount(lq[g].ravel(), minlength=nq + 1)[1:]
                pr_n += nq
                pr_ok += int((cov / area >= min_cover).sum())
    rec, prec = gt_hit / max(1, gt_n), pr_ok / max(1, pr_n)
    return {"obj_recall": rec, "obj_precision": prec, "obj_f1": 2 * rec * prec / max(1e-9, rec + prec),
            "n_gt": gt_n, "n_pred": pr_n}


# ----------------------------------------------------------------------------- main
def load_manifest(protocol: str, seed: int):
    if protocol == "tile":  # the official loader's protocol: shuffle individual tiles (sites leak across splits)
        df = pd.read_csv(ROOT / "research" / "splits" / f"site_seed{seed}.csv")
        tiles = df.FileName.tolist()
        np.random.RandomState(seed).shuffle(tiles)
        n_test, n_tv = int(0.2 * len(tiles)), int(0.8 * len(tiles))
        n_val = int(0.1 * n_tv)
        pos = {f: i for i, f in enumerate(df.FileName)}
        order = np.array([pos[f] for f in tiles])
        return df, {"test": order[:n_test], "val": order[n_test:n_test + n_val], "train": order[n_test + n_val:n_test + n_tv]}
    if protocol.startswith("official:"):
        sc = protocol.split(":", 1)[1]
        aux = pd.read_csv(ROOT / "datasets" / "tile_info224_v3_aux.csv")
        df = pd.read_csv(ROOT / "research" / "splits" / f"official_{sc}_seed{seed}.csv")
        assert (df.FileName.values == aux.FileName.values).all()
        df["ImageRawPath"] = aux.ImageRawPath.values
        return df, {k: np.flatnonzero(df.split.values == k) for k in ("train", "val", "test")}
    name = {"site": f"site_seed{seed}.csv", "w2e": f"west_to_east_seed{seed}.csv", "e2w": f"east_to_west_seed{seed}.csv"}[protocol]
    df = pd.read_csv(ROOT / "research" / "splits" / name)
    names = {"site": ("train", "val", "test"), "w2e": ("source_train", "source_val", "target_test"),
             "e2w": ("source_train", "source_val", "target_test")}[protocol]
    idx = {k: np.flatnonzero(df.split.values == n) for k, n in zip(("train", "val", "test"), names)}
    if protocol != "site":
        idx["pool"] = np.flatnonzero(df.split.values == "target_pool")
    return df, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="site", help="site | tile | w2e | e2w | official:<scenario>")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--model", default="unet_r34")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--loss", default="bce_dice", choices=["bce_dice", "bce_ftversky"])
    ap.add_argument("--norm", default="global", choices=["global", "tile", "both"])
    ap.add_argument("--aux", action="store_true", help="auxiliary distance + boundary heads (training only)")
    ap.add_argument("--photo-aug", action="store_true")
    ap.add_argument("--upscale", type=int, default=1)
    ap.add_argument("--adabn", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--data-dir", default=str(ROOT / "datasets" / "packed"))
    ap.add_argument("--out", default=str(ROOT / "research" / "reports" / "runs"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--recipe", choices=["dev", "paper"], default="dev",
                    help="paper: batch 32, AdamW 1e-4, ExponentialLR(0.95), <=100 epochs, early stop on val loss "
                         "(patience 5), checkpoint by val loss, threshold 0.5 (TreeFinder Sec. 4.1)")
    ap.add_argument("--max-train", type=int, default=0, help="debug: limit tiles per split")
    ap.add_argument("--in-memory", action="store_true")
    ap.add_argument("--num-threads", type=int, default=4)
    a = ap.parse_args()

    paper = a.recipe == "paper"
    a.epochs = a.epochs or (100 if paper else 30)
    a.lr = a.lr or (1e-4 if paper else 3e-4)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays = Arrays(Path(a.data_dir), a.in_memory)
    df, idx = load_manifest(a.protocol, a.seed)
    if a.max_train:
        rng = np.random.default_rng(a.seed)
        idx = {k: np.sort(rng.choice(v, min(len(v), a.max_train if k == "train" else max(8, a.max_train // 4)), replace=False)) for k, v in idx.items()}

    # global per-band statistics from (a subset of) the training tiles only
    sub = np.sort(np.random.default_rng(0).choice(idx["train"], min(len(idx["train"]), 1500), replace=False))
    xs, _ = arrays.batch(sub, "cpu")
    x5 = Prep.make5(xs)
    mean = x5.mean((0, 2, 3)).view(1, 5, 1, 1).to(device)
    std = x5.std((0, 2, 3)).view(1, 5, 1, 1).to(device)
    prep = Prep(a.norm, mean, std)

    model = Upscaled(build_model(a.model, 10 if a.norm == "both" else 5, 3 if a.aux else 1), a.upscale).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(idx["train"]) // a.bs)
    if paper:
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)  # stepped once per epoch
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=False)

    name = f"{a.protocol.replace(':', '-')}_{a.model}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)
    best = {"f1": -1.0, "epoch": 0}
    bad = 0
    ckpt = out / "best.pt"
    history = []
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        perm = np.random.permutation(idx["train"])
        tot = 0.0
        for s in range(steps_per_epoch):
            xb, yb = arrays.batch(perm[s * a.bs:(s + 1) * a.bs], device)
            x, t, v = prep(xb, yb, augment=True, photo=a.photo_aug)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
            logits = logits.float()
            loss = seg_loss(logits[:, :1], t, v, a.loss)
            if a.aux:
                loss = loss + aux_loss(logits, t, v)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if not paper:
                sched.step()
            tot += loss.item()
        h = predict_hist(model, arrays, idx["val"], prep, device, a.bs * 2).sum(0)
        thr, f1 = best_threshold(h)
        row = {"epoch": epoch, "train_loss": tot / steps_per_epoch, "val_f1_best_thr": f1, "val_thr": thr,
               "val_ap": average_precision(h), "elapsed_min": (time.time() - t0) / 60}
        history.append(row)
        print(json.dumps(row), flush=True)
        if paper:
            vl = val_loss(model, arrays, idx["val"], prep, device, a.bs, a.loss)
            row["val_loss"] = vl
            sched.step()
            if vl < best.get("loss", float("inf")):
                best = {"f1": f1, "epoch": epoch, "loss": vl}
                bad = 0
                torch.save(model.state_dict(), ckpt)
            else:
                bad += 1
                if bad >= 5:  # early stopping, patience 5
                    print("early stop", epoch, flush=True)
                    break
        elif f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch}
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    hv = predict_hist(model, arrays, idx["val"], prep, device, a.bs * 2, a.tta)
    thr_val, f1_val = best_threshold(hv.sum(0))

    def evaluate(tag: str):
        ht = predict_hist(model, arrays, idx["test"], prep, device, a.bs * 2, a.tta)
        order = np.sort(idx["test"])
        site = df.ImageRawPath.values[order]
        s_hist = pd.DataFrame({"site": site}).groupby("site").indices
        per_site = {k: ht[v].sum(0) for k, v in s_hist.items()}
        total = ht.sum(0)
        thr_oracle, f1_oracle = best_threshold(total)
        return {
            "official": official_metrics(total, 0.5),
            "objects_at_val_thr": object_metrics(model, arrays, idx["test"], prep, device, a.bs * 2, thr_val),
            "at_0.5": metrics_from_hist(total, 0.5),
            "at_val_thr": metrics_from_hist(total, thr_val),
            "oracle_thr": {"thr": thr_oracle, **metrics_from_hist(total, thr_oracle)},
            "ap": average_precision(total),
            "positive_rate": float(total[0].sum() / max(1, total.sum())),
        }, per_site

    res, per_site = evaluate("test")
    report = {"args": vars(a), "params_M": n_params, "best_epoch": best["epoch"], "val_f1": f1_val, "val_thr": thr_val,
              "history": history, "test": res, "minutes": (time.time() - t0) / 60}
    def dump_sites(per, thr, fname):
        rows = [{"site": k, **{q: m[q] for q in ("tp", "fp", "fn")}} for k, h in per.items() for m in [metrics_from_hist(h, thr)]]
        pd.DataFrame(rows).to_csv(out / fname, index=False)

    dump_sites(per_site, thr_val, "test_by_site.csv")
    if a.adabn and a.protocol in ("w2e", "e2w"):
        if adabn(model, arrays, idx["pool"], prep, device, a.bs):
            # threshold is still chosen on *source* validation data, re-scored with the adapted statistics
            hv2 = predict_hist(model, arrays, idx["val"], prep, device, a.bs * 2, a.tta)
            thr_val, _ = best_threshold(hv2.sum(0))
            report["test_adabn"], per_adapt = evaluate("test_adabn")
            dump_sites(per_adapt, thr_val, "test_by_site_adabn.csv")
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    t = res
    print("RESULT", name, json.dumps({"val_f1": round(f1_val, 4), "test_f1@0.5": round(t["at_0.5"]["f1"], 4),
          "test_f1@val_thr": round(t["at_val_thr"]["f1"], 4), "AP": round(t["ap"], 4),
          "oracle_f1": round(t["oracle_thr"]["f1"], 4),
          "official_dead_f1": round(t["official"]["dead_f1"], 4), "official_all_f1": round(t["official"]["all_f1"], 4)}), flush=True)


if __name__ == "__main__":
    main()
