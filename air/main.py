"""Train / evaluate TreeFinder models inside the benchmark's own pipeline (same splits, loaders, metric code).

    python air/main.py --config configs/res_random.yaml --model treemortseg --aux fixed --seed 2025 --exp_id 5000
    python air/main.py --config configs/res_random_site.yaml --model convnext_upernet --norm tile --photo \
        --sampler site --vrex 1.0 --ema 0.999 --tta --exp_id 7000

--model : unet | deeplabv3 (benchmark builders)  |  convnext_upernet | treemortseg (air/models.py)
--aux   : none | released | fixed      (auxiliary-target mode, only meaningful for treemortseg)
AIR components (all off by default, so the plain runs are unchanged):
  --norm tile    per-tile z-score over valid pixels, applied inside the model (train and test)
  --photo        per-tile, per-band gain / bias / gamma augmentation (NDVI recomputed)
  --sampler site batches of G sites x K tiles (site-grouped sampling)
  --vrex L       variance-of-risks penalty across the sites of a batch (needs --sampler site)
  --ema D        exponential moving average of the weights, evaluated and checkpointed instead of the raw model
  --tta          flip / transpose averaging at test time
Recipe  : AdamW(wd 0.01), ExponentialLR(0.95), batch 32, <=100 epochs, BCE+Dice mask loss; early stopping on the
          *mask* validation loss, so that all variants are selected on the same criterion.
Metrics : exps.evaluate.evaluate_model (threshold 0.5, pooled counts), identical to the benchmark.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from air.dataset import AirDataset                      # noqa: E402
from air.losses import EdgeBoundaryLoss, dist_loss       # noqa: E402
from exps.evaluate import evaluate_model                 # noqa: E402
from exps.train import DiceLoss                          # noqa: E402
from utils.tools import setup_logging, set_seed          # noqa: E402


class AirInput(nn.Module):
    """Input-side robustness: per-tile standardisation (valid = any band non-zero) and optional flip TTA."""

    def __init__(self, net, norm="none", tta=False, up=1.0):
        super().__init__()
        self.net, self.norm, self.tta, self.up = net, norm, tta, up

    def _prep(self, x):
        if self.norm != "tile":
            return x
        valid = (x[:, :4].sum(1, keepdim=True) > 0).float()
        n = valid.sum((2, 3), keepdim=True).clamp_min(1)
        mu = (x * valid).sum((2, 3), keepdim=True) / n
        var = (((x - mu) ** 2) * valid).sum((2, 3), keepdim=True) / n
        return (x - mu) / (var.sqrt() + 1e-3) * valid

    def _fwd(self, x):
        size = x.shape[2:]
        x = self._prep(x)
        if self.up != 1.0:
            x = torch.nn.functional.interpolate(x, scale_factor=self.up, mode="bilinear", align_corners=False)
        out = self.net(x)
        if self.up != 1.0:
            rs = lambda o: torch.nn.functional.interpolate(o, size=size, mode="bilinear", align_corners=False)
            out = [rs(o) for o in out] if isinstance(out, (tuple, list)) else rs(out)
        return out

    def forward(self, x):
        if not (self.tta and not self.training):
            return self._fwd(x)
        outs = []
        for flip in (None, (3,), (2,), (2, 3)):
            xi = x if flip is None else x.flip(flip)
            o = self._fwd(xi)
            o = o[0] if isinstance(o, (tuple, list)) else o
            outs.append(o if flip is None else o.flip(flip))
        return torch.stack(outs).mean(0)


class MaskOnly(nn.Module):
    """Expose a multi-output model as a single-logit model for the benchmark's evaluate_model."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return out[0] if isinstance(out, (tuple, list)) else out


class SiteGroupSampler(Sampler):
    """Each batch = `groups` sites drawn uniformly (with replacement) x `per_group` tiles from each site."""

    def __init__(self, site_ids, groups=8, per_group=4, n_batches=None, seed=0):
        self.by_site = {}
        for i, s in enumerate(site_ids):
            self.by_site.setdefault(int(s), []).append(i)
        self.sites = sorted(self.by_site)
        self.groups, self.per_group = groups, per_group
        self.n_batches = n_batches or max(1, len(site_ids) // (groups * per_group))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_batches * self.groups * self.per_group

    def __iter__(self):
        for _ in range(self.n_batches):
            for s in self.rng.choice(len(self.sites), self.groups, replace=True):
                tiles = self.by_site[self.sites[s]]
                yield from self.rng.choice(tiles, self.per_group, replace=True).tolist()


def build(model_name, in_ch, gate=False):
    core = _build(model_name, in_ch)
    if gate:
        from air.distill import GateWrap
        core = GateWrap(core, 768)
    return core


def _build(model_name, in_ch):
    if model_name in ("unet", "deeplabv3"):
        from models import get_model
        cfg = yaml.safe_load(open(ROOT / "configs" / "res_random.yaml"))["model"]
        cfg["name"] = model_name
        return get_model(cfg)
    from air.models import build_air_model
    return build_air_model(model_name, in_channels=in_ch)


def photometric(x):
    """x: [B,5,H,W] = R,G,B,NIR in [0,1] and rescaled NDVI. Gain/bias/gamma per tile and band; NDVI recomputed."""
    b = x.shape[0]
    valid = (x[:, :4].sum(1, keepdim=True) > 0).float()
    g = torch.empty(b, 4, 1, 1, device=x.device).uniform_(0.75, 1.25)
    o = torch.empty(b, 4, 1, 1, device=x.device).uniform_(-0.08, 0.08)
    gm = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.8, 1.25)
    bands = ((x[:, :4] * g + o).clamp(0, 1) ** gm) * valid
    ndvi = (bands[:, 3] - bands[:, 0]) / (bands[:, 3] + bands[:, 0] + 1e-10)
    ndvi = ((ndvi.clamp(-1, 1) + 1) / 2).unsqueeze(1) * valid
    return torch.cat([bands, ndvi], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/res_random.yaml")
    ap.add_argument("--model", default="treemortseg")
    ap.add_argument("--aux", default="none", choices=["none", "released", "fixed"])
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--exp_id", default="5000")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--w_edge", type=float, default=2.0)
    ap.add_argument("--w_dist", type=float, default=5.0)
    ap.add_argument("--w_boundary", type=float, default=0.2)
    ap.add_argument("--norm", default="none", choices=["none", "tile"])
    ap.add_argument("--photo", action="store_true")
    ap.add_argument("--sampler", default="random", choices=["random", "site"])
    ap.add_argument("--vrex", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--up", type=float, default=1.0, help="bilinear input upscaling factor (output resized back)")
    ap.add_argument("--sched", default="exp", choices=["exp", "cosine"])
    ap.add_argument("--loss", default="bce_dice", choices=["bce_dice", "bce_ftversky"])
    ap.add_argument("--split_seed", type=int, default=None,
                    help="seed that selects the data split (default: --seed); lets us vary training noise on a fixed split")
    ap.add_argument("--lr_backbone_mult", type=float, default=1.0, help="lr multiplier for the pretrained ViT of dinov2_b")
    ap.add_argument("--dino_align", type=float, default=0.0, help="D1: weight of cosine alignment to frozen DINOv2 patch features")
    ap.add_argument("--distill", default="", help="D2: comma-separated teacher experiment names (soft-label distillation)")
    ap.add_argument("--ckpt_dirs", default="checkpoints", help="comma-separated dirs searched for teacher checkpoints")
    ap.add_argument("--distill_alpha", type=float, default=0.5)
    ap.add_argument("--distill_T", type=float, default=1.0)
    ap.add_argument("--selfdistill", type=float, default=0.0, help="D3: weight of soft targets from the EMA of the student (needs --ema)")
    ap.add_argument("--copypaste", type=float, default=0.0, help="probability of pasting a small dead crown from another tile")
    ap.add_argument("--gate", type=float, default=0.0, help="M2: weight of the tile-level presence gate loss")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max_samples", type=int, default=0, help="debug only")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    assert a.vrex == 0 or a.sampler == "site", "--vrex needs --sampler site"

    cfg = yaml.safe_load(open(ROOT / a.config))
    cfg["experiment"]["seed"] = a.seed if a.split_seed is None else a.split_seed   # used only to build the split
    if a.workers is not None:
        cfg["experiment"]["num_workers"] = a.workers
    tag = a.model + (f"-{a.aux}" if a.model == "treemortseg" else "") + (f"-{a.tag}" if a.tag else "")
    exp_name = f"air{a.exp_id.zfill(4)}_{tag}"
    setup_logging(cfg["logging"]["log_dir"], exp_name)
    log = logging.getLogger("air")
    log.info("args=%s", vars(a))
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    aux_mode = a.aux if a.model == "treemortseg" else "none"
    train_ds = AirDataset(cfg, "train", aux_mode)
    val_ds = AirDataset(cfg, "val", "none")
    test_ds = AirDataset(cfg, "test", "none")
    if a.max_samples:
        for ds in (train_ds, val_ds, test_ds):
            ds.base.paths = ds.base.paths[:a.max_samples]
            ds.site_ids = ds.site_ids[:a.max_samples]
    nw = cfg["experiment"].get("num_workers", 8)
    bs = cfg["training"].get("batch_size", 32)

    def mk(ds, shuffle, sampler=None):
        return DataLoader(ds, batch_size=bs, shuffle=shuffle and sampler is None, sampler=sampler, num_workers=nw,
                          pin_memory=True, persistent_workers=nw > 0, prefetch_factor=2 if nw > 0 else None)

    sampler = SiteGroupSampler(train_ds.site_ids, 8, bs // 8, seed=a.seed) if a.sampler == "site" else None
    train_loader, val_loader, test_loader = mk(train_ds, True, sampler), mk(val_ds, False), mk(test_ds, False)

    core = build(a.model, in_ch=cfg["model"].get("in_channels", 5), gate=a.gate > 0)
    model = AirInput(core, norm=a.norm, tta=False, up=a.up).to(device)
    log.info("params: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)
    base = core.core if a.gate > 0 else core
    aligner = dino = None
    feats = {}
    if a.dino_align > 0:
        from air.distill import Aligner, DinoTeacher
        aligner, dino = Aligner(base.backbone.DIMS["small"][2], 768).to(device), DinoTeacher().to(device)
        base.backbone.register_forward_hook(lambda m, i, o: feats.__setitem__("s", o[2]))
    teachers = []
    if a.distill:
        from air.distill import load_teacher
        dirs = a.ckpt_dirs.split(",")
        teachers = [load_teacher(e, dirs, build, lambda n: AirInput(n, norm="none"), device) for e in a.distill.split(",")]
        log.info("teachers: %s", a.distill)
    assert a.selfdistill == 0 or a.ema > 0, "--selfdistill needs --ema"
    if a.lr_backbone_mult != 1.0:
        bb = [p for n, p in model.named_parameters() if ".vit." in "." + n]
        rest = [p for n, p in model.named_parameters() if ".vit." not in "." + n]
        opt = torch.optim.AdamW([{"params": bb, "lr": a.lr * a.lr_backbone_mult}, {"params": rest, "lr": a.lr}], weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    if aligner is not None:
        opt.add_param_group({"params": list(aligner.parameters()), "lr": a.lr, "weight_decay": 0.01})
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs) if a.sched == "cosine"
             else torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95))
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(a.ema)) if a.ema > 0 else None
    bce = nn.BCEWithLogitsLoss(reduction="none")
    dice = DiceLoss()
    edge_crit = EdgeBoundaryLoss(a.w_boundary, standard_dice=(aux_mode == "fixed"))
    use_aux = aux_mode != "none" and a.model == "treemortseg"

    def mask_loss(logits, y, valid):
        if a.loss == "bce_ftversky":          # recall-weighted focal Tversky instead of Dice
            base = (bce(logits, y) * valid).sum() / valid.sum()
            p, t = torch.sigmoid(logits) * valid, y * valid
            tp, fp, fn = (p * t).sum(), (p * (1 - t)).sum(), ((1 - p) * t).sum()
            return base + (1 - (tp + 1) / (tp + 0.3 * fp + 0.7 * fn + 1)) ** 0.75
        return (bce(logits, y) * valid).sum() / valid.sum() + dice(logits, y, valid)

    def eval_model():
        return ema.module if ema is not None else model

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"]) / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best, bad, hist, t0 = float("inf"), 0, [], time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for b in train_loader:
            x = b["image"].to(device, non_blocking=True)
            y = b["label"].unsqueeze(1).float().to(device)
            valid = (~b["no_data_mask"].unsqueeze(1)).float().to(device)
            if a.copypaste > 0:
                from air.distill import copy_paste
                x, y, valid = copy_paste(x, y, valid, prob=a.copypaste)
            x_clean = x                                      # teachers see the clean tile, the student the augmented one
            if a.photo:
                x = photometric(x)
            soft = None
            if teachers or a.selfdistill > 0:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    if teachers:
                        outs = [t(x_clean) for t in teachers]
                        lg = [(o[0] if isinstance(o, (tuple, list)) else o).float() for o in outs]
                        tw, tl = a.distill_alpha, torch.stack(lg).mean(0)
                    else:
                        ema.module.eval()
                        o = ema.module(x_clean)
                        tw, tl = a.selfdistill, (o[0] if isinstance(o, (tuple, list)) else o).float()
                soft = (tw, torch.sigmoid(tl / a.distill_T))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x)
                if dino is not None:
                    tgt = dino(x_clean)
            out = [o.float() for o in out] if isinstance(out, (tuple, list)) else [out.float()]
            if a.vrex > 0 and len(y) % 8 == 0:
                per = len(y) // 8                       # sampler yields 8 consecutive groups of `per` tiles
                risks = torch.stack([mask_loss(out[0][i * per:(i + 1) * per], y[i * per:(i + 1) * per],
                                               valid[i * per:(i + 1) * per]) for i in range(8)])
                loss = risks.mean() + a.vrex * risks.var()
            else:
                loss = mask_loss(out[0], y, valid)
            if soft is not None:                              # soft-label term: BCE(student/T, teacher prob) * T^2
                w, p_t = soft
                kd = ((bce(out[0] / a.distill_T, p_t) * valid).sum() / valid.sum()) * a.distill_T ** 2
                loss = (1 - w) * loss + w * kd
            if a.gate > 0:
                has = ((y * valid).flatten(1).sum(1, keepdim=True) > 0).float()
                loss = loss + a.gate * nn.functional.binary_cross_entropy_with_logits(out[1], has)
            if aligner is not None:
                loss = loss + a.dino_align * aligner(feats["s"], tgt)
            if use_aux:
                edge = b["edge"].unsqueeze(1).to(device)
                dist = b["dist"].unsqueeze(1).to(device)
                bd = b["boundary_dist"].unsqueeze(1).to(device)
                loss = loss + a.w_edge * edge_crit(out[1], edge, bd, valid) + a.w_dist * dist_loss(out[2], dist)
            loss.backward()
            opt.step()
            if ema is not None:
                ema.update_parameters(model)
                for be, bm in zip(ema.module.buffers(), model.buffers()):   # BN statistics are copied, not averaged
                    be.copy_(bm)
            tot += loss.item()
            nb += 1
        sched.step()
        ev = eval_model()
        ev.eval()
        vl = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for b in val_loader:
                x = b["image"].to(device)
                y = b["label"].unsqueeze(1).float().to(device)
                valid = (~b["no_data_mask"].unsqueeze(1)).float().to(device)
                o = ev(x)
                o = o[0] if isinstance(o, (tuple, list)) else o
                vl += mask_loss(o.float(), y, valid).item()
        vl /= max(1, len(val_loader))
        hist.append({"epoch": epoch, "train_loss": tot / max(1, nb), "val_mask_loss": vl, "min": (time.time() - t0) / 60})
        log.info("epoch %d train %.4f val_mask %.4f (%.1f min)", epoch, hist[-1]["train_loss"], vl, hist[-1]["min"])
        if vl < best:
            best, bad = vl, 0
            torch.save(MaskOnly(ev).state_dict(), ckpt_dir / f"{exp_name}_best.pth")
        else:
            bad += 1
            if bad >= a.patience:
                log.info("early stopping at epoch %d", epoch)
                break

    final = eval_model()
    final.tta = a.tta                                            # evaluate_model reloads the best checkpoint itself
    res = evaluate_model(MaskOnly(final), test_loader, cfg, exp_name)
    out = {"args": vars(a), "exp_name": exp_name, "best_val_mask_loss": best, "epochs": len(hist),
           "minutes": (time.time() - t0) / 60, "metrics": res, "history": hist}
    rd = Path(cfg["output"]["results_dir"]) / exp_name
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    log.info("Combined metrics: %s", json.dumps(res, default=float))


if __name__ == "__main__":
    main()
