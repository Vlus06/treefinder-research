"""Training-time helpers for the knowledge-transfer experiments (all optional, off by default).

  GateWrap       tile-level "contains dead trees" head on the deepest encoder map; its log-probability is added to the
                 mask logit, so empty-looking tiles are suppressed at train AND test time (targets empty-tile false positives)
  DinoTeacher    frozen DINOv2-B on the RGB bands (input x2); patch tokens are the target of the representation alignment loss
  Aligner        1x1 projection of the ConvNeXt stride-16 map to DINOv2 width + cosine loss (no labels involved)
  copy_paste     paste small labelled dead crowns from other tiles of the batch (rare, small objects are the dominant miss)
  load_teacher   restore a finished air/ checkpoint as a frozen teacher for soft-label distillation
"""
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


class GateWrap(nn.Module):
    def __init__(self, core, feat_dim=768):
        super().__init__()
        self.core = core
        self.gate = nn.Linear(feat_dim, 1)
        self._f = None
        core.backbone.register_forward_hook(self._grab)          # bound method: survives copy.deepcopy (EMA)

    def _grab(self, m, i, o):
        self._f = o

    def forward(self, x):
        out = self.core(x)
        logit = out[0] if isinstance(out, (tuple, list)) else out
        g = self.gate(self._f[-1].float().mean((2, 3)))                 # [B,1]
        return logit + F.logsigmoid(g)[:, :, None, None].to(logit.dtype), g


class DinoTeacher(nn.Module):
    def __init__(self, up=2):
        super().__init__()
        from transformers import Dinov2Model
        self.vit = Dinov2Model.from_pretrained("facebook/dinov2-base").eval()
        for p in self.vit.parameters():
            p.requires_grad_(False)
        self.up = up
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        x = (x[:, :3] - self.mean) / self.std
        x = F.interpolate(x, scale_factor=self.up, mode="bilinear", align_corners=False)
        h, w = x.shape[2] // 14, x.shape[3] // 14
        t = self.vit(pixel_values=x[:, :, :h * 14, :w * 14]).last_hidden_state[:, 1:]
        return t.transpose(1, 2).reshape(t.shape[0], -1, h, w).float()


class Aligner(nn.Module):
    def __init__(self, c_student=384, c_teacher=768):
        super().__init__()
        self.proj = nn.Conv2d(c_student, c_teacher, 1)

    def forward(self, s, t):
        s = self.proj(s.float())
        t = F.adaptive_avg_pool2d(t, s.shape[2:])
        return (1 - F.cosine_similarity(s, t, dim=1)).mean()


def copy_paste(x, y, valid, prob=0.5, min_px=15, max_px=1000, margin=4, rng=np.random):
    """In-place on clones: for each tile, with prob `prob`, paste the window around one small dead crown of another tile."""
    x, y, valid = x.clone(), y.clone(), valid.clone()
    lab = y[:, 0].cpu().numpy() > 0.5
    B, _, H, W = x.shape
    src_x, src_y, src_v = x.clone(), y.clone(), valid.clone()
    for i in range(B):
        if rng.rand() > prob:
            continue
        j = int(rng.randint(B))
        if j == i or not lab[j].any():
            continue
        cc, n = ndimage.label(lab[j])
        sizes = ndimage.sum(lab[j], cc, range(1, n + 1))
        ok = [k + 1 for k, s in enumerate(sizes) if min_px <= s <= max_px]
        if not ok:
            continue
        k = ok[rng.randint(len(ok))]
        ys, xs = np.where(cc == k)
        y0, y1 = max(0, ys.min() - margin), min(H, ys.max() + 1 + margin)
        x0, x1 = max(0, xs.min() - margin), min(W, xs.max() + 1 + margin)
        h, w = y1 - y0, x1 - x0
        ty, tx = rng.randint(0, H - h + 1), rng.randint(0, W - w + 1)
        x[i, :, ty:ty + h, tx:tx + w] = src_x[j, :, y0:y1, x0:x1]
        y[i, :, ty:ty + h, tx:tx + w] = src_y[j, :, y0:y1, x0:x1]
        valid[i, :, ty:ty + h, tx:tx + w] = src_v[j, :, y0:y1, x0:x1]
    return x, y, valid


def load_teacher(exp, ckpt_dirs, build, wrap, device):
    """exp like air0042_convnext_upernet / air0043_treemortseg-none / air0044_dinov2_b-base. Returns a frozen eval model."""
    name = re.sub(r"^air\d+_", "", exp)
    model_name = name.split("-")[0]
    ck = next(Path(d) / exp / f"{exp}_best.pth" for d in ckpt_dirs if (Path(d) / exp / f"{exp}_best.pth").exists())
    net = wrap(build(model_name, 5))                       # AirInput(core), as in training
    sd = {k[4:] if k.startswith("net.") else k: v for k, v in torch.load(ck, map_location="cpu").items()}   # drop MaskOnly
    (net if any(k.startswith("net.") for k in sd) else net.net).load_state_dict(sd)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net
