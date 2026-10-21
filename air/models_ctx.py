"""Two-scale context network.

Input: the 3x3 mosaic around a tile (672 x 672 px, built by air/context.py with margin=224).  The central 224 px tile is
processed at full resolution; the whole mosaic is average-pooled by 3 (to 224 px, i.e. 1.8 m / px) and encoded by the
SAME encoder, giving a coarse view of the surrounding 400 m.  The coarse features are cropped to the footprint of the
central tile, projected (zero-initialised, so training starts from the plain ConvNeXt-UPerNet) and added to the encoder
features of the central tile at strides 8, 16 and 32.  The logits are written into a 672 x 672 canvas so that the benchmark
loader / metric (which only scores the central tile) can be used unchanged.  Cost: 2 encoder passes at 224 px (~2x), instead
of 9x for feeding the full mosaic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from air.models import ConvNeXtAdapter, ConvNeXtUPerNet

T = 224


class DualScaleConvNeXt(nn.Module):
    def __init__(self, in_channels=5, pretrained=True, variant="small"):
        super().__init__()
        net = ConvNeXtUPerNet(in_channels=in_channels, num_classes=1, pretrained=pretrained, variant=variant)
        self.backbone, self.neck, self.head = net.backbone, net.neck, net.head
        dims = ConvNeXtAdapter.DIMS[variant]
        self.proj = nn.ModuleList([nn.Conv2d(d, d, 1) for d in dims[1:]])        # strides 8, 16, 32
        for p in self.proj:
            nn.init.zeros_(p.weight)
            nn.init.zeros_(p.bias)

    @staticmethod
    def _center(g, size):
        """g: [B,C,h,w] coarse map covering 3x3 tiles -> its central third resized to `size`."""
        h, w = g.shape[2:]
        g = F.interpolate(g, size=(3 * size[0], 3 * size[1]), mode="bilinear", align_corners=False)
        return g[:, :, size[0]:2 * size[0], size[1]:2 * size[1]]

    def forward(self, x):
        B, _, S, _ = x.shape
        assert S == 3 * T, "DualScaleConvNeXt expects a 672 px mosaic (use --ctx 224)"
        center = x[:, :, T:2 * T, T:2 * T]
        coarse = F.avg_pool2d(x, 3)
        fc = self.backbone(center)
        fg = self.backbone(coarse)
        for k, proj in zip((1, 2, 3), self.proj):
            fc[k] = fc[k] + proj(self._center(fg[k], fc[k].shape[2:]))
        logit = F.interpolate(self.head(self.neck(fc)["Fused"]), size=(T, T), mode="bilinear", align_corners=False)
        canvas = x.new_full((B, 1, S, S), -20.0, dtype=logit.dtype)
        canvas[:, :, T:2 * T, T:2 * T] = logit
        return canvas
