# air/models_modular.py -- backbone x decoder grid used to *select* the architecture instead of assuming it.
# Model names look like  m_<backbone>_<decoder>,  e.g.  m_cnxs_upernet, m_r50_unet, m_cnx2b_dlv3p.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation

BACKBONES = {
    "cnxs": "convnext_small.fb_in22k_ft_in1k",          # ConvNeXt-S, ImageNet-22k pre-training
    "cnx2t": "convnextv2_tiny.fcmae_ft_in22k_in1k",     # ConvNeXt-V2 tiny (FCMAE self-supervised + 22k)
    "cnx2b": "convnextv2_base.fcmae_ft_in22k_in1k",     # ConvNeXt-V2 base
    "r50": "resnet50.a1_in1k",                          # classic CNN control
    "regy4": "regnety_040.ra3_in1k",                    # RegNetY-4GF (efficient CNN)
}
DECODERS = ("upernet", "fpn", "unet", "dlv3p")


class TimmBackbone(nn.Module):
    """Returns the four feature maps with strides 4, 8, 16, 32."""

    def __init__(self, key, in_ch=5, pretrained=True):
        super().__init__()
        import timm
        self.m = timm.create_model(BACKBONES[key], pretrained=pretrained, features_only=True, in_chans=in_ch)
        red = self.m.feature_info.reduction()
        self.idx = [i for i, r in enumerate(red) if r in (4, 8, 16, 32)]
        assert len(self.idx) == 4, red
        self.channels = [self.m.feature_info.channels()[i] for i in self.idx]

    def forward(self, x):
        fs = self.m(x)
        return [fs[i] for i in self.idx]


def _up(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class UPerNetDec(nn.Module):
    def __init__(self, chs, c=128):
        super().__init__()
        from .models import UPerNetNeck
        self.neck = UPerNetNeck(in_channels_list=chs, out_channels=c)
        self.head = nn.Sequential(Conv2dNormActivation(4 * c, c, 3), nn.Dropout2d(0.1), nn.Conv2d(c, 1, 1))

    def forward(self, fs, size):
        return _up(self.head(self.neck(fs)["Fused"]), size)


class FPNDec(nn.Module):
    """Top-down feature pyramid with the same head as UPerNet but without the pyramid-pooling module."""

    def __init__(self, chs, c=128):
        super().__init__()
        self.lat = nn.ModuleList([Conv2dNormActivation(k, c, 1) for k in chs])
        self.smooth = nn.ModuleList([Conv2dNormActivation(c, c, 3) for _ in chs])
        self.head = nn.Sequential(Conv2dNormActivation(4 * c, c, 3), nn.Dropout2d(0.1), nn.Conv2d(c, 1, 1))

    def forward(self, fs, size):
        p = [l(f) for l, f in zip(self.lat, fs)]
        for i in (2, 1, 0):
            p[i] = p[i] + _up(p[i + 1], p[i].shape[2:])
        p = [s(x) for s, x in zip(self.smooth, p)]
        y = torch.cat([_up(x, p[0].shape[2:]) for x in p], 1)
        return _up(self.head(y), size)


class _Double(nn.Sequential):
    def __init__(self, i, o):
        super().__init__(Conv2dNormActivation(i, o, 3), Conv2dNormActivation(o, o, 3))


class UNetDec(nn.Module):
    """U-Net decoder with skips at strides 16/8/4, then two un-skipped x2 refinements up to the full resolution."""

    def __init__(self, chs, dec=(256, 128, 64)):
        super().__init__()
        self.b3 = _Double(chs[3], 384)
        self.d = nn.ModuleList()
        prev = 384
        for skip, o in zip((chs[2], chs[1], chs[0]), dec):
            self.d.append(_Double(prev + skip, o))
            prev = o
        self.r1 = _Double(prev, 32)
        self.r2 = _Double(32, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, fs, size):
        x = self.b3(fs[3])
        for blk, skip in zip(self.d, (fs[2], fs[1], fs[0])):
            x = blk(torch.cat([_up(x, skip.shape[2:]), skip], 1))
        x = self.r1(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        x = self.r2(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return _up(self.head(x), size)


class ASPP(nn.Module):
    def __init__(self, i, o=256, rates=(3, 6, 9)):
        super().__init__()
        self.b = nn.ModuleList([Conv2dNormActivation(i, o, 1)] +
                               [Conv2dNormActivation(i, o, 3, dilation=r, padding=r) for r in rates])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), Conv2dNormActivation(i, o, 1))
        self.proj = nn.Sequential(Conv2dNormActivation(o * (len(rates) + 2), o, 1), nn.Dropout2d(0.1))

    def forward(self, x):
        y = [b(x) for b in self.b] + [_up(self.pool(x), x.shape[2:])]
        return self.proj(torch.cat(y, 1))


class DLv3pDec(nn.Module):
    """DeepLabV3+: ASPP on the stride-16 map (small rates because a 224 px tile gives a 14x14 map) + low-level skip."""

    def __init__(self, chs):
        super().__init__()
        self.aspp = ASPP(chs[2])
        self.low = Conv2dNormActivation(chs[0], 48, 1)
        self.fuse = nn.Sequential(Conv2dNormActivation(256 + 48, 256, 3), Conv2dNormActivation(256, 128, 3), nn.Dropout2d(0.1),
                                  nn.Conv2d(128, 1, 1))

    def forward(self, fs, size):
        hi = _up(self.aspp(fs[2]), fs[0].shape[2:])
        return _up(self.fuse(torch.cat([hi, self.low(fs[0])], 1)), size)


class ModularSeg(nn.Module):
    def __init__(self, bb, dec, in_ch=5, pretrained=True):
        super().__init__()
        self.backbone = TimmBackbone(bb, in_ch, pretrained)
        chs = self.backbone.channels
        self.decoder = {"upernet": UPerNetDec, "fpn": FPNDec, "unet": UNetDec, "dlv3p": DLv3pDec}[dec](chs)

    def forward(self, x):
        return self.decoder(self.backbone(x), x.shape[2:])


def build_modular(name, in_ch=5, pretrained=True):
    """name = m_<backbone>_<decoder>; returns None when the name is not of that form."""
    if not name.startswith("m_"):
        return None
    _, bb, dec = name.split("_", 2)
    return ModularSeg(bb, dec, in_ch, pretrained)
