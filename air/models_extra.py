# air/models_extra.py -- additional strong baselines. All take the 5-channel RGB+NIR+NDVI tile and return one logit map
# at the input resolution.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation


def _resize(t, size):
    return F.interpolate(t, size=size, mode="bilinear", align_corners=False)


def widen_first_conv(conv: nn.Conv2d, in_ch: int) -> nn.Conv2d:
    """Keep the pretrained RGB filters; extra channels start from the mean RGB filter."""
    if conv.in_channels == in_ch:
        return conv
    new = nn.Conv2d(in_ch, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight[:, :3].copy_(conv.weight[:, :3])
        new.weight[:, 3:].copy_(conv.weight[:, :3].mean(1, keepdim=True).repeat(1, in_ch - 3, 1, 1))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def replace_first_conv(root: nn.Module, in_ch: int) -> str:
    for name, m in root.named_modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3:
            parent = root
            *path, leaf = name.split(".")
            for q in path:
                parent = getattr(parent, q)
            setattr(parent, leaf, widen_first_conv(m, in_ch))
            return name
    raise RuntimeError("no 3-channel stem conv found")


class SegFormerHF(nn.Module):
    """SegFormer (MiT) using all five channels. B5 is the encoder used by DT-V1 (Mohring et al. 2025)."""

    def __init__(self, variant="b5", in_channels=5, pretrained=True):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                f"nvidia/segformer-{variant}-finetuned-ade-640-640", num_labels=1, ignore_mismatched_sizes=True)
        else:
            cfg = {"b5": dict(depths=[3, 6, 40, 3], hidden_sizes=[64, 128, 320, 512], num_attention_heads=[1, 2, 5, 8])}[variant]
            self.net = SegformerForSemanticSegmentation(SegformerConfig(num_labels=1, **cfg))
        replace_first_conv(self.net, in_channels)

    def forward(self, x):
        return _resize(self.net(pixel_values=x).logits, x.shape[2:])


class UperNetSwin(nn.Module):
    """UPerNet with a Swin backbone (HF openmmlab/upernet-swin-*), ADE20K-pretrained decode head."""

    def __init__(self, variant="small", in_channels=5, pretrained=True):
        super().__init__()
        from transformers import SwinConfig, UperNetConfig, UperNetForSemanticSegmentation
        if pretrained:
            self.net = UperNetForSemanticSegmentation.from_pretrained(
                f"openmmlab/upernet-swin-{variant}", num_labels=1, ignore_mismatched_sizes=True, use_auxiliary_head=False)
        else:
            sw = SwinConfig(embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=7,
                            out_features=["stage1", "stage2", "stage3", "stage4"])
            self.net = UperNetForSemanticSegmentation(UperNetConfig(backbone_config=sw, num_labels=1, use_auxiliary_head=False))
        replace_first_conv(self.net.backbone, in_channels)

    def forward(self, x):
        return _resize(self.net(pixel_values=x).logits, x.shape[2:])


class EffUNet(nn.Module):
    """U-Net with an EfficientNet-B7 encoder ("EfficientUNet", a baseline in TreeMortSeg)."""

    def __init__(self, in_channels=5, pretrained=True):
        super().__init__()
        import segmentation_models_pytorch as smp
        w = "imagenet" if pretrained else None
        try:
            self.net = smp.Unet("efficientnet-b7", encoder_weights=w, in_channels=in_channels, classes=1)
        except Exception as exc:      # smp downloads these weights from GitHub; fall back to timm's HF-hosted copy
            print("EffUNet: smp weights unavailable (%s); using timm tu-efficientnet_b7" % type(exc).__name__)
            self.net = smp.Unet("tu-efficientnet_b7", encoder_weights=w, in_channels=in_channels, classes=1)

    def forward(self, x):
        return self.net(x)


class Dinov2Seg(nn.Module):
    """DINOv2 ViT-B/14 + multi-level convolutional decoder. The tile is upsampled x2 internally so that a ~9 px dead crown
    spans more than one 14 px patch. Hidden states of layers 3/6/9/12 feed a progressive-upsampling decoder."""

    def __init__(self, in_channels=5, pretrained=True, up=2, name="facebook/dinov2-base", layers=(3, 6, 9, 12)):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model
        self.up, self.layers = up, tuple(layers)
        self.vit = Dinov2Model.from_pretrained(name) if pretrained else Dinov2Model(Dinov2Config.from_pretrained(name))
        pe = self.vit.embeddings.patch_embeddings
        pe.projection = widen_first_conv(pe.projection, in_channels)
        pe.num_channels = in_channels
        d = self.vit.config.hidden_size
        self.proj = nn.ModuleList([Conv2dNormActivation(d, 128, kernel_size=1) for _ in range(4)])
        self.fuse = Conv2dNormActivation(512, 192, kernel_size=3)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 Conv2dNormActivation(192, 96, kernel_size=3))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 Conv2dNormActivation(96, 48, kernel_size=3))
        self.head = nn.Conv2d(48, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        size = x.shape[2:]
        x = torch.cat([(x[:, :3] - self.mean) / self.std, x[:, 3:]], 1)
        x = F.interpolate(x, scale_factor=self.up, mode="bilinear", align_corners=False)
        h, w = x.shape[2] // 14, x.shape[3] // 14
        x = x[:, :, : h * 14, : w * 14]
        hs = self.vit(pixel_values=x, output_hidden_states=True).hidden_states
        feats = []
        for proj, i in zip(self.proj, self.layers):
            t = hs[i][:, 1:]                                               # drop the CLS token
            feats.append(proj(t.transpose(1, 2).reshape(t.shape[0], -1, h, w)))
        y = self.up2(self.up1(self.fuse(torch.cat(feats, 1))))
        return _resize(self.head(y), size)


class HRNetSeg(nn.Module):
    """HRNet-W32 (high-resolution branch kept to the end) with the usual concatenate-all-branches head."""

    def __init__(self, in_channels=5, pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model("hrnet_w32", pretrained=pretrained, features_only=True,
                                          out_indices=(1, 2, 3, 4), in_chans=in_channels)
        ch = self.backbone.feature_info.channels()
        self.head = nn.Sequential(Conv2dNormActivation(sum(ch), 256, kernel_size=1),
                                  Conv2dNormActivation(256, 128, kernel_size=3), nn.Dropout2d(0.1), nn.Conv2d(128, 1, 1))

    def forward(self, x):
        fs = self.backbone(x)
        tgt = fs[0].shape[2:]
        y = torch.cat([f if f.shape[2:] == tgt else _resize(f, tgt) for f in fs], 1)
        return _resize(self.head(y), x.shape[2:])


def build_extra(name: str, in_channels: int = 5, pretrained: bool = True):
    if name.startswith("m_"):
        from .models_modular import build_modular
        return build_modular(name, in_channels, pretrained)
    table = {
        "segformer_b5": lambda: SegFormerHF("b5", in_channels, pretrained),
        "upernet_swin_s": lambda: UperNetSwin("small", in_channels, pretrained),
        "effunet_b7": lambda: EffUNet(in_channels, pretrained),
        "dinov2_b": lambda: Dinov2Seg(in_channels, pretrained),
        "dinov2_s": lambda: Dinov2Seg(in_channels, pretrained, name="facebook/dinov2-small", layers=(3, 6, 9, 12)),
        "dinov2_l": lambda: Dinov2Seg(in_channels, pretrained, name="facebook/dinov2-large", layers=(6, 12, 18, 24)),
        "hrnet_w32": lambda: HRNetSeg(in_channels, pretrained),
    }
    return table[name]() if name in table else None
