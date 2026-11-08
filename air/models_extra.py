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


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B(A(x)); W is frozen, A and B are trained (B starts at zero, so the layer starts as the pretrained one)."""

    def __init__(self, base: nn.Linear, r=16, alpha=16, dropout=0.0):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + self.B(self.A(self.drop(x))) * self.scale


class _ResidualConvUnit(nn.Module):
    """RefineNet residual unit used by the DPT fusion blocks: ReLU-conv-ReLU-conv + skip."""

    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class _FusionBlock(nn.Module):
    """DPT feature-fusion block: (optional skip input through a residual unit) + residual unit + x2 bilinear upsampling + 1x1 conv."""

    def __init__(self, c):
        super().__init__()
        self.rcu1, self.rcu2 = _ResidualConvUnit(c), _ResidualConvUnit(c)
        self.out = nn.Conv2d(c, c, 1)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.rcu1(skip)
        x = self.rcu2(x)
        return self.out(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))


class DPTDecoder(nn.Module):
    """Dense-prediction-transformer decoder (Ranftl et al., ICCV 2021), re-implemented for 4 ViT taps on a single-resolution token grid:
    reassemble (project + resample to x4 / x2 / x1 / x0.5 of the token grid) -> 3x3 conv to a common width -> RefineNet-style fusion from the
    deepest to the shallowest level -> head. Class token is ignored ("ignore" read operation)."""

    def __init__(self, d=768, out_channels=(96, 192, 384, 768), features=128):
        super().__init__()
        self.project = nn.ModuleList([nn.Conv2d(d, c, 1) for c in out_channels])
        self.resample = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1)])
        self.rn = nn.ModuleList([nn.Conv2d(c, features, 3, padding=1, bias=False) for c in out_channels])
        self.fusion = nn.ModuleList([_FusionBlock(features) for _ in range(4)])
        self.head = nn.Sequential(nn.Conv2d(features, features // 2, 3, padding=1), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                  nn.Conv2d(features // 2, 32, 3, padding=1), nn.ReLU(True), nn.Conv2d(32, 1, 1))

    def forward(self, maps):
        """maps: 4 tensors [B, d, h, w] taken from the shallow-to-deep ViT layers."""
        lv = [self.rn[i](self.resample[i](self.project[i](m))) for i, m in enumerate(maps)]
        y = self.fusion[3](lv[3])
        y = self.fusion[2](y, lv[2])
        y = self.fusion[1](y, lv[1])
        y = self.fusion[0](y, lv[0])
        return self.head(y)


class Dinov2Seg(nn.Module):
    """DINOv2 ViT-B/14 + multi-level convolutional decoder. The tile is upsampled x2 internally so that a ~9 px dead crown
    spans more than one 14 px patch. Hidden states of layers 3/6/9/12 feed a progressive-upsampling decoder."""

    def __init__(self, in_channels=5, pretrained=True, up=2, name="facebook/dinov2-base", layers=(3, 6, 9, 12), lora=0, film=False, decoder="ours"):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model
        self.up, self.layers, self.use_film = up, tuple(layers), film
        if pretrained:
            self.vit = Dinov2Model.from_pretrained(name)
        else:
            try:
                self.vit = Dinov2Model(Dinov2Config.from_pretrained(name))
            except Exception:
                self.vit = Dinov2Model(Dinov2Config())
        pe = self.vit.embeddings.patch_embeddings
        pe.projection = widen_first_conv(pe.projection, in_channels)
        pe.num_channels = in_channels
        if lora > 0:                               # parameter-efficient adaptation: frozen ViT + low-rank updates of query/value
            for p_ in self.vit.parameters():
                p_.requires_grad_(False)
            for layer in self.vit.encoder.layer:
                att = layer.attention.attention
                att.query = LoRALinear(att.query, lora, lora)
                att.value = LoRALinear(att.value, lora, lora)
            for p_ in pe.projection.parameters():  # the widened 5-channel patch embedding must be learned
                p_.requires_grad_(True)
        d = self.vit.config.hidden_size
        self.decoder_type = decoder
        if decoder == "dpt":                       # comparison decoder: DPT-style reassemble + RefineNet fusion
            self.dpt = DPTDecoder(d)
        self.proj = nn.ModuleList([Conv2dNormActivation(d, 128, kernel_size=1) for _ in range(4)])
        self.fuse = Conv2dNormActivation(512, 192, kernel_size=3)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 Conv2dNormActivation(192, 96, kernel_size=3))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 Conv2dNormActivation(96, 48, kernel_size=3))
        self.head = nn.Conv2d(48, 1, 1)
        if film:                                   # radiometric conditioning: the tile's own band mean/std modulate the decoder
            self.film = nn.Sequential(nn.Linear(8, 64), nn.SiLU(), nn.Linear(64, 2 * 192))
            nn.init.zeros_(self.film[2].weight)
            nn.init.zeros_(self.film[2].bias)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def _tile_stats(x):
        v = x[:, :4]
        valid = (v.sum(1, keepdim=True) > 0).float()
        n = valid.sum((2, 3), keepdim=True).clamp_min(1)
        mu = (v * valid).sum((2, 3), keepdim=True) / n
        sd = ((((v - mu) ** 2) * valid).sum((2, 3), keepdim=True) / n).sqrt()
        return (torch.cat([mu.flatten(1), sd.flatten(1)], 1) - 0.25) / 0.15

    def forward(self, x):
        size = x.shape[2:]
        stats = self._tile_stats(x) if self.use_film else None
        x = torch.cat([(x[:, :3] - self.mean) / self.std, x[:, 3:]], 1)
        x = F.interpolate(x, scale_factor=self.up, mode="bilinear", align_corners=False)
        h, w = x.shape[2] // 14, x.shape[3] // 14
        x = x[:, :, : h * 14, : w * 14]
        hs = self.vit(pixel_values=x, output_hidden_states=True).hidden_states
        if self.decoder_type == "dpt":
            maps = []
            for i in self.layers:
                t = hs[i][:, 1:]
                maps.append(t.transpose(1, 2).reshape(t.shape[0], -1, h, w))
            return _resize(self.dpt(maps), size)
        feats = []
        for proj, i in zip(self.proj, self.layers):
            t = hs[i][:, 1:]                                               # drop the CLS token
            feats.append(proj(t.transpose(1, 2).reshape(t.shape[0], -1, h, w)))
        y = self.fuse(torch.cat(feats, 1))
        if stats is not None:
            g, b = self.film(stats.to(y.dtype)).chunk(2, dim=1)
            y = y * (1 + g[:, :, None, None]) + b[:, :, None, None]
        y = self.up2(self.up1(y))
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
        "dinov2_b_dpt": lambda: Dinov2Seg(in_channels, pretrained, decoder="dpt"),
        "dinov2_b_lora": lambda: Dinov2Seg(in_channels, pretrained, lora=16),
        "dinov2_b_film": lambda: Dinov2Seg(in_channels, pretrained, film=True),
        "dinov2_b_lora_film": lambda: Dinov2Seg(in_channels, pretrained, lora=16, film=True),
        "dinov2_s": lambda: Dinov2Seg(in_channels, pretrained, name="facebook/dinov2-small", layers=(3, 6, 9, 12)),
        "dinov2_l": lambda: Dinov2Seg(in_channels, pretrained, name="facebook/dinov2-large", layers=(6, 12, 18, 24)),
        "hrnet_w32": lambda: HRNetSeg(in_channels, pretrained),
    }
    return table[name]() if name in table else None
