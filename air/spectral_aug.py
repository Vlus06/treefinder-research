"""Spectral / frequency augmentations for single-source domain generalisation (all act on the 5-channel batch
[R,G,B,NIR,NDVI] in [0,1], recompute NDVI so that the channels stay physically consistent, and keep no-data pixels at 0).

  photometric_parts  per-tile gain / bias / gamma / NIR-gain, each switchable, with a global strength `scale`
  bank_aug           "statistics bank": re-express a tile with the per-band mean/std of another tile drawn from a bank
                     (MixStyle / AdaIN at the pixel level, interpolated by lam ~ U(0,1)); the bank is built from per-tile band statistics
  fda_aug            Fourier amplitude mix (FACT): mix the amplitude spectrum of each band with another tile of the batch, keep the phase
"""
import numpy as np
import pandas as pd
import torch


def _valid(x):
    return (x[:, :4].sum(1, keepdim=True) > 0).float()


def _finish(bands, valid):
    """bands: [B,4,H,W] clamped to [0,1] -> [B,5,H,W] with a recomputed NDVI (rescaled to [0,1])."""
    bands = bands.clamp(0, 1) * valid
    ndvi = (bands[:, 3] - bands[:, 0]) / (bands[:, 3] + bands[:, 0] + 1e-10)
    ndvi = ((ndvi.clamp(-1, 1) + 1) / 2).unsqueeze(1) * valid
    return torch.cat([bands, ndvi], 1)


def photometric_parts(x, parts=("gain", "bias", "gamma"), scale=1.0):
    b, dev = x.shape[0], x.device
    valid = _valid(x)
    v = x[:, :4]
    if "gain" in parts:
        v = v * torch.empty(b, 4, 1, 1, device=dev).uniform_(1 - 0.25 * scale, 1 + 0.25 * scale)
    if "nir" in parts:
        g = torch.exp(torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.4 * scale, 0.4 * scale))
        v = torch.cat([v[:, :3], v[:, 3:4] * g], 1)
    if "bias" in parts:
        v = v + torch.empty(b, 4, 1, 1, device=dev).uniform_(-0.08 * scale, 0.08 * scale)
    v = v.clamp(0, 1)
    if "gamma" in parts:
        v = v ** torch.exp(torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.223 * scale, 0.223 * scale))
    return _finish(v, valid)


class StatBank:
    """Per-tile band mean / std (tensor [N,8]) used as the distribution of acquisition conditions.

    lam   : 'u' (U(0,1)) or 'beta' (Beta(b,b): mostly keep or fully replace the style)
    indep : pick the statistics of every band from a different bank tile
    gen   : s > 0 -> sample the target statistics from a Gaussian fitted to the bank with its covariance scaled by s^2
            (s > 1 extrapolates beyond the acquisition conditions seen in the source domain)
    p     : probability that a sample is restyled at all
    """

    def __init__(self, csv_path, keep_files=None, device="cpu", lam="u", beta=0.1, indep=False, gen=0.0, p=1.0):
        df = pd.read_csv(csv_path)
        if keep_files is not None:
            df = df[df.FileName.isin(set(keep_files))]
        cols = [f"m{i}" for i in range(4)] + [f"s{i}" for i in range(4)]
        self.t = torch.tensor(df[cols].values, dtype=torch.float32, device=device)
        self.lam, self.beta, self.indep, self.gen, self.p = lam, beta, indep, gen, p
        mu = self.t.mean(0)
        cov = torch.cov(self.t.T) + 1e-6 * torch.eye(8, device=device)
        self.mu, self.chol = mu, torch.linalg.cholesky(cov)
        self.lo = torch.tensor([0.02] * 4 + [0.01] * 4, device=device)
        self.hi = torch.tensor([0.98] * 4 + [0.40] * 4, device=device)

    def __len__(self):
        return len(self.t)

    def _stats(self, b, dev):
        if self.gen > 0:
            pick = (self.mu + self.gen * (torch.randn(b, 8, device=dev) @ self.chol.T)).clamp(self.lo, self.hi)
        elif self.indep:
            idx = torch.randint(0, len(self.t), (b, 4), device=dev)
            ar = torch.arange(4, device=dev)
            pick = torch.cat([self.t[:, :4][idx, ar], self.t[:, 4:][idx, ar]], 1)
        else:
            pick = self.t[torch.randint(0, len(self.t), (b,), device=dev)]
        return pick[:, :4].reshape(b, 4, 1, 1), pick[:, 4:].reshape(b, 4, 1, 1)

    def __call__(self, x):
        b, dev = x.shape[0], x.device
        valid = _valid(x)
        v = x[:, :4]
        n = valid.sum((2, 3), keepdim=True).clamp_min(1)
        mu = (v * valid).sum((2, 3), keepdim=True) / n
        sd = ((((v - mu) ** 2) * valid).sum((2, 3), keepdim=True) / n).sqrt()
        mb, sb = self._stats(b, dev)
        if self.lam == "beta":
            lam = torch.distributions.Beta(self.beta, self.beta).sample((b, 1, 1, 1)).to(dev)
        else:
            lam = torch.rand(b, 1, 1, 1, device=dev)
        if self.p < 1.0:
            lam = lam * (torch.rand(b, 1, 1, 1, device=dev) < self.p).float()
        mt, st = lam * mb + (1 - lam) * mu, lam * sb + (1 - lam) * sd
        return _finish((v - mu) / (sd + 1e-4) * st + mt, valid)

    def nearest(self, x, lam=1.0):
        """Test-time: restyle every tile with the statistics of its nearest bank entry (in standardised 8-d space)."""
        valid = _valid(x)
        v = x[:, :4]
        n = valid.sum((2, 3), keepdim=True).clamp_min(1)
        mu = (v * valid).sum((2, 3), keepdim=True) / n
        sd = ((((v - mu) ** 2) * valid).sum((2, 3), keepdim=True) / n).sqrt()
        cur = torch.cat([mu.flatten(1), sd.flatten(1)], 1)
        scale = self.t.std(0) + 1e-6
        d = (((cur[:, None, :] - self.t[None]) / scale) ** 2).sum(-1)
        pick = self.t[d.argmin(1)]
        mb, sb = pick[:, :4].reshape(-1, 4, 1, 1), pick[:, 4:].reshape(-1, 4, 1, 1)
        mt, st = lam * mb + (1 - lam) * mu, lam * sb + (1 - lam) * sd
        return _finish((v - mu) / (sd + 1e-4) * st + mt, valid)


def rel_channels(x, k=31):
    """Two tile-relative channels: NDVI and brightness minus their local (k x k) mean over valid pixels, scaled x4."""
    import torch.nn.functional as F
    valid = _valid(x)
    pad = k // 2
    den = F.avg_pool2d(valid, k, 1, pad, count_include_pad=False).clamp_min(1e-3)
    out = []
    for ch in (x[:, 4:5], x[:, :3].mean(1, keepdim=True)):
        lm = F.avg_pool2d(ch * valid, k, 1, pad, count_include_pad=False) / den
        out.append((ch - lm) * valid * 4.0)
    return torch.cat([x] + out, 1)


def fda_aug(x, p=0.5):
    b = x.shape[0]
    if b < 2:
        return x
    valid = _valid(x)
    f = torch.fft.rfft2(x[:, :4].float())
    amp, ph = f.abs(), f.angle()
    lam = torch.rand(b, 1, 1, 1, device=x.device) * (torch.rand(b, 1, 1, 1, device=x.device) < p).float()
    amp2 = (1 - lam) * amp + lam * amp.roll(1, 0)
    v = torch.fft.irfft2(torch.polar(amp2, ph), s=x.shape[2:])
    return _finish(v, valid)


class MixStyle(torch.nn.Module):
    """Feature-level MixStyle (Zhou et al., ICLR 2021): mix the per-channel mean/std of the feature maps of different samples of a batch
    (training only).  A standard domain-generalisation baseline to compare the statistics-bank augmentation with."""

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p, self.eps = p, eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        if torch.rand(1).item() > self.p or x.shape[0] < 2:
            return x
        b = x.shape[0]
        xf = x.float()
        mu = xf.mean((2, 3), keepdim=True)
        sig = (xf.var((2, 3), keepdim=True) + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        lam = self.beta.sample((b, 1, 1, 1)).to(x.device)
        perm = torch.randperm(b, device=x.device)
        out = (xf - mu) / sig * (lam * sig + (1 - lam) * sig[perm]) + (lam * mu + (1 - lam) * mu[perm])
        return out.to(x.dtype)
