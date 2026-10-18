# air/losses.py -- EdgeBoundaryLoss / auxiliary losses.
# "released" follows RONGHUA-LI/TreeMortSeg utils/losses.py (MIT, (c) 2026 Ronghua Li) including its quirks;
# "fixed" is the loss as written in the paper (standard Dice on a 0/1 edge map + distance-weighted edge penalty).
import torch
import torch.nn as nn


class DiceNoFactor(nn.Module):
    """inter / union, as in the TreeFinder / TreeMortSeg code (no factor 2)."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets, mask=None):
        p = torch.sigmoid(logits)
        if mask is not None:
            p = p * mask
        p, t = p.reshape(-1), targets.float().reshape(-1)
        return 1 - ((p * t).sum() + self.eps) / (p.sum() + t.sum() + self.eps)


class DiceStd(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets, mask=None):
        p = torch.sigmoid(logits)
        t = targets.float()
        if mask is not None:
            p, t = p * mask, t * mask
        p, t = p.reshape(-1), t.reshape(-1)
        return 1 - (2 * (p * t).sum() + self.eps) / (p.sum() + t.sum() + self.eps)


class EdgeBoundaryLoss(nn.Module):
    def __init__(self, lambda_boundary=0.2, max_dist=10.0, standard_dice=False):
        super().__init__()
        self.dice = DiceStd() if standard_dice else DiceNoFactor()
        self.lam, self.max_dist = lambda_boundary, max_dist

    def forward(self, logits, gts, dist_maps, mask):
        loss_dice = self.dice(logits, gts, mask)
        probs = torch.sigmoid(logits) * mask
        d = torch.clamp(dist_maps, 0, self.max_dist) / self.max_dist * mask
        return loss_dice + self.lam * (probs * d).sum() / (mask.sum() + 1e-6)


def dist_loss(out_dist_logit, dist_target, positive_mask=None):
    """L1 between sigmoid(prediction) and the normalised distance map, restricted to positive pixels."""
    if positive_mask is None:
        positive_mask = (dist_target > 0).float()
    l1 = (torch.sigmoid(out_dist_logit) - dist_target).abs() * positive_mask
    return l1.sum() / (positive_mask.sum() + 1e-8)
