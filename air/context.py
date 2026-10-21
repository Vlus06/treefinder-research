"""Spatial-context input: each sample is the 224 px tile plus a `margin` px border taken from its neighbouring tiles.

Tiles are contiguous 224 px (134.4 m) blocks of one source image (see research/tile_geo.py), so the border is real imagery.
The label / no-data mask are padded so that only the central tile is scored (border pixels are flagged no-data), hence the
benchmark metric (exps/evaluate.py) and the loss still look at exactly the same pixels as the plain runs.
"""
import os
import random

import numpy as np
import pandas as pd
import torch

from air.dataset import AirDataset
from data_loader.utils import load_tile

T = 224


class ContextAirDataset(AirDataset):
    def __init__(self, cfg, split, aux_mode="none", margin=112):
        super().__init__(cfg, split, aux_mode)
        self.margin, self.split, self.cfg = margin, split, cfg
        geo = pd.read_csv(os.path.join(cfg["data"]["root_dir"], "tile_geo.csv"))
        self.nb = {r["FileName"]: [r[f"nb_{dr}_{dc}"] if isinstance(r[f"nb_{dr}_{dc}"], str) else "" for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                                   if not (dr == 0 and dc == 0)] for _, r in geo.iterrows()}
        self.offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]
        self.dir = os.path.dirname(self.base.paths[0])

    def __getitem__(self, i):
        path = self.base.paths[i]
        name = os.path.basename(path)
        img, lab, nod = load_tile(path, no_data_value=self.cfg["data"]["no_data_value"], normalize=self.cfg["data"]["normalize"])
        m = self.margin
        S = T + 2 * m
        x = np.zeros((5, S, S), np.float32)
        y = np.zeros((S, S), np.uint8)
        nd = np.ones((S, S), bool)
        x[:, m:m + T, m:m + T], y[m:m + T, m:m + T], nd[m:m + T, m:m + T] = img, lab, nod
        for (dr, dc), nb in zip(self.offsets, self.nb[name]):
            if not nb:
                continue
            r0, c0 = m + dr * T, m + dc * T                        # top-left of the neighbour in crop coordinates
            ra, rb, ca, cb = max(r0, 0), min(r0 + T, S), max(c0, 0), min(c0 + T, S)
            if ra >= rb or ca >= cb:
                continue
            nimg, _, _ = load_tile(os.path.join(self.dir, nb), no_data_value=self.cfg["data"]["no_data_value"], normalize=self.cfg["data"]["normalize"])
            x[:, ra:rb, ca:cb] = nimg[:, ra - r0:rb - r0, ca - c0:cb - c0]
        if self.split == "train":
            aug = self.cfg["data"]["augmentation"]
            if aug.get("random_flip", False):
                if random.random() < 0.5:
                    x, y, nd = x[:, :, ::-1], y[:, ::-1], nd[:, ::-1]
                if random.random() < 0.5:
                    x, y, nd = x[:, ::-1, :], y[::-1, :], nd[::-1, :]
            if aug.get("rotation", {}).get("type") == "90":
                k = random.choice([0, 1, 2, 3])
                x, y, nd = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k), np.rot90(nd, k)
        return {"image": torch.from_numpy(np.ascontiguousarray(x)).float(), "label": torch.from_numpy(np.ascontiguousarray(y)).long(),
                "no_data_mask": torch.from_numpy(np.ascontiguousarray(nd)).bool(),
                "cls_label": torch.tensor(bool((y > 0).any()), dtype=torch.long),
                "site": torch.tensor(self.site_ids[i], dtype=torch.long)}
