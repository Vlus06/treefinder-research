# air/dataset.py -- wraps the benchmark's own dataset classes and attaches auxiliary targets.
import numpy as np
import torch
from torch.utils.data import Dataset

from .aux_labels import make_aux


class AirDataset(Dataset):
    def __init__(self, cfg: dict, split: str, aux_mode: str = "none"):
        method = cfg["data"]["split"]["method"]
        if method == "random":
            from data_loader.random_split import RandomSplitDataset as DS
        elif method == "by_site":
            from data_loader.site_split import SiteSplitDataset as DS
        elif method == "by_state":
            from data_loader.by_state_split import StateSplitDataset as DS
        elif method == "by_climate":
            from data_loader.by_climate_split import ClimateSplitDataset as DS
        elif method == "by_tree_type":
            from data_loader.by_tree_split import TreeSplitDataset as DS
        else:
            raise ValueError(method)
        self.base = DS(cfg, split=split)
        self.aux_mode = aux_mode
        # site (source image) index of every tile, used for site-grouped sampling
        import os
        import pandas as pd
        aux = pd.read_csv(os.path.join(cfg["data"]["root_dir"], "tile_info224_v3_aux.csv"), usecols=["FileName", "ImageRawPath"])
        codes = {p: i for i, p in enumerate(sorted(aux.ImageRawPath.unique()))}
        by_file = dict(zip(aux.FileName, aux.ImageRawPath.map(codes)))
        self.site_ids = [by_file[os.path.basename(p)] for p in self.base.paths]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        item["site"] = torch.tensor(self.site_ids[i], dtype=torch.long)
        if self.aux_mode != "none":
            edge, dist, bd = make_aux(item["label"].numpy(), item["no_data_mask"].numpy(), self.aux_mode)
            item["edge"], item["dist"], item["boundary_dist"] = (
                torch.from_numpy(np.ascontiguousarray(a)).float() for a in (edge, dist, bd))
        return item
