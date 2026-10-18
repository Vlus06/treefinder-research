# File: data_loader/site_split.py
# Additive split (not in the original benchmark): site-disjoint random split.
# All tiles cut from the same source image (ImageRawPath) go to the same partition, so that no test tile has a
# neighbour from the same image in train/val.  Manifest: research/splits/site_seed<seed>.csv (see
# research/make_site_split.py; stratified by state and presence of dead trees, ~70/10/20 of sites).
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from .utils import load_tile, augment_tile


class SiteSplitDataset(Dataset):
    def __init__(self, cfg: dict, split: str = 'train'):
        self.cfg = cfg
        self.split = split
        seed = cfg['experiment']['seed']
        manifest = cfg['data']['split'].get('site_manifest_dir', 'research/splits')
        df = pd.read_csv(os.path.join(manifest, f'site_seed{seed}.csv'), usecols=['FileName', 'split'])
        df = df[df['split'] == split]
        base = os.path.join(cfg['data']['root_dir'], cfg['data']['dataset_dir'])
        self.paths = [os.path.join(base, f) for f in df['FileName']]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img, lab, no_data_mask = load_tile(self.paths[idx], no_data_value=self.cfg['data']['no_data_value'],
                                           normalize=self.cfg['data']['normalize'])
        if self.split == 'train':
            img, lab, no_data_mask = augment_tile(img, lab, no_data_mask, self.cfg['data']['augmentation'])
        return {
            'image': torch.from_numpy(img).float(),
            'label': torch.from_numpy(lab).long(),
            'no_data_mask': torch.from_numpy(no_data_mask).bool(),
            'cls_label': torch.tensor(bool((lab > 0).any()), dtype=torch.long),
        }
