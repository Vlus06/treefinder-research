#!/bin/bash
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1
cd $D/code_air12
python air/tile_stats.py --config configs_hpc/res_random.yaml --split_seed 2025 --out $D/tile_stats_2025.csv \
  --exps air6007_convnext_upernet air6008_treemortseg-none air8028_dinov2_b-base \
  --ckpt_dirs $D/code_air/checkpoints $D/code_air7/checkpoints
