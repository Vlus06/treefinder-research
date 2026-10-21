#!/bin/bash
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1
cd $D/code_air12
python air/tile_stats.py --config configs_hpc/res_state_CO.yaml --split_seed 2025 --out $D/tile_stats_CO.csv \
  --exps air9507_convnext_upernet-convnext air9902_convnext_upernet-co_photo air9330_convnext_upernet-co_bank_src_cons air9950_treemortseg-co_tms_fixed_w25 \
  --ckpt_dirs $D/code_air9/checkpoints $D/code_air12/checkpoints
