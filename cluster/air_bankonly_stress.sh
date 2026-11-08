#!/bin/bash
# Spectral stress test (13 conditions, no retraining) for the bank-only DINOv2 runs of one geographic scenario.
S=$1
D=/datastore/phuocnth/treefinder
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
cd $D/code_air
$D/env/bin/python air/style_shift_eval.py --logs $D/logs --ckpts checkpoints --configs_dir configs_hpc --config_name res_state_$S \
  --group bankonly=$(echo $S | tr A-Z a-z)_lr001_bank --models dinov2_b --workers 4 --out $D/style_shift_bankonly_$S.json
