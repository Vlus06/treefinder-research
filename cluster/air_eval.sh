#!/bin/bash
# Re-evaluate every saved air/ checkpoint (AP, F1@val-threshold, per-site counts).
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1 PYTHONPATH=$D/code_air4
cd $D/code_air4
python air/eval_extra.py --logs $D/logs --ckpts $D/code_air/checkpoints $D/code_air2/checkpoints $D/code_air3/checkpoints \
    --configs_dir configs_hpc --out $D/eval_extra --workers 8
echo "eval finished"
