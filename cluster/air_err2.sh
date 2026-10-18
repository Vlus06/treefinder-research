#!/bin/bash
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1 PYTHONPATH=$D/code_air8
cd $D/code_air8
mkdir -p $D/records $D/pred_dump
python air/error_objects.py --evals $D/eval_extra --ckpts $D/code_air/checkpoints $D/code_air2/checkpoints $D/code_air3/checkpoints \
  --configs_dir configs_hpc --out $D/error_objects2.json --workers 8 --records $D/records/err --dump $D/pred_dump --n_dump 60 \
  --only "_convnext_upernet" "_treemortseg-none" "_deeplabv3" "_unet"
echo "err2 finished"
