#!/bin/bash
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1 PYTHONPATH=$D/code_air6
cd $D/code_air6
python air/error_objects.py --evals $D/eval_extra --ckpts $D/code_air/checkpoints $D/code_air2/checkpoints $D/code_air3/checkpoints \
  --configs_dir configs_hpc --out $D/error_objects.json --workers 8 \
  --only "_convnext_upernet" "_treemortseg-none" "_treemortseg-fixed" "_deeplabv3" "_unet"
echo "err finished"
