#!/bin/bash
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1
cd $D/code_air12
CK=checkpoints
for T in co_lr001_p_ck ew_lr001_p_ck; do
  python air/eval_extra.py --logs $D/logs --ckpts $CK --configs_dir configs_hpc --out $D/eval_extra_dino --only "_dinov2_b-$T" --workers 4
done
python air/error_objects.py --evals $D/eval_extra_dino --ckpts $CK --configs_dir configs_hpc --out $D/error_objects_dino_plain.json --workers 4   --only "_dinov2_b-co_lr001_p_ck" "_dinov2_b-ew_lr001_p_ck"
echo "diag2 finished"
