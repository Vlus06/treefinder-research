#!/bin/bash
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1
cd $D/code_air12
CK=checkpoints
for T in co_fz001 ew_fz001 we_fz001 site_dino_fz001 tile3_dino_fz001; do
  python air/eval_extra.py --logs $D/logs --ckpts $CK --configs_dir configs_hpc --out $D/eval_extra_dino --only "_dinov2_b-$T" --workers 4
done
python air/error_objects.py --evals $D/eval_extra_dino --ckpts $CK --configs_dir configs_hpc --out $D/error_objects_dino.json --workers 4   --only "_dinov2_b-co_fz001" "_dinov2_b-ew_fz001" "_dinov2_b-we_fz001" "_dinov2_b-site_dino_fz001" "_dinov2_b-tile3_dino_fz001"
echo "diag finished"
