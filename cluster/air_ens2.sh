#!/bin/bash
# seed ensembles (+ flip TTA) of finished DINOv2 / TreeMortSeg runs on the three cross-region scenarios
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1
cd $D/code_air12
for sc in "res_state_CO co_dino_plain co_dino_method co_tms_fixed_w25" "res_state_EW ew_dino_plain ew_dino_method ew_tms_fixed" "res_state_WE we_dino_plain we_dino_method we_tms_fixed"; do
  set -- $sc; cfg=$1; shift
  python air/seed_ens_eval.py --logs $D/logs --ckpts $D/code_air9/checkpoints $D/code_air12/checkpoints --configs_dir configs_hpc \
    --config_name $cfg --tags "$@" --models dinov2_b treemortseg --out $D/seed_ens_$cfg.json
done
