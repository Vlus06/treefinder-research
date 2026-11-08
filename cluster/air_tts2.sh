#!/bin/bash
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1
cd $D/code_air12
for sc in "res_state_CO co_dino" "res_state_EW ew_dino" "res_state_WE we_dino"; do
  set -- $sc
  python air/tts_eval.py --logs $D/logs --ckpts $D/code_air9/checkpoints $D/code_air12/checkpoints --configs_dir configs_hpc --config_name $1 \
    --models dinov2_b --tags $2_plain $2_method --workers 4 --out $D/tts_dino_$1.json
done
