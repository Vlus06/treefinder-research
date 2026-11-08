#!/bin/bash
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache
cd $D/code_air12
for S in CO EW; do
  python air/style_shift_eval.py --logs $D/logs --ckpts checkpoints --configs_dir configs_hpc --config_name res_state_$S     --group plain=$(echo $S | tr A-Z a-z)_lr001_p_ck --group kaleido=$(echo $S | tr A-Z a-z)_fz001 --models dinov2_b --out $D/style_shift_$S.json
done
