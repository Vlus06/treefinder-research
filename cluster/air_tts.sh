#!/bin/bash
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1
cd $D/code_air12
python air/tts_eval.py --logs $D/logs --ckpts $D/code_air9/checkpoints $D/code_air12/checkpoints --configs_dir configs_hpc --config_name res_state_CO \
  --tags convnext co_base co_photo co_bank_src co_bank_src_cons co_mixstyle --out $D/tts_CO.json
