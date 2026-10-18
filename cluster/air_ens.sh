#!/bin/bash
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1 PYTHONPATH=$D/code_air11
cd $D/code_air11
python air/ensemble_eval.py --logs $D/logs --ckpts $D/code_air/checkpoints $D/code_air2/checkpoints $D/code_air3/checkpoints $D/code_air7/checkpoints \
   --configs_dir configs_hpc --out $D/ensemble_results.json --workers 8
python air/ensemble_eval.py --logs $D/logs --ckpts $D/code_air/checkpoints $D/code_air2/checkpoints $D/code_air3/checkpoints $D/code_air7/checkpoints \
   --configs_dir configs_hpc --out $D/ensemble_results_tta.json --workers 8 --tta --splits tile
echo "ens finished"
