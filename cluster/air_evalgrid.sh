#!/bin/bash
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1 PYTHONPATH=$D/code_air11
cd $D/code_air11
python air/eval_extra.py --logs $D/logs --ckpts $D/code_air10/checkpoints --configs_dir configs_hpc --out $D/eval_grid --workers 8 --only "_m_"
echo "evalgrid finished"
