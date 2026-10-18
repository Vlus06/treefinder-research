#!/bin/bash
# Re-run the evaluation for a subset of runs, this time keeping the per-site score histograms.
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1 PYTHONPATH=$D/code_air5
cd $D/code_air5
for pat in convnext_upernet treemortseg-none treemortseg-fixed deeplabv3 unet; do
  python air/eval_extra.py --logs $D/logs --ckpts $D/code_air/checkpoints $D/code_air2/checkpoints $D/code_air3/checkpoints \
      --configs_dir configs_hpc --out $D/eval_hist --workers 8 --only "_$pat" 
done
echo "eval2 finished"
