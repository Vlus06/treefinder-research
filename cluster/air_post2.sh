#!/bin/bash
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1
cd $D/code_air12
F=$D/find_exps.py
CK=checkpoints
# 1) controlled colour shifts for West->East and site-disjoint
python air/style_shift_eval.py --logs $D/logs --ckpts $CK --configs_dir configs_hpc --config_name res_state_WE   --group plain=we_lr001_p_ck --group kaleido=we_fz001 --models dinov2_b --out $D/style_shift_WE.json
python air/style_shift_eval.py --logs $D/logs --ckpts $CK --configs_dir configs_hpc --config_name res_random_site   --group plain=site_lr001_p_ck --group kaleido=site_dino_fz001 --models dinov2_b --out $D/style_shift_Site.json
# 2) per-tile statistics and the style-distance analysis for the four scenarios
for S in "CO res_state_CO co_lr001_p_ck co_fz001" "EW res_state_EW ew_lr001_p_ck ew_fz001" "WE res_state_WE we_lr001_p_ck we_fz001" "Site res_random_site site_lr001_p_ck site_dino_fz001"; do
  set -- $S
  P=$(python $F $2 $3); K=$(python $F $2 $4)
  echo "== $1 baseline: $P | kaleido: $K"
  python air/tile_stats.py --config configs_hpc/$2.yaml --split_seed 2025 --out $D/tile_stats_$1.csv --exps ${P//,/ } ${K//,/ } --ckpt_dirs $CK --workers 4
  python research/style_distance_analysis.py --config configs_hpc/$2.yaml --split_seed 2025 --tile_stats $D/tile_stats_$1.csv --group baseline=$P --group kaleido=$K --out $D/style_distance_$1.json
done
echo "post2 finished"
