#!/bin/bash
# per-tile counts of the DINOv2 runs on the random split (plain vs method, all seeds) -> tile_stats_random.csv
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1
cd $D/code_air12
EXPS=$(python - <<PY
import re,ast,glob
tags={"tile_dino_plain","tile3_dino_plain","tile_dino_method","tile3_dino_method"}
out=[]
for f in glob.glob("$D/logs/air12_res_random_dinov2_b_*.log"):
    t=open(f,errors="ignore").read(); m=re.search(r"args=(\{.*\})",t)
    if not m or "Combined metrics" not in t: continue
    a=ast.literal_eval(m.group(1))
    if a["tag"] in tags and a.get("split_seed")==2025: out.append("air%s_dinov2_b-%s"%(str(a["exp_id"]).zfill(4),a["tag"]))
print(" ".join(sorted(out)))
PY
)
echo "runs: $EXPS"
python air/tile_stats.py --config configs_hpc/res_random.yaml --split_seed 2025 --out $D/tile_stats_random.csv --exps $EXPS --ckpt_dirs $D/code_air12/checkpoints
