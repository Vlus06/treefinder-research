#!/bin/bash
# Run the authors' unmodified main.py, one task at a time.  task line: <code_dir> <config> <exp_id> <model> <seed>
TASKS=$1
D=/datastore/phuocnth/treefinder
source ~/tfenv/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1
mkdir -p $D/logs
while read -r CODE CFG EXP MODEL SEED; do
  [ -z "$CODE" ] && continue
  key="${CODE}_${CFG%.yaml}_${EXP}_${MODEL}_s${SEED}"
  [ -f "$D/out/$key.done" ] && continue
  echo "[$(date +%T)] start $key"
  cd $D/$CODE
  python main.py --config configs_hpc/$CFG --exp_id $EXP --train_ratio 0.8 --random_seed $SEED \
      --model_name $MODEL --gpu_id 0 --overwrite_cfg True > $D/logs/$key.log 2>&1 \
    && { touch $D/out/$key.done; echo "[$(date +%T)] done  $key: $(grep -h 'Combined metrics' $D/logs/$key.log | tail -1 | cut -c1-600)"; } \
    || echo "[$(date +%T)] FAIL  $key"
done < "$TASKS"
echo "official worker finished"
