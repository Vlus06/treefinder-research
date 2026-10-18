#!/bin/bash
# Run air/main.py tasks one at a time.  task line: <config> <model> <aux> <seed> <exp_id>
TASKS=$1
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1
mkdir -p $D/logs $D/out
while read -r CFG MODEL AUX SEED EXP; do
  [ -z "$CFG" ] && continue
  key="air_${CFG%.yaml}_${MODEL}_${AUX}_s${SEED}_${EXP}"
  [ -f "$D/out/$key.done" ] && continue
  echo "[$(date +%T)] start $key"
  cd $D/code_air
  python air/main.py --config configs_hpc/$CFG --model $MODEL --aux $AUX --seed $SEED --exp_id $EXP \
      > $D/logs/$key.log 2>&1 \
    && { touch $D/out/$key.done; echo "[$(date +%T)] done  $key: $(grep -h 'Combined metrics' $D/logs/$key.log | tail -1 | cut -c1-700)"; } \
    || echo "[$(date +%T)] FAIL  $key"
done < "$TASKS"
echo "air worker finished"
