#!/bin/bash
# like air_worker.sh but uses code_air12 and forwards extra flags.  line: <config> <model> <aux> <seed> <exp_id> [extra flags...]
TASKS=$1
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1
mkdir -p $D/logs $D/out
while read -r CFG MODEL AUX SEED EXP EXTRA; do
  [ -z "$CFG" ] && continue
  key="air12_${CFG%.yaml}_${MODEL}_${AUX}_s${SEED}_${EXP}"
  [ -f "$D/out/$key.done" ] && continue
  echo "[$(date +%T)] start $key $EXTRA"
  cd $D/code_air12
  python air/main.py --config configs_hpc/$CFG --model $MODEL --aux $AUX --seed $SEED --exp_id $EXP $EXTRA \
      > $D/logs/$key.log 2>&1 \
    && { touch $D/out/$key.done; echo "[$(date +%T)] done  $key: $(grep -h 'Combined metrics' $D/logs/$key.log | tail -1 | cut -c1-700)"; } \
    || echo "[$(date +%T)] FAIL  $key"
done < "$TASKS"
echo "air12 worker finished"
