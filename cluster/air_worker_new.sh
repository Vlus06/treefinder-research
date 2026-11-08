#!/bin/bash
# air/ pipeline worker in the rebuilt environment.  line: <config> <model> <aux> <seed> <exp_id> [extra flags...]
TASKS=$1
D=/datastore/phuocnth/treefinder
PY=$D/env/bin/python
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p $D/logs $D/out
while read -r CFG MODEL AUX SEED EXP EXTRA; do
  [ -z "$CFG" ] && continue
  key="airN_${CFG%.yaml}_${MODEL}_${AUX}_s${SEED}_${EXP}"
  [ -f "$D/out/$key.done" ] && continue
  echo "[$(date +%T)] start $key $EXTRA"
  cd $D/code_air
  $PY air/main.py --config configs_hpc/$CFG --model $MODEL --aux $AUX --seed $SEED --exp_id $EXP --workers 4 $EXTRA \
      > $D/logs/$key.log 2>&1 \
    && { touch $D/out/$key.done; echo "[$(date +%T)] done  $key"; } \
    || echo "[$(date +%T)] FAIL  $key"
done < "$TASKS"
echo "air worker finished"
