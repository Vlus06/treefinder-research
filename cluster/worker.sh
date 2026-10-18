#!/bin/bash
# Pull trainer command lines from a task file until none are left. Several workers can share one file.
# usage: worker.sh <tasks_file> [parallel_slots]
set -u
TASKS=$1; SLOTS=${2:-1}
source ~/tfenv/bin/activate
export PYTHONPATH=$HOME/tf HF_HOME=$HOME/hf_cache OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
LOCAL=/tmp/tf_packed_$USER
mkdir -p $LOCAL ~/tf/runs ~/tf/locks ~/tf/logs
( flock 9; if [ ! -f $LOCAL/ok ]; then cp ~/tf/datasets/packed/x.npy ~/tf/datasets/packed/y.npy $LOCAL/ && touch $LOCAL/ok; fi ) 9>$LOCAL.lock
run_one() {
  local n=$1 args=$2 key
  key=$(echo "$args" | md5sum | cut -c1-12)
  mkdir "$HOME/tf/locks/$key" 2>/dev/null || return 0     # claimed by another worker
  echo "[$(date +%T)] start $n: $args"
  python ~/tf/research/trainer.py --data-dir $LOCAL --out ~/tf/runs $args > ~/tf/logs/$key.log 2>&1 \
    && echo "[$(date +%T)] done  $n: $(grep RESULT ~/tf/logs/$key.log | tail -1)" \
    || { echo "[$(date +%T)] FAIL  $n: $args"; rmdir "$HOME/tf/locks/$key"; }
}
i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  i=$((i+1))
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do sleep 5; done
  run_one $i "$line" &
done < "$TASKS"
wait
echo "worker finished"
