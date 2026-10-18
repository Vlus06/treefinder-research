#!/bin/bash
# usage: submit.sh <tasks_file> <qos> <gres>      (one job, one process at a time)
source /etc/profile.d/modules.sh; module load slurm >/dev/null 2>&1
TASKS=$1; QOS=$2; GRES=$3; SLOTS=1; N=1
CPUS=$((4*SLOTS))
for w in $(seq 1 $N); do
sbatch -A uit -p normal --qos=$QOS --gres=gpu:$GRES:1 --cpus-per-task=$CPUS --mem=$((24*SLOTS))G -t 3-00:00:00 \
  -J tf_$(basename $TASKS .txt)_$QOS -o ~/tf/logs/slurm_%j.out --wrap "bash ~/tf/cluster/worker.sh $TASKS $SLOTS"
done
