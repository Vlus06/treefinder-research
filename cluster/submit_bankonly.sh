#!/bin/bash
# Train bank-only DINOv2 (3 seeds) per geographic scenario, then run the spectral stress test on its checkpoints.
source /etc/profile.d/modules.sh; module load slurm >/dev/null 2>&1
D=/datastore/phuocnth/treefinder
for S in CO EW WE; do
  j=$(sbatch --parsable -A uit -p normal --qos=mig2-35 --gres=gpu:2g.35gb:1 --cpus-per-task=4 --mem=48G -t 3-00:00:00 \
      -J bo_$S -o $D/logs/slurm_%j.out --wrap "bash $D/air_worker_new.sh $D/air_tasks_bankonly_$S.txt")
  sbatch -A uit -p normal --qos=mig2-35 --gres=gpu:2g.35gb:1 --cpus-per-task=4 --mem=48G -t 12:00:00 \
      -J bo_stress_$S -d afterok:$j -o $D/logs/slurm_%j.out --wrap "bash $D/air_bankonly_stress.sh $S"
done
squeue -u phuocnth -o "%i %j %T %R"
