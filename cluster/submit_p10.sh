#!/bin/bash
# Re-run the six dataset baselines with patience 10 (lr 1e-4) in all five scenarios: one job per training seed.
source /etc/profile.d/modules.sh; module load slurm >/dev/null 2>&1
D=/datastore/phuocnth/treefinder
for s in 2025 2024 2023; do
  sbatch -A uit -p normal --qos=mig2-35 --gres=gpu:2g.35gb:1 --cpus-per-task=4 --mem=48G -t 3-00:00:00 \
    -J p10_s$s -o $D/logs/slurm_%j.out --wrap "bash $D/official_worker2.sh $D/official_tasks_p10_s$s.txt"
done
squeue -u phuocnth -o "%i %j %T %R"
