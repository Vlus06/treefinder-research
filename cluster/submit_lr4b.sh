#!/bin/bash
# Submit the lr 1e-4 re-runs of the dataset baselines (random and site-disjoint, 6 models x 3 seeds) in the rebuilt environment.
source /etc/profile.d/modules.sh; module load slurm >/dev/null 2>&1
D=/datastore/phuocnth/treefinder
for t in rand_s2025 rand_s2024 rand_s2023 site_s2025 site_s2024 site_s2023; do
  sbatch -A uit -p normal --qos=mig2-35 --gres=gpu:2g.35gb:1 --cpus-per-task=4 --mem=48G -t 3-00:00:00 \
    -J lr4_$t -o $D/logs/slurm_%j.out \
    --wrap "bash $D/official_worker2.sh $D/official_tasks_lr4_$t.txt"
done
squeue -u phuocnth -o "%i %j %T %R"
