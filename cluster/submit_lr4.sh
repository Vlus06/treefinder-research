#!/bin/bash
# Re-run the dataset baselines (authors' code) on the random and site-disjoint splits with lr 1e-4 (the value the TreeFinder paper reports).
# Creates the lr 1e-4 config copies, then submits one job per (split, seed) with the six baselines run one after the other.
source /etc/profile.d/modules.sh; module load slurm >/dev/null 2>&1
D=/datastore/phuocnth/treefinder
cp -n $D/code_head/configs_hpc/res_random.yaml $D/code_head/configs_hpc/res_random_lr4.yaml
sed -i 's/learning_rate: 1e-5/learning_rate: 1e-4/' $D/code_head/configs_hpc/res_random_lr4.yaml
cp -n $D/code_site/configs_hpc/res_random_site_fix.yaml $D/code_site/configs_hpc/res_random_site_fix_lr4.yaml
sed -i 's/learning_rate: 1e-5/learning_rate: 1e-4/' $D/code_site/configs_hpc/res_random_site_fix_lr4.yaml
grep -H "learning_rate" $D/code_head/configs_hpc/res_random_lr4.yaml $D/code_site/configs_hpc/res_random_site_fix_lr4.yaml
for t in rand_s2025 rand_s2024 rand_s2023 site_s2025 site_s2024 site_s2023; do
  sbatch -A uit -p normal --qos=mig2-35 --gres=gpu:2g.35gb:1 --cpus-per-task=4 --mem=48G -t 3-00:00:00 \
    -J lr4_$t -o $D/logs/slurm_%j.out \
    --wrap "bash $D/official_worker.sh $D/official_tasks_lr4_$t.txt"
done
squeue -u phuocnth -o "%i %j %T %R"
