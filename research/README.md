# TreeFinder research experiments

This folder adds site-disjoint evaluation and a small-label transfer experiment to the [official TreeFinder code](https://github.com/zhwang0/treefinder). The original code and CC BY 4.0 attribution are retained. The raw TreeFinder dataset is distributed separately by its authors on [Hugging Face](https://huggingface.co/datasets/zhwang1/TreeFinder).

## Research question

When a dead-tree segmenter is moved to a previously unseen region, which target sites should be labeled first to maximize detection performance per labeled site? The `active_transfer.py` experiment compares random selection, prediction uncertainty, spectral diversity, and a 50/50 combination. Every strategy starts from the same source checkpoint and receives the same number of labeled target sites. Target test masks are used only for final evaluation.

The acquisition score uses the model's top 1% most uncertain valid pixels from up to four tiles per site. Spectral diversity uses site means and standard deviations of RGB, NIR, and NDVI. These are **initial hypotheses**, not established improvements.

## Kaggle T4

Create a Kaggle notebook with **GPU T4** and **Internet on**, then run:

```bash
!git clone https://github.com/Vlus06/treefinder-research.git
%cd /kaggle/working/treefinder-research
!bash research/setup_kaggle.sh
!python research/train_baseline.py --epochs 15 --batch-size 8
!python research/active_transfer.py --direction west_to_east --budgets 5 10 20 --tiles-per-site 4 --batch-size 8
!python research/active_transfer.py --direction east_to_west --budgets 5 10 20 --tiles-per-site 4 --batch-size 8
```

Download `research/reports/` from the Kaggle notebook's output files after execution. The download is about 2 GB compressed; extraction needs roughly 4 GB more. `setup_kaggle.sh` downloads only the three dataset files required for these experiments and ignores macOS metadata in the ZIP.

With the downloaded results ZIP on a machine that also has the dataset metadata, run `python research/analyze_selection.py /path/to/treefinder_results.zip` to compare how many positive sites, positive pixels, and tiles each strategy acquired.
For a new run, use `python research/bootstrap_sites.py research/reports/active_transfer/west_to_east/seed2025/source_replay_tiles4_random_10_by_site.csv` to calculate a site-cluster bootstrap confidence interval for F1. Each active-transfer run saves per-site confusion counts for the fixed 0.5 threshold and source-validation-calibrated threshold.

For a quick pipeline check before the full run:

```bash
!python research/train_baseline.py --smoke
!python research/active_transfer.py --smoke --budgets 1 --max-score-tiles 1
```

Smoke metrics are **not scientific results**: they come from one epoch and deliberately small, label-stratified samples.

If continuing an earlier Kaggle session, run `!git pull` inside `/kaggle/working/treefinder-research` before the commands above. Re-run the baseline and **both** transfer directions: older checkpoints were selected by lowest validation loss and sometimes predicted only background. The scripts now select the epoch with highest source-validation F1 at threshold 0.5, breaking ties by validation loss. They save `best_f1.pt`, `source_model_f1.pt`, and `pool_scores_f1.json` so old checkpoints and scores are not reused. `--tiles-per-site 4` makes the labeled tile count equal for all acquisition strategies; omit it to reproduce the earlier whole-site labeling experiment. `--no-source-replay` reproduces target-only fine-tuning. The code saves these runs to separate JSON files.

To prepare additional splits, run `TREEFINDER_SEEDS="2025 2026 2027" bash research/setup_kaggle.sh`, then pass `--seed 2026` or `--seed 2027` to each training command. Source checkpoints and results are stored in separate seed folders.

## Data and split audit

The official random loader defaults to shuffling individual `FileName` tiles. Its test set can therefore contain tiles from source images also seen in train/validation. `make_site_split.py` groups by `ImageRawPath`, stratifies sites by state and presence of positives, then emits a 70/10/20 train/validation/test split. It also writes `research/reports/split_audit.json`, which reproduces the official random loader's site overlap for seed 2025. This audit proves overlap, not the magnitude of its effect on model scores.

`make_transfer_split.py` creates east-to-west and west-to-east experiments using the state groups in the authors' configuration. Within each direction, the source gets disjoint train/validation sites and the target gets disjoint acquisition-pool/test sites. The seed-2025 site manifests are committed for reproducibility.

The baseline is a deliberately small U-Net (`[16, 32, 64]` channels) with five inputs (RGB, NIR, computed NDVI), masked BCE plus Dice loss, and global positive-class pixel metrics. This is not numerically identical to the larger U-Net in the paper. The no-data mask is applied to *both* prediction and target terms in Dice loss. Reports include test confusion counts for later site-level bootstrap intervals. The new reports also evaluate a threshold selected on **source validation** data, never on target test data.

## Reporting before submission

Run at least three seeds for full claims; seed 2025 is only the first experiment. Compare each method at the same number of labeled sites and tiles (use `--tiles-per-site 4`), and report labeled valid hectares and positive sites. The source-replay and target-only fine-tuning modes should be treated as an ablation. Add site-bootstrap confidence intervals and inspect false positives per valid hectare before making deployment claims. Pixel F1 is not instance F1; TreeFinder's raster masks can merge neighboring crowns.
