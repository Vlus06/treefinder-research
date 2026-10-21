# TreeFinder research fork: reproduction, analysis and a competitive single-model recipe

Fork of [zhwang0/treefinder](https://github.com/zhwang0/treefinder) (NeurIPS 2025 D&B, dead-tree segmentation on NAIP tiles).
It adds (1) a **faithful reproduction** of the dataset paper's baselines with the authors' unmodified code, (2) a separate
pipeline (`air/`) for **stronger baselines and new methods**, evaluated on the *same* splits, loaders and metric, and
(3) analysis scripts and the cluster files used to run everything.
The authors' original README is kept in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

> Status: work in progress (course project / conference paper draft). All numbers are our own re-runs; see section 5 for scope and caveats.

## 1. What is whose (provenance)

| Part | Origin |
|---|---|
| `main.py`, `exps/`, `models/` (except `mask2former_rgb.py`), `data_loader/` (except `site_split.py`; 2 added lines in `__init__.py`), `utils/`, `scripts/`, `configs/res_*.yaml` (except `res_random_site.yaml`) | **Authors' code (upstream), unchanged.** Reproduction runs use it as is. |
| `models/mask2former_rgb.py` | Authors' Mask2Former from before commit `274e112` (RGB-only), to match the code that existed when the paper was written |
| `air/models.py` (TreeMortSeg, ConvNeXt+UPerNet) | TreeMortSeg architecture copied from [RONGHUA-LI/TreeMortSeg](https://github.com/RONGHUA-LI/TreeMortSeg) (MIT, 2026); only `ConvNeXtUPerNet` and the builder are new |
| everything else in `air/`, `cluster/`, `research/`, `data_loader/site_split.py`, `configs/res_random_site.yaml` | **This project** |

Upstream license: CC BY 4.0 (see `LICENSE`). Dataset: [Hugging Face zhwang1/TreeFinder](https://huggingface.co/datasets/zhwang1/TreeFinder) (not included).

## 2. Repository tree

```
.
├── main.py, exps/, models/, data_loader/, utils/, scripts/   authors' training / evaluation code (upstream)
├── configs/                  authors' YAML configs per scenario (+ res_random_site.yaml: site-disjoint split, ours)
│
├── air/                      OUR pipeline: same splits/loaders/metric as upstream, different models and recipe
│   ├── main.py               train + evaluate one run (all options below)
│   ├── models.py             ConvNeXt-S+UPerNet, TreeMortSeg (MIT, from RONGHUA-LI/TreeMortSeg)
│   ├── models_extra.py       SegFormer-B5, UPerNet-Swin-S, EfficientUNet-B7, DINOv2-B + conv decoder, HRNet-W32
│   ├── models_modular.py     backbone x decoder grid (m_<backbone>_<decoder>)
│   ├── spectral_aug.py       PROPOSED METHOD: statistics-bank augmentation (+ photometric, Fourier-mix, MixStyle, relative channels)
│   ├── distill.py            DINOv2 alignment, teacher loading, copy-paste, tile gate (negative results)
│   ├── context.py, models_ctx.py   neighbouring-tile context input / two-scale network (helps only on the leaky tile split)
│   ├── adabn_eval.py, tts_eval.py  test-time adaptation (AdaBN, test-time style normalisation) of finished runs
│   ├── tile_stats.py         per-tile tp/fp/fn of finished runs (regime and per-state analysis)
│   ├── dataset.py, losses.py, aux_labels.py   dataset wrapper (site ids, aux targets), losses, aux-label builders
│   ├── eval_extra.py         AP, F1 at validation threshold, per-site tp/fp/fn
│   ├── ensemble_eval.py      probability-averaging ensembles of finished runs
│   └── error_objects.py      crown-level error anatomy (size bins, missed crowns, false-positive objects)
│
├── data_loader/site_split.py site-disjoint split (no source image shared between train and test)
├── cluster/                  Slurm workers + task lists used on the UIT H200 cluster (paths are examples; edit them)
│   ├── official_worker.sh    runs the authors' main.py for a task list
│   ├── air_worker*.sh        runs air/main.py for a task list (one line = one run)
│   └── air_tasks_*.txt       the exact experiment lists (config model aux seed exp_id [flags])
│
├── research/                 analysis, splits and documents
│   ├── REPRODUCTION.md       how the paper baselines were reproduced, results, known differences
│   ├── METHOD_DESIGN.md      hypotheses and method design notes
│   ├── make_official_splits.py   exports the authors' splits (per seed) as CSV manifests
│   ├── splits/               split manifests (official_*_seed*.csv, site_seed*.csv, ...)
│   ├── build_experiment_sheet.py builds the experiment table (xlsx) from cluster logs
│   ├── tile_band_stats.py    per-tile band mean/std for every tile (the statistics bank; no labels read)
│   ├── tile_geo.py, fetch_prior_naip.py   tile georeferencing / neighbours; pilot of prior-year NAIP (not used in the final method)
│   ├── pack_dataset.py, dataset_report.py, shift_analysis.py, mega_analysis.py, feature_gallery.py,
│   │   pred_gallery.py, make_preview.py   dataset statistics, domain-shift analysis, figures
│   └── trainer.py, modal_app.py, train_baseline.py, ...   LEGACY first-pass trainer (not valid as a paper baseline)
└── docs/UPSTREAM_README.md   authors' README
```

Not in git (see `.gitignore`): `datasets/`, `checkpoints/`, `logs/`, `results/`, `research/reports/` (figures, cluster backups), `paper/`.

## 3. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.12
pip install torch==2.5.1 torchvision numpy==1.26.4 rasterio pandas pyyaml timm scipy \
            transformers==4.46.3 segmentation-models-pytorch
```
`numpy<2` is required by the authors' loader; `transformers==4.46.3` keeps the authors' SegFormer working.
Download the dataset from Hugging Face (revision `ffc676e`, 15,489 tiles) into `datasets/` as described in the upstream README.

## 4. Usage

Reproduce a paper baseline with the authors' code (unmodified):
```bash
python main.py --config configs/res_random.yaml --model deeplabv3 --seed 2025
```

Train with our pipeline (same data and metric, `air/` models and recipe):
```bash
python air/main.py --config configs/res_random.yaml --model convnext_upernet --seed 2025 --exp_id 5000
python air/main.py --config configs/res_state_EW.yaml --model treemortseg --aux none --seed 2025 --exp_id 5001
```

### Proposed method: DINOv2 + statistics bank + consistency (single-source domain generalisation)

Idea: spectral values differ mostly between source images (acquisition conditions), not between forest types, so a model trained in one
region sees only a few acquisition conditions. During training every source tile is restyled with the per-band mean/std of a random tile
of a **bank built from the training tiles only** (per-pixel colour transfer, NDVI recomputed, labels unchanged), and the prediction on the
restyled tile is asked to match the (detached) prediction on the original tile. Nothing is used from the target domain; inference is unchanged.

```bash
# 1) once: per-tile band statistics (no labels are read)
python research/tile_band_stats.py --info datasets/tile_info224_v3_aux.csv --tiles datasets/tiles224_v3 --out datasets/tile_band_stats.csv
# 2) train on one scenario (CO shown; also res_state_EW.yaml, res_state_WE.yaml)
python air/main.py --config configs/res_state_CO.yaml --model dinov2_b --lr_backbone_mult 0.1 \
       --bank src --cons 1.0 --seed 2025 --exp_id 7000 --tag method
```
`--bank src` takes the bank from the training tiles only (`--bank all` would use statistics of every tile, i.e. of the target domain; it is
only an analysis variant). `--cons` is the weight of the consistency loss; `--cons_ratio`, `--cons_w`, `--views`, `--tilt` are loss variants.
`--model dinov2_b` is DINOv2-B with a multi-level convolutional decoder (input up-sampled x2, 5 channels, encoder lr x0.1).
Task lists used for the reported runs: `cluster/air_tasks_dino*.txt`, `cluster/air_tasks_conf*.txt`.

Main options of `air/main.py` (all off by default):

| Flag | Meaning |
|---|---|
| `--model` | `unet, deeplabv3, convnext_upernet, treemortseg, segformer_b5, upernet_swin_s, effunet_b7, dinov2_b, hrnet_w32, m_<bb>_<dec>` |
| `--aux none/released/fixed` | TreeMortSeg auxiliary targets: none, the released ones, or our corrected ones |
| `--seed`, `--split_seed` | training seed / seed that selects the split (use `--split_seed 2025` to vary only training noise) |
| `--norm tile --photo --sampler site --vrex --ema --tta` | robustness options we tested (none helped reliably) |
| `--dino_align L` | D1: cosine alignment of ConvNeXt stride-16 features to frozen DINOv2 patch features |
| `--distill EXP1,EXP2 --ckpt_dirs DIRS --distill_alpha --distill_T` | D2: soft-label distillation from finished checkpoints |
| `--selfdistill A --ema 0.999` | D3: soft targets from the EMA of the student |
| `--copypaste P`, `--gate W` | copy-paste of small dead crowns; tile-level presence gate |
| `--bank src/all`, `--cons W`, `--bank_gen S`, `--bank_lam`, `--bank_indep`, `--views K`, `--tilt T`, `--cons_ratio R`, `--cons_w G` | statistics-bank augmentation and consistency-loss variants (proposed method) |
| `--photo`, `--photo_parts`, `--photo_scale`, `--fda P`, `--mixstyle P`, `--rel` | photometric / Fourier-mix / feature MixStyle / relative-channel baselines |
| `--ctx PX` | neighbouring-tile context (border of PX pixels from adjacent tiles) |

Recipe (all `air/` runs): AdamW (wd 0.01), lr 1e-4, ExponentialLR 0.95, batch 32, bf16, early stopping (patience 10) on the validation mask loss.
Metric: `exps/evaluate.py` (dead-tree class, threshold 0.5, pooled pixel counts).

Run a batch on Slurm: edit the paths in `cluster/air_worker12.sh`, then `bash cluster/air_worker12.sh cluster/air_tasks_kd_A.txt`.
Rebuild the experiment table: `python research/build_experiment_sheet.py --logs <dir containing logs/> --out experiment_log.xlsx`.
Further evaluation of finished runs: `air/eval_extra.py`, `air/ensemble_eval.py` (see their docstrings).

## 5. Results so far (dead-tree F1 in %, mean of 3 training seeds 2025 / 2024 / 2023, split fixed by the scenario)

Work in progress; numbers are from our own re-runs on the cluster and are not final. "TreeMortSeg (re-run)" = TreeMortSeg with the paper's loss weights
(1 / 2.5 / 2.5) and auxiliary labels rebuilt from the dead-tree mask as the paper describes; "reported" = the single value in the TreeMortSeg paper (no seeds/variance given).

**Main scenarios (the three cross-region transfers used by TreeMortSeg) + Random as a sanity check**

| Scenario | **DINOv2 + bank + consistency** | TreeMortSeg (re-run) | TreeMortSeg (reported) |
|---|---|---|---|
| CO -> non-CO | **23.1** | 17.8 | 19.55 |
| East -> West | **35.4** | 29.2 | 28.84 |
| West -> East | **26.3** | 22.0 | 23.67 |
| Random split | 63.0 (seeds 2025/1/2; being re-run with 2025/2024/2023) | 63.3 | 66.29 +- 0.40 |

**Where the gain comes from** (same three scenarios)

| | CO | E-W | W-E |
|---|---|---|---|
| ConvNeXt-S + UPerNet | 14.2 | 24.4 | 20.0 |
| ConvNeXt-S + UPerNet + bank + consistency | 19.1 | 27.2 | 21.7 |
| DINOv2-B | 21.2 | 33.9 | 25.5 |
| **DINOv2-B + bank + consistency** | **23.1** | **35.4** | **26.3** |

Most of the gain comes from the DINOv2 backbone; the bank + consistency augmentation adds about +0.8 to +1.9 points (not significant seed by seed with only 3 seeds).

**Other scenarios (not the target of the method; reported for completeness):** Climate 35.6 vs TreeMortSeg 34.0; Climate-hard 24.6 vs 28.7 (TreeMortSeg is better);
Forest 41.3 (TreeMortSeg re-run still running); Forest-hard not finished.

Findings so far
- Released TreeMortSeg auxiliary labels are built from the no-data mask; rebuilding them from the dead-tree mask lifts TreeMortSeg on CO from about 8.5 to 17.8.
- The random tile split leaks sites (857 of 860 test sites also appear in training); neighbouring-tile context only helps on that split.
- Negative results (kept in the repo): DINOv2 feature alignment, distillation, self-training, Fourier mix, AdaBN, copy-paste, tile gate.
- Variance is large (CO: 13-21 F1 for one model across seeds), so every comparison is made on 3 seeds, with the same seeds for all methods.
- Caveats: configuration choices (e.g. the consistency weight) were explored on CO with the same three seeds, so the CO number can be optimistic;
  E-W and W-E use the same configuration without tuning. DINOv2's pre-training data may contain aerial imagery.

Seeds: for Random, 2025/2024/2023 are the seeds of the authors' `run_random*.sh`; for the other scenarios the authors only ran seed 2025, so extra seeds are ours.
In the authors' protocol the Random seed also changes the train/test split; our re-runs keep the split fixed (`--split_seed 2025`) and vary only the training seed.

## 6. Credits

Please cite the TreeFinder paper (Wang et al., NeurIPS 2025 D&B) and TreeMortSeg (Li et al., 2026) when using this work.
