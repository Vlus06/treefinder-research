# TreeFinder research fork: reproduction, analysis and a competitive single-model recipe

Fork of [zhwang0/treefinder](https://github.com/zhwang0/treefinder) (NeurIPS 2025 D&B, dead-tree segmentation on NAIP tiles).
It adds (1) a **faithful reproduction** of the dataset paper's baselines with the authors' unmodified code, (2) a separate
pipeline (`air/`) for **stronger baselines and new methods**, evaluated on the *same* splits, loaders and metric, and
(3) analysis scripts and the cluster files used to run everything.
The authors' original README is kept in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

> Status: work in progress (course project / conference paper draft). Numbers marked *preliminary* are single-seed.

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
│   ├── distill.py            DINOv2 alignment, teacher loading, copy-paste, tile gate
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

Recipe (all `air/` runs): AdamW (wd 0.01), lr 1e-4, ExponentialLR 0.95, batch 32, bf16, early stopping (patience 10) on the validation mask loss.
Metric: `exps/evaluate.py` (dead-tree class, threshold 0.5, pooled pixel counts).

Run a batch on Slurm: edit the paths in `cluster/air_worker12.sh`, then `bash cluster/air_worker12.sh cluster/air_tasks_kd_A.txt`.
Rebuild the experiment table: `python research/build_experiment_sheet.py --logs <dir containing logs/> --out experiment_log.xlsx`.
Further evaluation of finished runs: `air/eval_extra.py`, `air/ensemble_eval.py` (see their docstrings).

## 5. Results so far (dead-tree F1 in %, tile split, seed 2025 unless stated)

Reproduction of the dataset paper (authors' code, seed 2025; all scenarios in `research/REPRODUCTION.md`): 36 of 46 cells within +-5 points of the paper;
Random U-Net / DeepLabV3 (3 seeds) 46.8 / 52.3 vs the paper's 46.1 / 49.7.

| Method | F1 | Note |
|---|---|---|
| ConvNeXt-S + UPerNet (ours) | 63.1 +- 0.4 | 3 training seeds, split fixed |
| TreeMortSeg re-run | 61 to 65 | its authors report 66.29 +- 0.40 |
| Ensemble ConvNeXt + DINOv2 + TreeMortSeg | 65.8 | 3x inference cost |
| D2: distillation from that ensemble (*preliminary*) | 64.7 | 1 seed, single model |
| D1: DINOv2 feature alignment | 57.7 to 61.7 | worse than the baseline |
| copy-paste / tile gate (*preliminary*) | 63.3 / 63.9 | within noise |

Findings so far: the tile split leaks sites (857 of 860 test sites also appear in training); 36 "mega" tiles hold about 30% of all dead pixels and dominate pooled F1;
TreeMortSeg's released auxiliary labels are built from the no-data mask; auxiliary multi-task heads gave no reliable gain; in the cross-region scenarios ConvNeXt and TreeMortSeg are within noise of each other.

Seeds: for Random, 2025/2024/2023 are the seeds of the authors' `run_random*.sh`; for the other scenarios the authors only ran seed 2025, so extra seeds are ours.
In Random, the seed also changes the train/test split.

## 6. Credits

Please cite the TreeFinder paper (Wang et al., NeurIPS 2025 D&B) and TreeMortSeg (Li et al., 2026) when using this work.
