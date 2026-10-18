# TreeFinder baseline reproduction — status and results

Generated 2026-10-15 from runs on the UIT H200 cluster. **Work in progress**: only the cells marked with a number
have been run; `—` means queued and not yet available. Nothing here is an estimate.

## 1. How the baselines are reproduced

| Item | Setting |
|---|---|
| Code | Authors' repository `zhwang0/treefinder`, `main.py` **unmodified** (clone at `8c63341`), run via `scripts/run_*.sh`-equivalent calls |
| Data | HuggingFace `zhwang1/TreeFinder` revision `ffc676e` (v3, 15,489 tiles), sha256-checked |
| Configs | `configs/res_*.yaml` as released (Nov 2025 update: seed 2025, **lr 1e-5 for `res_random.yaml`**, 1e-4 otherwise). Only change: `num_workers` 40 → 8 (copies in `configs_hpc/`) |
| Optimiser / loss | AdamW (wd 0.01), ExponentialLR(0.95), batch 32, ≤100 epochs, BCE + Dice, early stop on val loss (patience 5) — all from the config |
| Precision | fp32 (the authors' default) |
| Metric | Exactly `exps/evaluate.py`: threshold 0.5, counts pooled over the test set; "Dead tree" = class 1, "All" = mean of both classes |
| Environment | Python 3.12, torch 2.5.1+cu124, numpy 1.26.4 (the released code fails under numpy 2: `torch.tensor(np.bool_, dtype=long)`), transformers 4.46.3 |
| Hardware | one 2g.35gb MIG slice of an H200, one run at a time |
| Seeds | 2025 first; then 2024, 2023 (the seeds used by `scripts/run_random*.sh`; the paper only says "three runs") |

Model names are those the authors' scripts actually call: `unet`, `deeplabv3`, `vitseg3`, `segformer_large3`,
`mask2former`, `dofa`. A second Mask2Former run uses the repository **before** commit `274e112`
(RGB-only forward) to match the code that existed when the paper was written.

### Scenario → config mapping (my reading; the paper does not name configs)

| Paper scenario | Config | Train → test |
|---|---|---|
| Random | `res_random.yaml` | 80 % tiles (10 % of it val) → 20 % tiles |
| W – E | `res_state_WE.yaml` | western states → eastern states |
| E – W | `res_state_EW.yaml` | eastern → western |
| CO | `res_state_CO.yaml` | Colorado → all other states |
| Climate | `res_climate1.yaml` | Cfa → all other climates |
| Climate-hard | `res_climate3.yaml` | Dfa (humid continental) → rest |
| Forest | `res_treetype1.yaml` | top-5 tree types → rest |
| Forest-hard | `res_treetype3.yaml` | maple → rest |

Ambiguity: the paper's text does not say whether "Climate" is `climate1` (train on Cfa, 37 % of tiles) or
`climate2` (train on everything but Cfa). I assumed `climate1`; the swapped variants are in the appendix tables.

## 2. Completed runs (seed 2025, `main.py` unmodified)

F1 / precision / recall / IoU are for the dead-tree class, in %.

| Scenario | Model | F1 | P | R | IoU | F1 (All) | epochs | Paper F1 (mean ± sd, 3 runs) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Random | U-Net | **47.4** | 37.1 | 65.6 | 31.0 | 73.5 | 33 | 46.1 ± 7.1 (P 35.7, R 65.6, IoU 30.1, All 72.9) | within 1 sd — reproduced |
| Random | DeepLabV3 | **52.7** | 54.4 | 51.2 | 35.8 | 76.3 | 45 | 49.7 ± 8.7 (P 49.6, R 49.9, IoU 33.4, All 74.7) | within 1 sd — reproduced |

Both completed cells are single-seed; they fall within one reported standard deviation of the paper.
This is *consistent with* reproduction, not yet a statistical confirmation.

## 3. Full grid (seed 2025) — paper values for reference

Dead-tree F1 in %. "—" = not yet run. Paper numbers are from Tables 1–3 of the TreeFinder paper
(Random is mean over three runs; the other scenarios are a single value).

| Scenario | U-Net paper / mine | DeepLabV3 | ViT | SegFormer | Mask2Former | DOFA |
|---|---|---|---|---|---|---|
| Random | 46.1 / **47.4** | 49.7 / **52.7** | 43.6 / — | 47.8 / — | 51.9 / — | 29.2 / — |
| W – E | 13.9 / — | 15.0 / — | 13.4 / — | 22.8 / — | 21.4 / — | 4.4 / — |
| E – W | 16.1 / — | 20.5 / — | 3.8 / — | 21.8 / — | 21.6 / — | 4.8 / — |
| CO | 10.2 / — | 15.4 / — | 6.3 / — | 17.5 / — | 11.6 / — | 10.2 / — |
| Climate | 18.6 / — | 25.8 / — | 21.6 / — | 28.0 / — | 29.8 / — | 12.9 / — |
| Climate-hard | 19.9 / — | 18.0 / — | 11.4 / — | 22.6 / — | 15.4 / — | 4.7 / — |
| Forest | 35.3 / — | 36.1 / — | 26.5 / — | 40.1 / — | 38.6 / — | 13.9 / — |
| Forest-hard | 14.7 / — | 21.5 / — | 19.9 / — | 37.7 / — | 11.1 / — | 14.3 / — |

## 4. Status of the queue

* Job `3637` (single process, sequential): 56 runs = 8 scenarios × (6 models + Mask2Former RGB-only), seed 2025.
* Measured speed: ≈ 50–60 min per run on one 2g.35gb slice, so the 56 runs need roughly 2–3 days of wall time.
* Completed: Random/U-Net, Random/DeepLabV3. In progress: Random/ViT (RGB-only).
* Not yet queued: seeds 2024 and 2023; swapped climate/forest scenarios; site-split control experiment.

## 5. Known differences and caveats

1. **Earlier numbers from my own fast trainer (`research/trainer.py`, Modal and first HPC jobs) are not
   reproduction results.** They used z-score normalisation (the authors only divide by 255), bf16 autocast,
   per-batch rot90 and other differences. They are kept only as exploratory data and must not be cited as baselines.
2. **SegFormer and ViT in the authors' scripts see RGB only** (`x[:, :3]`, models `segformer_large3`, `vitseg3`);
   U-Net/DeepLab use all five channels. Before commit `274e112` (2026-07-03) Mask2Former was RGB-only too.
3. The builder named `segformer` in `models/` is a B0-sized network initialised almost from scratch (pretrained
   B2 weights are only copied where shapes match). The released scripts do not use it; do not use it as a baseline.
4. v2 of the dataset (used by the first public commit, seed 2020, lr 1e-4) is no longer on HuggingFace.
   The paper reports 15,489 tiles, which matches v3.
5. `res_random.yaml` has lr = 1e-5 but the paper text says 1e-4. The U-Net run above used the config value and matched
   the paper; the 1e-4 variant has not been run.
6. Exact bitwise reproduction is not possible (GPU non-determinism, library versions, unreported seeds).
   The target is agreement within the paper's reported standard deviation.
7. For cross-region tables the paper gives a single number with no deviation, so "reproduced" there can only be
   judged by absolute difference; I will report my 3-seed mean ± sd next to the paper value.

## 6. Files

* Results on the cluster: `/datastore/phuocnth/treefinder/code_head/results/exp*_*/`, logs in `.../logs/`.
* Task list and runner: `cluster/official_tasks_all.txt`, `cluster/official_worker.sh`.
* Split manifests that replicate the authors' splitters: `research/splits/official_*_seed*.csv`.
