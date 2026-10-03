#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root in a Kaggle notebook with Internet enabled.
python -m pip install -q rasterio
mkdir -p datasets

base='https://huggingface.co/datasets/zhwang1/TreeFinder/resolve/ffc676e1443a4dac0ed42d06cec36bcde224c770'
for name in tile_info224_v3.csv tile_info224_v3_aux.csv tiles224_v3.zip; do
  case "$name" in
    tile_info224_v3.csv) expected=1442557 ;;
    tile_info224_v3_aux.csv) expected=1818051 ;;
    tiles224_v3.zip) expected=1970236262 ;;
  esac
  actual=$(stat -c%s "datasets/$name" 2>/dev/null || echo 0)
  if [[ "$actual" != "$expected" ]]; then
    curl -L --fail --retry 5 --continue-at - --output "datasets/$name" "$base/$name"
  fi
done
sha256sum -c research/dataset.sha256

python research/extract_tiles.py
for seed in ${TREEFINDER_SEEDS:-2025}; do
  python research/make_site_split.py --seed "$seed"
  python research/make_transfer_split.py --seed "$seed"
done
python -c 'import torch; print("torch", torch.__version__, "CUDA", torch.cuda.is_available(), "GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
