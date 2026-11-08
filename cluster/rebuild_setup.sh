#!/bin/bash
# Rebuild /datastore/phuocnth/treefinder after it was deleted: dataset (pinned HF commit), venv (versions reported in the paper), pretrained weights.
# Run on the login node with nohup; progress is written to $D/setup.log.
set -u
D=/datastore/phuocnth/treefinder
mkdir -p $D/datasets $D/logs $D/out $D/hf_cache $D/torch_cache $D/pretrained_weights/dofa
exec >> $D/setup.log 2>&1
echo "[$(date +%T)] setup start"

# ---- 1. dataset (same pinned commit and checksums as before)
cd $D/datasets
base='https://huggingface.co/datasets/zhwang1/TreeFinder/resolve/ffc676e1443a4dac0ed42d06cec36bcde224c770'
for n in tile_info224_v3.csv tile_info224_v3_aux.csv tiles224_v3.zip; do
  curl -L --fail --retry 5 -C - -o $n $base/$n || echo "DOWNLOAD FAILED $n"
done
cd $D && sha256sum -c dataset.sha256 && echo "[$(date +%T)] dataset checksums OK" || echo "CHECKSUM MISMATCH"
cd $D/datasets && [ -d tiles224_v3 ] || unzip -q tiles224_v3.zip
echo "[$(date +%T)] tiles: $(ls $D/datasets/tiles224_v3 | wc -l)"

# ---- 2. python environment (Python 3.12, PyTorch 2.5.1 / CUDA 12.4, Transformers 4.46.3 as stated in the paper)
if [ ! -x $D/env/bin/python ]; then
  /usr/bin/python3.12 -m venv $D/env
  $D/env/bin/pip install -q --upgrade pip
  $D/env/bin/pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
  $D/env/bin/pip install -q transformers==4.46.3 "numpy<2" pandas scikit-learn scipy pyyaml tqdm rasterio huggingface_hub
fi
$D/env/bin/python -c "import torch,torchvision,transformers,numpy,rasterio;print('env OK',torch.__version__,torch.version.cuda,torchvision.__version__,transformers.__version__,numpy.__version__)"

# ---- 3. pretrained weights
export HF_HOME=$D/hf_cache TORCH_HOME=$D/torch_cache
$D/env/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for r in ["google/vit-base-patch16-224-in21k", "nvidia/segformer-b2-finetuned-ade-512-512", "facebook/mask2former-swin-tiny-ade-semantic"]:
    print("hf", r, snapshot_download(r))
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models import resnet50, ResNet50_Weights
deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
print("torchvision weights cached")
PY
curl -L --fail --retry 5 -C - -o $D/pretrained_weights/dofa/DOFA_ViT_base_e100.pth https://huggingface.co/XShadow/DOFA/resolve/main/DOFA_ViT_base_e100.pth || echo "DOFA DOWNLOAD FAILED"
ls -la $D/pretrained_weights/dofa
echo "[$(date +%T)] setup done"
