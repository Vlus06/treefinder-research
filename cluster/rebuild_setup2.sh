#!/bin/bash
# Second part of the rebuild: the venv had no pip, so bootstrap pip, install the paper's versions, then fetch the pretrained weights.
set -u
D=/datastore/phuocnth/treefinder
exec >> $D/setup.log 2>&1
echo "[$(date +%T)] setup2 start"
if ! $D/env/bin/python -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o $D/get-pip.py && $D/env/bin/python $D/get-pip.py
fi
$D/env/bin/python -m pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
$D/env/bin/python -m pip install -q transformers==4.46.3 "numpy<2" pandas scikit-learn scipy pyyaml tqdm rasterio huggingface_hub
$D/env/bin/python -c "import torch,torchvision,transformers,numpy,rasterio;print('env OK',torch.__version__,torch.version.cuda,torchvision.__version__,transformers.__version__,numpy.__version__)"
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
echo "[$(date +%T)] setup2 done"
