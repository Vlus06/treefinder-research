"""Run TreeFinder experiments on Modal GPUs.

One-time data preparation (downloads from Hugging Face inside Modal, packs into a Volume):
    modal run research/modal_app.py::prepare
Run a list of trainer configurations in parallel (one GPU container each):
    modal run research/modal_app.py::sweep --grid pilot
Results are written to research/reports/runs/<run>/metrics.json (and kept on the Volume).
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co/datasets/zhwang1/TreeFinder/resolve/ffc676e1443a4dac0ed42d06cec36bcde224c770"
FILES = {"tile_info224_v3.csv": 1442557, "tile_info224_v3_aux.csv": 1818051, "tiles224_v3.zip": 1970236262}

app = modal.App("treefinder-research")
volume = modal.Volume.from_name("treefinder-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy<2", "pandas", "scipy", "rasterio",
                 "segmentation-models-pytorch==0.4.0", "timm", "transformers==4.46.3", "huggingface_hub")
    .add_local_dir(str(ROOT), "/root/tf", copy=False, ignore=[
        ".git", ".venv", "datasets", "paper", "checkpoints", "logs", "results", "research/reports", "__pycache__", "*.pyc"])
)


@app.function(image=image, volumes={"/data": volume}, timeout=3 * 3600, cpu=8, memory=24576)
def prepare():
    import subprocess
    import os
    d = Path("/root/tf/datasets")
    d.mkdir(parents=True, exist_ok=True)
    for name, size in FILES.items():
        subprocess.run(["curl", "-L", "--fail", "--retry", "5", "-o", str(d / name), f"{HF}/{name}"], check=True)
        assert (d / name).stat().st_size == size, name
    env = {**os.environ, "PYTHONPATH": "/root/tf"}
    subprocess.run(["python", "research/extract_tiles.py"], cwd="/root/tf", check=True, env=env)
    subprocess.run(["python", "research/pack_dataset.py", "--workers", "8"], cwd="/root/tf", check=True, env=env)
    out = Path("/data/packed")
    out.mkdir(parents=True, exist_ok=True)
    for f in ("x.npy", "y.npy", "tile_stats.csv"):
        subprocess.run(["cp", str(d / "packed" / f), str(out / f)], check=True)
    volume.commit()
    return "prepared"


@app.function(image=image, volumes={"/data": volume}, gpu="L4", timeout=3 * 3600, memory=20480, cpu=4)
def train(argline: str) -> dict:
    import subprocess
    import sys
    args = shlex.split(argline)
    cmd = [sys.executable, "research/trainer.py", "--data-dir", "/data/packed", "--out", "/data/runs",
           "--in-memory", *args]
    proc = subprocess.run(cmd, cwd="/root/tf", capture_output=True, text=True,
                          env={"PYTHONPATH": "/root/tf", "PATH": "/usr/local/bin:/usr/bin:/bin", "HF_HOME": "/tmp/hf"})
    tail = "\n".join(proc.stdout.splitlines()[-4:])
    if proc.returncode != 0:
        return {"argline": argline, "error": proc.stderr[-3000:], "stdout_tail": tail}
    seed = "2025"
    protocol, model, tag = "site", "unet_r34", ""
    for i, a in enumerate(args):
        if a == "--protocol": protocol = args[i + 1]
        if a == "--model": model = args[i + 1]
        if a == "--seed": seed = args[i + 1]
        if a == "--tag": tag = args[i + 1]
    name = f"{protocol}_{model}_s{seed}" + (f"_{tag}" if tag else "")
    metrics = json.loads(Path(f"/data/runs/{name}/metrics.json").read_text())
    by_site = Path(f"/data/runs/{name}/test_by_site.csv").read_text()
    ad = Path(f"/data/runs/{name}/test_by_site_adabn.csv")
    by_site_adabn = ad.read_text() if ad.exists() else None
    volume.commit()
    return {"argline": argline, "name": name, "metrics": metrics, "by_site": by_site, "by_site_adabn": by_site_adabn}


def _abl(model, protocols, tag, flags, seeds=(2025,), epochs=30):
    return [f"--protocol {p} --model {model} --epochs {epochs} --seed {s} --tag {tag} {flags}".strip()
            for p in protocols for s in seeds]


ALL3 = ("site", "w2e", "e2w")
GRIDS = {
    # leakage: same models under the official tile-random protocol
    "leak": [f"--protocol tile --model {m} --epochs 30 --tag base" for m in ("unet_small", "unet_r34", "dlv3p_r50", "segformer_b2")],
    # recipe ablations on the fast U-Net-R34 (single seed; triage only)
    # multi-seed comparison of the candidate recipes (U-Net-R34)
    "abl2": (_abl("unet_r34", ALL3, "base", "--adabn", seeds=(2026, 2027))
             + _abl("unet_r34", ALL3, "photo", "--photo-aug --adabn", seeds=(2026, 2027))
             + _abl("unet_r34", ALL3, "ftv", "--loss bce_ftversky --adabn", seeds=(2026, 2027))
             + _abl("unet_r34", ALL3, "photo_ftv", "--photo-aug --loss bce_ftversky --adabn", seeds=(2025, 2026, 2027))
             + _abl("unet_r34", ALL3, "both_photo_ftv", "--norm both --photo-aug --loss bce_ftversky --adabn", seeds=(2025, 2026, 2027))
             + _abl("unet_r34", ALL3, "aux_photo_ftv", "--aux --photo-aug --loss bce_ftversky --adabn", seeds=(2025, 2026, 2027))),
    "abl1": (_abl("unet_r34", ALL3, "photo", "--photo-aug --adabn")
             + _abl("unet_r34", ALL3, "tile", "--norm tile --adabn")
             + _abl("unet_r34", ALL3, "photo_tile", "--norm tile --photo-aug --adabn")
             + _abl("unet_r34", ALL3, "up2", "--upscale 2 --adabn")
             + _abl("unet_r34", ALL3, "ftv", "--loss bce_ftversky --adabn")
             + _abl("unet_r34", ALL3, "base2", "--adabn")),
    # first look: three protocols x four architectures, plain recipe
    "pilot": [f"--protocol {p} --model {m} --epochs 30 --tag base"
              for p in ("site", "w2e", "e2w") for m in ("unet_small", "unet_r34", "dlv3p_r50", "segformer_b2")],
}


@app.local_entrypoint()
def sweep(grid: str = "pilot", extra: str = ""):
    if grid in GRIDS:
        configs = GRIDS[grid]
    else:  # a path to a text file: one trainer argument line per row
        configs = [l.strip() for l in Path(grid).read_text().splitlines() if l.strip() and not l.startswith("#")]
    configs = [f"{c} {extra}".strip() for c in configs]
    out_root = ROOT / "research" / "reports" / "runs"
    for res in train.map(configs, return_exceptions=True):
        if isinstance(res, Exception) or "error" in res:
            print("FAILED", res if isinstance(res, Exception) else res["argline"], "\n", (res if isinstance(res, Exception) else res["error"])[-1500:])
            continue
        d = out_root / res["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_text(json.dumps(res["metrics"], indent=2), encoding="utf-8")
        (d / "test_by_site.csv").write_text(res["by_site"], encoding="utf-8")
        if res.get("by_site_adabn"):
            (d / "test_by_site_adabn.csv").write_text(res["by_site_adabn"], encoding="utf-8")
        t = res["metrics"]["test"]
        print(res["name"], "val_f1=%.3f test_f1@.5=%.3f @val=%.3f AP=%.3f oracle=%.3f (%.1f min)" % (
            res["metrics"]["val_f1"], t["at_0.5"]["f1"], t["at_val_thr"]["f1"], t["ap"], t["oracle_thr"]["f1"],
            res["metrics"]["minutes"]), flush=True)
