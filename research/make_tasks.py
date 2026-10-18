"""Write task lists (one trainer command line per row) for the HPC workers.

    python research/make_tasks.py            # writes cluster/tasks_*.txt
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cluster"
OUT.mkdir(exist_ok=True)
SEEDS = (2025, 2026, 2027)
SCEN = ("random", "w2e", "e2w", "co", "climate", "climate_hard", "forest", "forest_hard")
OFFICIAL = ("official_unet", "official_deeplab", "official_segformer", "official_mask2former")


def line(protocol, model, seed, tag, flags=""):
    return f"--protocol {protocol} --model {model} --seed {seed} --recipe paper --tag {tag} {flags}".strip()


def write(name, lines):
    (OUT / f"tasks_{name}.txt").write_text("\n".join(lines) + "\n")
    print(name, len(lines))


# 1. paper baselines, authors' model code, under the paper's recipe and official splits
write("baselines", [line(f"official:{sc}", m, s, "paper") for s in SEEDS for sc in SCEN for m in OFFICIAL])
write("baselines_seed2025", [line(f"official:{sc}", m, 2025, "paper") for sc in SCEN for m in OFFICIAL])
write("segformer3_seed2025", [line(f"official:{sc}", "official_segformer3", 2025, "paper") for sc in SCEN])
# 3. candidates directly on the official splits (all reported, none selected per scenario)
OFF_CAND = {"base": "", "photo": "--photo-aug", "ftv": "--loss bce_ftversky", "photo_ftv": "--photo-aug --loss bce_ftversky",
            "both_photo_ftv": "--norm both --photo-aug --loss bce_ftversky", "aux_photo_ftv": "--aux --photo-aug --loss bce_ftversky"}
write("official_cand_seed2025", [line(f"official:{sc}", "unet_r34", 2025, f"cand_{k}", f) for sc in SCEN for k, f in OFF_CAND.items()])
# 4. input-band effect: transformers as the scripts ran them (RGB only) vs with NIR+NDVI
EXTRA = ("official_mask2former_rgb", "official_vit3", "segformer_b2")
write("input_bands_seed2025", [line(f"official:{sc}", m, 2025, "paper") for sc in SCEN for m in EXTRA])
# 5. controlled leakage test: identical model/recipe/seeds, only the unit of splitting differs (tile vs site)
write("leakage_control", [line(p, m, s, "ctrl") for s in SEEDS for m in ("official_unet", "official_deeplab") for p in ("official:random", "site")])
write("smoke", [line("official:random", "official_unet", 2025, "smoke", "--epochs 2")])

# 2. development: candidate recipes on my own site-disjoint / cross-region splits (never the official test splits)
DEV = ("site", "w2e", "e2w")
CAND = {
    "base": "",
    "photo": "--photo-aug",
    "ftv": "--loss bce_ftversky",
    "photo_ftv": "--photo-aug --loss bce_ftversky",
    "both_photo_ftv": "--norm both --photo-aug --loss bce_ftversky",
    "aux_photo_ftv": "--aux --photo-aug --loss bce_ftversky",
}
write("dev", [line(p, "unet_r34", s, f"dev_{k}", f) for k, f in CAND.items() for p in DEV for s in SEEDS])
