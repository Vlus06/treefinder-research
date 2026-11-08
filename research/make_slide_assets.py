"""Figures and the result table for the report slides (slides/main.tex).

    python research/make_slide_assets.py          # writes slides/figs/*.png and slides/results_table.tex
Numbers come from research/reports/table_metrics.json (see research/table_metrics.py); tiles are real TreeFinder tiles.
"""
import csv
import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "slides"
FIG = OUT / "figs"
FIG.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("archfig", ROOT / "research" / "make_architecture_figures_v2.py")
arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arch)

D = json.load(open(ROOT / "research" / "reports" / "table_metrics.json"))
SCEN = [("Random", "Random"), ("CO", "CO→non-CO"), ("EW", "East→West"), ("WE", "West→East"), ("Site", "Site-disjoint")]
GREY, BLUE, ORANGE = "#7C8794", "#4C8BC9", "#E8892B"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def f1(sc, name):
    v = D[sc][name]
    return (v["dead_f1"][0], v["dead_f1"][1]) if v["n"] >= 3 else (np.nan, np.nan)


# ------------------------------------------------------------------ sample tiles with dead-tree labels
rows = list(csv.DictReader(open(ROOT / "datasets" / "tile_info224_v3.csv")))
good = [r for r in rows if int(r["NoDataSize"]) == 0 and 900 <= int(r["LabelSize"]) <= 2600]
pick = {}
for r in good:
    pick.setdefault(r["State"], r)
chosen = [pick[s] for s in ("CO", "FL", "NY") if s in pick][:3]
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.7))
for ax, r in zip(axs, chosen):
    a = arch.read_tile(r["FileName"])
    img = arch.rgb(a).copy()
    m = a[..., 4] == 1
    img[m] = 0.55 * img[m] + 0.45 * np.array([1.0, 0.1, 0.55])
    ax.imshow(img)
    ax.set_title(r["State"], fontsize=13)
    ax.axis("off")
fig.suptitle("Real tiles (224×224); pink = dead-tree label", fontsize=11, y=0.02, va="bottom")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(FIG / "sample_tiles.png", dpi=170)
plt.close(fig)

# ------------------------------------------------------------------ F1 bars: TreeMortSeg vs DINOv2+decoder vs Kaleido
names = [("TreeMortSeg", "TreeMortSeg", GREY),
         ("DINOv2 + TreeMortBank (m=0.01)", "Kaleido (Ours)", ORANGE)]
fig, ax = plt.subplots(figsize=(10.5, 4.6))
w = 0.34
for k, (key, lab, col) in enumerate(names):
    vals = [f1(sc, key) for sc, _ in SCEN]
    x = np.arange(len(SCEN)) + (k - 0.5) * w
    ax.bar(x, [v[0] for v in vals], w, yerr=[v[1] for v in vals], capsize=3, color=col, label=lab, error_kw={"lw": 1})
    for xi, v in zip(x, vals):
        if not np.isnan(v[0]):
            ax.text(xi, v[0] + v[1] + 0.8, f"{v[0]:.1f}", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(xi, 2, "running", ha="center", va="bottom", fontsize=9, rotation=90, color="#777")
ax.set_xticks(np.arange(len(SCEN)))
ax.set_xticklabels([s for _, s in SCEN])
ax.set_ylabel("F1 of the dead-tree class (%)")
ax.set_ylim(0, 78)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.text(0.99, 0.97, "mean ± std over 3 seeds", transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#555")
fig.tight_layout()
fig.savefig(FIG / "f1_bars.png", dpi=170)
plt.close(fig)

# ------------------------------------------------------------------ colour-shift robustness on CO->non-CO: baseline vs Kaleido
DIAG = ROOT / "research" / "reports" / "diag"
sh = json.load(open(DIAG / "style_shift_CO.json"))
SHIFT_LABEL = {"identity": "no shift", "bright0.70": "bright. x0.7", "bright0.85": "bright. x0.85", "bright1.15": "bright. x1.15",
               "bright1.30": "bright. x1.3", "gamma0.70": "gamma 0.7", "gamma1.40": "gamma 1.4", "castRB+": "R x1.2, B x0.8",
               "castRB-": "R x0.8, B x1.2", "contrast0.60": "contrast x0.6", "contrast1.40": "contrast x1.4", "nir0.70": "NIR x0.7",
               "nir1.30": "NIR x1.3"}


def shift_ms(group, key):
    v = [100 * sh[group]["per_seed"][s][key] for s in sh[group]["per_seed"]]
    return float(np.mean(v)), float(np.std(v, ddof=1))


fig, ax = plt.subplots(figsize=(11, 4.4))
w = 0.38
for k, (grp, lab, col) in enumerate([("plain", "DINOv2 + decoder (m=0.01)", BLUE), ("kaleido", "Kaleido (Ours)", ORANGE)]):
    vals = [shift_ms(grp, key) for key in SHIFT_LABEL]
    ax.bar(np.arange(len(SHIFT_LABEL)) + (k - 0.5) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals], capsize=2, color=col, label=lab,
           error_kw={"lw": 0.9})
ax.set_xticks(np.arange(len(SHIFT_LABEL)))
ax.set_xticklabels(list(SHIFT_LABEL.values()), rotation=35, ha="right", fontsize=10)
ax.set_ylabel("F1 of the dead-tree class (%)")
ax.set_ylim(0, 36)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout()
fig.savefig(FIG / "shift_co.png", dpi=170)
plt.close(fig)

# ------------------------------------------------------------------ architecture figures (copy)
for n in ("arch_overview", "arch_detail", "arch_restyle"):
    shutil.copy(ROOT / "proposal" / "figures" / f"{n}.png", FIG / f"{n}.png")

# ------------------------------------------------------------------ compact result table (dead-tree F1 mean ± std)
ORDER = [("U-Net", "U-Net"), ("DeepLabV3+", "DeepLabV3+"), ("ViT", "ViT"), ("SegFormer", "SegFormer"), ("Mask2Former", "Mask2Former"),
         ("DOFA", "DOFA"), ("ConvNeXt-UPerNet", "ConvNeXt-UPerNet"), ("TreeMortSeg", "TreeMortSeg"),
         ("DINOv2 + TreeMortBank (m=0.1)", "Kaleido ($m{=}0.1$) (Ours)"), ("DINOv2 + TreeMortBank (m=0.01)", "\\textbf{Kaleido (Ours)}")]
best = {sc: max((f1(sc, k)[0] for k, _ in ORDER if not np.isnan(f1(sc, k)[0])), default=None) for sc, _ in SCEN}
lines = ["\\begin{tabular}{lccccc}", "\\toprule",
         "\\textbf{Model} & \\textbf{Random} & \\textbf{CO$\\to$non-CO} & \\textbf{E$\\to$W} & \\textbf{W$\\to$E} & \\textbf{Site-disj.} \\\\", "\\midrule"]
for k, lab in ORDER:
    cells = []
    for sc, _ in SCEN:
        m, s = f1(sc, k)
        if np.isnan(m):
            cells.append("\\textit{running}")
        else:
            t = f"{m:.1f}$\\pm${s:.1f}"
            cells.append(f"\\textbf{{{t}}}" if abs(m - best[sc]) < 1e-9 else t)
    if k == "DINOv2 + TreeMortBank (m=0.1)":
        lines.append("\\midrule")
    lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
(OUT / "results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf8")

# ------------------------------------------------------------------ error anatomy of Kaleido (m=0.01): crown-level statistics, mean +- std over seeds
eo = json.load(open(DIAG / "error_objects_dino.json"))
groups = {}
for r in eo.values():
    key = r["args"]["tag"].split("_")[0]
    if key in ("co", "ew", "we", "site"):
        groups.setdefault(key, []).append(r)
el = ["\\begin{tabular}{lccccc}", "\\toprule",
      "\\textbf{Kịch bản} & \\textbf{Tán phát hiện} & \\textbf{Tán $>256$ px phát hiện} & \\textbf{Pixel thuộc tán $>256$ px} & "
      "\\textbf{Bỏ sót nguyên tán} & \\textbf{FP tách biệt} \\\\", "\\midrule"]
for key, name in (("co", "CO$\\to$non-CO"), ("ew", "East$\\to$West"), ("we", "West$\\to$East"), ("site", "Site-disjoint")):
    per = []
    for r in groups[key]:
        a_, o_ = r["acc"], r["other"]
        per.append([100 * sum(a_["det"]) / sum(a_["n"]), 100 * a_["det"][3] / a_["n"][3], 100 * a_["px"][3] / sum(a_["px"]),
                    100 * o_["fn_whole"] / (o_["fn_whole"] + o_["fn_rim"]), 100 * o_["fp_iso"] / o_["fp_total"]])
    per = np.array(per)
    cells = [f"{m:.1f}" if i == 2 else f"{m:.1f}$\\pm${s:.1f}" for i, (m, s) in enumerate(zip(per.mean(0), per.std(0, ddof=1)))]
    el.append(f"{name} & " + " & ".join(cells) + " \\\\")
el += ["\\bottomrule", "\\end{tabular}"]
(OUT / "error_table.tex").write_text("\n".join(el) + "\n", encoding="utf8")
# ------------------------------------------------------------------ the full tables of the paper (8 metrics, mean +- std), one file per table
import re

_spec = importlib.util.spec_from_file_location("lt", ROOT / "research" / "make_latex_tables.py")
lt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lt)
lt.PAPER = True


def tabular_only(t):
    """Strip the float wrapper (caption, resizebox) of a generated paper table and keep the tabular."""
    m = re.search(r"\\begin\{tabular\}.*\\end\{tabular\}", t, re.S)
    return m.group(0) + "\n"


for sc, _ in SCEN:
    rows_sc = [(n, D[sc][n]) for n in lt.ORDER]
    (OUT / f"table_{sc.lower()}.tex").write_text(tabular_only(lt.table("", rows_sc)), encoding="utf8")
rows_ab = [(lt.ABL[k], D["Ablation"][k]) for k in ["1", "2", "4", "5"]]
(OUT / "table_ablation.tex").write_text(tabular_only(lt.table("", rows_ab, first_col_name="Configuration", numbered=True)), encoding="utf8")
(OUT / "table_shift.tex").write_text(tabular_only(lt.shift_table(only=["CO"])), encoding="utf8")
(OUT / "table_errors.tex").write_text(tabular_only(lt.err_table()), encoding="utf8")

print("assets written to", OUT)

