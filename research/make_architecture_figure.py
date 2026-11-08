"""Block diagram of the whole method (training view), in the style of a TreeMortSeg-like architecture figure.
All connectors are horizontal / vertical (elbow) lines.

    python research/make_architecture_figure.py            # writes proposal/figures/architecture.{pdf,png}
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "proposal" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C = {"data": "#E8F1FB", "bank": "#FFF1D6", "style": "#FDE3E1", "net": "#E6F4EA", "enc": "#D1E9D8", "dec": "#EAF5E6",
     "loss": "#EDE4F7", "edge": "#44505C", "text": "#1F2933", "arrow": "#44505C", "grad": "#B3261E", "sg": "#6B7785"}

fig, ax = plt.subplots(figsize=(19, 9.6))
ax.set_xlim(0, 19)
ax.set_ylim(-0.4, 9.95)
ax.axis("off")


def box(x, y, w, h, text, fc, fs=9.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=C["edge"], lw=1.2, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=C["text"],
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(p0, p1, color=None, lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, lw=lw, color=color or C["arrow"], ls=ls,
                                 shrinkA=0, shrinkB=0))


def path(points, color=None, lw=1.5, ls="-"):
    """Elbow connector through `points`; the last segment carries the arrow head."""
    color = color or C["arrow"]
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="butt")
    arrow(points[-2], points[-1], color=color, lw=lw, ls=ls)


def group(x, y, w, h, title, ec="#8A96A3"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc="none", ec=ec, lw=1.3, ls=(0, (5, 3))))
    ax.text(x + 0.15, y + h - 0.2, title, ha="left", va="top", fontsize=10.5, fontweight="bold", color=C["text"])


ax.text(0.15, 9.75, "Training (source region only; nothing from the target region is used)", fontsize=12.5, fontweight="bold",
        color=C["text"], va="center")

# ============================================================ restyling group (top left)
group(0.15, 4.1, 8.25, 5.3, "Spectral restyling with the statistics bank")
box(0.4, 6.7, 2.4, 1.2, "Statistics bank  $\\mathcal{B}$\n$K$ source tiles $\\times$ 8 numbers\n($\\mu,\\sigma$ of R, G, B, NIR)", C["bank"], 9)   # centre y 7.3
box(0.4, 5.0, 2.4, 1.3, "Tile  $x$\n$5\\times224\\times224$\nR, G, B, NIR, NDVI", C["data"], 9.5)                                         # centre y 5.65
box(3.45, 5.0, 3.15, 3.0,
    "Restyle  ($\\mu,\\sigma$: statistics of $x$)\n$b=(\\mu_b,\\sigma_b)\\sim\\mathcal{B}$,  $\\alpha\\sim\\mathcal{U}(0,1)$\n"
    "$\\mu_t=\\alpha\\mu_b+(1-\\alpha)\\mu$\n$\\sigma_t=\\alpha\\sigma_b+(1-\\alpha)\\sigma$\n"
    "$x'=\\frac{x-\\mu}{\\sigma}\\,\\sigma_t+\\mu_t$\nNDVI recomputed, labels unchanged", C["style"], 9)
box(7.0, 5.9, 1.2, 1.2, "Restyled\ntile  $x'$", C["data"], 9.5)                                                                     # centre y 6.5
arrow((2.8, 7.3), (3.45, 7.3))
arrow((2.8, 5.65), (3.45, 5.65))
arrow((6.6, 6.5), (7.0, 6.5))

# ============================================================ segmentation network group (top right)
group(8.6, 4.1, 10.25, 5.3, "Segmentation network")
box(8.8, 5.5, 1.55, 2.0, "Preprocess\n$\\times2$ bilinear\n($448\\times448$)\nImageNet norm\non RGB", C["net"], 8.8)                  # centre y 6.5
box(10.6, 5.5, 1.4, 2.0, "Patch embed\n$14\\times14$,\n5 channels\n$N=1024$\ntokens", C["net"], 8.8)
arrow((8.2, 6.5), (8.8, 6.5))
arrow((10.35, 6.5), (10.6, 6.5))

ex, ew = 12.4, 1.9
ax.text(ex + ew / 2, 8.72, "DINOv2-B encoder", fontsize=10, fontweight="bold", ha="center", color=C["text"])
stack = [("layers 1-3", 7.8), ("layers 4-6", 7.18), ("layers 7-9", 6.56), ("layers 10-12", 5.94)]
for name, y in stack:
    box(ex, y, ew, 0.5, name, C["enc"], 9)
for (_, y), (_, y2) in zip(stack[:-1], stack[1:]):
    arrow((ex + ew / 2, y), (ex + ew / 2, y2 + 0.5), lw=1.0)
path([(12.0, 6.5), (12.2, 6.5), (12.2, 8.05), (ex, 8.05)], lw=1.3)          # patch embed -> first layers (elbow)

dx, dw = 15.0, 3.65
ax.text(dx + dw / 2, 8.72, "Multi-level decoder", fontsize=10, fontweight="bold", ha="center", color=C["text"])
box(dx, 7.45, dw, 1.1, "$1\\times1$ conv + norm + act per tap\n(768 $\\to$ 128 each)", C["dec"], 8.8)
box(dx, 6.55, dw, 0.62, "concat (512)  $\\to$  conv $3\\times3$ (192)", C["dec"], 8.8)
box(dx, 5.75, dw, 0.55, "up $\\times2$ + conv $3\\times3$ (96)", C["dec"], 8.8)
box(dx, 4.95, dw, 0.55, "up $\\times2$ + conv $3\\times3$ (48)", C["dec"], 8.8)
box(dx, 4.25, dw, 0.5, "conv $1\\times1$, resize $224^2$, $\\sigma(\\cdot)$", C["dec"], 8.8)
for y0, y1 in [(7.45, 7.17), (6.55, 6.3), (5.75, 5.5), (4.95, 4.75)]:
    arrow((dx + dw / 2, y0), (dx + dw / 2, y1), lw=1.0)
# four taps merge on a vertical bus, then enter the first decoder box
bus = 14.65
for _, y in stack:
    ax.plot([ex + ew, bus], [y + 0.25, y + 0.25], color=C["sg"], lw=1.1, solid_capstyle="butt")
ax.plot([bus, bus], [stack[-1][1] + 0.25, 8.05], color=C["sg"], lw=1.1, solid_capstyle="butt")
arrow((bus, 8.05), (dx, 8.05), color=C["sg"], lw=1.1)
ax.text(13.35, 5.55, "taps at layers 3, 6, 9, 12", fontsize=8, color=C["sg"], ha="center")

# ============================================================ clean branch group (bottom left)
group(0.15, 0.2, 8.25, 3.6, "Clean view (reference)")
box(0.4, 2.0, 2.4, 1.1, "Tile  $x$\n(unchanged)", C["data"], 9.5)
box(3.45, 1.85, 3.15, 1.4, "Same network, shared weights\n(forward only, stop-gradient)", C["net"], 9.5, ls=(0, (4, 2)))
box(7.0, 2.2, 1.2, 0.7, "$p$", C["data"], 10.5)
arrow((2.8, 2.55), (3.45, 2.55))
arrow((6.6, 2.55), (7.0, 2.55))
arrow((1.6, 5.0), (1.6, 3.1), lw=1.3)
ax.text(1.72, 4.1, "$x$", fontsize=10, color=C["text"])
ax.text(4.85, 3.5, "$\\operatorname{sg}(p)$: no gradient to this branch", fontsize=8.5, color=C["sg"], ha="center")
ax.text(4.3, 0.75, "Inference: only the clean tile  $x\\to$ network $\\to\\sigma(\\cdot)\\to$ dead-tree map\n(threshold 0.5); no restyling, no extra cost.",
        fontsize=9.3, color=C["text"], ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="#F6F8FA", ec="#8A96A3", lw=1.0))

# ============================================================ objective group (bottom right)
group(8.6, 0.2, 10.25, 3.6, "Objective")
box(9.3, 0.55, 3.3, 1.35, "Consistency loss\n$\\mathcal{L}_{\\mathrm{cons}}=\\frac{1}{|\\Omega|}\\sum_{i\\in\\Omega}(p'_i-\\operatorname{sg}(p_i))^2$", C["loss"], 9)
box(13.0, 0.55, 2.9, 1.35, "Mask loss\n$\\mathcal{L}_{\\mathrm{mask}}(p',y)$\nBCE + Dice, valid pixels", C["loss"], 9)
box(16.3, 0.55, 2.35, 1.35, "$\\mathcal{L}=\\mathcal{L}_{\\mathrm{mask}}+\\lambda\\,\\mathcal{L}_{\\mathrm{cons}}$\n$\\lambda=1$", C["loss"], 9.8, bold=True)
box(13.0, 2.3, 1.3, 0.55, "Label  $y$", C["data"], 9.5)
pcx = dx + dw / 2                                                          # 16.825: p' sits directly under the decoder output
box(pcx - 0.8, 2.85, 1.6, 0.6, "$p'$  (restyled)", C["data"], 9.5)
arrow((pcx, 4.25), (pcx, 3.45), lw=1.4, color=C["grad"])                    # decoder output -> p'
path([(pcx - 0.8, 3.15), (11.0, 3.15), (11.0, 1.9)], lw=1.4, color=C["grad"])   # p' -> consistency loss
path([(pcx, 2.85), (pcx, 2.45), (15.2, 2.45), (15.2, 1.9)], lw=1.4, color=C["grad"])  # p' -> mask loss
arrow((13.65, 2.3), (13.65, 1.9), lw=1.3)                                   # label -> mask loss
path([(8.2, 2.55), (8.95, 2.55), (8.95, 1.22), (9.3, 1.22)], lw=1.4, color=C["sg"], ls=(0, (4, 2)))   # p -> consistency (stop-gradient)
arrow((15.9, 1.2), (16.3, 1.2), lw=1.3)                                     # mask -> total
path([(11.0, 0.55), (11.0, 0.38), (17.5, 0.38), (17.5, 0.55)], lw=1.3)      # consistency -> total

ax.text(0.15, -0.28, "red arrows: gradient flows;  grey dashed: stop-gradient reference.", fontsize=8.5, color=C["sg"])
fig.savefig(OUT / "architecture.pdf", bbox_inches="tight")
fig.savefig(OUT / "architecture.png", dpi=200, bbox_inches="tight")
print("saved", OUT)
