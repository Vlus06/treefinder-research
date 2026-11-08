"""Architecture figures in the style of the TreeMortSeg paper: (1) an overview figure with image thumbnails, (2) a detailed figure
with 3-D feature-map blocks. Thumbnails of the tile / restyled tile / label are real (read from datasets/tiles224_v3); the two
probability-map thumbnails (p, p') are ILLUSTRATIVE (blurred label), to be replaced by real predictions of the trained model.

    python research/make_architecture_figures_v2.py     # writes proposal/figures/arch_overview.{pdf,png}, arch_detail.{pdf,png}
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Rectangle
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "proposal" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TILES = ROOT / "datasets" / "tiles224_v3"

C = {"data": "#E8F1FB", "bank": "#FFF1D6", "style": "#FDE3E1", "net": "#E6F4EA", "enc": "#D1E9D8", "dec": "#EAF5E6",
     "loss": "#EDE4F7", "edge": "#44505C", "text": "#1F2933", "arrow": "#44505C", "grad": "#B3261E", "sg": "#6B7785",
     "orange": "#F5A623"}
SLAB = {"enc": "#9ACBA6", "enc_t": "#C6E3CD", "enc_s": "#7DB38C",
        "c1": "#F4B6C2", "c2": "#F9D49A", "c3": "#A9D6A5", "c4": "#A9C6EA",
        "dec": "#B9DDB4", "dec_t": "#DCEED9", "dec_s": "#98C792",
        "in": "#B9CDEA", "in_t": "#D8E4F5", "in_s": "#9DB6DB",
        "out": "#F7D6A0", "out_t": "#FBE8C6", "out_s": "#E8BD78"}


# ------------------------------------------------------------------ data helpers
def read_tile(name):
    """Minimal reader for the uncompressed, chunky, 5-sample 224x224 GeoTIFFs of TreeFinder (R, G, B, NIR, label)."""
    import struct
    b = (TILES / name).read_bytes()
    off = struct.unpack("<I", b[4:8])[0]
    n = struct.unpack("<H", b[off:off + 2])[0]
    tags = {}
    for i in range(n):
        e = b[off + 2 + 12 * i: off + 14 + 12 * i]
        tag, typ, cnt = struct.unpack("<HHI", e[:8])
        size = {3: 2, 4: 4}.get(typ, 1)
        data = e[8:8 + size * cnt] if size * cnt <= 4 else b[struct.unpack("<I", e[8:12])[0]:][:size * cnt]
        if typ in (3, 4):
            tags[tag] = struct.unpack("<" + ("H" if typ == 3 else "I") * cnt, data)
    offs, cnts = tags[273], tags[279]
    raw = b"".join(b[o:o + c] for o, c in zip(offs, cnts))
    return np.frombuffer(raw, np.uint8).reshape(224, 224, 5)


def tile_stats(a):
    img = a[..., :4].astype(np.float32)
    valid = (a[..., 4] != 255) & (img.sum(-1) > 0)
    mu = np.array([img[..., c][valid].mean() for c in range(4)])
    sd = np.array([img[..., c][valid].std() for c in range(4)])
    return mu, sd


def rgb(a):
    return np.clip(a[..., :3].astype(np.float32) / 255.0, 0, 1)


def restyle(a, mu_b, sd_b, alpha):
    img = a[..., :4].astype(np.float32)
    mu, sd = tile_stats(a)
    mt, st = alpha * mu_b + (1 - alpha) * mu, alpha * sd_b + (1 - alpha) * sd
    out = (img - mu) / (sd + 1e-4) * st + mt
    return np.clip(out[..., :3] / 255.0, 0, 1)


def mask_img(a, blur=0.0):
    m = (a[..., 4] == 1).astype(np.float32)
    if blur > 0:
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32) / 255.0
        m = np.clip(m * 2.2, 0, 1)
    bg = np.array([0.94, 0.94, 0.94])
    fg = np.array([0.96, 0.65, 0.14])
    return bg[None, None] * (1 - m[..., None]) + fg[None, None] * m[..., None]


def pick_tiles():
    rows = list(csv.DictReader(open(ROOT / "datasets" / "tile_info224_v3.csv")))
    good = [r for r in rows if int(r["NoDataSize"]) == 0]
    main = [r for r in good if r["State"] == "CO" and 700 <= int(r["LabelSize"]) <= 2500]
    main = main[len(main) // 3] if main else good[0]
    cand = {}
    for r in good[::97]:
        cand.setdefault(r["State"], r)
    bank = []
    for st in ["FL", "CA", "NY", "MN", "AZ", "OR", "ME", "GA"]:
        if st in cand:
            bank.append(cand[st])
    return main["FileName"], [r["FileName"] for r in bank]


# ------------------------------------------------------------------ drawing helpers
def make_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc, fs=9.5, bold=False, ls="-", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=C["edge"], lw=lw, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=C["text"], fontweight="bold" if bold else "normal",
            linespacing=1.35, zorder=3)


def arrow(ax, p0, p1, color=None, lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, lw=lw, color=color or C["arrow"], ls=ls, shrinkA=0,
                                 shrinkB=0, zorder=4))


def path(ax, pts, color=None, lw=1.5, ls="-"):
    color = color or C["arrow"]
    xs, ys = zip(*pts[:-1])
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=4)
    arrow(ax, pts[-2], pts[-1], color, lw, ls)


def group(ax, x, y, w, h, title, ec="#8A96A3"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc="none", ec=ec, lw=1.3, ls=(0, (5, 3)), zorder=1))
    ax.text(x + 0.15, y + h - 0.15, title, ha="left", va="top", fontsize=10.5, fontweight="bold", color=C["text"])


def thumb(ax, img, x, y, s, frame="#44505C", lw=1.2):
    ax.imshow(img, extent=[x, x + s, y, y + s], aspect="auto", zorder=3, interpolation="nearest" if img.shape[0] < 20 else "bilinear")
    ax.add_patch(Rectangle((x, y), s, s, fc="none", ec=frame, lw=lw, zorder=5))


def slab(ax, x, y, w, h, d, col, label=None, fs=8.5, label_pos="below", z=3):
    """Isometric feature-map block: front face (x, y, w, h), depth d drawn to the upper right."""
    dx, dy = d * 0.6, d * 0.42
    front = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], fc=col[0], ec=C["edge"], lw=0.9, zorder=z)
    top = Polygon([(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)], fc=col[1], ec=C["edge"], lw=0.9, zorder=z)
    side = Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)], fc=col[2], ec=C["edge"], lw=0.9, zorder=z)
    for p in (front, top, side):
        ax.add_patch(p)
    if label:
        ty = y - 0.18 if label_pos == "below" else y + h + dy + 0.12
        ax.text(x + (w + dx) / 2, ty, label, ha="center", va="top" if label_pos == "below" else "bottom", fontsize=fs, color=C["text"],
                linespacing=1.25, zorder=6)
    return (x + w + dx, y + h / 2)                                     # right-middle anchor for connectors


# ================================================================== figure 1: overview
def prep_tiles(main, bank_names):
    a = read_tile(main)
    banks = [read_tile(n) for n in bank_names[:4]]
    stats = [tile_stats(t) for t in banks]
    mu_a, _ = tile_stats(a)
    j = 3 if len(banks) > 3 else int(np.argmax([np.linalg.norm(s[0][:3] - mu_a[:3]) for s in stats]))   # yellowish entry: clearest illustration
    return a, banks, stats, j


def overview(a, banks, stats, j):
    xr = restyle(a, stats[j][0], stats[j][1], 0.85)
    fig, ax = make_ax(19, 7.6)
    ax.text(0.15, 7.35, "Training (source region only; nothing from the target region is used)", fontsize=12.5, fontweight="bold",
            color=C["text"], va="center")
    # statistics bank
    group(ax, 0.15, 5.45, 5.4, 1.65, "Statistics bank $\\mathcal{B}$")
    for k, t in enumerate(banks):
        thumb(ax, rgb(t), 0.4 + 0.4 * k, 5.55 + 0.06 * k, 0.85)
    ax.text(3.35, 6.1, "$K$ source tiles;  per tile\n$\\mu,\\sigma$ of R, G, B, NIR\n(8 numbers)", fontsize=8.8, ha="left", va="center", color=C["text"])
    # input tile and restyling
    thumb(ax, rgb(a), 0.5, 3.0, 1.6)
    ax.text(1.3, 2.8, "Tile $x$\n(R, G, B, NIR, NDVI)", ha="center", va="top", fontsize=9, color=C["text"])
    box(ax, 3.1, 4.2, 1.9, 1.0, "Spectral\nrestyling", C["style"], 10, bold=True)
    path(ax, [(2.1, 3.8), (2.6, 3.8), (2.6, 4.45), (3.1, 4.45)])
    path(ax, [(4.05, 5.45), (4.05, 5.2)])
    thumb(ax, xr, 5.75, 3.95, 1.5)
    ax.text(6.5, 3.75, "Restyled tile $x'$", ha="center", va="top", fontsize=9, color=C["text"])
    arrow(ax, (5.0, 4.7), (5.75, 4.7))
    ax.text(5.38, 4.85, "$\\alpha$", fontsize=9, color=C["sg"], ha="center")
    # shared network: two rows with the same weights
    group(ax, 8.0, 0.9, 4.35, 5.7, "Segmentation network (shared weights)")
    box(ax, 8.3, 4.1, 3.75, 1.2, "DINOv2-B encoder\n+ multi-level decoder", C["net"], 10, bold=True)
    box(ax, 8.3, 1.4, 3.75, 1.2, "DINOv2-B encoder\n+ multi-level decoder", C["net"], 10, bold=True, ls=(0, (4, 2)))
    ax.annotate("", xy=(10.17, 2.62), xytext=(10.17, 4.08), arrowprops=dict(arrowstyle="<->", color=C["sg"], lw=1.3))
    ax.text(10.35, 3.35, "same weights", fontsize=8.5, color=C["sg"], ha="left", va="center")
    arrow(ax, (7.25, 4.7), (8.3, 4.7), color=C["grad"], lw=1.6)
    ax.text(7.75, 4.9, "gradient", fontsize=8, color=C["grad"], ha="center")
    # clean branch
    thumb(ax, rgb(a), 5.9, 1.4, 1.2)
    ax.text(6.5, 1.2, "Clean tile $x$ (reference)", ha="center", va="top", fontsize=9, color=C["text"])
    path(ax, [(1.3, 3.0), (1.3, 2.0), (5.9, 2.0)], lw=1.4)
    arrow(ax, (7.1, 2.0), (8.3, 2.0), color=C["sg"], lw=1.6, ls=(0, (4, 2)))
    ax.text(7.7, 1.8, "no gradient", fontsize=8, color=C["sg"], ha="center", va="top")
    # outputs
    mask = mask_img(a, 1.6)
    thumb(ax, mask, 13.0, 4.0, 1.4)
    ax.text(13.7, 5.55, "$p'$ (restyled)", ha="center", va="bottom", fontsize=9, color=C["text"])
    thumb(ax, mask, 13.0, 1.3, 1.4, frame=C["sg"])
    ax.text(13.7, 2.85, "$p$ (clean)", ha="center", va="bottom", fontsize=9, color=C["text"])
    arrow(ax, (12.05, 4.7), (13.0, 4.7), color=C["grad"], lw=1.6)
    arrow(ax, (12.05, 2.0), (13.0, 2.0), color=C["sg"], lw=1.6, ls=(0, (4, 2)))
    # losses
    box(ax, 15.2, 4.25, 1.9, 0.9, "$\\mathcal{L}_{\\mathrm{mask}}$\nBCE + Dice", C["loss"], 10)
    thumb(ax, mask_img(a, 0), 15.65, 5.7, 1.0)
    ax.text(16.8, 6.2, "label $y$", fontsize=9, ha="left", va="center", color=C["text"])
    arrow(ax, (16.15, 5.7), (16.15, 5.15), lw=1.3)
    arrow(ax, (14.4, 4.7), (15.2, 4.7), color=C["grad"], lw=1.6)
    box(ax, 15.2, 1.55, 1.9, 0.9, "$\\mathcal{L}_{\\mathrm{cons}}$\n$(p'-\\operatorname{sg}(p))^2$", C["loss"], 10)
    arrow(ax, (14.4, 2.0), (15.2, 2.0), color=C["sg"], lw=1.6, ls=(0, (4, 2)))
    ax.add_patch(Circle((14.75, 4.7), 0.05, fc=C["grad"], ec=C["grad"], zorder=5))
    path(ax, [(14.75, 4.7), (14.75, 3.3), (16.15, 3.3), (16.15, 2.45)], color=C["grad"], lw=1.6)
    # total loss
    box(ax, 17.45, 2.85, 1.45, 1.0, "Total\nloss", C["loss"], 10, bold=True)
    path(ax, [(17.1, 4.7), (18.17, 4.7), (18.17, 3.85)])
    path(ax, [(17.1, 2.0), (18.17, 2.0), (18.17, 2.85)])
    ax.text(18.3, 1.7, "$\\lambda=1$", fontsize=8.5, color=C["sg"], ha="left")
    ax.text(0.15, 0.2, "Inference: only the clean tile $x$ goes through the network ($\\to\\sigma(\\cdot)\\to$ dead-tree map, threshold 0.5); no restyling.   "
            "Probability-map thumbnails are illustrative.", fontsize=8.5, color=C["sg"])
    fig.savefig(OUT / "arch_overview.pdf", bbox_inches="tight")
    fig.savefig(OUT / "arch_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return xr


# ================================================================== figure 2: detailed network with 3-D blocks
def detail(a, xr):
    fig, ax = make_ax(20.8, 7.3)
    group(ax, 0.15, 0.15, 20.5, 7.0, "Network (detail): DINOv2-B encoder with four taps and multi-level decoder")
    yc = 3.3

    def S(x, w, h, d, col, label=None, fs=8.2, yy=None):
        y = (yc if yy is None else yy) - h / 2
        slab(ax, x, y, w, h, d, col, None)
        ext = w + 0.6 * d
        if label:
            ax.text(x + ext / 2, y - 0.22, label, ha="center", va="top", fontsize=fs, color=C["text"], linespacing=1.25, zorder=6)
        return ext

    ENC = (SLAB["enc"], SLAB["enc_t"], SLAB["enc_s"])
    DEC = (SLAB["dec"], SLAB["dec_t"], SLAB["dec_s"])
    # input, preprocessing, patch embedding
    thumb(ax, xr, 0.45, yc - 0.65, 1.3)
    ax.text(1.1, yc - 0.8, "$x'$,  $5\\times224^2$", ha="center", va="top", fontsize=8.5, color=C["text"])
    box(ax, 2.1, yc - 0.5, 1.35, 1.0, "$\\times2$ up\nImageNet norm\n(RGB)", C["net"], 8.3)
    box(ax, 3.8, yc - 0.5, 1.35, 1.0, "Patch embed\n$14\\times14$\n5 channels", C["net"], 8.3)
    arrow(ax, (1.75, yc), (2.1, yc))
    arrow(ax, (3.45, yc), (3.8, yc))
    # encoder slabs
    exs = [5.5, 7.1, 8.7, 10.3]
    names = ["layers 1-3", "layers 4-6", "layers 7-9", "layers 10-12"]
    taps = ["3", "6", "9", "12"]
    arrow(ax, (5.15, yc), (exs[0], yc))
    for k, (x, nm, tp) in enumerate(zip(exs, names, taps)):
        ext = S(x, 0.7, 0.7, 0.8, ENC, f"{nm}\n$\\ell={tp}$")
        if k < 3:
            arrow(ax, (x + ext, yc), (exs[k + 1], yc), lw=1.2)
    ax.text(8.5, 1.3, "DINOv2-B encoder: 768 channels, $32\\times32$ tokens", fontsize=9.5, fontweight="bold", ha="center")
    # taps: 1x1 conv (128 ch) slabs above each encoder slab, four colours
    cols = ["c1", "c2", "c3", "c4"]
    ytap = 5.0
    for x, ck in zip(exs, cols):
        xc = x + 0.37
        slab(ax, xc, ytap - 0.25, 0.5, 0.5, 0.5, (SLAB[ck], "#FFFFFF", SLAB[ck]), None)
        arrow(ax, (xc + 0.25, yc + 0.35 + 0.34 + 0.05), (xc + 0.25, ytap - 0.25), color=C["sg"], lw=1.1)
    ax.text(0.5, 4.9, "taps: $1\\times1$ conv + norm + act,\n768 $\\to$ 128 channels (one per tap)", fontsize=8.8, ha="left", va="center", color=C["text"])
    # bus above the taps to the concat block
    cx = 12.4
    bus_y = 6.45
    cxt = cx + (0.7 + 0.6) / 2
    for x in exs:
        xc = x + 0.37
        ax.plot([xc + 0.4, xc + 0.4], [ytap + 0.25 + 0.21, bus_y], color=C["sg"], lw=1.1, zorder=4)
    ax.plot([exs[0] + 0.77, cxt], [bus_y, bus_y], color=C["sg"], lw=1.1, zorder=4)
    arrow(ax, (cxt, bus_y), (cxt, yc + 0.45 + 0.4 + 0.02), color=C["sg"], lw=1.1)
    # concat + decoder
    ext = S(cx, 0.7, 0.9, 1.0, (SLAB["c2"], "#FBE3BB", SLAB["c2"]), "concat\n512 ch, $32^2$")
    x1 = cx + ext + 0.4
    arrow(ax, (cx + ext, yc), (x1, yc), lw=1.2)
    ext1 = S(x1, 0.55, 0.65, 0.6, DEC, "conv $3\\times3$\n192 ch, $32^2$")
    x2 = x1 + ext1 + 0.4
    arrow(ax, (x1 + ext1, yc), (x2, yc), lw=1.2)
    ext2 = S(x2, 0.8, 0.8, 0.5, DEC, "up $\\times2$ + conv\n96 ch, $64^2$")
    x3 = x2 + ext2 + 0.4
    arrow(ax, (x2 + ext2, yc), (x3, yc), lw=1.2)
    ext3 = S(x3, 1.05, 1.05, 0.3, DEC, "up $\\times2$ + conv\n48 ch, $128^2$")
    ax.text((x1 + x3 + ext3) / 2, 1.3, "multi-level decoder", fontsize=9.5, fontweight="bold", ha="center")
    x4 = x3 + ext3 + 0.45
    thumb(ax, mask_img(a, 1.6), x4, yc - 0.65, 1.3)
    arrow(ax, (x3 + ext3, yc), (x4, yc), lw=1.3, color=C["grad"])
    ax.text(x4 + 0.65, yc - 0.8, "$1\\times1$ conv, resize $224^2$,\n$\\sigma(\\cdot)\\to p'$", ha="center", va="top", fontsize=8.3, color=C["text"])
    fig.savefig(OUT / "arch_detail.pdf", bbox_inches="tight")
    fig.savefig(OUT / "arch_detail.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================== figure 3: statistics bank and restyling with real tiles
def restyle_fig(a, banks, stats, j):
    fig, ax = make_ax(19, 5.6)
    group(ax, 0.15, 0.15, 18.7, 5.3, "Spectral restyling with the statistics bank (real tiles)")
    # bank
    ax.text(2.0, 4.6, "Statistics bank $\\mathcal{B}$", fontsize=10, ha="center", fontweight="bold")
    for k, (t, st) in enumerate(zip(banks, stats)):
        thumb(ax, rgb(t), 0.5 + 0.95 * k, 3.1, 0.85)
        ax.text(0.5 + 0.95 * k + 0.42, 3.0, "$b_%d$" % (k + 1), fontsize=8.5, ha="center", va="top")
    ax.text(2.4, 2.35, "$b_k=(\\mu_R,\\mu_G,\\mu_B,\\mu_N,\\sigma_R,\\sigma_G,\\sigma_B,\\sigma_N)$", fontsize=8.8, ha="center", color=C["text"])
    # chosen entry and source tile
    thumb(ax, rgb(banks[j]), 5.0, 2.9, 1.4)
    ax.text(5.7, 2.7, "bank entry $b$\n(only its $\\mu_b,\\sigma_b$ is used)", ha="center", va="top", fontsize=8.8)
    path(ax, [(3.9, 3.5), (4.45, 3.5), (4.45, 3.6), (5.0, 3.6)]) if False else arrow(ax, (3.9, 3.55), (5.0, 3.55), lw=1.2)
    thumb(ax, rgb(a), 5.0, 0.7, 1.4)
    ax.text(5.7, 0.5, "source tile $x$ ($\\mu,\\sigma$)", ha="center", va="top", fontsize=8.8)
    # alpha sweep
    xs = [8.2, 10.4, 12.6, 14.8]
    alphas = [0.0, 0.35, 0.7, 1.0]
    for x, al in zip(xs, alphas):
        img = restyle(a, stats[j][0], stats[j][1], al)
        thumb(ax, img, x, 1.9, 1.7)
        ax.text(x + 0.85, 1.7, "$\\alpha=%.2f$" % al, ha="center", va="top", fontsize=9.5)
    ax.text(11.9, 4.0, "$x'=\\frac{x-\\mu}{\\sigma}\\,\\sigma_t+\\mu_t,\\;\\;\\mu_t=\\alpha\\mu_b+(1-\\alpha)\\mu,\\;\\;\\sigma_t=\\alpha\\sigma_b+(1-\\alpha)\\sigma$",
            fontsize=10, ha="center")
    ax.text(8.2, 3.85, "$\\alpha=0$: unchanged", fontsize=8.5, color=C["sg"], ha="left", va="center")
    ax.text(16.5, 3.85, "$\\alpha=1$: style of $b$", fontsize=8.5, color=C["sg"], ha="right", va="center")
    arrow(ax, (6.4, 1.45), (8.2, 2.35), lw=1.2) if False else None
    arrow(ax, (6.5, 3.55), (7.7, 3.0), lw=1.0, color=C["sg"]) if False else None
    arrow(ax, (6.5, 1.4), (8.0, 1.4), lw=1.3, color="#FFFFFF") if False else None
    path(ax, [(6.45, 3.6), (7.7, 3.6), (7.7, 2.75), (8.2, 2.75)], lw=1.3)
    path(ax, [(6.45, 1.4), (7.7, 1.4), (7.7, 2.2), (8.2, 2.2)], lw=1.3) if False else None
    ax.text(17.7, 2.75, "NDVI recomputed\nlabels unchanged\n$\\alpha\\sim\\mathcal{U}(0,1)$\nper tile", fontsize=8.8, ha="center", va="center")
    fig.savefig(OUT / "arch_restyle.pdf", bbox_inches="tight")
    fig.savefig(OUT / "arch_restyle.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main_tile, bank_tiles = pick_tiles()
    print("tile", main_tile, "bank", bank_tiles[:4])
    a, banks, stats, j = prep_tiles(main_tile, bank_tiles)
    xr = overview(a, banks, stats, j)
    detail(a, xr)
    restyle_fig(a, banks, stats, j)
    print("saved", OUT)
