"""Dot plot of dead-tree F1 and macro-F1 (mean +- sample SD over three seeds) for every model and scenario, read from the paper's Table 1.

    python research/make_comparison_fig.py --table <paper>/sections/main_results_table.tex --out <paper>/images/comparison.pdf
"""
import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

B = chr(92)
CELL = re.compile(r"([0-9]+\.[0-9]+)" + re.escape(B) + r"pm([0-9]+\.[0-9]+)")
TITLES = {"Random": "Random", "CO": "CO → non-CO", "East": "East → West", "West": "West → East", "Site": "Site-disjoint"}
BLUE, ORANGE, GRID, SHADE, TXT = "#2f7ab9", "#e0812b", "#dfe3e8", "#eef2f6", "#333a40"


def label(name):
    name = name.replace("(Ours, ", "(").replace("--", "–").replace("ConvNeXt-", "ConvNeXt–")
    return re.sub(r"\$m\{=\}([0-9.]+)\$", r"$m = \1$", name)


def parse(path):
    blocks, cur = [], None
    for ln in open(path, encoding="utf8"):
        if ln.startswith(B + "multicolumn{9}"):
            key = re.search(re.escape(B) + r"textit\{([A-Za-z]+)", ln).group(1)
            cur = (TITLES[key], [])
            blocks.append(cur)
        elif cur and " & " in ln and "pm" in ln:
            name, rest = ln.split(" & ", 1)
            c = [(float(v), float(s)) for v, s in CELL.findall(rest)]
            cur[1].append((label(name), c[0], c[4]))  # dead-tree F1, macro F1
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    blocks = parse(a.table)
    names = [r[0] for r in blocks[0][1]]
    n = len(names)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9})
    fig, axes = plt.subplots(1, len(blocks), figsize=(7.8, 3.75), sharey=True)
    for k, (ax, (title, rows)) in enumerate(zip(axes, blocks)):
        assert [r[0] for r in rows] == names
        for i, (name, (fd, sd), (fm, sm)) in enumerate(rows):
            y = n - 1 - i
            if name.startswith("iTwin"):
                ax.axhspan(y - 0.5, y + 0.5, color=SHADE, zorder=0, lw=0)
            ax.errorbar(fd, y + 0.12, xerr=sd, fmt="o", ms=4.2, color=BLUE, ecolor=BLUE, elinewidth=1, capsize=2, zorder=3)
            ax.errorbar(fm, y - 0.12, xerr=sm, fmt="s", ms=4.0, color=ORANGE, ecolor=ORANGE, elinewidth=1, capsize=2, zorder=3)
        ax.set_xlim(-2, 102)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names[::-1])
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.set_title(title, color=TXT, fontsize=9.5, pad=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_visible(k == 0)
        ax.spines["bottom"].set_color("#888888")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8.5)
    fig.supxlabel("F1 (%)", color=TXT, fontsize=9.5, y=0.1)
    h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=4.5, label="Dead-tree F1"),
         plt.Line2D([], [], marker="s", ls="", color=ORANGE, ms=4.5, label="Macro-F1 (background + dead tree)")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.55, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.6)
    fig.savefig(a.out, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
