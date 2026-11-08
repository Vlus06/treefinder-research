"""Replace the dataset-baseline rows of the paper's Table 1 with the lr 1e-4 re-runs read from the authors' logs.

    python research/update_table_lr4.py --logs research/reports/lr4_logs --table <paper>/sections/main_results_table.tex --scenarios Site [Random]

Only rows whose three seeds (2025, 2024, 2023) have a "Combined metrics" line are replaced; bold / underline are recomputed per scenario block.
ViT is taken from the min-epoch re-runs (tags *_minep) when present, otherwise left unchanged.
"""
import argparse
import ast
import glob
import os
import re

import numpy as np

B = chr(92)
KEYS = ["class1_f1", "class1_precision", "class1_recall", "class1_iou", "res_f1_macro", "res_precision_macro", "res_recall_macro", "res_iou_macro"]
MODELS = {"unet": "U-Net", "deeplabv3": "DeepLabV3+", "vitseg3": "ViT", "segformer_large3": "SegFormer", "mask2former": "Mask2Former", "dofa": "DOFA"}
CFG = {"Random": ("res_random_lr4", "res_random_lr4_minep"), "Site": ("res_random_site_fix_lr4", "res_random_site_fix_lr4_minep")}
BLOCK = {"Random": "Random", "Site": "Site"}
CELL = re.compile(r"([0-9]+\.[0-9]+)" + re.escape(B) + r"pm([0-9]+\.[0-9]+)")


def metrics(path):
    line = [l for l in open(path, errors="ignore") if "Combined metrics" in l]
    if not line:
        return None
    d = ast.literal_eval(re.sub(r"np\.float64\(([^)]*)\)", r"\1", line[-1].split("Combined metrics: ")[1]))
    return [100 * d[k] for k in KEYS]


def runs(logs, cfg, model):
    out = {}
    for f in glob.glob(os.path.join(logs, f"code_*_{cfg}_*_{model}_s*.log")):
        m = metrics(f)
        if m:
            out[int(re.search(r"_s(\d+)\.log$", f).group(1))] = m
    return out


def fmt(v, s, mark):
    if mark == "b":
        return "$" + B + "mathbf{" + f"{v:.2f}" + B + "pm" + f"{s:.2f}" + "}$"
    core = "$" + f"{v:.2f}" + B + "pm" + f"{s:.2f}" + "$"
    return B + "underline{" + core + "}" if mark == "u" else core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--scenarios", nargs="+", default=["Site"])
    a = ap.parse_args()
    lines = open(a.table, encoding="utf8").read().split("\n")
    cur, block_idx = None, {}
    for i, ln in enumerate(lines):
        m = re.search(re.escape(B) + r"textit\{([A-Za-z]+)", ln)
        if ln.startswith(B + "multicolumn{9}"):
            cur = m.group(1)
            block_idx[cur] = []
        elif cur and " & " in ln and "pm" in ln:
            block_idx[cur].append(i)
    for sc in a.scenarios:
        new = {}
        for model, name in MODELS.items():
            r = runs(a.logs, CFG[sc][1], model) if model == "vitseg3" else {}
            if len(r) < 3:
                r = runs(a.logs, CFG[sc][0], model) if model != "vitseg3" else r
            if len(r) == 3:
                v = np.array([r[s] for s in (2025, 2024, 2023)])
                new[name] = list(zip(v.mean(0), v.std(0, ddof=1)))
                print(sc, name, "F1 %.2f +- %.2f" % new[name][0])
            else:
                print(sc, name, "kept (", len(r), "of 3 seeds)")
        rows = []
        for i in block_idx[BLOCK[sc]]:
            name, rest = lines[i].split(" & ", 1)
            cells = new.get(name) or [(float(x), float(y)) for x, y in CELL.findall(rest)]
            rows.append((i, name, cells))
        for j in range(8):
            vals = sorted({round(c[j][0], 2) for _, _, c in rows}, reverse=True)
            best, second = vals[0], vals[1]
            for k, (i, name, cells) in enumerate(rows):
                pass
        for i, name, cells in rows:
            out = []
            for j, (v, s) in enumerate(cells):
                vals = sorted({round(c[j][0], 2) for _, _, c in rows}, reverse=True)
                mark = "b" if round(v, 2) == vals[0] else ("u" if round(v, 2) == vals[1] else "")
                out.append(fmt(v, s, mark))
            lines[i] = name + " & " + " & ".join(out) + B + B
    open(a.table, "w", encoding="utf8", newline="").write("\n".join(lines))


if __name__ == "__main__":
    main()
