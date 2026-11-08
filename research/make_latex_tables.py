"""Turn table_metrics.json into the LaTeX result tables of the proposal and splice them between the markers
'% BEGIN RESULTS' and '% END RESULTS' of the .tex file.

    python research/make_latex_tables.py --json table_metrics.json --tex proposal/proposal.tex
Rows with fewer than three finished runs are shown as 'đang chạy'. Best value per column is bold, second best underlined
(computed over the finished rows of that table).
"""
import argparse
import json
import re
from pathlib import Path

COLS = ["dead_f1", "dead_p", "dead_r", "dead_iou", "all_f1", "all_p", "all_r", "all_iou"]
ORDER = ["U-Net", "DeepLabV3+", "ViT", "SegFormer", "Mask2Former", "DOFA", "ConvNeXt-UPerNet", "TreeMortSeg",
         "DINOv2 + TreeMortBank (m=0.1)", "DINOv2 + TreeMortBank (m=0.01)"]
TITLES = {"Random": "Kịch bản ngẫu nhiên (chia theo tile).",
          "CO": "Kịch bản CO$\\to$không-CO.",
          "EW": "Kịch bản Đông$\\to$Tây.",
          "WE": "Kịch bản Tây$\\to$Đông.",
          "Site": "Kịch bản site-disjoint (một phân vùng cố định)."}
ABL = {"1": "DINOv2 + decoder ($m=0{,}1$)",
       "2": "$+$ lower encoder learning rate ($m=0{,}01$)",
       "3": "$+$ photometric augmentation (gain, bias, gamma)",
       "4": "$+$ statistics bank",
       "5": "$+$ statistics bank $+$ consistency loss (full \\method{}, Ours)"}


def dec(x):
    return f"{x:.2f}".replace(".", "{,}")


LABELS = {"DINOv2": "DINOv2 + decoder ($m=0{,}1$) (Ours)",
          "DINOv2 (m=0.01)": "DINOv2 + decoder ($m=0{,}01$) (Ours)",
          "DINOv2 + DPT (m=0.01)": "DINOv2 + DPT ($m=0{,}01$)",
          "DINOv2 + TreeMortBank (m=0.01)": "\\method{} (Ours)",
          "DINOv2 + TreeMortBank (m=0.1)": "\\method{} ($m=0{,}1$) (Ours)",
          "ConvNeXt-UPerNet + TreeMortBank": "ConvNeXt-UPerNet + statistics bank + consistency loss (Ours)"}


PAPER = False


def label(name):
    t = LABELS.get(name, name)
    if PAPER and "Kaleido" not in t and "\\method" not in t:
        t = t.replace(" (Ours)", "")
    return t


def table(caption, rows, first_col_name="Model", numbered=False):
    done = {n: v for n, v in rows if v["n"] >= 3}
    best, second = {}, {}
    for c in COLS:
        vals = sorted((v[c][0] for v in done.values()), reverse=True)
        best[c] = vals[0] if vals else None
        second[c] = vals[1] if len(vals) > 1 else None
    head = ("\\begin{table}[h]\n\\centering\n\\caption{%s}\n\\resizebox{\\textwidth}{!}{%%\n\\begin{tabular}{%s}\n\\toprule\n"
            % (caption, "cl" + "c" * 8 if numbered else "l" + "c" * 8))
    lead = "\\multirow{2}{*}{Row} & " if numbered else ""
    head += "%s\\multirow{2}{*}{%s} & \\multicolumn{4}{c}{Dead tree} & \\multicolumn{4}{c}{All} \\\\\n" % (lead, first_col_name)
    head += "\\cmidrule(lr){%d-%d}\\cmidrule(lr){%d-%d}\n" % ((3, 6, 7, 10) if numbered else (2, 5, 6, 9))
    head += "%sF1 & Precision & Recall & IoU & F1 & Precision & Recall & IoU \\\\\n\\midrule\n" % ("& & " if numbered else "& ")
    body = []
    for key, (name, v) in enumerate(rows, 1):
        pre = f"{key} & " if numbered else ""
        if v["n"] < 3:
            body.append(f"{pre}{label(name)} & \\multicolumn{{8}}{{c}}{{\\pend}} \\\\")
            continue
        cells = []
        for c in COLS:
            mu, sd = v[c]
            s = f"{dec(mu)} $\\pm$ {dec(sd)}"
            if best[c] is not None and abs(mu - best[c]) < 1e-9:
                s = f"\\textbf{{{s}}}"
            elif second[c] is not None and abs(mu - second[c]) < 1e-9:
                s = f"\\underline{{{s}}}"
            cells.append(s)
        body.append(f"{pre}{label(name)} & " + " & ".join(cells) + " \\\\")
    return head + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"


def aug_table(rows):
    """Dead-tree F1 (mean +- std) of the augmentation variants on the three cross-region scenarios, per backbone group."""
    scs = [("CO", "CO$\\to$non-CO"), ("EW", "East$\\to$West"), ("WE", "West$\\to$East")]
    head = ("\\begin{table}[h]\n\\centering\n\\caption{So sánh các cách tăng cường phổ trên ba kịch bản liên vùng (F1 của lớp cây chết, "
            "trung bình $\\pm$ độ lệch chuẩn, ba seed). Mỗi khối dùng cùng backbone và cùng điều kiện huấn luyện.}\n"
            "\\begin{tabular}{lccc}\n\\toprule\nConfiguration & " + " & ".join(n for _, n in scs) + " \\\\\n\\midrule\n")
    body, last = [], None
    for grp in dict.fromkeys(r["group"] for r in rows):
        sub = [r for r in rows if r["group"] == grp]
        best = {sc: max([r["cells"][sc]["dead_f1"][0] for r in sub if r["cells"][sc]["n"] >= 3] or [None], key=lambda v: -1 if v is None else v) for sc, _ in scs}
        if last is not None:
            body.append("\\midrule")
        last = grp
        for r in sub:
            cells = []
            for sc, _ in scs:
                c = r["cells"][sc]
                if c["n"] < 3:
                    cells.append("\\pend")
                else:
                    t = f"{dec(c['dead_f1'][0])} $\\pm$ {dec(c['dead_f1'][1])}"
                    cells.append(f"\\textbf{{{t}}}" if best[sc] is not None and abs(c["dead_f1"][0] - best[sc]) < 1e-9 else t)
            body.append(f"{r['label']} & " + " & ".join(cells) + " \\\\")
    return head + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


DIAG = Path(__file__).resolve().parent / "reports" / "diag"
SHIFT_NAMES = {"identity": "Không lệch", "bright0.70": "Tối (×0,7)", "bright0.85": "Tối nhẹ (×0,85)", "bright1.15": "Sáng nhẹ (×1,15)",
               "bright1.30": "Sáng (×1,3)", "gamma0.70": "Gamma 0,7", "gamma1.40": "Gamma 1,4", "castRB+": "Lệch màu R×1,2, B×0,8",
               "castRB-": "Lệch màu R×0,8, B×1,2", "contrast0.60": "Tương phản ×0,6", "contrast1.40": "Tương phản ×1,4",
               "nir0.70": "NIR ×0,7", "nir1.30": "NIR ×1,3"}


def shift_table(only=None):
    """F1 of the dead-tree class under controlled colour shifts of the test tiles: baseline (lr-only) vs Kaleido."""
    import numpy as np
    scs = [("CO", "CO$\\to$non-CO"), ("EW", "East$\\to$West"), ("WE", "West$\\to$East"), ("Site", "Site-disjoint")]
    if only:
        scs = [x for x in scs if x[0] in only]
    data = {sc: json.load(open(DIAG / f"style_shift_{sc}.json")) for sc, _ in scs}

    def ms(sc, grp, sh):
        v = [100 * data[sc][grp]["per_seed"][k][sh] for k in data[sc][grp]["per_seed"]]
        return float(np.mean(v)), float(np.std(v, ddof=1))
    head = ("\\begin{table}[h]\n\\centering\n\\caption{Độ bền với lệch màu có kiểm soát" + (" (%s)" % scs[0][1] if len(scs) == 1 else "") + ": F1 (\\%) của lớp cây chết, trung bình $\\pm$ độ lệch chuẩn trên ba seed, "
            "khi bốn kênh ảnh của test tile bị biến đổi (NDVI tính lại). Baseline là DINOv2 + decoder ($m=0{,}01$) không có statistics bank và consistency loss. Hai dòng cuối là trung bình và giá trị thấp nhất trên $13$ điều kiện (tính cho từng seed rồi lấy trung bình trên ba seed). In đậm: giá trị cao hơn trong mỗi cặp.}\n"
            "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{l" + "cc" * len(scs) + "}\n\\toprule\n\\multirow{2}{*}{Kiểu lệch} & "
            + " & ".join("\\multicolumn{2}{c}{%s}" % n for _, n in scs) + " \\\\\n"
            + "".join("\\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(len(scs))) + "\n"
            + "& " + " & ".join("Baseline & \\method{} (Ours)" for _ in scs) + " \\\\\n\\midrule\n")
    body = []
    for sh in SHIFT_NAMES:
        cells = []
        for sc, _ in scs:
            b, k = ms(sc, "plain", sh), ms(sc, "kaleido", sh)
            for (m, sd), better in ((b, b[0] >= k[0]), (k, k[0] > b[0])):
                t = f"{dec(m)} $\\pm$ {dec(sd)}"
                cells.append(f"\\textbf{{{t}}}" if better else t)
        body.append(f"{SHIFT_NAMES[sh]} & " + " & ".join(cells) + " \\\\")

    def summary(sc, grp, fn):
        v = [fn([100 * data[sc][grp]["per_seed"][k][s] for s in SHIFT_NAMES]) for k in data[sc][grp]["per_seed"]]
        return float(np.mean(v)), float(np.std(v, ddof=1))
    for lab, fn in (("Trung bình $13$ điều kiện", np.mean), ("Thấp nhất (điều kiện xấu nhất)", np.min)):
        cells = []
        for sc, _ in scs:
            b, k = summary(sc, "plain", fn), summary(sc, "kaleido", fn)
            for (m, sd), better in ((b, b[0] >= k[0]), (k, k[0] > b[0])):
                t = f"{dec(m)} $\\pm$ {dec(sd)}"
                cells.append(f"\\textbf{{{t}}}" if better else t)
        body.append(("\\midrule\n" if fn is np.mean else "") + f"{lab} & " + " & ".join(cells) + " \\\\")
    return head + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"


def distance_table():
    """F1 by quartile of the style distance between a test tile and the nearest tile of the source bank (natural spectral shift)."""
    scs = [("CO", "CO$\\to$non-CO"), ("EW", "East$\\to$West"), ("WE", "West$\\to$East"), ("Site", "Site-disjoint")]
    head = ("\\begin{table}[h]\n\\centering\n\\caption{F1 (\\%) của lớp cây chết theo nhóm tứ phân vị của khoảng cách phổ giữa test tile và tile gần nhất trong "
            "statistics bank nguồn (Q1: gần bank nhất, Q4: xa nhất); trung bình $\\pm$ độ lệch chuẩn trên ba seed. Khoảng cách tính trong không gian "
            "tám thống kê (trung bình và độ lệch chuẩn của R, G, B, NIR) đã chuẩn hóa. Dòng $\\Delta$ là hiệu \\method{} trừ baseline.}\n"
            "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{llccccc}\n\\toprule\nKịch bản & Mô hình & Q1 & Q2 & Q3 & Q4 & Tất cả \\\\\n\\midrule\n")
    body = []
    for sc, name in scs:
        d = json.load(open(DIAG / f"style_distance_{sc}.json"))["groups"]
        for key, lab in (("baseline", "Baseline"), ("kaleido", "\\method{} (Ours)")):
            m, sd = d[key]["mean"], d[key]["std"]
            body.append(f"{name if key == 'baseline' else ''} & {lab} & " + " & ".join(f"{dec(a)} $\\pm$ {dec(b)}" for a, b in zip(m, sd)) + " \\\\")
        delta = [k - b for k, b in zip(d["kaleido"]["mean"], d["baseline"]["mean"])]
        body.append(" & $\\Delta$ & " + " & ".join(("+" if x >= 0 else "") + dec(x) for x in delta) + " \\\\")
        body.append("\\midrule")
    body = body[:-1]
    return head + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"


def diag_table():
    """Threshold-free metrics and crown-level error anatomy of the baseline (lr-only) and Kaleido on CO and East->West."""
    import glob
    ev, eo = {}, {}
    for f in glob.glob(str(DIAG / "eval_extra_dino" / "*.json")):
        r = json.load(open(f))
        ev.setdefault(r["args"]["tag"], []).append(r)
    for fn_ in ("error_objects_dino.json", "error_objects_dino_plain.json"):
        for r in json.load(open(DIAG / fn_)).values():
            eo.setdefault(r["args"]["tag"], []).append(r)
    rows = [("CO$\\to$non-CO", "Baseline", "co_lr001_p_ck"), ("CO$\\to$non-CO", "\\method{} (Ours)", "co_fz001"),
            ("East$\\to$West", "Baseline", "ew_lr001_p_ck"), ("East$\\to$West", "\\method{} (Ours)", "ew_fz001")]
    head = ("\\begin{table}[h]\n\\centering\n\\caption{Chẩn đoán lỗi (trung bình ba seed). AP: average precision ở mức pixel (không phụ thuộc ngưỡng); "
            "F1@val: F1 ở ngưỡng chọn trên validation set. Tán cây được coi là phát hiện nếu ít nhất $30\\%$ pixel của nó được dự đoán (ngưỡng chọn trên validation set). "
            "``Bỏ sót nguyên tán'': tỉ lệ pixel cây chết bị bỏ sót thuộc về các tán bị bỏ sót hoàn toàn; ``FP tách biệt'': tỉ lệ pixel dương tính giả nằm trong "
            "cụm dự đoán không chạm nhãn nào.}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{llccccccc}\n\\toprule\n"
            "Kịch bản & Mô hình & AP & F1@0,5 & F1@val & Tán phát hiện (\\%) & Phát hiện, tán $>256$ px (\\%) & Bỏ sót nguyên tán (\\%) & FP tách biệt (\\%) \\\\\n\\midrule\n")
    body = []
    for sc, name, tag in rows:
        e, o = ev[tag], eo[tag]
        mean = lambda f: sum(f(r) for r in e) / len(e)
        n = sum(sum(r["acc"]["n"]) for r in o)
        det = sum(sum(r["acc"]["det"]) for r in o)
        big_n = sum(r["acc"]["n"][3] for r in o)
        big_det = sum(r["acc"]["det"][3] for r in o)
        fnw = sum(r["other"]["fn_whole"] for r in o)
        fnr = sum(r["other"]["fn_rim"] for r in o)
        fpi = sum(r["other"]["fp_iso"] for r in o)
        fpt = sum(r["other"]["fp_total"] for r in o)
        cells = [dec(mean(lambda r: r["ap"])), dec(100 * mean(lambda r: r["at_0.5"]["f1"])), dec(100 * mean(lambda r: r["at_val_thr"]["f1"])),
                 dec(100 * det / n), dec(100 * big_det / big_n), dec(100 * fnw / (fnw + fnr)), dec(100 * fpi / fpt)]
        body.append(f"{sc} & {name} & " + " & ".join(cells) + " \\\\")
    return head + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"


def err_table():
    """Object-level error anatomy of Kaleido (m=0.01) on the four cross-region scenarios: mean +- std over the three seeds."""
    import numpy as np
    d = json.load(open(DIAG / "error_objects_dino.json"))
    groups = {}
    for r in d.values():
        sc = r["args"]["tag"].split("_")[0]
        if sc in ("co", "ew", "we", "site"):
            groups.setdefault(sc, []).append(r)
    names = [("co", "CO$\\to$non-CO"), ("ew", "East$\\to$West"), ("we", "West$\\to$East"), ("site", "Site-disjoint")]
    head = ("\\begin{table}[h]\n\\centering\n\\caption{Phân tích lỗi của \\method{} ($m=0{,}01$), trung bình $\\pm$ độ lệch chuẩn trên ba seed. "
            "Tán cây (thành phần liên thông 8-liên thông của nhãn) được coi là phát hiện nếu ít nhất $30\\%$ pixel của nó được dự đoán, ngưỡng chọn trên validation set. "
            "``Pixel thuộc tán $>256$ px'': tỉ lệ pixel cây chết nằm trong các tán lớn. ``Bỏ sót nguyên tán'': tỉ lệ pixel cây chết bị bỏ sót nằm trong các tán bị bỏ sót hoàn toàn. "
            "``FP tách biệt'': tỉ lệ pixel dương tính giả nằm trong cụm dự đoán không chạm nhãn nào.}\n\\label{tab:errors}\n"
            "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lccccc}\n\\toprule\n"
            "Kịch bản & Tán phát hiện (\\%) & Tán $>256$ px phát hiện (\\%) & Pixel thuộc tán $>256$ px (\\%) & Bỏ sót nguyên tán (\\%) & FP tách biệt (\\%) \\\\\n\\midrule\n")
    body = []
    for key, name in names:
        rs = groups[key]
        per = []
        for r in rs:
            a, o = r["acc"], r["other"]
            per.append([100 * sum(a["det"]) / sum(a["n"]), 100 * a["det"][3] / a["n"][3], 100 * a["px"][3] / sum(a["px"]),
                        100 * o["fn_whole"] / (o["fn_whole"] + o["fn_rim"]), 100 * o["fp_iso"] / o["fp_total"]])
        per = np.array(per)
        body.append(f"{name} & " + " & ".join(dec(m) if k == 2 else f"{dec(m)} $\\pm$ {dec(s)}" for k, (m, s) in enumerate(zip(per.mean(0), per.std(0, ddof=1)))) + " \\\\")
    return head + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--tex", required=True)
    ap.add_argument("--profile", default="full", choices=["full", "paper"])
    a = ap.parse_args()
    global PAPER
    PAPER = a.profile == "paper"
    d = json.load(open(a.json))
    parts = []
    for sc in ["Random", "CO", "EW", "WE", "Site"]:
        parts.append(table(TITLES[sc], [(n, d[sc][n]) for n in ORDER]))
    rows = [(ABL[k], d["Ablation"][k]) for k in ["1", "2", "4", "5"]]
    parts.append(table("Ablation trên kịch bản CO$\\to$không-CO (DINOv2, ba seed). Từ dòng 2 trở đi dùng $m=0{,}01$.", rows,
                       first_col_name="Configuration", numbered=True))
    if a.profile == "paper":
        if (DIAG / "style_shift_CO.json").exists():
            for sc_ in ("CO", "EW", "WE"):
                parts.append(shift_table(only=[sc_]))
        if (DIAG / "error_objects_dino.json").exists():
            parts.append(err_table())
    else:
        if "AugCmp" in d:
            parts.append(aug_table(d["AugCmp"]))
        if (DIAG / "style_shift_CO.json").exists():
            parts.append(shift_table())
            parts.append(diag_table())
            if (DIAG / "style_distance_CO.json").exists():
                parts.append(distance_table())
    if a.profile == "paper":
        labels = ["tab:res_random", "tab:res_co", "tab:res_ew", "tab:res_we", "tab:res_site", "tab:ablation", "tab:shift", "tab:shift_ew", "tab:shift_we"]
        parts = [p.replace("\\resizebox", "\\label{%s}\n\\resizebox" % lab, 1) if (lab and "\\label" not in p) else p
                 for p, lab in zip(parts, labels + [None] * len(parts))]
    block = "% BEGIN RESULTS\n" + "\n".join(parts) + "% END RESULTS"
    t = Path(a.tex).read_text(encoding="utf8")
    t2 = re.sub(r"% BEGIN RESULTS.*?% END RESULTS", lambda m: block, t, flags=re.S)
    Path(a.tex).write_text(t2, encoding="utf8")
    print("tables written:", len(parts))


if __name__ == "__main__":
    main()
