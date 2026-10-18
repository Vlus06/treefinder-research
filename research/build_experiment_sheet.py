"""Build the single-sheet experiment log (xlsx -> import to Google Sheets) from cluster logs.

    python research/build_experiment_sheet.py --logs <dir with logs/ + ensemble_results*.json> --out experiment_log.xlsx
"""
import argparse
import ast
import json
import re
import statistics as st
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

SCEN = {"res_random": "Random", "res_state_WE": "W–E", "res_state_EW": "E–W", "res_state_CO": "CO",
        "res_climate1": "Climate", "res_climate3": "Climate-hard", "res_treetype1": "Forest", "res_treetype3": "Forest-hard",
        "res_random_site": "Site-disjoint (tự tạo)", "res_climate2": "Climate (đảo)", "res_treetype2": "Forest (đảo)"}
SCEN_ORDER = list(SCEN.values())
MODEL = {"unet": "U-Net", "deeplabv3": "DeepLabV3", "vitseg3": "ViT", "segformer_large3": "SegFormer", "mask2former": "Mask2Former",
         "dofa": "DOFA"}
PAPER_ORDER = ["U-Net", "DeepLabV3", "ViT", "SegFormer", "Mask2Former", "DOFA"]
# TreeFinder paper (NeurIPS 2025) dead-tree F1 [%], Tables 1-3, as transcribed in research/REPRODUCTION.md
PAPER = {
    "Random": [46.1, 49.7, 43.6, 47.8, 51.9, 29.2], "W–E": [13.9, 15.0, 13.4, 22.8, 21.4, 4.4],
    "E–W": [16.1, 20.5, 3.8, 21.8, 21.6, 4.8], "CO": [10.2, 15.4, 6.3, 17.5, 11.6, 10.2],
    "Climate": [18.6, 25.8, 21.6, 28.0, 29.8, 12.9], "Climate-hard": [19.9, 18.0, 11.4, 22.6, 15.4, 4.7],
    "Forest": [35.3, 36.1, 26.5, 40.1, 38.6, 13.9], "Forest-hard": [14.7, 21.5, 19.9, 37.7, 11.1, 14.3]}
PAPER_EXTRA = {("Random", "U-Net"): dict(sd=7.1, P=35.7, R=65.6, IoU=30.1, All=72.9),
               ("Random", "DeepLabV3"): dict(sd=8.7, P=49.6, R=49.9, IoU=33.4, All=74.7)}
TMS = {"Random": (66.29, "± 0.40"), "E–W": (28.84, ""), "W–E": (23.67, ""), "CO": (19.55, "")}   # TreeMortSeg (Li et al., EMS 2026)

OWN_NAMES = {"convnext_upernet": "ConvNeXt-S + UPerNet", "treemortseg": "TreeMortSeg (tái hiện)", "convnext_b_upernet": "ConvNeXt-B + UPerNet",
             "treemortseg_b": "TreeMortSeg-B", "segformer_b5": "SegFormer-B5 (5 kênh)", "upernet_swin_s": "UPerNet-Swin-S",
             "effunet_b7": "EfficientUNet-B7", "dinov2_b": "DINOv2-B + conv decoder", "hrnet_w32": "HRNet-W32",
             "unet": "U-Net (pipeline của mình)", "deeplabv3": "DeepLabV3 (pipeline của mình)"}


def metrics_of(text):
    ms = re.findall(r"Combined metrics: (\{.*\})", text)
    if not ms:
        return None
    d = ast.literal_eval(ms[-1].replace("nan", "float('nan')")) if "nan" not in ms[-1] else None
    if d is None:
        d = json.loads(ms[-1])
    return {"F1": 100 * d["class1_f1"], "P": 100 * d["class1_precision"], "R": 100 * d["class1_recall"],
            "IoU": 100 * d["class1_iou"], "All": 100 * d["res_f1_macro"]}


def epochs_of(text):
    n = re.findall(r"epoch (\d+) train", text) or re.findall(r"Epoch\s*\[?(\d+)", text)
    return int(n[-1]) if n else None


def air_desc(g):
    parts = []
    if g.get("norm", "none") != "none": parts.append("chuẩn hoá từng tile")
    if g.get("photo"): parts.append("photometric aug")
    if g.get("sampler") == "site": parts.append("site-grouped sampler")
    if g.get("vrex"): parts.append(f"V-REx {g['vrex']}")
    if g.get("ema"): parts.append(f"EMA {g['ema']}")
    if g.get("tta"): parts.append("TTA")
    if g.get("up", 1.0) != 1.0: parts.append(f"input ×{g['up']}")
    if g.get("sched", "exp") != "exp": parts.append("cosine lr")
    if g.get("loss", "bce_dice") != "bce_dice": parts.append(g["loss"])
    if g.get("lr", 1e-4) != 1e-4: parts.append(f"lr {g['lr']}")
    if g.get("lr_backbone_mult", 1.0) != 1.0: parts.append(f"lr backbone ×{g['lr_backbone_mult']}")
    if g.get("patience", 10) != 10: parts.append(f"patience {g['patience']}")
    for k, lbl in (("dino_align", "căn chỉnh DINOv2"), ("distill", "distill"), ("selfdistill", "self-distill"),
                   ("copypaste", "copy-paste"), ("gate", "tile gate")):
        if g.get(k): parts.append(f"{lbl} {g[k]}")
    if g["model"] == "treemortseg":
        parts.append({"none": "không aux (chỉ mask)", "released": "aux nhãn phát hành", "fixed": "aux nhãn sửa lỗi"}[g.get("aux", "none")])
    if g.get("split_seed"): parts.append(f"split seed {g['split_seed']}")
    if g.get("tag"): parts.append(f"tag={g['tag']}")
    return "; ".join(parts) or "recipe chuẩn (AdamW 1e-4, patience 10)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.logs)
    rows = []   # dict: kind, source, scen, model, desc, seed, metrics, epochs, paperF1, note, ref

    # --- paper rows
    for sc in SCEN_ORDER:
        if sc not in PAPER: continue
        for m, f in zip(PAPER_ORDER, PAPER[sc]):
            ex = PAPER_EXTRA.get((sc, m), {})
            rows.append(dict(order=0, source="PAPER (TreeFinder, NeurIPS'25) – số báo cáo", scen=sc, model=m, desc="Số liệu trong bảng của paper dataset",
                             seed="TB 3 lần" if sc == "Random" else "1 giá trị", F1=f, P=ex.get("P"), R=ex.get("R"), IoU=ex.get("IoU"),
                             All=ex.get("All"), note=("± %.1f" % ex["sd"]) if ex else ""))
    for sc, (f, sd) in TMS.items():
        rows.append(dict(order=0, source="PAPER (TreeMortSeg, EMS'26) – số báo cáo", scen=sc, model="TreeMortSeg", desc="Số liệu báo cáo trong paper TreeMortSeg (đối thủ cần cạnh tranh)",
                         seed="paper", F1=f, note=sd))

    # --- official runs
    for p in sorted(root.glob("logs/code_*_res_*.log")):
        m = re.match(r"code_(head|rgbm2f|site)_(res_.+?)_(\d{4})_(\w+?)_s(\d+)\.log", p.name)
        if not m: continue
        code, cfg, expid, model, seed = m.groups()
        t = p.read_text(errors="ignore"); mt = metrics_of(t)
        if not mt: continue
        mn = MODEL[model]; sc = SCEN[cfg]
        if code == "rgbm2f":
            src, desc = "TÁI HIỆN paper (code tác giả, bản trước commit 274e112)", "Mask2Former chỉ nhận RGB, giống code lúc viết paper"
        elif code == "site":
            src, desc = "TỰ THÊM (thí nghiệm kiểm soát) – code tác giả", "Code tác giả nguyên bản nhưng chia theo site (không rò rỉ site)"
        else:
            src, desc = "TÁI HIỆN paper (code tác giả, không sửa)", "main.py + configs của tác giả, chỉ đổi num_workers"
        pf = PAPER.get(sc, [None] * 6)[PAPER_ORDER.index(mn)] if code != "site" else None
        rows.append(dict(order=1, source=src, scen=sc, model=mn + (" (RGB)" if code == "rgbm2f" else ""), desc=desc, seed=int(seed), epochs=epochs_of(t),
                         paperF1=pf if code == "head" or code == "rgbm2f" else None, log=p.stem, **mt))

    # --- own (air) runs
    for p in sorted(root.glob("logs/air*.log")):
        t = p.read_text(errors="ignore")
        am = re.search(r"args=(\{.*\})", t)
        mt = metrics_of(t)
        if not am or not mt: continue
        g = ast.literal_eval(am.group(1))
        if g.get("max_samples") or str(g["exp_id"]).startswith("99"): continue
        cfg = Path(g["config"]).stem; sc = SCEN.get(cfg, cfg)
        mname = OWN_NAMES.get(g["model"], g["model"])
        if g["model"].startswith("m_"):
            mname = "Grid: " + g["model"][2:]
        if g["model"] in ("unet", "deeplabv3"): src = "TỰ CHẠY lại baseline của paper trong pipeline của mình"
        elif g["model"] == "treemortseg": src = "BASELINE từ paper TreeMortSeg – tự chạy lại (code tác giả MIT)"
        elif g["model"] in ("convnext_upernet", "convnext_b_upernet"): src = "TỰ THÊM (ConvNeXt+UPerNet = backbone/neck của TreeMortSeg, bỏ aux)"
        else: src = "TỰ THÊM (baseline mạnh hơn, không có trong paper)"
        if any(g.get(k) for k in ("dino_align", "distill", "selfdistill", "copypaste", "gate")): src = "PHƯƠNG PHÁP ĐỀ XUẤT (distill / thành phần mới)"
        pf = None
        if g["model"] == "treemortseg" and sc in TMS: pf = TMS[sc][0]
        if g["model"] in ("unet", "deeplabv3") and sc in PAPER: pf = PAPER[sc][PAPER_ORDER.index(MODEL[g["model"]])]
        seed = g["seed"]; ss = g.get("split_seed")
        rows.append(dict(order=2, source=src, scen=sc, model=mname, desc=air_desc(g), seed=f"train {seed}" + (f" / split {ss}" if ss else ""),
                         seedkey=(seed, ss), epochs=epochs_of(t) or None, paperF1=pf, log=p.stem, **mt))

    # --- ensembles
    for fn, tta in (("ensemble_results.json", False), ("ensemble_results_tta.json", True)):
        fp = root / fn
        if not fp.exists(): continue
        try:
            ens = json.load(open(fp))
        except ValueError:
            continue
        for k, r in ens.items():
            if len(r["members"]) < 2: continue
            mm = "+".join({"cnx": "ConvNeXt", "dino": "DINOv2", "tms": "TreeMortSeg"}[x] for x in r["members"])
            sc = "Random" if r["split"] == "tile" else "Site-disjoint (tự tạo)"
            a5 = r["at_0.5"]
            rows.append(dict(order=3, source="TỰ THÊM (ensemble xác suất, chưa phải novelty)", scen=sc, model="Ensemble " + mm + (" +TTA" if tta else ""),
                             desc="Trung bình xác suất các checkpoint đã train (cùng seed); tốn %d× chi phí" % len(r["members"]), seed=r["seed"],
                             F1=100 * a5["f1"], P=100 * a5["precision"], R=100 * a5["recall"], IoU=None, All=None,
                             paperF1=TMS["Random"][0] if sc == "Random" else None, note="F1@val-thr=%.1f, AP=%.1f" % (100 * r["at_val_thr"]["f1"], 100 * r["ap"])))

    # --- group means for multi-seed (own+official) rows
    from collections import defaultdict
    grp = defaultdict(list)
    for r in rows:
        if r["order"] in (1, 2) and r.get("F1") is not None:
            grp[(r["order"], r["source"], r["scen"], r["model"], r["desc"].split("; split seed")[0])].append(r)
    means = []
    for (o, src, sc, mn, desc), rs in grp.items():
        if len(rs) < 2: continue
        f = [x["F1"] for x in rs]
        means.append(dict(order=o, source=src, scen=sc, model=mn, desc=desc, seed=f"TRUNG BÌNH {len(rs)} lần chạy", mean=True,
                          F1=st.mean(f), P=st.mean(x["P"] for x in rs), R=st.mean(x["R"] for x in rs), IoU=st.mean(x["IoU"] for x in rs),
                          All=st.mean(x["All"] for x in rs), paperF1=rs[0].get("paperF1"), note="± %.1f (sd, n=%d)" % (st.stdev(f), len(f))))
    rows += means

    def sk(r):
        sci = SCEN_ORDER.index(r["scen"]) if r["scen"] in SCEN_ORDER else 99
        return (sci, r["order"], r["model"], r["desc"], str(r["seed"]))
    rows.sort(key=sk)

    wb = Workbook(); ws = wb.active; ws.title = "ExperimentTable"
    hdr = ["#", "Kịch bản (scenario)", "Nguồn / loại baseline", "Mô hình (model)", "Experiment (cấu hình chi tiết)", "Seed", "F1 dead-tree (%)",
           "Precision (%)", "Recall (%)", "IoU (%)", "F1 All (%)", "Epochs", "F1 paper báo cáo", "Δ so với paper (điểm)", "Ghi chú", "Log"]
    ws.append(hdr)
    fills = {0: "FFE699", 1: "C6E0B4", 2: "BDD7EE", 3: "F8CBAD"}
    for i, r in enumerate(rows, 1):
        f1 = r.get("F1"); pf = r.get("paperF1")
        delta = round(f1 - pf, 2) if (f1 is not None and pf is not None and r["order"] != 0) else None
        rr = lambda v: None if v is None else round(v, 2)
        ws.append([i, r["scen"], r["source"], r["model"], r["desc"], r["seed"], rr(f1), rr(r.get("P")), rr(r.get("R")), rr(r.get("IoU")), rr(r.get("All")),
                   r.get("epochs"), pf, delta, r.get("note", ""), r.get("log", "")])
        c = fills.get(r["order"], "FFFFFF")
        for col in range(1, len(hdr) + 1):
            cell = ws.cell(row=i + 1, column=col)
            cell.fill = PatternFill("solid", fgColor=c)
            if r.get("mean"): cell.font = Font(bold=True)
    for col in range(1, len(hdr) + 1):
        h = ws.cell(row=1, column=col); h.font = Font(bold=True, color="FFFFFF"); h.fill = PatternFill("solid", fgColor="1F4E78")
        h.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    widths = [5, 20, 44, 28, 60, 18, 12, 12, 12, 10, 10, 8, 14, 14, 30, 40]
    for i, w in enumerate(widths, 1): ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "E2"; ws.auto_filter.ref = ws.dimensions
    ws.append([]); ws.append(["Chú giải màu: vàng = số liệu paper tự báo cáo; xanh lá = tái hiện paper bằng code tác giả; xanh dương = tự chạy / tự thêm (pipeline của mình); cam = ensemble."])

    add_insights(wb)
    wb.save(a.out)
    print(len(rows), "rows ->", a.out)


INSIGHTS = [
    ("Dữ liệu", "Tile split rò rỉ site", "857/860 site của tập test cũng có tile trong train ở split ngẫu nhiên chuẩn", "Điểm Random cao một phần do thuộc lòng site; đã tạo split tách site để làm bằng chứng phụ (không phải đóng góp chính)"),
    ("Dữ liệu", "Hai chế độ nhãn: 'mega tile'", "36 tile (0,2%) có >20% pixel chết, chứa ~30% tổng pixel chết (polygon cấp lâm phần)", "F1 gộp theo pixel bị các tile này chi phối; bỏ chúng thì F1 ≈ 46 ở mọi split"),
    ("Dữ liệu", "Nhãn aux phát hành của TreeMortSeg dựng từ vùng no-data (lỗi)", "Đã phát hiện khi đối chiếu code; có chế độ 'fixed' sửa lại", "Đóng góp của aux/multitask trong paper có thể bị thổi phồng"),
    ("Tái hiện", "Tái hiện baseline paper dataset tốt", "36/46 ô nằm trong ±5 điểm; Random U-Net 46.8 vs 46.1, DeepLab 52.3 vs 49.7 (TB 3 seed)", "Pipeline/data/seed đáng tin để so sánh công bằng"),
    ("Tái hiện", "Chưa tái hiện đúng số TreeMortSeg", "TreeMortSeg tự chạy 61–65 F1 trên split ngẫu nhiên vs 66.29±0.40 báo cáo; cross-region thấp hơn ~3–4 điểm", "Cần nói rõ khi so sánh; sd giữa các split lớn hơn sd báo cáo"),
    ("Tái hiện", "Recipe quan trọng hơn kiến trúc", "lr 1e-4 + patience 10 nâng DeepLab 49.7→58.2; ViT sụp do early stopping theo val loss", "Phải đối xử cùng recipe cho mọi baseline trước khi kết luận về kiến trúc"),
    ("Mô hình", "ConvNeXt-S + UPerNet là baseline mạnh", "63.1±0.4 (tile seed 2025), ngang/hơn TreeMortSeg khi bỏ aux", "Phần lớn lợi ích của TreeMortSeg đến từ backbone/neck, không phải đầu aux"),
    ("Mô hình", "Multitask (edge/distance) không đem thêm thông tin ngoài mask", "Chỉ liên quan lỗi ở rìa tán (8–26% pixel bị bỏ sót); không thắng ổn định qua nhiều seed", "Dẫn tới hướng thay multitask bằng distillation / căn chỉnh biểu diễn"),
    ("Mô hình", "Ensemble xác suất cho mức tăng lớn nhưng tốn 3×", "+1.6 F1 và ~+9 AP so với model đơn tốt nhất (tile split); ConvNeXt+DINOv2+TreeMortSeg ≈ 65.8", "Ngang TreeMortSeg 66.29 nhưng không phải novelty; làm 'giáo viên' cho distillation"),
    ("Mô hình", "Các mẹo không giúp / không ổn định", "Chuẩn hoá từng tile (+9 chỉ ở 1 split, TB −8 trên 3 split), EMA, TTA, sampler theo site, V-REx", "Loại khỏi danh sách đề xuất"),
    ("Lỗi", "~45% pixel false positive nằm trong tile không có cây chết", "Phân tích error anatomy mức tán cây", "Gợi ý cổng hiện diện ở mức tile (M2)"),
    ("Lỗi", "Tán nhỏ bị bỏ sót nhiều", "Bỏ sót theo kích thước: tán nhỏ chiếm phần lớn FN", "Gợi ý copy-paste tán nhỏ và tinh chỉnh độ phân giải cao (M1)"),
    ("Đánh giá", "Điểm số dao động mạnh theo split", "sd 4–7 điểm khi đổi seed/split (split thay đổi theo seed)", "Dùng split cố định 2025 và đổi seed huấn luyện (nhiễu ≈ ±0.4) để so sánh ghép cặp"),
]
DIRECTIONS = [
    ("1", "D1 – Căn chỉnh biểu diễn với DINOv2 đóng băng", "Đặc trưng stride-16 của ConvNeXt thiếu ngữ nghĩa mà DINOv2 có; ép cosine tới patch feature sẽ giảm FP ngoài rừng", "--dino_align λ trên split 2025, 3 seed huấn luyện, so với 63.1±0.4", "Đã code, chưa chạy"),
    ("2", "D2 – Distill ensemble → 1 student", "Nhận được phần lớn +1.6 F1/+9 AP của ensemble với chi phí suy luận 1×", "--distill (teacher ConvNeXt+DINOv2+TreeMortSeg), α∈{0.5}, T∈{1,2}; lưu ý teacher đã thấy dữ liệu train", "Đã code, chưa chạy"),
    ("3", "D3 – Self-distillation / mean teacher", "Soft label từ EMA của chính model giảm nhiễu nhãn mức pixel", "--ema 0.999 --selfdistill α", "Đã code, chưa chạy"),
    ("4", "Copy-paste tán chết nhỏ", "Tán nhỏ là lỗi chính; thêm ví dụ nhỏ tăng recall", "--copypaste 0.5", "Đã code, chưa chạy"),
    ("5", "M2 – Cổng hiện diện ở mức tile", "Nhắm vào 45% FP ở tile rỗng", "--gate w; theo dõi precision/AP", "Đã code, chưa chạy"),
    ("6", "M1 – Tinh chỉnh độ phân giải cao cho tán nhỏ", "Stride-4 của decoder quá thô cho tán ~9 px", "Chưa code; thử input ×2 hoặc nhánh refine", "Ý tưởng"),
    ("7", "Kết hợp tốt nhất (D1+D2, v.v.) + ablation đa seed", "Các thành phần bổ trợ nhau", "3 seed, kiểm định ghép cặp; sau đó chạy cross-region và split tách site", "Sau bước 1–5"),
    ("8", "Cross-region (E–W, W–E, CO)", "Mục tiêu so với TreeMortSeg 28.84 / 23.67 / 19.55", "Đang chạy ConvNeXt/TreeMortSeg 3 seed (job 4333/4334)", "Đang chạy"),
    ("9", "Hoàn thiện tái hiện & viết paper", "Cần số 3 seed cho các baseline chính, Climate/Forest đảo, training-size sweep, PROVENANCE.md tách code tác giả/code mình", "Theo REPRODUCTION.md", "Chưa làm"),
]


def add_insights(wb):
    ws = wb.create_sheet("Insights & Hướng phát triển")
    ws.append(["Phát hiện / Insight"]); ws.append(["Nhóm", "Phát hiện", "Bằng chứng", "Hàm ý"])
    for r in INSIGHTS: ws.append(list(r))
    ws.append([]); ws.append(["Hướng phát triển"]); ws.append(["Ưu tiên", "Hướng", "Giả thuyết", "Thí nghiệm", "Trạng thái"])
    for r in DIRECTIONS: ws.append(list(r))
    bold = Font(bold=True, color="FFFFFF")
    for row in ws.iter_rows():
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        if row[0].value in ("Nhóm", "Ưu tiên") or row[0].value in ("Phát hiện / Insight", "Hướng phát triển"):
            for c in row:
                c.font = bold; c.fill = PatternFill("solid", fgColor="1F4E78")
    for i, w in enumerate([12, 45, 60, 60, 20], 1): ws.column_dimensions[get_column_letter(i)].width = w


if __name__ == "__main__":
    main()
