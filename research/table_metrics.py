"""Collect the full metric set (dead-tree and 'All' F1 / precision / recall / IoU) of the runs behind every cell of the result tables.

Run on the cluster (logs live there; standard library only):
    python research/table_metrics.py --logs /datastore/phuocnth/treefinder/logs --out table_metrics.json
Each cell = mean and sample standard deviation over the three training seeds (2025, 2024, 2023, or the equivalent early seeds
1 and 2 where the first runs used those labels). A cell with fewer than three finished runs is stored with n < 3.
"""
import argparse
import ast
import glob
import json
import math
import os
import re
from pathlib import Path

KEYS = {"dead_f1": "class1_f1", "dead_p": "class1_precision", "dead_r": "class1_recall", "dead_iou": "class1_iou",
        "all_f1": "res_f1_macro", "all_p": "res_precision_macro", "all_r": "res_recall_macro", "all_iou": "res_iou_macro"}
SEEDS = (2025, 2024, 2023)
OFFICIAL = {"unet": "U-Net", "deeplabv3": "DeepLabV3+", "vitseg3": "ViT", "segformer_large3": "SegFormer",
            "mask2former": "Mask2Former", "dofa": "DOFA"}


def read_metrics(path):
    t = open(path, errors="ignore").read()
    m = re.findall(r"Combined metrics: (\{.*\})", t)
    if not m:
        return None
    d = ast.literal_eval(m[-1])
    return {k: 100.0 * d[v] for k, v in KEYS.items() if v in d}


def index_air(logs):
    idx = {}
    for f in glob.glob(os.path.join(logs, "air*.log")):
        t = open(f, errors="ignore").read()
        a = re.search(r"args=(\{.*\})", t)
        if not a or "Combined metrics" not in t:
            continue
        g = ast.literal_eval(a.group(1))
        if g.get("tag") == "smoke" or g.get("max_samples"):
            continue
        key = (Path(g["config"]).stem, g["model"], g["aux"], g.get("tag", "") or "", int(g["seed"]))
        prev = idx.get(key)
        if prev is None or int(g["exp_id"]) > prev[0]:
            m = read_metrics(f)
            if m:
                idx[key] = (int(g["exp_id"]), m)
    return {k: v[1] for k, v in idx.items()}


def index_official(logs):
    idx = {}
    pat = re.compile(r"^(code_head|code_site)_(res_.+?)_(\d{4})_(unet|deeplabv3|vitseg3|segformer_large3|mask2former|dofa)_s(\d+)\.log$")
    for f in glob.glob(os.path.join(logs, "code_*.log")):
        m = pat.match(os.path.basename(f))
        if not m:
            continue
        met = read_metrics(f)
        if met:
            idx[(m.group(2), m.group(4), int(m.group(5)))] = met
    return idx


def stat(runs):
    out = {"n": len(runs)}
    for k in KEYS:
        v = [r[k] for r in runs if k in r]
        mu = sum(v) / len(v) if v else float("nan")
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        out[k] = [mu, sd]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    air, off = index_air(a.logs), index_official(a.logs)

    def air_cell(picks):
        runs = [air[p] for p in picks if p in air]
        return stat(runs)

    def off_cell(cfgs_by_seed, model):
        runs = []
        for s in SEEDS:
            for cfg in cfgs_by_seed(model, s):
                if (cfg, model, s) in off:
                    runs.append(off[(cfg, model, s)])
                    break
        return stat(runs)

    def uni(cfg, model, aux, tag):
        return [(cfg, model, aux, tag, s) for s in SEEDS]

    res = {}
    # ---- cross-region scenarios: one tag per row
    state = {"CO": ("res_state_CO", "co_tms_fixed_w25", "convnext", "co_dino_plain", "co_bank_src_cons", "co_dino_method", "co_fz001"),
             "EW": ("res_state_EW", "ew_tms_fixed", "convnext", "ew_dino_plain", "ew_method", "ew_dino_method", "ew_fz001"),
             "WE": ("res_state_WE", "we_tms_fixed", "convnext", "we_dino_plain", "we_method", "we_dino_method", "we_fz001")}
    for sc, (cfg, tms, cnx, dpl, cpp, d01, d001) in state.items():
        t = {}
        for m in OFFICIAL:
            t[OFFICIAL[m]] = off_cell(lambda mm, s, c=cfg: [c], m)
        t["ConvNeXt-UPerNet"] = air_cell(uni(cfg, "convnext_upernet", "none", cnx))
        t["TreeMortSeg"] = air_cell(uni(cfg, "treemortseg", "fixed", tms))
        t["DINOv2"] = air_cell(uni(cfg, "dinov2_b", "none", dpl))
        t["DINOv2 (m=0.01)"] = air_cell(uni(cfg, "dinov2_b", "none", {"CO": "co_lr001_p", "EW": "ew_lr001_p", "WE": "we_lr001_p"}[sc]))
        t["DINOv2 + DPT (m=0.01)"] = air_cell(uni(cfg, "dinov2_b_dpt", "none", {"CO": "co_dpt_lr001", "EW": "ew_dpt_lr001", "WE": "we_dpt_lr001"}[sc]))
        t["ConvNeXt-UPerNet + TreeMortBank"] = air_cell(uni(cfg, "convnext_upernet", "none", cpp))
        t["DINOv2 + TreeMortBank (m=0.1)"] = air_cell(uni(cfg, "dinov2_b", "none", d01))
        t["DINOv2 + TreeMortBank (m=0.01)"] = air_cell(uni(cfg, "dinov2_b", "none", d001))
        res[sc] = t
    # ---- random split: early runs were labelled with seeds 1, 2, 2025 or 2025, 2024, 2023 depending on the tag
    r = "res_random"
    t = {}
    for m in OFFICIAL:
        t[OFFICIAL[m]] = off_cell(lambda mm, s: [r], m)
    t["ConvNeXt-UPerNet"] = air_cell([(r, "convnext_upernet", "none", "", 2025), (r, "convnext_upernet", "none", "tile3_cnx_plain", 2024),
                                      (r, "convnext_upernet", "none", "tile3_cnx_plain", 2023)])
    t["TreeMortSeg"] = air_cell(uni(r, "treemortseg", "fixed", "tile_tms_fixed_w25"))
    t["DINOv2"] = air_cell([(r, "dinov2_b", "none", "tile_dino_plain", 2025), (r, "dinov2_b", "none", "tile3_dino_plain", 2024),
                            (r, "dinov2_b", "none", "tile3_dino_plain", 2023)])
    t["DINOv2 (m=0.01)"] = air_cell(uni(r, "dinov2_b", "none", "tile3_dino_lr001_p"))
    t["DINOv2 + DPT (m=0.01)"] = air_cell(uni(r, "dinov2_b_dpt", "none", "tile3_dpt_lr001"))
    t["ConvNeXt-UPerNet + TreeMortBank"] = air_cell([(r, "convnext_upernet", "none", "tile_method", 2025), (r, "convnext_upernet", "none", "tile3_cnx_method", 2024),
                                                      (r, "convnext_upernet", "none", "tile3_cnx_method", 2023)])
    t["DINOv2 + TreeMortBank (m=0.1)"] = air_cell([(r, "dinov2_b", "none", "tile_dino_method", 2025), (r, "dinov2_b", "none", "tile3_dino_method", 2024),
                                                    (r, "dinov2_b", "none", "tile3_dino_method", 2023)])
    t["DINOv2 + TreeMortBank (m=0.01)"] = air_cell(uni(r, "dinov2_b", "none", "tile3_dino_fz001"))
    res["Random"] = t
    # ---- site-disjoint split (partition 2025 for every model)
    s_ = "res_random_site"
    t = {}

    def site_cfgs(model, seed):
        if model in ("unet", "deeplabv3") and seed == 2025:
            return [s_, s_ + "_fix"]
        return [s_ + "_fix"]
    for m in OFFICIAL:
        t[OFFICIAL[m]] = off_cell(site_cfgs, m)
    t["ConvNeXt-UPerNet"] = air_cell([(s_, "convnext_upernet", "none", "plain_ts1", 1), (s_, "convnext_upernet", "none", "plain_ts2", 2),
                                      (s_, "convnext_upernet", "none", "", 2025)])
    t["TreeMortSeg"] = air_cell(uni(s_, "treemortseg", "fixed", "site_tms_fixed"))
    t["DINOv2"] = air_cell(uni(s_, "dinov2_b", "none", "site_dino_plain"))
    t["DINOv2 (m=0.01)"] = air_cell(uni(s_, "dinov2_b", "none", "site_dino_lr001_p"))
    t["DINOv2 + DPT (m=0.01)"] = air_cell(uni(s_, "dinov2_b_dpt", "none", "site_dpt_lr001"))
    t["ConvNeXt-UPerNet + TreeMortBank"] = air_cell(uni(s_, "convnext_upernet", "none", "site_cnx_method"))
    t["DINOv2 + TreeMortBank (m=0.1)"] = air_cell(uni(s_, "dinov2_b", "none", "site_dino_method"))
    t["DINOv2 + TreeMortBank (m=0.01)"] = air_cell(uni(s_, "dinov2_b", "none", "site_dino_fz001"))
    res["Site"] = t
    # ---- ablation on CO (DINOv2)
    c = "res_state_CO"
    res["Ablation"] = {
        "1": air_cell(uni(c, "dinov2_b", "none", "co_dino_plain")),
        "2": air_cell(uni(c, "dinov2_b", "none", "co_lr001_p")),
        "3": air_cell(uni(c, "dinov2_b", "none", "co_lr001_photo")),
        "4": air_cell(uni(c, "dinov2_b", "none", "co_lr001_bank")),
        "5": air_cell(uni(c, "dinov2_b", "none", "co_fz001")),
    }
    e = "res_state_EW"
    res["AblationEW"] = {
        "1": air_cell(uni(e, "dinov2_b", "none", "ew_dino_plain")),
        "2": air_cell(uni(e, "dinov2_b", "none", "ew_lr001_p")),
        "3": air_cell(uni(e, "dinov2_b", "none", "ew_lr001_photo")),
        "4": air_cell(uni(e, "dinov2_b", "none", "ew_lr001_bank")),
        "5": air_cell(uni(e, "dinov2_b", "none", "ew_fz001")),
    }
    cfgs = {"CO": "res_state_CO", "EW": "res_state_EW", "WE": "res_state_WE"}
    AUG = [("cnx", "ConvNeXt-UPerNet", "convnext_upernet", {"CO": "convnext", "EW": "convnext", "WE": "convnext"}),
           ("cnx", "+ MixStyle", "convnext_upernet", {"CO": "co_mixstyle", "EW": "ew_mixstyle", "WE": "we_mixstyle"}),
           ("cnx", "+ statistics bank", "convnext_upernet", {"CO": "co_bank_src", "EW": "ew_bank_src", "WE": "we_bank_src"}),
           ("cnx", "+ statistics bank + consistency loss", "convnext_upernet", {"CO": "co_bank_src_cons", "EW": "ew_method", "WE": "we_method"}),
           ("cnx", "+ background-only bank + consistency loss", "convnext_upernet", {"CO": "co_cnx_bg", "EW": "ew_cnx_bg", "WE": "we_cnx_bg"}),
           ("dino", "DINOv2 + decoder ($m{=}0.01$)", "dinov2_b", {"CO": "co_lr001_p", "EW": "ew_lr001_p", "WE": "we_lr001_p"}),
           ("dino", "+ MixStyle", "dinov2_b", {"CO": "co_lr001_ms", "EW": "ew_lr001_ms", "WE": "we_lr001_ms"}),
           ("dino", "+ statistics bank", "dinov2_b", {"CO": "co_lr001_bank", "EW": "ew_lr001_bank", "WE": "we_lr001_bank"}),
           ("dino", "+ statistics bank + consistency loss (Kaleido)", "dinov2_b", {"CO": "co_fz001", "EW": "ew_fz001", "WE": "we_fz001"}),
           ("dino", "+ background-only bank + consistency loss", "dinov2_b", {"CO": "co_lr001_bg", "EW": "ew_lr001_bg", "WE": "we_lr001_bg"})]
    res["AugCmp"] = [{"group": g, "label": lab, "cells": {sc: air_cell(uni(cfgs[sc], model, "none", tags[sc])) for sc in cfgs}} for g, lab, model, tags in AUG]
    Path(a.out).write_text(json.dumps(res, indent=1))
    for sc, rows in res.items():
        if not isinstance(rows, dict):
            continue
        print("==", sc)
        for name, v in rows.items():
            print("  %-36s n=%d  deadF1 %.1f  allF1 %.1f" % (name, v["n"], v["dead_f1"][0], v["all_f1"][0]))


if __name__ == "__main__":
    main()
