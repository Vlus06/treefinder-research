"""Dataset characteristics used in the paper's data analysis (reads datasets/packed/tile_stats.csv)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEST = set("AR AZ CA CO IA ID KS LA MN MO MT ND NE NM NV OK OR SD TX UT WA WY".split())

info = pd.read_csv(ROOT / "datasets" / "tile_info224_v3_aux.csv")
st = pd.read_csv(ROOT / "datasets" / "packed" / "tile_stats.csv")
d = pd.concat([info, st.drop(columns=["FileName", "index"])], axis=1)
d["region"] = np.where(d.State.isin(WEST), "West", "East")
d["pos"] = d.n_dead > 0
out = {}

valid = d.n_valid.sum()
out["tiles"] = len(d)
out["sites"] = int(d.ImageRawPath.nunique())
out["valid_ha"] = float(valid * 0.36 / 1e4)
out["positive_pixel_rate"] = float(d.n_dead.sum() / valid)
out["positive_tile_frac"] = float(d.pos.mean())
out["nodata_frac"] = float(1 - valid / (len(d) * 224 * 224))
out["tiles_per_site"] = d.groupby("ImageRawPath").size().describe().round(1).to_dict()

per_tile_rate = (d.n_dead / d.n_valid.clip(lower=1))[d.pos]
out["positive_tile_dead_rate_quantiles"] = per_tile_rate.quantile([.1, .25, .5, .75, .9, .99]).round(4).to_dict()

comps = np.concatenate([json.loads(s) for s in d.component_sizes if s != "[]"])
area_m2 = comps * 0.36
out["n_components"] = int(len(comps))
out["component_pixels_quantiles"] = {str(q): float(np.quantile(comps, q)) for q in (.1, .25, .5, .75, .9, .99)}
out["component_area_m2_median"] = float(np.median(area_m2))
out["components_le_16px_frac"] = float((comps <= 16).mean())
out["components_le_64px_frac"] = float((comps <= 64).mean())
out["pixel_mass_in_components_le_64px"] = float(comps[comps <= 64].sum() / comps.sum())
out["components_per_positive_tile_median"] = float(d.loc[d.pos, "n_components"].median())

# per-region imbalance and radiometric shift
reg = d.groupby("region").agg(tiles=("FileName", "size"), sites=("ImageRawPath", "nunique"),
                              dead=("n_dead", "sum"), valid=("n_valid", "sum"),
                              pos_tiles=("pos", "mean"))
reg["dead_rate"] = reg.dead / reg.valid
out["by_region"] = reg.round(5).to_dict("index")
bands = ["mean_R", "mean_G", "mean_B", "mean_N", "mean_ndvi", "dead_ndvi", "live_ndvi"]
out["band_means_by_region"] = d.groupby("region")[bands].mean().round(3).to_dict("index")
out["band_means_dead_vs_live"] = {
    "dead": {b: float(d[f"dead_mean_{b}"].mean()) for b in "RGBN"},
    "live": {b: float(d[f"live_mean_{b}"].mean()) for b in "RGBN"},
    "dead_ndvi": float(d.dead_ndvi.mean()), "live_ndvi": float(d.live_ndvi.mean()),
}
# how well can a single global threshold separate dead/live within vs across sites?
site = d.groupby("ImageRawPath").agg(live_ndvi=("live_ndvi", "mean"), dead_ndvi=("dead_ndvi", "mean"), n=("n_dead", "sum"))
site = site[site.n > 0].dropna()
out["site_level_ndvi_std_of_live"] = float(site.live_ndvi.std())
out["site_level_ndvi_gap_dead_minus_live_mean"] = float((site.dead_ndvi - site.live_ndvi).mean())
out["site_level_ndvi_gap_std"] = float((site.dead_ndvi - site.live_ndvi).std())
# dead pixel stats by state: concentration
by_state = d.groupby("State").agg(dead=("n_dead", "sum"), valid=("n_valid", "sum"), sites=("ImageRawPath", "nunique"))
by_state["rate"] = by_state.dead / by_state.valid
out["state_dead_rate_quantiles"] = by_state.rate.quantile([0, .25, .5, .75, 1]).round(5).to_dict()
out["top5_states_share_of_dead_pixels"] = float(by_state.dead.sort_values(ascending=False).head(5).sum() / by_state.dead.sum())
site_dead = d.groupby("ImageRawPath").n_dead.sum().sort_values(ascending=False)
out["top10pct_sites_share_of_dead_pixels"] = float(site_dead.head(len(site_dead) // 10).sum() / site_dead.sum())
out["sites_without_dead"] = float((site_dead == 0).mean())
out["climate_top"] = d.ClimateType.value_counts().head(8).to_dict()
out["treetype_top"] = d.TreeTypes.value_counts().head(8).to_dict()

Path(ROOT / "research" / "reports").mkdir(exist_ok=True)
(ROOT / "research" / "reports" / "dataset_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=1))
