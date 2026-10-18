"""What are the 'mega' tiles (dead fraction > 20 % of valid pixels)? Statistics + gallery."""
import json, glob, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

x = np.load("datasets/packed/x.npy", mmap_mode="r"); y = np.load("datasets/packed/y.npy", mmap_mode="r")
info = pd.read_csv("datasets/tile_info224_v3_aux.csv"); st = pd.read_csv("datasets/packed/tile_stats.csv")
d = pd.concat([info, st.drop(columns=["FileName", "index"])], axis=1)
d["frac"] = d.n_dead / d.n_valid.clip(lower=1)
d["mega"] = d.frac > 0.20
sites = d.groupby("ImageRawPath").agg(State=("State", "first"), tiles=("FileName", "size"), mega_tiles=("mega", "sum"), dead=("n_dead", "sum"),
                                     valid=("n_valid", "sum"), climate=("ClimateType", "first"), tree=("TreeTypes", lambda s: s.dropna().mode().iat[0] if s.notna().any() else ""))
sites["dead_frac_%"] = 100 * sites.dead / sites.valid
sites["share_of_all_dead_%"] = 100 * sites.dead / d.n_dead.sum()
top = sites.sort_values("dead", ascending=False).head(12)
print("== 12 sites with the most dead pixels (of %d sites)" % len(sites))
print(top[["State", "tiles", "mega_tiles", "dead_frac_%", "share_of_all_dead_%", "climate", "tree"]].round(1).to_string())
print("cumulative share of the top-5/10/20 sites: %.1f / %.1f / %.1f %%" % tuple(sites.sort_values("dead", ascending=False).dead.cumsum().iloc[[4, 9, 19]] / d.n_dead.sum() * 100))

# spectral signature of what is labelled in mega tiles vs normal tiles
def sig(idx):
    out = []
    for i in idx:
        t = np.asarray(x[i]).astype(np.float32); lab = np.asarray(y[i]); v = lab != 255
        dead = lab == 1; live = v & ~dead
        if dead.sum() < 20 or live.sum() < 20: continue
        R, G, B, N = t; nd = (N - R) / (N + R + 1e-6)
        out.append((nd[dead].mean(), nd[live].mean(), (R + G + B)[dead].mean() / 3, (R + G + B)[live].mean() / 3, N[dead].mean(), N[live].mean()))
    return np.array(out).mean(0)
m_idx = d[d.mega].index.tolist(); n_idx = d[(d.n_dead > 100) & (~d.mega)].sample(400, random_state=0).index.tolist()
for nm, idx in (("mega tiles", m_idx), ("normal tiles with dead trees", n_idx)):
    s = sig(idx)
    print("%-30s NDVI dead %.3f vs live %.3f | brightness dead %.0f vs live %.0f | NIR dead %.0f vs live %.0f" % (nm, *s[[0, 1, 2, 3, 4, 5]]))

# per mega-site detection when it lands in a test split
amap = {p: i for i, p in enumerate(sorted(info.ImageRawPath.unique()))}
inv = {v: k for k, v in amap.items()}
rec = {}
for f in glob.glob("research/reports/cluster_backup/2026-10-17/eval_extra/*.json"):
    r = json.load(open(f)); a = r["args"]
    if "site" not in a["config"] or a["model"] not in ("convnext_upernet", "treemortseg") or a.get("tag") or (a["model"] == "treemortseg" and a["aux"] != "none"):
        continue
    for s in r["per_site"]:
        p = inv[s["site"]]
        if p in sites.index and sites.loc[p, "mega_tiles"] > 0 and s["tp"] + s["fn"] > 0:
            rec.setdefault(p, []).append(s["tp"] / (s["tp"] + s["fn"]))
print("\n== recall on mega-containing sites when they are in a TEST split (site split, ConvNeXt + TreeMortSeg runs)")
for p, v in sorted(rec.items(), key=lambda kv: -sites.loc[kv[0], "dead"]):
    print("%-34s %s  n_test_runs=%d  recall: %s" % (p, sites.loc[p, "State"], len(v), " ".join("%.0f%%" % (100 * q) for q in v)))

# gallery: 8 mega tiles from different sites
pick = []
seen = set()
for i in d[d.mega].sort_values("n_dead", ascending=False).index:
    s = d.ImageRawPath[i]
    if s in seen: continue
    seen.add(s); pick.append(i)
    if len(pick) == 8: break
def stretch(a, v):
    lo, hi = np.percentile(a[v], (2, 98)); return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
fig, ax = plt.subplots(2, 4, figsize=(15, 7.6))
for k, i in enumerate(pick):
    t = np.asarray(x[i]); lab = np.asarray(y[i]); v = lab != 255; dead = lab == 1
    rgb = np.dstack([stretch(t[c].astype(np.float32), v) for c in range(3)]); rgb[~v] = .5
    a = ax[k // 4, k % 4]; a.imshow(rgb)
    o = np.zeros(dead.shape + (4,)); e = dead & ~ndimage.binary_erosion(dead); o[e] = [1, 1, 0, 1]; a.imshow(o)
    a.set_title("%s %s  %.0f%% labelled dead\n%s" % (info.State[i], info.FileName[i], 100 * d.frac[i], str(info.TreeTypes[i]).split(" - ")[0]), fontsize=9)
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("Tiles with >20 % of pixels labelled dead, one per site (yellow = label outline)", fontsize=12)
plt.tight_layout(); plt.savefig("research/reports/mega_tiles.png", dpi=95)
