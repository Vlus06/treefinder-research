"""print experiment names (airNNNN_model-tag, comma separated) of the finished runs of a config and a tag: python find_exps.py CONFIG_STEM TAG"""
import ast, glob, re, sys
cfg, tag = sys.argv[1], sys.argv[2]
out = []
for f in sorted(glob.glob("/datastore/phuocnth/treefinder/logs/air*.log")):
    t = open(f, errors="ignore").read()
    m = re.search(r"args=(\{.*\})", t)
    if not m or "Combined metrics" not in t:
        continue
    g = ast.literal_eval(m.group(1))
    if g["config"].split("/")[-1].replace(".yaml", "") == cfg and g.get("tag", "") == tag:
        out.append(f"air{str(g['exp_id']).zfill(4)}_{g['model']}-{tag}")
print(",".join(sorted(set(out))))
