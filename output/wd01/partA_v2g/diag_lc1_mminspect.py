"""Inspect the LC1 counting-audit mismatches (diag support)."""
import json
from collections import Counter

HERE = r"output/wd01/partA_v2g"

rep = json.load(open(HERE + r"\results\lc1\audit_report.json"))
print("first mismatches from report:")
for m in rep["window_mismatches"][:10]:
    print(" ", m["key"], "eval", m["eval"], "ind", m["independent"])

eval_lc = {}
for line in open(HERE + r"\results\eval_a2\raw_rows.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("model") == "v2g":
        eval_lc[(r["world"], round(float(r["w"]), 3), int(r["H"]))] = \
            r["lc_true"]

hi_eval = hi_ind = 0
byH = Counter()
byworld = Counter()
diffs = []
for line in open(HERE + r"\results\lc1\truth_windows.jsonl"):
    w = json.loads(line)
    key = (w["world"], round(w["w"], 3), w["H"])
    ref = eval_lc.get(key)
    if ref is None:
        continue
    mine = {"merge": w["merge"], "disc_left": w["disc_left"],
            "disc_right": w["disc_right"]}
    if mine != ref:
        d = {k: mine[k] - ref[k] for k in ref if ref[k] != mine[k]}
        hi_eval += sum(1 for v in d.values() if v < 0)
        hi_ind += sum(1 for v in d.values() if v > 0)
        byH[key[2]] += 1
        byworld[key[0]] += 1
        diffs.append((key, ref, mine))

print("mismatch field-direction: eval-higher", hi_eval,
      "independent-higher", hi_ind)
print("by H:", dict(byH))
print("worlds affected:", len(byworld), dict(list(byworld.items())[:10]))
print("total mismatch rows:", len(diffs))
for d in diffs[:10]:
    print(" ", d)
