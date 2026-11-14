"""Validate the generated LPO artifacts (self-check before commit)."""
import hashlib
import json
from pathlib import Path

OUT = Path(r"E:/rd-wt/ws-p5/_out/lpo")

files = ["windows_train_noPI.json", "windows_val.json", "pi_eval_states.json",
         "pi_excluded_runs.json", "BUILD_RECEIPT.json"]
data = {}
for name in files:
    raw = (OUT / name).read_bytes()
    data[name] = json.loads(raw)
    print(f"{name:28s} {len(raw):>10,d} B  sha256 {hashlib.sha256(raw).hexdigest()[:16]}...")

rec = data["BUILD_RECEIPT.json"]
print("\nreceipt:", {k: rec[k] for k in (
    "n_train", "n_val", "n_pi_eval", "n_pi_excluded_keys", "n_unverifiable",
    "n_corpus_train_windows")})
print("unverifiable runs:")
for k in rec["unverifiable_runs"]:
    print("  ", k)

tr = data["windows_train_noPI.json"]
va = data["windows_val.json"]
ev = data["pi_eval_states.json"]
ex = data["pi_excluded_runs.json"]

# 1. counts integrity
assert tr["n_windows"] == len(tr["windows"]) == rec["n_train"]
assert va["n_windows"] == len(va["windows"]) == rec["n_val"]
assert ev["n_windows"] == len(ev["windows"]) == rec["n_pi_eval"]
assert ex["n_runs"] == len(ex["keys"]) == rec["n_pi_excluded_keys"]

# 2. disjointness + coverage (keys are pack-scoped: pack_damping re-packs the
#    same base runs as pack_v3, so (pack, key) is the window identity)
keys_tr = {(w["pack"], w["key"]) for w in tr["windows"]}
keys_va = {(w["pack"], w["key"]) for w in va["windows"]}
keys_ev = {(w["pack"], w["key"]) for w in ev["windows"]}
assert not (keys_tr & keys_va), "train/val overlap"
assert not (keys_tr & keys_ev) and not (keys_va & keys_ev)
keys_un = {(w["pack"], w["key"]) for w in ex["unverifiable"]}
n_total = len(keys_tr) + len(keys_va) + len(keys_ev) + len(keys_un)
assert n_total == 3060, n_total
assert set(ex["keys"]) == {k for _, k in keys_ev | keys_un}

# 3. no PI policy in kept lists
for w in tr["windows"] + va["windows"]:
    assert w["policy"] not in ("pi", "fleet-pi"), w

# 4. val worlds disjoint from train worlds
worlds_tr = {w["world"] for w in tr["windows"]}
worlds_va = {w["world"] for w in va["windows"]}
assert not (worlds_tr & worlds_va), "world leakage between train and val"
print(f"\nworlds: train {len(worlds_tr)}, val {len(worlds_va)}, "
      f"val fraction of windows {len(keys_va) / (len(keys_tr) + len(keys_va)):.3f}")

# 5. counts blocks present (per controller + per world)
for lst, name in ((tr, "train"), (va, "val")):
    c = lst["counts"]
    print(f"{name}: by_policy {c['by_policy']}")
    assert c["total"] == lst["n_windows"]
    assert sum(c["by_world"].values()) == lst["n_windows"]

# 6. eval sources + policy check
srcs = {}
for w in ev["windows"]:
    srcs.setdefault(w["pi_source"], set()).add(w["policy"])
print("pi_eval sources:", {k: (len(v), sorted(v)) for k, v in srcs.items()})
assert all(w["policy"] in ("pi", "switching", "fixed") for w in ev["windows"])

# 7. determinism spot-check: val_world rule
import sys
sys.path.insert(0, r"E:/rd-wt/ws-p5/output/wd01/wm_n1")
from lpo.build_lpo import val_world
assert all(val_world(w) for w in worlds_va)
assert not any(val_world(w) for w in worlds_tr)
print("\nALL CHECKS PASSED")
