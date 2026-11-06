"""Combine r10_weight_rule.py shards (different sampler seeds, same checkpoint/settings) into the rule file.

weight = G_brake / G_cutin ; each G is the geometric mean over ALL measured batches of the accumulated gradient
norm (braking terms summed after their own geometric means, as in tools/r10_weight_rule.py).
    python tools/r10_weight_combine.py --out docs/r10_weight_rule.json tmp_r10/w10/m1.json tmp_r10/w10/m2.json ...
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BRAKING = ("brake_persist", "brake_persist_quantile", "asym_speed", "decel_weighted")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("shards", nargs="+")
    args = p.parse_args(argv)
    shards = [json.loads(Path(s).read_text(encoding="utf-8")) for s in args.shards]
    rows = [r for s in shards for r in s["rows"]]
    names = sorted({k for r in rows for k in r["norms"]})
    gmean = {}
    for name in names:
        vals = [r["norms"][name] for r in rows if r["norms"].get(name, 0.0) > 0]
        gmean[name] = math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0
    braking = {k: v for k, v in gmean.items() if k in BRAKING and v > 0}
    g_brake, g_cut = sum(braking.values()), gmean["cutin_reaction"]
    ratios = [sum(v for k, v in r["norms"].items() if k in BRAKING) / r["norms"]["cutin_reaction"]
              for r in rows if r["norms"].get("cutin_reaction", 0.0) > 0]
    weight = float(f"{g_brake / g_cut:.4g}")
    ref = shards[0]
    out = {
        "rule": "weight = G_brake / G_cutin(weight=1): geometric-mean accumulated gradient norm of the explicit braking "
                "terms (brake_persist + brake_persist_quantile) over the round-10 trainable parameters, divided by the "
                "gradient norm of the unweighted cut-in term; tools/r10_weight_rule.py + tools/r10_weight_combine.py",
        "status": "MEASURED",
        "chosen_weight": weight,
        "chosen_weight_note": "weight_geomean_ratio rounded to 4 significant digits (no other adjustment)",
        "weight_geomean_ratio": g_brake / g_cut,
        "g_brake": g_brake, "g_cutin_at_weight_1": g_cut,
        "geomean_grad_norm": gmean, "braking_terms": braking,
        "per_batch_ratio": ratios,
        "per_batch_ratio_min_max": [min(ratios), max(ratios)] if ratios else None,
        "n_batches": len(rows), "shards": [Path(s).name for s in args.shards],
        "sampler_seeds": [s.get("sampler_seed") for s in shards],
        "method": ref["method"],
        "checkpoint": "data/r9_train/wm_s_stage300.pt (round-8 stage-300 weights; init of the round-9 draft; "
                      "merge_pivot new tensors from the round-8 init report)",
        "windows": "local TRAIN episode w963004m1900p40t300__native, 1 s stride, windows whose recorded ramp->lane-1 "
                   "insertion is 24 steps after the window start (117 candidates), unroll 20 s, truncation 40 steps, "
                   "batch size 2, CPU",
        "overrides": ref["overrides"] + ["loss.cutin_reaction.weight=1.0 (tool default)"],
        "caveats": ["merge_pivot loss term disabled for the measurement (loss.merge_core.tf_weight=0, metrics=false): "
                    "the pre-window replay made one batch cost >25 min on CPU; per-term gradient norms are measured "
                    "term by term, so brake/cut-in norms do not depend on its presence",
                    "single local episode, 20 s windows (production unroll 120/300 s), CPU"],
        "trainable_parameters": ref["trainable_parameters"],
    }
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("chosen_weight", "weight_geomean_ratio", "per_batch_ratio", "n_batches")}, indent=1))


if __name__ == "__main__":
    main()
