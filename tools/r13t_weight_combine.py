"""Combine tools/r13t_weight_rule.py shards (different sampler seeds, same checkpoint/settings) into the rule file.

accel_weight = (G_brake/2)/G_accel_tail, jerk_weight = (G_brake/2)/G_jerk_tail; every G is the geometric mean over
ALL measured batches (braking terms summed after their own geometric means), rounded to 4 significant digits.
    python tools/r13t_weight_combine.py --out docs/r13t_weight_rule.json --status ... shard1.json shard2.json ...
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BRAKING = ("brake_persist", "brake_persist_quantile")


def _gmean(values):
    values = [v for v in values if v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else 0.0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--status", default="MEASURED")
    p.add_argument("--note", action="append", default=[])
    p.add_argument("shards", nargs="+")
    args = p.parse_args(argv)
    shards = [json.loads(Path(s).read_text(encoding="utf-8")) for s in args.shards]
    rows = [r for s in shards for r in s["rows"]]
    names = sorted({k for r in rows for k in r["norms"]})
    gmean = {n: _gmean([r["norms"].get(n, 0.0) for r in rows]) for n in names}
    g_brake = sum(gmean.get(k, 0.0) for k in BRAKING)
    g_a, g_j = gmean["accel_tail"], gmean["jerk_tail"]
    per_batch = [{"accel": sum(r["norms"][k] for k in BRAKING) / 2 / r["norms"]["accel_tail"],
                  "jerk": sum(r["norms"][k] for k in BRAKING) / 2 / r["norms"]["jerk_tail"]} for r in rows]
    ref = shards[0]
    out = {"rule": ref["rule"] + "; tools/r13t_weight_rule.py + tools/r13t_weight_combine.py",
           "status": args.status, "notes": args.note,
           "accel_weight": float(f"{g_brake / 2 / g_a:.4g}"), "jerk_weight": float(f"{g_brake / 2 / g_j:.4g}"),
           "g_brake": g_brake, "g_accel_tail_w1": g_a, "g_jerk_tail_w1": g_j, "geomean_grad_norm": gmean,
           "per_batch_weights": per_batch, "n_batches": len(rows),
           "sampler_seeds": [s.get("sampler_seed") for s in shards], "shards": [Path(s).name for s in args.shards],
           "config": ref["config"], "ckpt": ref["ckpt"], "table": ref["table"], "roots": ref["roots"],
           "overrides": ref["overrides"], "batch_size": ref["batch_size"], "unroll_s": ref["unroll_s"],
           "truncation_steps": ref["truncation_steps"]}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("accel_weight", "jerk_weight", "per_batch_weights", "n_batches")}, indent=1))


if __name__ == "__main__":
    main()
