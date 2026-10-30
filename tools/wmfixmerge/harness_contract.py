"""Emit the wmfix_harness_contract/1 receipt: source-clock contract + measured shift.

The WMREQUAL3 scorer summed the post-service queue samples q_1..q_n while the
recorded labels (tools/wmrequal/labels.py) sum the left boundary q_0..q_{n-1}
(S-GAPCHECK rectangle).  Both samplers now seed the pre-rollout Q0 sample and
drop the final post-service sample, so the predicted and recorded rectangles
share one clock.  This builder quantifies the one-tick shift that the WMREQUAL3
numbers carried, directly from the recorded labels of the validation panel
(no model): per row ``.5*(q_h - q_0)`` and per paired contrast
``.5*(q_h^arm - q_h^native)`` (paired starts share Q0).
"""
from __future__ import annotations

import argparse
import json

from tools.wmfixmerge.data import digest, safe

HORIZONS = ("60", "90", "120", "180", "300")
FILES = ("tools/wmrequal/labels.py", "tools/wmrequal/score.py",
         "tools/wmrequal/native_lanes.py", "tools/wmfixmerge/runtime.py")


def measure(real_rows):
    """Per-horizon row/contrast shift statistics from recorded validation labels."""
    rows = [r for r in real_rows if r.get("source_available")]
    if not rows:
        raise ValueError("no recorded rows with exact source labels")
    stats = {}
    for h in HORIZONS:
        shifts, contrasts = [], []
        natives = {r["pair_key"]: r for r in rows if r["native"] and h in r["values"]}
        for r in rows:
            if h not in r["values"]:
                continue
            shifts.append(.5 * (float(r["values"][h]["queue"]) - float(r["queue0_total"])))
            ref = natives.get(r["pair_key"])
            if ref is not None and not r["native"]:
                contrasts.append(.5 * (float(r["values"][h]["queue"])
                                       - float(ref["values"][h]["queue"])))
        if not shifts or not contrasts:
            raise ValueError(f"horizon {h} lacks rows or paired contrasts")
        def agg(values):
            import numpy as np
            a = np.abs(np.asarray(values, float))
            return dict(mean=float(a.mean()), median=float(np.median(a)),
                        max=float(a.max()), n=len(values))
        source = [abs(float(r["values"][h]["source"])) for r in rows if h in r["values"]]
        stats[h] = dict(row_shift_veh_s=agg(shifts), contrast_shift_veh_s=agg(contrasts),
                        mean_abs_source_veh_s=float(sum(source) / len(source)))
    return stats


def build(corrected_report, prepared, real, residual_remains, out):
    doc = json.loads(safe(real).read_text())
    rows = (doc["panels"]["validation"] if "panels" in doc else doc)
    contract = dict(
        schema="wmfix_harness_contract/1",
        source_clock=("labels left boundary q_0..q_{n-1} of post-service samples; "
                      "runtime/scorer seed the pre-rollout Q0 and drop the final sample"),
        corrected_report_sha256=digest(corrected_report),
        prepared_sha256=digest(prepared),
        real_sha256=digest(real),
        native_lanes_sha256=digest("tools/wmrequal/native_lanes.py"),
        scorer_sha256=digest("tools/wmrequal/score.py"),
        runtime_sha256=digest("tools/wmfixmerge/runtime.py"),
        labels_sha256=digest("tools/wmrequal/labels.py"),
        wmrequal3_one_tick_shift=measure(rows),
        residual_remains=bool(residual_remains),
    )
    contract["source_clock_match"] = all(
        contract[key] == digest(path) for key, path in
        (("scorer_sha256", "tools/wmrequal/score.py"),
         ("runtime_sha256", "tools/wmfixmerge/runtime.py"),
         ("labels_sha256", "tools/wmrequal/labels.py"),
         ("native_lanes_sha256", "tools/wmrequal/native_lanes.py")))
    if not contract["source_clock_match"]:
        raise ValueError("contract digests must match the tree they are built from")
    target = safe(out)
    if not target.parent.is_dir():
        raise ValueError("output parent must exist")
    with target.open("x") as stream:
        json.dump(contract, stream, indent=2, allow_nan=False)
    return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected-report", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--real", required=True)
    parser.add_argument("--residual-remains", type=int, choices=(0, 1), required=True,
                        help="1 iff the zero-head entry path leaves the upstream residual")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build(args.corrected_report, args.prepared, args.real, args.residual_remains, args.out)
