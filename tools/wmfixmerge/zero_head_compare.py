"""Compare the zero-head entry path against WM2 on the corrected validation panel.

WM2's WMREQUAL3 batches were scored under the pre-fix source clock (q_1..q_n).
The clock fix changes no rollout, only the rectangle: per row and horizon,
``J_fixed = J_old - .5 * (q_n - q_0)`` with q_n = the row's recorded end-of-horizon
queue and q_0 = the pair's recorded Q0 (paired starts share it).  Both sides are
therefore compared on the labels' left-boundary clock without re-running WM2.

Retarget rule (addendum 03:55 item 4), threshold-free: with SE = bootstrap
standard error of WM2's upstream paired MAE over worlds, the residual is GONE if
either side's upstream MAE is within 2*SE of zero, CHANGED if the sides differ
by more than 2*SE, and REMAINS otherwise (train the head only if it remains).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HORIZONS = (60, 90, 120, 180, 300)
ARMS = ("ours_c1", "coopoff_c1", "mappo")
COMPONENTS = ("source", "upstream", "downstream", "discharge")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_batches(folder):
    rows = []
    for path in sorted(Path(folder).glob("batch_*.json")):
        doc = json.loads(path.read_text())
        rows.extend(doc["rows"])
    if not rows:
        raise ValueError("no batches found in " + str(folder))
    return rows


def index_side(rows, real_native, clock_correct):
    """(pair_key, arm, draw) -> per-horizon component dict, clock-corrected."""
    table = {}
    for r in rows:
        q0 = float(real_native[r["pair_key"]]["queue0_total"])
        values = {}
        for h in HORIZONS:
            cell = dict(r["values"][str(h)])
            if clock_correct:
                shift = .5 * (float(cell["queue"]) - q0)
                cell["source"] -= shift
                cell["J"] -= shift
            values[h] = cell
        table[(r["pair_key"], r["arm"], int(r["draw"]))] = values
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, help="WMREQUAL3 WM2 validation batches")
    parser.add_argument("--entry-dir", required=True, help="zero-head validation batches")
    parser.add_argument("--real", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import numpy as np
    from tools.wmfixmerge.verdict_metrics import metric
    real = json.loads(Path(args.real).read_text())["panels"]["validation"]
    real_native = {r["pair_key"]: r for r in real if r["native"]}
    base = index_side(load_batches(args.base_dir), real_native, True)
    entry = index_side(load_batches(args.entry_dir), real_native, False)
    rows = []
    for key in sorted(real_native):
        native = next(r for r in real if r["pair_key"] == key and r["native"])
        for arm in ARMS:
            armrow = next(r for r in real if r["pair_key"] == key and r["arm"] == arm)
            for h in HORIZONS:
                cell = dict(world=native["world"], pair_key=key, arm=arm, horizon=h)
                for c in ("J",) + COMPONENTS:
                    cell["real" + ("" if c == "J" else "_" + c)] = float(
                        armrow["values"][str(h)][c] - native["values"][str(h)][c])
                for name, table in (("wm2", base), ("entry", entry)):
                    cell[name + "_draws"] = [
                        float(table[(key, arm, d)][h]["J"] - table[(key, "native", d)][h]["J"])
                        for d in range(8)]
                    for c in COMPONENTS:
                        cell[f"{name}_{c}_draws"] = [
                            float(table[(key, arm, d)][h][c] - table[(key, "native", d)][h][c])
                            for d in range(8)]
                rows.append(cell)
    worlds = sorted({r["world"] for r in rows})
    groups = [[r for r in rows if r["world"] == w] for w in worlds]
    draws_idx = np.random.default_rng(0).integers(0, len(groups), (2000, len(groups)))

    def panel(subset):
        return {name: metric(subset, name) for name in ("wm2", "entry")}

    def component_mae(subset, name, c):
        return float(np.mean([abs(float(np.median(r[f"{name}_{c}_draws"]))
                                  - r["real" + ("" if c == "J" else "_" + c)]) for r in subset]))

    report = dict(schema="wmfix_zero_head_compare/1",
                  base_batches={p.name: digest(p) for p in
                                sorted(Path(args.base_dir).glob("batch_*.json"))},
                  entry_batches={p.name: digest(p) for p in
                                 sorted(Path(args.entry_dir).glob("batch_*.json"))},
                  real_sha256=digest(args.real), pooled=panel(rows), horizons={},
                  components={}, n_worlds=len(worlds), n_rows=len(rows))
    for h in HORIZONS:
        subset = [r for r in rows if r["horizon"] == h]
        report["horizons"][str(h)] = panel(subset)
        for c in COMPONENTS:
            report["components"][f"{h}s_{c}"] = {name: component_mae(subset, name, c)
                                                 for name in ("wm2", "entry")}
    tail = [r for r in rows if r["horizon"] in (120, 180, 300)]

    def boot_se(fn):
        values = [float(np.mean(fn([r for i in draw for r in groups[i]]))) for draw in draws_idx]
        return float(np.std(values, ddof=1))

    se = boot_se(lambda subset: [abs(float(np.median(r["wm2_upstream_draws"]))
                                     - r["real_upstream"]) for r in subset])
    wm2_up = component_mae(tail, "wm2", "upstream")
    entry_up = component_mae(tail, "entry", "upstream")
    gone = min(wm2_up, entry_up) <= 2 * se
    changed = abs(entry_up - wm2_up) > 2 * se
    report["retarget"] = dict(upstream_paired_mae_120_300=dict(wm2=wm2_up, entry=entry_up),
                              bootstrap_se=se, gone=bool(gone), changed=bool(changed),
                              residual_remains=bool(not (gone or changed)))
    out = Path(args.out)
    with out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report["retarget"], indent=2))


if __name__ == "__main__":
    main()
