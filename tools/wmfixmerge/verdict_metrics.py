"""Registered paired-world verdict gates; complete inputs only, no legacy sigma."""
from __future__ import annotations

import numpy as np


def metric(rows, which):
    real = np.array([r["real"] for r in rows], float)
    draws = np.array([r[which + "_draws"] for r in rows], float)
    if draws.shape != (len(rows), 8) or not np.isfinite(draws).all() or not np.isfinite(real).all():
        raise ValueError("finite K8 contrasts required")
    pred = np.median(draws, 1)
    nonzero = real != 0
    confident = np.quantile(-draws, .1, axis=1) > 0
    sign = float((np.sign(pred[nonzero]) == np.sign(real[nonzero])).mean()) if nonzero.any() else None
    selected = confident & nonzero
    accuracy = float((real[selected] < 0).mean()) if selected.any() else None
    corr = float(np.corrcoef(pred, real)[0, 1]) if pred.std() and real.std() else None
    return dict(mae=float(abs(pred - real).mean()), sign=sign, r=corr, bias=float((pred - real).mean()),
                confident_gain_accuracy=accuracy, confident_gain_count=int(confident.sum()),
                confident_gain_nonzero_count=int(selected.sum()),
                confident_gain_coverage=float(confident.mean()))


def comparisons(rows):
    if not rows:
        raise ValueError("empty panel")
    names = ("base", "entry", "candidate")
    values = {n: metric(rows, n) for n in names}
    worlds = sorted({r["world"] for r in rows})
    groups = [[r for r in rows if r["world"] == w] for w in worlds]
    draws = np.random.default_rng(0).integers(0, len(groups), (2000, len(groups)))
    metrics = ("mae", "sign", "r", "confident_gain_accuracy")
    pairs = (("base", "entry"), ("entry", "candidate"), ("base", "candidate"))
    boot = {f"{b}>{c}": {m: [] for m in metrics} for b, c in pairs}
    for draw in draws:
        sampled = [r for i in draw for r in groups[i]]
        m = {n: metric(sampled, n) for n in names}
        for b, c in pairs:
            for x in metrics:
                if m[b][x] is not None and m[c][x] is not None:
                    boot[f"{b}>{c}"][x].append(m[c][x] - m[b][x])
    ci = {k: {m: np.quantile(v[m], [.025, .975]).tolist() if v[m] else None
              for m in metrics} for k, v in boot.items()}
    # gates compare the trained candidate against WM2 only; entry (zero-head
    # entry path) and head-over-entry deltas are reported for attribution.
    return dict(base=values["base"], entry=values["entry"], candidate=values["candidate"],
                delta_ci95=ci["base>candidate"],
                attribution_delta_ci95={k: ci[k] for k in ("base>entry", "entry>candidate")},
                n_worlds=len(worlds), n_rows=len(rows))


def validation(doc):
    if doc.get("schema") != "wmfix_validation/2" or doc.get("K") != 8 \
            or doc.get("mix") != "ON" or doc.get("native_lanes") is not True:
        raise ValueError("not corrected native-lane K8/mix ON three-way validation")
    rows = doc["rows"]
    if {r["arm"] for r in rows} != {"ours_c1", "coopoff_c1", "mappo"}:
        raise ValueError("unexpected validation arm")
    keys = [(r["world"], r["pair_key"], r["arm"], int(r["horizon"])) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate validation rows")
    horizons = (60, 90, 120, 180, 300)
    if len({r["pair_key"] for r in rows}) != 36:
        raise ValueError("validation must contain all 36 registered groups")
    for key in {r["pair_key"] for r in rows}:
        for arm in ("ours_c1", "coopoff_c1", "mappo"):
            if {int(r["horizon"]) for r in rows if r["pair_key"] == key and r["arm"] == arm} != set(horizons):
                raise ValueError("incomplete validation arm/horizon grid")
    # MAPPO's lane-policy arm is reported but excluded from native-lane qualification.
    primary = [r for r in rows if r["arm"] in ("ours_c1", "coopoff_c1")]
    report = dict(pooled=comparisons(primary), horizons={}, diagnostic_mappo={}, reasons=[])
    pooled = report["pooled"]
    if pooled["delta_ci95"]["mae"][1] >= 0:
        report["reasons"].append("pooled paired-J MAE improvement CI does not exclude zero")
    for m in ("sign", "r", "confident_gain_accuracy"):
        b, c = pooled["base"][m], pooled["candidate"][m]
        if b is None or c is None or c <= b:
            report["reasons"].append("pooled " + m + " not improved/available")
    for h in horizons:
        panel = comparisons([r for r in primary if int(r["horizon"]) == h])
        report["horizons"][str(h)] = panel
        report["diagnostic_mappo"][str(h)] = {
            n: metric([r for r in rows if r["arm"] == "mappo" and int(r["horizon"]) == h], n)
            for n in ("base", "entry", "candidate")}
        for m, ci in panel["delta_ci95"].items():
            if ci is None or (ci[0] > 0 if m == "mae" else ci[1] < 0):
                report["reasons"].append(f"{h}s {m} significant regression/unavailable")
    report["passed"] = not report["reasons"]
    report["H_G"] = None
    return report


def speedhead(doc):
    if doc.get("schema") != "wmfix_speedhead/1" or doc.get("K") != 8 \
            or doc.get("mix") != "ON" or doc.get("native_lanes") is not True \
            or doc.get("matched_t300_history_queue") is not True \
            or doc.get("closed_loop_all_cavs") is not True:
        raise ValueError("not exact-start four-arm closed-loop SPEEDHEAD K8 panel")
    rows = doc["rows"]
    expected = {(str(w), a, h) for w in range(963880, 963888)
                for a in ("MAPPO", "PI", "CIRCLES") for h in (300, 600)}
    actual = [(str(r["world"]), r["arm"], int(r["horizon"])) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("SPEEDHEAD missing/duplicate worlds, arms, or horizons")
    absolute = doc["absolute"]
    absolute_keys = [(str(r["world"]), r["arm"], int(r["horizon"])) for r in absolute]
    if len(absolute_keys) != 64 or set(absolute_keys) != {
            (str(w), a, h) for w in range(963880, 963888)
            for a in ("CTRL0", "MAPPO", "PI", "CIRCLES") for h in (300, 600)}:
        raise ValueError("all four absolute arm grids must be retained")
    lookup = {(str(r["world"]), r["arm"], int(r["horizon"])): r for r in absolute}
    for arm in absolute:
        components = arm["components"]
        if not np.isclose(components["real"]["road"] + components["real"]["source"], arm["real"]):
            raise ValueError("SPEEDHEAD real road/source do not sum to J")
        for name in ("base", "candidate"):
            sums = np.asarray(components[name]["road"]) + np.asarray(components[name]["source"])
            if not np.allclose(sums, arm[name + "_draws"], rtol=1e-10, atol=1e-8):
                raise ValueError("SPEEDHEAD WM road/source do not sum to J")
    for row in rows:
        arm = lookup[str(row["world"]), row["arm"], int(row["horizon"])]
        ref = lookup[str(row["world"]), "CTRL0", int(row["horizon"])]
        for name in ("base", "candidate"):
            expected_draw = np.asarray(arm[name + "_draws"]) - np.asarray(ref[name + "_draws"])
            if not np.allclose(expected_draw, row[name + "_draws"], rtol=1e-10, atol=1e-8):
                raise ValueError("arm-minus-CTRL0 draw mismatch")
        if not np.isclose(arm["real"] - ref["real"], row["real"], rtol=1e-10, atol=1e-8):
            raise ValueError("real arm-minus-CTRL0 mismatch")
    report = dict(per_world=rows, absolute=absolute, horizons={}, reasons=[])
    for h in (300, 600):
        panel = [r for r in rows if int(r["horizon"]) == h]
        b, c = metric(panel, "base"), metric(panel, "candidate")
        arms = {a: dict(base=metric([r for r in panel if r["arm"] == a], "base"),
                       candidate=metric([r for r in panel if r["arm"] == a], "candidate"))
                for a in ("MAPPO", "PI", "CIRCLES")}
        report["horizons"][str(h)] = dict(base=b, candidate=c, arms=arms)
        if c["mae"] >= b["mae"]:
            report["reasons"].append(f"{h}s pooled magnitude MAE not lower")
        if b["sign"] is None or c["sign"] is None or c["sign"] < b["sign"]:
            report["reasons"].append(f"{h}s fewer correct directions/unavailable")
        for arm, values in arms.items():
            if values["candidate"]["mae"] > values["base"]["mae"]:
                report["reasons"].append(f"{h}s {arm} MAE worse")
    report["passed"] = not report["reasons"]
    return report


def c1(doc):
    if doc.get("schema") != "wmfix_c1_regression/1" or doc.get("reference_label") != "C1" \
            or not doc.get("canonical_cohort_sha256"):
        raise ValueError("not the unchanged hash-bound C1 zero-shot cohort")
    rows = doc["rows"]
    report = dict(cohorts={}, reasons=[])
    for cohort in ("PI", "unseen_wall"):
        part = [r for r in rows if r["cohort"] == cohort]
        if not part or len({r["world"] for r in part}) != len(part):
            raise ValueError("empty/duplicate C1 cohort worlds")
        expected = doc.get("canonical_worlds", {}).get(cohort)
        if not expected or set(expected) != {r["world"] for r in part}:
            raise ValueError("C1 cohort incomplete or changed")
        for row in part:
            if any(not np.isfinite(row[k + "_error"]) or row[k + "_error"] < 0 for k in ("base", "candidate")) \
                    or any(not isinstance(row[k + "_blowups"], int) or row[k + "_blowups"] < 0
                           for k in ("base", "candidate")):
                raise ValueError("invalid C1 error/blowup measurement")
        difference = np.array([r["candidate_error"] - r["base_error"] for r in part], float)
        if not np.isfinite(difference).all():
            raise ValueError("nonfinite C1 state error")
        draw = np.random.default_rng(0).integers(0, len(part), (2000, len(part)))
        ci = np.quantile(difference[draw].mean(1), [.025, .975]).tolist()
        blowups = [sum(r[k + "_blowups"] for r in part) for k in ("base", "candidate")]
        report["cohorts"][cohort] = dict(n_worlds=len(part), error_delta_ci95=ci, blowups=blowups)
        if ci[0] > 0 or blowups[1] > blowups[0]:
            report["reasons"].append(cohort + " state-error/blowup regression")
    report["passed"] = not report["reasons"]
    return report
