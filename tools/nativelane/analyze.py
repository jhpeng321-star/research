"""Registered world-cluster comparisons and fail-closed NATIVELANE decision."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path.home() / "nativelane"
sys.path[:0] = [str(Path.home() / "mappodiag"), str(Path.home() / "actqual/tools")]
import analyze_b as A  # noqa: E402
import coopbehav as CB  # noqa: E402
import qual_analyze as Q  # noqa: E402

Q.RNG_SEED = 0
Q.NBOOT = 2000


def save(name, doc):
    p = ROOT / name
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=float) + "\n")
    tmp.replace(p)


def boot(rows, stat):
    return A.boot(np.arange(len(rows)), [r["world"] for r in rows], n=2000, seed=0,
                  stat=lambda ii: stat([rows[int(i)] for i in ii]))


def rate(rows, side, event="vacate"):
    return 3600 * sum(r[side][event] for r in rows) / sum(r[side]["exposure_cav_s"] for r in rows)


def ordinary_wm(path):
    with np.load(path) as z:
        ln, pr, cav = z["lane"][:, :, 1], z["present"][:, :, 1].astype(bool), z["is_cav"][:, :, 1].astype(bool)
        good = pr[:, 1:] & pr[:, :-1] & cav[:, 1:] & cav[:, :-1]
        changed = good & (ln[:, 1:] != ln[:, :-1]) & (ln[:, 1:] >= 0) & (ln[:, :-1] >= 0)
        ordinary = changed & (ln[:, 1:] < 2) & (ln[:, :-1] < 2)
        return {"count": float(ordinary.sum(1).sum(1).mean()),
                "all_count": float(changed.sum(1).sum(1).mean()),
                "exposure_cav_s": float((pr[:, 1:] & cav[:, 1:]).sum(1).sum(1).mean()),
                "samples": ln.shape[0]}


def ordinary_sumo(path, t0, horizon):
    df = CB.load_traj(path)
    df = df[df.cav].sort_values(["vid", "t"])
    prev = df.groupby("vid", sort=False)[["t", "lane"]].shift()
    good = (df.t - prev.t == 1) & (df.t > t0) & (df.t <= t0 + horizon)
    changed = good & (df.lane != prev.lane) & (df.lane >= 0) & (prev.lane >= 0)
    ordinary = changed & (df.lane < 2) & (prev.lane < 2)
    return {"count": int(ordinary.sum()), "all_count": int(changed.sum()),
            "exposure_cav_s": int(((df.t > t0) & (df.t <= t0 + horizon)).sum())}


def native_counts(off, on):
    byon = {r["unit"]: r for r in on}
    rows = []
    for a in off:
        b = byon[a["unit"]]
        seed, t0 = a["seed"], a["t0"]
        jobs = list((Path.home() / "coopmeas/run/jobs").glob(f"main_branch_s{seed}_p*"))
        assert len(jobs) == 1
        rows.append({"unit": a["unit"], "world": a["world"],
                     "off": ordinary_wm(ROOT / "off/wm_coop" / (a["unit"] + ".npz")),
                     "on": ordinary_wm(ROOT / "on/wm_coop" / (b["unit"] + ".npz")),
                     "sumo": ordinary_sumo(jobs[0] / f"traj_t{t0}_B.npz", t0, 60)})
    res = {"rows": rows, "n": len(rows)}
    for side in ("off", "on", "sumo"):
        res[side] = {"ordinary_total": sum(r[side]["count"] for r in rows),
                     "all_lane_total": sum(r[side]["all_count"] for r in rows),
                     "ordinary_per_start_ci95": boot(rows, lambda rr: np.mean([r[side]["count"] for r in rr])),
                     "ordinary_rate_per_cav_h_ci95": boot(rows, lambda rr: rate(rr, side, "count"))}
    def improvement(rr):
        s = sum(r["sumo"]["count"] for r in rr)
        return (abs(sum(r["off"]["count"] for r in rr) - s)
                - abs(sum(r["on"]["count"] for r in rr) - s)) / len(rr)
    res["count_gap_reduction_per_start_ci95"] = boot(rows, improvement)
    res["rate_gap_reduction_ci95"] = boot(rows, lambda rr:
        abs(rate(rr, "off", "count") - rate(rr, "sumo", "count"))
        - abs(rate(rr, "on", "count") - rate(rr, "sumo", "count")))
    res["counts_move_toward_sumo"] = res["count_gap_reduction_per_start_ci95"][0] > 0
    return res


def qualification():
    tasks = json.loads((ROOT / "QUAL_TASKS.json").read_text())
    assert tasks["complete"], tasks
    sp, sc = [str(Path.home() / f"c4/runs/{s}") for s in ("pulse", "coop")]
    for tag in ("off", "on"):
        for k in range(6):
            for job in sorted((Path(sp) / "jobs").glob("s*_t300")):
                for tape in job.glob("traj__*.npz"):
                    assert (ROOT / tag / "pulse/wm" / f"k{k}" / job.name / tape.name).is_file(), "missing pulse tape"
            var = "as_is" if k == 0 else f"as_is_k{k}"
            for job in sorted((Path(sc) / "jobs").glob("s*_t300")):
                if (job / "traj__A__noop.npz").exists():
                    for name in ("traj__A__noop.npz", "traj__A__interv.npz"):
                        assert (ROOT / tag / "c9/wm" / var / job.name / name).is_file(), "missing gap-open tape"
    tol = json.loads((Path.home() / "actqual/reg/TOLERANCES_v2.json").read_text())
    sumo = Q.add_contrasts(Q.pulse_events(sp)) + [r for r in Q.c9_events(sc) if r["item"] == "gap_open"]
    units = ("pulse_m/follower@5", "pulse_m/follower@10", "gap_open/follower@10", "gap_open/follower@30")
    out = {}
    for tag in ("off", "on"):
        out[tag] = {}
        for sname, ks in Q.SETS.items():
            def loader(k):
                pr, cr = str(ROOT / tag / "pulse"), str(ROOT / tag / "c9")
                var = "as_is" if k == 0 else f"as_is_k{k}"
                rows = Q.pulse_events(sp, pr, k)
                rows += [r for r in Q.c9_events(sc, cr, var) if r["item"] == "gap_open"]
                return Q.add_contrasts(rows)
            wm = Q.wm_set_rows(loader, ks)
            assert wm and all(r.get("n_samples") == 3 for r in wm.values()), "missing qualification sample"
            scores = Q.score_units(sumo, wm, tol)
            out[tag][sname] = {u: scores[u] for u in units}
    flips = []
    for sname in Q.SETS:
        for u in units:
            if out["off"][sname][u]["sign_verdict"] == "PASS" and out["on"][sname][u]["sign_verdict"] == "FAIL":
                flips.append(f"{sname}:{u}")
    out["pass_to_fail"] = flips
    out["complete"] = True
    return out


def parity():
    checks = json.loads((ROOT / "CHECKS.json").read_text())
    assert checks["passed"]
    result = {"synthetic_bit_identity": checks["bitid"], "historical_off": {}}
    for tag in ("off", "on"):
        for suite, expected, horizon in (("coop", 33, 60), ("c", 12, 300)):
            units = sorted((ROOT / tag / f"wm_{suite}").glob("w*_t*.json"))
            assert len(units) == expected
            for p in units:
                unit = json.loads(p.read_text())
                assert "error" not in unit and unit.get("finite") is True, p.name
                for name in ("delay", "source_wait"):
                    values = np.asarray(unit["J"][f"{name}@{horizon}"])
                    assert values.shape == (4, 2) and np.isfinite(values).all(), p.name
    for suite, expected in (("coop", 33), ("c", 12)):
        bad, n = [], 0
        for p in sorted((ROOT / f"off/wm_{suite}").glob("w*_t*.npz")):
            prior = Path.home() / f"mappodiag/wm_{suite}" / p.name
            with np.load(p) as a, np.load(prior) as b:
                if any(not np.array_equal(a[k], b[k], equal_nan=True) for k in a.files):
                    bad.append(p.name)
            n += 1
        assert n == expected
        result["historical_off"][suite] = {"n": n, "differing_trajectories": bad}
    return result


def main():
    off, on = [json.loads((ROOT / tag / "B_RESULTS.json").read_text()) for tag in ("off", "on")]
    for d in (off, on):
        assert len(d["coop_points"]) == 33 and len(d["c_points"]) == 12
    aa, bb = off["coop_points"], on["coop_points"]
    assert [r["unit"] for r in aa] == [r["unit"] for r in bb]
    rows = [{"world": a["world"], "off_all": a["wm_all"], "off_b": a["wm_b"],
             "on_all": b["wm_all"], "on_b": b["wm_b"], "sumo_all": a["sumo_all"], "sumo_b": a["sumo_b"]}
            for a, b in zip(aa, bb)]

    def contrast(rr, side):
        return rate(rr, f"{side}_all") - rate(rr, f"{side}_b")
    contrasts = {side: boot(rows, lambda rr: contrast(rr, side)) for side in ("off", "on", "sumo")}
    gap = boot(rows, lambda rr: abs(contrast(rr, "off") - contrast(rr, "sumo"))
               - abs(contrast(rr, "on") - contrast(rr, "sumo")))
    sign_ok = np.sign(contrasts["on"][0]) == np.sign(contrasts["sumo"][0])
    counts, qual, bitid = native_counts(aa, bb), qualification(), parity()
    adopted = bool((sign_ok or gap[1] > 0) and counts["counts_move_toward_sumo"] and not qual["pass_to_fail"])
    result = {"schema": "nativelane_decision/1", "adopt_scientific": adopted,
              "adopt": adopted and time.time() < datetime.fromisoformat("2026-10-26T09:00:00+08:00").timestamp(),
              "decision_time": time.time(), "primary_horizon_s": 60,
              "vacate_contrast_per_cav_h_ci95": contrasts, "absolute_gap_reduction_ci95": gap,
              "sign_agrees": bool(sign_ok), "native_counts": counts, "qualification": qual,
              "parity": bitid, "summary_off": {k: v for k, v in off.items() if k not in ("coop_points", "c_points")},
              "summary_on": {k: v for k, v in on.items() if k not in ("coop_points", "c_points")},
              "rule": "(60s contrast sign agrees OR paired absolute-gap reduction lower95>0) AND native count discrepancy shrinks AND no exploratory sign PASS->FAIL",
              "main14_action": "pending prearm verification" if adopted else "leave arm untouched"}
    save("DECISION.json", result)
    save("QUAL_NATIVE_LANE.json", qual)
    save("NATIVE_LANE_COUNTS.json", counts)
    lines = ["# NATIVELANE measured result", "", f"Scientific adoption: **{adopted}**.",
             f"Lane-vacate contrasts OFF / ON / SUMO (estimate, lower95, upper95): {contrasts}.",
             f"Absolute contrast-gap reduction: {gap}; sign agrees: {bool(sign_ok)}.",
             "Native ordinary lane-change totals OFF / ON / SUMO: " +
             str([counts[k]["ordinary_total"] for k in ("off", "on", "sumo")]) + ".",
             f"Exploratory PASS-to-FAIL units: {qual['pass_to_fail']}.",
             "Full gains, metering, world-cluster intervals, raw counts, qualification and parity: DECISION.json.",
             f"Main14 action: {result['main14_action']}."]
    (ROOT / "RESULT.md").write_text("\n\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in
                      ("adopt", "adopt_scientific", "vacate_contrast_per_cav_h_ci95", "absolute_gap_reduction_ci95", "main14_action")}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save("DECISION.json", {"adopt": False, "reason": "incomplete required evidence", "error": repr(exc), "time": time.time()})
        raise
