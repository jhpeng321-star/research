"""PL-04m bounded REAL SUMO check (the ONLY SUMO this task is allowed).

OPT-IN: skipped unless ``PLANNER_REAL_SUMO=1``. Budget: ONE master SUMO
run of 120 sim-s on its OWN free port, waited on ONLY through the
harness's own Popen child (never by image name or pattern); no lookahead
branches exist (the native-only master never reaches a decision point:
warmup == horizon). Output goes to ``E:/rd-wt/pl35/tmp_pl04m`` (never
``runs/`` or ``output/planner_v1``).

What it verifies, from the produced rows only:

(a) the master's roster is the base run's FULL designated ``cav_ids`` set;
    every LIVE designated CAV at every tick has speed_mode 27 -- including
    the CAVs inserted after t0 (listed with their first-seen times);
(b) replay equivalence with the recorded campaign native branch
    (``w8568m2600p05t360__native/run/physics_steps.jsonl``) over
    [360.5, 480]: per tick the same vehicle set, and the max abs
    position/speed/acceleration difference.

TRAIN world only; HOLDOUT/CALIBRATION outcomes are never read.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PLANNER_REAL_SUMO") != "1",
    reason="opt-in bounded real-SUMO check (set PLANNER_REAL_SUMO=1)",
)

PLANNER_ROOT = Path(__file__).resolve().parents[2]
BASE_RUN = Path(
    "E:/research-dense/runs/campaign_stage1_chunk3/w8568m2600p05t360__base"
)
NATIVE_BRANCH_PHYSICS = Path(
    "E:/research-dense/runs/campaign_stage1_chunk3/"
    "w8568m2600p05t360__native/run/physics_steps.jsonl"
)
OUT_ROOT = PLANNER_ROOT / "tmp_pl04m"
RUN_ID = "pl04m_real_check"
T0_S = 360.0
HORIZON_S = 120.0
WINDOW_START = 360.5
WINDOW_END = 480.0


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_pl04m_native_master_full_roster_and_replay():
    from traffic_shepherd.planner.closed_loop import (
        ClosedLoopConfig,
        ClosedLoopRunner,
        build_real_components,
        resolve_base_run,
    )

    resolved = resolve_base_run(BASE_RUN)
    assert resolved.t0_s == pytest.approx(T0_S)
    args = SimpleNamespace(
        predictor="perfect_sim",
        seed=resolved.seed,
        base_run=str(BASE_RUN),
        decision_period_s=30.0,
        lookahead_horizon_s=5.0,
        lookahead_commit_s=None,
    )
    components = build_real_components(args)
    roster = tuple(components.roster)
    evidence = dict(components.roster_evidence or {})
    print("\n[roster] source:", evidence.get("roster_source"))
    print("[roster] size:", len(roster), "first row t:", evidence.get("roster_first_row_time_s"))
    print("[roster] n_inserted_le_t0:", evidence.get("n_inserted_le_t0"))
    print("[roster] n_inserted_after_t0_or_never:", evidence.get("n_inserted_after_t0_or_never"))

    config = ClosedLoopConfig(
        out_root=OUT_ROOT,
        run_id=RUN_ID,
        world="w8568m2600p05t360",
        seed=resolved.seed,
        horizon_s=HORIZON_S,
        # native-only master: the first would-be decision point sits
        # OFF the 0.5 s physics grid and after it the horizon-end check
        # completes the run -- no decision point, no lookahead branch,
        # while the master still steps the FULL [360.5, 480] window
        warmup_s=HORIZON_S - 0.001,
        decision_period_s=30.0,
        predictor="perfect_sim",
        lookahead_horizon_s=5.0,
        lookahead_commit_s=None,
        k_samples=1,
        # the projection is conservative over the WOULD-BE decision point
        # (which never lands on the 0.5 s grid); the ACTUAL ledger spends
        # only the master's 120 sim-s (asserted below)
        max_sim_s=800.0,
        base_run=resolved.base_run,
        base_t0_s=resolved.t0_s,
        base_state_sha256=resolved.state_sha256,
        walls="off",
    )
    receipt = ClosedLoopRunner(config, components).run()
    assert receipt["exit_status"] == "complete", receipt["exit_status"]
    print("[receipt] exit_status:", receipt["exit_status"])
    print("[receipt] cav_roster size:", len(receipt["cav_roster"]))
    print("[receipt] roster_source:", receipt["roster_source"])
    print(
        "[receipt] n_inserted_le_t0 / after_or_never:",
        receipt["n_inserted_le_t0"],
        "/",
        receipt["n_inserted_after_t0_or_never"],
    )
    assert receipt["roster_source"] == "base_run_physics_cav_ids"
    assert receipt["cav_roster"] == list(roster)
    (master_launch,) = [r for r in receipt["sumo_launches"] if r["kind"] == "master"]
    print("[sumo] pid:", master_launch["pid"], "port:", master_launch["port"])
    print("[sumo] cmd:", " ".join(master_launch["cmd"]))
    print("[sumo] master sim-s:", receipt["sim_s"]["master_s"])

    run_dir = OUT_ROOT / RUN_ID
    rows = _read_jsonl(run_dir / "physics_steps.jsonl")
    assert rows and rows[0]["time_s"] == pytest.approx(WINDOW_START)

    # (a) every live designated CAV has speed_mode 27 from the tick AFTER
    # its first live row on. The FIRST live row itself races the control
    # tick that configures the mode (a CAV inserted on a physics tick is
    # recorded one 0.5 s step at the SUMO default 31; the RECORDED native
    # branch shows the identical one-tick pattern at ITS insertion tick,
    # e.g. mainline.323 at 454.0), so the first tick is reported, not a
    # violation.
    live_at_t0 = set(rows[0]["live_ids"])
    post_t0 = sorted(set(roster) - live_at_t0)
    first_seen = {}
    first_tick_31 = []
    mode_violations = []
    for row in rows:
        designated_live = set(row["live_ids"]) & set(row["cav_ids"])
        assert designated_live == set(row["speed_modes"]), (
            f"t={row['time_s']}: speed_modes keys != live designated set"
        )
        for cav, mode in row["speed_modes"].items():
            first = first_seen.setdefault(cav, row["time_s"])
            if mode != 27:
                if row["time_s"] == first:
                    first_tick_31.append((row["time_s"], cav, mode))
                else:
                    mode_violations.append((row["time_s"], cav, mode))
    assert not mode_violations, mode_violations[:5]
    late_first_seen = {cav: first_seen[cav] for cav in post_t0 if cav in first_seen}
    never_seen = sorted(cav for cav in post_t0 if cav not in first_seen)
    print(f"[a] ticks checked: {len(rows)}; mode violations after first tick: 0")
    print(f"[a] designated CAVs live at t0: {len(live_at_t0 & set(roster))}")
    print(f"[a] post-t0 CAVs: {len(post_t0)}")
    print(f"[a] post-t0 first-seen times: {len(late_first_seen)} entries")
    for cav in sorted(late_first_seen, key=lambda c: first_seen[c]):
        print(f"[a]   {cav}: first seen t={first_seen[cav]:g}")
    print(f"[a] post-t0 never seen live in window: {never_seen}")
    print(f"[a] one-tick 31 at the CAV's own first live row: {first_tick_31}")

    # (b) replay equivalence vs the recorded campaign native branch
    reference = _read_jsonl(NATIVE_BRANCH_PHYSICS)
    ref_by_t = {round(float(r["time_s"]), 6): r for r in reference}
    max_diff = {"position_m": 0.0, "speed_mps": 0.0, "accel_mps2": 0.0}
    first_divergent = None
    set_mismatches = []
    compared = 0
    for row in rows:
        t = round(float(row["time_s"]), 6)
        if t < WINDOW_START - 1e-9 or t > WINDOW_END + 1e-9:
            continue
        ref = ref_by_t.get(t)
        assert ref is not None, f"t={t}: no reference row"
        ours = row["vehicles"]
        theirs = ref["vehicles"]
        if set(ours) != set(theirs):
            set_mismatches.append((t, sorted(set(ours) ^ set(theirs))[:5]))
            continue
        compared += 1
        for vid in ours:
            ours_v, ref_v = ours[vid], theirs[vid]
            for keys, bucket in (
                (("lane_position_m", "s_m"), "position_m"),
                (("speed_mps", "v_mps"), "speed_mps"),
                (("acceleration_mps2", "accel_mps2"), "accel_mps2"),
            ):
                ours_key = next((k for k in keys if k in ours_v), None)
                ref_key = next((k for k in keys if k in ref_v), None)
                assert ours_key and ref_key, f"t={t} {vid}: missing {keys}"
                diff = abs(float(ours_v[ours_key]) - float(ref_v[ref_key]))
                if diff > max_diff[bucket]:
                    max_diff[bucket] = diff
                    if diff > 1e-6 and first_divergent is None:
                        first_divergent = (t, vid, ours_key, diff)
    print(f"[b] ticks compared over [360.5, 480]: {compared} "
          f"(of {len([t for t in rows if WINDOW_START - 1e-9 <= round(float(t['time_s']), 6) <= WINDOW_END + 1e-9])} window ticks; "
          f"{len(set_mismatches)} ticks had a vehicle-set mismatch)")
    print(f"[b] max abs diffs over matching-set ticks: {max_diff}")
    print(f"[b] first divergent tick/vehicle: {first_divergent}")
    print(f"[b] first vehicle-set mismatch: {set_mismatches[:3]}")
    exact = all(v <= 1e-6 for v in max_diff.values()) and not set_mismatches
    print(f"[b] replay verdict: {'EXACT/<=1e-6' if exact else 'NOT EXACT (reported; nothing tuned)'}")

    # the verdict is informational: equivalence is EXPECTED but a mismatch
    # is REPORTED (first divergent tick + vehicle + set mismatches), never
    # tuned away
    verdict = {
        "roster_size": len(roster),
        "post_t0_count": len(post_t0),
        "post_t0_first_seen": late_first_seen,
        "post_t0_never_seen": never_seen,
        "mode_violations_after_first_tick": 0,
        "first_tick_31": first_tick_31,
        "window_ticks": len(rows),
        "ticks_compared": compared,
        "vehicle_set_mismatches": len(set_mismatches),
        "max_abs_diffs": max_diff,
        "first_divergent": list(first_divergent) if first_divergent else None,
        "replay_within_1e-6": exact,
    }
    (run_dir / "PL04M_REAL_CHECK_VERDICT.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert receipt["sim_s"]["master_s"] <= 120.0 + 1e-9
    assert len(rows) == 240  # the full [360.5, 480] window at the 0.5 s grid
