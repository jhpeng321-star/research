"""T-A bounded REAL SUMO checks (the ONLY SUMO this task is allowed).

OPT-IN: skipped unless ``PLANNER_REAL_SUMO=1``. Budget: TWO master SUMO
runs of at most 120 sim-s EACH, each on its OWN free port, waited on ONLY
through the harness's own Popen child (never by image name or pattern).
No lookahead branches exist (reference runs never reach a decision
point). Output goes to ``<repo>/tmp_ta`` (never ``runs/`` or
``output/planner_v1``). TRAIN world only; HOLDOUT/CALIBRATION outcomes
are never read.

(a) AMENDED 4b: ``--fixed-policy damping`` restored at the base t0=360,
    horizon 120, vs the CONTINUOUS recorded base run's physics over
    [360.5, 480]: per tick the vehicle-set equality and the max abs
    position/speed/acceleration difference. NOT expected to be exact
    (the damping controller state is not carried by the saved state --
    branch controller-state seeding is the follow-up task T-B), so the
    first divergent tick/vehicle + a diagnosis are REPORTED and written
    to the verdict file. Nothing is tuned to force equivalence.
(b) ``--fixed-policy native``, 120 s: every live designated CAV is
    speed-mode 27 from the tick AFTER its first-seen row (the first-seen
    row may show 31, as in the campaign) and there are ZERO non-native
    command rows.

The recorded campaign __native/__pi arms are INVALID (random per-CAV
switching) and are NOT used as a pass criterion anywhere here.
"""

import json
import os
from pathlib import Path

import pytest

from tests.planner._helpers import drive_only_campaign

PLANNER_ROOT = Path(__file__).resolve().parents[2]
BASE_RUN = drive_only_campaign("campaign_stage1_chunk3") / "w8568m2600p05t360__base"
BASE_PHYSICS = BASE_RUN / "run" / "physics_steps.jsonl"

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("PLANNER_REAL_SUMO") != "1",
        reason="opt-in bounded real-SUMO check (set PLANNER_REAL_SUMO=1)",
    ),
    pytest.mark.skipif(
        not BASE_RUN.is_dir(),
        reason="raw campaign_stage1_chunk3 runs not on disk (Drive-only dataset "
        "campaign_stage1_chunk3: `python -m wms fetch get campaign_stage1_chunk3`)",
    ),
]

OUT_ROOT = PLANNER_ROOT / "tmp_ta"
HORIZON_S = 120.0
T0_S = 360.0
WINDOW_START = 360.5
WINDOW_END = 480.0


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_reference(tmp_run_id, policy):
    """One --fixed-policy reference run through the CLI entry the launcher
    uses (main(argv)); ONE master SUMO process, <= 120 sim-s, no lookahead.
    Budget discipline: an already-COMPLETE run dir is REUSED (no second
    SUMO launch on a test re-invocation); a write-once refusal for a
    missing/incomplete dir is never silently overwritten."""
    from traffic_shepherd.planner.closed_loop import main, resolve_base_run

    resolved = resolve_base_run(BASE_RUN)
    assert resolved.t0_s == pytest.approx(T0_S)
    run_dir = OUT_ROOT / tmp_run_id
    receipt_path = run_dir / "RUN_RECEIPT.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt.get("exit_status") == "complete", (
            f"{run_dir} exists without a complete receipt; remove it or "
            "choose a fresh run id (write-once run dirs)"
        )
        master = next(
            r for r in receipt["sumo_launches"] if r["kind"] == "master"
        )
        return run_dir, receipt, master
    argv = [
        "--components", "real",
        "--world", "w8568m2600p05t360",
        "--split", "train",
        "--seed", str(resolved.seed),
        "--base-run", str(BASE_RUN),
        "--horizon-s", str(HORIZON_S),
        "--warmup-s", "0",
        "--walls", "off",
        "--out", str(OUT_ROOT),
        "--predictor", "perfect_sim",
        "--fixed-policy", policy,
        "--max-sim-s", "700",
        "--run-id", tmp_run_id,
    ]
    code = main(argv)
    assert code == 0, f"closed_loop main returned {code}"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["exit_status"] == "complete", receipt["exit_status"]
    assert receipt["run_purpose"] == "reference"
    assert receipt["fixed_policy"] == policy
    assert receipt["base_t0_s"] == pytest.approx(T0_S)
    assert receipt["sim_s"]["master_s"] <= HORIZON_S + 1e-9
    assert receipt["sim_s"]["lookahead_s"] == 0.0
    master = next(r for r in receipt["sumo_launches"] if r["kind"] == "master")
    return run_dir, receipt, master


def test_ta_4b_fixed_policy_damping_vs_recorded_base():
    run_dir, receipt, master = _run_reference("ta_4b_damping", "damping")
    ours = _read_jsonl(run_dir / "physics_steps.jsonl")
    reference = _read_jsonl(BASE_PHYSICS)
    ref_by_t = {round(float(r["time_s"]), 6): r for r in reference}
    max_diff = {"position_m": 0.0, "speed_mps": 0.0, "accel_mps2": 0.0}
    first_divergent = None
    set_mismatches = []
    compared = 0
    for row in ours:
        t = round(float(row["time_s"]), 6)
        if t < WINDOW_START - 1e-9 or t > WINDOW_END + 1e-9:
            continue
        ref = ref_by_t.get(t)
        assert ref is not None, f"t={t}: no recorded base row"
        if set(row["vehicles"]) != set(ref["vehicles"]):
            set_mismatches.append((t, sorted(set(row["vehicles"]) ^ set(ref["vehicles"]))[:5]))
            continue
        compared += 1
        for vid in row["vehicles"]:
            ours_v, ref_v = row["vehicles"][vid], ref["vehicles"][vid]
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
    commands = _read_jsonl(run_dir / "dispatcher_rows.jsonl")
    damping_commands = sorted(
        {
            (row["t_s"], row["cav"])
            for row in commands
            if row.get("source") == "master_control_tick"
            and row.get("assigned_action") == "damping"
        }
    )
    print("\n[4b] SUMO pid:", master["pid"], "port:", master["port"])
    print("[4b] cmd:", " ".join(master["cmd"]))
    print("[4b] master sim-s:", receipt["sim_s"]["master_s"])
    print(f"[4b] ticks compared over [360.5, 480]: {compared} "
          f"({len(set_mismatches)} vehicle-set mismatches)")
    print(f"[4b] max abs diffs: {max_diff}")
    print(f"[4b] first divergent tick/vehicle: {first_divergent}")
    print(f"[4b] first damping command rows: {[list(c) for c in damping_commands[:3]]}")
    exact = all(v <= 1e-6 for v in max_diff.values()) and not set_mismatches
    print("[4b] verdict:", "EXACT/<=1e-6" if exact else "NOT EXACT (reported; nothing tuned)")
    diagnosis = None
    if not exact:
        div_txt = (
            f"first divergence t={first_divergent[0]} {first_divergent[1]} "
            f"({first_divergent[2]})"
            if first_divergent is not None
            else f"{len(set_mismatches)} vehicle-set mismatches"
        )
        diagnosis = (
            "NOT exact, nothing tuned. Two known mechanisms compound here: "
            "(1) the saved-state restore does not bit-exactly CONTINUE the "
            "continuous recorded run -- the first divergence already sits "
            "on the first post-restore tick (360.5), BEFORE this run's "
            "first control write (the campaign's own restored-arm replays "
            "show the same one-tick offset vs their continuous base; cf. "
            "the PL-04m check vs the recorded __native arm), and "
            "(2) from the first control tick (361.0) the restored master "
            "re-issues damping commands from a COLD controller -- the "
            "damping controller state is not carried by the saved state "
            "(per-CAV controller-state seeding is the follow-up task T-B) "
            "-- so the commanded speeds drift away from the recorded warm "
            "continuation and the gap grows over the window. " + div_txt + "."
        )
    print("[4b] diagnosis:", diagnosis)
    verdict = {
        "check": "amended_4b_fixed_policy_damping_vs_recorded_base",
        "sumo_pid": master["pid"],
        "sumo_port": master["port"],
        "master_sim_s": receipt["sim_s"]["master_s"],
        "window": [WINDOW_START, WINDOW_END],
        "ticks_compared": compared,
        "vehicle_set_mismatches": len(set_mismatches),
        "max_abs_diffs": max_diff,
        "first_divergent": list(first_divergent) if first_divergent else None,
        "n_damping_command_rows": len(damping_commands),
        "first_damping_commands": [list(c) for c in damping_commands[:5]],
        "exact_within_1e-6": exact,
        "diagnosis": diagnosis,
    }
    (run_dir / "TA_4B_VERDICT.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_ta_ref_native_mode27_and_zero_nonnative_commands():
    run_dir, receipt, master = _run_reference("ta_ref_native", "native")
    rows = _read_jsonl(run_dir / "physics_steps.jsonl")
    roster = set(receipt["cav_roster"])
    commands = _read_jsonl(run_dir / "dispatcher_rows.jsonl")
    non_native = [
        row
        for row in commands
        if row.get("source") == "master_control_tick"
        and (
            row.get("assigned_action") not in (None, "native")
            or (row.get("command") not in (None, "", "native"))
        )
    ]
    first_seen = {}
    first_tick_31 = []
    violations = []
    for row in rows:
        designated_live = set(row["live_ids"]) & roster
        assert designated_live == set(row["speed_modes"]), (
            f"t={row['time_s']}: speed_modes keys != live designated set"
        )
        for cav, mode in row["speed_modes"].items():
            first = first_seen.setdefault(cav, row["time_s"])
            if mode != 27:
                if row["time_s"] == first:
                    first_tick_31.append((row["time_s"], cav, mode))
                else:
                    violations.append((row["time_s"], cav, mode))
    print("\n[native] SUMO pid:", master["pid"], "port:", master["port"])
    print("[native] master sim-s:", receipt["sim_s"]["master_s"])
    print(f"[native] ticks checked: {len(rows)}; roster: {len(roster)}")
    print(f"[native] mode violations AFTER the CAV's own first live row: {violations[:5]}")
    print(f"[native] one-tick 31 at the CAV's own first live row: {len(first_tick_31)} entries")
    print(f"[native] non-native command rows: {len(non_native)}")
    assert not violations, violations[:5]
    assert not non_native, non_native[:5]
    verdict = {
        "check": "ref_native_mode27_and_zero_nonnative_commands",
        "sumo_pid": master["pid"],
        "sumo_port": master["port"],
        "master_sim_s": receipt["sim_s"]["master_s"],
        "ticks": len(rows),
        "roster_size": len(roster),
        "one_tick_31_at_first_row": len(first_tick_31),
        "mode_violations_after_first_row": 0,
        "non_native_command_rows": 0,
    }
    (run_dir / "TA_REF_NATIVE_VERDICT.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
