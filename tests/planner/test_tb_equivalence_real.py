"""T-B bounded REAL SUMO equivalence checks (opt-in; the ONLY SUMO of T-B).

Skipped unless ``PLANNER_REAL_SUMO=1``. Budget (the owner's hard limits):
at most FOUR real closed-loop runs, each master horizon <= 120 s, plus
their lookahead branches, <= 1,500 simulated seconds in total. Outputs go
ONLY under ``<worktree>/tmp_tb/``. TRAIN base run only (read-only):
``campaign_stage1_chunk3/w8568m2600p05t360__base`` (t0 = 360). HOLDOUT /
CALIBRATION outcomes are never read. Complete run dirs are REUSED on a
re-invocation (no second SUMO launch); nothing is ever overwritten.

Runs (3 masters):

* R1 ``tb_r1_damping`` -- ``--fixed-policy damping`` restored at t0=360,
  horizon 120. Serves TWO checks:
  (4b) its physics rows vs the CONTINUOUS recorded base over [360.5, 480];
  (b)  at t=390 a snapshot is taken from the RUNNING master and ONE 30 s
       lookahead branch is restored from it under the SAME plan (damping on
       the whole roster); the branch's state at every 0.5 s tick must equal
       the master's own rows over the window, and its start state must
       equal the master's row at the snapshot time.
* R2 ``tb_r2_native`` -- ``--fixed-policy native``, horizon 120 (the
  reference for the null planner).
* R3 ``tb_r3_null`` -- the FULL planner machinery (perfect-sim lookahead
  branches) with a test-only infinite margin so it can never switch:
  ``--max-decision-units 1``, lookahead 30 s, horizon 60 (ONE decision at
  t=360). (a) its master rows must equal R2's over [360.5, 420]: the
  branch processes never perturb the master.
"""

import json
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PLANNER_REAL_SUMO") != "1",
    reason="opt-in bounded real-SUMO check (set PLANNER_REAL_SUMO=1)",
)

# the PI adapter's frozen modules (output/controller_verification_v2/v4)
# live only in the main tree; the worktree carries a partial ``output``
# namespace package -- append the main tree AFTER the worktree.
_MAIN = "E:/research-dense"
if _MAIN not in sys.path:
    sys.path.append(_MAIN)

PLANNER_ROOT = Path(__file__).resolve().parents[2]
BASE_RUN = Path("E:/research-dense/runs/campaign_stage1_chunk3/w8568m2600p05t360__base")
BASE_PHYSICS = BASE_RUN / "run" / "physics_steps.jsonl"
OUT_ROOT = PLANNER_ROOT / "tmp_tb"
WORLD = "w8568m2600p05t360"
T0_S = 360.0
SNAP_T = 390.0
BRANCH_S = 30.0
TOL = 1e-6
FIELDS = (
    ("lane_position_m", "position_m"),
    ("speed_mps", "speed_mps"),
    ("acceleration_mps2", "accel_mps2"),
)
VERDICTS = OUT_ROOT / "TB_VERDICTS.json"


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _record_verdict(key, payload):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    data = json.loads(VERDICTS.read_text(encoding="utf-8")) if VERDICTS.is_file() else {}
    data[key] = payload
    VERDICTS.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _argv(run_id, horizon_s, extra):
    from traffic_shepherd.planner.closed_loop import resolve_base_run

    seed = resolve_base_run(BASE_RUN).seed
    return [
        "--components", "real",
        "--world", WORLD,
        "--split", "train",
        "--seed", str(seed),
        "--base-run", str(BASE_RUN),
        "--horizon-s", str(horizon_s),
        "--warmup-s", "0",
        "--walls", "off",
        "--out", str(OUT_ROOT),
        "--predictor", "perfect_sim",
        *extra,
        "--run-id", run_id,
    ]


def _complete(run_id):
    receipt_path = OUT_ROOT / run_id / "RUN_RECEIPT.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt.get("exit_status") == "complete", (
            f"{receipt_path.parent} exists without a complete receipt; choose a "
            "fresh run id (write-once run dirs)"
        )
        return receipt
    return None


def _compare(rows_a, rows_b, t_lo, t_hi):
    """Per-tick vehicle-set equality + max |dpos|,|dv|,|da| over [t_lo, t_hi]."""
    by_b = {round(float(r["time_s"]), 6): r for r in rows_b}
    max_diff = {name: 0.0 for _, name in FIELDS}
    first = None
    set_mismatches = []
    compared = 0
    for row in rows_a:
        t = round(float(row["time_s"]), 6)
        if t < t_lo - 1e-9 or t > t_hi + 1e-9:
            continue
        other = by_b.get(t)
        assert other is not None, f"t={t}: no reference row"
        if set(row["vehicles"]) != set(other["vehicles"]):
            set_mismatches.append(
                (t, sorted(set(row["vehicles"]) ^ set(other["vehicles"]))[:6])
            )
            continue
        compared += 1
        for vid in sorted(row["vehicles"]):
            for key, name in FIELDS:
                diff = abs(float(row["vehicles"][vid][key]) - float(other["vehicles"][vid][key]))
                max_diff[name] = max(max_diff[name], diff)
                if diff > TOL and first is None:
                    first = (t, vid, key, diff)
    exact = compared > 0 and not set_mismatches and all(v <= TOL for v in max_diff.values())
    return {
        "ticks_compared": compared,
        "vehicle_set_mismatches": len(set_mismatches),
        "first_set_mismatch": set_mismatches[0] if set_mismatches else None,
        "max_abs_diffs": max_diff,
        "first_divergent": list(first) if first else None,
        "exact_within_1e-6": exact,
    }


def _read_state(connection):
    vehicles = {}
    for raw in connection.vehicle.getIDList():
        vid = str(raw)
        vehicles[vid] = {
            "lane_position_m": float(connection.vehicle.getLanePosition(vid)),
            "speed_mps": float(connection.vehicle.getSpeed(vid)),
            "acceleration_mps2": float(connection.vehicle.getAcceleration(vid)),
        }
    return {"vehicles": vehicles}


# ---------------------------------------------------------------- R1 (4b + b)
_R1 = {}


def _run_r1():
    """R1: fixed damping master (4b) with ONE mid-run branch (b)."""
    if "receipt" in _R1:
        return _R1
    from traffic_shepherd.planner import closed_loop as cl

    run_id = "tb_r1_damping"
    receipt = _complete(run_id)
    branch_file = OUT_ROOT / run_id / "TB_BRANCH_ROWS.json"
    if receipt is not None:
        assert branch_file.is_file(), "R1 complete but its branch rows are missing"
        _R1.update(receipt=receipt, branch=json.loads(branch_file.read_text(encoding="utf-8")))
        return _R1

    captured = {"spy_rows": {}, "done": False}
    orig_build = cl.build_real_components
    orig_step = cl.RealMasterHarness.step_physics

    def build(args):
        components = orig_build(args)
        captured["roster"] = tuple(components.roster)
        orig_wm = components.make_world_model

        def make_wm(context):
            model = orig_wm(context)
            captured["model"] = model
            runner = model.branch_runner
            orig_factory = runner._tick_hook_factory

            def spy_factory(connection, session, plan):
                inner = orig_factory(connection=connection, session=session, plan=plan)
                base_cls = type(inner)

                class Spy(base_cls):
                    def __call__(self, session, now):
                        # the state a master row at ``now`` reports: read
                        # BEFORE this control tick (tick precedes the step)
                        captured["spy_rows"][round(float(now), 6)] = _read_state(
                            self.connection
                        )
                        return base_cls.__call__(self, session, now)

                inner.__class__ = Spy
                return inner

            runner._tick_hook_factory = spy_factory
            return model

        components.make_world_model = make_wm
        return components

    def step(self):
        row = orig_step(self)
        if not captured["done"] and abs(float(row["t_s"]) - SNAP_T) < 1e-9:
            captured["done"] = True
            model = captured["model"]
            snap = model.snapshotter.save(t0_s=SNAP_T, controller_state={}, kept_wall_ids=[])
            plan = cl.fixed_policy_for_roster(captured["roster"], "damping")
            record = model._run_branch_in_fresh_process(
                plan,
                process_label="tb_b:damping",
                manifest=snap,
                draw_id="tb_b",
                departures=(),
                horizon_s=BRANCH_S,
                safety_window_s=BRANCH_S,
                controller_state={},
                kept_wall_ids=[],
                stop_on_window_collision=False,
                commit_s=None,
                continuation=None,
            )
            captured["branch_record"] = record
            captured["carried_restores"] = model.branch_runner.carried_state_restores
            captured["snapshot_harness_state_keys"] = sorted(
                snap["manifest"]["harness_state"]
            )
            captured["branch_launch"] = list(model.branch_runner.process_factory.launched)
        return row

    cl.build_real_components = build
    cl.RealMasterHarness.step_physics = step
    try:
        code = cl.main(_argv(run_id, 120.0, ["--fixed-policy", "damping", "--max-sim-s", "700"]))
    finally:
        cl.build_real_components = orig_build
        cl.RealMasterHarness.step_physics = orig_step
    assert code == 0, f"closed_loop main returned {code}"
    assert captured["done"], "the mid-run snapshot/branch hook never fired"
    receipt = json.loads((OUT_ROOT / run_id / "RUN_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["exit_status"] == "complete"
    record = captured["branch_record"]
    branch = {
        "spy_rows": {str(t): v for t, v in captured["spy_rows"].items()},
        "record_steps": [
            {
                "time_s": s.time_s,
                "vehicles": {
                    vid: {
                        "lane_position_m": vs.s_m,
                        "speed_mps": vs.speed_mps,
                        "acceleration_mps2": vs.accel_mps2,
                    }
                    for vid, vs in s.vehicles.items()
                },
            }
            for s in record.steps
        ],
        # branch p4-evaluator row cadence: EVERY 0.5 s physics step
        "physics_times": [round(float(x.time_s), 6) for x in record.physics_steps],
        "carried_state_restores": captured["carried_restores"],
        "snapshot_harness_state_keys": captured["snapshot_harness_state_keys"],
        "branch_launch": captured["branch_launch"],
        "collision_times": list(record.collision_times),
    }
    branch_file.write_text(json.dumps(branch, sort_keys=True) + "\n", encoding="utf-8")
    _R1.update(receipt=receipt, branch=branch)
    return _R1


def test_tb_4b_fixed_policy_damping_vs_recorded_base():
    r1 = _run_r1()
    receipt = r1["receipt"]
    run_dir = OUT_ROOT / "tb_r1_damping"
    ours = _read_jsonl(run_dir / "physics_steps.jsonl")
    result = _compare(ours, _read_jsonl(BASE_PHYSICS), 360.5, 480.0)
    master = next(r for r in receipt["sumo_launches"] if r["kind"] == "master")
    result.update(
        check="4b_fixed_policy_damping_vs_recorded_base",
        window=[360.5, 480.0],
        sumo_pid=master["pid"],
        sumo_port=master["port"],
        master_sim_s=receipt["sim_s"]["master_s"],
        harness_state_restored=receipt["harness_state_restored"],
        first_control_tick_s=receipt["first_control_tick_s"],
        restore_source_manifest_sha256=receipt["restore_source_manifest_sha256"],
    )
    print("\n[4b]", json.dumps(result, sort_keys=True))
    _record_verdict("4b", result)
    assert receipt["harness_state_restored"] is True
    assert receipt["first_control_tick_s"] == pytest.approx(T0_S)
    assert result["exact_within_1e-6"], result


def test_tb_b_master_continuation_equals_branch_same_plan():
    r1 = _run_r1()
    branch = r1["branch"]
    master_rows = _read_jsonl(OUT_ROOT / "tb_r1_damping" / "physics_steps.jsonl")
    spy = [
        {"time_s": float(t), "vehicles": v["vehicles"]}
        for t, v in sorted(branch["spy_rows"].items(), key=lambda kv: float(kv[0]))
    ]
    # start state (t = 390.0) and every 0.5 s tick of the window
    result = _compare(spy, master_rows, SNAP_T, SNAP_T + BRANCH_S - 0.5)
    # and the 1 s telemetry the record itself carries, incl. the final tick
    steps = _compare(branch["record_steps"], master_rows, SNAP_T + 1.0, SNAP_T + BRANCH_S)
    result.update(
        check="b_master_continuation_equals_branch_same_plan",
        window=[SNAP_T, SNAP_T + BRANCH_S],
        record_steps_1s=steps,
        carried_state_restores=branch["carried_state_restores"],
        branch_launch=[
            {k: v for k, v in rec.items() if k != "cmd"} for rec in branch["branch_launch"]
        ],
        collisions_in_branch=branch["collision_times"],
    )
    print("\n[b]", json.dumps(result, sort_keys=True))
    _record_verdict("b", result)
    assert branch["carried_state_restores"] == 1
    if "physics_times" in branch:  # absent only in run dirs cached before the cadence fix
        master_times = [
            round(float(r["time_s"]), 6)
            for r in master_rows
            if SNAP_T + 1e-9 < float(r["time_s"]) <= SNAP_T + BRANCH_S + 1e-9
        ]
        assert branch["physics_times"] == master_times, "branch row cadence != master's"
    assert result["exact_within_1e-6"], result
    assert steps["exact_within_1e-6"], steps


# ---------------------------------------------------------- R2 + R3 (a)
def _run_cli(run_id, horizon_s, extra):
    from traffic_shepherd.planner import closed_loop as cl

    receipt = _complete(run_id)
    if receipt is not None:
        return receipt
    code = cl.main(_argv(run_id, horizon_s, extra))
    assert code == 0, f"closed_loop main returned {code}"
    return json.loads((OUT_ROOT / run_id / "RUN_RECEIPT.json").read_text(encoding="utf-8"))


def test_tb_a_null_planner_equals_native_reference():
    from traffic_shepherd.config import load_config  # noqa: F401 (import parity)
    from traffic_shepherd.planner import closed_loop as cl
    from traffic_shepherd.planner.config import PlannerConfig

    native = _run_cli("tb_r2_native", 120.0, ["--fixed-policy", "native", "--max-sim-s", "700"])

    # test-only override: an effectively infinite switch margin => the
    # planner evaluates every candidate through the real lookahead but can
    # NEVER switch (native everywhere)
    orig = cl._planner_config_from_actions

    def never_switch(actions):
        return PlannerConfig(actions=("native", "damping"), margin_floor_s=1.0e12)

    cl._planner_config_from_actions = never_switch
    try:
        null = _run_cli(
            "tb_r3_null",
            60.0,
            [
                "--decision-period-s", "60",
                "--lookahead-horizon-s", "30",
                "--lookahead-commit-s", "30",
                "--k-samples", "1",
                "--max-decision-units", "1",
                "--max-sim-s", "500",
            ],
        )
    finally:
        cl._planner_config_from_actions = orig

    ours = _read_jsonl(OUT_ROOT / "tb_r3_null" / "physics_steps.jsonl")
    ref = _read_jsonl(OUT_ROOT / "tb_r2_native" / "physics_steps.jsonl")
    result = _compare(ours, ref, 360.5, 420.0)
    decisions = _read_jsonl(OUT_ROOT / "tb_r3_null" / "decisions.jsonl")
    branches = [r for r in null["sumo_launches"] if r["kind"] == "branch"]
    master = next(r for r in null["sumo_launches"] if r["kind"] == "master")
    switched = [d for d in decisions if d.get("switched")]
    non_native = [d for d in decisions if any(a != "native" for a in d["actions"].values())]
    result.update(
        check="a_null_planner_equals_native_reference",
        window=[360.5, 420.0],
        sumo_master_pid=master["pid"],
        sumo_master_port=master["port"],
        n_branch_processes=len(branches),
        branch_pids_ports=[[b["pid"], b["port"]] for b in branches],
        decisions=len(decisions),
        decisions_switched=len(switched),
        decisions_with_non_native_actions=len(non_native),
        master_sim_s=null["sim_s"]["master_s"],
        lookahead_sim_s=null["sim_s"]["lookahead_s"],
        ref_master_sim_s=native["sim_s"]["master_s"],
    )
    print("\n[a]", json.dumps(result, sort_keys=True))
    _record_verdict("a", result)
    assert null["run_purpose"] == "planner"
    assert decisions and branches, "the null planner must have run lookahead branches"
    assert not switched and not non_native
    assert result["exact_within_1e-6"], result
