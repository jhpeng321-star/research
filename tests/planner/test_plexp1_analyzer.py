"""PL-EXP1b offline analyzer tests (zero SUMO; tmp_path only).

(a) REAL-INTERFACE: a planner run dir is produced by the REAL closed-loop
runner (``ClosedLoopRunner.run`` over the REAL ``build_real_components``
wiring; only ``traci`` and the SUMO binary are stubbed -- the fixtures/
helpers are IMPORTED from tests/planner/test_pl04i_e2e_real_run.py), then
the analyzer runs over the registered PL-EXP1b layout (addendum 1):

- runs root: ``<world>__planner_oracle`` + the TRUE comparators
  ``<world>__ref_native`` / ``<world>__ref_pi`` (synthetic REFERENCE
  masters: receipt ``run_purpose='reference'``, ``fixed_policy``,
  closed_loop-format physics rows);
- campaign chunk: ``<world>__base`` (the DAMPING reference) PLUS decoy
  ``<world>__native`` / ``<world>__pi`` recorded arms -- fully populated,
  VALID-looking, and (path-spy test) NEVER opened, because the real
  campaign arms ran random per-CAV switching and are invalid.

TRUE-GAP TTC (gap #18): every arm's scenario requests.rou.xml declares
vTypes; ``normal_car`` has no length attribute (SUMO passenger default
5.0 m), ``hdv_car`` declares 4.6 m. The tiny world's restored-state
vehicles (mainline.6/7, hdv.9) are registered in the planner run's copied
scenario (documented in-test fixture fix) so the fail-closed length map
resolves.

Sign convention (AGENTS.md): delta = planner - reference; NEGATIVE =
improvement. Pooling gate: a world enters pooled aggregates only if every
run used is delay_reliable (per-arm n + excluded worlds reported).
"""

import builtins
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from tests.planner import test_pl04i_e2e_real_run as e2e_mod
from traffic_shepherd.planner.closed_loop import (
    ClosedLoopConfig,
    ClosedLoopRunner,
    build_real_components,
    initial_plan_for_roster,
    resolve_base_run,
)

PLANNER_ROOT = Path(__file__).resolve().parents[2]
ANALYZER_PATH = PLANNER_ROOT / "output" / "planner_v1" / "plexp1_analysis" / "analyze_plexp1.py"

WORLD = "w_pl04i"
SEED = e2e_mod.SEED  # 8468 (the tiny world's seed)
T0 = e2e_mod.BASE_T0_S  # 240.0
HORIZON = 20.0
LANE = "1191554636_0"  # WM-30 chain lane (main_right), edge 1191554636
ROAD = "1191554636"
VEHICLES = ("mainline.6", "mainline.7", "hdv.9")
START_POS = {"mainline.6": 200.0, "mainline.7": 260.0, "hdv.9": 330.0}

#: every fixture scenario declares vTypes: normal_car WITHOUT a length
#: attribute (SUMO passenger default 5.0 m applies) and hdv_car WITH 4.6 m
ROU_XML = (
    "<routes>"
    '<vType id="normal_car" carFollowModel="IDM" speedDev="0"/>'
    '<vType id="hdv_car" carFollowModel="IDM" length="4.6"/>'
    '<route id="mainline_route" edges="e0 e1"/>'
    + "".join(
        f'<vehicle id="{v}" type="{"hdv_car" if v == "hdv.9" else "normal_car"}"'
        ' route="mainline_route" depart="1.0"/>'
        for v in VEHICLES
    )
    + "</routes>"
)


def _load_analyzer():
    name = "plexp1_analyzer_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ANALYZER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


az = _load_analyzer()


def _p4_available() -> bool:
    try:
        az.p4_delay_provider(az.P4_SEARCH_PATHS)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _p4_available(), reason="WD-01 p4 delay library absent")


@pytest.fixture(autouse=True)
def short_window(monkeypatch):
    """The synthetic worlds are 20 s long, not the registered 600 s."""
    monkeypatch.setattr(az, "WINDOW_S", HORIZON)


# ---------------------------------------------------------- real run dirs
def _produce_real_run(
    root: Path, *, initial_plan: str = None, lane: str = None, hdv_speed: float = 22.0
) -> Path:
    """One REAL closed-loop run (real writers; stubbed traci + SUMO binary),
    written to ``root/plexp1_runs/<world>__planner_oracle``.

    In-test fixture fix (documented): the run's COPIED scenario is written
    with the roster vehicles declared (the tiny world's restored state
    predates the scenario <vehicle> list), so the TRUE-GAP length map
    resolves (normal_car -> SUMO default 5.0 m, hdv_car -> 4.6 m)."""
    root.mkdir(parents=True, exist_ok=True)
    base = e2e_mod.make_base_run_pl04i(root / "fixture", hdv_speed=hdv_speed)
    resolved = resolve_base_run(base)
    mp = pytest.MonkeyPatch()
    try:
        mp.setenv("SUMO_BINARY", sys.executable)
        mp.delenv("SUMO_HOME", raising=False)
        mp.setitem(sys.modules, "traci", e2e_mod._TinyTraci(lane=lane))
        components = build_real_components(e2e_mod._real_args(base))
        if initial_plan:
            components.initial_plan = initial_plan_for_roster(components.roster, initial_plan)
        config = ClosedLoopConfig(
            out_root=root / "plexp1_runs",
            run_id=f"{WORLD}__planner_oracle",
            world=WORLD,
            seed=SEED,
            horizon_s=HORIZON,
            warmup_s=0.0,
            decision_period_s=10.0,
            predictor="perfect_sim",
            lookahead_horizon_s=5.0,
            lookahead_commit_s=None,
            k_samples=1,
            max_sim_s=500.0,
            base_run=resolved.base_run,
            base_t0_s=resolved.t0_s,
            base_state_sha256=resolved.state_sha256,
            max_decision_units=2,
            walls="off",
            initial_plan_action=initial_plan,
        )
        ClosedLoopRunner(config, components).run()
    finally:
        mp.undo()
    run_dir = root / "plexp1_runs" / f"{WORLD}__planner_oracle"
    scenario = run_dir / "scenario"
    scenario.mkdir(parents=True, exist_ok=True)
    (scenario / "requests.rou.xml").write_text(ROU_XML, encoding="utf-8")
    return run_dir


@pytest.fixture(scope="module")
def real_planner_run(tmp_path_factory):
    return _produce_real_run(tmp_path_factory.mktemp("real_planner"))


@pytest.fixture(scope="module")
def real_chain_run(tmp_path_factory):
    """PL-04k: a REAL-runner run whose fake world drives on the WM-30 chain
    lane (so the rows' road_id/lane_id map onto the p4 chain) with a slow
    HDV ahead of the CAVs (a closing pair -> a finite min TTC)."""
    return _produce_real_run(
        tmp_path_factory.mktemp("real_chain"), lane=LANE, hdv_speed=18.0
    )


@pytest.fixture(scope="module")
def real_proof_run(tmp_path_factory):
    return _produce_real_run(tmp_path_factory.mktemp("real_proof"), initial_plan="damping")


@pytest.fixture(scope="module")
def shared_lib():
    return az.chunk1_layout(), az.p4_delay_provider(az.P4_SEARCH_PATHS)


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ------------------------------------------- synthetic PL-EXP1b run dirs
def _campaign_rows(speed_by_vehicle, t_first, t_last, *, t0=T0):
    """Campaign-format physics rows (time_s / road_id / lane_id /
    lane_position_m / speed_mps / acceleration_mps2 + colliding_ids)."""
    rows = []
    t = t_first
    while t <= t_last + 1e-9:
        rows.append(
            {
                "time_s": round(t, 6),
                "colliding_ids": [],
                "arrived_ids": [],
                "vehicles": {
                    vid: {
                        "road_id": ROAD,
                        "lane_id": LANE,
                        "lane_position_m": START_POS[vid] + speed_by_vehicle[vid] * (t - t0),
                        "speed_mps": speed_by_vehicle[vid],
                        "acceleration_mps2": 0.0,
                    }
                    for vid in VEHICLES
                },
            }
        )
        t = round(t + 0.5, 6)
    return rows


def _integrated_rows(phases, t_first, t_last, *, t0=T0, colliding_at=(), start_pos=None):
    """Closed-loop-format physics rows with piecewise-constant speeds:
    ``phases`` maps vehicle id -> [(segment_end_offset_s, speed_mps), ...]
    (the last segment runs to the horizon). Positions are integrated, so
    closing pairs yield hand-checkable TRUE-gap TTC values."""
    starts = start_pos or START_POS

    def pos_at(vid, offset):
        pos = starts[vid]
        done = 0.0
        for until, spd in phases[vid]:
            step = min(until, offset) - done
            if step > 0:
                pos += spd * step
                done += step
            if done >= offset - 1e-12:
                break
        return pos

    def spd_at(vid, offset):
        for until, spd in phases[vid]:
            if offset <= until + 1e-9:
                return spd
        return phases[vid][-1][1]

    rows = []
    t = t_first
    while t <= t_last + 1e-9:
        offset = t - t0
        rows.append(
            {
                "t_s": round(t, 6),
                "colliding_ids": list(colliding_at),
                "arrived_ids": [],
                "cav_ids": sorted(phases),
                "cav_actions": {},
                "vehicles": {
                    vid: {
                        "s_m": round(pos_at(vid, offset), 9),
                        "v_mps": spd_at(vid, offset),
                        "accel_mps2": 0.0,
                        "road_id": ROAD,
                        "lane_id": LANE,
                    }
                    for vid in phases
                },
            }
        )
        t = round(t + 0.5, 6)
    return rows


def make_ref_run(runs_root: Path, world: str, arm: str, *, speed: float,
                 coverage: float = HORIZON, receipt_overrides: dict = None) -> Path:
    """A PL-EXP1b REFERENCE master ``<world>__ref_<arm>`` (closed-loop
    format rows; receipt run_purpose='reference' + fixed_policy)."""
    run = Path(runs_root) / f"{world}__ref_{arm}"
    run.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "closed_loop_receipt/1",
        "exit_status": "complete",
        "run_purpose": "reference",
        "fixed_policy": arm,
        "world": world,
        "seed": SEED,
        "split": "TRAIN",
    }
    receipt.update(receipt_overrides or {})
    (run / "RUN_RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")
    _write_jsonl(
        run / "physics_steps.jsonl",
        _integrated_rows({v: [(coverage, speed)] for v in VEHICLES}, T0 + 0.5, T0 + coverage),
    )
    scenario = run / "scenario"
    scenario.mkdir(parents=True, exist_ok=True)
    (scenario / "requests.rou.xml").write_text(ROU_XML, encoding="utf-8")
    return run


def make_chunk_world(chunk_dir: Path, world: str, *, damping_speed: float = 22.0,
                     coverage: float = HORIZON, decoys: tuple = ()) -> Path:
    """``<world>__base`` (the DAMPING reference) in the campaign chunk, plus
    optional fully-populated decoy ``__native`` / ``__pi`` recorded arms
    (INVALID in the real campaign: random per-CAV switching; the analyzer
    must never read them)."""
    run = chunk_dir / f"{world}__base"
    run.mkdir(parents=True, exist_ok=True)
    (run / "RUN_RECEIPT.json").write_text(
        json.dumps({"schema": "wm44_branch_receipt/1", "exit_status": "complete"}),
        encoding="utf-8",
    )
    _write_jsonl(
        run / "run" / "physics_steps.jsonl",
        _campaign_rows({v: damping_speed for v in VEHICLES}, T0 - 1.0, T0 + coverage),
    )
    scenario = run / "run" / "scenario"
    scenario.mkdir(parents=True, exist_ok=True)
    (scenario / "requests.rou.xml").write_text(ROU_XML, encoding="utf-8")
    manifest = run / "saved_state" / "STATE_MANIFEST.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "branch_point_t0_s": T0,
                "physics_row_at_t0": {"time_s": T0, "live_ids": list(VEHICLES)},
            }
        ),
        encoding="utf-8",
    )
    for arm in decoys:
        decoy = chunk_dir / f"{world}__{arm}"
        decoy.mkdir(parents=True, exist_ok=True)
        (decoy / "RUN_RECEIPT.json").write_text(
            json.dumps({"schema": "wm44_branch_receipt/1", "exit_status": "complete"}),
            encoding="utf-8",
        )
        _write_jsonl(
            decoy / "run" / "physics_steps.jsonl",
            _campaign_rows({v: 18.0 for v in VEHICLES}, T0 + 0.5, T0 + coverage),
        )
        (decoy / "run" / "branch_record.json").write_text(
            json.dumps({"branch_type": arm}), encoding="utf-8"
        )
        decoy_scenario = decoy / "run" / "scenario"
        decoy_scenario.mkdir(parents=True, exist_ok=True)
        (decoy_scenario / "requests.rou.xml").write_text(ROU_XML, encoding="utf-8")
    return chunk_dir


def build_world(tmp_path: Path, planner_src: Path = None, *, world: str = WORLD,
                native_speed: float = 18.0, pi_speed: float = 21.0,
                damping_speed: float = 22.0, coverage: float = HORIZON,
                damping_coverage: float = None, decoys: tuple = (),
                refs: bool = True, planner_name: str = None) -> tuple:
    """A complete PL-EXP1b fixture: (runs root, campaign root). The runs
    root carries the planner run (copied from ``planner_src``) plus the
    ``__ref_native`` / ``__ref_pi`` reference masters; the campaign chunk
    carries ``__base`` plus the INVALID recorded decoy arms."""
    runs = tmp_path / "plexp1b"
    runs.mkdir(parents=True, exist_ok=True)
    if planner_src is not None:
        shutil.copytree(planner_src, runs / (planner_name or planner_src.name))
    if refs:
        make_ref_run(runs, world, "native", speed=native_speed, coverage=coverage)
        make_ref_run(runs, world, "pi", speed=pi_speed, coverage=coverage)
    chunk = tmp_path / "campaign" / "campaign_stage1_chunk2"
    make_chunk_world(
        chunk, world, damping_speed=damping_speed,
        coverage=HORIZON if damping_coverage is None else damping_coverage,
        decoys=decoys,
    )
    return runs, tmp_path / "campaign"


def worlds_file(tmp_path, entries=None, name="worlds.json"):
    entries = entries or [
        {"chunk": "c2", "world": WORLD, "seed": SEED, "penetration": 0.02,
         "loading_veh_h": 2700, "t0_s": T0, "split": "TRAIN"}
    ]
    path = tmp_path / name
    path.write_text(json.dumps({"worlds": entries}), encoding="utf-8")
    return path


def run_world(world_entry, runs_root, campaign, shared):
    layout, provider = shared
    return az.analyze_world(world_entry, Path(runs_root), Path(campaign), layout, provider)


WORLD_ENTRY = {"chunk": "c2", "world": WORLD, "seed": SEED, "penetration": 0.02,
               "loading_veh_h": 2700, "t0_s": T0, "split": "TRAIN"}


def independent_delay(rows, provider, layout, *, departs, t0_present, start, window):
    """A delay recomputed WITHOUT the analyzer: campaign-format rows fed
    straight to the p4 provider over the same window."""
    from traffic_shepherd.planner.recorded import ticks_from_rows, window_ticks

    kept = window_ticks(ticks_from_rows(rows, layout), start, start + window)
    delay, _waits, _n, _diag = provider(
        kept, t0_s=start, window_s=window, layout=layout, departs=departs,
        t0_present_ids=t0_present,
    )
    return delay


# ============================================================ (a) real I/F
class TestRealInterface:
    def test_real_rows_behaviour_integrity_and_numeric_delay(
        self, real_chain_run, tmp_path, shared_lib
    ):
        runs, campaign = build_world(tmp_path, real_chain_run, decoys=("native", "pi"))
        record = run_world(WORLD_ENTRY, runs, campaign, shared_lib)

        # ---- behaviour: computed from the REAL decisions.jsonl / receipt
        rows = read_jsonl(runs / real_chain_run.name / "decisions.jsonl")
        receipt = json.loads(
            (runs / real_chain_run.name / "RUN_RECEIPT.json").read_text(encoding="utf-8"))
        beh = record["behaviour"]
        assert record["exit_status"] == "complete" == receipt["exit_status"]
        assert beh["decisions_logged"] == len(rows) >= 1
        assert beh["decisions_abstained"] == sum(1 for r in rows if r["abstain"])
        assert beh["decisions_made"] == len(rows) - beh["decisions_abstained"]
        expected_actions = {}
        for r in rows:
            for action in r["actions"].values():
                expected_actions[action] = expected_actions.get(action, 0) + 1
        assert beh["actions_chosen"] == expected_actions
        assert beh["lookahead_sim_s_rows"] == pytest.approx(sum(r["lookahead_sim_s"] for r in rows))
        assert beh["lookahead_sim_s_receipt"] == pytest.approx(receipt["sim_s"]["lookahead_s"])
        assert beh["sim_s_cap"] == 500.0
        assert beh["total_sim_s_receipt"] == pytest.approx(receipt["sim_s"]["total_s"])
        assert beh["expected_decision_points"] == 2
        assert beh["gate_refusals"] == receipt["gate_refusals"] == 0
        assert beh["decisions_cadence_skipped"] == 0

        # ---- integrity (a)-(c) via the reused run_cl_smoke readers
        integ = record["integrity"]
        assert integ["c_world_seed_everywhere"] is True
        assert integ["receipt_complete"] is True
        assert integ["fresh_process_per_launch"] is True
        assert integ["n_master_launches"] == 1
        # T-A #1: the lookahead now optimises the p4 evaluation delay, so
        # the chain fixture's planner CAN find a real non-native saving
        # (the fake world's PI speed-up reduces the advance-based delay).
        # The integrity block must be CONSISTENT with whatever the planner
        # decided, and a non-native decision demonstrates the REAL command
        # path (a) with the branch-control evidence (b).
        non_native = any(
            action != "native" for row in rows for action in row["actions"].values()
        )
        assert integ["a_all_decisions_native"] is (not non_native)
        if non_native:
            assert integ["a_cav_command_differs_from_native"] is True
            assert integ["b_applicable"] is True
            assert integ["b_branch_control_rows_nonempty"] is True
        assert integ["b_applicable"] is True and integ["b_branch_control_rows_nonempty"] is True

        # ---- the REAL rows carry the campaign fields: NUMERIC delay /
        # min TTC / collision with the TRUE gap (no UNAVAILABLE)
        ident = record["planner_row_identity"]
        assert ident["row_format"] == "closed_loop"
        assert ident["delay_feedable"] is True and ident["collision_observable"] is True
        planner = record["planner"]
        assert isinstance(planner["delay_s"], float) and planner["delay_s"] > 0
        assert isinstance(planner["min_ttc_s"], float) and planner["min_ttc_s"] > 0
        assert planner["has_closing_pair"] is True
        assert planner["collision"] is False
        assert isinstance(planner["ttc_lt1p5_exposure_s"], float)
        assert isinstance(planner["ttc_lt3_exposure_s"], float)
        physics = read_jsonl(runs / real_chain_run.name / "physics_steps.jsonl")
        hard = sum(
            0.5 for r in physics for v in r["vehicles"].values() if v["accel_mps2"] < -3.0
        )
        assert planner["hard_brake_veh_s"] == pytest.approx(hard)
        assert not any("UNAVAILABLE" in reason for reason in record["unavailable"])
        # ---- TRUE comparators + window + gate
        assert record["comparator_status"] == {"native": "ok", "pi": "ok", "damping": "ok"}
        assert record["comparator_sources"]["native"].endswith(f"{WORLD}__ref_native")
        assert record["comparator_sources"]["pi"].endswith(f"{WORLD}__ref_pi")
        assert record["comparator_sources"]["damping"].endswith(f"{WORLD}__base")
        assert record["window"]["start_s"] == pytest.approx(T0 + 0.5)
        assert record["gating"]["included_in_pooled"] is True
        native, pi, damping = (
            record["arms"][a]["delay_s"] for a in ("native", "pi", "damping")
        )
        assert native > pi > damping > 0  # slower arm = more delay (p4 library)
        assert record["delta_delay_vs"]["native"]["delta_delay_s"] == pytest.approx(
            planner["delay_s"] - native)
        assert record["owner_rule_primary"]["evaluable"] is True
        assert record["owner_rule_primary"]["abs_no_collision"] is True

    def test_thin_legacy_rows_still_report_unavailable(
        self, real_chain_run, tmp_path, shared_lib
    ):
        """The guard is intact: rows WITHOUT the identity fields (a pre-PL-04k
        run) are still UNAVAILABLE, never imputed -- and the world is then
        EXCLUDED from the pooled aggregates (delay unreliable)."""
        runs, campaign = build_world(tmp_path, real_chain_run)
        run = runs / real_chain_run.name
        physics = read_jsonl(run / "physics_steps.jsonl")
        for row in physics:
            del row["colliding_ids"]
            for state in row["vehicles"].values():
                del state["road_id"], state["lane_id"]
        _write_jsonl(run / "physics_steps.jsonl", physics)
        record = run_world(WORLD_ENTRY, runs, campaign, shared_lib)
        ident = record["planner_row_identity"]
        assert ident["delay_feedable"] is False
        assert {"vehicles[*].road_id", "vehicles[*].lane_id", "colliding_ids"} <= set(
            ident["missing_fields"]
        )
        for key in ("delay_s", "min_ttc_s", "ttc_lt1p5_exposure_s",
                    "ttc_lt3_exposure_s", "collision"):
            assert record["planner"][key] is None
        assert any("UNAVAILABLE" in reason for reason in record["unavailable"])
        assert record["gating"]["included_in_pooled"] is False
        assert any(r.startswith("planner: delay unreliable")
                   for r in record["gating"]["exclusion_reasons"])

    def test_adapter_maps_real_rows_to_campaign_format(
        self, real_planner_run, tmp_path, shared_lib
    ):
        """The SAME real rows + the identity fields the runner does not yet
        write (road_id, lane_id, colliding_ids) are fed to the p4 library
        through the Layout mapping; the number equals a delay recomputed
        from those rows re-keyed by hand into the campaign format."""
        layout, provider = shared_lib
        runs, campaign = build_world(tmp_path, real_planner_run)
        run = runs / real_planner_run.name
        physics = read_jsonl(run / "physics_steps.jsonl")
        for row in physics:
            row["colliding_ids"] = []
            for state in row["vehicles"].values():
                state["road_id"] = ROAD
                state["lane_id"] = LANE
        physics[0]["vehicles"]["mainline.7"]["accel_mps2"] = -4.0  # 1 tick of hard brake
        physics[1]["vehicles"]["mainline.7"]["accel_mps2"] = -4.0  # 2 ticks: 1.0 veh-s
        _write_jsonl(run / "physics_steps.jsonl", physics)

        record = run_world(WORLD_ENTRY, runs, campaign, shared_lib)
        ident = record["planner_row_identity"]
        assert ident["delay_feedable"] and ident["collision_observable"]
        planner = record["planner"]
        assert planner["collision"] is False
        assert planner["hard_brake_veh_s"] == pytest.approx(1.0)
        assert record["planner_hard_brake_count"] == 2

        campaign_rows = [
            {
                "time_s": r["t_s"],
                "colliding_ids": [],
                "arrived_ids": [],
                "vehicles": {
                    vid: {
                        "road_id": s["road_id"],
                        "lane_id": s["lane_id"],
                        "lane_position_m": s["s_m"],
                        "speed_mps": s["v_mps"],
                        "acceleration_mps2": s["accel_mps2"],
                    }
                    for vid, s in r["vehicles"].items()
                },
            }
            for r in physics
        ]
        start = record["window"]["start_s"]
        assert start == pytest.approx(T0 + 0.5)
        expected = independent_delay(
            campaign_rows, provider, layout,
            departs={v: 1.0 for v in VEHICLES}, t0_present=set(VEHICLES),
            start=start, window=HORIZON,
        )
        assert planner["delay_s"] == pytest.approx(expected, rel=1e-9, abs=1e-9)
        # SIGN (AGENTS.md): delta = planner - reference, negative = better.
        # The planner (20 m/s) is FASTER than the native reference (18 m/s)
        # -> negative delta; SLOWER than damping (22 m/s) -> positive delta.
        native = record["arms"]["native"]["delay_s"]
        damping = record["arms"]["damping"]["delay_s"]
        assert record["delta_delay_vs"]["native"]["delta_delay_s"] == pytest.approx(
            planner["delay_s"] - native)
        assert record["delta_delay_vs"]["native"]["delta_delay_s"] < 0
        assert record["delta_delay_vs"]["damping"]["delta_delay_s"] == pytest.approx(
            planner["delay_s"] - damping)
        assert record["delta_delay_vs"]["damping"]["delta_delay_s"] > 0
        assert record["delta_delay_vs"]["native"]["delta_delay_pct_of_reference"] == (
            pytest.approx(100.0 * (planner["delay_s"] - native) / native))
        # hard-brake 1.0 veh-s exceeds native's 0.0 -> the relative rule
        # fails -> PRIMARY pass False
        assert record["owner_rule_primary"]["evaluable"] is True
        assert record["owner_rule_primary"]["abs_ok"] is True
        assert record["owner_rule_primary"]["rel_ok"] is False
        assert record["owner_rule_primary"]["pass"] is False

    def test_safe_win_flag_and_full_cli_outputs(self, real_planner_run, tmp_path, shared_lib):
        """Same, without the injected hard brake: no safety regression vs
        the native reference -> PRIMARY pass; the CLI writes the report
        JSON and CSV carrying the label, the sign convention and the gate."""
        runs, campaign = build_world(tmp_path, real_planner_run, decoys=("native", "pi"))
        out = tmp_path / "out"
        code = az.main([
            "--runs-root", str(runs), "--worlds", str(worlds_file(tmp_path)),
            "--campaign-runs-root", str(campaign), "--out", str(out),
        ])
        assert code == 0
        results = json.loads((out / "PLEXP1_RESULTS.json").read_text(encoding="utf-8"))
        assert results["label"] == "ORACLE (perfect-sim), TRAIN, descriptive n<=10"
        assert "NEGATIVE delta = the planner improves" in results["sign_convention"]
        world = results["worlds"][0]
        assert world["owner_rule_primary"]["pass"] is True
        assert world["owner_rule_secondary_3s"]["pass"] is True
        assert world["gating"]["included_in_pooled"] is True
        summary = results["summary"]
        assert summary["owner_rule_primary"]["pooled"]["n_pass"] == 1
        assert summary["per_arm_n"] == {"native": 1, "pi": 1, "damping": 1}
        assert summary["excluded_worlds"] == []
        assert summary["n_included_in_pooled"] == 1
        assert results["meta"]["primary_ttc_exposure_s"] == 1.5
        assert results["meta"]["secondary_ttc_exposure_s"] == 3.0
        report = (out / "PLEXP1_REPORT.md").read_text(encoding="utf-8")
        assert "PL-EXP1b closed-loop planner analysis" in report
        assert "ORACLE (perfect-sim), TRAIN, descriptive n<=10" in report
        assert "negative = improvement" in report
        assert "PRIMARY pass (rule C, TTC<1.5)" in report
        assert "RULE C" in report
        assert "never read" in report  # the invalid recorded arms statement
        csv_text = (out / "worlds.csv").read_text(encoding="utf-8")
        assert csv_text.splitlines()[0].startswith("label,world,chunk")
        assert "ORACLE (perfect-sim), TRAIN, descriptive n<=10" in csv_text.splitlines()[1]
        assert "owner_rule_primary_pass" in csv_text.splitlines()[0]


# ========================================================= (b) comparators
class TestComparators:
    def test_comparators_are_true_reference_runs(
        self, real_planner_run, tmp_path, shared_lib
    ):
        """native/pi metrics come from the PL-EXP1b reference masters in the
        runs root; damping from the campaign chunk base run."""
        runs, campaign = build_world(
            tmp_path, real_planner_run, native_speed=16.0, pi_speed=21.0,
            damping_speed=24.0,
        )
        record = run_world(WORLD_ENTRY, runs, campaign, shared_lib)
        assert record["comparator_status"] == {"native": "ok", "pi": "ok", "damping": "ok"}
        # the native reference runs SLOWEST (16 m/s): its delay must exceed
        # the PI (21) and damping (24) references'
        assert (
            record["arms"]["native"]["delay_s"]
            > record["arms"]["pi"]["delay_s"]
            > record["arms"]["damping"]["delay_s"] > 0
        )

    def test_recorded_native_and_pi_arms_never_opened(
        self, real_planner_run, tmp_path, shared_lib, monkeypatch
    ):
        """PATH SPY: the fixture contains fully-populated, VALID-looking
        recorded ``<world>__native`` / ``<world>__pi`` campaign arms (in the
        chunk root AND in the runs root). The analyzer must not OPEN a
        single file inside them (the arms ran random per-CAV switching and
        are invalid)."""
        runs, campaign = build_world(tmp_path, real_planner_run, decoys=("native", "pi"))
        # runs-root decoys too (the old suffix layout), equally populated
        for arm in ("native", "pi"):
            decoy = runs / f"{WORLD}__{arm}"
            decoy.mkdir()
            (decoy / "RUN_RECEIPT.json").write_text(
                json.dumps({"exit_status": "complete"}), encoding="utf-8")
            _write_jsonl(
                decoy / "physics_steps.jsonl",
                _campaign_rows({v: 18.0 for v in VEHICLES}, T0 + 0.5, T0 + HORIZON),
            )
        chunk = campaign / "campaign_stage1_chunk2"
        for arm in ("native", "pi"):
            assert (chunk / f"{WORLD}__{arm}" / "RUN_RECEIPT.json").is_file()

        opened = []
        real_builtin_open = builtins.open
        real_path_open = Path.open
        forbidden_parts = {f"{WORLD}__native", f"{WORLD}__pi"}

        def spy_builtin(file, *a, **k):
            try:
                opened.append(os.fspath(file))
            except TypeError:
                pass
            return real_builtin_open(file, *a, **k)

        def spy_path_open(self, *a, **k):
            opened.append(str(self))
            return real_path_open(self, *a, **k)

        monkeypatch.setattr(builtins, "open", spy_builtin)
        monkeypatch.setattr(Path, "open", spy_path_open)
        try:
            record = run_world(WORLD_ENTRY, runs, campaign, shared_lib)
        finally:
            monkeypatch.undo()

        assert record["comparator_status"] == {"native": "ok", "pi": "ok", "damping": "ok"}
        hits = [p for p in opened if set(Path(p).parts) & forbidden_parts]
        assert hits == [], f"the INVALID recorded arms were opened: {hits}"

    def test_missing_reference_refused(self, real_planner_run, tmp_path, shared_lib):
        """A MISSING reference run refuses with the named error (it is never
        replaced by the invalid recorded arm, never imputed)."""
        runs, campaign = build_world(tmp_path, real_planner_run)
        shutil.rmtree(runs / f"{WORLD}__ref_pi")
        with pytest.raises(az.ReferenceRunRefusedError, match="ref_pi"):
            run_world(WORLD_ENTRY, runs, campaign, shared_lib)
        assert az.main([
            "--runs-root", str(runs), "--worlds", str(worlds_file(tmp_path)),
            "--campaign-runs-root", str(campaign), "--out", str(tmp_path / "o"),
        ]) == 2

    def test_reference_receipt_identity_refused(
        self, real_planner_run, tmp_path, shared_lib
    ):
        """A reference receipt with the WRONG run_purpose or fixed_policy is
        refused with the named error."""
        runs, campaign = build_world(tmp_path, real_planner_run)
        pi_receipt = runs / f"{WORLD}__ref_pi" / "RUN_RECEIPT.json"
        receipt = json.loads(pi_receipt.read_text(encoding="utf-8"))
        receipt["fixed_policy"] = "native"
        pi_receipt.write_text(json.dumps(receipt), encoding="utf-8")
        with pytest.raises(az.ReferenceRunRefusedError, match="fixed_policy"):
            run_world(WORLD_ENTRY, runs, campaign, shared_lib)

        runs2, campaign2 = build_world(tmp_path / "w2", real_planner_run)
        native_receipt = runs2 / f"{WORLD}__ref_native" / "RUN_RECEIPT.json"
        receipt = json.loads(native_receipt.read_text(encoding="utf-8"))
        receipt["run_purpose"] = "planner"
        native_receipt.write_text(json.dumps(receipt), encoding="utf-8")
        with pytest.raises(az.ReferenceRunRefusedError, match="run_purpose"):
            run_world(WORLD_ENTRY, runs2, campaign2, shared_lib)

        # a reference relabelled control-path-proof is refused too: the
        # identity guard fires first (a non-'reference' purpose can never be
        # a valid PL-EXP1b reference receipt)
        runs3, campaign3 = build_world(tmp_path / "w3", real_planner_run)
        proof_receipt = runs3 / f"{WORLD}__ref_native" / "RUN_RECEIPT.json"
        receipt = json.loads(proof_receipt.read_text(encoding="utf-8"))
        receipt["run_purpose"] = "control_path_proof"
        proof_receipt.write_text(json.dumps(receipt), encoding="utf-8")
        with pytest.raises(az.ReferenceRunRefusedError, match="control_path_proof"):
            run_world(WORLD_ENTRY, runs3, campaign3, shared_lib)


# ==================================================== (c) TRUE-gap analyzer
class TestTrueGapAnalyzer:
    def test_analyzer_min_ttc_uses_true_gap(self, real_planner_run, tmp_path, shared_lib):
        """A closing pair with leader length 5.0 m: leader mainline.7 at 260
        (15 m/s), follower mainline.6 at 200 closing at 10 m/s for 2 s.
        TRUE bumper-to-bumper gap at t0+0.5: 25 - 5 = 20 m -> TTC 2.0 s and
        the TRUE-gap minimum is 3.5 s; the OLD front-difference gap would
        give 4.0 s. The analyzer must report the TRUE number."""
        runs, campaign = build_world(tmp_path, real_planner_run)
        run = runs / real_planner_run.name
        _write_jsonl(
            run / "physics_steps.jsonl",
            _integrated_rows(
                {
                    "mainline.6": [(2.0, 25.0), (HORIZON, 15.0)],
                    "mainline.7": [(HORIZON, 15.0)],
                    "hdv.9": [(HORIZON, 22.0)],
                },
                T0 + 0.5, T0 + HORIZON,
            ),
        )
        record = run_world(WORLD_ENTRY, runs, campaign, shared_lib)
        assert record["planner"]["min_ttc_s"] == pytest.approx(3.5)
        assert record["planner"]["has_closing_pair"] is True
        # the arms run at constant speeds: no closing pair, exposures zero
        for arm in ("native", "pi", "damping"):
            assert record["arms"][arm]["has_closing_pair"] is False
            assert record["arms"][arm]["ttc_lt1p5_exposure_s"] == 0.0
            assert record["arms"][arm]["ttc_lt3_exposure_s"] == 0.0

    def test_primary_1p5_passes_while_secondary_3s_fails(
        self, real_planner_run, tmp_path, shared_lib
    ):
        """The addendum-2 scenario: the planner's TTC<3 exposure RISES vs the
        native reference while its TTC<1.5 exposure does not -> PRIMARY pass,
        SECONDARY fail. The planner pair closes at 3 m/s with a true gap of
        25 m for 6 s (TTC 2.33..7.83 s: two ticks under 3 s, none under
        1.5 s); the native reference has no closing pair at all."""
        runs, campaign = build_world(tmp_path, real_planner_run)
        run = runs / real_planner_run.name
        # leader mainline.7 repositioned to 230 m (15 m/s): true gap 25 m,
        # closing 3 m/s while mainline.6 runs 18 m/s (t' <= 6 s)
        _write_jsonl(
            run / "physics_steps.jsonl",
            _integrated_rows(
                {
                    "mainline.6": [(6.0, 18.0), (HORIZON, 14.0)],
                    "mainline.7": [(HORIZON, 15.0)],
                    "hdv.9": [(HORIZON, 30.0)],
                },
                T0 + 0.5, T0 + HORIZON,
                start_pos={"mainline.6": 200.0, "mainline.7": 230.0, "hdv.9": 330.0},
            ),
        )
        record = run_world(WORLD_ENTRY, runs, campaign, shared_lib)
        planner = record["planner"]
        assert planner["min_ttc_s"] == pytest.approx((25.0 - 18.0) / 3.0)  # 2.33 s
        assert planner["ttc_lt1p5_exposure_s"] == pytest.approx(0.0)
        assert planner["ttc_lt3_exposure_s"] == pytest.approx(1.0)  # 2 ticks x 0.5 s
        native = record["arms"]["native"]
        assert native["ttc_lt1p5_exposure_s"] == 0.0
        assert native["ttc_lt3_exposure_s"] == 0.0
        primary = record["owner_rule_primary"]
        secondary = record["owner_rule_secondary_3s"]
        assert primary["evaluable"] is True and primary["pass"] is True
        assert primary["rel_detail"]["ttc_lt1p5_exposure_s_not_worse"] is True
        assert secondary["evaluable"] is True and secondary["pass"] is False
        assert secondary["rel_detail"]["ttc_lt3_exposure_s_not_worse"] is False
        # secondary (report-only) verdicts vs PI and damping references
        for arm in ("pi", "damping"):
            assert record["secondary_reference_verdicts"][arm]["pass"] is True


# ====================================================== (d) sign convention
class TestSignConvention:
    def test_delta_delay_sign_pinned(self, real_planner_run, tmp_path, shared_lib):
        """delta = planner - reference (AGENTS.md): NEGATIVE = improvement.
        The planner (20 m/s) beats the slower native reference (18 m/s):
        delta negative; loses to damping (22 m/s): delta positive."""
        runs, campaign = build_world(tmp_path, real_planner_run)
        record = run_world(WORLD_ENTRY, runs, campaign, shared_lib)
        planner = record["planner"]["delay_s"]
        for arm, sign in (("native", -1), ("damping", +1), ("pi", +1)):
            delta = record["delta_delay_vs"][arm]["delta_delay_s"]
            ref = record["arms"][arm]["delay_s"]
            assert delta == pytest.approx(planner - ref)
            assert (delta < 0) is (sign < 0)
        assert record["safety_delta_vs"]["native"]["hard_brake_veh_s"] == pytest.approx(0.0)
        out = tmp_path / "out"
        assert az.main([
            "--runs-root", str(runs), "--worlds", str(worlds_file(tmp_path)),
            "--campaign-runs-root", str(campaign), "--out", str(out),
        ]) == 0
        results = json.loads((out / "PLEXP1_RESULTS.json").read_text(encoding="utf-8"))
        assert results["sign_convention"].startswith(
            "AGENTS.md sign convention: delta = planner - reference")
        report = (out / "PLEXP1_REPORT.md").read_text(encoding="utf-8")
        assert "Sign convention" in report and "negative = improvement" in report


# ============================================================= (e) gating
class TestGating:
    def test_unreliable_world_excluded_with_per_arm_n(
        self, real_planner_run, tmp_path, shared_lib
    ):
        """A world whose planner delay is unreliable (thin rows) is EXCLUDED
        from the pooled aggregates; the summary reports per-arm n and lists
        the excluded world with its reason."""
        world_b = "w_pl04i_b"
        entry_b = {"chunk": "c2", "world": world_b, "seed": SEED, "penetration": 0.05,
                   "loading_veh_h": 2400, "t0_s": T0, "split": "TRAIN"}
        runs, campaign = build_world(tmp_path, real_planner_run)
        make_ref_run(runs, world_b, "native", speed=18.0)
        make_ref_run(runs, world_b, "pi", speed=21.0)
        make_chunk_world(campaign / "campaign_stage1_chunk2", world_b)
        thin = runs / f"{world_b}__planner_oracle"
        shutil.copytree(real_planner_run, thin)
        physics = read_jsonl(thin / "physics_steps.jsonl")
        for row in physics:
            del row["colliding_ids"]
            for state in row["vehicles"].values():
                del state["road_id"], state["lane_id"]
        _write_jsonl(thin / "physics_steps.jsonl", physics)

        out = tmp_path / "out"
        assert az.main([
            "--runs-root", str(runs), "--worlds",
            str(worlds_file(tmp_path, [WORLD_ENTRY, entry_b])),
            "--campaign-runs-root", str(campaign), "--out", str(out),
        ]) == 0
        results = json.loads((out / "PLEXP1_RESULTS.json").read_text(encoding="utf-8"))
        summary = results["summary"]
        assert summary["n_registered_worlds"] == 2
        assert summary["n_included_in_pooled"] == 1
        assert [w["world"] for w in summary["excluded_worlds"]] == [world_b]
        assert any("delay unreliable" in r for r in summary["excluded_worlds"][0]["reasons"])
        assert summary["per_arm_n"] == {"native": 1, "pi": 1, "damping": 1}
        by_world = {r["world"]: r for r in results["worlds"]}
        assert by_world[WORLD]["gating"]["included_in_pooled"] is True
        assert by_world[world_b]["gating"]["included_in_pooled"] is False
        # the excluded world's numbers are still REPORTED (never deleted)
        assert by_world[world_b]["planner"]["delay_s"] is None
        report = (out / "PLEXP1_REPORT.md").read_text(encoding="utf-8")
        assert world_b in report and "delay unreliable" in report


# ================================================================ (f) guards
class TestGuards:
    def test_non_oracle_receipt_refused(self, real_planner_run, tmp_path, shared_lib):
        runs, campaign = build_world(tmp_path, real_planner_run)
        run = runs / real_planner_run.name
        receipt = json.loads((run / "RUN_RECEIPT.json").read_text(encoding="utf-8"))
        receipt["predictor_class"] = "learned"
        (run / "RUN_RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")
        with pytest.raises(az.PredictorClassRefusedError, match="predictor_class"):
            run_world(WORLD_ENTRY, runs, campaign, shared_lib)

    def test_control_path_proof_run_refused(self, real_proof_run, tmp_path, shared_lib):
        # a REAL --initial-plan run (run_purpose='control_path_proof' in the
        # receipt and every decisions.jsonl row)
        receipt = json.loads((real_proof_run / "RUN_RECEIPT.json").read_text(encoding="utf-8"))
        assert receipt["run_purpose"] == "control_path_proof"
        runs, campaign = build_world(tmp_path, real_proof_run)
        with pytest.raises(az.RunPurposeRefusedError, match="control_path_proof"):
            run_world(WORLD_ENTRY, runs, campaign, shared_lib)

    def test_control_path_proof_rows_refused_even_if_receipt_edited(
        self, real_proof_run, tmp_path, shared_lib
    ):
        runs, campaign = build_world(tmp_path, real_proof_run)
        run = runs / real_proof_run.name
        receipt = json.loads((run / "RUN_RECEIPT.json").read_text(encoding="utf-8"))
        receipt["run_purpose"] = "planner"  # relabelled receipt, rows still say proof
        (run / "RUN_RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")
        with pytest.raises(az.RunPurposeRefusedError):
            run_world(WORLD_ENTRY, runs, campaign, shared_lib)

    def test_non_train_world_refused(self, tmp_path):
        entry = dict(WORLD_ENTRY, split="HOLDOUT")
        path = worlds_file(tmp_path, [entry])
        with pytest.raises(az.NonTrainWorldRefusedError, match="TRAIN"):
            az.load_worlds(path)
        assert az.main([
            "--runs-root", str(tmp_path / "r"), "--worlds", str(path),
            "--campaign-runs-root", str(tmp_path / "c"), "--out", str(tmp_path / "o"),
        ]) == 2

    def test_non_train_receipt_refused(self, real_planner_run, tmp_path, shared_lib):
        runs, campaign = build_world(tmp_path, real_planner_run)
        run = runs / real_planner_run.name
        receipt = json.loads((run / "RUN_RECEIPT.json").read_text(encoding="utf-8"))
        receipt["split"] = "holdout"
        (run / "RUN_RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")
        with pytest.raises(az.NonTrainWorldRefusedError):
            run_world(WORLD_ENTRY, runs, campaign, shared_lib)

    def test_pooling_without_per_penetration_refused(self):
        with pytest.raises(az.PoolingGuardError):
            az.assert_stratified({"saving": {"pooled": {"n": 3, "median": 1.0}}})
        az.assert_stratified({"saving": {"pooled": {"n": 3}, "by_penetration": {}}})
        # every statistic the analyzer builds is stratified by construction
        rows = [_synthetic_row(0.02, 2700, 10.0), _synthetic_row(0.05, 2600, 20.0)]
        summary = az.build_summary(rows)
        for key in ("delta_delay_vs_native_s", "delta_delay_vs_native_pct",
                    "delta_delay_vs_damping_s", "delta_delay_vs_pi_s",
                    "owner_rule_primary", "owner_rule_secondary_3s"):
            assert "by_penetration" in summary[key] and "pooled" in summary[key]

    def test_out_dir_guard(self, tmp_path):
        with pytest.raises(az.OutDirRefusedError):
            az.guard_out_dir(tmp_path / "runs" / "x")
        with pytest.raises(az.OutDirRefusedError):
            az.guard_out_dir("D" + ":/anything")
        with pytest.raises(az.OutDirRefusedError):
            az.guard_out_dir(tmp_path / "in" / "out", tmp_path / "in")


# =============================================== (g) missing run/comparator
class TestMissing:
    def test_missing_planner_run_reported_not_imputed(self, tmp_path, shared_lib):
        runs, campaign = build_world(tmp_path, planner_src=None)
        record = run_world(WORLD_ENTRY, runs, campaign, shared_lib)
        assert record["exit_status"] == "missing"
        assert record["planner"] is None and record["behaviour"] is None
        assert record["integrity"] is None
        assert all(v["delta_delay_s"] is None for v in record["delta_delay_vs"].values())
        assert record["owner_rule_primary"]["pass"] is None
        assert any("planner run missing" in r for r in record["unavailable"])
        # the TRUE comparators are still real numbers
        assert record["arms"]["native"]["delay_s"] > 0
        assert record["arms"]["damping"]["delay_s"] > 0
        # nothing unreliable -> the world still enters the pooled aggregates
        assert record["gating"]["included_in_pooled"] is True

    def test_missing_and_incomplete_damping_reported(self, real_planner_run, tmp_path,
                                                     shared_lib):
        runs, campaign = build_world(tmp_path, real_planner_run)
        shutil.rmtree(campaign / "campaign_stage1_chunk2" / f"{WORLD}__base")
        record = run_world(WORLD_ENTRY, runs, campaign, shared_lib)
        assert record["comparator_status"]["damping"] == "missing"
        assert record["arms"]["damping"] is None
        assert record["delta_delay_vs"]["damping"]["delta_delay_s"] is None
        assert record["delta_delay_vs"]["native"]["delta_delay_s"] is not None
        assert any("comparator damping: missing" in r for r in record["unavailable"])

        # an incomplete base receipt is 'not_complete', never silently mixed
        runs2, campaign2 = build_world(tmp_path / "w2", real_planner_run)
        base = campaign2 / "campaign_stage1_chunk2" / f"{WORLD}__base"
        (base / "RUN_RECEIPT.json").write_text(
            json.dumps({"exit_status": "running"}), encoding="utf-8")
        record = run_world(WORLD_ENTRY, runs2, campaign2, shared_lib)
        assert record["comparator_status"]["damping"].startswith("not_complete")

    def test_short_damping_window_reported_incomplete(self, tmp_path, shared_lib):
        runs, campaign = build_world(
            tmp_path, planner_src=None, damping_coverage=HORIZON - 5.0)
        record = run_world(WORLD_ENTRY, runs, campaign, shared_lib)
        assert record["comparator_status"]["damping"].startswith("window_incomplete")
        assert record["arms"]["damping"] is None
        assert record["comparator_status"]["native"] == "ok"

    def test_cli_with_everything_missing_refuses(self, tmp_path):
        """No planner run and no reference runs: the named reference refusal
        stops the CLI with exit code 2 (no outputs are written)."""
        out = tmp_path / "out"
        code = az.main([
            "--runs-root", str(tmp_path / "none"), "--worlds", str(worlds_file(tmp_path)),
            "--campaign-runs-root", str(tmp_path / "nocamp"), "--out", str(out),
        ])
        assert code == 2
        assert not (out / "PLEXP1_RESULTS.json").exists()


# ======================================================= (h) stratification
def _synthetic_row(pen, veh_h, delta, primary=True, secondary=None, actions=None):
    """A per-world record with only the fields the stratifiers read."""
    return {
        "world": f"w{pen}_{veh_h}", "chunk": "c2", "penetration": pen, "loading_veh_h": veh_h,
        "demand_class": az.demand_class(veh_h), "exit_status": "complete", "unavailable": [],
        "delta_delay_vs": {
            "native": {"delta_delay_s": delta, "delta_delay_pct_of_reference": delta / 10.0},
            "damping": {"delta_delay_s": delta / 2, "delta_delay_pct_of_reference": delta / 20.0},
            "pi": {"delta_delay_s": None, "delta_delay_pct_of_reference": None},
        },
        "owner_rule_primary": {"pass": primary},
        "owner_rule_secondary_3s": {"pass": secondary},
        # Rule C: report-only verdicts (old rule A, A + 3 s, relative-only)
        "owner_rule_a_report_only": {"pass": primary},
        "owner_rule_a_3s_report_only": {"pass": secondary},
        "owner_rule_relative_only_report_only": {"pass": primary},
        "behaviour": {"actions_chosen": actions or {"native": 2}, "gate_refusals": 0},
    }


class TestStratification:
    def test_per_penetration_and_demand_never_pooled_alone(self):
        rows = [
            _synthetic_row(0.02, 2700, 10.0, True, True, {"native": 3, "damping": 1}),
            _synthetic_row(0.02, 2400, 30.0, False, False, {"pi-external-u": 2}),
            _synthetic_row(0.05, 2700, 50.0, True, True),
            _synthetic_row(0.05, 2600, -20.0, None, None),  # not evaluable
            _synthetic_row(0.4, 2700, 100.0, True, False),
        ]
        summary = az.build_summary(rows)
        s = summary["delta_delay_vs_native_s"]
        # deltas (planner - reference): negative = improvement
        assert s["pooled"] == {"n": 5, "median": 30.0, "mean": pytest.approx(34.0)}
        assert s["by_penetration"]["0.02"] == {"n": 2, "median": 20.0, "mean": 20.0}
        assert s["by_penetration"]["0.05"]["median"] == pytest.approx(15.0)
        assert s["by_penetration"]["0.4"]["n"] == 1
        assert s["by_demand"]["2700"]["n"] == 3 and s["by_demand"]["<2700"]["n"] == 2
        assert s["by_penetration_demand"]["0.02|<2700"]["median"] == 30.0
        # the unavailable PI comparator is n=0 everywhere, not imputed as 0
        assert summary["delta_delay_vs_pi_s"]["pooled"] == {"n": 0, "median": None, "mean": None}
        assert summary["per_arm_n"] == {"native": 5, "pi": 0, "damping": 5}
        frac = summary["owner_rule_primary"]
        assert frac["pooled"]["n_registered"] == 5
        assert frac["pooled"]["n_evaluable"] == 4  # the None row is not evaluable
        assert frac["pooled"]["n_pass"] == 3
        assert frac["pooled"]["fraction_of_evaluable"] == pytest.approx(0.75)
        assert frac["pooled"]["fraction_of_registered"] == pytest.approx(0.6)
        assert frac["by_penetration"]["0.05"]["n_evaluable"] == 1
        sec = summary["owner_rule_secondary_3s"]
        assert sec["pooled"]["n_pass"] == 2 and sec["pooled"]["n_evaluable"] == 4
        assert summary["actions_chosen_by_penetration"]["0.02"] == {
            "damping": 1, "native": 3, "pi-external-u": 2}
        az.assert_stratified(summary)

    def test_report_lists_penetration_strata_before_pooled(self, tmp_path):
        rows = [_synthetic_row(0.02, 2700, 10.0), _synthetic_row(0.05, 2600, 20.0)]
        for r in rows:
            r.update(
                seed=1, t0_s=0.0, split="TRAIN", planner=None, arms={}, planner_row_identity=None,
                integrity=None, comparator_status={}, comparator_sources={}, window={},
                behaviour=None, safety_delta_vs={}, secondary_reference_verdicts={
                    "pi": {"pass": None}, "damping": {"pass": None}},
                delta_delay_vs=r["delta_delay_vs"], gating={"included_in_pooled": True,
                                                            "exclusion_reasons": []},
                planner_hard_brake_count=None,
            )
            r["owner_rule_primary"].update(abs_ok=None, rel_ok=None)
            r["owner_rule_secondary_3s"].update(abs_ok=None, rel_ok=None)
            for key in ("owner_rule_a_report_only", "owner_rule_a_3s_report_only",
                        "owner_rule_relative_only_report_only"):
                r[key].update(abs_ok=None, rel_ok=None)
        summary = az.build_summary(rows)
        text = az.render_report(
            rows, summary,
            {"worlds": "w", "runs_root": "r", "campaign_runs_root": "c"},
        )
        assert text.index("penetration | 0.02") < text.index("POOLED")
        assert "penetration | 0.05" in text and "demand | <2700" in text
        assert "n<=10" in text


# ============================================ (i) RULE C (PL-EXP1c) + refs root
def _m(min_ttc, *, collision=False, hb=0.0, e15=0.0, e3=0.0, pair=True):
    return {"collision": collision, "hard_brake_veh_s": hb, "ttc_lt1p5_exposure_s": e15,
            "ttc_lt3_exposure_s": e3, "min_ttc_s": min_ttc,
            "has_closing_pair": (min_ttc is not None) if pair is True else pair}


class TestRuleCAnalyzer:
    def test_primary_is_rule_c_below_1p5_but_not_below_native(self):
        planner, native = _m(1.2), _m(1.0)
        c = az.owner_rule(planner, native, 1.5, "ttc_lt1p5_exposure_s")
        assert c["pass"] is True and c["rule"] == "C"
        assert c["ttc_floor_applied_s"] == pytest.approx(1.0)
        a = az.owner_rule(planner, native, 1.5, "ttc_lt1p5_exposure_s", az.RULE_A)
        assert a["pass"] is False and a["ttc_floor_applied_s"] == 1.5

    def test_below_native_fails_and_equal_passes(self):
        assert az.owner_rule(_m(0.9), _m(1.0), 1.5, "ttc_lt1p5_exposure_s")["pass"] is False
        assert az.owner_rule(_m(1.0), _m(1.0), 1.5, "ttc_lt1p5_exposure_s")["pass"] is True
        assert az.owner_rule(_m(1.0 - 5e-7), _m(1.0), 1.5, "ttc_lt1p5_exposure_s")["pass"] is True

    def test_native_at_least_1p5_or_no_pair_floor_is_1p5(self):
        assert az.owner_rule(_m(1.4), _m(3.0), 1.5, "ttc_lt1p5_exposure_s")["pass"] is False
        no_pair = _m(None)
        assert az.owner_rule(_m(1.4), no_pair, 1.5, "ttc_lt1p5_exposure_s")["pass"] is False
        assert az.owner_rule(_m(1.5), no_pair, 1.5, "ttc_lt1p5_exposure_s")["pass"] is True

    def test_guards_remain(self):
        # collision, hard-brake and exposure still fail rule C
        assert az.owner_rule(_m(1.0, collision=True), _m(1.0), 1.5,
                             "ttc_lt1p5_exposure_s")["pass"] is False
        assert az.owner_rule(_m(1.0, hb=1.0), _m(1.0), 1.5, "ttc_lt1p5_exposure_s")["pass"] is False
        assert az.owner_rule(_m(1.0, e15=1.0), _m(1.0), 1.5,
                             "ttc_lt1p5_exposure_s")["pass"] is False
        assert az.owner_rule(None, _m(1.0), 1.5, "ttc_lt1p5_exposure_s")["pass"] is None

    def test_relative_only_ignores_ttc_and_collision_bar(self):
        r = az.owner_rule(_m(0.2, collision=True), _m(3.0), 1.5, "ttc_lt1p5_exposure_s",
                          az.RULE_REL_ONLY)
        assert r["pass"] is True

    def test_world_record_carries_primary_c_and_report_only_variants(
        self, real_planner_run, tmp_path, shared_lib
    ):
        runs, campaign = build_world(tmp_path, real_planner_run)
        record = run_world(WORLD_ENTRY, runs, campaign, shared_lib)
        assert record["owner_rule_primary"]["rule"] == "C"
        assert record["owner_rule_a_report_only"]["rule"] == "A"
        assert record["owner_rule_a_3s_report_only"]["rule"] == "A"
        assert record["owner_rule_relative_only_report_only"]["rule"] == "relative_only"
        assert record["owner_rule_secondary_3s"]["ttc_exposure_key"] == "ttc_lt3_exposure_s"

    def test_refs_root_split_and_default(self, real_planner_run, tmp_path, shared_lib):
        # references in plexp1b, planner run ONLY in plexp1c
        refs_runs, campaign = build_world(tmp_path, None)
        planner_root = tmp_path / "plexp1c"
        planner_root.mkdir()
        shutil.copytree(real_planner_run, planner_root / real_planner_run.name)
        layout, provider = shared_lib
        record = az.analyze_world(
            WORLD_ENTRY, planner_root, Path(campaign), layout, provider, 1.5,
            Path(refs_runs))
        assert record["comparator_sources"]["native"] == str(refs_runs / f"{WORLD}__ref_native")
        assert record["planner"] is not None
        assert record["arms"]["native"] is not None
        # without --refs-root the references are looked up in the runs root: refused
        with pytest.raises(az.ReferenceRunRefusedError):
            az.analyze_world(WORLD_ENTRY, planner_root, Path(campaign), layout, provider)
        # the CLI takes --refs-root and keeps the reference receipt guards
        out = tmp_path / "out_split"
        code = az.main([
            "--runs-root", str(planner_root), "--refs-root", str(refs_runs),
            "--worlds", str(worlds_file(tmp_path)),
            "--campaign-runs-root", str(campaign), "--out", str(out),
        ])
        assert code == 0
        results = json.loads((out / "PLEXP1_RESULTS.json").read_text(encoding="utf-8"))
        assert results["meta"]["refs_root"] == str(refs_runs)
        report = (out / "PLEXP1_REPORT.md").read_text(encoding="utf-8")
        assert "RULE C" in report and "rule A" in report
        # a bad reference receipt in the split root is still refused (exit 2)
        rcpt = refs_runs / f"{WORLD}__ref_native" / "RUN_RECEIPT.json"
        data = json.loads(rcpt.read_text(encoding="utf-8"))
        data["fixed_policy"] = "pi"
        rcpt.write_text(json.dumps(data), encoding="utf-8")
        assert az.main([
            "--runs-root", str(planner_root), "--refs-root", str(refs_runs),
            "--worlds", str(worlds_file(tmp_path)),
            "--campaign-runs-root", str(campaign), "--out", str(tmp_path / "o2"),
        ]) == 2
