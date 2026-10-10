"""PL-10: PerfectSimWorldModel contract-v2 outputs against REAL call shapes.

Covers PREDICT_CONTRACT_v2.md (read-only, WD-01 worktree):
commit-window semantics (continuation='native'), realised future_* heads,
fail-closed validity, future wall heads, N1, CRN, the PL-04 refusal, the
WM-41/44 snapshot exactness guard (vendored thin adapter), and the kit's own
``validate_prediction`` (v2 validator, WD-01 worktree's predict/)
run on the model's outputs. The fake traci/branch runner mimics the real
saved_state/suffix_runner call shapes (guarded restore; branch_types-style
dispatcher payloads with commit_horizon_s=0, strict_roster, WM-40 walls).

Zero simulation: no SUMO/traci/libsumo import; the traci surface is a fake
api object that writes/reads real state files.
"""

import dataclasses
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from traffic_shepherd.planner import PlannerConfig, PlanCandidate, WallSpec, plan_key
from traffic_shepherd.planner.perfect_sim import (
    BranchRecord,
    BranchRunner,
    PerfectSimWorldModel,
    SimpleArrivalDraws,
    StepRecord,
    TraciStateSnapshotter,
    VehicleStep,
    WallEventRecord,
)
from traffic_shepherd.planner.perfect_sim_state import (
    SCHEMA,
    StateSnapshotError,
    validate_state_file_for_exactness,
)
from traffic_shepherd.planner.paths import rd_wt_root
from traffic_shepherd.planner.safety import (
    FutureSafetyRefusal,
    enforce_future_gate,
    safety_check,
)

T0 = 40.0
COMMIT = 10.0
HORIZON = 20.0
PART_A = 20.0  # model part_a_horizon_s (v2: commit_s <= part_a_horizon_s)

# -- the WD-01 v2 validator kit (read-only sibling tree). The v1/v1.1-only
#    kit (wd01b) may already own the `predict` package name for the
#    conformance tests, and the kit uses lazy intra-package imports -- so
#    this module loads the v2-capable tree under an ALIAS
#    ("wd01_predict_v2_kit"): both kits coexist, neither can poison the
#    other's module identity. --
KIT_ROOT = rd_wt_root() / "wd01" / "output" / "wd01"
KIT_PACKAGE = "wd01_predict_v2_kit"

if not (KIT_ROOT / "predict").is_dir():
    pytest.skip(
        f"WD-01 predict kit (v2 validator) not found at {KIT_ROOT}",
        allow_module_level=True,
    )


def _load_kit():
    import importlib.util

    existing = sys.modules.get(KIT_PACKAGE)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        KIT_PACKAGE,
        KIT_ROOT / "predict" / "__init__.py",
        submodule_search_locations=[str(KIT_ROOT / "predict")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[KIT_PACKAGE] = module
    spec.loader.exec_module(module)
    return module


try:
    kit = _load_kit()
    kit_validate = importlib.import_module(f"{KIT_PACKAGE}.validate")
except ImportError as exc:  # pragma: no cover - kit present but broken
    pytest.skip(
        f"WD-01 predict kit is not importable from {KIT_ROOT}: {exc!r}",
        allow_module_level=True,
    )

WM34_REAL_ROOT = str(rd_wt_root() / "wm34" / "output")

# PL-10c: the rig's declared scenario, CAMPAIGN FORMAT (explicit <vehicle>
# list, fixed type=, speedDev="0"): no post-t0 randomness, so the
# scenario-exactness guard passes it and the restored-RNG kit-ArrivalDraw
# test stays runnable. Flow-style scenario fixtures are refused by design
# (tests/planner/test_perfect_sim_scenario_guard.py).
CAMPAIGN_ROUTES = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
  <vType id="hdv" carFollowModel="IDM" sigma="0" speedDev="0" speedFactor="1.0"/>
  <route id="mainline" edges="e0 e1"/>
  <vehicle id="v0" type="hdv" route="mainline" depart="0.00" departLane="free" departSpeed="max"/>
  <vehicle id="v1" type="hdv" route="mainline" depart="1.50" departLane="free" departSpeed="max"/>
</routes>
"""

# safe everywhere: gap 30 / closing 10 -> TTC 3.0 (>= the 1.5 floor; the
# exposure threshold at < 3 s stays 0); the 0.8 s scripts exercise vetoes.
SAFE_GAP = 30.0

GOOD_STATE = (
    '<state><rngState seed="7"/><vehicle id="c0" pos="12.34567890123456789" speed="20.0"/></state>'
)
LOW_PRECISION_STATE = (
    '<state><rngState seed="7"/><vehicle id="c0" pos="12.35" speed="20.0"/></state>'
)
NO_RNG_STATE = '<state><vehicle id="c0" pos="12.34567890123456789"/></state>'

V2_KW = dict(horizon_s=HORIZON, commit_s=COMMIT, continuation="native")

PLAN_A = PlanCandidate(per_cav={"c0": "damping"})
PLAN_B = PlanCandidate(per_cav={"c0": "native"})


# ------------------------------------------------------------------ fakes
class FakeSimApi:
    """traci-shaped fake: simulation.saveState/loadState/getTime only."""

    def __init__(self, state_text, time_s=T0):
        self._text = state_text
        self._time_s = float(time_s)
        self.saved = []
        self.loaded = []

    def saveState(self, path):
        Path(path).write_text(self._text, encoding="utf-8")
        self.saved.append(str(path))

    def loadState(self, path):
        self.loaded.append(str(path))

    def getTime(self):
        return self._time_s


def fake_api(state_text=GOOD_STATE, time_s=T0):
    return SimpleNamespace(simulation=FakeSimApi(state_text, time_s))


class FakeProcessFactory:
    """PL-10d fake launcher: ONE fresh fake traci connection per branch.

    Each connection records its own loadState calls (FakeSimApi.loaded), so
    the tests can prove every branch restored exactly once into its own
    process and that the master api never saw a loadState."""

    def __init__(self, state_text=GOOD_STATE, time_s=T0):
        self._state_text = state_text
        self._time_s = float(time_s)
        self.started = []
        self.closed = []

    def start_process(self, *, label=None):
        connection = fake_api(self._state_text, self._time_s)
        connection.label = label
        self.started.append(connection)
        return connection

    def close_process(self, connection):
        assert connection in self.started, "closed a process that never started"
        assert connection not in self.closed, "a process was closed twice"
        self.closed.append(connection)


class FakeSuffixBranchRunner(BranchRunner):
    """Scripted branches through the REAL guarded restore call shape.

    Mimics the real machinery: the guarded restore
    (``load_branch_state(api, world_dir, expected_t_b=..., sumo_version=...,
    scenario_digests=...)`` inside ``snapshotter.restore``) runs before
    anything else (suffix_runner.run_suffix shape), then the plan is staged
    as a branch_types-style dispatcher payload (strict full roster,
    commit_horizon_s=0, WM-40 wall tuples) and the scripted physics record
    is returned."""

    def __init__(
        self,
        snapshotter,
        *,
        process_factory=None,
        t0_s=T0,
        wait_s=None,
        gap_m=SAFE_GAP,
        leader_speed_mps=10.0,
        wall_ids=("w0",),
        teleport_times=(),
        run_error=False,
        wall_assembled_at=T0 + 5.0,
    ):
        super().__init__(process_factory=process_factory)
        self.snapshotter = snapshotter
        self.t0_s = float(t0_s)
        self.wait_s = dict(wait_s or {})
        self.gap_m = gap_m
        self.leader_speed_mps = leader_speed_mps
        self.wall_ids = tuple(wall_ids)
        self.teleport_times = tuple(teleport_times)
        self.run_error = run_error
        self.wall_assembled_at = wall_assembled_at
        self.calls = []

    def run(
        self,
        plan,
        *,
        connection,
        manifest,
        draw_id,
        departures,
        horizon_s,
        safety_window_s,
        controller_state,
        kept_wall_ids,
        stop_on_window_collision=True,
        commit_s=None,
        continuation=None,
    ):
        is_native = getattr(plan, "label", None) == "native"
        # -- real call shape 1: guarded restore INTO THE BRANCH'S OWN
        #    PROCESS (PL-10d; fails loudly on a bad state, refuses a reused
        #    process with ProcessReuseError before any loadState)
        self.snapshotter.restore(manifest, plan=plan, draw_id=draw_id, api=connection)
        # -- real call shape 2: the dispatcher payload (branch_types.make_plan)
        roster = sorted(controller_state or {"c0": {}})
        key = "native" if is_native else plan_key(plan)
        payload = {
            "assignments": {} if is_native else dict(plan.per_cav),
            "plan_id": f"{'native' if is_native else 'plan'}@{draw_id}",
            "commit_horizon_s": 0.0,  # the planner re-decides; dwell 0
            "walls": tuple(
                (w.wall_id, tuple(w.members), float(w.speed_mps), w.phase)
                for w in getattr(plan, "walls", ())
            ),
            "strict_roster": True,
        }
        if is_native:
            payload["assignments"] = {cav: "native" for cav in roster}
        else:
            for _wid, members, _speed, _phase in payload["walls"]:
                # wall members are committed to wall-v2 (the dispatcher's
                # group action), never named in per_cav
                for member in members:
                    payload["assignments"][member] = "wall-v2"
        assert set(payload["assignments"]) == set(roster), "strict_roster violated"
        self.calls.append(
            {
                "plan": key,
                "connection": connection,
                "draw_id": draw_id,
                "departures": tuple(departures),
                "horizon_s": float(horizon_s),
                "safety_window_s": float(safety_window_s),
                "commit_s": commit_s,
                "continuation": continuation,
                "payload": payload,
            }
        )
        n_steps = int(horizon_s) + 1
        steps = []
        for i in range(n_steps):
            vehicles = {
                "c0": VehicleStep(
                    s_m=20.0 * i,
                    lane_id=0.0,
                    speed_mps=20.0,
                    accel_mps2=0.0,
                    entry_wait_s=float(self.wait_s.get(key, 0.0)),
                    on_section=True,
                    leader_gap_m=self.gap_m,
                    leader_speed_mps=self.leader_speed_mps if self.gap_m is not None else None,
                )
            }
            steps.append(StepRecord(time_s=self.t0_s + i, dt_s=1.0, vehicles=vehicles))
        walls = {}
        for wall in getattr(plan, "walls", ()):
            if wall.wall_id in self.wall_ids:
                walls[wall.wall_id] = WallEventRecord(assembled_at_s=self.wall_assembled_at)
        return BranchRecord(
            plan_label=str(key),
            draw_id=draw_id,
            t0_s=self.t0_s,
            horizon_s=float(horizon_s),
            dt_s=1.0,
            free_flow_s={"c0": 30.0},
            steps=tuple(steps),
            wall_events=walls,
            tracked=("c0",),
            run_error=self.run_error,
            teleport_times=tuple(self.teleport_times),
        )


class _CollidingRunner(FakeSuffixBranchRunner):
    """The plan branch collides inside the commit window: early stop."""

    def run(self, plan, **kw):
        record = super().run(plan, **kw)
        if getattr(plan, "label", None) == "native":
            return record
        fields = {name: getattr(record, name) for name in record.__dataclass_fields__}
        return BranchRecord(
            **{
                **fields,
                "collision_times": (T0 + 5.0,),
                "stopped_early": True,
                "stopped_at_s": T0 + 5.0,
            }
        )


def make_sample(k=1, arrivals=None):
    out = []
    for i in range(k):
        payload = (
            {
                "sample_id": f"k{i}",
                "departures": ((T0 + 5.0, "main", 20.0, False, "passenger"),),
            }
            if arrivals is None
            else arrivals(i)
        )
        out.append(
            SimpleNamespace(
                hybrid={"roster": ["c0"]},
                arrivals=payload,
                meta={
                    "sample_id": f"k{i}",
                    "time_s": T0,
                    "merge_observed": True,
                    "coverage": 1.0,
                },
            )
        )
    return out


class Setup:
    """One test rig: shared snapshotter (decision dirs increment), runner,
    and model factories for the three contract versions. Models declare the
    rig's campaign-format scenario (the PL-10c exactness guard runs on every
    predict_many)."""

    def __init__(self, root: Path, runner=None, snapshotter=None, api=None):
        self.root = root
        self.api = api or fake_api()
        self.factory = FakeProcessFactory()
        self.routes = self.root / "campaign.rou.xml"
        self.routes.parent.mkdir(parents=True, exist_ok=True)
        self.routes.write_text(CAMPAIGN_ROUTES, encoding="utf-8")
        self.snapshotter = snapshotter or TraciStateSnapshotter(
            self.api,
            root / "root",
            wm34_output_root=str(root / "no-such-wm34-root (vendored)"),
        )
        self.runner = runner or FakeSuffixBranchRunner(
            self.snapshotter, process_factory=self.factory
        )

    def v2_model(self, **over):
        return PerfectSimWorldModel(
            self.snapshotter,
            SimpleArrivalDraws(),
            self.runner,
            part_a_horizon_s=PART_A,
            contract_version="wd01_predict/2",
            model_id="perfect_sim/v2-test",
            sumo_version="1.27.1",
            network_sha256="f00d",
            scenario_route_files=[self.routes],
            **over,
        )

    def model(self, contract_version, **over):
        return PerfectSimWorldModel(
            self.snapshotter,
            SimpleArrivalDraws(),
            self.runner,
            part_a_horizon_s=PART_A,
            contract_version=contract_version,
            scenario_route_files=[self.routes],
            **over,
        )

    def v2_predict(self, samples=None, plans=(PLAN_A,), controller_state=None, **over):
        model = self.v2_model(**over.pop("model", {}))
        out = model.predict_many(
            samples if samples is not None else make_sample(1),
            list(plans),
            controller_state,
            **{**V2_KW, **over},
        )
        return out, model


@pytest.fixture()
def setup(tmp_path):
    return Setup(tmp_path)


# ------------------------------------------------------------- v2 shapes
class TestV2Semantics:
    def test_commit_window_split_and_meta(self, setup):
        setup.runner.wait_s = {plan_key(PLAN_A): 0.5}
        (pred,), _ = setup.v2_predict()
        meta = pred.meta
        assert meta["contract_version"] == "wd01_predict/2"
        assert meta["commit_s"] == COMMIT
        assert meta["continuation"] == "native"
        assert meta["continuation_policy_id"] is None
        assert meta["value_model_id"] == "perfect_sim"
        assert meta["value_model_sha256"] == ["simulator:sumo-1.27.1+f00d"]
        assert meta["safety_window_s"] == COMMIT  # the commit window IS the window
        assert meta["future_min_ttc_validated"] is True
        assert meta["future_min_ttc_validation_id"] == "perfect_sim"
        assert meta["relative_safety_validated"] is True
        assert meta["relative_safety_validation_id"] == "perfect_sim"
        # part_a covers [t0, t0+commit]: 11 steps at wait 0.5 vs native wait 0
        assert pred.delay_part_a_s[0, 0] == pytest.approx((11.0 - 30.0) + 11 * 0.5)
        # part_b is the remainder (10 steps) under the native continuation
        assert pred.delay_part_b_s[0, 0] == pytest.approx(10.0 + 10 * 0.5)
        assert pred.plan_delay_s[0, 0] == pytest.approx(
            pred.delay_part_a_s[0, 0] + pred.delay_part_b_s[0, 0]
        )
        assert pred.delay_vs_native_s[0, 0] == pytest.approx(21.0 * 0.5)

    def test_safety_window_is_commit_not_part_a(self, setup):
        rig = Setup(setup.root)
        rig.runner.gap_m = 8.0
        (pred,), _ = rig.v2_predict()
        # TTC 0.8 s in EVERY step; the window is the 11 commit steps only
        assert pred.min_ttc_s[0, 0] == pytest.approx(0.8)
        assert pred.ttc_lt3_exposure_s[0, 0] == pytest.approx(11.0)
        assert pred.native_ttc_lt3_exposure_s[0, 0] == pytest.approx(11.0)
        # the future window (t0+commit, t0+horizon] holds the other 10 steps
        assert pred.future_ttc_lt3_exposure_s[0, 0] == pytest.approx(10.0)
        assert pred.native_future_ttc_lt3_exposure_s[0, 0] == pytest.approx(10.0)
        assert pred.future_min_ttc_q05_s[0, 0] == pytest.approx(0.8)

    def test_realised_future_heads(self, setup):
        (pred,), _ = setup.v2_predict()
        assert pred.p_future_collision[0, 0] == 0.0
        assert pred.native_p_future_collision[0, 0] == 0.0
        assert pred.future_min_ttc_q05_s[0, 0] == pytest.approx(3.0)  # realised min
        assert bool(pred.future_has_closing_pair[0, 0])
        assert bool(pred.native_future_has_closing_pair[0, 0])
        assert pred.future_hard_brake_s[0, 0] == 0.0
        assert pred.future_cell_valid[0, 0]
        assert pred.native_future_cell_valid[0, 0]

    def test_no_closing_pair_nan_q05(self, setup):
        rig = Setup(setup.root)
        rig.runner.gap_m = None
        (pred,), _ = rig.v2_predict()
        assert np.isnan(pred.min_ttc_s[0, 0])
        assert not bool(pred.has_closing_pair[0, 0])
        assert np.isnan(pred.future_min_ttc_q05_s[0, 0])
        assert not bool(pred.future_has_closing_pair[0, 0])
        # valid cells + NaN q05 WITHOUT a closing pair: contract-clean

    def test_future_wall_heads(self, setup):
        wall = WallSpec(
            wall_id="w0", members=("c0",), speed_mps=16.0, phase="assemble", hold_s=60.0
        )
        plan = PlanCandidate(per_cav={}, walls=(wall,))
        (pred,), _ = setup.v2_predict(plans=(plan,))
        block = pred.walls["w0"]
        assert float(block.p_assembly_completes[0, 0]) == 1.0
        assert block.expected_assembly_time_s[0, 0] == pytest.approx(T0 + 5.0)
        assert float(block.p_assembly_abort[0, 0]) == 0.0

    def test_missing_wall_block_reports_zero_probability(self, setup):
        wall = WallSpec(
            wall_id="w0", members=("c0",), speed_mps=16.0, phase="assemble", hold_s=60.0
        )
        plan = PlanCandidate(per_cav={}, walls=(wall,))
        rig = Setup(setup.root)
        rig.runner.wall_ids = ()  # never assembles
        (pred,), _ = rig.v2_predict(plans=(plan,))
        block = pred.walls["w0"]
        assert float(block.p_assembly_completes[0, 0]) == 0.0
        assert np.isnan(block.expected_assembly_time_s[0, 0])
        assert not block.assembled[0, 0]

    def test_runner_receives_commit_and_continuation(self, setup):
        _, _ = setup.v2_predict()
        for call in setup.runner.calls:
            assert call["commit_s"] == COMMIT
            assert call["continuation"] == "native"
            assert call["safety_window_s"] == COMMIT
            assert call["payload"]["commit_horizon_s"] == 0.0
            assert call["payload"]["strict_roster"] is True


# ------------------------------------------------------------ fail-closed
class TestValidityHeads:
    def test_teleport_in_window_invalidates_cell(self, tmp_path):
        rig = Setup(tmp_path)
        rig.runner.teleport_times = (T0 + 5.0,)
        (pred,), _ = rig.v2_predict()
        assert not bool(pred.cell_valid[0, 0])
        assert not bool(pred.native_cell_valid[0, 0])

    def test_teleport_in_future_invalidates_future_only(self, tmp_path):
        rig = Setup(tmp_path)
        rig.runner.teleport_times = (T0 + 15.0,)
        (pred,), _ = rig.v2_predict()
        assert bool(pred.cell_valid[0, 0])  # window untouched
        assert not bool(pred.future_cell_valid[0, 0])
        assert np.isnan(pred.future_ttc_lt3_exposure_s[0, 0])
        assert np.isnan(pred.p_future_collision[0, 0])

    def test_run_error_invalidates_and_nans_delay(self, tmp_path):
        rig = Setup(tmp_path)
        rig.runner.run_error = True
        (pred,), _ = rig.v2_predict()
        assert np.isnan(pred.plan_delay_s[0, 0])
        assert np.isnan(pred.delay_vs_native_s[0, 0])
        assert not bool(pred.cell_valid[0, 0])
        assert not bool(pred.future_cell_valid[0, 0])

    def test_window_collision_vetoed_by_planner(self, tmp_path):
        rig = Setup(tmp_path)
        rig.runner.__class__ = _CollidingRunner
        (pred,), _ = rig.v2_predict()
        assert pred.collision[0, 0]
        assert pred.meta["early_stop_cells"] == [0]
        assert not bool(pred.cell_valid[0, 0])
        verdict = safety_check(pred, PlannerConfig(mode="controlled_experiment"))
        assert not verdict.ok


# ------------------------------------------------------------ kit bridge
def to_kit_prediction(pred):
    """Planner Prediction -> kit Prediction. The G-2 meta fields are
    planner-side (safety.py reads the meta mapping): the kit Meta predates
    them and rejects unknown keys, so they do not cross the bridge."""
    meta = dict(pred.meta)
    is_v2 = meta["contract_version"] == "wd01_predict/2"
    kit_meta = kit.Meta(
        contract_version=meta["contract_version"],
        model_id=meta["model_id"],
        weights_sha256=tuple(meta["weights_sha256"]),
        sample_ids=tuple(meta["sample_ids"]),
        member_ids=tuple(meta["member_ids"]),
        part_a_horizon_s=float(meta["part_a_horizon_s"]),
        safety_window_s=float(meta["safety_window_s"]),
        horizon_s=float(meta["horizon_s"]),
        native_kept_walls=tuple(meta["native_kept_walls"]),
        native_unlocked_available=bool(meta["native_unlocked_available"]),
        device=str(meta["device"]),
        wall_clock_s=float(meta["wall_clock_s"]),
        commit_s=meta.get("commit_s"),
        continuation=meta.get("continuation"),
        continuation_policy_id=meta.get("continuation_policy_id"),
        value_model_id=meta.get("value_model_id"),
        value_model_sha256=(
            tuple(meta["value_model_sha256"]) if meta.get("value_model_sha256") else None
        ),
        future_min_ttc_validated=bool(meta.get("future_min_ttc_validated", False)),
        future_min_ttc_validation_id=meta.get("future_min_ttc_validation_id"),
    )
    walls = {}
    for wall_id, block in pred.walls.items():
        walls[wall_id] = kit.WallOutcome(
            assembled=np.asarray(block.assembled),
            assembled_at_s=np.asarray(block.assembled_at_s),
            safety_abort=np.asarray(block.safety_abort),
            abort_reason=np.asarray(block.abort_reason),
            **(
                {
                    "p_assembly_completes": np.asarray(block.p_assembly_completes),
                    "expected_assembly_time_s": np.asarray(block.expected_assembly_time_s),
                    "p_assembly_abort": np.asarray(block.p_assembly_abort),
                }
                if is_v2
                else {}
            ),
        )
    optional = {}
    if meta["contract_version"] in ("wd01_predict/1.1", "wd01_predict/2"):
        optional.update(
            {
                "cell_valid": np.asarray(pred.cell_valid),
                "native_cell_valid": np.asarray(pred.native_cell_valid),
                "has_closing_pair": np.asarray(pred.has_closing_pair),
                "native_has_closing_pair": np.asarray(pred.native_has_closing_pair),
            }
        )
    if is_v2:
        optional.update(
            {
                "future_cell_valid": np.asarray(pred.future_cell_valid),
                "native_future_cell_valid": np.asarray(pred.native_future_cell_valid),
                "future_has_closing_pair": np.asarray(pred.future_has_closing_pair),
                "native_future_has_closing_pair": np.asarray(pred.native_future_has_closing_pair),
                "future_ttc_lt3_exposure_s": np.asarray(pred.future_ttc_lt3_exposure_s),
                "future_hard_brake_s": np.asarray(pred.future_hard_brake_s),
                "p_future_collision": np.asarray(pred.p_future_collision),
                "future_min_ttc_q05_s": np.asarray(pred.future_min_ttc_q05_s),
                "native_future_ttc_lt3_exposure_s": np.asarray(
                    pred.native_future_ttc_lt3_exposure_s
                ),
                "native_future_hard_brake_s": np.asarray(pred.native_future_hard_brake_s),
                "native_p_future_collision": np.asarray(pred.native_p_future_collision),
                "native_future_min_ttc_q05_s": np.asarray(pred.native_future_min_ttc_q05_s),
            }
        )
    return kit.Prediction(
        delay_vs_native_s=np.asarray(pred.delay_vs_native_s),
        plan_delay_s=np.asarray(pred.plan_delay_s),
        native_delay_s=np.asarray(pred.native_delay_s),
        delay_part_a_s=np.asarray(pred.delay_part_a_s),
        delay_part_b_s=np.asarray(pred.delay_part_b_s),
        collision=np.asarray(pred.collision),
        min_ttc_s=np.asarray(pred.min_ttc_s),
        ttc_lt3_exposure_s=np.asarray(pred.ttc_lt3_exposure_s),
        hard_brake_s=np.asarray(pred.hard_brake_s),
        native_collision=np.asarray(pred.native_collision),
        native_min_ttc_s=np.asarray(pred.native_min_ttc_s),
        native_ttc_lt3_exposure_s=np.asarray(pred.native_ttc_lt3_exposure_s),
        native_hard_brake_s=np.asarray(pred.native_hard_brake_s),
        trajectories_tracked=dict(pred.trajectories_tracked),
        native_trajectories_tracked=dict(pred.native_trajectories_tracked),
        walls=walls,
        meta=kit_meta,
        **optional,
    )


class TestKitValidation:
    def test_v2_output_passes_kit_validator(self, setup):
        (pred,), _ = setup.v2_predict()
        kit_pred = to_kit_prediction(pred)
        # trajectory_s=None: T derives from meta.commit_s (validator section 2)
        kit_validate.validate_prediction(kit_pred, M=1, K=1, n_plans=1, trajectory_s=None)
        assert kit_pred.meta.contract_version == "wd01_predict/2"

    def test_v2_two_plans_two_samples_pass_kit(self, setup):
        setup.runner.wait_s = {plan_key(PLAN_A): 0.5}
        model = setup.v2_model()
        out = model.predict_many(make_sample(2), [PLAN_A, PLAN_B], None, **V2_KW)
        for pred in out:
            kit_validate.validate_prediction(
                to_kit_prediction(pred), M=1, K=2, n_plans=2, trajectory_s=None
            )

    def test_v1_1_output_passes_kit_validator(self, tmp_path):
        rig = Setup(tmp_path)
        (pred,) = rig.model("wd01_predict/1.1").predict_many(
            make_sample(1), [PLAN_A], None, horizon_s=HORIZON
        )
        assert pred.meta["contract_version"] == "wd01_predict/1.1"
        assert pred.meta["safety_window_s"] == PART_A
        kit_validate.validate_prediction(
            to_kit_prediction(pred), M=1, K=1, n_plans=1, trajectory_s=None
        )

    def test_v1_output_still_finite_and_kit_valid(self, tmp_path):
        rig = Setup(tmp_path)
        (pred,) = rig.model("wd01_predict/1").predict_many(
            make_sample(1), [PLAN_A], None, horizon_s=HORIZON
        )
        assert pred.meta["contract_version"] == "wd01_predict/1"
        assert np.isfinite(pred.plan_delay_s).all()
        kit_validate.validate_prediction(
            to_kit_prediction(pred), M=1, K=1, n_plans=1, trajectory_s=PART_A
        )

    def test_kit_state_sample_feeds_the_model(self, setup):
        """A kit StateSample (dataclass meta, ArrivalDraw arrivals) drives the
        model end to end via the declared K=1 restored-RNG mode."""
        sample = kit.StateSample(
            hybrid={"note": "opaque wm30 hybrid state"},
            arrivals=kit.ArrivalDraw(demand_model_id="dm", seed=7),
            meta=kit.SampleMeta(sample_id="k0", time_s=T0, merge_observed=True, coverage=1.0),
        )
        kit_validate.validate_state_samples([sample])
        (pred,), _ = setup.v2_predict(samples=[sample])
        kit_validate.validate_prediction(to_kit_prediction(pred), M=1, K=1, n_plans=1)
        assert len(setup.runner.calls) == 2  # native reference + the plan
        for call in setup.runner.calls:
            assert call["draw_id"] == "restored-rng:dm:7"
            assert call["departures"] == ()  # nothing explicit: the snapshot RNG


# --------------------------------------------------------- planner gates
class TestPlannerGates:
    def test_live_mode_gate_and_safety_accept_v2(self, setup):
        setup.runner.wait_s = {plan_key(PLAN_A): 0.5}
        (pred,), _ = setup.v2_predict()
        config = PlannerConfig(mode="live")
        assert enforce_future_gate(pred, config) is None  # G-1 AND G-2 pass
        verdict = safety_check(pred, config)
        assert verdict.ok
        assert verdict.future_ok
        assert not verdict.p_future_collision_veto
        assert not verdict.g1_future_floor_veto
        assert not verdict.window_relative_veto

    def test_g1_floor_applies_to_validated_q05(self, tmp_path):
        # TTC 0.8 s everywhere: the future q05 (0.8) is below the 1.5 floor
        # and the head IS validated -> the G-1 conditional future floor vetoes
        rig = Setup(tmp_path)
        rig.runner.gap_m = 8.0
        (pred,), _ = rig.v2_predict()
        # pinned to rule A: under rule C plan == native (both 0.8 s) is admitted
        config = PlannerConfig(mode="live", absolute_ttc_rule="absolute")
        assert enforce_future_gate(pred, config) is None  # the MODE gate passes
        verdict = safety_check(pred, config)
        assert verdict.g1_future_floor_veto
        assert not verdict.ok

    def test_unvalidated_relative_safety_refused_in_live_mode(self, setup):
        from traffic_shepherd.planner.interface import Prediction

        (pred,), _ = setup.v2_predict()
        stripped = Prediction(
            **{
                **{
                    f.name: getattr(pred, f.name)
                    for f in dataclasses.fields(Prediction)
                    if f.name != "meta"
                },
                "meta": {
                    **pred.meta,
                    "relative_safety_validated": False,
                    "relative_safety_validation_id": None,
                },
            }
        )
        with pytest.raises(FutureSafetyRefusal, match="relative-safety"):
            enforce_future_gate(stripped, PlannerConfig(mode="live"))


# ------------------------------------------------- continuation refusal
class TestPlannerContinuationRefusal:
    def test_continuation_planner_raises_naming_pl04(self, setup):
        model = setup.v2_model()
        with pytest.raises(NotImplementedError, match="PL-04"):
            model.predict_many(
                make_sample(1),
                [PLAN_A],
                None,
                horizon_s=HORIZON,
                commit_s=COMMIT,
                continuation="planner",
            )
        assert setup.runner.calls == []  # nothing ran: refusal precedes branching
        assert setup.api.simulation.saved == []  # no snapshot either

    def test_unknown_continuation_rejected(self, setup):
        model = setup.v2_model()
        with pytest.raises(ValueError, match="continuation"):
            model.predict_many(
                make_sample(1),
                [PLAN_A],
                None,
                horizon_s=HORIZON,
                commit_s=COMMIT,
                continuation="oracle",
            )

    def test_commit_window_validation(self, setup):
        model = setup.v2_model()
        with pytest.raises(ValueError, match="part_a_horizon_s"):
            model.predict_many(
                make_sample(1),
                [PLAN_A],
                None,
                horizon_s=HORIZON,
                commit_s=25.0,
                continuation="native",
            )
        with pytest.raises(ValueError, match="commit_s"):
            model.predict_many(
                make_sample(1),
                [PLAN_A],
                None,
                horizon_s=HORIZON,
                commit_s=0.0,
                continuation="native",
            )
        with pytest.raises(ValueError, match="safety window"):
            model.predict_many(
                make_sample(1),
                [PLAN_A],
                None,
                horizon_s=5.0,
                commit_s=COMMIT,
                continuation="native",
            )


# --------------------------------------------------------------- N1 / CRN
class TestN1AndCRN:
    def test_n1_assemble_kept_hold_dissolved_v2(self, tmp_path):
        for phase, expected in (("assemble", ["w0"]), ("hold", []), ("release", [])):
            rig = Setup(tmp_path / phase)  # per-phase root: no decision collision
            state = {
                "c0": {
                    "action": "wall",
                    "wall_id": "w0",
                    "wall_phase": phase,
                    "locked": True,
                    "time_in_phase_s": 1.0,
                    "wall_speed_mps": 16.0,
                }
            }
            (pred,), _ = rig.v2_predict(controller_state=state)
            assert pred.meta["native_kept_walls"] == expected
            for call in rig.runner.calls:
                if call["plan"] == "native":
                    assert call["payload"]["assignments"] == {"c0": "native"}

    def test_crn_identical_draws_across_plans_and_native(self, setup):
        model = setup.v2_model()
        model.predict_many(make_sample(2), [PLAN_A, PLAN_B], None, **V2_KW)
        by_plan: dict = {}
        for call in setup.runner.calls:
            by_plan.setdefault(call["plan"], []).append(call)
        assert set(by_plan) == {"native", plan_key(PLAN_A), plan_key(PLAN_B)}
        for calls in by_plan.values():
            assert [c["draw_id"] for c in calls] == ["k0", "k1"]
            assert all(c["departures"] != () for c in calls)
        cross = {
            tuple((c["draw_id"], c["departures"]) for c in calls) for calls in by_plan.values()
        }
        assert len(cross) == 1  # CRN for each k

    def test_snapshot_taken_once_per_call(self, setup):
        model = setup.v2_model()
        model.predict_many(make_sample(2), [PLAN_A, PLAN_B], None, **V2_KW)
        assert len(setup.runner.calls) == 6  # (native + 2 plans) x K=2
        assert len(setup.api.simulation.saved) == 1  # one t0 snapshot
        # PL-10d: one FRESH process per branch (native included), each
        # closed exactly once, each seeing EXACTLY ONE loadState; the
        # master api never restores (it only saves)
        factory = setup.factory
        assert len(factory.started) == 6
        assert factory.closed == factory.started
        assert [c["connection"] for c in setup.runner.calls] == factory.started
        for connection in factory.started:
            assert len(connection.simulation.loaded) == 1
        assert setup.api.simulation.loaded == []


# ------------------------------------------------------- snapshot guard
class TestSnapshotGuard:
    def _snapshotter(self, tmp_path, text, **over):
        api = fake_api(text)
        over.setdefault("wm34_output_root", str(tmp_path / "no-such-wm34-root"))
        return TraciStateSnapshotter(api, tmp_path / "root", **over), api

    def test_good_snapshot_saves_and_restores(self, tmp_path):
        snapshotter, _ = self._snapshotter(tmp_path, GOOD_STATE)
        manifest = snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
        assert manifest["manifest"]["schema"] == SCHEMA
        assert manifest["manifest"]["state_options"]["rng_state_in_file"] is True
        assert manifest["manifest"]["state_options"]["save_state_precision_decimals"] > 2
        branch = fake_api(GOOD_STATE)
        snapshotter.restore(manifest, plan=PLAN_A, draw_id="k0", api=branch)
        assert branch.simulation.loaded  # loadState ran on the BRANCH process

    def test_low_precision_snapshot_refused(self, tmp_path):
        snapshotter, _ = self._snapshotter(tmp_path, LOW_PRECISION_STATE)
        with pytest.raises(StateSnapshotError, match="precision"):
            snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])

    def test_no_rng_snapshot_refused(self, tmp_path):
        snapshotter, _ = self._snapshotter(tmp_path, NO_RNG_STATE)
        with pytest.raises(StateSnapshotError, match="rngState"):
            snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])

    def test_guard_direct(self, tmp_path):
        path = tmp_path / "state.xml"
        path.write_text(LOW_PRECISION_STATE, encoding="utf-8")
        check = validate_state_file_for_exactness(path)
        assert check["has_rng"] and check["max_pos_decimals"] == 2 and check["problems"]

    def test_restore_refuses_doctored_manifest(self, tmp_path):
        snapshotter, _ = self._snapshotter(tmp_path, GOOD_STATE)
        manifest = snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
        manifest["manifest"]["state_options"]["rng_state_in_file"] = False
        with pytest.raises(StateSnapshotError, match="rng"):
            snapshotter.restore(manifest, plan=PLAN_A, draw_id="k0", api=fake_api())

    def test_restore_refuses_digest_mismatch(self, tmp_path):
        snapshotter, _ = self._snapshotter(tmp_path, GOOD_STATE)
        manifest = snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
        state_file = Path(manifest["saved_state_dir"]) / "state.xml"
        state_file.write_text(LOW_PRECISION_STATE, encoding="utf-8")  # tampered
        with pytest.raises(StateSnapshotError):
            snapshotter.restore(manifest, plan=PLAN_A, draw_id="k0", api=fake_api())

    def test_restore_refuses_wrong_branch_time(self, tmp_path):
        snapshotter, _ = self._snapshotter(tmp_path, GOOD_STATE)
        manifest = snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
        manifest["t0_s"] = T0 + 1.0
        with pytest.raises(StateSnapshotError, match="branch time"):
            snapshotter.restore(manifest, plan=PLAN_A, draw_id="k0", api=fake_api())

    @pytest.mark.skipif(not Path(WM34_REAL_ROOT).is_dir(), reason="wm34 sibling tree not present")
    def test_real_saved_state_module_reuse(self, tmp_path):
        """The REAL campaign_pilot_v1.saved_state owns the save when the
        read-only wm34 tree is importable (read-only import, not vendored)."""
        snapshotter, _ = self._snapshotter(tmp_path, GOOD_STATE, wm34_output_root=WM34_REAL_ROOT)
        manifest = snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
        assert manifest["manifest"]["implementation"] == "campaign_pilot_v1.saved_state"
        assert manifest["manifest"]["schema"] == SCHEMA
        branch = fake_api(GOOD_STATE)
        snapshotter.restore(manifest, plan=PLAN_A, draw_id="k0", api=branch)
        assert branch.simulation.loaded


# ---------------------------------------------------------- v1 unchanged
class TestV1RegressionWithinV2File:
    def test_v1_mode_ignores_commit_and_continuation(self, tmp_path):
        rig = Setup(tmp_path)
        (pred,) = rig.model("wd01_predict/1").predict_many(
            make_sample(1),
            [PLAN_A],
            None,
            horizon_s=HORIZON,
            commit_s=COMMIT,  # ignored on v1
            continuation="planner",  # ignored (and NOT a refusal) on v1
        )
        assert pred.meta["contract_version"] == "wd01_predict/1"
        assert pred.meta["safety_window_s"] == PART_A
        assert np.isfinite(pred.plan_delay_s).all()
