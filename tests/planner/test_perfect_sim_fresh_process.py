"""PL-10d: ONE fresh SUMO process per restored branch (fake-tested).

Evidence (re-smoke, output/planner_v1/resmoke/RESMOKE_REPORT.md): the FIRST
restore into a fresh SUMO process was bit-exact vs the uninterrupted run,
while a native branch restored in the SAME process after a controlled
(damping) branch diverged at t+5 s with max dpos/dspeed identical to every
digit to the plan branch's -- simulator state carries over across loadState.
The contract under test:

- PerfectSimWorldModel.predict_many drives ONE fresh process per
  (plan, draw) branch, native reference included, via the runner's
  BranchProcessFactory (start_process -> connection; close_process after
  the branch, exceptions included);
- each branch process sees EXACTLY ONE loadState (the master process that
  saved the snapshot never restores);
- a second restore on a connection raises ProcessReuseError BEFORE
  loadState is invoked;
- the factory/launcher abstraction is fake: no SUMO/traci/libsumo import at
  module import time.

Zero simulation: every traci surface here is a fake.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from traffic_shepherd.planner import PlanCandidate
from traffic_shepherd.planner.perfect_sim import (
    NATIVE_PLAN,
    MissingControlWriteError,
    PerfectSimWorldModel,
    SimpleArrivalDraws,
    TraciBranchRunner,
    TraciStateSnapshotter,
)
from traffic_shepherd.planner.perfect_sim_state import (
    ProcessReuseError,
    StateSnapshotError,
)

T0 = 40.0
COMMIT = 10.0
HORIZON = 20.0
PART_A = 20.0  # model part_a_horizon_s (v2: commit_s <= part_a_horizon_s)
DT = 0.5

GOOD_STATE = (
    '<state><rngState seed="7"/><vehicle id="c0" pos="12.34567890123456789" speed="20.0"/></state>'
)

V2_KW = dict(horizon_s=HORIZON, commit_s=COMMIT, continuation="native")

PLAN_A = PlanCandidate(per_cav={"c0": "damping"})
PLAN_B = PlanCandidate(per_cav={"c0": "native"})


# ------------------------------------------------------------------ fakes
class FakeBranchSim:
    """Per-process fake traci.simulation: loadState recorded, time advances
    only when THIS process steps (branches are independent processes)."""

    def __init__(self, t0, dt):
        self._time = float(t0)
        self._dt = float(dt)
        self.loaded = []

    def loadState(self, path):
        self.loaded.append(str(path))

    def getTime(self):
        return self._time

    def simulationStep(self):
        self._time += self._dt

    def getArrivedIDList(self):
        return []  # T-B: the control write reads arrivals (run_suffix lifecycle)

    def getStartingTeleportIDList(self):
        return ()

    def getCollidingVehiclesIDList(self):
        return ()


class FakeBranchVehicle:
    def __init__(self):
        self.added = []

    def add(self, rid, route, **kw):
        self.added.append((str(rid), str(route), kw))

    def getIDList(self):
        return ("c0",)

    def getLeader(self, rid, range_m):
        return ("lead", 30.0)

    def getLanePosition(self, rid):
        return 12.5

    def getLaneIndex(self, rid):
        return 0

    def getSpeed(self, rid):
        return 20.0

    def getAcceleration(self, rid):
        return 0.0


class FakeBranchConnection:
    """One fake SUMO process: fresh physics, its OWN loadState ledger.

    Shaped like the real traci module surface the runner uses:
    ``connection.simulationStep()`` at the top level, plus
    ``connection.simulation.*`` and ``connection.vehicle.*``."""

    def __init__(self, label=None, t0=T0, dt=DT):
        self.label = label
        self.simulation = FakeBranchSim(t0, dt)
        self.vehicle = FakeBranchVehicle()

    def simulationStep(self):
        self.simulation.simulationStep()


class FreshProcessFactory:
    """PL-10d fake launcher: one FakeBranchConnection per start_process."""

    def __init__(self, t0=T0):
        self.started = []
        self.closed = []
        self.labels = []
        self._t0 = float(t0)

    def start_process(self, *, label=None):
        connection = FakeBranchConnection(label=label, t0=self._t0)
        self.started.append(connection)
        self.labels.append(label)
        return connection

    def close_process(self, connection):
        assert any(connection is c for c in self.started), "never started"
        assert not any(connection is c for c in self.closed), "closed twice"
        self.closed.append(connection)


class BombFactory(FreshProcessFactory):
    """Poisons the N-th started branch's process: its physics step raises
    mid-run, so the world model must still retire that process."""

    def __init__(self, fail_index, t0=T0):
        super().__init__(t0=t0)
        self._fail_index = int(fail_index)

    def start_process(self, *, label=None):
        connection = super().start_process(label=label)
        if len(self.started) - 1 == self._fail_index:

            def explode():
                raise RuntimeError("branch exploded mid-run")

            connection.simulation.simulationStep = explode
        return connection


class MasterSim:
    """traci.simulation of the master process: saveState writes the real
    state file (the exactness guard reads it back from disk)."""

    def __init__(self, state_text):
        self._state_text = state_text
        self.saved = []
        self.loaded = []

    def saveState(self, path):
        Path(path).write_text(self._state_text, encoding="utf-8")
        self.saved.append(str(path))

    def loadState(self, path):
        self.loaded.append(str(path))


class MasterApi:
    """The snapshotter's master process: saveState only. PL-10d says the
    master NEVER restores (it keeps a loaded ledger to prove that)."""

    def __init__(self, state_text=GOOD_STATE):
        self.simulation = MasterSim(state_text)

    @property
    def saved(self):
        return self.simulation.saved

    @property
    def loaded(self):
        return self.simulation.loaded


class FakeDispatcherSession:
    """apply_plan sink (the WM-40 dispatcher stand-in); no wall records."""

    def __init__(self):
        self.applied = []

    wall_records = ()

    def apply_plan(self, plan, now_s=0.0):
        self.applied.append((plan, float(now_s)))


class FakeExecPlan:
    """plan_factory product: the runner only hands it to the session."""

    def __init__(self, plan_id):
        self.plan_id = plan_id


def _no_control_write(session, now):
    """PL-04c fail-closed opt-out: this rig's fake plan factory writes no
    control, and the runner now refuses a non-native branch (PLAN_A is a
    damping plan) without a control write -- the stub declares one."""
    del session, now


class Rig:
    """Full-stack rig: REAL TraciStateSnapshotter + TraciBranchRunner over
    fake processes, driven by the REAL PerfectSimWorldModel (v2)."""

    def __init__(self, root: Path, factory=None, **model_over):
        self.master = MasterApi()
        self.snapshotter = TraciStateSnapshotter(
            self.master,
            root / "state_root",
            wm34_output_root=str(root / "no-such-wm34-root (vendored)"),
        )
        self.factory = factory or FreshProcessFactory(t0=T0)
        self.dispatcher = FakeDispatcherSession()
        self.runner = TraciBranchRunner(
            self.master,
            snapshotter=self.snapshotter,
            cav_ids=("c0",),
            dispatcher_factory=lambda controller_state, kept: self.dispatcher,
            plan_factory=lambda plan, plan_id, controller_state, kept: FakeExecPlan(plan_id),
            tick_hook=_no_control_write,
            process_factory=self.factory,
        )
        self.model = PerfectSimWorldModel(
            self.snapshotter,
            SimpleArrivalDraws(),
            self.runner,
            part_a_horizon_s=PART_A,
            contract_version="wd01_predict/2",
            model_id="perfect_sim/fresh-process-test",
            sumo_version="1.27.1",
            network_sha256="f00d",
            **model_over,
        )


def make_sample(k=2):
    """K samples with explicit option-A departures (exactness guard passes
    with no declared scenario)."""
    return [
        SimpleNamespace(
            hybrid={"roster": ["c0"]},
            arrivals={
                "sample_id": f"k{i}",
                "departures": ((T0 + 5.0 + i, "main", 20.0, False, "passenger"),),
            },
            meta={
                "sample_id": f"k{i}",
                "time_s": T0,
                "merge_observed": True,
                "coverage": 1.0,
            },
        )
        for i in range(k)
    ]


# ------------------------------------------------- one process per branch
class TestOneFreshProcessPerBranch:
    def test_predict_many_opens_one_process_per_branch_including_native(self, tmp_path):
        rig = Rig(tmp_path)
        rig.model.predict_many(make_sample(2), [PLAN_A, PLAN_B], None, **V2_KW)
        # (native reference + 2 plans) x K=2 draws = 6 branches = 6 processes
        assert len(rig.factory.started) == 6
        assert len({id(c) for c in rig.factory.started}) == 6  # every one fresh
        # the native reference is a branch like any other: own process per k
        assert rig.factory.labels[0] == "native:k0"
        assert sorted(lbl for lbl in rig.factory.labels if lbl.startswith("native")) == [
            "native:k0",
            "native:k1",
        ]
        assert rig.factory.closed == rig.factory.started  # all closed, once

    def test_each_process_loads_state_once_and_master_never_loads(self, tmp_path):
        rig = Rig(tmp_path)
        rig.model.predict_many(make_sample(2), [PLAN_A], None, **V2_KW)
        for connection in rig.factory.started:
            assert len(connection.simulation.loaded) == 1  # EXACTLY ONE loadState
            assert connection.simulation.getTime() > T0  # the branch RAN there
            assert connection.vehicle.added  # the draw installed on THIS process
        assert len(rig.master.saved) == 1  # the master saved the snapshot ...
        assert rig.master.loaded == []  # ... and NEVER restored

    def test_fresh_connection_after_spent_one_still_restores(self, tmp_path):
        """Positive control: the ledger refuses only the SAME connection --
        the next branch's fresh process restores fine."""
        rig = Rig(tmp_path)
        manifest = rig.snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
        first = rig.factory.start_process(label="b0")
        rig.snapshotter.restore(manifest, plan=PLAN_A, draw_id="k0", api=first)
        second = rig.factory.start_process(label="b1")
        rig.snapshotter.restore(manifest, plan=PLAN_B, draw_id="k0", api=second)
        assert len(first.simulation.loaded) == 1
        assert len(second.simulation.loaded) == 1

    def test_screening_rescore_stage_also_opens_fresh_processes(self, tmp_path):
        rig = Rig(tmp_path, screen_horizon_s=15.0, rescore_top_n=1, rescore_always=(0,))
        rig.model.predict_many(make_sample(1), [PLAN_A, PLAN_B], None, **V2_KW)
        # screen stage: (native + 2 plans) x K=1 = 3 processes; rescore:
        # native + top-1 = 2 more; every one closed exactly once
        assert len(rig.factory.started) == 5
        assert rig.factory.closed == rig.factory.started
        assert all(len(c.simulation.loaded) == 1 for c in rig.factory.started)


# -------------------------------------------------------- reuse refusal
class TestReuseRefusal:
    def _spent_connection(self, tmp_path):
        rig = Rig(tmp_path)
        manifest = rig.snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
        connection = rig.factory.start_process(label="branch")
        rig.snapshotter.restore(manifest, plan=PLAN_A, draw_id="k0", api=connection)
        return rig, manifest, connection

    def test_second_restore_refused_before_load_state(self, tmp_path):
        rig, manifest, connection = self._spent_connection(tmp_path)
        assert len(connection.simulation.loaded) == 1  # the first (legal) load
        with pytest.raises(ProcessReuseError, match="PL-10d"):
            rig.snapshotter.restore(manifest, plan=PLAN_B, draw_id="k1", api=connection)
        assert len(connection.simulation.loaded) == 1  # NO second loadState

    def test_refusal_applies_after_a_native_branch_too(self, tmp_path):
        rig = Rig(tmp_path)
        rig.model.predict_many(make_sample(1), [], None, **V2_KW)  # native only
        native_connection = rig.factory.started[0]
        assert "native" in str(native_connection.label)
        assert len(native_connection.simulation.loaded) == 1  # the native branch ran
        manifest = rig.snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
        with pytest.raises(ProcessReuseError, match="EXACTLY ONE"):
            rig.snapshotter.restore(manifest, plan=NATIVE_PLAN, draw_id="k0", api=native_connection)
        assert len(native_connection.simulation.loaded) == 1

    def test_process_reuse_error_is_in_the_saved_state_error_family(self):
        assert issubclass(ProcessReuseError, StateSnapshotError)


# ------------------------------------------------------- process closing
class TestProcessLifecycle:
    def test_every_process_closed_after_its_branch(self, tmp_path):
        rig = Rig(tmp_path)
        rig.model.predict_many(make_sample(2), [PLAN_A, PLAN_B], None, **V2_KW)
        assert rig.factory.closed == rig.factory.started  # 6/6, none twice

    def test_process_closed_even_when_the_branch_raises(self, tmp_path):
        rig = Rig(tmp_path, factory=BombFactory(fail_index=2))  # plan1@k0
        with pytest.raises(RuntimeError, match="branch exploded"):
            rig.model.predict_many(make_sample(2), [PLAN_A, PLAN_B], None, **V2_KW)
        # the poisoned process AND both earlier ones: started -> closed
        assert len(rig.factory.started) == 3
        assert rig.factory.closed == rig.factory.started


# ------------------------------------------------------ factory required
class TestFactoryRequired:
    def test_runner_without_factory_refused_before_any_snapshot(self, tmp_path):
        rig = Rig(tmp_path)
        rig.runner.process_factory = None
        with pytest.raises(NotImplementedError, match="PL-10d"):
            rig.model.predict_many(make_sample(1), [PLAN_A], None, **V2_KW)
        assert rig.master.saved == []  # refused BEFORE the t0 snapshot ...
        assert rig.factory.started == []  # ... and before any process

    def test_start_process_without_factory_names_the_contract(self, tmp_path):
        runner = TraciBranchRunner(
            MasterApi(), snapshotter=None, cav_ids=("c0",)
        )  # no process_factory
        with pytest.raises(NotImplementedError, match="PL-10d"):
            runner.start_process()

    def test_branch_runner_run_requires_a_connection(self, tmp_path):
        """The production runner has NO in-process fallback: run() without
        the fresh connection is a refusal before any restore."""
        rig = Rig(tmp_path)
        manifest = rig.snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
        with pytest.raises(NotImplementedError, match="fresh"):
            rig.runner.run(
                PLAN_A,
                connection=None,
                manifest=manifest,
                draw_id="k0",
                departures=(),
                horizon_s=HORIZON,
                safety_window_s=COMMIT,
                controller_state={},
                kept_wall_ids=[],
                commit_s=COMMIT,
                continuation="native",
            )
        assert rig.factory.started == []


# ------------------------------------------- PL-04c fail-closed control write
class TestMissingControlWriteFailClosed:
    """PL-04c: a branch that assigns any non-native action or wall refuses
    to run WITHOUT a control write -- BEFORE the restore (no loadState is
    spent on a branch that would silently simulate as native). A
    native-only plan may run bare."""

    @staticmethod
    def _bare_runner(rig):
        """A production runner with NO control write (no tick_hook)."""
        return TraciBranchRunner(
            rig.master,
            snapshotter=rig.snapshotter,
            cav_ids=("c0",),
            dispatcher_factory=lambda controller_state, kept: rig.dispatcher,
            plan_factory=lambda plan, plan_id, controller_state, kept: FakeExecPlan(plan_id),
            process_factory=rig.factory,
        )

    @staticmethod
    def _run_kw(manifest, **over):
        kw = dict(
            manifest=manifest,
            draw_id="k0",
            departures=(),
            horizon_s=HORIZON,
            safety_window_s=COMMIT,
            controller_state={},
            kept_wall_ids=[],
            commit_s=COMMIT,
            continuation="native",
        )
        kw.update(over)
        return kw

    def test_damping_plan_without_tick_hook_refused_before_restore(self, tmp_path):
        rig = Rig(tmp_path)
        runner = self._bare_runner(rig)
        manifest = rig.snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
        connection = rig.factory.start_process(label="bare")
        with pytest.raises(MissingControlWriteError, match="control write"):
            runner.run(PLAN_A, connection=connection, **self._run_kw(manifest))
        # fail closed BEFORE restore/loadState: the process is unspent
        assert connection.simulation.loaded == []

    def test_native_only_plan_runs_without_a_tick_hook(self, tmp_path):
        rig = Rig(tmp_path)
        runner = self._bare_runner(rig)
        manifest = rig.snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
        connection = rig.factory.start_process(label="bare_native")
        record = runner.run(NATIVE_PLAN, connection=connection, **self._run_kw(manifest))
        # the all-native reference (no kept walls) needs no control write
        assert record.plan_label.startswith("native")
        assert record.steps  # the branch RAN (bare) and reported telemetry
        assert len(connection.simulation.loaded) == 1

    def test_native_plan_with_kept_walls_without_tick_hook_refused(self, tmp_path):
        """The N1 reference continues its kept walls at the staged speed --
        wall-v2 members are control writes too, so bare is refused."""
        rig = Rig(tmp_path)
        runner = self._bare_runner(rig)
        manifest = rig.snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
        connection = rig.factory.start_process(label="bare_kept_walls")
        with pytest.raises(MissingControlWriteError, match="non-native actions or walls"):
            runner.run(
                NATIVE_PLAN,
                connection=connection,
                **self._run_kw(manifest, kept_wall_ids=["w1"]),
            )
        assert connection.simulation.loaded == []

    def test_wall_plan_without_tick_hook_refused(self, tmp_path):
        rig = Rig(tmp_path)
        runner = self._bare_runner(rig)
        manifest = rig.snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
        wall_plan = SimpleNamespace(
            label="wallplan", per_cav={"c0": "native"}, walls=(object(),)
        )
        connection = rig.factory.start_process(label="bare_wall")
        with pytest.raises(MissingControlWriteError, match="silently simulate as native"):
            runner.run(wall_plan, connection=connection, **self._run_kw(manifest))
        assert connection.simulation.loaded == []

    def test_both_control_write_injections_refused_at_construction(self):
        with pytest.raises(ValueError, match="ONE control write"):
            TraciBranchRunner(
                MasterApi(),
                snapshotter=None,
                cav_ids=("c0",),
                tick_hook=lambda session, now: None,
                tick_hook_factory=lambda connection, session, plan: None,
            )

    def test_factory_returning_a_non_callable_refused(self, tmp_path):
        rig = Rig(tmp_path)
        runner = TraciBranchRunner(
            rig.master,
            snapshotter=rig.snapshotter,
            cav_ids=("c0",),
            dispatcher_factory=lambda controller_state, kept: rig.dispatcher,
            plan_factory=lambda plan, plan_id, controller_state, kept: FakeExecPlan(plan_id),
            tick_hook_factory=lambda connection, session, plan: None,
            process_factory=rig.factory,
        )
        manifest = rig.snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
        connection = rig.factory.start_process(label="bare_bad_factory")
        with pytest.raises(MissingControlWriteError, match="per-branch control"):
            runner.run(PLAN_A, connection=connection, **self._run_kw(manifest))
        # the factory failure is not silently skipped: the branch refuses
        # rather than simulating as native (the restore already landed)
        assert len(connection.simulation.loaded) == 1
