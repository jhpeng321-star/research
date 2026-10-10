"""PL-04l: master/branch traci connection routing (the cl_smoke_2 defect).

Real traci's ``init(label=...)`` defaults to ``doSwitch=True``: it SWITCHES
the module-level current connection, and every module-level call (the
``simulation``/``vehicle``/... domains, ``simulationStep``, ``close``) then
acts on the CURRENT connection. The first real closed-loop smoke
(cl_smoke_2.log) died at the SECOND ``predict_many`` of the first decision
with ``FatalTraCIError: Connection already closed``:

- the master harness held the traci MODULE as its api;
- the first lookahead branch's ``traci.init(label=...)`` had switched the
  module to that branch (doSwitch default True);
- the branch closed (its own Connection object);
- the master's next ``saveState`` went through the module to the DEAD branch
  connection.

The PL-04i e2e fakes are upgraded to model this exactly
(:class:`tests.planner.test_pl04i_e2e_real_run._TinyTraci`): ``init``
switches when ``doSwitch``, module-level surfaces route to the current
connection, ``getConnection(label)`` returns the labelled one, and any call
on a closed connection raises ``FatalTraCIError('Connection already
closed')``. On the pre-PL-04l code the e2e below reproduces the exact
cl_smoke_2 symptom; on the fixed code the master holds and passes ITS OWN
Connection object everywhere, and branch launches use
``init(..., doSwitch=False)`` + their own connection only.
"""

import sys
from types import SimpleNamespace

import pytest

from tests.planner import test_pl04i_e2e_real_run as e2e_mod
from traffic_shepherd.planner.closed_loop import (
    ClosedLoopConfig,
    ClosedLoopRunner,
    build_real_components,
    resolve_base_run,
)

MASTER_LABEL = "pl04"


def _planner_e2e(tmp_path, monkeypatch, *, run_id="pl04l"):
    """One REAL-wiring planner e2e (the PL-04j near-merge fixture: the
    planner decides with >= 2 candidate plans at the first decision). The
    runner's own harness/model are captured for the identity assertions."""
    base = e2e_mod.make_base_run_pl04j(tmp_path / "fixture")
    resolved = resolve_base_run(base)
    monkeypatch.setenv("SUMO_BINARY", sys.executable)
    monkeypatch.delenv("SUMO_HOME", raising=False)
    traci = e2e_mod._TinyTraci(lane=e2e_mod.PRE_MERGE_LANE)
    monkeypatch.setitem(sys.modules, "traci", traci)
    components = build_real_components(e2e_mod._real_args(base))
    made = {}
    real_harness = components.make_harness
    real_model = components.make_world_model

    def recording_harness(context):
        made["harness"] = real_harness(context)
        return made["harness"]

    def recording_model(context):
        made["model"] = real_model(context)
        return made["model"]

    components.make_harness = recording_harness
    components.make_world_model = recording_model
    config = ClosedLoopConfig(
        out_root=tmp_path,
        run_id=run_id,
        world="w_pl04l",
        seed=e2e_mod.SEED,
        horizon_s=10.0,
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
    )
    return SimpleNamespace(
        base=base,
        traci=traci,
        components=components,
        config=config,
        run_dir=tmp_path / run_id,
        made=made,
    )


def _branch_labels(traci):
    return [c["label"] for c in traci.init_calls if c["label"] != MASTER_LABEL]


# ------------------------------------------------ the fake IS real traci
class TestFakeModelsRealTraciModuleSwitching:
    """Unit pins of the module semantics the defect depended on: a fake that
    does not switch would hide the routing defect again (that is exactly why
    the pre-PL-04l fakes passed while the real run died)."""

    def test_init_with_doswitch_switches_the_module_surface(self):
        traci = e2e_mod._TinyTraci()
        traci.init(label="master", port=1)
        traci.init(label="branch", port=2)  # doSwitch default True
        assert traci.current_label == "branch"
        traci.simulation.getTime()
        traci.vehicle.getIDList()
        traci.simulationStep()
        assert traci.module_calls == [
            "simulation", "vehicle", "simulationStep",
        ]
        # the module stepped the CURRENT (branch) connection's world
        assert traci.connections["master"].world.time == 0.0
        assert traci.connections["branch"].world.time == 0.5

    def test_init_without_doswitch_leaves_the_current_connection_alone(self):
        traci = e2e_mod._TinyTraci()
        traci.init(label="master", port=1, doSwitch=True)
        traci.init(label="branch", port=2, doSwitch=False)
        assert traci.current_label == "master"
        traci.simulation.getTime()
        assert traci.connections["master"].calls == [
            "simulation.getTime",
        ]

    def test_getconnection_returns_the_labelled_connection(self):
        traci = e2e_mod._TinyTraci()
        traci.init(label="a", port=1)
        traci.init(label="b", port=2)
        assert traci.getConnection("a") is traci.connections["a"]
        assert traci.getConnection("b") is traci.connections["b"]
        assert traci.getConnection() is traci.connections["b"]  # current

    def test_module_close_acts_on_the_current_connection(self):
        traci = e2e_mod._TinyTraci()
        traci.init(label="a", port=1)
        traci.init(label="b", port=2)  # current -> b
        b = traci.getConnection("b")
        traci.close()
        assert b.closed_calls == [True]
        assert "b" not in traci.connections  # deleted, like real traci
        assert "a" in traci.connections

    def test_any_call_on_a_closed_connection_raises_like_real_traci(self):
        traci = e2e_mod._TinyTraci()
        traci.init(label="a", port=1)
        traci.init(label="b", port=2)  # current -> b
        traci.connections["b"].close(wait=False)
        # module-level saveState now lands on the CLOSED branch connection:
        # the exact cl_smoke_2 chain
        with pytest.raises(e2e_mod._FatalTraCIError, match="Connection already closed"):
            traci.simulation.saveState("unused_path")
        with pytest.raises(e2e_mod._FatalTraCIError, match="Connection already closed"):
            traci.simulationStep()


# ------------------------------------------------------------ the e2e proof
class TestPL04lE2eBranchesClosedBetweenMasterSaves:
    """The cl_smoke_2 sequence, end to end: >= 2 predict_many batches at one
    decision, lookahead branches launched AND closed between the master's
    saveStates. On the pre-PL-04l code (master api = the traci module) this
    run dies with ``_FatalTraCIError: Connection already closed`` at the
    second batch's master saveState; on the fixed code it completes."""

    def test_two_predict_many_batches_survive_branches_closed_between_saves(
        self, tmp_path, monkeypatch
    ):
        rig = _planner_e2e(tmp_path, monkeypatch)
        receipt = ClosedLoopRunner(rig.config, rig.components).run()

        assert receipt["exit_status"] == "complete"
        decisions = e2e_mod.read_jsonl(rig.run_dir / "decisions.jsonl")
        assert decisions and decisions[0]["abstain"] is False
        assert decisions[0]["n_candidates_evaluated"] >= 2

        # the master's own connection carried >= 2 lookahead saveStates...
        master = rig.traci.connections[MASTER_LABEL]
        saves = [op for op in master.calls if op == "simulation.saveState"]
        assert len(saves) >= 2
        # ...and branches were launched and closed BETWEEN those saves (the
        # global ordered event log proves the interleaving)
        events = rig.traci.events
        save_positions = [
            i
            for i, (label, kind, op) in enumerate(events)
            if label == MASTER_LABEL and op == "simulation.saveState"
        ]
        first, last = save_positions[0], save_positions[-1]
        between = events[first + 1 : last]
        assert any(
            label != MASTER_LABEL and kind == "init" for label, kind, _op in between
        ), "no branch was launched between two master saveStates"
        assert any(
            label != MASTER_LABEL and kind == "close" for label, kind, _op in between
        ), "no branch was closed between two master saveStates"
        assert len(_branch_labels(rig.traci)) >= 2  # real branch launches
        for label in _branch_labels(rig.traci):
            assert rig.traci.connections[label].closed_calls == [False]


class TestPL04lNoCrossTalk:
    """Assertion-level no-cross-talk over a full e2e run: master
    step_physics/control/observe calls are recorded ONLY on the master
    connection, branch calls ONLY on their own connections, and nothing
    routes through the traci module."""

    def test_master_and_branch_calls_recorded_only_on_their_own_connections(
        self, tmp_path, monkeypatch
    ):
        rig = _planner_e2e(tmp_path, monkeypatch)
        receipt = ClosedLoopRunner(rig.config, rig.components).run()
        assert receipt["exit_status"] == "complete"

        master = rig.traci.getConnection(MASTER_LABEL)
        # (0) the runner's own parts hold the master CONNECTION OBJECT and
        # hand it to every consumer (never the traci module)
        harness = rig.made["harness"]
        model = rig.made["model"]
        assert harness._traci is master
        assert harness.master_api is master
        assert harness._control_write.connection is master  # control write
        assert harness.snapshotter._traci_api is master  # save/load state
        bound = model.snapshotter._inner
        assert bound is not None and bound.api is master  # MasterBound save
        # the branch runner's symmetry-only api resolves to the same object
        assert model.branch_runner.api() is master

        # (1) ZERO module-level traci use anywhere in the run: every
        # master/branch call went through its OWN connection object
        assert rig.traci.module_calls == []

        # (2) master-only ops on the master connection: the ONE base-state
        # loadState, master physics to the horizon, and the master control/
        # observe path (sensor captures + ownership speed-mode checks + the
        # start speedMode config) all recorded on the master's OWN log
        assert master.calls.count("simulation.loadState") == 1
        # master step_physics: the harness drives the connection's own
        # simulationStep (>= 20 x 0.5 s to reach the 10 s horizon)
        assert master.calls.count("simulationStep") >= 20
        assert "vehicle.setSpeedMode" in master.calls  # start config (27)
        assert master.calls.count("vehicle.getAllowedSpeed") >= 2  # captures
        assert master.calls.count("vehicle.getSpeedMode") >= 2  # ownership ticks
        assert master.world.time == pytest.approx(250.0)  # stepped to the end

        # (3) every branch: exactly one loadState on ITS OWN connection,
        # never a saveState, closed exactly once, stepped to its own horizon
        for label in _branch_labels(rig.traci):
            conn = rig.traci.connections[label]
            assert conn.calls.count("simulation.loadState") == 1
            assert "simulation.saveState" not in conn.calls
            assert conn.closed_calls == [False]
            assert conn.world.time == pytest.approx(245.0)
        # the non-native lookahead branches ran the REAL per-branch control
        # write ON THEIR OWN connections (slowDown issued by a branch's
        # ownership, never by the master's)
        assert any(
            "vehicle.slowDown" in rig.traci.connections[label].calls
            for label in _branch_labels(rig.traci)
        )

        # (4) window isolation from the global ordered event log: while a
        # branch is open, ONLY that branch's ops appear (a master step or
        # control tick routed to the open/closed branch would show here).
        # The master's own init/close events are outside every branch window.
        open_branch = None
        for label, kind, _op in rig.traci.events:
            if kind == "init":
                open_branch = None if label == MASTER_LABEL else label
            elif kind == "close":
                if label != MASTER_LABEL:
                    assert label == open_branch, f"unexpected close of {label!r}"
                    open_branch = None
            else:
                assert open_branch in (None, label), (
                    f"cross-talk: op on {label!r} while {open_branch!r} is open"
                )
