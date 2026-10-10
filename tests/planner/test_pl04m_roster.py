"""PL-04m: the CAV roster is the campaign's FULL designated set.

Before PL-04m, ``roster_from_base_run`` kept only the CAVs that received a
damping command in the base run AND were inserted <= t0 -- so every CAV
inserted after t0 stayed undesignated on the master (speedMode 31, no
ownership) and the planner abstained once the t0 CAVs left. PL-04m makes
the roster the base run's CONSTANT ``cav_ids`` list (the AUTHORITATIVE
designation source carried by every ``run/physics_steps.jsonl`` row),
fail-closed on a missing/empty/inconsistent source, and proves that a CAV
first appearing after t0 is designated on its first live tick -- on the
master AND in every perfect-sim lookahead branch.

Zero SUMO: the designation tests run the REAL component wiring over the
established tiny fake traci world (test_pl04i_e2e_real_run) with one
scheduled post-t0 insertion. The bounded REAL-SUMO replay check is the
opt-in ``test_pl04m_real_check`` module (PLANNER_REAL_SUMO=1).
"""

import json
import sys
from types import SimpleNamespace

import pytest

from tests.planner._helpers import make_base_run
from tests.planner.test_closed_loop import (  # the established rig fakes
    CAV_SPEED_MODE,
    MasterRig,
)
from tests.planner.test_pl04i_e2e_real_run import (  # the tiny fake SUMO world
    BASE_T0_S,
    CAVS,
    PRE_MERGE_LANE,
    SEED,
    _TinyConnection,
    _TinyTraci,
    _TinyWorld,
    _real_args,
    make_base_run_pl04i as _make_base_run_pl04i,
    make_base_run_pl04j,
    read_jsonl,
)
from traffic_shepherd.planner import PlannerConfig, PlanCandidate
from traffic_shepherd.planner.closed_loop import (
    BaseRunRefusedError,
    ClosedLoopConfig,
    ClosedLoopRunner,
    RosterSourceError,
    build_real_components,
    initial_plan_for_roster,
    resolve_base_run,
    roster_from_base_run,
)
from traffic_shepherd.planner.loop import Planner, select_decision_units
from traffic_shepherd.planner.mock import MockWorldModel

LATE_CAV = "mainline.99"
INSERT_AT_S = 242.5  # AFTER t0=240; inside a 240->245 lookahead branch too
# insertion positions that keep the roster leader inside sensor coverage on
# each fixture geometry (the group moved INSERT_AT_S - t0 ahead by then)
INSERT_POS_BY_LANE = {PRE_MERGE_LANE: 440.0, None: 150.0}  # None = the default L0


# ------------------------------------------------- the roster source (unit)
class TestRosterFromBaseRunPL04m:
    """The roster = the base run's constant physics cav_ids, fail closed."""

    def test_full_roster_including_post_t0_ids(self, tmp_path):
        base = make_base_run(tmp_path)  # cav_ids incl. the post-t0 late_cav
        roster, evidence = roster_from_base_run(base, 240.0)
        assert roster == ("mainline.6", "mainline.7", "mainline.99")
        assert evidence["roster_source"] == "base_run_physics_cav_ids"
        assert evidence["roster_size"] == 3
        assert evidence["physics_rows_read"] == 2
        assert evidence["roster_first_row_time_s"] == 0.5
        # the receipt counts come from insertions.jsonl (read-only), which
        # is NEVER a designation source
        assert evidence["n_inserted_le_t0"] == 2
        assert evidence["n_inserted_after_t0_or_never"] == 1
        assert evidence["insertions_read"] == 4

    def test_roster_includes_a_cav_never_inserted_in_the_run(self, tmp_path):
        # a cav_ids entry without an insertions row at all counts as
        # "after t0 or never"; the roster still carries it
        base = make_base_run(tmp_path, cav_ids=("mainline.6", "mainline.7000"))
        roster, evidence = roster_from_base_run(base, 240.0)
        assert roster == ("mainline.6", "mainline.7000")
        assert evidence["n_inserted_le_t0"] == 1
        assert evidence["n_inserted_after_t0_or_never"] == 1

    def test_inconsistent_rows_refused(self, tmp_path):
        base = make_base_run(tmp_path)
        physics = base / "run" / "physics_steps.jsonl"
        rows = [json.loads(line) for line in physics.read_text().splitlines() if line]
        rows[1]["cav_ids"] = ["mainline.6"]  # the later row disagrees
        physics.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        with pytest.raises(RosterSourceError, match="disagrees"):
            roster_from_base_run(base, 240.0)

    def test_row_without_cav_ids_refused(self, tmp_path):
        base = make_base_run(tmp_path)
        physics = base / "run" / "physics_steps.jsonl"
        rows = [json.loads(line) for line in physics.read_text().splitlines() if line]
        del rows[1]["cav_ids"]
        physics.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        with pytest.raises(RosterSourceError, match="no usable cav_ids"):
            roster_from_base_run(base, 240.0)

    def test_empty_cav_ids_refused(self, tmp_path):
        base = make_base_run(tmp_path, cav_ids=[])
        with pytest.raises(RosterSourceError, match="no usable cav_ids"):
            roster_from_base_run(base, 240.0)

    def test_missing_physics_file_refused(self, tmp_path):
        base = make_base_run(tmp_path)
        (base / "run" / "physics_steps.jsonl").unlink()
        with pytest.raises(RosterSourceError, match="physics_steps.jsonl"):
            roster_from_base_run(base, 240.0)

    def test_rowless_physics_file_refused(self, tmp_path):
        base = make_base_run(tmp_path)
        (base / "run" / "physics_steps.jsonl").write_text("", encoding="utf-8")
        with pytest.raises(RosterSourceError, match="no physics rows"):
            roster_from_base_run(base, 240.0)

    def test_malformed_physics_row_refused(self, tmp_path):
        base = make_base_run(tmp_path)
        physics = base / "run" / "physics_steps.jsonl"
        physics.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(RosterSourceError, match="not valid JSON"):
            roster_from_base_run(base, 240.0)

    def test_missing_insertions_file_refused(self, tmp_path):
        # the counts need insertions.jsonl (read-only); without it the
        # receipt could not be truthful -> fail closed
        base = make_base_run(tmp_path)
        (base / "run" / "insertions.jsonl").unlink()
        with pytest.raises(RosterSourceError, match="insertions.jsonl"):
            roster_from_base_run(base, 240.0)

    def test_refusal_is_a_baserun_refused_subclass(self, tmp_path):
        base = make_base_run(tmp_path)
        (base / "run" / "physics_steps.jsonl").unlink()
        with pytest.raises(BaseRunRefusedError):
            roster_from_base_run(base, 240.0)

    def test_t0_is_recorded_not_a_filter(self, tmp_path):
        # the roster is t0-independent now: the same list at any t0
        base = make_base_run(tmp_path)
        roster_at_0, _ = roster_from_base_run(base, 0.0)
        roster_at_big, _ = roster_from_base_run(base, 1e6)
        assert roster_at_0 == roster_at_big == ("mainline.6", "mainline.7", "mainline.99")


# --------------------------------- the real build path carries the full set
class TestRealComponentsFullRoster:
    def test_build_takes_the_full_roster_and_records_evidence(self, tmp_path, monkeypatch):
        base = make_base_run(tmp_path, seed=8468)
        resolved = resolve_base_run(base)
        monkeypatch.delenv("SUMO_BINARY", raising=False)
        monkeypatch.setenv("SUMO_HOME", "E:/sumo_home_fake_for_test")
        monkeypatch.setitem(sys.modules, "traci", None)  # build touches no traci
        args = SimpleNamespace(
            predictor="perfect_sim",
            seed=resolved.seed,
            base_run=str(base),
            decision_period_s=10.0,
            lookahead_horizon_s=5.0,
            lookahead_commit_s=None,
        )
        components = build_real_components(args)
        assert components.roster == ("mainline.6", "mainline.7", "mainline.99")
        evidence = components.roster_evidence
        assert evidence["roster_source"] == "base_run_physics_cav_ids"
        assert evidence["roster_size"] == 3
        assert evidence["n_inserted_le_t0"] == 2
        assert evidence["n_inserted_after_t0_or_never"] == 1


# --------------------------------------------- the designation probe (e2e)
class _InsertWorld(_TinyWorld):
    """The tiny world + one scheduled post-t0 insertion (master + every
    branch world models the same scenario insertion)."""

    def __init__(self, lane=None):
        super().__init__(lane=lane)
        self._inserted = False
        self._insert_pos = INSERT_POS_BY_LANE.get(lane, 150.0)

    def step(self, dt=0.5):
        if not self._inserted and self.time + dt > INSERT_AT_S + 1e-9:
            self._inserted = True
            self.vehicles[LATE_CAV] = {
                "pos": self._insert_pos,
                "speed": 20.0,
                "lane": self.LANE,
                "target": None,
            }
        super().step(dt=dt)


class _InsertConnection(_TinyConnection):
    def __init__(self, label, lane=None, bus=None):
        super().__init__(label, lane=lane, bus=bus)
        self.world = _InsertWorld(lane=lane)


def make_base_run_pl04m(tmp_path, *, near_merge=True):
    """A pl04j/pl04i base run + the PL-04m roster source: physics rows
    carrying the FULL designated cav_ids (incl. the post-t0 LATE_CAV) + its
    insertion row. ``near_merge`` picks the fixture geometry: the
    pre-merge-lane variant (merge-observed decisions) or the mid-lane
    variant (a leader that stays inside sensor coverage for the damping
    command rows)."""
    base = (
        make_base_run_pl04j(tmp_path) if near_merge else _make_base_run_pl04i(tmp_path)
    )
    run = base / "run"
    with (run / "physics_steps.jsonl").open("w", encoding="utf-8") as stream:
        for row_time in (0.5, float(BASE_T0_S)):
            stream.write(
                json.dumps(
                    {
                        "time_s": row_time,
                        "live_ids": [],
                        "departed_ids": [],
                        "arrived_ids": [],
                        "colliding_ids": [],
                        "cav_ids": [CAVS[0], CAVS[1], LATE_CAV],
                        "speed_modes": {},
                        "cav_actions": {},
                    }
                )
                + "\n"
            )
    with (run / "insertions.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "vehicle_id": LATE_CAV,
                    "scheduled_depart_s": INSERT_AT_S,
                    "inserted_s": INSERT_AT_S,
                    "entrance": "mainline",
                    "is_cav": False,
                    "type_id": "normal_car",
                }
            )
            + "\n"
        )
    return base


@pytest.fixture()
def pl04m_e2e(request, tmp_path, monkeypatch):
    near_merge = getattr(request, "param", True)
    base = make_base_run_pl04m(tmp_path, near_merge=near_merge)
    resolved = resolve_base_run(base)
    monkeypatch.setenv("SUMO_BINARY", sys.executable)  # harmless children
    monkeypatch.delenv("SUMO_HOME", raising=False)
    traci = _TinyTraci(lane=PRE_MERGE_LANE if near_merge else None)
    monkeypatch.setitem(sys.modules, "traci", traci)
    import tests.planner.test_pl04i_e2e_real_run as e2e_mod

    monkeypatch.setattr(e2e_mod, "_TinyConnection", _InsertConnection)
    args = _real_args(base)
    components = build_real_components(args)
    return SimpleNamespace(
        base=base,
        traci=traci,
        components=components,
        run_dir=tmp_path / "w_pl04m",
        resolved=resolved,
        e2e_mod=e2e_mod,
    )


def _pl04m_config(pl04m_e2e, tmp_path, *, initial_plan_action):
    return ClosedLoopConfig(
        out_root=tmp_path,
        run_id="w_pl04m",
        world="w_pl04m",
        seed=SEED,
        horizon_s=15.0,
        warmup_s=0.0,
        decision_period_s=10.0,
        predictor="perfect_sim",
        lookahead_horizon_s=5.0,
        lookahead_commit_s=None,
        k_samples=1,
        max_sim_s=500.0,
        base_run=pl04m_e2e.resolved.base_run,
        base_t0_s=pl04m_e2e.resolved.t0_s,
        base_state_sha256=pl04m_e2e.resolved.state_sha256,
        max_decision_units=None,
        walls="off",
        initial_plan_action=(
            None if initial_plan_action == "native" else initial_plan_action
        ),
    )


class TestPL04mLateCAVDesignationE2E:
    """A CAV first appearing at t0+2.5 is designated on its FIRST live
    tick -- on the master AND inside the perfect-sim lookahead branches."""

    def test_native_default_native_record_then_decision_eligibility(
        self, pl04m_e2e, tmp_path, capsys
    ):
        # no initial plan: the default is native for the WHOLE roster, so
        # the late CAV's first record is the native selected_native one
        receipt = ClosedLoopRunner(
            _pl04m_config(pl04m_e2e, tmp_path, initial_plan_action="native"),
            pl04m_e2e.components,
        ).run()
        assert receipt["exit_status"] == "complete"
        # the receipt carries the truthful roster source fields (PL-04m)
        assert receipt["cav_roster"] == [CAVS[0], CAVS[1], LATE_CAV]
        assert receipt["roster_source"] == "base_run_physics_cav_ids"
        assert receipt["roster_size"] == 3
        assert receipt["n_inserted_le_t0"] == 2
        assert receipt["n_inserted_after_t0_or_never"] == 1

        # designation on the FIRST live tick: the late CAV goes live at
        # 242.5; the next control tick (243.0) configures speedMode 27 and
        # records the native command
        master = pl04m_e2e.traci.connections["pl04"]
        assert master.world.speed_modes[LATE_CAV] == CAV_SPEED_MODE
        control = [
            row
            for row in read_jsonl(pl04m_e2e.run_dir / "dispatcher_rows.jsonl")
            if row.get("source") == "master_control_tick" and row.get("cav") == LATE_CAV
        ]
        assert control, "the late CAV was never commanded on the master"
        first = min(control, key=lambda row: row["t_s"])
        assert first["t_s"] == 243.0
        assert first["command"] == "native"
        assert first["fallback"] == "selected_native"

        # the run's own physics rows carry the FULL cav_ids list and the
        # late CAV's mode is 27 from the row AFTER its first control tick
        # (T-B: the tick runs before the step, exactly like the campaign, so
        # the first-seen rows still show the default 31)
        physics = read_jsonl(pl04m_e2e.run_dir / "physics_steps.jsonl")
        assert physics[0]["cav_ids"] == [CAVS[0], CAVS[1], LATE_CAV]
        late_modes = {
            row["time_s"]: row["speed_modes"].get(LATE_CAV)
            for row in physics
            if row["time_s"] > INSERT_AT_S + 0.5
        }
        assert late_modes and set(late_modes.values()) == {CAV_SPEED_MODE}
        assert min(late_modes) == 243.5  # the row after its first control tick (243.0)

        # decision-unit eligibility: both decisions' action maps cover the
        # late CAV (it is a free unit like every roster CAV; it went live
        # after t0 and is NOT excluded anywhere)
        decisions = read_jsonl(pl04m_e2e.run_dir / "decisions.jsonl")
        assert [row["t_s"] for row in decisions] == [240.0, 250.0]
        for row in decisions:
            assert LATE_CAV in row["actions"]

        # lookahead branches: the plan branches carry the SAME full roster
        # -- the late CAV inserted mid-lookahead (242.5) is designated in
        # the branch (speedMode 27 there too)
        branch_rows = [
            row
            for row in read_jsonl(pl04m_e2e.run_dir / "branch_control_rows.jsonl")
            if row.get("cav") == LATE_CAV
        ]
        assert branch_rows, "the late CAV was never designated in a lookahead branch"
        branch_labels = [
            call["label"] for call in pl04m_e2e.traci.init_calls if call["label"] != "pl04"
        ]
        branch_modes = {
            label: pl04m_e2e.traci.connections[label].world.speed_modes.get(LATE_CAV)
            for label in branch_labels
        }
        assert CAV_SPEED_MODE in set(branch_modes.values())

    @pytest.mark.parametrize("pl04m_e2e", [False], indirect=True)  # mid-lane geometry
    def test_non_native_plan_commands_the_late_cav(self, pl04m_e2e, tmp_path):
        # --initial-plan damping: a non-native plan covers the WHOLE roster
        # (initial_plan_for_roster over the full set) -- the late CAV gets
        # real damping command rows on the master and in the branches
        components = pl04m_e2e.components
        components.initial_plan = initial_plan_for_roster(components.roster, "damping")
        receipt = ClosedLoopRunner(
            _pl04m_config(pl04m_e2e, tmp_path, initial_plan_action="damping"),
            components,
        ).run()
        assert receipt["exit_status"] == "complete"
        assert receipt["initial_plan"] == "damping"
        master = pl04m_e2e.traci.connections["pl04"]
        assert master.world.speed_modes[LATE_CAV] == CAV_SPEED_MODE
        control = [
            row
            for row in read_jsonl(pl04m_e2e.run_dir / "dispatcher_rows.jsonl")
            if row.get("source") == "master_control_tick" and row.get("cav") == LATE_CAV
        ]
        assert control, "the non-native plan never commanded the late CAV"
        first = min(control, key=lambda row: row["t_s"])
        assert first["t_s"] == 243.0
        assert first["command"] == "damping"
        assert first["assigned_action"] == "damping"

        branch_rows = [
            row
            for row in read_jsonl(pl04m_e2e.run_dir / "branch_control_rows.jsonl")
            if row.get("cav") == LATE_CAV
        ]
        assert branch_rows
        assert any(row["command"] == "damping" for row in branch_rows)


# ------------------------------------- focused harness-level unit (fakes)
class TestPL04mLateCAVHarnessUnit:
    """MasterRig-level: a CAV appearing after start() is designated on its
    first live control tick (native selected_native), stays eligible as a
    decision unit, and a non-native plan can command it."""

    def test_late_cav_designated_on_first_live_tick(self, tmp_path, monkeypatch):
        rig = MasterRig(
            monkeypatch,
            tmp_path,
            run_id="pl04m_u1",
            vehicle_ids=("c0", "c1"),  # c9 NOT live yet (inserted after t0)
        )
        # rebuild the harness with the FULL roster (c9 in) over the same
        # fake module surface
        rig.args.cav_ids = "c0,c1,c9"
        rig.components = build_real_components(rig.args)
        harness = rig.components.make_harness(rig.context)
        rig._harness = harness  # the rig's `sensor` property reads this
        harness.start()
        assert rig.traci.speed_modes == {"c0": 27, "c1": 27}  # live at start
        harness.step_physics()  # 0.5: physics tick
        harness.step_physics()  # 1.0: control tick (c0, c1 native)
        # the late CAV appears
        rig.traci.vehicle_ids = ("c0", "c1", "c9")
        harness.step_physics()  # 1.5: physics tick
        harness.step_physics()  # 2.0: control tick -- c9's FIRST live tick
        owner = rig.master_owner
        last = owner.steps[-1]
        assert set(last["owners"]) == {"c0", "c1", "c9"}
        assert last["owners"]["c9"] == "native"
        assert rig.traci.speed_modes["c9"] == CAV_SPEED_MODE
        rows = harness.drain_control_rows()
        c9_rows = [row for row in rows if row["cav"] == "c9"]
        assert c9_rows and c9_rows[0]["command"] == "native"
        assert c9_rows[0]["fallback"] == "selected_native"

        # the late CAV is a FULL decision citizen: observe() reports it in
        # the controller_state, the distance cap ranks it, and a damping
        # plan commands it through the same control write
        rig.sensor.frames_by_id = {
            "c0": {"distance_to_merge_m": 120.0},
            "c1": {"distance_to_merge_m": 80.0},
            "c9": {"distance_to_merge_m": 20.0},  # nearest upstream
        }
        samples, controller_state = harness.observe()
        assert "c9" in controller_state
        units, held = select_decision_units(tuple(controller_state), samples, 2)
        assert units[0] == "c9"  # eligible: ranked (and chosen) like any CAV
        assert "c9" not in held

        harness.apply_plan(
            rig.v3_plan({"c0": "native", "c1": "native", "c9": "damping"}), 2.0
        )
        harness.step_physics()  # 2.5: physics tick
        harness.step_physics()  # 3.0: control tick
        last = rig.master_owner.steps[-1]
        assert last["owners"]["c9"] == "damping"
        assert any(v == "c9" for v, _, _ in rig.traci.master_slow_downs)
        rows = harness.drain_control_rows()
        assert any(
            row["cav"] == "c9" and row["command"] == "damping" for row in rows
        )
        harness.close()

    def test_late_cav_is_a_planner_decision_unit(self):
        # loop-level: with the full roster, a post-t0 CAV in the samples is
        # ranked and the planner may move it (no t0 filter anywhere)
        cav = ("c0", "c1", "c9")
        model = MockWorldModel(
            cav, part_a_horizon_s=30.0, contract_version="wd01_predict/2", model_id="mock/m"
        )
        model.meta_script.update(
            {
                "future_min_ttc_validated": True,
                "future_min_ttc_validation_id": "test-validated",
                "relative_safety_validated": True,
                "relative_safety_validation_id": "test-validated",
            }
        )
        model.action_delay = {c: {"damping": -5.0} for c in cav}

        def sample(time_s, distances):
            return SimpleNamespace(
                hybrid=None,
                arrivals={"seed": 0},
                meta={
                    "sample_id": "s0",
                    "time_s": time_s,
                    "merge_observed": True,
                    "coverage": 1.0,
                },
                frames={c: time_s for c in distances},
                frames_distance_to_merge_m=dict(distances),
            )

        planner = Planner(
            model,
            PlannerConfig(
                mode="closed_loop", decide_every_s=1.0, trajectory_s=None,
                margin_floor_s=0.0,
                max_decision_units=1,  # only the NEAREST unit may move
            ),
            current_plan=PlanCandidate(per_cav={c: "native" for c in cav}),
        )
        decision = planner.decide(
            [sample(0.0, {"c0": 300.0, "c1": 200.0, "c9": 50.0})],
            0.0,
            controller_state={
                c: {
                    "action": "native",
                    "wall_id": None,
                    "wall_phase": None,
                    "time_in_phase_s": 0.0,
                    "wall_speed_mps": None,
                    "locked": False,
                }
                for c in cav
            },
        )
        # the post-t0 CAV is a ranked unit (the nearest one) and the chosen
        # plan covers it -- only c9 may move under the cap
        units, held = select_decision_units(
            cav, [sample(0.0, {"c0": 300.0, "c1": 200.0, "c9": 50.0})], 1
        )
        assert units == ("c9",) and held == ("c0", "c1")
        assert decision.plan.per_cav == {"c0": "native", "c1": "native", "c9": "damping"}
