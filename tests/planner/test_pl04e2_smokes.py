"""PL-04e2 smoke-wrapper unit tests (FAKES ONLY -- zero SUMO, zero launches).

- run_confirm.py (the PL-10 confirmation smoke): a fake ConfirmLauncher over
  deterministic closed-form fake SUMO connections proves the N1/plan/N2
  fresh-process protocol, the verdict keys, and the FAIL/budget/guard paths.
- run_cl_smoke.py (the PL-04 real closed-loop smoke): a fake closed-loop
  launcher writing a canned REAL-shaped run dir proves the split-ladder
  projection refusal, the exact launch argv, and the a/b/c verdict reader
  (PASS / INCONCLUSIVE / FAIL).

Neither wrapper is executed against SUMO here: a campaign is running, and
the wrappers join the coordinated pause-window smoke."""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

PLANNER_ROOT = Path(__file__).resolve().parents[2]


def _load_wrapper(name, relative):
    spec = importlib.util.spec_from_file_location(name, PLANNER_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_confirm = _load_wrapper("pl04e2_run_confirm", "output/planner_v1/resmoke/run_confirm.py")
run_cl_smoke = _load_wrapper("pl04e2_run_cl_smoke", "output/planner_v1/closed_loop_smoke/run_cl_smoke.py")

from tests.planner._helpers import make_base_run, scratch_root  # noqa: E402


# ----------------------------------------------------- run_confirm.py fakes
class _FakeSumoConn:
    """A deterministic closed-form fake SUMO connection: every vehicle sits
    at 100 + speed*time (speed 19 damped / 20 native), so the continuous
    master run and the restored native branches agree per tick and the
    damping branch diverges from its first control tick. saveState writes a
    17-decimal rngState file (the real exactness guard passes); loadState
    lands on the sibling manifest's t0."""

    def __init__(self, label):
        self.label = label
        self.confirm_phase = label
        self.time = 0.0
        self.damped = False
        self.closed_calls = 0
        self.simulation = _FakeSumoConn._Sim(self)
        self.vehicle = _FakeSumoConn._Vehicle(self)

    def getVersion(self):
        return (27, " FakeSUMO 1.27.1")

    def simulationStep(self):
        self.simulation.simulationStep()  # the real Connection's top-level step

    def close(self, wait=True):
        self.closed_calls += 1

    class _Sim:
        def __init__(self, conn):
            self._conn = conn

        def getTime(self):
            return self._conn.time

        def simulationStep(self):
            self._conn.time = round(self._conn.time + 0.5, 10)

        def saveState(self, path):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            pos = 100.0 + (19.0 if self._conn.damped else 20.0) * self._conn.time
            path.write_text(
                '<state><rngState seed="8468"/>'
                f'<vehicle id="veh0" pos="{pos:.17f}" speed="20.00000000000000000"/>'
                "</state>",
                encoding="utf-8",
            )

        def loadState(self, path):
            manifest = json.loads(
                (Path(path).parent / "STATE_MANIFEST.json").read_text(encoding="utf-8")
            )
            self._conn.time = float(manifest["branch_point_t0_s"])
            self._conn.damped = False

        def getArrivedIDList(self):
            return []  # T-B: the control write reads arrivals (run_suffix lifecycle)

        def getStartingTeleportIDList(self):
            return ()

        def getCollidingVehiclesIDList(self):
            return ()

    class _Vehicle:
        def __init__(self, conn):
            self._conn = conn

        def getIDList(self):
            return ("veh0", "veh1")

        @staticmethod
        def _speed(conn):
            return 19.0 if conn.damped else 20.0

        def getLanePosition(self, vid):
            return 100.0 + self._speed(self._conn) * self._conn.time

        def getSpeed(self, vid):
            return self._speed(self._conn)

        def getAcceleration(self, vid):
            return 0.0

        def getLaneID(self, vid):
            return "lane_0"

        def getLaneIndex(self, vid):
            return 0


class FakeConfirmLauncher(run_confirm.ConfirmLauncher):
    """The injectable launcher's fake: fresh fake connections, no children."""

    def __init__(self, *, damp_plan=True):
        self.damp_plan = damp_plan
        self.started_phases = []
        self.closed_phases = []
        self.controls = []
        self.launches = []  # the real launcher records its children here

    def start(self, phase):
        self.started_phases.append(phase)
        return _FakeSumoConn(phase)

    def close(self, phase, conn):
        self.closed_phases.append(phase)

    @staticmethod
    def version(conn):
        return "1.27.1"

    def adapter_config(self):
        return SimpleNamespace(sensor_range_m=100.0)

    def plan_control(self, conn, roster):
        stats = dict(run_confirm.empty_control_stats())

        def control(now):
            stats["control_ticks"] += 1
            stats["command_rows"] += len(roster)
            if self.damp_plan:
                conn.damped = True
                stats["non_fallback_rows"] += len(roster)
                stats["rows_by_owner"]["damping"] = (
                    stats["rows_by_owner"].get("damping", 0) + len(roster)
                )
                stats["rows_by_mode"]["damping"] = (
                    stats["rows_by_mode"].get("damping", 0) + len(roster)
                )
            else:
                stats["fallbacks"] += len(roster)
                stats["rows_by_owner"]["native"] = (
                    stats["rows_by_owner"].get("native", 0) + len(roster)
                )
                stats["rows_by_mode"]["native"] = (
                    stats["rows_by_mode"].get("native", 0) + len(roster)
                )
                stats["fallback_reason_histogram"]["selected_native"] = (
                    stats["fallback_reason_histogram"].get("selected_native", 0)
                    + len(roster)
                )

        control.stats = stats
        control.release = lambda: None
        self.controls.append(control)
        return control


@pytest.fixture()
def confirm_scenario(tmp_path):
    """A campaign-format scenario dir: two vehicles at t0, one after."""
    scenario = tmp_path / "scenario_src"
    scenario.mkdir()
    (scenario / "requests.rou.xml").write_text(
        "<routes>\n"
        '  <vType id="normal_car" speedDev="0"/>\n'
        '  <route id="r0" edges="e0 e1"/>\n'
        '  <vehicle id="veh0" type="normal_car" route="r0" depart="0.0"/>\n'
        '  <vehicle id="veh1" type="normal_car" route="r0" depart="1.0"/>\n'
        '  <vehicle id="veh2" type="normal_car" route="r0" depart="31.0"/>\n'
        "</routes>\n",
        encoding="utf-8",
    )
    (scenario / "scenario.sumocfg").write_text(
        "<configuration>\n  <input>\n"
        '    <net-file value="legacy-mount/assets/net.xml.gz"/>\n'
        '    <route-files value="requests.rou.xml"/>\n'
        "  </input>\n</configuration>\n",
        encoding="utf-8",
    )
    return scenario


def _write_base_run(base: Path) -> Path:
    """A fake campaign base run in the REAL layout: ``<base>/run/
    telemetry.jsonl`` (command rows designate CAVs; other rows ignored) and
    ``<base>/run/insertions.jsonl`` (``inserted_s``). At t0=30 the re-smoke
    rule yields exactly ["veh0", "veh3"]: veh0 commands+early; hdv.9 early
    but NO command rows; veh1 commands but inserted AFTER t0; veh3
    commands+early (its second row is owner native -- command rows
    designate regardless of owner, exactly like the re-smoke reader)."""
    run = base / "run"
    run.mkdir(parents=True)
    (run / "insertions.jsonl").write_text(
        json.dumps({"vehicle_id": "veh0", "inserted_s": 5.0}) + "\n"
        + json.dumps({"vehicle_id": "hdv.9", "inserted_s": 10.0}) + "\n"
        + json.dumps({"vehicle_id": "veh1", "inserted_s": 31.0}) + "\n"
        + json.dumps({"vehicle_id": "veh3", "inserted_s": 7.0}) + "\n",
        encoding="utf-8",
    )
    (run / "telemetry.jsonl").write_text(
        json.dumps({"kind": "fleet_switch_attempt", "timestamp_s": 0.0}) + "\n"
        + json.dumps({"kind": "command", "timestamp_s": 6.0, "vehicle_id": "veh0",
                      "owner": "damping"}) + "\n"
        + json.dumps({"kind": "command", "timestamp_s": 8.0, "vehicle_id": "veh3",
                      "owner": "damping"}) + "\n"
        + json.dumps({"kind": "command", "timestamp_s": 9.0, "vehicle_id": "veh3",
                      "owner": "native"}) + "\n",
        encoding="utf-8",
    )
    return base


class TestRunConfirm:
    def test_pass_verdict_from_a_fake_run(self, tmp_path, confirm_scenario):
        out = tmp_path / "confirm_out"
        base_run = _write_base_run(tmp_path / "campaign_base")
        launcher = FakeConfirmLauncher()
        verdict = run_confirm.run_confirm_smoke(
            launcher,
            out_dir=out,
            scenario_dir=confirm_scenario,
            base_seed=8468,
            base_run=base_run,
        )
        # the verdict keys, exactly as the brief requires
        assert verdict["identity_native_vs_continuous"] is True
        assert verdict["restored_vs_restored_identical"] is True
        assert verdict["plan_differs_from_native"] is True
        # PL-04h: the plan branch COMMANDED (>= 1 non-fallback row)
        assert verdict["plan_non_fallback_command_rows"] >= 1
        assert verdict["sim_s_total"] == 78.0
        assert verdict["process_start_s_measured"] >= 0.0
        assert verdict["sumo_version"] == "1.27.1"
        assert verdict["verdict"] == "PASS"
        assert "reasons" not in verdict
        written = json.loads((out / "CONFIRM_VERDICT.json").read_text(encoding="utf-8"))
        assert written["verdict"] == "PASS"
        # 4 fresh processes: master + N1 + P + N2, each closed exactly once
        assert len(launcher.started_phases) == 4
        assert sorted(launcher.closed_phases) == sorted(launcher.started_phases)
        # the scenario copy lives inside --out, net rewritten
        copied = out / "scenario" / "scenario.sumocfg"
        assert copied.is_file()
        assert "assets/research/highway_merge_shepherd_long" in copied.read_text("utf-8")
        # roster: the base-run rule (READ-ONLY), ids/rule/source recorded
        roster = verdict["report"]["roster"]
        assert roster["ids"] == ["veh0", "veh3"]
        assert "damping command row" in roster["rule"] and "inserted_s <= t0" in roster["rule"]
        assert roster["source"] == str(base_run)
        assert roster["designated_cav_count_total"] == 2
        assert roster["command_rows_read"] == 3
        assert roster["insertions_read"] == 4
        # PL-04h diagnostics: per-branch owner/mode counts + the histogram,
        # empty for the native branches, damping-only for the plan branch
        branches = verdict["report"]["branches"]
        assert branches[0]["control"] == run_confirm.empty_control_stats()
        assert branches[2]["control"] == run_confirm.empty_control_stats()
        plan_control = branches[1]["control"]
        assert plan_control["rows_by_owner"] == {"damping": plan_control["command_rows"]}
        assert plan_control["rows_by_mode"] == {"damping": plan_control["command_rows"]}
        assert plan_control["non_fallback_rows"] == plan_control["command_rows"]
        diagnostics = verdict["report"]["plan_control_diagnostics"]
        assert diagnostics["non_fallback_command_rows"] == (
            verdict["plan_non_fallback_command_rows"]
        )
        assert diagnostics["rows_by_owner"] == {
            "damping": verdict["plan_non_fallback_command_rows"]
        }

    def test_inert_plan_control_fails_the_verdict(self, tmp_path, confirm_scenario):
        launcher = FakeConfirmLauncher(damp_plan=False)
        verdict = run_confirm.run_confirm_smoke(
            launcher,
            out_dir=tmp_path / "confirm_inert",
            scenario_dir=confirm_scenario,
            base_seed=8468,
            cav_ids_arg="veh0",
        )
        assert verdict["plan_differs_from_native"] is False
        assert verdict["plan_non_fallback_command_rows"] == 0
        assert verdict["verdict"] == "FAIL"
        assert any("plan branch" in reason for reason in verdict["reasons"])
        # PL-04h: an all-fallback plan branch FAILS the non-fallback gate too
        assert any(
            "NO non-fallback command row" in reason for reason in verdict["reasons"]
        )

    def test_hard_budget_refuses_up_front(self, tmp_path, confirm_scenario):
        with pytest.raises(run_confirm.ConfirmSmokeError, match="HARD"):
            run_confirm.run_confirm_smoke(
                FakeConfirmLauncher(),
                out_dir=tmp_path / "confirm_budget",
                scenario_dir=confirm_scenario,
                base_seed=8468,
                budget_s=50.0,
            )

    def test_out_dir_guard(self, tmp_path):
        with pytest.raises(run_confirm.OutDirRefusedError, match="runs/"):
            run_confirm.resolve_out_dir(tmp_path / "runs" / "somewhere")
        with pytest.raises(run_confirm.OutDirRefusedError, match="output/planner_v1"):
            run_confirm.resolve_out_dir(PLANNER_ROOT / "output" / "planner_v1" / "x")
        assert run_confirm.resolve_out_dir(tmp_path / "ok") == (tmp_path / "ok").resolve()

    def test_cav_ids_narrow_the_roster(self, tmp_path, confirm_scenario):
        out = tmp_path / "confirm_narrow"
        verdict = run_confirm.run_confirm_smoke(
            FakeConfirmLauncher(),
            out_dir=out,
            scenario_dir=confirm_scenario,
            base_seed=8468,
            cav_ids_arg="veh1",
        )
        assert verdict["report"]["roster"]["rule"].startswith("explicit --cav-ids")
        assert verdict["report"]["roster"]["source"] == "--cav-ids (command line)"
        assert verdict["report"]["roster"]["ids"] == ["veh1"]
        assert verdict["verdict"] == "PASS"

    def test_cav_ids_override_the_base_run_roster(self, tmp_path, confirm_scenario):
        base_run = _write_base_run(tmp_path / "campaign_base")
        verdict = run_confirm.run_confirm_smoke(
            FakeConfirmLauncher(),
            out_dir=tmp_path / "confirm_override",
            scenario_dir=confirm_scenario,
            base_seed=8468,
            base_run=base_run,
            cav_ids_arg="veh1",
        )
        roster = verdict["report"]["roster"]
        assert roster["rule"].startswith("explicit --cav-ids")
        assert roster["source"] == "--cav-ids (command line)"
        assert roster["ids"] == ["veh1"]  # NOT the base-run roster ["veh0", "veh3"]
        assert verdict["verdict"] == "PASS"

    def test_cav_ids_missing_at_t0_refused(self, tmp_path, confirm_scenario):
        with pytest.raises(run_confirm.ConfirmSmokeError, match="not present at t0"):
            run_confirm.run_confirm_smoke(
                FakeConfirmLauncher(),
                out_dir=tmp_path / "confirm_missing",
                scenario_dir=confirm_scenario,
                base_seed=8468,
                cav_ids_arg="veh2",  # departs at 31 > t0
            )

    # ------------------------- PL-04h: the roster source contract ----------
    def test_no_roster_source_refused_by_name(self, tmp_path, confirm_scenario, capsys):
        """The removed default: NEITHER --base-run nor --cav-ids -> the named
        refusal, both at the API and the CLI level."""
        with pytest.raises(run_confirm.RosterSourceRequiredError, match="--base-run"):
            run_confirm.run_confirm_smoke(
                FakeConfirmLauncher(),
                out_dir=tmp_path / "confirm_nosource",
                scenario_dir=confirm_scenario,
                base_seed=8468,
            )
        code = run_confirm.main(
            [
                "--out", str(tmp_path / "confirm_cli"),
                "--scenario-dir", str(confirm_scenario),
                "--base-seed", "8468",
            ]
        )
        assert code == 2
        assert "RosterSourceRequiredError" in capsys.readouterr().err

    def test_base_run_roster_rule(self, tmp_path):
        base_run = _write_base_run(tmp_path / "campaign_base")
        roster, evidence = run_confirm.roster_from_base_run(base_run, 30.0)
        assert roster == ("veh0", "veh3")
        assert evidence["source"] == str(base_run)
        assert "damping command row" in evidence["rule"]
        assert "inserted_s <= t0" in evidence["rule"]
        assert evidence["telemetry_file"] == str(base_run / "run" / "telemetry.jsonl")
        assert evidence["insertions_file"] == str(base_run / "run" / "insertions.jsonl")
        assert evidence["designated_cav_count_total"] == 2  # hdv.9 had NO command row
        assert evidence["command_rows_read"] == 3  # the fleet_switch row is skipped
        assert evidence["insertions_read"] == 4
        assert evidence["present_at_t0_s"] == 30.0

    def test_base_run_missing_evidence_refused(self, tmp_path):
        base_run = tmp_path / "empty_base"
        base_run.mkdir()
        with pytest.raises(run_confirm.ConfirmSmokeError, match="missing campaign evidence"):
            run_confirm.roster_from_base_run(base_run, 30.0)

    def test_base_run_empty_roster_refused(self, tmp_path):
        base_run = _write_base_run(tmp_path / "campaign_base")
        with pytest.raises(run_confirm.ConfirmSmokeError, match="EMPTY roster"):
            run_confirm.roster_from_base_run(base_run, 4.0)  # every inserted_s > 4.0


# --------------------------------------------------- run_cl_smoke.py fakes
SEED = run_cl_smoke.SMOKE_SEED


def _write_fake_run(
    run_dir: Path,
    *,
    non_native=True,
    with_receipt=True,
    seed=SEED,
    run_purpose=None,
    initial_plan=None,
    control_command=None,
):
    """A canned REAL-shaped closed-loop run dir for the verdict reader.
    PL-04f: ``run_purpose``/``initial_plan`` stamp the receipt and the
    decisions row (control-path proof runs); ``control_command`` overrides
    the master_control_tick command (the proof mode judges those rows
    directly, so an all-native decision run with a damping control row is
    its PASS shape)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    actions = (
        {"mainline.6": "damping", "mainline.7": "native"}
        if non_native
        else {"mainline.6": "native", "mainline.7": "native"}
    )
    if with_receipt:
        receipt = {
            "schema": "pl04_closed_loop_receipt/1",
            "run_id": run_dir.name,
            "world": "w8468m2600p40t240",
            "seed": seed,
            "predictor": "perfect_sim",
            "predictor_class": "oracle",
            "horizon_s": 15.0,
            "walls": "off",
            "max_decision_units": 2,
            "cav_roster": ["mainline.6", "mainline.7"],
            "start_mode": "base_state",
            "base_run": "E:/research-dense/runs/campaign_stage1_chunk1/w8468m2600p40t240__base",
            "base_t0_s": 240.0,
            "base_state_sha256": "b" * 64,
            "sumo_launches": [
                {
                    "kind": "master",
                    "label": "pl04_master",
                    "port": 9000,
                    "pid": 111,
                    "cmd": ["sumo", "-c", "cfg.sumocfg", "--seed", str(seed)],
                },
                {
                    "kind": "branch",
                    "label": "pl04_branch1",
                    "port": 9001,
                    "pid": 112,
                    "cmd": ["sumo", "-c", "cfg.sumocfg", "--seed", str(seed)],
                },
            ],
            "branch_control_rows": {"count": 2, "by_branch": {"1:plan": 2}},
            "sim_s": {"master_s": 15.0, "lookahead_s": 60.0, "total_s": 75.0},
            "decisions": 1,
            "exit_status": "complete",
        }
        if run_purpose is not None:
            receipt["run_purpose"] = run_purpose
        if initial_plan is not None:
            receipt["initial_plan"] = initial_plan
        (run_dir / "RUN_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
        )
    decision_row = {
        "schema": "pl04_decision/1",
        "t_s": 240.0,
        "predictor": "perfect_sim",
        "predictor_class": "oracle",
        "actions": actions,
        "walls": [],
        "abstain": False,
    }
    if run_purpose is not None:
        decision_row["run_purpose"] = run_purpose
    (run_dir / "decisions.jsonl").write_text(
        json.dumps(decision_row) + "\n", encoding="utf-8"
    )
    dispatcher = [
        {"kind": "command", "t_s": 240.0, "cav": "mainline.6", "command": actions["mainline.6"]},
    ]
    dispatcher.append(
        {
            "kind": "control",
            "t_s": 241.0,
            "cav": "mainline.6",
            "command": control_command or ("damping" if non_native else "native"),
            "owner": "planner",
            "source": "master_control_tick",
        }
    )
    (run_dir / "dispatcher_rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in dispatcher), encoding="utf-8"
    )
    (run_dir / "physics_steps.jsonl").write_text(
        json.dumps(
            {
                "kind": "physics",
                "t_s": 241.0,
                "dt_s": 0.5,
                "vehicles": {"mainline.6": {"s_m": 12.5, "v_mps": 19.0}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if non_native:
        (run_dir / "branch_control_rows.jsonl").write_text(
            json.dumps(
                {
                    "kind": "branch_control",
                    "branch": "1:plan",
                    "t_s": 240.0,
                    "cav": "mainline.6",
                    "command": "damping",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    manifest = {"schema": "wm41_saved_state/1", "seed": seed, "branch_point_t0_s": 240.0}
    for rel in (
        "checkpoints/ckpt_000000/master_state/attempt_000001/decision_000001/saved_state",
        "lookahead_state/p1/decision_000001/saved_state",
    ):
        path = run_dir / rel
        path.mkdir(parents=True, exist_ok=True)
        (path / "STATE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")


class TestRunClSmoke:
    def test_pass_verdict_and_exact_argv(self, tmp_path):
        base = make_base_run(tmp_path, world="w8468m2600p40t240", seed=SEED)
        out = tmp_path / "cl_out"
        calls = []

        def fake_launcher(argv, cwd):
            calls.append(list(argv))
            run_dir = out / "w8468m2600p40t240__cl_smoke"
            _write_fake_run(run_dir, non_native=True)
            return {"returncode": 0, "stdout": "", "stderr": ""}

        code = run_cl_smoke.main(
            ["--out", str(out), "--base-run", str(base)], launcher=fake_launcher
        )
        assert code == 0
        assert len(calls) == 1
        argv = calls[0]
        # the exact PL-04e2 real-run flags (the selected split's times)
        def flagged(flag):
            return argv[argv.index(flag) + 1]

        assert argv[0] == "--components" and argv[1] == "real"
        assert flagged("--predictor") == "perfect_sim"
        assert flagged("--base-run") == str(base)
        assert flagged("--seed") == str(SEED)
        assert flagged("--walls") == "off"
        assert flagged("--max-decision-units") == "2"
        assert flagged("--max-sim-s") == "80.0"
        assert flagged("--world") == "w8468m2600p40t240"
        # the real roster (2 damping CAVs) projects 270/150/75: the LAST rung
        # fits the 80 s budget
        assert flagged("--horizon-s") == "15.0"
        assert flagged("--decision-period-s") == "15.0"
        assert flagged("--lookahead-horizon-s") == "5.0"
        assert flagged("--lookahead-commit-s") == "5.0"
        verdict = json.loads(
            (out / "CL_SMOKE_VERDICT.json").read_text(encoding="utf-8")
        )
        assert verdict["verdict"] == "PASS"
        assert verdict["a_cav_command_differs_from_native"] is True
        assert verdict["a_evidence"]["cav"] == "mainline.6"
        assert verdict["a_evidence"]["command"] == "damping"
        assert verdict["a_evidence"]["t"] == 241.0
        assert verdict["b_branch_control_rows_nonempty_for_nonnative"] is True
        assert verdict["b_counts"]["branch_control_rows"] == 2
        assert verdict["c_world_seed_everywhere"] is True
        assert verdict["c_seeds_seen"] == [SEED]
        assert verdict["receipt_complete"] is True
        assert verdict["sim_s_total"] == 75.0
        assert verdict["closed_loop_exit"] == 0
        # the projection was printed and carried into the verdict
        assert verdict["projections"][-1]["projected_sim_s"] == 75.0
        assert verdict["selected_split"]["horizon_s"] == 15.0

    def test_all_native_decisions_are_inconclusive(self, tmp_path):
        base = make_base_run(tmp_path, world="w8468m2600p40t240", seed=SEED)
        out = tmp_path / "cl_inconclusive"

        def fake_launcher(argv, cwd):
            run_dir = out / "w8468m2600p40t240__cl_smoke"
            _write_fake_run(run_dir, non_native=False)
            return {"returncode": 0, "stdout": "", "stderr": ""}

        code = run_cl_smoke.main(
            ["--out", str(out), "--base-run", str(base)], launcher=fake_launcher
        )
        assert code == 1
        verdict = json.loads(
            (out / "CL_SMOKE_VERDICT.json").read_text(encoding="utf-8")
        )
        assert verdict["verdict"] == "INCONCLUSIVE"
        assert verdict["a_cav_command_differs_from_native"] is False
        assert any("native at every decision point" in reason for reason in verdict["reasons"])

    def test_missing_receipt_fails(self, tmp_path):
        base = make_base_run(tmp_path, world="w8468m2600p40t240", seed=SEED)
        out = tmp_path / "cl_fail"

        def fake_launcher(argv, cwd):
            run_dir = out / "w8468m2600p40t240__cl_smoke"
            _write_fake_run(run_dir, with_receipt=False)
            return {"returncode": 0, "stdout": "", "stderr": ""}

        code = run_cl_smoke.main(
            ["--out", str(out), "--base-run", str(base)], launcher=fake_launcher
        )
        assert code == 1
        verdict = json.loads(
            (out / "CL_SMOKE_VERDICT.json").read_text(encoding="utf-8")
        )
        assert verdict["verdict"] == "FAIL"
        assert verdict["receipt_complete"] is False

    def test_seed_drift_fails(self, tmp_path):
        base = make_base_run(tmp_path, world="w8468m2600p40t240", seed=SEED)
        out = tmp_path / "cl_seed"

        def fake_launcher(argv, cwd):
            run_dir = out / "w8468m2600p40t240__cl_smoke"
            _write_fake_run(run_dir, seed=999)
            return {"returncode": 0, "stdout": "", "stderr": ""}

        code = run_cl_smoke.main(
            ["--out", str(out), "--base-run", str(base)], launcher=fake_launcher
        )
        assert code == 1
        verdict = json.loads(
            (out / "CL_SMOKE_VERDICT.json").read_text(encoding="utf-8")
        )
        assert verdict["verdict"] == "FAIL"
        assert verdict["c_world_seed_everywhere"] is False
        # every drifted artifact consistently says 999: the wrapper's world
        # seed (8468) is nowhere, which is exactly the failure
        assert verdict["c_seeds_seen"] == [999]

    def test_no_split_fits_the_hard_budget_refuses_before_launch(self, tmp_path, capsys):
        base = make_base_run(tmp_path, world="w8468m2600p40t240", seed=SEED)
        out = tmp_path / "cl_budget"
        calls = []

        def fake_launcher(argv, cwd):
            calls.append(argv)
            return {"returncode": 0, "stdout": "", "stderr": ""}

        code = run_cl_smoke.main(
            ["--out", str(out), "--base-run", str(base), "--budget-s", "50"],
            launcher=fake_launcher,
        )
        assert code == 2
        assert calls == []  # nothing was launched
        text = capsys.readouterr().err
        assert "no split of the ladder fits" in text

    def test_out_dir_guard(self, tmp_path):
        with pytest.raises(run_cl_smoke.OutDirRefusedError, match="runs/"):
            run_cl_smoke.resolve_out_dir("E:/research-dense/runs/x")
        with pytest.raises(run_cl_smoke.OutDirRefusedError, match="output/planner_v1"):
            run_cl_smoke.resolve_out_dir(PLANNER_ROOT / "output" / "planner_v1" / "x")


# ------------------- PL-04f: --control-path-proof (fakes only, NO launches)
class TestRunClSmokeControlPathProof:
    """The control-path-proof smoke: --initial-plan damping is passed, the
    <=80 sim-s projection check is kept, criterion (a) is judged from the
    REAL master_control_tick command rows only (non-native at/after t0),
    the verdict carries label='control_path_proof' and INCONCLUSIVE is
    impossible (missing (a) -> FAIL). The default (no flag) behaviour --
    including its verdict bytes and the planner-smoke label -- is
    unchanged."""

    def test_proof_pass_with_all_native_decisions_and_exact_argv(self, tmp_path):
        base = make_base_run(tmp_path, world="w8468m2600p40t240", seed=SEED)
        out = tmp_path / "cl_proof"
        calls = []

        def fake_launcher(argv, cwd):
            calls.append(list(argv))
            run_dir = out / "w8468m2600p40t240__cl_smoke"
            _write_fake_run(
                run_dir,
                non_native=False,  # the PLANNER decided native everywhere ...
                run_purpose="control_path_proof",
                initial_plan="damping",
                control_command="damping",  # ... yet the pre-set plan reached
                # the control path: a REAL non-native command at/after t0
            )
            return {"returncode": 0, "stdout": "", "stderr": ""}

        code = run_cl_smoke.main(
            ["--out", str(out), "--base-run", str(base), "--control-path-proof"],
            launcher=fake_launcher,
        )
        assert code == 0
        argv = calls[0]

        def flagged(flag):
            return argv[argv.index(flag) + 1]

        # the proof flag is on the argv; the HARD <=80 sim-s check is kept
        assert flagged("--initial-plan") == "damping"
        assert flagged("--max-sim-s") == "80.0"
        assert flagged("--base-run") == str(base)
        assert flagged("--seed") == str(SEED)
        assert flagged("--horizon-s") == "15.0"  # the same split ladder
        verdict = json.loads(
            (out / "CL_SMOKE_VERDICT.json").read_text(encoding="utf-8")
        )
        assert verdict["label"] == "control_path_proof"
        assert verdict["verdict"] == "PASS"
        assert verdict["a_cav_command_differs_from_native"] is True
        assert verdict["a_evidence"]["cav"] == "mainline.6"
        assert verdict["a_evidence"]["command"] == "damping"
        assert verdict["a_evidence"]["t"] == 241.0
        assert verdict["a_evidence"]["initial_plan_t0_s"] == 240.0
        assert verdict["b_branch_control_rows_nonempty_for_nonnative"] is True
        assert verdict["c_world_seed_everywhere"] is True
        assert verdict["receipt_complete"] is True
        assert verdict["sim_s_total"] == 75.0
        assert "reasons" not in verdict

    def test_proof_without_nonnative_command_fails_never_inconclusive(self, tmp_path):
        base = make_base_run(tmp_path, world="w8468m2600p40t240", seed=SEED)
        out = tmp_path / "cl_proof_fail"

        def fake_launcher(argv, cwd):
            run_dir = out / "w8468m2600p40t240__cl_smoke"
            _write_fake_run(
                run_dir,
                non_native=False,
                run_purpose="control_path_proof",
                initial_plan="damping",
                # control_command stays native: the pre-set plan did NOT
                # reach the control path
            )
            return {"returncode": 0, "stdout": "", "stderr": ""}

        code = run_cl_smoke.main(
            ["--out", str(out), "--base-run", str(base), "--control-path-proof"],
            launcher=fake_launcher,
        )
        assert code == 1
        verdict = json.loads(
            (out / "CL_SMOKE_VERDICT.json").read_text(encoding="utf-8")
        )
        assert verdict["label"] == "control_path_proof"
        assert verdict["verdict"] == "FAIL"  # INCONCLUSIVE is impossible here
        assert verdict["a_cav_command_differs_from_native"] is False
        assert any(
            "did not reach the control path" in reason for reason in verdict["reasons"]
        )

    def test_proof_shaped_run_without_the_flag_stays_planner_inconclusive(self, tmp_path):
        """Default-mode byte-for-byte: a proof-shaped run judged WITHOUT the
        flag gets the unchanged planner-smoke verdict (no label key,
        INCONCLUSIVE when every decision is native)."""
        base = make_base_run(tmp_path, world="w8468m2600p40t240", seed=SEED)
        out = tmp_path / "cl_default"

        def fake_launcher(argv, cwd):
            assert "--initial-plan" not in argv  # the default argv is unchanged
            run_dir = out / "w8468m2600p40t240__cl_smoke"
            _write_fake_run(
                run_dir,
                non_native=False,
                run_purpose="control_path_proof",
                initial_plan="damping",
                control_command="damping",
            )
            return {"returncode": 0, "stdout": "", "stderr": ""}

        code = run_cl_smoke.main(
            ["--out", str(out), "--base-run", str(base)], launcher=fake_launcher
        )
        assert code == 1
        verdict = json.loads(
            (out / "CL_SMOKE_VERDICT.json").read_text(encoding="utf-8")
        )
        assert "label" not in verdict  # the planner-smoke output is unchanged
        assert verdict["verdict"] == "INCONCLUSIVE"
        assert any("native at every decision point" in reason for reason in verdict["reasons"])


# ------- PL-04g: the REAL Popen/run paths over a harmless python child
class TestRealSubprocessLaunchers:
    """PL-04g: every real launcher seam is exercised against a REAL child
    process -- the SUMO binary/command is monkeypatched to a harmless
    ``python -c`` payload, NEVER Popen itself (faked Popen hid the
    run_confirm Path-as-stream crash: Popen(stdout=<WindowsPath>) raised
    AttributeError 'WindowsPath' object has no attribute 'fileno' before
    any child started). Only the launchers' OWN Popen children are ever
    waited here; no SUMO, no signalling by image name or pattern."""

    CHILD = "import sys; sys.stdout.write('ok'); sys.stderr.write('err')"

    @staticmethod
    def _install_traci_stub(monkeypatch, *, fail_init=False):
        """A sys.modules['traci'] stub: init/getConnection only (Popen itself
        is NEVER patched -- the launchers run their REAL subprocess path)."""
        conn = SimpleNamespace(close=lambda wait=True: None, closed=False)

        class _TraciStub:
            calls: list = []

            @staticmethod
            def init(**kwargs):
                _TraciStub.calls.append(kwargs)
                if fail_init:
                    proc = kwargs.get("proc")
                    if proc is not None:
                        try:
                            proc.wait(timeout=30)  # a dead child's diag is
                        except Exception:  # noqa: BLE001
                            pass  # already flushed when init gives up
                    raise ConnectionError("no traci server (stub)")

            @staticmethod
            def getConnection(label):
                return conn

        monkeypatch.setitem(sys.modules, "traci", _TraciStub)
        return conn, _TraciStub

    # -- run_confirm.ConfirmLauncher (the fixed Path-as-stream seam) -------
    def test_confirm_start_success_keeps_handle_and_close_closes_it(
        self, tmp_path, monkeypatch
    ):
        conn, _stub = self._install_traci_stub(monkeypatch)
        launcher = run_confirm.ConfirmLauncher(
            sumo_binary=sys.executable,
            cfg="<payload below>",  # the -c payload; monkeypatched, not Popen
            seed=1,
            diag_dir=tmp_path,
        )
        launcher.cfg = self.CHILD
        diag_files = sorted(tmp_path.glob("sumo_launch_*.txt"))
        assert diag_files == []
        got = launcher.start("master")
        assert got is conn
        assert len(launcher.launches) == 1
        (_label, proc, handle) = next(iter(launcher._children.values()))
        assert proc.pid > 0  # a REAL child process started
        assert not handle.closed  # the diag handle is kept with the child
        launcher.close("master", conn)
        assert proc.wait(timeout=0) == 0  # exits 0 (already reaped by close)
        assert handle.closed  # closed once the child is waited
        text = next(iter(tmp_path.glob("sumo_launch_*.txt"))).read_text("utf-8")
        assert "ok" in text and "err" in text  # both streams landed in the diag

    def test_confirm_start_retry_closes_handle_and_raises_diag_error(
        self, tmp_path, monkeypatch
    ):
        self._install_traci_stub(monkeypatch, fail_init=True)
        launcher = run_confirm.ConfirmLauncher(
            sumo_binary=sys.executable,
            cfg="<payload below>",
            seed=1,
            diag_dir=tmp_path,
        )
        launcher.cfg = self.CHILD
        with pytest.raises(run_confirm.ConfirmSmokeError, match="SUMO exited on launch"):
            launcher.start("master")
        assert launcher._children == {}  # nothing stayed registered
        text = next(iter(tmp_path.glob("sumo_launch_*.txt"))).read_text("utf-8")
        assert "ok" in text and "err" in text  # the REAL child's output

    # -- run_cl_smoke.subprocess_launcher ----------------------------------
    def test_cl_smoke_subprocess_launcher_runs_a_real_mock_child(self):
        with tempfile.TemporaryDirectory(dir=scratch_root()) as scratch:
            out = str(Path(scratch).resolve())
            result = run_cl_smoke.subprocess_launcher(
                [
                    "--components", "mock",
                    "--predictor", "mock",
                    "--run-id", "pl04g_child",
                    "--world", "w-pl04g",
                    "--horizon-s", "100",
                    "--lookahead-horizon-s", "30",
                    "--max-sim-s", "100000",
                    "--out", out,
                ],
                cwd=str(PLANNER_ROOT),
            )
            assert result["returncode"] == 0, result["stderr"]
            assert '"receipt"' in result["stdout"]
            assert (Path(out) / "pl04g_child" / "RUN_RECEIPT.json").is_file()
