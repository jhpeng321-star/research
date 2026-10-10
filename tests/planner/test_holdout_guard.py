"""PL-12 tests: the HOLDOUT exclusion guard wired into the planner.

The reconstructor-owned ``tools/holdout_guard.py`` +
``protocols/holdout_exclusions.json`` are the SINGLE SOURCE OF TRUTH: every
test here installs a tmp_path FIXTURE copy of the guard code plus a tiny
fixture exclusions JSON via ``RESEARCH_DENSE_ROOT`` and never reads (or
copies) the real list.  Zero simulation; no real HOLDOUT outcome file is
opened (the report refusal is proven with a path spy).
"""

import builtins
import io
import json
import os
from pathlib import Path

import pytest

from traffic_shepherd.planner import holdout as holdout_guard
from traffic_shepherd.planner.closed_loop import ClosedLoopConfig, SplitRefusedError

pytest.importorskip("output.pl02a_offline_v1.report_chunk1_final")
from output.pl02a_offline_v1 import report_chunk1_final as rcf1  # noqa: E402
from output.pl02a_offline_v1 import report_chunk_final as rcfn  # noqa: E402
from output.pl02a_offline_v1 import run as pl02a_run  # noqa: E402

pytest.importorskip(
    "output.planner_v1.plexp1_analysis.analyze_plexp1",
    reason="PL-EXP1 analyzer module (output/planner_v1) lives in the planner worktree only",
)
from output.planner_v1.plexp1_analysis import analyze_plexp1  # noqa: E402

from tests.planner import test_recorded as tr  # noqa: E402
from traffic_shepherd.planner.recorded import Layout  # noqa: E402

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

EXCLUDED = "wx_excluded_holdout"
DISCLOSED = "wx_disclosed_kept"
PLAIN_HOLDOUT = "wx_clean_holdout"

_EXCLUSIONS = {
    "schema": "holdout_exclusions/1",
    "rule": "tmp fixture ONLY (never the real list)",
    "expected_excluded_count": 1,
    "excluded": [{"world": EXCLUDED, "reason": "fixture: exposed root"}],
    "kept_disclosed": [DISCLOSED],
}


def _install_guard(
    tmp_path,
    monkeypatch,
    *,
    exclusions=None,
    guard_source=True,
    broken_guard=False,
    missing=False,
):
    """Point RESEARCH_DENSE_ROOT at tmp_path holding a FIXTURE guard copy.

    The workspace's preserved network is copied (read-only citation) so the
    report pipeline's p4 geometry keeps working under the fixture root."""
    root = tmp_path / "rdroot"
    (root / "tools").mkdir(parents=True)
    if not missing:
        if guard_source:
            real = _WORKSPACE_ROOT / "tools" / "holdout_guard.py"
            text = real.read_text(encoding="utf-8")
        else:
            text = "raise RuntimeError('not the real guard')\n"
        if broken_guard:
            text = "def broken(:\n"
        (root / "tools" / "holdout_guard.py").write_text(text, encoding="utf-8")
    if exclusions is not None:
        (root / "protocols").mkdir(parents=True, exist_ok=True)
        (root / "protocols" / "holdout_exclusions.json").write_text(
            json.dumps(exclusions), encoding="utf-8"
        )
    net = _WORKSPACE_ROOT / "assets" / "research" / "highway_merge_shepherd_long" / (
        "osm.net.xml.gz"
    )
    if net.is_file():
        net_dir = root / "assets" / "research" / "highway_merge_shepherd_long"
        net_dir.mkdir(parents=True, exist_ok=True)
        (net_dir / net.name).write_bytes(net.read_bytes())
    monkeypatch.setenv("RESEARCH_DENSE_ROOT", str(root))
    return root


@pytest.fixture()
def guard_env(tmp_path, monkeypatch):
    """Fixture guard: EXCLUDED exposed, DISCLOSED kept-with-disclosure."""
    _install_guard(tmp_path, monkeypatch, exclusions=_EXCLUSIONS)
    return tmp_path


# ----------------------------------------------------------------- helper
class TestGuardLoader:
    def test_load_guard_resolves_via_env_root(self, guard_env):
        assert holdout_guard.guard_path() == (
            Path(os.environ["RESEARCH_DENSE_ROOT"]) / "tools" / "holdout_guard.py"
        )
        guard = holdout_guard.load_guard()
        data = guard.load_holdout_exclusions()
        assert guard.excluded_roots(data) == {EXCLUDED}
        assert holdout_guard.holdout_disclosures() == [DISCLOSED]
        assert holdout_guard.load_guard().filter_final_holdout(
            [f"{EXCLUDED}__pi", f"{PLAIN_HOLDOUT}__base"]
        ) == [f"{PLAIN_HOLDOUT}__base"]

    def test_assert_refuses_excluded_world_and_run_names(self, guard_env):
        with pytest.raises(ValueError, match=EXCLUDED):
            holdout_guard.assert_holdout_eligible(
                [f"{EXCLUDED}__base", f"{PLAIN_HOLDOUT}__pi"], holdout_requested=True
            )
        assert (
            holdout_guard.assert_holdout_eligible(
                [f"{PLAIN_HOLDOUT}__base", "wx_train"], holdout_requested=True
            )
            is not None
        )

    # (d) missing guard + HOLDOUT requested -> named error; not requested -> OK
    def test_missing_guard_fails_closed_only_when_requested(self, tmp_path, monkeypatch):
        _install_guard(tmp_path, monkeypatch, missing=True)
        with pytest.raises(holdout_guard.HoldoutGuardUnavailableError, match="missing"):
            holdout_guard.load_guard()
        with pytest.raises(holdout_guard.HoldoutGuardUnavailableError):
            holdout_guard.assert_holdout_eligible(["w"], holdout_requested=True)
        assert (
            holdout_guard.assert_holdout_eligible(["w"], holdout_requested=False) is None
        )
        assert holdout_guard.filter_holdout(["w"], holdout_requested=False) == ["w"]

    def test_unparseable_guard_fails_closed_only_when_requested(self, tmp_path, monkeypatch):
        _install_guard(tmp_path, monkeypatch, broken_guard=True)
        with pytest.raises(holdout_guard.HoldoutGuardUnavailableError, match="failed to load"):
            holdout_guard.assert_holdout_eligible(["w"], holdout_requested=True)
        assert (
            holdout_guard.assert_holdout_eligible(["w"], holdout_requested=False) is None
        )


# ------------------------------------------------------------ path spy
@pytest.fixture()
def open_spy(monkeypatch):
    opened = []
    real_builtins_open = builtins.open
    real_io_open = io.open

    def spy(file, *args, **kwargs):
        try:
            opened.append(os.fspath(file))
        except TypeError:
            opened.append(repr(file))
        return real_builtins_open(file, *args, **kwargs)

    def spy_io(file, *args, **kwargs):
        try:
            opened.append(os.fspath(file))
        except TypeError:
            opened.append(repr(file))
        return real_io_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy_io, raising=False)
    return opened


def _plan_with_excluded_holdout(tmp_path):
    plan = tmp_path / "world_plan" / "WORLD_PLAN.json"
    tr.write_world_plan(
        plan,
        [
            tr.world_entry("wx_train", arms=("fixed1",)),
            tr.world_entry(EXCLUDED, split="HOLDOUT", arms=("fixed1",)),
        ],
    )
    return plan


# (a) the pl02a report refuses an excluded world BEFORE any outcome is read
class TestPl02aReportGuard:
    def test_chunk1_report_refuses_excluded_holdout_before_outcomes(
        self, guard_env, tmp_path, open_spy
    ):
        plan = _plan_with_excluded_holdout(tmp_path)
        runs_root = tmp_path / "runs"  # empty: nothing may be read at all
        with pytest.raises(ValueError, match=EXCLUDED):
            rcf1.build_report(
                runs_root=runs_root,
                world_plan=plan,
                tally_path=tmp_path / "absent_tally.txt",
                chunk_summary=None,
                out_dir=tmp_path / "out",
                cache_dir=tmp_path / "cache",
                splits=("HOLDOUT",),
                layout=Layout(),
            )
        assert not any(EXCLUDED in p for p in open_spy)
        assert not any(str(runs_root) in p for p in open_spy)

    def test_generic_report_refuses_excluded_holdout_before_outcomes(
        self, guard_env, tmp_path, open_spy
    ):
        plan = _plan_with_excluded_holdout(tmp_path)
        runs_root = tmp_path / "runs"
        with pytest.raises(ValueError, match=EXCLUDED):
            rcfn.build_report(
                runs_root=runs_root,
                world_plan=plan,
                tally_path=None,
                chunk_summary=None,
                out_dir=tmp_path / "out",
                cache_dir=tmp_path / "cache",
                splits=("TRAIN", "HOLDOUT"),
                layout=Layout(),
                spec=rcfn.CHUNKS[1],
            )
        assert not any(EXCLUDED in p for p in open_spy)

    def test_main_with_owner_flag_still_refuses_excluded(self, guard_env, tmp_path):
        plan = _plan_with_excluded_holdout(tmp_path)
        out = tmp_path / "out"
        with pytest.raises(ValueError, match=EXCLUDED):
            rcf1.main(
                [
                    "--runs-root", str(tmp_path / "runs"),
                    "--plan", str(plan),
                    "--out", str(out / "final"),
                    "--cache", str(out / "cache"),
                    "--tally", str(out / "TALLY.txt"),
                    "--splits", "TRAIN,HOLDOUT",
                    "--owner-allow-holdout",
                ]
            )
        assert not list(out.rglob("RESULTS.json"))

    def test_report_without_holdout_runs_without_guard(self, tmp_path, monkeypatch, open_spy):
        """(d) counterpart: no HOLDOUT in --splits -> no guard file needed;
        the build proceeds past the guard (and only then fails on the
        deliberately-absent campaign/tally fixture)."""
        _install_guard(tmp_path, monkeypatch, missing=True)
        plan = _plan_with_excluded_holdout(tmp_path)
        with pytest.raises(Exception) as excinfo:  # noqa: PT011 - any failure BUT the guard's
            rcf1.build_report(
                runs_root=tmp_path / "runs",
                world_plan=plan,
                tally_path=tmp_path / "absent_tally.txt",
                chunk_summary=None,
                out_dir=tmp_path / "out",
                cache_dir=tmp_path / "cache",
                splits=("TRAIN", "CALIBRATION"),
                layout=Layout(),
            )
        assert not isinstance(excinfo.value, holdout_guard.HoldoutGuardUnavailableError)

    # (a) run.py scoped smoke: --allow-non-train no longer admits the exposed world
    def test_run_allow_non_train_refuses_excluded_holdout(self, guard_env, tmp_path):
        plan = _plan_with_excluded_holdout(tmp_path)
        with pytest.raises(ValueError, match=EXCLUDED):
            pl02a_run.enforce_train_only(
                plan, roots_scope=[f"{EXCLUDED}__base"], allow_non_train=True
            )

    def test_run_allow_non_train_calibration_ok_without_guard(self, tmp_path, monkeypatch):
        _install_guard(tmp_path, monkeypatch, missing=True)
        plan = tmp_path / "world_plan" / "WORLD_PLAN.json"
        tr.write_world_plan(
            plan,
            [
                tr.world_entry("wx_train", arms=("fixed1",)),
                tr.world_entry("wx_calib", split="CALIBRATION", arms=("fixed1",)),
            ],
        )
        logged = []
        pl02a_run.enforce_train_only(
            plan,
            roots_scope=["wx_calib__base"],
            allow_non_train=True,
            log=logged.append,
        )
        assert logged and "wx_calib" in logged[0]


# (b) the plexp1 analyzer calls the guard defensively
class TestPlexp1AnalyzerGuard:
    def _train_worlds(self, world):
        return [
            {
                "chunk": "c2",
                "world": world,
                "seed": 7,
                "penetration": 0.2,
                "loading_veh_h": 2400,
                "t0_s": 60.0,
                "split": "TRAIN",
            }
        ]

    def test_analyzer_refuses_excluded_world(self, guard_env):
        with pytest.raises(analyze_plexp1.NonTrainWorldRefusedError, match=EXCLUDED):
            analyze_plexp1.refuse_excluded_worlds(self._train_worlds(EXCLUDED))
        assert analyze_plexp1.refuse_excluded_worlds(self._train_worlds("wx_ok")) is None

    def test_analyzer_train_only_ok_with_missing_guard(self, tmp_path, monkeypatch):
        _install_guard(tmp_path, monkeypatch, missing=True)
        assert analyze_plexp1.refuse_excluded_worlds(self._train_worlds("wx_ok")) is None

    def test_load_worlds_still_refuses_non_train_first(self, guard_env, tmp_path):
        worlds = self._train_worlds(EXCLUDED)
        worlds[0]["split"] = "HOLDOUT"
        path = tmp_path / "worlds.json"
        path.write_text(json.dumps({"worlds": worlds}), encoding="utf-8")
        with pytest.raises(analyze_plexp1.NonTrainWorldRefusedError, match="TRAIN"):
            analyze_plexp1.load_worlds(path)


# (c) closed_loop refuses an excluded world even with --owner-allow-holdout
class TestClosedLoopGuard:
    def _config(self, world, allow=True, split="train"):
        return ClosedLoopConfig(
            out_root=Path("E:/rd-wt/pl34/output"),
            run_id="pl12",
            world=world,
            split=split,
            allow_holdout=allow,
            max_sim_s=1e9,
        )

    def test_excluded_world_refused_with_owner_flag(self, guard_env):
        with pytest.raises(ValueError, match=EXCLUDED):
            self._config(EXCLUDED)
        with pytest.raises(ValueError, match=EXCLUDED):
            self._config(f"{EXCLUDED}__rerun")  # run-name style also refused

    def test_clean_holdout_world_passes_with_owner_flag(self, guard_env):
        assert self._config(PLAIN_HOLDOUT).world == PLAIN_HOLDOUT

    def test_missing_guard_fails_closed_when_holdout_requested(self, tmp_path, monkeypatch):
        _install_guard(tmp_path, monkeypatch, missing=True)
        with pytest.raises(holdout_guard.HoldoutGuardUnavailableError):
            self._config("some_HOLDOUT_world")
        with pytest.raises(SplitRefusedError, match="HOLDOUT"):
            self._config("some_HOLDOUT_world", allow=False)
        # HOLDOUT not requested: a missing guard never fails the config
        assert self._config("wx_plain_world").world == "wx_plain_world"
        assert self._config("wx_plain_world", allow=False).world == "wx_plain_world"


# (e) kept_disclosed worlds are listed in a HOLDOUT report
class TestHoldoutDisclosureListing:
    @pytest.fixture()
    def full_campaign(self, tmp_path):
        """The proven six-world chunk-1 fixture campaign (copied from
        tests.planner.test_recorded.TestChunk1FinalReport.final_campaign)."""
        runs_root = tmp_path / "runs"
        window = 20.0
        schedule = {"events": [{"t_s": tr.T0, "per_cav": {"c0": "pi-external-u"}}]}

        def colliding_row(t):
            return {
                "time_s": t,
                "vehicles": {
                    "cav_f": tr.veh(lane="lane0", pos=90.0, speed=3.0, accel=-3.0)
                    | {"type_id": "cautious_car"},
                    "lead": tr.veh(lane="lane0", pos=100.0, speed=0.0, accel=0.0)
                    | {"type_id": "normal_car"},
                },
                "colliding_ids": ["cav_f", "lead"],
                "cav_actions": {"cav_f": {"action": "native"}},
                "cav_ids": ["cav_f"],
                "arrived_ids": [],
            }

        def world_runs(label, *, base_collide_t=None, fixed_collide_t=None,
                       native_collide_t=None, fixed_speed=20.0, fixed_arrivals=False,
                       fixed_hard_brake=False):
            base_rows = tr.steady_rows(0.0, tr.T0 + window, ("v1",), cav_ids=("c0",))
            if base_collide_t is not None:
                base_rows = (
                    [row for row in base_rows if row["time_s"] < base_collide_t]
                    + [colliding_row(base_collide_t)]
                    + [
                        row
                        for row in tr.steady_rows(
                            base_collide_t + tr.ROW_DT, tr.T0 + window, ("v1",),
                            cav_ids=("c0",),
                        )
                    ]
                )
            tr.write_run(
                runs_root, f"{label}__base", arm="base", is_base=True, rows=base_rows,
                rou_vehicles=[("v_late", tr.T0)],
            )
            native_rows = tr.steady_rows(tr.T0, tr.T0 + window, ("v1",), cav_ids=("c0",))
            if native_collide_t is not None:
                native_rows = [
                    row for row in native_rows if row["time_s"] < native_collide_t
                ] + [colliding_row(native_collide_t)]
            tr.write_run(
                runs_root, f"{label}__native", arm="native", is_base=False,
                rows=native_rows,
                branch_record={"arm": "native", "root_id": label,
                               "branch_point_t0_s": tr.T0, "seed": 1},
            )
            fixed_rows = tr.steady_rows(
                tr.T0, tr.T0 + window, ("v1",), speed=fixed_speed, cav_ids=("c0",)
            )
            if fixed_hard_brake:
                for row in fixed_rows[:4]:
                    row["vehicles"]["v1"]["acceleration_mps2"] = -4.0
            if fixed_collide_t is not None:
                fixed_rows = [
                    row for row in fixed_rows if row["time_s"] < fixed_collide_t
                ] + [colliding_row(fixed_collide_t)]
            tr.write_run(
                runs_root, f"{label}__fixed1", arm="fixed1", is_base=False,
                rows=fixed_rows,
                arrivals=[("v1", tr.T0 + 15.0)] if fixed_arrivals else None,
                schedule=schedule,
                dispatcher=[
                    {"t_s": tr.T0 + t, "actions": {"c0": "pi-external-u"}}
                    for t in (0.0, 5.0)
                ],
                branch_record={"arm": "fixed1", "root_id": label,
                               "branch_point_t0_s": tr.T0, "seed": 1},
            )

        world_runs("w1")
        world_runs("w3")
        world_runs("w2", base_collide_t=tr.T0 + 10.0)
        world_runs("w4", fixed_collide_t=tr.T0 + 5.0)
        world_runs("w5", native_collide_t=tr.T0 + 7.0, fixed_speed=25.0,
                   fixed_arrivals=True, fixed_hard_brake=True)
        world_runs("w6", base_collide_t=tr.T0 - 5.0)
        layout = Layout()
        (runs_root / "w4__fixed1" / layout.receipt_name).unlink()
        (runs_root / "w5__native" / layout.receipt_name).unlink()
        plan_path = tmp_path / "world_plan" / "WORLD_PLAN.json"
        tr.write_world_plan(
            plan_path,
            [
                tr.world_entry("w1", arms=("fixed1",)),
                tr.world_entry("w2", split="HOLDOUT", arms=("fixed1",)),
                tr.world_entry("w3", penetration=0.4, split="CALIBRATION",
                               arms=("fixed1",)),
                tr.world_entry("w4", arms=("fixed1",)),
                tr.world_entry("w5", arms=("fixed1",)),
                tr.world_entry("w6", split="CALIBRATION", arms=("fixed1",)),
            ],
        )
        tally_rows = [
            "w2__base | p20 | base | t0=60 | t=70.0 | in_window "
            "| CAV-behind-human | follower cav_f cautious_car v=3.00 a=-3.00 "
            "| leader lead normal_car v=0.00 | receipt=yes",
            "w4__fixed1 | p20 | fixed1 | t0=60 | t=65.0 | in_window "
            "| CAV-behind-human | follower cav_f cautious_car v=3.00 a=-3.00 "
            "| leader lead normal_car v=0.00 | receipt=NO",
            "w5__native | p20 | native | t0=60 | t=67.0 | in_window "
            "| CAV-behind-human | follower cav_f cautious_car v=3.00 a=-3.00 "
            "| leader lead normal_car v=0.00 | receipt=NO",
            "w6__base | p20 | base | t0=60 | t=55.0 | pre_t0 "
            "| CAV-behind-human | follower cav_f cautious_car v=3.00 a=-3.00 "
            "| leader lead normal_car v=0.00 | receipt=yes",
        ]
        tally = tmp_path / "TALLY_FINAL.txt"
        tally.write_text(
            "# test FINAL tally\n"
            "# run | penetration | arm | t0 | first tick | class | kind "
            "| follower | leader | receipt\n" + "\n".join(tally_rows) + "\n",
            encoding="utf-8",
        )
        return runs_root, plan_path, tally, tmp_path / "out"

    def test_kept_disclosed_listed_in_holdout_report(self, tmp_path, monkeypatch,
                                                     full_campaign):
        # fixture guard: nothing excluded, w2 KEPT but must be LISTED
        _install_guard(
            tmp_path,
            monkeypatch,
            exclusions={
                "schema": "holdout_exclusions/1",
                "expected_excluded_count": 0,
                "excluded": [],
                "kept_disclosed": ["w2"],
            },
        )
        runs_root, plan, tally, out = full_campaign
        results = rcf1.build_report(
            runs_root=runs_root,
            world_plan=plan,
            tally_path=tally,
            chunk_summary=None,
            out_dir=out / "final",
            cache_dir=out / "cache",
            splits=("HOLDOUT",),
            layout=Layout(),
        )
        report = (out / "final" / "REPORT.md").read_text(encoding="utf-8")
        assert "kept_disclosed" in report and "w2" in report
        assert "HOLDOUT exclusions (PL-12)" in report
        assert results["guard"]["kept_disclosed_listed"] == ["w2"]
        assert "tools/holdout_guard.py" in results["guard"]["exclusion_guard"]

    def test_no_disclosure_line_without_holdout(self, tmp_path, monkeypatch,
                                                full_campaign):
        _install_guard(tmp_path, monkeypatch, exclusions=_EXCLUSIONS)
        runs_root, plan, tally, out = full_campaign
        results = rcf1.build_report(
            runs_root=runs_root,
            world_plan=plan,
            tally_path=tally,
            chunk_summary=None,
            out_dir=out / "final",
            cache_dir=out / "cache2",
            splits=("TRAIN", "CALIBRATION"),
            layout=Layout(),
        )
        report = (out / "final" / "REPORT.md").read_text(encoding="utf-8")
        assert "kept_disclosed" not in report
        assert results["guard"]["kept_disclosed_listed"] == []
        assert results["guard"]["exclusion_guard"] is None
