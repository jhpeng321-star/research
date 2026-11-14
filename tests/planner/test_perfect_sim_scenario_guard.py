"""PL-10c: the scenario-exactness refusal guard (PL-10 smoke diagnosis).

The PL-10 smoke's restored runs diverged because the SCENARIO is flow-based
(``<flow>`` + ``<vTypeDistribution>`` + vTypes without ``speedDev``): a
vehicle created by a flow AFTER ``loadState`` re-draws its type/speedFactor
from a different RNG stream position per branch. The owner decision: the
perfect simulator must REFUSE such scenarios -- raising, not marking
exact=False -- unless explicit option-A arrival draws are installed; the K=1
restored-RNG mode is NOT an exception.

Covers :func:`traffic_shepherd.planner.perfect_sim_state.check_scenario_exactness`
(plain and .gz route XML, .sumocfg indirection, every offending element
listed) and its ENFORCEMENT on the planner path (PerfectSimWorldModel
refuses before the snapshot and any BranchRunner.run).

Zero simulation: no SUMO/traci/libsumo import; the traci surface is fakes.
"""

import gzip
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from traffic_shepherd.planner import PlanCandidate
from traffic_shepherd.planner.perfect_sim import (
    BranchRecord,
    BranchRunner,
    PINNED_SUMO_VERSION,
    PerfectSimWorldModel,
    SimpleArrivalDraws,
    StepRecord,
    TraciStateSnapshotter,
    VehicleStep,
)
from traffic_shepherd.planner.perfect_sim_state import (
    ScenarioNotExactError,
    StateSnapshotError,
    check_scenario_exactness,
)
from tests.planner._helpers import RecordingProcessFactory

REPO_ROOT = Path(__file__).resolve().parents[2]
# the REAL flow-based chunk-1 TRAIN scenario of the PL-10 smoke (never
# HOLDOUT/TEST): <flow> x2, <vTypeDistribution>, nested vTypes w/o speedDev
REAL_FLOW_ROUTES = REPO_ROOT / "scenarios" / "idm_stage1b" / "idm_m2200_seed7001.rou.xml"

# PL-10c follow-up: the REAL campaign-format fixture (chunk-1 TRAIN root
# w8411m2700p02t240 base scenario, TRAIN per the PL-02a final roots.csv):
# element-form <route-files value=...> under a <configuration> root -- the
# shape that used to fail the guard open (vacuous pass).
CAMPAIGN_FIXTURE_DIR = (
    REPO_ROOT / "tests" / "planner" / "fixtures" / "campaign_scenario_w8411_train"
)
CAMPAIGN_FIXTURE_CFG = CAMPAIGN_FIXTURE_DIR / "scenario.sumocfg"
CAMPAIGN_FIXTURE_ROUTES = CAMPAIGN_FIXTURE_DIR / "requests.rou.xml"

T0 = 40.0
GOOD_STATE = (
    '<state><rngState seed="7"/><vehicle id="c0" pos="12.34567890123456789" speed="20.0"/></state>'
)

CAMPAIGN_ROUTES = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
  <vType id="hdv" carFollowModel="IDM" sigma="0" speedDev="0" speedFactor="1.0"/>
  <vType id="cav" carFollowModel="IDM" sigma="0" speedDev="0" speedFactor="1.0"/>
  <route id="mainline" edges="e0 e1"/>
  <vehicle id="v0" type="hdv" route="mainline" depart="0.00" departLane="free" departSpeed="max"/>
  <vehicle id="v1" type="hdv" route="mainline" depart="1.50" departLane="free" departSpeed="max"/>
  <vehicle id="v2" type="cav" route="mainline" depart="2.75" departLane="free" departSpeed="max"/>
</routes>
"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _routes_xml(*body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n'
        + "".join(line + "\n" for line in body)
        + "</routes>\n"
    )


# ------------------------------------------------- function-level checks
class TestCheckScenarioExactness:
    def test_real_flow_scenario_refused_naming_all_offenders(self):
        """NEGATIVE, real file: the smoke's scenario is refused with EVERY
        offending element named (flow + vTypeDistribution + speedDev)."""
        assert REAL_FLOW_ROUTES.is_file()
        with pytest.raises(ScenarioNotExactError) as excinfo:
            check_scenario_exactness(REAL_FLOW_ROUTES, explicit_arrivals_installed=False)
        message = str(excinfo.value)
        assert "mainline" in message and "ramp" in message  # both <flow>s
        assert "<flow" in message
        assert "<vTypeDistribution" in message and "hdv_mix" in message
        assert "speedDev" in message  # the nested vTypes lack it
        assert "option-A" in message  # the one exception is named

    def test_real_flow_scenario_exception_with_explicit_arrivals(self):
        """The option-A exception: explicit arrival draws installed -> the
        SAME dirty scenario passes (post-t0 departures are data, not draws)."""
        report = check_scenario_exactness(REAL_FLOW_ROUTES, explicit_arrivals_installed=True)
        assert report["exact"] is False
        assert report["explicit_arrivals_installed"] is True
        assert report["offenses"]  # still reported for the audit trail

    def test_campaign_format_passes(self, tmp_path):
        """POSITIVE: explicit <vehicle> list, fixed types, speedDev="0"."""
        path = _write(tmp_path / "campaign.rou.xml", CAMPAIGN_ROUTES)
        report = check_scenario_exactness(path, explicit_arrivals_installed=False)
        assert report["exact"] is True
        assert report["offenses"] == []
        assert report["route_files"] == [str(path)]

    @pytest.mark.parametrize(
        "body, needle",
        [
            # single-defect synthetic cases: each one alone is refused
            (
                '<vType id="car" speedDev="0"/><route id="r" edges="e"/>'
                '<flow id="f" type="car" route="r" begin="0" end="100" vehsPerHour="10"/>',
                "<flow",
            ),
            (
                '<vType id="car" speedDev="0"/>'
                '<vTypeDistribution id="mix"><vType id="a" speedDev="0"/>'
                '<vType id="b" speedDev="0"/></vTypeDistribution>',
                "<vTypeDistribution",
            ),
            ('<route id="r" edges="e"/><vType id="car"/>', "no speedDev"),
            ('<route id="r" edges="e"/><vType id="car" speedDev="0.1"/>', "speedDev"),
            (
                '<route id="r" edges="e"/><vType id="car" speedDev="0" speedFactor="norm(1,0.1)"/>',
                "distribution string",
            ),
            (
                '<vType id="car" speedDev="0"/><route id="r" edges="e"/>'
                '<vehicle id="v0" type="car" route="r" depart="0" '
                'speedFactor="normc(1.0,0.1,0,2)"/>',
                "distribution string",
            ),
        ],
    )
    def test_single_defect_refused(self, tmp_path, body, needle):
        path = _write(tmp_path / "one_defect.rou.xml", _routes_xml(body))
        with pytest.raises(ScenarioNotExactError) as excinfo:
            check_scenario_exactness(path, explicit_arrivals_installed=False)
        assert "refusing a scenario" in str(excinfo.value)
        assert needle in str(excinfo.value)

    def test_sumocfg_indirection_and_gzip(self, tmp_path):
        """A .sumocfg's comma-separated route-files (relative to the cfg dir)
        are followed, and .gz route files are parsed."""
        clean = _write(tmp_path / "clean.rou.xml", _routes_xml('<vType id="car" speedDev="0"/>'))
        gz = tmp_path / "nested" / "dirty.rou.xml.gz"
        gz.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(gz, "wb") as stream:
            stream.write(
                _routes_xml(
                    '<vType id="car" speedDev="0"/><route id="r" edges="e"/>'
                    '<vehicle id="v9" type="car" route="r" depart="3" '
                    'speedFactor="norm(1,0.2)"/>'
                ).encode("utf-8")
            )
        cfg = _write(
            tmp_path / "scenario.sumocfg",
            '<configuration>\n  <input net-file="net.net.xml" '
            'route-files="clean.rou.xml, nested/dirty.rou.xml.gz"/>\n'
            "</configuration>\n",
        )
        with pytest.raises(ScenarioNotExactError) as excinfo:
            check_scenario_exactness(cfg, explicit_arrivals_installed=False)
        message = str(excinfo.value)
        assert "dirty.rou.xml.gz" in message  # the .gz WAS parsed
        assert "norm(1,0.2)" in message
        # relative names resolve against the cfg directory, in order
        report = check_scenario_exactness(cfg, explicit_arrivals_installed=True)
        assert report["route_files"] == [str(clean), str(gz)]

    def test_missing_route_file_refused(self, tmp_path):
        with pytest.raises(ScenarioNotExactError, match="missing"):
            check_scenario_exactness(tmp_path / "nope.rou.xml", explicit_arrivals_installed=True)

    def test_error_family_is_state_snapshot_error(self):
        assert issubclass(ScenarioNotExactError, StateSnapshotError)


# --------------------------------- real campaign fixture + fail-closed cfg
@pytest.mark.skipif(
    not CAMPAIGN_FIXTURE_CFG.is_file(),
    reason="campaign fixture (tests/planner/fixtures/campaign_scenario_w8411_train) not present",
)
class TestRealCampaignFixtureAndFailClosedCfg:
    def test_real_campaign_sumocfg_resolves_to_requests_rou_exact(self):
        """(a) the REAL campaign .sumocfg (element-form <route-files
        value=...> under a <configuration> root -- previously a VACUOUS
        pass) resolves to exactly the one requests.rou.xml and is exact."""
        report = check_scenario_exactness(CAMPAIGN_FIXTURE_CFG, explicit_arrivals_installed=False)
        assert report["exact"] is True
        assert report["offenses"] == []
        assert report["route_files"] == [str(CAMPAIGN_FIXTURE_ROUTES)]

    def test_real_campaign_requests_rou_exact_with_type_speeddev_audit(self):
        """(b) the REAL campaign requests.rou.xml is exact, and EVERY
        type= referenced by a <vehicle> resolves to a <vType> in the file
        whose speedDev is exactly "0"."""
        report = check_scenario_exactness(
            CAMPAIGN_FIXTURE_ROUTES, explicit_arrivals_installed=False
        )
        assert report["exact"] is True
        assert report["offenses"] == []
        root = ET.parse(str(CAMPAIGN_FIXTURE_ROUTES)).getroot()
        vtypes = {v.get("id"): v for v in root.iter("vType")}
        vehicles = list(root.iter("vehicle"))
        assert len(vehicles) == 1413  # the explicit campaign vehicle list
        for vehicle in vehicles:
            vtype = vtypes.get(vehicle.get("type"))
            assert vtype is not None, f"unresolved type= on {vehicle.get('id')!r}"
            assert vtype.get("speedDev") == "0", f"speedDev != 0 on {vehicle.get('id')!r}"

    @pytest.mark.parametrize("root_tag", ["configuration", "sumoConfiguration"])
    def test_element_form_cfg_with_flow_route_refused(self, tmp_path, root_tag):
        """(c) element-form <route-files value=...> under ANY root element
        is honoured: a flow file behind such a cfg is REFUSED (the old
        attribute-only lookup passed it vacuously)."""
        _write(tmp_path / "flow.rou.xml", _routes_xml('<flow id="f" begin="0" end="9"/>'))
        cfg = _write(
            tmp_path / "scenario.sumocfg",
            f'<{root_tag}>\n  <input>\n    <route-files value="flow.rou.xml"/>\n'
            f"  </input>\n</{root_tag}>\n",
        )
        with pytest.raises(ScenarioNotExactError, match="flow"):
            check_scenario_exactness(cfg, explicit_arrivals_installed=False)

    def test_cfg_without_route_files_refused(self, tmp_path):
        """(d) a cfg whose <input> carries no route-files resolves to ZERO
        route files: refused, and the explicit-arrivals exception does NOT
        bypass a structural failure."""
        cfg = _write(
            tmp_path / "empty.sumocfg",
            '<configuration>\n  <input>\n    <net-file value="net.net.xml"/>\n'
            "  </input>\n</configuration>\n",
        )
        with pytest.raises(ScenarioNotExactError, match="vacuously"):
            check_scenario_exactness(cfg, explicit_arrivals_installed=False)
        with pytest.raises(ScenarioNotExactError, match="vacuously"):
            check_scenario_exactness(cfg, explicit_arrivals_installed=True)

    def test_cfg_with_only_non_route_additional_files_refused(self, tmp_path):
        """additional-files entries count ONLY when they carry route
        content: an edges-only additional file still resolves to ZERO route
        files -> refused."""
        _write(
            tmp_path / "edges.add.xml",
            '<edges>\n  <edge id="e0" from="a" to="b" numLanes="1"/>\n</edges>\n',
        )
        cfg = _write(
            tmp_path / "scenario.sumocfg",
            '<configuration>\n  <input>\n    <additional-files value="edges.add.xml"/>\n'
            "  </input>\n</configuration>\n",
        )
        with pytest.raises(ScenarioNotExactError, match="vacuously"):
            check_scenario_exactness(cfg, explicit_arrivals_installed=False)

    def test_route_carrying_additional_files_is_scanned(self, tmp_path):
        """...and an additional-files entry that DOES carry routes/vTypes/
        flows is parsed the same way: a flow inside it is refused."""
        _write(tmp_path / "extra.rou.xml", _routes_xml('<flow id="extra" begin="0" end="9"/>'))
        cfg = _write(
            tmp_path / "scenario.sumocfg",
            '<configuration>\n  <input>\n    <additional-files value="extra.rou.xml"/>\n'
            "  </input>\n</configuration>\n",
        )
        with pytest.raises(ScenarioNotExactError, match="extra"):
            check_scenario_exactness(cfg, explicit_arrivals_installed=False)

    def test_cfg_pointing_to_missing_file_refused(self, tmp_path):
        """(e) a cfg route-files entry that does not exist is refused, even
        with the explicit-arrivals exception requested."""
        cfg = _write(
            tmp_path / "scenario.sumocfg",
            '<sumoConfiguration>\n  <input>\n    <route-files value="ghost.rou.xml"/>\n'
            "  </input>\n</sumoConfiguration>\n",
        )
        with pytest.raises(ScenarioNotExactError, match="missing"):
            check_scenario_exactness(cfg, explicit_arrivals_installed=False)
        with pytest.raises(ScenarioNotExactError, match="missing"):
            check_scenario_exactness(cfg, explicit_arrivals_installed=True)

    def test_cfg_pointing_to_unparseable_file_refused(self, tmp_path):
        """A route file that is not valid XML is refused, never skipped."""
        _write(tmp_path / "broken.rou.xml", '<routes><flow id="f">\n  not xml')
        cfg = _write(
            tmp_path / "scenario.sumocfg",
            '<configuration>\n  <input>\n    <route-files value="broken.rou.xml"/>\n'
            "  </input>\n</configuration>\n",
        )
        with pytest.raises(ScenarioNotExactError, match="unparseable"):
            check_scenario_exactness(cfg, explicit_arrivals_installed=False)


# --------------------------------------------------- model-level refusal
class FakeSnapshotter:
    """Skeleton-seam fake: records saves, restores nothing."""

    def __init__(self):
        self.saves = []
        self.restores = []

    def save(self, *, t0_s, controller_state, kept_wall_ids):
        self.saves.append(float(t0_s))
        return {
            "saved_state_dir": "fake/saved_state",
            "saved_state_id": "f00d",
            "manifest": {"saved_state_id": "f00d"},
            "t0_s": float(t0_s),
        }

    def restore(self, manifest, *, plan, draw_id):
        self.restores.append(draw_id)


class FakeRunner(BranchRunner):
    """BranchRunner fake: every run() call is recorded (guard tests assert
    the record stays EMPTY on a refusal). PL-10d: carries the recording
    process factory the world model requires before any branch."""

    def __init__(self):
        super().__init__(process_factory=RecordingProcessFactory())
        self.calls = []

    def run(self, plan, **kw):
        self.calls.append(kw["draw_id"])
        steps = tuple(
            StepRecord(
                time_s=T0 + i,
                dt_s=1.0,
                vehicles={
                    "c0": VehicleStep(s_m=20.0 * i, lane_id=0.0, speed_mps=20.0, accel_mps2=0.0)
                },
            )
            for i in range(int(kw["horizon_s"]) + 1)
        )
        return BranchRecord(
            plan_label="plan",
            draw_id=kw["draw_id"],
            t0_s=T0,
            horizon_s=float(kw["horizon_s"]),
            dt_s=1.0,
            free_flow_s={"c0": 30.0},
            steps=steps,
            tracked=("c0",),
        )


def explicit_sample():
    return SimpleNamespace(
        hybrid=None,
        arrivals={
            "sample_id": "k0",
            "departures": ((T0 + 5.0, "main", 20.0, False, "hdv"),),
        },
        meta={"sample_id": "k0", "time_s": T0, "merge_observed": True, "coverage": 1.0},
    )


def restored_rng_sample():
    # the kit ArrivalDraw shape: K=1 restored-RNG mode, nothing installed
    return SimpleNamespace(
        hybrid=None,
        arrivals={"demand_model_id": "restored_rng", "seed": 7001},
        meta={"sample_id": "k0", "time_s": T0, "merge_observed": True, "coverage": 1.0},
    )


def make_model(scenario):
    snapshotter = FakeSnapshotter()
    runner = FakeRunner()
    model = PerfectSimWorldModel(
        snapshotter,
        SimpleArrivalDraws(),
        runner,
        part_a_horizon_s=5.0,
        scenario_route_files=scenario,
    )
    return model, snapshotter, runner


class TestModelEnforcement:
    def test_restored_rng_refused_before_any_branch_or_snapshot(self, tmp_path):
        """(6) the refusal happens before ANY BranchRunner.run call -- and
        before the t0 snapshot; the fake runner records nothing."""
        dirty = _write(tmp_path / "flow.rou.xml", _routes_xml('<flow id="f" begin="0" end="9"/>'))
        model, snapshotter, runner = make_model(dirty)
        with pytest.raises(ScenarioNotExactError, match="flow"):
            model.predict_many(
                [restored_rng_sample()],
                [PlanCandidate(per_cav={"c0": "damping"})],
                None,
                horizon_s=10.0,
            )
        assert runner.calls == []
        assert snapshotter.saves == []  # refused BEFORE the snapshot too

    def test_explicit_arrivals_exception_passes_restored_rng_still_refused(self):
        """(5) with the REAL dirty scenario declared: the option-A exception
        lets an explicit-draw prediction through, while the restored-RNG mode
        on the SAME model is still refused (it is NOT an exception)."""
        model, snapshotter, runner = make_model(REAL_FLOW_ROUTES)
        (pred,) = model.predict_many(
            [explicit_sample()], [PlanCandidate(per_cav={"c0": "damping"})], None, horizon_s=10.0
        )
        assert len(runner.calls) == 2  # native reference + plan branched
        assert snapshotter.saves == [T0]
        runner.calls.clear()
        with pytest.raises(ScenarioNotExactError, match="mainline"):
            model.predict_many(
                [restored_rng_sample()],
                [PlanCandidate(per_cav={"c0": "damping"})],
                None,
                horizon_s=10.0,
            )
        assert runner.calls == []  # nothing branched on the refused call

    def test_clean_scenario_allows_restored_rng(self, tmp_path):
        """Campaign format + restored-RNG: bit-exact by construction (the
        WM-41 pilot), so the guard passes it."""
        clean = _write(tmp_path / "campaign.rou.xml", CAMPAIGN_ROUTES)
        model, _snapshotter, runner = make_model(clean)
        model.predict_many(
            [restored_rng_sample()],
            [PlanCandidate(per_cav={"c0": "damping"})],
            None,
            horizon_s=10.0,
        )
        assert len(runner.calls) == 2  # native + plan

    def test_no_scenario_declared_restored_rng_refused(self):
        """No silent fallback: without a declared scenario, restored-RNG
        cannot prove post-t0 determinism and is refused outright."""
        model, snapshotter, runner = make_model(None)
        with pytest.raises(ScenarioNotExactError, match="scenario_route_files"):
            model.predict_many(
                [restored_rng_sample()],
                [PlanCandidate(per_cav={"c0": "damping"})],
                None,
                horizon_s=10.0,
            )
        assert runner.calls == []
        assert snapshotter.saves == []
        # ... while the explicit-draw path needs no scenario (option A)
        model.predict_many(
            [explicit_sample()], [PlanCandidate(per_cav={"c0": "damping"})], None, horizon_s=10.0
        )
        assert len(runner.calls) == 2

    def test_empty_scenario_list_rejected_at_construction(self):
        with pytest.raises(ValueError, match="scenario_route_files"):
            make_model([])

    def test_mixed_explicit_and_restored_rng_refused(self, tmp_path):
        """One restored-RNG sample among explicit ones still draws from the
        scenario RNG: the exception requires EVERY sample explicit."""
        dirty = _write(tmp_path / "flow.rou.xml", _routes_xml('<flow id="f" begin="0" end="9"/>'))
        model, _snapshotter, runner = make_model(dirty)
        with pytest.raises(ScenarioNotExactError):
            model.predict_many(
                [explicit_sample(), restored_rng_sample()],
                [PlanCandidate(per_cav={"c0": "damping"})],
                None,
                horizon_s=10.0,
            )
        assert runner.calls == []


# --------------------------------------------- snapshotter version query
class _VersionApi:
    """traci-shaped fake WITH getVersion (the live-module surface). The
    saveState surface sits under ``simulation`` exactly as the real
    traci module (and the real campaign_pilot_v1.saved_state) call it."""

    def __init__(self):
        self.version_calls = 0
        self.simulation = SimpleNamespace(
            saveState=lambda path: Path(path).write_text(GOOD_STATE, encoding="utf-8")
        )

    def getVersion(self):
        self.version_calls += 1
        return (22, "SUMO 1.19.0 (queried, not pinned)")


class _BareApi(_VersionApi):
    """A fake WITHOUT the live getVersion surface (kept at the pinned
    default -- tests/fakes own no traci version)."""

    def __init__(self):
        super().__init__()
        self.getVersion = None  # no live version surface on this fake


class TestSnapshotterRecordsQueriedVersion:
    # the vendored save path (a NON-EXISTENT temp root keeps the real
    # campaign_pilot_v1.saved_state OUT of sys.modules: importing it here
    # would leak into every later snapshotter constructed in this session)
    VENDORED = str(Path(tempfile.gettempdir()) / "no-such-wm34-root (vendored)")

    def test_queried_version_recorded_not_caller_constant(self, tmp_path):
        api = _VersionApi()
        snapshotter = TraciStateSnapshotter(api, tmp_path / "root", wm34_output_root=self.VENDORED)
        manifest = snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
        assert api.version_calls == 1
        assert manifest["manifest"]["sumo_version"] == [1, 19, 0]
        assert snapshotter.sumo_version == (1, 19, 0)

    def test_fake_without_get_version_keeps_pinned_default(self, tmp_path):
        api = _BareApi()
        snapshotter = TraciStateSnapshotter(api, tmp_path / "root", wm34_output_root=self.VENDORED)
        manifest = snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
        assert api.version_calls == 0  # no live surface: never queried
        assert manifest["manifest"]["sumo_version"] == list(PINNED_SUMO_VERSION)

    def test_explicit_caller_version_still_honoured(self, tmp_path):
        api = _VersionApi()
        snapshotter = TraciStateSnapshotter(
            api, tmp_path / "root", sumo_version="1.16.0", wm34_output_root=self.VENDORED
        )
        manifest = snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
        assert api.version_calls == 0  # caller pinned it: no query
        assert manifest["manifest"]["sumo_version"] == [1, 16, 0]
