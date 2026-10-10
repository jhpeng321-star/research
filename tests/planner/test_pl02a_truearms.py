"""PL-02a TRUE-ARMS report: arm-root override, true-gap TTC, labels, deltas.

Zero simulation; tmp_path fixtures only (no real run directory is read).
The fixture mimics the recorded campaign: an ORIGINAL root (base, recorded
switching native/pi, fixed1) and a TRUE-ARMS root (fleet-native / fleet-PI)
for one TRAIN world and one HOLDOUT world. Path spies prove that (a) no
HOLDOUT directory of either root is ever opened and (b) the original root's
switching native/pi are never read when the override is given.
"""

import builtins
import copy
import dataclasses
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip(
    "output.pl02a_offline_v1.report_chunk_final",
    reason="PL-02a FINAL report module (output/pl02a_offline_v1) lives in the planner worktree only",
)
from output.pl02a_offline_v1 import arm_roots as ar  # noqa: E402
from output.pl02a_offline_v1 import report_chunk_final as rcf  # noqa: E402
from output.pl02a_offline_v1 import run as runner  # noqa: E402
from traffic_shepherd.planner import PlannerConfig  # noqa: E402
from traffic_shepherd.planner.recorded import Layout  # noqa: E402

T0 = 60.0
TRAIN = "wTRAIN1"
HOLD = "wHOLDX9"
LEADER_POS, FOLLOWER_POS, LEADER_SPEED = 50.0, 30.0, 10.0


def _veh(pos, speed):
    return {
        "road_id": "e0",
        "lane_id": 0.0,
        "lane_position_m": pos,
        "speed_mps": speed,
        "acceleration_mps2": 0.0,
    }


def _rows(start, end, follower_speed):
    """Static closing pair: leader 'l' at 50 m / 10 m/s, follower 'f' at
    30 m: front-difference TTC = 20 / (fs - 10), true TTC = 15 / (fs - 10)."""
    rows, t = [], start
    while t < end - 1e-9:
        rows.append(
            {
                "time_s": t,
                "vehicles": {
                    "l": _veh(LEADER_POS, LEADER_SPEED),
                    "f": _veh(FOLLOWER_POS, follower_speed),
                },
                "colliding_ids": [],
                "cav_actions": {},
                "boundary_crossings": [],
            }
        )
        t += 0.5
    return rows


def _write_run(root, name, *, is_base, follower_speed, receipt_extra=None, rou=True):
    layout = Layout()
    run_dir = root / name
    physics_dir = run_dir / ("run" if is_base else ".")
    physics_dir.mkdir(parents=True, exist_ok=True)
    receipt = {"run": name, layout.receipt_status_key: layout.receipt_complete_value}
    receipt.update(receipt_extra or {})
    (run_dir / layout.receipt_name).write_text(json.dumps(receipt))
    end = T0 + (542.0 if is_base else 600.0)
    start = 0.0 if is_base else T0
    (physics_dir / layout.physics_name).write_text(
        "".join(json.dumps(r) + "\n" for r in _rows(start, end, follower_speed))
    )
    if rou:
        scenario = run_dir / "run" / "scenario"
        scenario.mkdir(parents=True, exist_ok=True)
        (scenario / "requests.rou.xml").write_text(
            "<routes>"
            '<vType id="normal_car"/>'
            '<vehicle id="l" type="normal_car" depart="0"/>'
            '<vehicle id="f" type="normal_car" depart="0"/>'
            "</routes>"
        )
    if not is_base:
        arm = name.rsplit("__", 1)[-1]
        (run_dir / layout.branch_record_name).write_text(json.dumps({"arm": arm}))
        if arm == "fixed1":  # a recorded constant plan (schedule + dispatcher)
            plan = {"c0": "damping", "c1": "pi-external-u"}
            (run_dir / layout.switch_schedule_name).write_text(
                json.dumps({"events": [{"t_s": T0, "per_cav": plan}]})
            )
            (run_dir / layout.dispatcher_rows_name).write_text(
                "".join(json.dumps({"time_s": T0 + t, "actions": plan}) + "\n"
                        for t in (0.0, 5.0, 10.0))
            )
    return run_dir


def _world(label, split):
    branches = []
    for arm in ("native", "pi", "fixed1"):
        branches.append(
            {
                "arm": arm,
                "run": f"{label}__{arm}",
                "duration_s": 600.0,
                "restores_state": {"from_run": f"{label}__base", "at_s": T0},
                "native_sibling_run": f"{label}__native",
                "branch_seed": 1,
            }
        )
    return {
        "world": label,
        "seed": 1,
        "loading_veh_h": 2600,
        "penetration": 0.02,
        "split": split,
        "t0_s": T0,
        "base_arm": {
            "arm": "base",
            "run": f"{label}__base",
            "duration_s": 602.0,
            "saves_state": {"at_s": T0},
        },
        "branches": branches,
    }


@pytest.fixture()
def world(tmp_path):
    """orig root: base/fixed1 + RECORDED native/pi (follower 20 m/s -> front
    TTC 2.0 s); true root: fleet native/pi (follower 15 m/s -> front TTC
    4.0 s, true TTC 3.0 s). Both worlds exist in both roots."""
    orig, true = tmp_path / "runs", tmp_path / "runs_truearms"
    for label in (TRAIN, HOLD):
        _write_run(orig, f"{label}__base", is_base=True, follower_speed=15.0)
        _write_run(orig, f"{label}__fixed1", is_base=False, follower_speed=15.0)
        for arm in ("native", "pi"):
            _write_run(orig, f"{label}__{arm}", is_base=False, follower_speed=20.0)
            _write_run(true, f"{label}__{arm}", is_base=False, follower_speed=15.0,
                       receipt_extra={"world": label, "arm": arm, "saved_state_id": "S1"})
        # sibling receipts carry the saved-state id the overrides must match
        rc = orig / f"{label}__fixed1" / "RUN_RECEIPT.json"
        data = json.loads(rc.read_text())
        data["saved_state_id"] = "S1"
        rc.write_text(json.dumps(data))
    plan = tmp_path / "WORLD_PLAN.json"
    plan.write_text(json.dumps({"worlds": [_world(TRAIN, "TRAIN"), _world(HOLD, "HOLDOUT")]}))
    return SimpleNamespace(orig=orig, true=true, plan=plan, tmp=tmp_path)


def _build(world, out="out", cache="cache", **kw):
    out_dir, cache_dir = world.tmp / out, world.tmp / cache
    kw.setdefault("arm_roots", {"native": world.true, "pi": world.true})
    kw.setdefault("ttc_gap", "front")
    return rcf.build_report(
        runs_root=world.orig,
        world_plan=world.plan,
        tally_path=None,
        chunk_summary=None,
        out_dir=out_dir,
        cache_dir=cache_dir,
        splits=("TRAIN",),
        layout=Layout(),
        spec=rcf.CHUNKS[2],
        **kw,
    )


def _root(results):
    (record,) = [r for r in results["per_root"] if r["root"] == f"{TRAIN}__base"]
    return record


# ----------------------------------------------------------- path spies
class _Spy:
    def __init__(self, monkeypatch):
        self.seen = []
        for name in ("iterdir", "is_file", "is_dir", "exists", "stat", "read_text",
                     "read_bytes", "open", "glob", "rglob"):
            self._wrap_path(monkeypatch, name)
        for mod, name in ((os, "scandir"), (os, "listdir"), (os, "stat"), (builtins, "open")):
            self._wrap_fn(monkeypatch, mod, name)

    def _wrap_path(self, monkeypatch, name):
        orig = getattr(Path, name)
        seen = self.seen

        def wrapper(path_self, *a, **k):
            seen.append(str(path_self).replace("\\", "/"))
            return orig(path_self, *a, **k)

        monkeypatch.setattr(Path, name, wrapper)

    def _wrap_fn(self, monkeypatch, mod, name):
        orig = getattr(mod, name)
        seen = self.seen

        def wrapper(*a, **k):
            if a and isinstance(a[0], (str, bytes, os.PathLike)):
                seen.append(os.fsdecode(a[0]).replace("\\", "/"))
            return orig(*a, **k)

        monkeypatch.setattr(mod, name, wrapper)

    def touched(self, needle):
        return sorted({p for p in self.seen if needle in p})


def test_holdout_dirs_never_opened_in_either_root(world, monkeypatch):
    spy = _Spy(monkeypatch)
    results = _build(world)
    assert spy.touched(TRAIN), "sanity: the TRAIN runs were read"
    assert spy.touched(HOLD) == []  # no HOLDOUT dir of orig OR true root
    assert results["collisions"]["holdout_collision_pairs_count_only"] is None
    assert [r["root"] for r in results["per_root"]] == [f"{TRAIN}__base"]
    # neither root was ever listed
    assert str(world.orig).replace("\\", "/") not in [p for p in spy.seen if p.endswith("runs")]
    assert str(world.true).replace("\\", "/") not in spy.seen


def test_recorded_switching_native_pi_not_read_with_override(world, monkeypatch):
    spy = _Spy(monkeypatch)
    results = _build(world)
    orig = str(world.orig).replace("\\", "/")
    for arm in ("native", "pi"):
        assert spy.touched(f"{orig}/{TRAIN}__{arm}") == []
        assert spy.touched(f"{str(world.true).replace(chr(92), '/')}/{TRAIN}__{arm}")
    # the analysed native is the TRUE one: front TTC 4.0 s (recorded: 2.0 s)
    assert _root(results)["arms"]["native"]["min_ttc_s"] == pytest.approx(4.0)


def test_without_override_recorded_arms_are_used(world):
    results = _build(world, arm_roots={}, ttc_gap="front")
    assert _root(results)["arms"]["native"]["min_ttc_s"] == pytest.approx(2.0)
    # legacy pipeline: no new keys, plain cache file names
    for key in ("labels", "arm_roots", "sign_convention", "ttc_gap"):
        assert key not in results
    assert (world.tmp / "cache" / f"{TRAIN}__native.json").is_file()
    header = (world.tmp / "out" / "roots.csv").read_text().splitlines()[0]
    assert "delta" not in header


# ----------------------------------------------------------- true gap
def test_true_gap_vs_front_gap_ttc_through_runner(world):
    front = _build(world, out="o_front", cache="c_front", ttc_gap="front")
    true = _build(world, out="o_true", cache="c_true", ttc_gap="true")
    for arm in ("native", "pi"):
        assert _root(front)["arms"][arm]["min_ttc_s"] == pytest.approx(4.0)  # 20 / 5
        assert _root(true)["arms"][arm]["min_ttc_s"] == pytest.approx(3.0)  # (20 - 5) / 5
    assert _root(true)["arms"]["base"]["min_ttc_s"] == pytest.approx(3.0)
    assert _root(front)["arms"]["base"]["min_ttc_s"] == pytest.approx(4.0)


def test_true_gap_fails_closed_without_route_types(world):
    # the true native run's route file no longer declares the leader
    rou = world.true / f"{TRAIN}__native" / "run" / "scenario" / "requests.rou.xml"
    rou.write_text('<routes><vType id="normal_car"/>'
                   '<vehicle id="f" type="normal_car" depart="0"/></routes>')
    results = _build(world, ttc_gap="true")
    record = _root(results)
    assert record["usable"] is False
    assert "fail closed" in record["unusable_reason"]
    # the same data is fine under the (biased) front gap
    assert _build(world, out="o2", cache="c2", ttc_gap="front")["per_root"][0]["usable"] is True


def _entry(world, run_name, cache="cache"):
    (path,) = list((world.tmp / cache).glob(f"{run_name}.*.json"))
    return json.loads(path.read_text())


def test_branch_without_scenario_uses_base_lengths(world):
    """Restored branch arms have no scenario dir: the same root's base route
    file is used (source 'base', digest_unverified without a recorded
    digest); arms with their own scenario report source 'own'."""
    shutil.rmtree(world.true / f"{TRAIN}__native" / "run")
    shutil.rmtree(world.orig / f"{TRAIN}__fixed1" / "run", ignore_errors=True)
    # fixed1 keeps its physics at <run>/ (only the scenario dir is removed)
    results = _build(world, ttc_gap="true")
    record = _root(results)
    assert record["usable"] is True
    assert record["arms"]["native"]["min_ttc_s"] == pytest.approx(3.0)  # base lengths
    assert _entry(world, f"{TRAIN}__native")["true_gap_source"] == {
        "source": "base", "digest": "digest_unverified"}
    assert _entry(world, f"{TRAIN}__fixed1")["true_gap_source"]["source"] == "base"
    assert _entry(world, f"{TRAIN}__base")["true_gap_source"]["source"] == "own"
    assert _entry(world, f"{TRAIN}__pi")["true_gap_source"]["source"] == "own"


def _drop_scenario(world, arm_root, name):
    shutil.rmtree(arm_root / name / "run" / "scenario")


def test_base_fallback_digest_verified_and_mismatch_named_error(world):
    import hashlib

    from output.pl02a_offline_v1.run import TrueGapScenarioError, true_gap_source_for_run
    from traffic_shepherd.planner.recorded import CampaignIndex

    _drop_scenario(world, world.true, f"{TRAIN}__native")
    index = CampaignIndex(world.orig, world.plan, Layout())
    index.runs_root = ar.ScopedRunsRoot(
        {n: (world.true if n.endswith("__native") else world.orig) / n
         for n in (f"{TRAIN}__base", f"{TRAIN}__native", f"{TRAIN}__fixed1")}, world.orig)
    index.build()
    root = index.roots[f"{TRAIN}__base"]
    native = root.branches["native"]
    base_sha = hashlib.sha256(root.base.rou_paths[0].read_bytes()).hexdigest()
    # recorded digest equal -> verified
    native.branch_record = {"scenario_digests": {"scenario/requests.rou.xml": base_sha}}
    lengths, meta = true_gap_source_for_run(native, root.base)
    assert meta == {"source": "base", "digest": "verified"} and lengths["l"] == 5.0
    # recorded digest differs -> named error
    native.branch_record = {"scenario_digests": {"scenario/requests.rou.xml": "0" * 64}}
    with pytest.raises(TrueGapScenarioError, match="does not match"):
        true_gap_source_for_run(native, root.base)
    # base saved-state manifest digest is honoured too
    native.branch_record = None
    manifest = world.tmp / "STATE_MANIFEST.json"
    manifest.write_text(json.dumps({"scenario_digests": {"scenario/requests.rou.xml": "1" * 64}}))
    root.base.manifest_path = manifest
    with pytest.raises(TrueGapScenarioError, match="does not match"):
        true_gap_source_for_run(native, root.base)
    manifest.write_text(json.dumps({"scenario_digests": {"scenario/requests.rou.xml": base_sha}}))
    assert true_gap_source_for_run(native, root.base)[1]["digest"] == "verified"


def test_no_own_and_no_base_scenario_is_a_named_error(world):
    from output.pl02a_offline_v1.run import TrueGapScenarioError, true_gap_source_for_run

    run = SimpleNamespace(directory=Path("wX__fixed1"), rou_paths=(), branch_record=None,
                          receipt={})
    with pytest.raises(TrueGapScenarioError, match="fails closed"):
        true_gap_source_for_run(run, None)
    base = SimpleNamespace(directory=Path("wX__base"), rou_paths=(), manifest_path=None)
    with pytest.raises(TrueGapScenarioError):
        true_gap_source_for_run(run, base)
    # through the runner: base scenario gone -> the root fails closed
    shutil.rmtree(world.true / f"{TRAIN}__native" / "run")
    shutil.rmtree(world.orig / f"{TRAIN}__base" / "run" / "scenario")
    assert _root(_build(world, ttc_gap="true"))["usable"] is False


# ------------------------------------------------------------- cache key
def test_cache_key_includes_arm_roots_and_gap_mode(world):
    alt = world.tmp / "runs_truearms_copy"
    shutil.copytree(world.true, alt)
    _build(world, ttc_gap="front")
    _build(world, ttc_gap="true")
    _build(world, ttc_gap="true", arm_roots={"native": alt, "pi": alt})
    names = sorted(p.name for p in (world.tmp / "cache").glob(f"{TRAIN}__native.*.json"))
    assert len(names) == 3 and len(set(names)) == 3
    variants = [
        json.loads((world.tmp / "cache" / n).read_text())["key"]["variant"] for n in names
    ]
    assert {v["ttc_gap"] for v in variants} == {"front", "true"}
    assert len({json.dumps(v["arm_roots"], sort_keys=True) for v in variants}) == 2


def test_structural_key_legacy_unchanged_and_variant_changes_it(world):
    from traffic_shepherd.planner.recorded import CampaignIndex

    index = CampaignIndex(world.orig, world.plan, Layout()).build()
    run = index.roots[f"{TRAIN}__base"].base
    args = (run, Layout(), "sha", {"builtin": "b"}, "r", None, T0)
    legacy = runner.entry_structural_key(*args)
    assert "variant" not in legacy and set(legacy) == {
        "run_dir", "receipt_sha256", "code_sha256", "layout_fingerprint",
        "provider_versions", "t0_s", "aux_sha256"}
    assert runner.entry_structural_key(*args, None) == legacy
    other = runner.entry_structural_key(*args, {"ttc_gap": "true", "arm_roots": {}})
    assert other != legacy
    assert other != runner.entry_structural_key(*args, {"ttc_gap": "front", "arm_roots": {}})
    assert runner.cache_entry_path(Path("c"), "r", None).name == "r.json"


# ------------------------------------------------- identity / receipts
def test_override_receipt_identity_mismatch_removes_arm(world):
    rc = world.true / f"{TRAIN}__native" / "RUN_RECEIPT.json"
    data = json.loads(rc.read_text())
    data["saved_state_id"] = "OTHER"
    rc.write_text(json.dumps(data))
    results = _build(world)
    assert _root(results)["usable"] is False  # native removed -> fails closed
    failures = results["arm_root_identity"]["failures"]
    assert failures and "saved_state_id" in failures[0]


def test_incomplete_override_receipt_is_skipped(world):
    rc = world.true / f"{TRAIN}__native" / "RUN_RECEIPT.json"
    data = json.loads(rc.read_text())
    data["exit_status"] = "failed"
    rc.write_text(json.dumps(data))
    with pytest.raises(SystemExit):  # receipt/physics disagreement fails loudly
        _build(world)


# ----------------------------------------------------- labels and signs
def test_labels_rule_sign_and_delta_columns(world):
    results = _build(world, ttc_gap="true")
    labels = results["labels"]
    assert labels["native"] == "native = TRUE fleet-native (truearms)"
    assert labels["pi"] == "pi = TRUE fleet-PI (U=31.29)"
    assert labels["ttc"] == "TTC = true bumper gap"
    config = PlannerConfig()
    assert labels["safety_rule"]["fields"]["ttc_floor_s"] == config.ttc_floor_s
    assert str(config.ttc_floor_s) in labels["safety_rule"]["text"]
    report = (world.tmp / "out" / "REPORT.md").read_text(encoding="utf-8")
    for text in (labels["native"], labels["pi"], labels["ttc"], "OPPOSITE to AGENTS.md",
                 "saving = native - arm"):
        assert text in report
    assert "saving = native - arm" in results["sign_convention"]["analysis"]
    record = _root(results)
    for name, arm in record["arms"].items():
        assert arm["delta_vs_native_s"] == -arm["saving_vs_native_s"]
    header = (world.tmp / "out" / "roots.csv").read_text().splitlines()[0].split(",")
    assert header[-2:] == ["pi_delta_vs_native_s", "best_safe_delta_s"]
    assert results["arm_roots"] == {"native": str(world.true), "pi": str(world.true)}
    head = results["train_headline"]
    assert head["median_pi_delta_0filled_s_agents_convention"] == (
        None if head["pi_vs_native"]["median_pi_saving_0filled_s"] is None
        else -head["pi_vs_native"]["median_pi_saving_0filled_s"]
    )


def test_labels_for_recorded_arms_and_front_gap():
    labels = ar.build_labels({}, "front", PlannerConfig())
    assert "SWITCHING" in labels["native"] and "SWITCHING" in labels["pi"]
    assert "biased HIGH" in labels["ttc"]


def test_safety_rule_falls_back_when_absolute_ttc_rule_absent():
    plain = ar.safety_rule_description(PlannerConfig())
    if hasattr(PlannerConfig(), "absolute_ttc_rule"):
        assert "absolute_ttc_rule =" in plain["absolute_ttc_rule_note"]
    else:
        assert "not present" in plain["absolute_ttc_rule_note"]
        assert plain["absolute_ttc_rule"] is None
    fake = SimpleNamespace(ttc_floor_s=2.5, absolute_ttc_rule="rule-x", rel_safety_tol_s=1.0,
                           relative_ttc_threshold_s=1.5)
    described = ar.safety_rule_description(fake)
    assert described["absolute_ttc_rule"] == "rule-x"
    assert "2.5" in described["text"] and "rule-x" in described["absolute_ttc_rule_note"]


def test_add_delta_fields_sign():
    record = {"arms": {"pi": {"saving_vs_native_s": 12.0}, "x": {"saving_vs_native_s": None}},
              "best_safe_delay_s": 90.0, "native_delay_s": 100.0}
    ar.add_delta_fields([record])
    assert record["arms"]["pi"]["delta_vs_native_s"] == -12.0
    assert record["arms"]["x"]["delta_vs_native_s"] is None
    assert record["best_safe_delta_s"] == -10.0  # arm - native, negative = better


# ------------------------------------------------------------------ CLI
def test_parse_arm_roots():
    parsed = ar.parse_arm_roots(["native=/a/b", "pi=/c"])
    assert parsed == {"native": Path("/a/b"), "pi": Path("/c")}
    assert ar.parse_arm_roots(None) == {}
    for bad in (["native"], ["base=/x"], ["native=/a", "native=/b"], ["=/a"], ["pi="]):
        with pytest.raises(SystemExit):
            ar.parse_arm_roots(bad)


def test_documented_commands_parse_with_the_real_cli():
    parser = rcf.build_parser()
    lines = ar.render_commands().splitlines()
    assert len(lines) == 2
    for chunk, line in zip((2, 3), lines):
        args = parser.parse_args(line.split()[3:])
        assert args.chunk == chunk and args.ttc_gap == "true"
        roots = ar.parse_arm_roots(args.arm_root)
        assert set(roots) == {"native", "pi"}
        assert roots["native"].name == f"campaign_stage1_chunk{chunk}_truearms"
        assert args.out.as_posix().endswith(f"results_truearms_v1/chunk{chunk}")
        assert args.cache.name == "cache_truearms_v1"
    assert rcf.build_parser().parse_args(["--chunk", "2"]).ttc_gap == "true"
    assert 3 in rcf.CHUNKS


def test_scoped_run_dirs_come_from_plan_split_metadata_only(world):
    worlds = rcf.plan_worlds(world.plan)
    dirs = ar.scoped_run_dirs(worlds, ("TRAIN",), world.orig, {"native": world.true})
    assert set(dirs) == {f"{TRAIN}__{a}" for a in ("base", "native", "pi", "fixed1")}
    assert dirs[f"{TRAIN}__native"] == world.true / f"{TRAIN}__native"
    assert dirs[f"{TRAIN}__pi"] == world.orig / f"{TRAIN}__pi"
    scoped = ar.ScopedRunsRoot(dirs, world.orig)
    assert scoped.is_dir() and len(list(scoped.iterdir())) == 4
    # the original PlannerConfig / layout are untouched by the helpers
    assert dataclasses.is_dataclass(PlannerConfig()) and copy.copy(Layout()) == Layout()


# ------------------------------------------------ rule C (outcome level)
def _outcome(ttc, **kw):
    from traffic_shepherd.planner.recorded import BranchOutcome

    fields = {f.name for f in dataclasses.fields(BranchOutcome)}
    base = dict(cell_valid=True, collision=False, min_ttc_s=ttc)
    base.update(kw)
    return SimpleNamespace(**{k: v for k, v in base.items() if k in fields or True})


def test_outcome_safe_rule_c_vs_absolute():
    rule_c = PlannerConfig()
    rule_a = dataclasses.replace(rule_c, absolute_ttc_rule="absolute")
    assert rule_c.absolute_ttc_rule == "relative_to_native"
    native = _outcome(1.0)
    # below the 1.5 floor but >= native's: SAFE under rule C, UNSAFE under A
    assert runner.outcome_safe(_outcome(1.2), rule_c, native) is True
    assert runner.outcome_safe(_outcome(1.2), rule_a, native) is False
    # below native's (native < 1.5): UNSAFE under both
    assert runner.outcome_safe(_outcome(0.8), rule_c, native) is False
    # native without a closing pair -> floor 1.5
    assert runner.outcome_safe(_outcome(1.2), rule_c, _outcome(None)) is False
    assert runner.outcome_safe(_outcome(1.6), rule_c, _outcome(None)) is True
    assert runner.outcome_safe(_outcome(1.2), rule_c, None) is False
    # native above the floor: the floor stays 1.5
    assert runner.outcome_safe(_outcome(1.4), rule_c, _outcome(3.0)) is False
    # absolute ignores the native entirely
    assert runner.outcome_safe(_outcome(1.6), rule_a, native) is True
    # unchanged handling
    assert runner.outcome_safe(_outcome(None), rule_c, native) is True
    assert runner.outcome_safe(_outcome(float("nan")), rule_c, native) is False
    assert runner.outcome_safe(_outcome(5.0, collision=True), rule_c, native) is False
    assert runner.outcome_safe(_outcome(5.0, cell_valid=False), rule_c, native) is False


def test_rule_reported_in_labels_and_results(world):
    results = _build(world, ttc_gap="true")
    assert results["safety_rule"]["absolute_ttc_rule"] == "relative_to_native"
    assert "rule C" in results["safety_rule"]["text"]
    report = (world.tmp / "out" / "REPORT.md").read_text(encoding="utf-8")
    assert "min(1.5 s, native min TTC" in report
    rule_a = dataclasses.replace(PlannerConfig(), absolute_ttc_rule="absolute")
    text = ar.safety_rule_description(rule_a)["text"]
    assert "rule A" in text


def test_pipeline_safe_outcome_follows_rule(world):
    # native TTC (front) 4.0 > floor -> floor stays 1.5; pin the arms' rule
    for rule in ("relative_to_native", "absolute"):
        cfg = dataclasses.replace(PlannerConfig(), absolute_ttc_rule=rule)
        results = _build(world, out=f"o_{rule}", cache=f"c_{rule}", config=cfg)
        assert results["safety_rule"]["absolute_ttc_rule"] == rule
        assert _root(results)["arms"]["pi"]["safe_outcome"] is True
