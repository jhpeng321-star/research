"""MIG-05 evidence-regeneration harness tests (mocked; no SUMO is launched).

plan   is checked against the real manifest/runner/registration sources when the
       full local evidence tree is present (skipped on published clones).
run    exercises traffic_shepherd.workflow.evaluate only through mocks and a
       fake traci module; nothing touches runs/active or real run directories.
dataset checks the sha256 comparison record logic on tiny synthetic fixtures.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "output" / "evidence_regeneration_v1"
MANIFEST = ROOT / "output" / "worldmodel_dataset_v2" / "MANIFEST.json"
LOCAL_TREE = (ROOT / "archive").is_dir() and MANIFEST.is_file()

requires_local_tree = pytest.mark.skipif(
    not LOCAL_TREE, reason="needs the full local evidence tree (manifest + campaign sources)")


def _load_regenerate():
    spec = importlib.util.spec_from_file_location(
        "evidence_regeneration_v1_regenerate", HERE / "regenerate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reg():
    return _load_regenerate()


@pytest.fixture(scope="module")
def real_plan(reg, tmp_path_factory):
    plan_path = tmp_path_factory.mktemp("plan") / "PLAN.json"
    reg.write_json(plan_path, reg.build_plan())
    return json.loads(plan_path.read_text(encoding="utf-8")), plan_path


def _entry(plan, run_tag):
    return next(e for e in plan["runs"] if e["run_tag"] == run_tag)


def _evaluate_call(runner_path):
    """Independently read the workflow.evaluate keywords from a campaign runner."""
    tree = ast.parse(Path(runner_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "evaluate":
            return {kw.arg: kw.value for kw in node.keywords}, tree
    raise AssertionError(f"no evaluate call found in {runner_path}")


def _const(node):
    assert isinstance(node, ast.Constant), f"expected constant, got {ast.dump(node)}"
    return node.value


def _campaign_dir(campaign):
    name = "controller_verification_v26_wall" if campaign == "cv26" \
        else f"controller_verification_v{campaign[2:]}"
    return ROOT / "output" / name


def _registration(campaign):
    return json.loads((_campaign_dir(campaign) / "REGISTRATION.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

@requires_local_tree
def test_plan_covers_all_88_manifest_runs(reg, real_plan):
    plan, _ = real_plan
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_tags = [run["run_tag"] for run in manifest["runs"]]
    assert plan["totals"]["runs"] == 88
    assert sorted(e["run_tag"] for e in plan["runs"]) == sorted(manifest_tags)
    assert len({e["run_tag"] for e in plan["runs"]}) == 88
    for run in manifest["runs"]:
        entry = _entry(plan, run["run_tag"])
        assert entry["campaign"] == run["campaign"] and entry["job"] == run["job"]
        assert entry["split"] == run["split"]
        assert entry["expected_output_sha256"] == run["output_sha256"]
    counts = {}
    for entry in plan["runs"]:
        counts[entry["campaign"]] = counts.get(entry["campaign"], 0) + 1
    assert counts == {"cv17": 3, "cv19": 6, "cv20": 21, "cv21": 3, "cv23": 4, "cv24": 1,
                      "cv25": 4, "cv26": 1, "cv27": 3, "cv28": 3, "cv29": 1, "cv30": 18,
                      "cv31": 15, "cv32": 5}
    assert plan["totals"]["reproducible"] == 88
    assert all(e["reproducible"] for e in plan["runs"])
    divergent = {e["campaign"]: e["source_code_state"] for e in plan["runs"]}
    assert divergent["cv32"] == "divergent" and divergent["cv20"] == "divergent"
    cv32_entry = _entry(plan, "cv32_7531m2700-wall")
    assert {row["module"] for row in cv32_entry["source_hash_differences"]
            if row["status"] != "match"} == {"__init__.py", "verification.py"}


@requires_local_tree
def test_plan_refuses_to_overwrite_existing_plan(reg, tmp_path):
    out_path = tmp_path / "PLAN.json"
    assert reg.main(["plan", "--out", str(out_path)]) == 0
    with pytest.raises(reg.RegenerationError, match="refusing to overwrite"):
        reg.main(["plan", "--out", str(out_path)])


@requires_local_tree
def test_plan_marks_missing_job_unreproducible(reg, tmp_path):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    victim = dict(manifest["runs"][0])
    victim["job"] = "not-a-registered-job"
    victim["run_tag"] = "cvXX_not-a-registered-job"
    manifest["runs"] = [victim]
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out_path = tmp_path / "PLAN.json"
    assert reg.main(["plan", "--manifest", str(manifest_path),
                     "--regen-root", str(tmp_path / "regenerated"),
                     "--out", str(out_path)]) == 0
    plan = json.loads(out_path.read_text(encoding="utf-8"))
    entry = plan["runs"][0]
    assert entry["reproducible"] is False
    assert any("absent" in reason for reason in entry["unreproducible_reasons"])
    assert entry["evaluate_kwargs"] is None


# ---------------------------------------------------------------------------
# planned kwargs vs the runner sources (independent AST reading)
# ---------------------------------------------------------------------------

@requires_local_tree
def test_planned_kwargs_match_runner_source(reg, real_plan):
    plan, _ = real_plan

    # --- cv17: literal seed, suffix-conditional desired speed ----------------
    runner = _campaign_dir("cv17") / "runner.py"
    kwargs_expr, cv17_tree = _evaluate_call(runner)
    planned = _entry(plan, "cv17_native")["evaluate_kwargs"]
    seed_literal = _const(kwargs_expr["seed"])
    registration = _registration("cv17")
    assert seed_literal == planned["seed"] == registration["jobs"]["native"]["expected"]["seed"]
    assert planned["loading"] == registration["jobs"]["native"]["loading"]
    assert planned["penetration"] == _const(kwargs_expr["penetration"]) == 0.2
    assert planned["duration"] == _const(kwargs_expr["duration"]) == 2400
    assert planned["role"] == _const(kwargs_expr["role"]) == "exposed"
    assert planned["controller_name"] == _const(kwargs_expr["controller_name"]) == "damping"
    desired = kwargs_expr["desired_speed_mps"]
    assert isinstance(desired, ast.IfExp) and _const(desired.orelse) is None
    pi_token = _const(desired.test.comparators[0])
    pi_value = _const(desired.body)
    assert planned["desired_speed_mps"] is None  # native arm keeps the damping prefix
    pi_entry = _entry(plan, "cv17_pi-external-u")["evaluate_kwargs"]
    assert pi_entry["desired_speed_mps"] == pi_value == 31.29
    assert pi_token == "pi-external-u"
    fleet_args = kwargs_expr["fleet_switch"].args
    assert _const(fleet_args[0]) == planned["fleet_switch"]["prefix"]
    assert _const(fleet_args[2]) == planned["fleet_switch"]["switch_time_s"] == 600.0
    suffix_arg = fleet_args[1]
    assert isinstance(suffix_arg, ast.Name)
    suffix_assign = next(node for node in ast.walk(cv17_tree) if isinstance(node, ast.Assign)
                         and any(getattr(t, "id", None) == suffix_arg.id for t in node.targets))
    assert isinstance(suffix_assign.value, ast.IfExp)
    assert _const(suffix_assign.value.test.comparators[0]) == "retain"
    assert _const(suffix_assign.value.body) == "damping"
    retain_entry = _entry(plan, "cv17_retain")["evaluate_kwargs"]
    assert retain_entry["fleet_switch"] == \
        registration["jobs"]["retain"]["expected"]["fleet_switch"]

    # --- cv20/cv31: seed read from registration expected --------------------
    for campaign, native_tag, pi_tag in (("cv20", "cv20_7001m2000-native",
                                          "cv20_7001m2000-pi-external-u"),):
        runner = _campaign_dir(campaign) / "runner.py"
        kwargs_expr, _tree = _evaluate_call(runner)
        chain = kwargs_expr["seed"]
        assert isinstance(chain, ast.Subscript) and _const(chain.slice) == "seed"
        assert isinstance(chain.value, ast.Subscript) and _const(chain.value.slice) == "expected"
        registration = _registration(campaign)
        job = native_tag.split("_", 1)[1]
        planned = _entry(plan, native_tag)["evaluate_kwargs"]
        assert planned["seed"] == registration["jobs"][job]["expected"]["seed"]
        assert planned["loading"] == registration["jobs"][job]["loading"]
        assert planned["desired_speed_mps"] is None
        pi_planned = _entry(plan, pi_tag)["evaluate_kwargs"]
        assert pi_planned["desired_speed_mps"] == _const(
            kwargs_expr["desired_speed_mps"].body) == 31.29
        assert pi_planned["fleet_switch"] == \
            registration["jobs"][pi_tag.split("_", 1)[1]]["expected"]["fleet_switch"]

    # --- cv31 cell check added explicitly (second registration) -------------
    registration31 = _registration("cv31")
    job31 = "7531m2700-native"
    planned31 = _entry(plan, "cv31_7531m2700-native")["evaluate_kwargs"]
    assert planned31["seed"] == registration31["jobs"][job31]["expected"]["seed"]
    assert planned31["loading"] == registration31["jobs"][job31]["loading"]

    # --- cv26: wall-v2 suffix mode, desired None, no wall_suffix_path -------
    runner26 = _campaign_dir("cv26") / "runner.py"
    kwargs_expr, tree26 = _evaluate_call(runner26)
    suffix_mode_value = None
    for node in tree26.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "SUFFIX_MODE" for t in node.targets):
            suffix_mode_value = _const(node.value)
    planned26 = _entry(plan, "cv26_7012m2700-wall")["evaluate_kwargs"]
    assert suffix_mode_value == planned26["fleet_switch"]["suffix"] == "wall-v2"
    assert _const(kwargs_expr["desired_speed_mps"]) is None
    assert planned26["desired_speed_mps"] is None
    assert "wall_suffix_path" not in planned26
    assert planned26["fleet_switch"]["switch_time_s"] == _const(
        kwargs_expr["fleet_switch"].args[2]) == 600.0

    # --- cv32: wall-v2 + registered wall_suffix_path ------------------------
    runner32 = _campaign_dir("cv32") / "runner.py"
    kwargs_expr, _tree = _evaluate_call(runner32)
    planned32 = _entry(plan, "cv32_7531m2700-wall")["evaluate_kwargs"]
    literals = [node.value for node in sorted(
        (n for n in ast.walk(kwargs_expr["wall_suffix_path"])
         if isinstance(n, ast.Constant) and isinstance(n.value, str)),
        key=lambda n: (n.lineno, n.col_offset))]
    expected_path = str(ROOT.joinpath(*literals))
    assert planned32["wall_suffix_path"] == expected_path
    assert Path(planned32["wall_suffix_path"]).is_file()
    assert planned32["fleet_switch"]["suffix"] == "wall-v2"


# ---------------------------------------------------------------------------
# run (mocked evaluate / traci)
# ---------------------------------------------------------------------------

def _tmp_plan_entry(reg, tmp_path, source_entry):
    entry = json.loads(json.dumps(source_entry))
    regen = tmp_path / "regenerated_v1"
    target = regen / entry["campaign"] / entry["job"]
    entry["regenerated_run_dir"] = str(target)
    entry["evaluate_kwargs"]["output"] = str(target / "run")
    plan = {"schema": "evidence_regeneration_plan/1", "runs": [entry]}
    plan_path = tmp_path / "PLAN.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path, target


def _tmp_plan_multi(reg, tmp_path, source_entries):
    """Plan with several rewritten entries under tmp_path/regenerated_v1."""
    regen = tmp_path / "regenerated_v1"
    entries = []
    for source_entry in source_entries:
        entry = json.loads(json.dumps(source_entry))
        target = regen / entry["campaign"] / entry["job"]
        entry["regenerated_run_dir"] = str(target)
        entry["evaluate_kwargs"]["output"] = str(target / "run")
        entries.append(entry)
    plan = {"schema": "evidence_regeneration_plan/1", "runs": entries}
    plan_path = tmp_path / "PLAN.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path, regen


def _fake_evaluate_factory(calls):
    """Mock workflow.evaluate: creates the output dir + the files dataset needs."""

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output"])
        output.mkdir(parents=True)
        (output / "telemetry.jsonl").write_text("{}\n", encoding="utf-8")
        (output / "physics_steps.jsonl").write_text('{"time_s": 1}\n', encoding="utf-8")
        return {"ok": True}

    return fake_evaluate


@requires_local_tree
def test_run_executes_planned_kwargs_and_refuses_existing_dir(reg, real_plan, tmp_path,
                                                              monkeypatch):
    plan, _ = real_plan
    plan_path, target = _tmp_plan_entry(reg, tmp_path, _entry(plan, "cv17_native"))
    calls = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output"])
        output.mkdir(parents=True)
        (output / "telemetry.jsonl").write_text("{}\n", encoding="utf-8")
        return {"summary": "ok"}

    receipt = reg.run_single("cv17_native", plan_path, tmp_path / "regenerated_v1",
                             evaluate_fn=fake_evaluate)
    tmp_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(calls) == 1 and receipt["exit_status"] == "complete"
    assert calls[0]["seed"] == tmp_plan["runs"][0]["evaluate_kwargs"]["seed"]
    assert calls[0]["loading"] == tmp_plan["runs"][0]["evaluate_kwargs"]["loading"]
    assert Path(receipt["regenerated_run_dir"]).is_relative_to(tmp_path)
    assert "active" not in Path(receipt["regenerated_run_dir"]).parts
    assert receipt["started_utc"] and receipt["finished_utc"]
    assert receipt["evaluate_kwargs"] == tmp_plan["runs"][0]["evaluate_kwargs"]
    assert (target / "REGENERATION_RECEIPT.json").is_file()

    # A second attempt must refuse before evaluate is called again.
    with pytest.raises(reg.RegenerationError, match="refusing to write into existing"):
        reg.run_single("cv17_native", plan_path, tmp_path / "regenerated_v1",
                       evaluate_fn=fake_evaluate)
    assert len(calls) == 1

    # Pre-created empty target directory is refused too.
    (tmp_path / "second").mkdir()
    other_plan_path, other_target = _tmp_plan_entry(
        reg, tmp_path / "second", _entry(plan, "cv17_retain"))
    other_target.mkdir(parents=True)
    with pytest.raises(reg.RegenerationError, match="refusing to write into existing"):
        reg.run_single("cv17_retain", other_plan_path, tmp_path / "second" / "regenerated_v1",
                       evaluate_fn=fake_evaluate)


@requires_local_tree
def test_run_never_targets_runs_active(reg, real_plan, tmp_path):
    plan, _ = real_plan
    source = _entry(plan, "cv17_native")
    plan_path, _target = _tmp_plan_entry(reg, tmp_path, source)
    tampered = json.loads(plan_path.read_text(encoding="utf-8"))
    tampered["runs"][0]["job"] = "active/escape"
    tampered["runs"][0]["evaluate_kwargs"]["output"] = str(
        tmp_path / "regenerated_v1" / "active" / "escape" / "run")
    bad_plan = tmp_path / "BAD_PLAN.json"
    bad_plan.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(reg.RegenerationError, match="unsafe campaign/job name"):
        reg.run_single("cv17_native", bad_plan, tmp_path / "regenerated_v1",
                       evaluate_fn=lambda **kwargs: None)
    with pytest.raises(reg.RegenerationError, match="runs/active"):
        reg.run_single("cv17_native", plan_path,
                       ROOT / "runs" / "active", evaluate_fn=lambda **kwargs: None)


class _FakeTraci:
    """Minimal traci stand-in for the runner-side instrumentation."""

    def __init__(self):
        self.vehicle = types.SimpleNamespace(
            getIDList=lambda: ("cv0", "hd0"),
            getSpeedMode=lambda v: 31,
            getRoadID=lambda v: "E0",
            getLaneID=lambda v: "E0_0",
            getLanePosition=lambda v: 10.0,
            getPosition=lambda v: (1.0, 2.0),
            getSpeed=lambda v: 20.0,
            getAcceleration=lambda v: 0.5,
            getTypeID=lambda v: "passenger",
            slowDown=lambda *a: None,
            setSpeed=lambda *a: None,
        )
        self.simulation = types.SimpleNamespace(
            getTime=lambda: 1.0,
            getDepartedIDList=lambda: ("cv0",),
            getArrivedIDList=lambda: (),
        )


@requires_local_tree
def test_run_default_path_instruments_evaluate_with_mocked_traci(reg, real_plan, tmp_path,
                                                                 monkeypatch):
    plan, _ = real_plan
    plan_path, target = _tmp_plan_entry(reg, tmp_path, _entry(plan, "cv17_native"))
    monkeypatch.setitem(sys.modules, "traci", _FakeTraci())
    seen = {}

    def fake_evaluate(**kwargs):
        seen.update(kwargs)
        output = Path(kwargs["output"])
        output.mkdir(parents=True)
        (output / "telemetry.jsonl").write_text("{}\n", encoding="utf-8")
        import traci  # the fake injected above
        traci.vehicle.slowDown("cv0", 5.0)
        kwargs["progress_callback"](1.0)
        return {"manifest_status": "complete"}

    import traffic_shepherd.workflow as workflow
    monkeypatch.setattr(workflow, "evaluate", fake_evaluate)
    receipt = reg.run_single("cv17_native", plan_path, tmp_path / "regenerated_v1")
    assert receipt["exit_status"] == "complete"
    assert seen["controller_name"] == "damping"
    assert seen["penetration"] == 0.2 and seen["duration"] == 2400 and seen["role"] == "exposed"
    assert type(seen["fleet_switch"]).__name__ == "SingleSwitchSpec"
    assert seen["fleet_switch"].suffix == "native"
    physics = (target / "run" / "physics_steps.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(physics)["time_s"] == 1.0
    trace = (target / "run" / "writer_trace.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(trace)["method"] == "slowDown"
    assert receipt["current_source_hashes"]["workflow.py"]["sha256"]


@requires_local_tree
def test_run_records_failure_receipt_from_mock(reg, real_plan, tmp_path):
    plan, _ = real_plan
    plan_path, target = _tmp_plan_entry(reg, tmp_path, _entry(plan, "cv17_native"))

    def failing(**kwargs):
        raise RuntimeError("simulated failure")

    receipt = reg.run_single("cv17_native", plan_path, tmp_path / "regenerated_v1",
                             evaluate_fn=failing)
    assert receipt["exit_status"] == "failed"
    assert "simulated failure" in receipt["error"]
    stored = json.loads((target / "REGENERATION_RECEIPT.json").read_text(encoding="utf-8"))
    assert stored["exit_status"] == "failed"


# ---------------------------------------------------------------------------
# dataset rebuild + comparison records
# ---------------------------------------------------------------------------

def _dataset_plan(tmp_path, expected_sha, with_telemetry=True):
    run_dir = tmp_path / "regenerated_v1" / "cv17" / "native" / "run"
    run_dir.mkdir(parents=True)
    if with_telemetry:
        (run_dir / "telemetry.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "physics_steps.jsonl").write_text("{}\n", encoding="utf-8")
    entry = {
        "run_tag": "cv17_native", "campaign": "cv17", "job": "native", "split": "train",
        "expected_output_sha256": expected_sha, "manifest_output_file": "data/train/x.jsonl",
        "regenerated_run_dir": str(run_dir.parent),
    }
    plan_path = tmp_path / "PLAN.json"
    plan_path.write_text(json.dumps({"runs": [entry]}), encoding="utf-8")
    return plan_path, run_dir


def _fake_builder(payload):
    def build_run(run_dir, run_tag, out_path):
        Path(out_path).write_bytes(payload)
        return {"output_sha256": hashlib.sha256(payload).hexdigest(), "n_examples": 0,
                "events_total": 0, "rsu_events_total": 0}
    return build_run


def test_dataset_comparison_records_match_and_mismatch(reg, tmp_path):
    payload = b'{"example": 1}\n'
    good_sha = hashlib.sha256(payload).hexdigest()
    plan_path, _run_dir = _dataset_plan(tmp_path, good_sha)
    comparison = reg.dataset_single("cv17_native", plan_path, tmp_path / "dataset_v2",
                                    build_run_fn=_fake_builder(payload))
    assert comparison["match"] is True
    assert comparison["computed_output_sha256"] == good_sha
    built = tmp_path / "dataset_v2" / "train" / "cv17_native.jsonl"
    assert built.read_bytes() == payload
    assert (tmp_path / "dataset_v2" / "train" / "cv17_native.comparison.json").is_file()

    mismatch_plan, _ = _dataset_plan(tmp_path / "other", "0" * 64)
    mismatch = reg.dataset_single("cv17_native", mismatch_plan, tmp_path / "other" / "dataset_v2",
                                  build_run_fn=_fake_builder(payload))
    assert mismatch["match"] is False
    assert mismatch["expected_output_sha256"] != mismatch["computed_output_sha256"]

    with pytest.raises(reg.RegenerationError, match="refusing to overwrite"):
        reg.dataset_single("cv17_native", plan_path, tmp_path / "dataset_v2",
                           build_run_fn=_fake_builder(payload))


def test_dataset_requires_completed_run(reg, tmp_path):
    plan_path, _run_dir = _dataset_plan(tmp_path, "0" * 64, with_telemetry=False)
    with pytest.raises(reg.RegenerationError, match="telemetry.jsonl missing"):
        reg.dataset_single("cv17_native", plan_path, tmp_path / "dataset_v2",
                           build_run_fn=_fake_builder(b"x"))


# ---------------------------------------------------------------------------
# prefix mode (run --horizon-s)
# ---------------------------------------------------------------------------

@requires_local_tree
def test_run_horizon_validation_and_receipt(reg, real_plan, tmp_path):
    plan, _ = real_plan
    plan_path, _target = _tmp_plan_entry(reg, tmp_path, _entry(plan, "cv17_native"))
    regen = tmp_path / "regenerated_v1"
    for bad in (601.0, 2401.0):
        with pytest.raises(reg.RegenerationError, match="outside the allowed"):
            reg.run_single("cv17_native", plan_path, regen, evaluate_fn=lambda **k: None,
                           horizon_s=bad)
    calls = []
    receipt = reg.run_single("cv17_native", plan_path, regen,
                             evaluate_fn=_fake_evaluate_factory(calls), horizon_s=602)
    assert calls[0]["duration"] == 602.0
    assert receipt["horizon_s"] == 602 and receipt["horizon_mode"] == "prefix"
    stored = json.loads((tmp_path / "regenerated_v1" / "cv17" / "native" /
                         "REGENERATION_RECEIPT.json").read_text(encoding="utf-8"))
    assert stored["horizon_s"] == 602 and stored["horizon_mode"] == "prefix"

    (tmp_path / "full").mkdir()
    plan_path2, _t2 = _tmp_plan_entry(reg, tmp_path / "full", _entry(plan, "cv17_retain"))
    calls2 = []
    receipt2 = reg.run_single("cv17_retain", plan_path2, tmp_path / "full" / "regenerated_v1",
                              evaluate_fn=_fake_evaluate_factory(calls2))
    assert calls2[0]["duration"] == 2400
    assert receipt2["horizon_s"] == 2400 and receipt2["horizon_mode"] == "full"


@requires_local_tree
def test_prefix_run_feeds_dataset_step(reg, real_plan, tmp_path):
    plan, _ = real_plan
    plan_path, target = _tmp_plan_entry(reg, tmp_path, _entry(plan, "cv17_native"))
    calls = []
    reg.run_single("cv17_native", plan_path, tmp_path / "regenerated_v1",
                   evaluate_fn=_fake_evaluate_factory(calls), horizon_s=602)
    assert calls[0]["duration"] == 602
    payload = b'{"example": 1}\n'

    def build_run(run_dir, run_tag, out_path):
        Path(out_path).write_bytes(payload)
        return {"output_sha256": hashlib.sha256(payload).hexdigest(), "n_examples": 0}

    comparison = reg.dataset_single("cv17_native", plan_path, tmp_path / "dataset_v2",
                                    build_run_fn=build_run)
    assert (tmp_path / "dataset_v2" / "train" / "cv17_native.jsonl").read_bytes() == payload
    assert comparison["rebuilt_from"] == str(target / "run")


# ---------------------------------------------------------------------------
# batch (child processes mocked via the launch seam)
# ---------------------------------------------------------------------------

@requires_local_tree
def test_batch_dispatches_children_logs_and_skips(reg, real_plan, tmp_path):
    plan, _ = real_plan
    tags = ("cv17_retain", "cv17_native", "cv17_pi-external-u")
    plan_path, regen = _tmp_plan_multi(reg, tmp_path, [_entry(plan, t) for t in tags])
    launches = []

    def fake_launch(argv):
        argv = [str(a) for a in argv]
        launches.append(argv)
        return 0

    summary = reg.batch_single(tags=",".join(tags), plan_path=plan_path, regen_root=regen,
                               dataset_root=tmp_path / "dataset_v2",
                               batches_root=tmp_path / "batches", launch=fake_launch)
    assert summary["dispatched"] == 3 and summary["skipped"] == 0
    assert summary["matched"] == 3 and summary["mismatched"] == 0 and summary["failed"] == 0
    # each run executed as its own child process: one run + one dataset invocation per tag
    run_calls = [argv for argv in launches if argv[2] == "run"]
    dataset_calls = [argv for argv in launches if argv[2] == "dataset"]
    assert len(run_calls) == 3 and len(dataset_calls) == 3
    for argv in run_calls:
        assert argv[1].endswith("regenerate.py") and "--horizon-s" not in argv
    log_lines = [json.loads(line) for line in
                 next((tmp_path / "batches").glob("*.jsonl")).read_text(
                     encoding="utf-8").splitlines()]
    assert len(log_lines) == 3
    assert {line["tag"] for line in log_lines} == set(tags)
    assert all(line["run_exit_code"] == 0 and line["dataset_exit_code"] == 0
               and line["match"] is True and line["status"] == "ok" for line in log_lines)

    # A tag whose regeneration target already exists is skipped, not rerun.
    (regen / "cv17" / "native").mkdir(parents=True)
    launches.clear()
    (tmp_path / "batches2").mkdir()
    summary2 = reg.batch_single(tags=",".join(tags), plan_path=plan_path, regen_root=regen,
                                dataset_root=tmp_path / "dataset_v2",
                                batches_root=tmp_path / "batches2", launch=fake_launch)
    assert summary2["skipped"] == 1 and summary2["dispatched"] == 2
    skipped_lines = [json.loads(line) for line in
                     next((tmp_path / "batches2").glob("*.jsonl")).read_text(
                         encoding="utf-8").splitlines()]
    assert [line["status"] for line in skipped_lines].count("skipped") == 1

    with pytest.raises(reg.RegenerationError, match="--tags"):
        reg.batch_single(plan_path=plan_path, regen_root=regen, launch=fake_launch)
    with pytest.raises(reg.RegenerationError, match="outside the allowed"):
        reg.batch_single(tags="cv17_retain", horizon_s=100, plan_path=plan_path,
                         regen_root=regen, launch=fake_launch)


@requires_local_tree
def test_batch_stop_on_mismatch_and_parallel_workers(reg, real_plan, tmp_path):
    plan, _ = real_plan
    tags = ("cv17_retain", "cv17_native", "cv17_pi-external-u")
    plan_path, regen = _tmp_plan_multi(reg, tmp_path, [_entry(plan, t) for t in tags])
    launches = []

    def mismatching_launch(argv):
        argv = [str(a) for a in argv]
        launches.append(argv)
        if argv[2] == "dataset" and "--run-tag" in argv and \
                argv[argv.index("--run-tag") + 1] == "cv17_native":
            return 4
        return 0

    summary = reg.batch_single(tags=",".join(tags), workers=1, stop_on_mismatch=True,
                               plan_path=plan_path, regen_root=regen,
                               dataset_root=tmp_path / "dataset_v2",
                               batches_root=tmp_path / "batches",
                               launch=mismatching_launch)
    assert summary["mismatched"] == 1 and summary["stopped_early"] is True
    dispatched_tags = {argv[argv.index("--run-tag") + 1] for argv in launches if argv[2] == "run"}
    assert dispatched_tags == {"cv17_retain", "cv17_native"}

    launches.clear()
    summary2 = reg.batch_single(tags=",".join(tags), workers=2,
                                plan_path=plan_path, regen_root=regen,
                                dataset_root=tmp_path / "dataset_v2",
                                batches_root=tmp_path / "batches2",
                                launch=mismatching_launch)
    # without stop_on_mismatch the remaining run is still dispatched despite the mismatch
    assert summary2["dispatched"] == 3 and summary2["stopped_early"] is False


# ---------------------------------------------------------------------------
# restore-dataset
# ---------------------------------------------------------------------------

def _restore_setup(tmp_path, *, match=True, corrupt_source=False, dest_exists=False,
                   no_comparison=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = b'{"example": 1}\n'
    expected_sha = hashlib.sha256(payload).hexdigest()
    entry = {"run_tag": "cv17_native", "campaign": "cv17", "job": "native", "split": "train",
             "expected_output_sha256": expected_sha}
    plan_path = tmp_path / "PLAN.json"
    plan_path.write_text(json.dumps({"runs": [entry]}), encoding="utf-8")
    dataset_root = tmp_path / "dataset_v2"
    split_dir = dataset_root / "train"
    split_dir.mkdir(parents=True)
    source = split_dir / "cv17_native.jsonl"
    source.write_bytes(b'{"example": 2}\n' if corrupt_source else payload)
    (split_dir / "cv17_native.aggregate.json").write_text("{}", encoding="utf-8")
    if not no_comparison:
        (split_dir / "cv17_native.comparison.json").write_text(
            json.dumps({"match": match, "computed_output_sha256": expected_sha}),
            encoding="utf-8")
    restore_root = tmp_path / "worldmodel_v2"
    if dest_exists:
        destination = restore_root / "data" / "train" / "cv17_native.jsonl"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"pre-existing")
    return plan_path, dataset_root, restore_root, payload, expected_sha


def test_restore_restores_only_verified_matches(reg, tmp_path):
    plan_path, dataset_root, restore_root, payload, sha = _restore_setup(tmp_path)
    receipts_root = tmp_path / "restore_receipts"
    receipt = reg.restore_single(tags="cv17_native", plan_path=plan_path,
                                 dataset_root=dataset_root, dataset_v2_root=restore_root,
                                 receipts_root=receipts_root)
    assert [r["tag"] for r in receipt["restored"]] == ["cv17_native"]
    assert receipt["refused"] == []
    destination = restore_root / "data" / "train" / "cv17_native.jsonl"
    assert destination.read_bytes() == payload
    assert (restore_root / "aggregates" / "cv17_native.aggregate.json").is_file()
    assert receipt["receipt_path"] and Path(receipt["receipt_path"]).is_file()
    stored = json.loads(Path(receipt["receipt_path"]).read_text(encoding="utf-8"))
    assert stored["restored"][0]["sha256"] == sha

    cases = [
        (_restore_setup(tmp_path / "c1", match=False), "does not report match"),
        (_restore_setup(tmp_path / "c2", corrupt_source=True), "recomputed sha256"),
        (_restore_setup(tmp_path / "c3", dest_exists=True), "destination already exists"),
        (_restore_setup(tmp_path / "c4", no_comparison=True), "no comparison record"),
    ]
    for (plan_path, dataset_root_x, restore_root_x, _p, _s), reason in cases:
        outcome = reg.restore_single(tags="cv17_native", plan_path=plan_path,
                                     dataset_root=dataset_root_x,
                                     dataset_v2_root=restore_root_x,
                                     receipts_root=tmp_path / "rr")
        assert outcome["restored"] == []
        assert len(outcome["refused"]) == 1
        assert reason in outcome["refused"][0]["reason"]


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def test_cli_help_and_subcommands(reg, capsys):
    for argv in (["--help"], ["plan", "--help"], ["run", "--help"], ["batch", "--help"],
                 ["dataset", "--help"], ["restore-dataset", "--help"]):
        with pytest.raises(SystemExit) as excinfo:
            reg.main(argv)
        assert excinfo.value.code == 0
        capsys.readouterr()
    with pytest.raises(SystemExit) as excinfo:
        reg.main([])
    assert excinfo.value.code == 2
    capsys.readouterr()
    with pytest.raises(SystemExit) as excinfo:
        reg.main(["batch"])
    assert excinfo.value.code == 2
    capsys.readouterr()
