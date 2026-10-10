"""Synthetic launcher contracts; never use a real pack, shard, or model."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "run_train_stage.py"
spec = importlib.util.spec_from_file_location("wm_train_stage_launcher_test", LAUNCHER)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def _plan():
    return {"manifest_sha256": "a" * 64, "index_sha256": "b" * 64,
            "k_max": 192, "n_items": 1,
            "selection": [{"shard_rel": "chunk1/w1/native", "split": "TRAIN",
                           "key": "chunk1/w1__native", "position": 0,
                           "tick": 0, "plan_row": 0, "tick_start": 0,
                           "tick_end": 1, "dec_start": 0, "dec_end": 1,
                           "batch": "chunk1", "chunk": "chunk1", "world": "w1",
                           "arm": "native", "t0_s": 600, "n_ticks": 1,
                           "n_outcome_rows": 1, "input_sha256": "c" * 64}]}


def _mock_modules(monkeypatch, plan, calls):
    ids = types.ModuleType("train.identity_sidecar")
    ids._plan = lambda pack: calls.append(("plan", pack)) or plan
    ids.load_identity_sidecar = lambda pack, directory: (
        calls.append(("identity", pack, directory)) or object())
    label = types.ModuleType("train.label_isolation")
    label._validate = lambda pack, directory: calls.append(("label", pack, directory))
    label.build_sidecar = lambda pack, root, out: (
        calls.append(("label_build", pack, root, out)) or {"sidecar": {"n_rows": 1}})
    idea = types.ModuleType("train.idea_a")
    idea.IdeaATrainConfig = lambda **_kw: types.SimpleNamespace(
        batch_size=8, device="cpu", state_global_token=True,
        quick_tune=False, allow_incomplete_losses=False,
        subset_frac=1.0, epoch_frac=1.0)

    def main(argv):
        calls.append(("idea_main", argv))
        out = Path(argv[argv.index("--out") + 1])
        (out / "TRAIN_RECEIPT.json").write_text(json.dumps({
            "member": 1, "seed": 101, "steps": 40, "epochs": 1,
            "num_workers": int(argv[argv.index("--num-workers") + 1]),
            "state_global_token": True, "quick_tune": False,
            "allow_incomplete_losses": False,
            "required_loss_terms": ["a", "b", "c", "d", "e"],
            "active_loss_terms": ["a", "b", "c", "d", "e"]}), encoding="utf-8")
        return 0

    idea.main = main
    train = types.ModuleType("train")
    train.identity_sidecar, train.label_isolation, train.idea_a = ids, label, idea
    seals = types.ModuleType("evalh.seals")
    seals.require_frozen_amendment_6 = lambda: calls.append(("a6",)) or "d" * 64
    evalh = types.ModuleType("evalh")
    evalh.seals = seals
    torch = types.ModuleType("torch")
    torch.set_num_threads = lambda n: calls.append(("threads", n))
    for name, module in (("train", train), ("train.identity_sidecar", ids),
                         ("train.label_isolation", label), ("train.idea_a", idea),
                         ("evalh", evalh), ("evalh.seals", seals), ("torch", torch)):
        monkeypatch.setitem(sys.modules, name, module)
    return ids, label, idea


def _mock_guard(monkeypatch, stage, inputs, plan, calls):
    guard = types.ModuleType("train_stage_guard")
    guard.assert_startup = lambda **kw: calls.append(("guard_startup", kw))
    guard.ensure_required_wrappers = lambda: calls.append(("wrappers",))
    guard._lexnorm = lambda path: str(Path(path)).lower()
    pins = {}
    for rel in inputs.get("selected_shards", []):
        shard = Path(inputs["shards_root"]).joinpath(*rel.split("/"))
        for name in launcher.SHARD_COMMON_NAMES:
            pins[guard._lexnorm(str(shard / name))] = {"role": "data", "sha256": "x"}
    for rel in inputs.get("positive_shards", []):
        shard = Path(inputs["shards_root"]).joinpath(*rel.split("/"))
        pins[guard._lexnorm(str(shard / launcher.SHARD_OUTCOME_NAME))] = {
            "role": "data", "sha256": "d" * 64}
    guard._POLICY = types.SimpleNamespace(
        stage=stage, spec={"stage_inputs": inputs,
                           "selection_sha256": launcher.canonical_sha(plan)}, files=pins)
    monkeypatch.setitem(sys.modules, "train_stage_guard", guard)
    return guard


def test_cpu40_calls_current_idea_a_main_with_frozen_argv(monkeypatch, tmp_path):
    calls, plan = [], _plan()
    _mock_modules(monkeypatch, plan, calls)
    out = tmp_path / "new-cpu"
    inputs = {"pack": str(tmp_path / "pack"),
              "identity_dir": str(tmp_path / "identity"),
              "sidecar_dir": str(tmp_path / "labels"),
              "cpu_out": str(out), "num_workers": 2}
    _mock_guard(monkeypatch, "cpu40", inputs, plan, calls)
    monkeypatch.setenv("WM_N1_TRAIN_IDENTITIES", inputs["identity_dir"])
    monkeypatch.setenv("WM_N1_TRAIN_OUTCOMES", inputs["sidecar_dir"])
    result = launcher.run_stage_child()
    assert out.is_dir()
    assert result["steps_requested"] == 40
    assert [call[0] for call in calls] == ["guard_startup", "wrappers", "plan",
                                           "a6", "identity", "label", "threads",
                                           "idea_main"]
    argv = calls[-1][1]
    assert argv == ["--member", "1", "--seed", "101", "--corpus", inputs["pack"],
                    "--out", str(out), "--device", "cpu", "--ensemble-k", "5",
                    "--epochs", "1", "--steps-per-epoch", "40", "--k-max", "192",
                    "--num-workers", "2", "--log-every", "1"]


def test_cpu40_refuses_bad_identity_before_label_or_model(monkeypatch, tmp_path):
    calls, plan = [], _plan()
    ids, _, _ = _mock_modules(monkeypatch, plan, calls)
    ids.load_identity_sidecar = lambda *_args: (_ for _ in ()).throw(
        launcher.LauncherRefusal("identity receipt drift"))
    inputs = {"pack": str(tmp_path / "pack"),
              "identity_dir": str(tmp_path / "identity"),
              "sidecar_dir": str(tmp_path / "labels"),
              "cpu_out": str(tmp_path / "out"), "num_workers": 0}
    _mock_guard(monkeypatch, "cpu40", inputs, plan, calls)
    monkeypatch.setenv("WM_N1_TRAIN_IDENTITIES", inputs["identity_dir"])
    monkeypatch.setenv("WM_N1_TRAIN_OUTCOMES", inputs["sidecar_dir"])
    with pytest.raises(launcher.LauncherRefusal, match="identity receipt drift"):
        launcher.run_stage_child()
    assert [name for name, *_ in calls] == ["guard_startup", "wrappers", "plan", "a6"]
    assert not Path(inputs["cpu_out"]).exists()


def test_metadata_stage_calls_accepted_plan_and_writes_only_receipt(monkeypatch, tmp_path):
    calls, plan = [], _plan()
    _mock_modules(monkeypatch, plan, calls)
    out = tmp_path / "new-plan"
    inputs = {"pack": str(tmp_path / "pack"), "plan_out": str(out)}
    _mock_guard(monkeypatch, "metadata-plan", inputs, plan, calls)
    launcher.run_stage_child()
    receipt = json.loads((out / "PLAN_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["plan"] == plan
    assert receipt["selection_sha256"] == launcher.canonical_sha(plan)
    assert [name for name, *_ in calls] == ["guard_startup", "wrappers", "plan"]


def test_sidecar_stage_verifies_done_then_builds_identity_then_labels(monkeypatch, tmp_path):
    calls, plan = [], _plan()
    ids, _, _ = _mock_modules(monkeypatch, plan, calls)
    selection = plan["selection"][0]
    done = dict(selection, schema="wm_n1_features_index/1",
                outputs={"slot_index": "x", "slot_vid.json": "x",
                         "outcomes.parquet": "d" * 64})
    ids._file = lambda root, name: Path(root) / name
    ids._json = lambda _path: calls.append(("done",)) or done
    ids._relative_shard = lambda _done: selection["shard_rel"]
    ids._sha = lambda path: calls.append(("source_hash",)) or (
        "d" * 64 if Path(path).name == "outcomes.parquet" else "x")
    ids.build_identity_sidecar = lambda pack, root, out: (
        calls.append(("identity_build", pack, root, out)) or {"n_vehicles": 1})
    inputs = {"pack": str(tmp_path / "pack"),
              "shards_root": str(tmp_path / "shards"),
              "selected_shards": [selection["shard_rel"]],
              "positive_shards": [selection["shard_rel"]],
              "identity_out": str(tmp_path / "identity"),
              "sidecar_out": str(tmp_path / "labels")}
    _mock_guard(monkeypatch, "train-sidecars", inputs, plan, calls)
    launcher.run_stage_child()
    names = [name for name, *_ in calls]
    assert names.index("source_hash") < names.index("done")
    assert names.index("done") < names.index("identity_build")
    assert names.index("source_hash") < names.index("identity_build")
    assert names.index("identity_build") < names.index("label_build")


def test_sidecar_stage_refuses_done_drift_before_builders(monkeypatch, tmp_path):
    calls, plan = [], _plan()
    ids, _, _ = _mock_modules(monkeypatch, plan, calls)
    ids._file = lambda root, name: Path(root) / name
    ids._json = lambda _path: {"schema": "wrong"}
    ids._sha = lambda _path: "x"
    ids._relative_shard = lambda _done: "chunk1/w1/native"
    ids.build_identity_sidecar = lambda *_args: pytest.fail("identity builder ran")
    inputs = {"pack": str(tmp_path / "pack"),
              "shards_root": str(tmp_path / "shards"),
              "selected_shards": ["chunk1/w1/native"],
              "positive_shards": ["chunk1/w1/native"],
              "identity_out": str(tmp_path / "identity"),
              "sidecar_out": str(tmp_path / "labels")}
    _mock_guard(monkeypatch, "train-sidecars", inputs, plan, calls)
    with pytest.raises(launcher.LauncherRefusal, match="DONE schema/path"):
        launcher.run_stage_child()
    assert "label_build" not in [name for name, *_ in calls]


def test_zero_row_done_requires_absent_outcome_key_and_file(monkeypatch, tmp_path):
    calls, plan = [], _plan()
    plan["selection"][0]["n_outcome_rows"] = 0
    ids, _, _ = _mock_modules(monkeypatch, plan, calls)
    selection = plan["selection"][0]
    done = dict(selection, schema="wm_n1_features_index/1",
                outputs={"slot_index": "x", "slot_vid.json": "x"})
    ids._file = lambda root, name: Path(root) / name
    ids._json = lambda _path: done
    ids._relative_shard = lambda _done: selection["shard_rel"]
    ids._sha = lambda path: calls.append(("hash", Path(path).name)) or "x"
    ids.build_identity_sidecar = lambda *_args: {"n_vehicles": 1}
    inputs = {"pack": str(tmp_path / "pack"),
              "shards_root": str(tmp_path / "shards"),
              "selected_shards": [selection["shard_rel"]],
              "positive_shards": [],
              "identity_out": str(tmp_path / "identity"),
              "sidecar_out": str(tmp_path / "labels")}
    _mock_guard(monkeypatch, "train-sidecars", inputs, plan, calls)
    launcher.run_stage_child()
    assert ("hash", "outcomes.parquet") not in calls
    done["outputs"]["outcomes.parquet"] = "d" * 64
    with pytest.raises(launcher.LauncherRefusal, match="falsely records"):
        launcher.run_stage_child()
    del done["outputs"]["outcomes.parquet"]
    outcome = tmp_path / "shards" / "chunk1" / "w1" / "native" / "outcomes.parquet"
    outcome.parent.mkdir(parents=True)
    outcome.write_bytes(b"synthetic unexpected file")
    with pytest.raises(launcher.LauncherRefusal, match="unexpected outcomes file"):
        launcher.run_stage_child()


def test_stage_data_paths_require_outcomes_only_for_positive_train_runs(tmp_path):
    inputs = {"pack": str(tmp_path / "pack"),
              "plan_receipt": str(tmp_path / "prior" / "PLAN_RECEIPT.json"),
              "shards_root": str(tmp_path / "shards"),
              "positive_shards": ["chunk1/w1/native"]}
    selected = ["chunk1/w1/native", "chunk1/w2/native"]
    paths = launcher._expected_data("train-sidecars", inputs, selected)
    for rel in selected:
        shard = Path(inputs["shards_root"]).joinpath(*rel.split("/"))
        assert all(str(shard / name) in paths for name in launcher.SHARD_COMMON_NAMES)
        assert (str(shard / launcher.SHARD_OUTCOME_NAME) in paths) == (rel == selected[0])


def test_positive_outcome_subset_must_match_pinned_plan_before_shard_hash(
        monkeypatch, tmp_path):
    plan = _plan()
    receipt = tmp_path / "PLAN_RECEIPT.json"
    inputs = {"pack": str(tmp_path / "pack"), "plan_receipt": str(receipt),
              "selected_shards": ["chunk1/w1/native"], "positive_shards": []}
    receipt.write_text(json.dumps({"schema": launcher.PLAN_SCHEMA,
                                   "pack": inputs["pack"], "plan": plan,
                                   "selection_sha256": launcher.canonical_sha(plan)}),
                       encoding="utf-8")
    files = {str(receipt): {"role": "data", "sha256": launcher.digest(receipt)},
             str(Path(inputs["pack"]) / "manifest.json"):
                 {"role": "data", "sha256": "a" * 64},
             str(Path(inputs["pack"]) / "INDEX.json"):
                 {"role": "data", "sha256": "b" * 64}}
    monkeypatch.setattr(launcher, "policy_inputs",
                        lambda _policy: ("train-sidecars", inputs))
    original_digest = launcher.digest
    monkeypatch.setattr(launcher, "digest", lambda path:
                        original_digest(path) if Path(path) == receipt else
                        pytest.fail("hashed shard before positive subset validation"))
    with pytest.raises(launcher.LauncherRefusal, match="positive outcome shards"):
        launcher.plan_and_selection({"files": files})


def test_parent_refuses_zero_outcome_leaf_before_any_registered_data_hash(
        monkeypatch, tmp_path):
    plan = _plan()
    plan["selection"][0]["n_outcome_rows"] = 0
    selected = [plan["selection"][0]["shard_rel"]]
    inputs = {"pack": str(tmp_path / "pack"),
              "plan_receipt": str(tmp_path / "prior" / "PLAN_RECEIPT.json"),
              "shards_root": str(tmp_path / "shards"),
              "positive_shards": []}
    expected = launcher._expected_data("train-sidecars", inputs, selected)
    policy = {"files": {path: {"role": "data", "sha256": "a" * 64}
                        for path in expected}}
    leaf = tmp_path / "shards" / "chunk1" / "w1" / "native" / "outcomes.parquet"
    leaf.parent.mkdir(parents=True)
    leaf.write_bytes(b"synthetic unexpected file")
    monkeypatch.setattr(launcher, "policy_inputs",
                        lambda _policy: ("train-sidecars", inputs))
    monkeypatch.setattr(launcher, "digest",
                        lambda _path: pytest.fail("hashed before zero-row refusal"))
    with pytest.raises(launcher.LauncherRefusal, match="unexpected outcomes file"):
        launcher.validate_stage_data(policy, plan, selected)


def test_foreign_stage_file_registration_refuses_before_hash(monkeypatch, tmp_path):
    pack = tmp_path / "pack"
    inputs = {"pack": str(pack)}
    policy = {"files": {str(pack / "manifest.json"): {"role": "data"},
                        str(pack / "INDEX.json"): {"role": "data"},
                        str(pack / "outcomes.parquet"): {"role": "data"}}}
    monkeypatch.setattr(launcher, "policy_inputs", lambda _policy: ("metadata-plan", inputs))
    monkeypatch.setattr(launcher, "digest", lambda _path: pytest.fail("hashed before refusal"))
    with pytest.raises(launcher.LauncherRefusal, match="stage data pins"):
        launcher.verify_data_registrations(policy)


def test_cpu_stage_has_no_shard_data_registration(monkeypatch, tmp_path):
    inputs = {"pack": str(tmp_path / "pack"),
              "plan_receipt": str(tmp_path / "plan" / "PLAN_RECEIPT.json"),
              "identity_dir": str(tmp_path / "identity"),
              "sidecar_dir": str(tmp_path / "labels")}
    expected = launcher._expected_data("cpu40", inputs, [])
    assert not any("shards" in path or "outcomes.parquet" == Path(path).name
                   for path in expected)
    assert len(expected) == 2 + 1 + 15 + 4 + 2


def test_parent_metadata_planning_imports_no_project_or_native_code(monkeypatch):
    monkeypatch.setattr(launcher, "policy_inputs",
                        lambda _policy: ("metadata-plan", {"pack": "synthetic"}))
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "train" or name.startswith("train.") or name == "numpy":
            pytest.fail("parent imported project or native source")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    assert launcher.plan_and_selection({}) == (None, [])


def test_prior_plan_drift_refuses_before_selected_source_hash(monkeypatch, tmp_path):
    plan = _plan()
    path = tmp_path / "PLAN_RECEIPT.json"
    path.write_text(json.dumps({"schema": launcher.PLAN_SCHEMA,
                                "pack": str(tmp_path / "pack"), "plan": plan,
                                "selection_sha256": "0" * 64}), encoding="utf-8")
    inputs = {"pack": str(tmp_path / "pack"), "plan_receipt": str(path),
              "selected_shards": ["chunk1/w1/native"],
              "positive_shards": ["chunk1/w1/native"]}
    policy = {"files": {str(path): {"role": "data", "sha256": launcher.digest(path)},
                        str(tmp_path / "pack" / "manifest.json"):
                            {"role": "data", "sha256": "a" * 64},
                        str(tmp_path / "pack" / "INDEX.json"):
                            {"role": "data", "sha256": "b" * 64}}}
    monkeypatch.setattr(launcher, "policy_inputs",
                        lambda _policy: ("train-sidecars", inputs))
    with pytest.raises(launcher.LauncherRefusal, match="selection SHA"):
        launcher.plan_and_selection(policy)


def test_foreign_cal_config_refuses_before_data_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "WORKTREE", tmp_path)
    monkeypatch.setattr(launcher, "policy_inputs",
                        lambda _policy: ("metadata-plan", {"pack": "synthetic"}))
    foreign = tmp_path / "synthetic_CAL_SPLIT_A4.json"
    foreign.write_text("{}", encoding="utf-8")
    payload = {"version": 1, "stage": "metadata-plan", "sandbox": str(tmp_path / "new"),
               "stage_inputs": {}, "files": {str(foreign):
                                            {"role": "config", "sha256": "0" * 64}},
               "interpreter": {}, "pythonpath": []}
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    expected_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    seen = []

    def spy(path):
        seen.append(str(path))
        pytest.fail("read a registered file before role refusal")

    monkeypatch.setattr(launcher, "digest", spy)
    with pytest.raises(launcher.LauncherRefusal, match="only the frozen A6"):
        launcher.load_policy(str(policy_path), expected_sha)
    assert seen == []


def test_large_spec_is_file_backed_with_small_environment(monkeypatch, tmp_path):
    spec = {"stage": "metadata-plan", "pythonpath": [str(tmp_path)],
            "sandbox_tmp": str(tmp_path / "tmp"),
            "files": {f"synthetic_file_{i}.py": "a" * 64 for i in range(900)}}
    path = tmp_path / "child_spec.json"
    sha = launcher._write_spec(path, spec)
    assert path.stat().st_size > 32_000
    monkeypatch.setenv("WM_N1_CANONICAL_PACK", "foreign")
    monkeypatch.setenv("WM_N1_TRAIN_OUTCOMES", "foreign")
    env = launcher.child_env(spec, path, sha)
    assert env["WM_TRAINSTAGE_POLICY_SPEC_PATH"] == str(path)
    assert len(env["WM_TRAINSTAGE_POLICY_SPEC_PATH"] +
               env["WM_TRAINSTAGE_POLICY_SHA256"]) < 512
    assert "WM_N1_CANONICAL_PACK" not in env
    assert "WM_N1_TRAIN_OUTCOMES" not in env


def test_synthetic_fixture_root_must_be_narrow_before_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "SANDBOX_ROOT", tmp_path)
    pack = tmp_path / "pack"
    pack.mkdir()
    policy = {"stage": "metadata-plan", "sandbox": str(tmp_path / "new-sandbox"),
              "stage_inputs": {"pack": str(pack),
                               "plan_out": str(tmp_path / "new-sandbox" / "plan"),
                               "synthetic_test_mode": True,
                               "synthetic_root": str(tmp_path)}}
    monkeypatch.setattr(launcher, "digest", lambda _path: pytest.fail("hashed data"))
    with pytest.raises(launcher.LauncherRefusal, match="narrow disjoint"):
        launcher.policy_inputs(policy)
