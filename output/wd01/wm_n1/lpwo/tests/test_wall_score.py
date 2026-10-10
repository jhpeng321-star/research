"""Synthetic-only wall scoring tests: no model loading, training or SUMO."""
import copy
import io
import json
import sys
import types
from pathlib import Path

import pytest

WM_N1 = Path(__file__).resolve().parents[2]
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))

from lpwo import score_wall as S  # noqa: E402


def trajectory(speed=5.0, offset=0.0, t0=100.0):
    # Stay upstream, so section migration cannot change the synthetic delay label.
    return [{"time_s": t0 + i * 0.5, "vehicles": [
        {"id": "h", "s": -10000 + i * 0.5 * speed + (offset if i else 0),
         "v": speed if i else 5.0, "lane": 0, "is_cav": False},
        {"id": "c", "s": -5000 + i * 2.5, "v": 5.0, "lane": 1, "is_cav": True}]}
        for i in range(241)]


def record():
    native, wall = trajectory(), trajectory(8.0)
    return {"world": "synthetic-world", "key": "synthetic/run", "t0_s": 100.0,
            "horizon_s": 120.0, "dt_s": 0.5, "v0_mps": 10.0,
            "state": native[0], "recorded": {"native": native, "wall": wall}}


def predictor(state, controllers, _horizon, _seed):
    return copy.deepcopy(state["recorded"][controllers["arm"]])


class FakeGuard:
    def __init__(self):
        self.paths, self.records, self.reads = [], [], []
        self.audits = 0

    def check_path(self, path):
        self.paths.append(Path(path))

    def check_record(self, record):
        self.records.append(record["key"])

    def read_json(self, path):
        assert Path(path) in self.paths
        self.reads.append(Path(path))
        return {"dt_s": 0.5}

    def audit(self):
        self.audits += 1
        return {"n_paths": len(self.paths), "n_records": len(self.records)}


def guard_args():
    return ["--guard-spec", "synthetic/spec.json", "--guard-tool", "synthetic/tool.py",
            "--allowed-root", "synthetic"]


def test_exact_replay_hdv_horizons_and_wall_minus_native():
    r = S.evaluate_state(record(), predictor)
    for arm in S.ARMS:
        for h in S.HORIZONS:
            assert r["arms"][arm]["car"][str(h)]["hdv"]["ds_med"] == 0
    for pair in r["paired"].values():
        assert pair["recorded_delta"] < 0
        assert pair["predicted_delta"] == pair["recorded_delta"]
        assert pair["sign_agreement"] is True
        assert pair["selected_arm"] == "wall"
        assert pair["regret_s"] == 0
    # Independently derived delay: 3 m/s gain / 10 free speed, 239 half-second
    # frames after the common t0 (the 120 s endpoint is excluded).
    assert r["paired"]["120.0"]["recorded_delta"] == pytest.approx(-35.85)


def test_hdv_error_excludes_cavs():
    def shifted(state, controllers, horizon, seed):
        frames = predictor(state, controllers, horizon, seed)
        for f in frames[1:]:
            f["vehicles"][0]["s"] += 7
            f["vehicles"][1]["s"] += 1000
        return frames
    r = S.evaluate_state(record(), shifted)
    for h in S.HORIZONS:
        assert r["arms"]["wall"]["car"][str(h)]["hdv"]["ds_med"] == 7
        assert r["arms"]["wall"]["car"][str(h)]["cav"]["ds_med"] == 1000


@pytest.mark.parametrize("pred,rec,arm,agree,regret", [
    (-2, -3, "wall", True, 0), (2, -3, "native", False, 3),
    (-2, 3, "wall", False, 3), (0, -3, "native", False, 3),
    (0, 0, "native", True, 0), (2, 3, "native", True, 0)])
def test_rank_sign_regret_and_native_ties(pred, rec, arm, agree, regret):
    pair = S.rank_pair(pred, rec)
    assert (pair["selected_arm"], pair["sign_agreement"], pair["regret_s"]) == (arm, agree, regret)


def test_no_unsupported_wall_fallback():
    calls = []
    def unavailable(state, controllers, horizon, seed):
        calls.append(controllers["arm"])
        if controllers["arm"] == "wall":
            raise ValueError("unsupported wall")
        return predictor(state, controllers, horizon, seed)
    r = S.evaluate_state(record(), unavailable)
    assert calls == ["native", "wall"]
    assert "unsupported wall" in r["arms"]["wall"]["error"]
    assert r["paired"] == {}


@pytest.mark.parametrize("change,match", [
    (lambda r: r.update(recorded={}), "at least one inline"),
    (lambda r: r.update(recorded_paths={"wall": "outside"}), "inline materialised"),
    (lambda r: r.update(base_run_dir="outside"), "inline materialised"),
    (lambda r: r.update(dt_s=1), "dt_s=0.5"),
    (lambda r: r.update(horizon_s=60), "shorter"),
    (lambda r: r["recorded"]["wall"].pop(), "expected 241"),
    (lambda r: r["recorded"]["wall"][0]["vehicles"][0].update(s=1), "identical initial"),
    (lambda r: r["recorded"]["wall"][2].update(time_s=999), "expected 241"),
])
def test_invalid_records_fail_before_predictor(change, match):
    r = copy.deepcopy(record())
    change(r)
    def forbidden(*args):
        pytest.fail("invalid input reached predictor")
    with pytest.raises(S.WallScoreError, match=match):
        S.evaluate_state(r, forbidden)


def test_truncated_prediction_not_nearest_extrapolated():
    r = S.evaluate_state(record(), lambda *args: trajectory()[:21])
    assert all("expected 241" in r["arms"][a]["error"] for a in S.ARMS)
    assert r["paired"] == {}


def test_nonfinite_prediction_is_an_arm_error():
    def bad(state, controllers, horizon, seed):
        frames = predictor(state, controllers, horizon, seed)
        frames[-1]["vehicles"][0]["s"] = float("nan")
        return frames
    r = S.evaluate_state(record(), bad)
    assert "nonfinite s" in r["arms"]["wall"]["error"]


def test_streaming_c1_json_does_not_read_whole_file(monkeypatch):
    raw = json.dumps({"windows": [record(), record()]}).encode()
    class BoundedRead(io.BytesIO):
        def read(self, size=-1):
            assert 0 <= size <= 1 << 20
            return super().read(size)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: BoundedRead(raw))
    rows = list(S.iter_windows("synthetic.json", limit=1))
    assert len(rows) == 1
    assert rows[0]["world"] == "synthetic-world"


def test_common_state_denominators_bootstrap_and_regret():
    good = S.evaluate_state(record(), predictor)
    wrong = copy.deepcopy(good)
    missing = copy.deepcopy(good)
    missing["world"] = "missing-world"
    missing["arms"]["wall"] = {"error": "unsupported"}
    missing["paired"] = {}
    for p in wrong["paired"].values():
        p.update(S.rank_pair(-p["recorded_delta"], p["recorded_delta"]))
    report = S.summarize_results({"wms": [good, missing], "blackbox_traj": [wrong, missing],
                                  "blackbox_id": [wrong, missing]}, n_boot=20)
    for h in S.HORIZONS:
        hk = str(h)
        rank = report["models"]["wms"]["ranking_common_states"][hk]
        assert rank["n"] == 1 and rank["n_excluded"] == 1
        assert rank["sign_agreement"] == 1
        for bb in ("blackbox_traj", "blackbox_id"):
            ci = report["structured_minus_blackboxes"][bb][hk]["regret_s"]
            assert ci["point"] < 0 and ci["hi"] < 0
            assert ci["n_worlds"] == 1 and ci["n_pairs"] == 1
    assert report["models"]["wms"]["blowup"]["wall"]["n_excluded"] == 1


def test_duplicate_pairs_and_inconsistent_labels_refused():
    good = S.evaluate_state(record(), predictor)
    with pytest.raises(S.WallScoreError, match="duplicate"):
        S.summarize_results({"wms": [good, good]})
    bad = copy.deepcopy(good)
    bad["paired"]["10.0"]["recorded_delta"] += 1
    with pytest.raises(S.WallScoreError, match="labels differ"):
        S.summarize_results({"wms": [good], "blackbox_id": [bad]})


def test_wall_exec_cli_refuses_before_paths_or_models(capsys):
    assert S.main(["--mode", "wall_exec", "--models", "invalid", "--sets", "invalid",
                   "--out", "invalid", *guard_args()]) == 2
    assert "wall_exec unavailable" in capsys.readouterr().out


def test_outside_path_rejected_without_read():
    with pytest.raises(S.WallScoreError, match="outside workspace"):
        S.local_path(S.WORKSPACE.parent / "outside.json")


def test_missing_physics_refused_before_loading(monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda *args: False)
    with pytest.raises(S.WallScoreError, match="missing explicit checkpoint physics"):
        S.physics_source(S.WORKSPACE / "synthetic" / "wm_s.pt", FakeGuard())


@pytest.mark.parametrize("shape", ["paired", "wall_only", "seen_controls"])
def test_cli_streamed_output_and_report_with_synthetic_predictors(monkeypatch, capsys, shape):
    # In-memory filesystem; the real CLI orchestration never loads checkpoints.
    written, calls = {}, []
    guard = FakeGuard()
    def factory(spec, tool, allowed_roots):
        assert spec == Path.cwd() / "synthetic" / "spec.json"
        assert tool == Path.cwd() / "synthetic" / "tool.py"
        assert allowed_roots == [Path.cwd() / "synthetic"]
        return guard
    monkeypatch.setitem(sys.modules, "lpwo.guard", types.SimpleNamespace(DataGuard=factory))
    class StoredText(io.StringIO):
        def __init__(self, key):
            super().__init__()
            self.key = key
        def close(self):
            written[self.key] = self.getvalue()
            super().close()
    monkeypatch.setattr(Path, "is_file", lambda p: p in guard.paths)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(Path, "open", lambda p, *args, **kwargs: StoredText(p.name))
    monkeypatch.setattr(Path, "write_text", lambda p, data, **kwargs: written.update({p.name: data}))
    def streamed(path, limit=None):
        assert path in guard.paths
        row = record()
        if shape == "wall_only":
            row["recorded"].pop("native")
            row.update(arm="wall", source_arm="wall_arm", paired=False)
        elif shape == "seen_controls":
            row["recorded"] = {"arm0": row["recorded"]["native"]}
            row.update(arm="arm0", source_set="seen_controls", paired=False)
        yield row
    monkeypatch.setattr(S, "iter_windows", streamed)
    def loaded(spec, checkpoint, physics):
        assert checkpoint in guard.paths
        calls.append(spec)
        assert physics == {"dt_s": 0.5}
        def guarded_predictor(state, controllers, horizon, seed):
            assert state["key"] in guard.records
            return predictor(state, controllers, horizon, seed)
        return guarded_predictor
    monkeypatch.setattr(S, "checkpoint_predictor", loaded)
    rc = S.main(["--models", "wms=synthetic/wms/wm_s.pt", "--models",
                 "blackbox_traj=synthetic/traj/wm_s.pt", "--models",
                 "blackbox_id=synthetic/id/wm_s.pt", "--sets", "wall_c1=synthetic/states.json",
                 "--bootstrap", "0", "--out", "synthetic/new_score", *guard_args()])
    assert rc == 0
    assert calls == list(S.MODELS)
    report = json.loads(written["WALL_SCORE.json"])
    assert report["status"] == "scored"
    assert report["cav_contract"]["blackbox_id"].startswith("learned")
    assert report["flag_identity"] == "pending outside receipt confirmation"
    assert report["sets"]["wall_c1"]["models"]["wms"]["ranking_common_states"]["120.0"]["n"] == (shape == "paired")
    for model in S.MODELS:
        payload = json.loads(written[f"states__{model}__wall_c1.json"])
        assert payload["model"] == model and len(payload["states"]) == 1
        assert "wall" in payload["states"][0]["arms"]
    assert '"n_arm_errors": 0' in capsys.readouterr().out
    assert guard.audits == 1 and len(guard.reads) == 3
    assert "GUARD_AUDIT.json" in written


def test_wall_only_scores_hdv_without_predicting_missing_native():
    row = record()
    row["recorded"].pop("native")
    row.update(arm="wall", source_arm="wall_arm", paired=False)
    calls = []
    def wall_only(state, controllers, horizon, seed):
        calls.append(controllers["arm"])
        return predictor(state, controllers, horizon, seed)
    result = S.evaluate_state(row, wall_only)
    assert calls == ["wall"]
    assert result["arms"]["native"] == {"error": None, "unavailable": "missing_recorded_arm"}
    assert not result["paired"]
    report = S.summarize_results({m: [result] for m in S.MODELS}, n_boot=0)
    summary = report["models"]["wms"]
    assert summary["n_arm_errors"]["native"] == 0
    assert summary["n_arm_unavailable"]["native"] == 1
    assert summary["hdv_ds"]["wall"]["120.0"]["n"] == 1
    assert summary["ranking_common_states"]["120.0"]["n"] == 0
    assert report["structured_minus_blackboxes"]["blackbox_traj"]["120.0"]["hdv_ds_wall_m"]["n_pairs"] == 1


@pytest.mark.parametrize("arm", ["native", "base", "arm0"])
def test_seen_controls_use_exact_own_arm_no_alias_or_fallback(arm):
    row = record()
    row["recorded"] = {arm: row["recorded"]["native"]}
    row.update(arm=arm, source_set="seen_controls", paired=False)
    calls = []
    def own(state, controllers, horizon, seed):
        calls.append(controllers["arm"])
        return predictor(state, controllers, horizon, seed)
    result = S.evaluate_state(row, own)
    assert calls == [arm]
    assert result["source_arm"] == arm and result["source_set"] == "seen_controls"
    assert result["arms"][arm]["car"]["60.0"]["hdv"]["ds_med"] == 0
    assert result["paired"] == {}
    assert S.summarize_results({"wms": [result]})["models"]["wms"]["hdv_ds"][arm]["60.0"]["n"] == 1


def test_nonidentical_native_sibling_is_unavailable_not_a_pair():
    row = record()
    row["state"] = copy.deepcopy(row["state"])
    row["recorded"]["native"][0]["vehicles"][0]["s"] += 1
    result = S.evaluate_state(row, predictor)
    assert result["arms"]["native"]["unavailable"] == "initial_state_mismatch"
    assert not result["arms"]["wall"]["error"] and not result["paired"]


def test_declared_nonpair_not_promoted_from_compact_snapshot_match():
    row = record()
    row.update(paired=False, native_pair_status="physical_state_mismatch")
    result = S.evaluate_state(row, predictor)
    assert all(not result["arms"][a]["error"] for a in S.ARMS)
    assert not result["paired"] and result["pair_status"] == "physical_state_mismatch"


@pytest.mark.parametrize("field", ["key", "source_arm"])
def test_shared_four_part_identity_and_bootstrap_do_not_collapse_same_t0(field):
    one = S.evaluate_state(record(), predictor)
    two = copy.deepcopy(one)
    two[field] = "different-source"
    report = S.summarize_results({m: [one, two] for m in S.MODELS}, n_boot=10)
    assert report["n_union_states"] == report["n_shared_states"] == 2
    ci = report["structured_minus_blackboxes"]["blackbox_traj"]["120.0"]
    assert ci["regret_s"]["n_pairs"] == ci["hdv_ds_wall_m"]["n_pairs"] == 2
    mismatched = S.summarize_results({"wms": [one], "blackbox_traj": [two]}, n_boot=0)
    assert mismatched["n_shared_states"] == 0


@pytest.mark.parametrize("deny", ["checkpoint", "physics", "set"])
def test_guard_preflight_denial_precedes_all_data_opens(monkeypatch, deny):
    args = types.SimpleNamespace(mode="recorded_cav", models=[f"{m}=synthetic/{m}/wm_s.pt" for m in S.MODELS],
                                 sets=["wall=synthetic/states.json"], limit=None, bootstrap=0,
                                 out="synthetic/new_score")
    guard = FakeGuard()
    def check(path):
        if (deny == "checkpoint" and path.suffix == ".pt") or \
           (deny == "physics" and path.name == "physics.json") or \
           (deny == "set" and path.name == "states.json"):
            return False
    guard.check_path = check
    def forbidden(*args, **kwargs):
        pytest.fail("data accessed before complete guard preflight")
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(S, "checkpoint_predictor", forbidden)
    with pytest.raises(S.WallScoreError, match="guard denied path"):
        S.run(args, guard=guard)


def test_record_guard_denial_precedes_model_and_cannot_be_swallowed():
    guard = FakeGuard()
    guard.check_record = lambda record: False
    def forbidden(*args):
        pytest.fail("denied record reached a predictor")
    with pytest.raises(S.WallScoreError, match="guard denied record"):
        S.evaluate_state(record(), forbidden, guard=guard)


def test_required_guard_flags_and_missing_production_guard_fail_closed(monkeypatch, capsys):
    args = ["--models", "wms=synthetic/wm_s.pt", "--sets", "wall=synthetic/states.json",
            "--out", "synthetic/new_score"]
    with pytest.raises(SystemExit) as exc:
        S.main(args)
    assert exc.value.code == 2
    monkeypatch.setitem(sys.modules, "lpwo.guard", None)
    assert S.main([*args, *guard_args()]) == 2
    assert "ModuleNotFoundError" in capsys.readouterr().out


def test_main_audits_refusal_and_does_not_write_existing_output(monkeypatch, capsys):
    guard = FakeGuard()
    monkeypatch.setitem(sys.modules, "lpwo.guard", types.SimpleNamespace(DataGuard=lambda *args: guard))
    def refused(*args, **kwargs):
        raise S.WallScoreError("existing output")
    monkeypatch.setattr(S, "run", refused)
    monkeypatch.setattr(Path, "write_text", lambda *args, **kwargs: pytest.fail("overwrote old output"))
    assert S.main(["--models", "invalid", "--sets", "invalid", "--out", "synthetic/existing",
                   *guard_args()]) == 2
    assert guard.audits == 1 and "guard_audit" in capsys.readouterr().out


@pytest.mark.parametrize("model,kind", [("wms", "structured"),
                                       ("blackbox_traj", "blackbox_traj"),
                                       ("blackbox_id", "blackbox_id")])
def test_checkpoint_adapter_receives_explicit_physics(monkeypatch, model, kind):
    from wms_eval import wms_plugin
    calls = []
    class FakeAdapter:
        def __init__(self, checkpoint, flags, physics, model_kind):
            calls.append((checkpoint, physics, model_kind, flags.open_entries))
        predict_states = staticmethod(predictor)
    monkeypatch.setattr(wms_plugin, "WmsCheckpointModel", FakeAdapter)
    checkpoint = S.WORKSPACE / "synthetic" / "wm_s.pt"
    physics = {"dt_s": 0.5, "sumo_fidelity": True}
    assert S.checkpoint_predictor(model, checkpoint, physics) is predictor
    assert calls == [(checkpoint, physics, kind, False)]
