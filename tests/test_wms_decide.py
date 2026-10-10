import hashlib
import json

import numpy as np
import pytest

import wms_decide


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict_states(self, state, plan, horizon_s, seed):
        self.calls.append((plan, horizon_s, seed))
        times = np.arange(0.0, horizon_s + 0.5, 0.5)
        free = np.full((len(times), 1), 10.0)
        speed = np.full_like(free, 10.0 - float(plan))
        return {"time_s": times, "speed_mps": speed, "free_speed_mps": free}


def _truth(deficit):
    times = np.arange(0.0, 30.5, 0.5)
    free = np.full((len(times), 1), 10.0)
    speed = np.full_like(free, 10.0 - deficit)
    return {"time_s": times, "speed_mps": speed, "free_speed_mps": free}


def test_delay_score_uses_normalized_delay_seconds():
    # 2 m/s deficit at 10 m/s free flow for 30 s = (1 - 8/10) * 30 s = 6 s,
    # matching the registered WM-N1 window-delay definition.
    assert wms_decide.total_delay(_truth(2.0), t0_s=0.0, tau_s=30.0) == pytest.approx(6.0)


def test_total_delay_refuses_a_nonpositive_free_speed():
    states = _truth(2.0)
    states["free_speed_mps"] = np.full_like(states["free_speed_mps"], 0.0)
    with pytest.raises(wms_decide.DecisionRefusal, match="free-flow"):
        wms_decide.total_delay(states, t0_s=0.0, tau_s=30.0)


def test_score_rows_uses_same_seed_and_reports_pi_native_difference():
    model = FakeModel()
    rows = [{
        "world": "train-world",
        "t0_s": 0.0,
        "taus_s": [30.0],
        "state": {"id": 1},
        "candidates": {"native": 0.0, "pi": 1.0, "arm0": 0.5},
        "true": {"native": _truth(0.0), "pi": _truth(1.0), "arm0": _truth(0.5)},
    }]

    result = wms_decide.score_rows(model, rows, seed=17)
    assert [call[2] for call in model.calls] == [17, 17, 17]
    item = result["rows"][0]
    assert item["argmin"] == "native"
    assert item["paired_difference_vs_native_s"]["pi"] == pytest.approx(3.0)
    assert item["pi_native_true_difference_s"] == pytest.approx(3.0)
    assert item["argmin_regret_s"] == pytest.approx(0.0)


def test_decision_ties_default_to_native():
    model = FakeModel()
    rows = [{
        "world": "train-world",
        "t0_s": 0.0,
        "taus_s": [30.0],
        "state": {"id": 1},
        "candidates": {"native": 1.0, "arm0": 1.0, "arm1": 1.0},
    }]
    result = wms_decide.score_rows(model, rows, seed=17)
    item = result["rows"][0]
    assert item["predicted_delay_s"]["native"] == item["predicted_delay_s"]["arm0"]
    assert item["argmin"] == "native"


def _write_source(tmp_path, content: bytes = b"train-manifest", name: str = "run_base"):
    base = tmp_path / name
    (base / "saved_state").mkdir(parents=True)
    (base / "saved_state" / "STATE_MANIFEST.json").write_bytes(content)
    return base


def _provenanced_row(base, *, sha: str | None = None, split: str = "TRAIN"):
    row = {"split": split,
           "state": {"base_run_dir": str(base)},
           "source_sha256": sha or hashlib.sha256(
               (base / "saved_state" / "STATE_MANIFEST.json").read_bytes()).hexdigest()}
    return row


def test_release_loader_refuses_non_train(tmp_path):
    path = tmp_path / "release.json"
    path.write_text(json.dumps({"split": "CAL", "rows": [{}]}), encoding="utf-8")
    with pytest.raises(wms_decide.DecisionRefusal, match="TRAIN"):
        wms_decide._load_release(str(path))


def test_release_loader_rejects_a_nontrain_row_inside_a_train_document(tmp_path):
    good = _provenanced_row(_write_source(tmp_path, name="good"))
    bad_rows = [
        {"split": "CAL", "source_sha256": "b" * 64},
        {"split": "HOLDOUT", "source_sha256": "c" * 64},
        {"split": "test", "source_sha256": "d" * 64},
        {"split": "TRAIN"},
        {"source_sha256": "e" * 64},
    ]
    for index, bad in enumerate(bad_rows):
        path = tmp_path / f"release_{index}.json"
        path.write_text(
            json.dumps({"split": "TRAIN", "rows": [dict(good), bad]}),
            encoding="utf-8")
        with pytest.raises(wms_decide.DecisionRefusal, match="row 1"):
            wms_decide._load_release(str(path))


def test_release_loader_accepts_fully_provenanced_train_rows(tmp_path):
    first = _write_source(tmp_path, b"manifest-one", name="run_a")
    second = _write_source(tmp_path, b"manifest-two", name="run_b")
    uppercase_sha = hashlib.sha256(b"manifest-two").hexdigest().upper()
    path = tmp_path / "release.json"
    path.write_text(
        json.dumps({"split": "TRAIN", "rows": [
            _provenanced_row(first),
            _provenanced_row(second, sha=uppercase_sha),
        ]}),
        encoding="utf-8")
    assert len(wms_decide._load_release(str(path))) == 2


def test_release_loader_rejects_wellformed_but_wrong_source_hash(tmp_path):
    base = _write_source(tmp_path, b"real-manifest")
    path = tmp_path / "release.json"
    path.write_text(
        json.dumps({"split": "TRAIN",
                    "rows": [_provenanced_row(base, sha="e" * 64)]}),
        encoding="utf-8")
    with pytest.raises(wms_decide.DecisionRefusal,
                       match="row 0 .*does not match.*source"):
        wms_decide._load_release(str(path))


def test_release_loader_fails_closed_on_missing_source_artifact(tmp_path):
    base = tmp_path / "absent_run"
    path = tmp_path / "release.json"
    path.write_text(
        json.dumps({"split": "TRAIN",
                    "rows": [_provenanced_row(base, sha="a" * 64)]}),
        encoding="utf-8")
    with pytest.raises(wms_decide.DecisionRefusal, match="missing source artifact"):
        wms_decide._load_release(str(path))


def test_release_loader_refuses_rows_without_a_source_reference(tmp_path):
    path = tmp_path / "release.json"
    path.write_text(
        json.dumps({"split": "TRAIN",
                    "rows": [{"split": "TRAIN", "source_sha256": "a" * 64}]}),
        encoding="utf-8")
    with pytest.raises(wms_decide.DecisionRefusal, match="does not reference"):
        wms_decide._load_release(str(path))


def _frames(values):
    """values: list of per-step {id: (v, present)}."""
    out = []
    for step, rows in enumerate(values):
        out.append({"time_s": float(step), "vehicles": [
            {"id": vid, "v": v, "present": present, "s": 100.0, "lane": 1,
             "is_cav": True}
            for vid, (v, present) in rows.items()]})
    return out


def test_frames_to_arrays_tracks_presence_and_free_speed():
    frames = _frames([{"a": (10.0, True), "b": (10.0, True)},
                      {"a": (8.0, True), "b": (10.0, False)},
                      {"a": (8.0, True), "b": (10.0, False)}])
    arrays = wms_decide.frames_to_arrays(frames, free_speed_mps=10.0)
    assert arrays["speed_mps"].shape == (3, 2)
    assert bool(arrays["mask"][1, 1]) is False
    assert bool(arrays["mask"][1, 0]) is True
    assert wms_decide.total_delay(arrays, t0_s=0.0, tau_s=2.0) == pytest.approx(0.2)


def test_frames_to_arrays_refuses_empty_vehicle_stream():
    with pytest.raises(wms_decide.DecisionRefusal, match="no vehicles"):
        wms_decide.frames_to_arrays([{"time_s": 0.0, "vehicles": []},
                                     {"time_s": 0.5, "vehicles": []}], free_speed_mps=10.0)


def test_wms_eval_predictor_consumes_plugin_frames_per_arm():
    calls = []

    def predictor(state, controllers, horizon_s, seed):
        calls.append((dict(controllers), horizon_s, seed))
        return _frames([{"a": (10.0, True)}, {"a": (9.0, True)}])

    model = wms_decide.WmsEvalPredictor(predictor)
    state = {"v0_mps": 10.0, "cav_controllers": {"arm": "native"}}
    result = model.predict_states(state, "pi", 1.0, 5)
    assert calls == [({"arm": "pi"}, 1.0, 5)]
    assert result["speed_mps"].shape == (2, 1)


def test_recorded_delay_streams_normalized_seconds(tmp_path):
    lines = [
        {"time_s": 10.0, "live_ids": ["a"], "vehicles": {"a": {"speed_mps": 8.0}}},
        {"time_s": 10.5, "live_ids": ["a"], "vehicles": {"a": {"speed_mps": 8.0}}},
        {"time_s": 11.0, "live_ids": ["a"], "vehicles": {"a": {"speed_mps": 8.0}}},
    ]
    path = tmp_path / "physics_steps.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in lines) + "\n", encoding="utf-8")
    delays = wms_decide.recorded_delay_seconds(
        path, t0_s=10.0, taus_s=(1.0, 2.0), free_speed_mps=10.0)
    assert delays["1.0"] == pytest.approx(0.2)
    assert delays["2.0"] == pytest.approx(0.3)


def test_decision_summary_reports_sign_rank_and_native_regret():
    rows = [
        {"tau_s": 30.0, "argmin": "pi",
         "paired_difference_vs_native_s": {"native": 0.0, "pi": -2.0},
         "pi_native_pred_difference_s": -2.0, "pi_native_true_difference_s": -3.0,
         "true_delay_s": {"native": 10.0, "pi": 7.0}},
        {"tau_s": 30.0, "argmin": "native",
         "paired_difference_vs_native_s": {"native": 0.0, "pi": 1.0},
         "pi_native_pred_difference_s": 1.0, "pi_native_true_difference_s": 2.0,
         "true_delay_s": {"native": 4.0, "pi": 6.0}},
    ]
    summary = wms_decide.decision_summary(rows)["30.0"]
    assert summary["sign_agreement"] == pytest.approx(1.0)
    assert summary["rank_correlation"] == pytest.approx(1.0)
    assert summary["regret_s"] == pytest.approx(0.0)
    assert summary["always_native_regret_s"] == pytest.approx(3.0)
    assert summary["regret_improvement_vs_native_s"] == pytest.approx(3.0)
    assert summary["best_arm_accuracy"] == pytest.approx(1.0)
