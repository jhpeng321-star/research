import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "wd01" / "wm_n1"
sys.path.insert(0, str(PKG))

from wms_eval import c1  # noqa: E402
from wms_eval.data import DataError  # noqa: E402

VEHICLES = (
    {"id": "h1", "s": 10.0, "lane": 0, "v": 18.0, "present": True, "is_cav": False},
    {"id": "h2", "s": 35.0, "lane": 0, "v": 20.0, "present": True, "is_cav": False},
    {"id": "c1", "s": 55.0, "lane": 1, "v": 22.0, "present": True, "is_cav": True},
)


def _trajectory(t0_s: float, scale: float, n: int = 141):
    frames = []
    for step in range(n):
        t = t0_s + step * 0.5
        rows = []
        for vehicle in VEHICLES:
            speed = vehicle["v"] * scale
            rows.append({"id": vehicle["id"], "s": vehicle["s"] + speed * step * 0.5,
                         "lane": vehicle["lane"], "v": speed, "present": True,
                         "is_cav": vehicle["is_cav"]})
        frames.append({"time_s": t, "vehicles": rows})
    return frames


def _state(world: str, t0_s: float, seed: int) -> dict:
    return {
        "world": world, "t0_s": t0_s, "seed": seed, "horizon_s": 40.0, "v0_mps": 27.78,
        "initial_state": {"time_s": t0_s, "vehicles": [dict(v) for v in VEHICLES]},
        "recorded": {"native": _trajectory(t0_s, 1.0), "pi": _trajectory(t0_s, 0.96)},
    }


def _six_states(prefix: str = "w"):
    return [_state(f"{prefix}{index // 2}", 10.0 * index, index) for index in range(6)]


def _checkpoint(out: Path, seed: int = 0) -> Path:
    from wms.model import PhysicsConfig, WmSModel

    torch.manual_seed(seed)
    model = WmSModel(physics=PhysicsConfig(), history_features=6)
    path = out / f"wm_s_{seed}.pt"
    torch.save({"format": "wm_s/1", "model": model.state_dict(),
                "config": {"history_features": 6}}, path)
    return path


def _fake_blackbox(monkeypatch, transform):
    module = types.ModuleType("wms.blackbox_id")

    class _Fake:
        def __init__(self, checkpoint, flags=None):
            self.checkpoint = checkpoint

        def predict_states(self, state, controllers, horizon_s, seed):
            from wms_eval.data import initial_frame

            initial = initial_frame(state)
            t0 = float(initial["time_s"])
            steps = int(round(float(horizon_s) / 0.5))
            frames = []
            for step in range(steps + 1):
                rows = []
                for row in initial["vehicles"]:
                    row = dict(row)
                    row["s"] = transform(row["s"] + row["v"] * step * 0.5)
                    row["time_s"] = t0 + step * 0.5
                    rows.append(row)
                frames.append({"time_s": t0 + step * 0.5, "vehicles": rows})
            return frames

    module.build = _Fake
    monkeypatch.setitem(sys.modules, "wms.blackbox_id", module)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def test_bootstrap_is_deterministic_and_brackets_the_point():
    grouped = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
    first = c1.bootstrap_over_worlds(grouped, n_boot=200, seed=7)
    second = c1.bootstrap_over_worlds(grouped, n_boot=200, seed=7)
    assert first == second
    assert first["point"] == 3.5
    assert first["lo"] <= first["hi"]
    assert first["n_worlds"] == 2


def test_ratio_and_paired_confidence_intervals():
    ratio = c1.ratio_ci({"a": [2.0, 2.0]}, {"b": [1.0, 1.0]}, n_boot=64, seed=0)
    assert ratio["point"] == 2.0
    assert ratio["lo"] == 2.0 and ratio["hi"] == 2.0
    paired = c1.paired_ci({"w1": [3.0], "w2": [4.0]}, {"w1": [1.0], "w2": [1.0]},
                          n_boot=64, seed=0)
    assert paired["point"] == 2.5
    assert paired["lo"] <= paired["point"] <= paired["hi"]


def test_headline_metric_grammar():
    result = {"arms": {"native": {
        "error": None,
        "car": {"30.0": {"hdv": {"ds_med": 1.5}}},
        "traffic": {"merge": {"total_delay": {"mean": 4.0}}},
    }}}
    assert c1.headline_metric(result, "hdv_ds@30", "native") == 1.5
    assert c1.headline_metric(result, "traffic_total_delay@merge", "native") == 4.0
    with pytest.raises(c1.C1Error):
        c1.headline_metric(result, "hdv_ds", "native")
    with pytest.raises(c1.C1Error):
        c1.headline_metric(result, "bogus", "native")


def test_resolve_states_fails_closed_without_index():
    with pytest.raises(DataError):
        c1.resolve_states([{"world": "w", "key": "c1/w__native", "run": "w__native"}])


def test_parse_models_rejects_unknown_spec():
    with pytest.raises(c1.C1Error):
        c1.parse_models(["nope=x"])
    assert c1.parse_models(["keep_speed=", "wms=ck.pt"]) == [("keep_speed", None), ("wms", "ck.pt")]


# --------------------------------------------------------------------------- #
# end-to-end on six states with random checkpoints
# --------------------------------------------------------------------------- #
def test_parallel_run_model_keep_speed(tmp_path):
    results = c1.run_model("keep_speed", None, _six_states(), ["native", "pi"], "native",
                           c1.CAR_HORIZONS, 30.0, 70.0, workers=2)
    assert len(results) == 6
    assert all(r["arms"]["native"]["error"] is None for r in results)
    assert all(c1.headline_metric(r, "hdv_ds@60", "native") is not None for r in results)


def test_c1_cli_end_to_end_with_random_checkpoint_and_blackbox(tmp_path, monkeypatch):
    # the WM-S checkpoint loader is workspace-confined, so it lives under _out/
    ckpt_dir = ROOT / "_out" / "wms_c1_test"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint(ckpt_dir)
    blackbox = tmp_path / "bb.pt"
    blackbox.write_bytes(b"placeholder")
    _fake_blackbox(monkeypatch, lambda s: s - 1.0)

    unseen = {"windows": _six_states("u")}
    seen = {"windows": _six_states("s")[:4]}
    unseen_path = tmp_path / "unseen.json"
    seen_path = tmp_path / "seen.json"
    unseen_path.write_text(json.dumps(unseen), encoding="utf-8")
    seen_path.write_text(json.dumps(seen), encoding="utf-8")
    out = tmp_path / "c1_out"

    code = c1.main([
        "--models", "keep_speed=", "--models", f"wms={checkpoint}",
        "--models", f"blackbox_id={blackbox}",
        "--unseen", str(unseen_path), "--seen", str(seen_path), "--out", str(out),
        "--horizon", "40", "--metric", "hdv_ds@30", "--bootstrap", "64",
        "--workers", "1", "--paired-a", "wms", "--paired-b", "blackbox_id",
    ])
    assert code == 0
    report = json.loads((out / "C1.json").read_text(encoding="utf-8"))
    assert set(report["models"]) == {"keep_speed", "wms", "blackbox_id"}
    assert report["sets"]["unseen"]["n_states"] == 6
    assert report["sets"]["seen"]["n_states"] == 4
    assert report["sets"]["unseen"]["n_worlds"] == 3
    assert report["sets"]["seen"]["n_worlds"] == 2
    assert report["separation"]["world_overlap"] == []
    assert report["separation"]["refused"] is False
    assert "wms" in report["unseen_seen_ratio"]
    paired = report["paired_structured_vs_blackbox_unseen"]
    assert paired is not None and paired["a"] == "wms" and paired["b"] == "blackbox_id"
    assert (out / "C1.md").read_text(encoding="utf-8").startswith("# C1")
    assert (out / "RESULTS.json").exists()


def test_c1_cli_refuses_to_overwrite(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    unseen_path = tmp_path / "unseen.json"
    unseen_path.write_text(json.dumps({"windows": _six_states()}), encoding="utf-8")
    code = c1.main(["--models", "keep_speed=", "--unseen", str(unseen_path),
                    "--out", str(out), "--workers", "1"])
    assert code == 2


# --------------------------------------------------------------------------- #
# seen/unseen and training separation
# --------------------------------------------------------------------------- #
def test_check_separation_reports_counts_and_refuses_overlap():
    disjoint = c1.check_separation({"unseen": _six_states("u"), "seen": _six_states("s")[:4]})
    assert disjoint["refused"] is False
    assert disjoint["world_overlap"] == []
    assert disjoint["unseen_worlds"] == 3 and disjoint["seen_worlds"] == 2
    with pytest.raises(c1.C1Error) as excinfo:
        c1.check_separation({"unseen": _six_states("w"), "seen": _six_states("w")[:4]})
    assert "world overlap=2" in str(excinfo.value)


def test_check_separation_refuses_unseen_training_key_or_run_leak():
    train = [{"key": "chunk1/w0__native", "run": "w0__native", "world": "trn"}]
    leaked = _state("u0", 0.0, 0)
    leaked.update({"key": "chunk1/w0__native", "run": "w0__native"})
    clean = _state("u0", 0.0, 0)
    clean.update({"key": "chunk9/w0__pi", "run": "w0__pi"})
    with pytest.raises(c1.C1Error) as excinfo:
        c1.check_separation({"unseen": [leaked]}, train)
    message = str(excinfo.value)
    assert "key overlap=1" in message and "run overlap=1" in message
    report = c1.check_separation({"unseen": [clean]}, train)
    assert report["key_overlap"] == [] and report["run_overlap"] == []
    assert report["train_windows"] == 1


def test_c1_cli_refuses_overlapping_worlds(tmp_path):
    overlapping = {"windows": _six_states("w")}
    unseen_path = tmp_path / "unseen.json"
    seen_path = tmp_path / "seen.json"
    unseen_path.write_text(json.dumps(overlapping), encoding="utf-8")
    seen_path.write_text(json.dumps({"windows": overlapping["windows"][2:]}), encoding="utf-8")
    out = tmp_path / "c1_out"
    code = c1.main([
        "--models", "keep_speed=", "--unseen", str(unseen_path), "--seen", str(seen_path),
        "--out", str(out), "--horizon", "40", "--workers", "1",
    ])
    assert code == 2
    assert not out.exists()


def test_c1_cli_refuses_unseen_state_in_training_windows(tmp_path):
    world = "u0"
    unseen_state = _state(world, 0.0, 0)
    unseen_state.update({"key": "chunk2/u0__native", "run": "u0__native"})
    unseen_path = tmp_path / "unseen.json"
    train_path = tmp_path / "train.json"
    unseen_path.write_text(json.dumps({"windows": [unseen_state]}), encoding="utf-8")
    train_path.write_text(json.dumps({"windows": [
        {"key": "chunk2/u0__native", "run": "u0__native", "world": world}]}), encoding="utf-8")
    out = tmp_path / "c1_out"
    code = c1.main([
        "--models", "keep_speed=", "--unseen", str(unseen_path),
        "--train-windows", str(train_path), "--out", str(out),
        "--horizon", "40", "--workers", "1",
    ])
    assert code == 2
    assert not out.exists()
