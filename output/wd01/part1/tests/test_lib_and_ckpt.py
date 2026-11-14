import json

import numpy as np
import pytest
import torch

import wd1lib as L


# ---------------- device resolution -----------------------------------------

def test_device_default_is_cpu_without_cuda():
    import os
    os.environ.pop("WD01_DEVICE", None)
    assert L.device().type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_device_env_override(monkeypatch):
    monkeypatch.setenv("WD01_DEVICE", "cpu")
    assert L.device() == torch.device("cpu")


# ---------------- tag resolution --------------------------------------------

def test_resolve_tag_prefers_retry1(tmp_path):
    retry = tmp_path / "regen_retry1"
    regen = tmp_path / "regen1"
    for root in (retry, regen):
        (root / "cv23" / "7012m2700-retain" / "run").mkdir(parents=True)
    pref, run = L.resolve_tag("cv23_7012m2700-retain", retry, regen)
    assert pref == "retry1"
    assert run.is_dir()


def test_resolve_tag_falls_back_when_retry_missing(tmp_path):
    retry = tmp_path / "regen_retry1"          # does not exist at all
    regen = tmp_path / "regen1"
    (regen / "cv17" / "native" / "run").mkdir(parents=True)
    pref, run = L.resolve_tag("cv17_native", retry, regen)
    assert pref == "regen1"
    assert run.is_dir()


def test_resolve_tag_rejects_bad_shape(tmp_path):
    assert L.resolve_tag("not_a_tag", tmp_path, tmp_path) == (None, None)


def test_saved_tag_map_consistency():
    m = L.load_json(L.RESULTS / "tag_run_map.json")
    assert m["n_tags"] == 88 and m["n_problems"] == 0
    assert m["n_retry1"] == 8
    assert all(r["physics_sha256"] and r["telemetry_sha256"]
               for r in m["rows"])
    assert all(r["manifest_sha_match"] and r["plan_dir_match"]
               and r["plan_sha_match"] and r["meta_match"]
               for r in m["rows"])
    splits = {r["split"] for r in m["rows"]}
    assert splits == {"train", "holdout"}


# ---------------- modern checkpoint format ----------------------------------

def _tiny_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(),
                               torch.nn.Linear(3, 1))


def test_modern_ckpt_roundtrip_weights_only(tmp_path):
    model = _tiny_model()
    meta = {"format": "wd01_part1_modern_v1", "in_dim": 4,
            "scale": np.float64(1.5), "path": tmp_path,
            "x_mean": np.arange(4.0)}
    p = L.save_modern_ckpt(tmp_path / "tiny.pt", model, meta)
    state, loaded_meta = L.load_modern_ckpt(p)      # weights_only=True
    ref = model.state_dict()
    assert sorted(state) == sorted(ref)
    for k in ref:
        assert torch.equal(state[k], ref[k])
    assert loaded_meta["format"] == "wd01_part1_modern_v1"
    assert loaded_meta["scale"] == pytest.approx(1.5)
    assert loaded_meta["x_mean"] == [0.0, 1.0, 2.0, 3.0]
    assert loaded_meta["path"] == str(tmp_path)


def test_surviving_checkpoint_sha256_pinned():
    assert L.SURV_CKPT.is_file()
    assert L.sha256_file(L.SURV_CKPT) == L.SURV_CKPT_SHA256
    assert L.ENSEMBLE.is_file()
    assert L.sha256_file(L.ENSEMBLE) == L.ENSEMBLE_SHA256


def test_retrained_modern_ckpt_loads_weights_only_and_matches_surviving():
    p = L.IDENTITY / "retrained" / "vehicle_dynamics_v3_retrained.pt"
    if not p.is_file():        # pragma: no cover - retrain not run yet
        pytest.skip("retrained checkpoint not built yet")
    state, meta = L.load_modern_ckpt(p)             # weights_only=True
    assert meta["format"] == "wd01_part1_modern_v1"
    ck = torch.load(L.SURV_CKPT, map_location="cpu", weights_only=False)
    assert sorted(state) == sorted(ck["model_state"])
    max_diff = max(float((state[k] - ck["model_state"][k]).abs().max())
                   for k in state)
    assert max_diff == 0.0          # bit-identical retrain
