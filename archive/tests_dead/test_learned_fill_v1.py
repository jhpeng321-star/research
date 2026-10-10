"""WM-21 learned-fill package tests (synthetic grids; no SUMO, no fit).

Contract under test (output/learned_fill_v1/REGISTRATION.md): the model
writes ONLY provenance-unknown valid cells; observed and declared-empty
cells keep the arm-B count and speed exactly; the training split refuses
every WM-18 world; the loss touches unknown valid cells only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "output" / "learned_fill_v1"
RP_V1 = Path(__file__).resolve().parents[1] / "output" / "radius_penetration_v1"
for p in (HERE, RP_V1, HERE.parents[1]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from wm20_common import P_OBSERVED, P_UNKNOWN, WM20Error  # noqa: E402
from model import FillNet, apply_fill, fill_loss  # noqa: E402
from wm21_common import (assert_train_scene, features_from_b_grid,  # noqa: E402
                         scene_runs)


def _b_grid():
    """300x4 grid: (10,0) seen, (11,0) declared empty, (12,0) unknown."""
    count = np.full((300, 4), np.nan)
    speed = np.full((300, 4), np.nan)
    provenance = np.full((300, 4), P_UNKNOWN, dtype=np.int64)
    valid = np.zeros((300, 4), dtype=bool)
    valid[10:13, 0] = True
    count[10, 0], speed[10, 0], provenance[10, 0] = 1.0, 20.0, P_OBSERVED
    count[11, 0], provenance[11, 0] = 0.0, P_OBSERVED
    return {"count": count, "speed": speed, "provenance": provenance,
            "valid": valid, "coverage": valid & (provenance != P_UNKNOWN)}


def _pred(value=3.0, speed=12.0):
    count = np.zeros((300, 4))
    spd = np.zeros((300, 4))
    count[:, :] = value
    spd[:, :] = speed
    return count, spd


def test_fill_writes_only_unknown_cells():
    grid = _b_grid()
    pred_count, pred_speed = _pred(3.0, 12.0)
    filled = apply_fill(grid, pred_count, pred_speed)
    unknown = grid["provenance"] == P_UNKNOWN
    declared = ~unknown
    # observed and declared-empty cells keep arm-B count and speed EXACTLY
    assert np.array_equal(filled["count"][declared], grid["count"][declared],
                          equal_nan=True)
    assert np.array_equal(filled["speed"][declared], grid["speed"][declared],
                          equal_nan=True)
    # unknown valid cells carry the model output (count clamped at 0)
    assert filled["count"][12, 0] == pytest.approx(3.0)
    assert filled["speed"][12, 0] == pytest.approx(12.0)
    # invalid cells stay NaN whatever the model says
    assert np.all(~np.isfinite(filled["count"][~grid["valid"]]))


def test_unknown_cell_can_change():
    grid = _b_grid()
    pred_count, pred_speed = _pred(2.0, 7.0)
    filled = apply_fill(grid, pred_count, pred_speed)
    assert not np.isfinite(grid["count"][12, 0])
    assert filled["count"][12, 0] == pytest.approx(2.0)
    assert filled["filled_mask"][12, 0] and filled["filled_mask"].sum() == 1


def test_fill_zero_count_hides_speed():
    grid = _b_grid()
    filled = apply_fill(grid, *_pred(0.0, 9.0))
    assert filled["count"][12, 0] == 0.0
    assert not np.isfinite(filled["speed"][12, 0])


def test_wm18_world_rejected_by_trainer_split():
    for world in ("7003m2000", "7008m2700", "7536m2400"):
        with pytest.raises(Exception, match="refused"):
            assert_train_scene(world)
    with pytest.raises(Exception, match="not a registered training scene"):
        assert_train_scene("cv20_7003m2000-retain")  # not a scene name
    for scene in ("cv17", "4107", "4108", "7001m2000", "7002m2000"):
        assert scene_runs(scene)          # registered scenes stay open
    assert len(scene_runs("cv17")) == 3   # arms of one scene are one scene


def test_loss_touches_unknown_cells_only():
    model = FillNet()
    features = np.zeros((5, 300, 4), dtype=np.float32)
    pred_count, pred_speed = model.predict(features)
    import torch
    pc = torch.as_tensor(pred_count[None])
    ps = torch.as_tensor(pred_speed[None])
    mask = np.zeros((1, 300, 4), dtype=bool)
    mask[0, 12, 0] = True                   # unknown
    truth_a = np.zeros((1, 300, 4))
    truth_a[0, 12, 0] = 2.0
    truth_b = truth_a.copy()
    truth_b[0, 10, 0] = 5.0                 # declared cell perturbed
    speeds = np.full((1, 300, 4), 15.0)
    total_a, count_a, _ = fill_loss(pc, ps, torch.as_tensor(truth_a),
                                    torch.as_tensor(speeds), mask)
    total_b, count_b, _ = fill_loss(pc, ps, torch.as_tensor(truth_b),
                                    torch.as_tensor(speeds), mask)
    assert float(total_a) == pytest.approx(float(total_b))
    assert float(count_a) == pytest.approx(float(count_b))
    truth_c = truth_a.copy()
    truth_c[0, 12, 0] = 4.0                 # unknown cell perturbed
    total_c, _, _ = fill_loss(pc, ps, torch.as_tensor(truth_c),
                              torch.as_tensor(speeds), mask)
    assert float(total_c) != pytest.approx(float(total_a))


def test_features_from_b_grid_channels():
    grid = _b_grid()
    features, unknown = features_from_b_grid(grid)
    assert features.shape == (5, 300, 4) and unknown[12, 0]
    assert not unknown[10, 0] and not unknown[11, 0]
    assert features[0, 10, 0] == 1.0 and features[1, 10, 0] == 20.0
    assert features[0, 12, 0] == 0.0            # unknown count never leaks
    assert features[2, 10, 0] == 1.0            # seen indicator
    assert features[3, 11, 0] == 1.0            # declared-empty indicator
    assert features[4, 12, 0] == 1.0            # unknown indicator
    # model input carries no truth/penetration/driver-type channel:
    # only the five observation-derived channels exist
    from model import INPUT_CHANNELS
    assert INPUT_CHANNELS == 5


def test_checkpoint_round_trip(tmp_path):
    model = FillNet()
    state = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    path = tmp_path / "checkpoint.npz"
    np.savez(path, **state)
    restored = FillNet()
    with np.load(path, allow_pickle=False) as npz:
        restored.load_state_dict({k: npz[k] for k in npz.files})
    features = np.random.default_rng(0).random((5, 300, 4)).astype(np.float32)
    c1, s1 = model.predict(features)
    c2, s2 = restored.predict(features)
    assert np.array_equal(c1, c2) and np.array_equal(s1, s2)


def test_checkpoint_guards(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "wm21_evaluate_cli", HERE / "evaluate.py")
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)
    spec = importlib.util.spec_from_file_location(
        "wm21_train_cli", HERE / "train.py")
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)
    missing = tmp_path / "missing.npz"
    with pytest.raises(WM20Error, match="checkpoint not found"):
        eval_mod.load_checkpoint(missing)
    outside = tmp_path / "outside.npz"
    np.savez(outside, x=np.zeros(1))
    with pytest.raises(WM20Error, match="output.learned_fill_v1"):
        eval_mod.load_checkpoint(outside)
    existing = tmp_path / "exists.npz"
    existing.write_bytes(b"x")
    with pytest.raises(WM20Error, match="refusing to overwrite"):
        train_mod.train("cv17", epochs=0, checkpoint=existing)
    with pytest.raises(WM20Error, match="under output"):
        train_mod.train("cv17", epochs=0, checkpoint=tmp_path / "elsewhere.npz")


def test_cli_help(capsys):
    import importlib.util

    def load(name_and_file):
        name, file_name = name_and_file
        spec = importlib.util.spec_from_file_location(name, HERE / file_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    train_mod = load(("wm21_train_cli", "train.py"))
    eval_mod = load(("wm21_evaluate_cli", "evaluate.py"))
    for main_fn in (train_mod.main, eval_mod.main):
        with pytest.raises(SystemExit) as excinfo:
            main_fn(["--help"])
        assert excinfo.value.code == 0
        capsys.readouterr()
