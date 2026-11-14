"""P11 gate-logic tests: PASS and FAIL paths with fakes, plus piece-missing."""
from __future__ import annotations

import pytest

from fakes import (FakeFn, FakeResolver, clean_torch_tree, dirty_torch_tree,
                   settings)
from preflight.gates import (g1_zero_residual as g1, g2_ctrl_parity as g2,
                             g3_label_leak as g3, g4_overfit as g4,
                             g5_e2e_smoke as g5, g6_heldout_dryrun as g6)


# ---------------------------------------------------------------------- G1 ---
def _g1_pass():
    return {"ok": True, "integrator": "v2g", "windows": ["w0", "w1"],
            "max_position_error_m": 1e-9, "max_speed_error_mps": 1e-9}


def test_g1_pass(tmp_path):
    res = g1.run(settings(tmp_path), FakeResolver(
        {g1.ENTRY_POINTS[0]: FakeFn(_g1_pass())}))
    assert res.passed and res.checks["n_windows"] == 2


def test_g1_fail_divergence(tmp_path):
    bad = dict(_g1_pass(), max_position_error_m=0.5)
    res = g1.run(settings(tmp_path), FakeResolver(
        {g1.ENTRY_POINTS[0]: FakeFn(bad)}))
    assert not res.passed


def test_g1_fail_unexplained_tolerance(tmp_path):
    bad = dict(_g1_pass(), tolerance=1e-3)
    res = g1.run(settings(tmp_path), FakeResolver(
        {g1.ENTRY_POINTS[0]: FakeFn(bad)}))
    assert not res.passed
    bad["tolerance_explained"] = "v2g uses ballistic integration"
    res = g1.run(settings(tmp_path), FakeResolver(
        {g1.ENTRY_POINTS[0]: FakeFn(bad)}))
    assert res.passed


def test_g1_piece_missing(tmp_path):
    res = g1.run(settings(tmp_path), FakeResolver())
    assert not res.passed and res.error == "piece-missing"


# ---------------------------------------------------------------------- G2 ---
def _strategies(err=1e-5, n=2):
    return {s: {"max_error": err, "n_windows": n}
            for s in g2.REQUIRED_STRATEGIES + g2.NEW_RULE_STRATEGIES}


def test_g2_pass(tmp_path):
    res = g2.run(settings(tmp_path), FakeResolver(
        {g2.ENTRY_POINTS[0]: FakeFn({"ok": True, "tolerance": 1e-3,
                                     "strategies": _strategies()})}))
    assert res.passed and res.checks["new_rule"] is True


def test_g2_fail_missing_wall_v22(tmp_path):
    s = _strategies()
    s.pop("wall-v2.2")
    res = g2.run(settings(tmp_path), FakeResolver(
        {g2.ENTRY_POINTS[0]: FakeFn({"ok": True, "strategies": s})}))
    assert not res.passed


def test_g2_fail_too_few_windows(tmp_path):
    res = g2.run(settings(tmp_path), FakeResolver(
        {g2.ENTRY_POINTS[0]: FakeFn({"ok": True,
                                     "strategies": _strategies(n=1)})}))
    assert not res.passed


def test_g2_piece_missing(tmp_path):
    res = g2.run(settings(tmp_path), FakeResolver())
    assert not res.passed and res.error == "piece-missing"


# ---------------------------------------------------------------------- G3 ---
def _kin_fields():
    return ["s_down_m", "lane_main_left", "v_mps", "a_mps2", "length_m"]


def test_g3_pass(tmp_path):
    res = g3.run(settings(tmp_path), FakeResolver({
        g3.ENTRY_POINTS[0]: FakeFn({"fields": _kin_fields()}),
        g3.ENTRY_POINTS[1]: FakeFn({"identical": True, "max_abs_delta": 0.0,
                                    "n_perturbations": 3})}))
    assert res.passed


def test_g3_fail_cav_channel(tmp_path):
    res = g3.run(settings(tmp_path), FakeResolver({
        g3.ENTRY_POINTS[0]: FakeFn({"fields": _kin_fields() + ["is_cav"]}),
        g3.ENTRY_POINTS[1]: FakeFn({"identical": True, "max_abs_delta": 0.0,
                                    "n_perturbations": 3})}))
    assert not res.passed


def test_g3_fail_perturbation_changes_output(tmp_path):
    res = g3.run(settings(tmp_path), FakeResolver({
        g3.ENTRY_POINTS[0]: FakeFn({"fields": _kin_fields()}),
        g3.ENTRY_POINTS[1]: FakeFn({"identical": False,
                                    "max_abs_delta": 1e-3,
                                    "n_perturbations": 3})}))
    assert not res.passed


def test_g3_piece_missing(tmp_path):
    res = g3.run(settings(tmp_path), FakeResolver({
        g3.ENTRY_POINTS[0]: FakeFn({"fields": _kin_fields()})}))
    assert not res.passed and res.error == "piece-missing"


# ---------------------------------------------------------------------- G4 ---
def _overfit(**kw):
    base = {"ok": True, "device": "cpu", "n_windows": 2, "threshold": 1e-2,
            "final_loss": 1e-4, "loss_curve": [1.0, 0.1, 1e-3, 1e-4]}
    base.update(kw)
    return base


def test_g4_pass(tmp_path):
    res = g4.run(settings(tmp_path), FakeResolver(
        {g4.ENTRY_POINTS[0]: FakeFn(_overfit())}))
    assert res.passed


def test_g4_fail_above_threshold(tmp_path):
    res = g4.run(settings(tmp_path), FakeResolver(
        {g4.ENTRY_POINTS[0]: FakeFn(_overfit(final_loss=0.5))}))
    assert not res.passed


def test_g4_fail_one_window(tmp_path):
    res = g4.run(settings(tmp_path), FakeResolver(
        {g4.ENTRY_POINTS[0]: FakeFn(_overfit(n_windows=1))}))
    assert not res.passed


def test_g4_piece_missing(tmp_path):
    res = g4.run(settings(tmp_path), FakeResolver())
    assert not res.passed and res.error == "piece-missing"


# ---------------------------------------------------------------------- G5 ---
def _e2e(**kw):
    base = {"ok": True, "device": "cpu", "timing_seconds": 30.0,
            "stages": list(g5.STAGES), "train_steps": 3, "n_worlds": 1}
    base.update(kw)
    return base


def test_g5_pass(tmp_path):
    s = settings(tmp_path, wm_n1_dir=clean_torch_tree(tmp_path))
    res = g5.run(s, FakeResolver({g5.ENTRY_POINTS[0]: FakeFn(_e2e())}))
    assert res.passed, res.detail
    assert res.checks["torch"]["ok"] is True


def test_g5_fail_missing_stage(tmp_path):
    s = settings(tmp_path, wm_n1_dir=clean_torch_tree(tmp_path))
    res = g5.run(s, FakeResolver(
        {g5.ENTRY_POINTS[0]: FakeFn(_e2e(stages=["convert", "timing"]))}))
    assert not res.passed


def test_g5_fail_forbidden_torch_api(tmp_path):
    s = settings(tmp_path, wm_n1_dir=dirty_torch_tree(tmp_path))
    res = g5.run(s, FakeResolver({g5.ENTRY_POINTS[0]: FakeFn(_e2e())}))
    assert not res.passed
    assert res.checks["torch"]["scan"]["forbidden_hits"]


def test_g5_piece_missing(tmp_path):
    s = settings(tmp_path, wm_n1_dir=clean_torch_tree(tmp_path))
    res = g5.run(s, FakeResolver())
    assert not res.passed and res.error == "piece-missing"


# ---------------------------------------------------------------------- G6 ---
def _splits():
    return {"splits": {
        "heldout-i-j4b": {"train": ["j4", "native"], "eval": ["j4b"],
                          "excluded": ["j4b"]},
        "heldout-ii-leave-pi": {"train": ["native", "damping"],
                                "eval": ["pi"], "excluded": ["pi"]},
        "heldout-iii-leave-highest-wall-speed": {
            "train": ["wall-speed-low", "native"],
            "eval": ["wall-speed-highest"],
            "excluded": ["wall-speed-highest"]},
    }}


def test_g6_pass(tmp_path):
    res = g6.run(settings(tmp_path), FakeResolver(
        {g6.ENTRY_POINTS[0]: FakeFn(_splits())}))
    assert res.passed


def test_g6_fail_leaked_arm(tmp_path):
    bad = _splits()
    bad["splits"]["heldout-i-j4b"]["train"] = ["j4", "j4b"]
    res = g6.run(settings(tmp_path), FakeResolver(
        {g6.ENTRY_POINTS[0]: FakeFn(bad)}))
    assert not res.passed


def test_g6_fail_missing_split(tmp_path):
    bad = _splits()
    bad["splits"].pop("heldout-ii-leave-pi")
    res = g6.run(settings(tmp_path), FakeResolver(
        {g6.ENTRY_POINTS[0]: FakeFn(bad)}))
    assert not res.passed


def test_g6_piece_missing(tmp_path):
    res = g6.run(settings(tmp_path), FakeResolver())
    assert not res.passed and res.error == "piece-missing"
