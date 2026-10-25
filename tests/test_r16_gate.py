"""R16 gate contracts (CPU only; no training, no data)."""
from dataclasses import replace

import pytest

from wms.train import round16_gate as g


def test_configs_are_r14_plus_exactly_the_registered_r16_changes():
    out = g.check_configs()
    assert out["a"].model.ramp_speed_adapt_learnable and out["b"].model.ramp_speed_adapt_learnable
    assert out["a"].model.init_from == g.INIT_REL
    assert out["b"].model.init_from == "runs/c4_wm_v10_round16_a/wm_s.pt"


@pytest.mark.parametrize("stage,change", [
    ("a", lambda c: replace(c, model=replace(c.model, ramp_speed_adapt_learnable=False))),
    ("a", lambda c: replace(c, optim=replace(c.optim, lr_scale={**c.optim.lr_scale, "ramp_speed_adapt": 10.0}))),
    ("b", lambda c: replace(c, optim=replace(c.optim, freeze_prefixes=c.optim.freeze_prefixes[:-1]))),
    ("a", lambda c: replace(c, physics=replace(c.physics, secure_gap_veto=False))),
])
def test_any_other_difference_is_refused(monkeypatch, stage, change):
    original = g.expected_config

    def tampered(s):
        expected, cfg = original(s)
        return expected, (change(cfg) if s == stage else cfg)
    monkeypatch.setattr(g, "expected_config", tampered)
    with pytest.raises(Exception):
        g.check_configs()


def test_built_stage_a_model_holds_the_train_fitted_init():
    from wms.cli.assemble import build_model, build_physics
    cfg = g.check_configs()["a"]
    g.check_ramp_init(build_model(cfg, build_physics(cfg)))


def _report(tmp_path, before, new):
    import hashlib
    path = tmp_path / "wm_s.pt"
    path.write_bytes(b"x")
    return path, {"sha256": hashlib.sha256(b"x").hexdigest(), "loaded": sorted(before),
                  "new_in_model": sorted(new), "expanded": [], "shape_mismatch": [],
                  "unused_from_checkpoint": []}


def test_init_check_a_requires_exactly_the_three_new_ramp_tensors(tmp_path):
    before = ["driver_encoder.head.0.weight", "merge_yield.log_decel"]
    path, report = _report(tmp_path, before, g.RAMP_TENSORS)
    g.check_init(report, before, path, "a")
    _, bad = _report(tmp_path, before, g.RAMP_TENSORS[:2])
    with pytest.raises(Exception):
        g.check_init(bad, before, path, "a")
    with pytest.raises(Exception):
        g.check_init(report, [*before, g.RAMP_TENSORS[0]], path, "a")


def test_launch_is_refused():
    with pytest.raises(SystemExit, match="preparation only"):
        g.main(["launch"])
