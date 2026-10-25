"""R16 gate contracts (CPU only; no training, no data)."""
from dataclasses import replace

import pytest

from wms.train import round13e_gate as g13
from wms.train import round16_gate as g

_MUTATED = ("VARIANT", "TAG", "RUN", "GS", "FLAG_KEYS", "operational_root", "check_configs", "check_init",
            "YIELD_TENSORS")


@pytest.fixture(autouse=True)
def restore_13e_gate_globals():
    """install() re-points round13e_gate module globals; restore them for later test modules."""
    saved = {k: getattr(g13, k) for k in _MUTATED if hasattr(g13, k)}
    variants, configs = dict(g13.VARIANTS), dict(g13.CONFIGS)
    yield
    for k, v in saved.items():
        setattr(g13, k, v)
    g13.VARIANTS.clear()
    g13.VARIANTS.update(variants)
    g13.CONFIGS.clear()
    g13.CONFIGS.update(configs)


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
                  "unused_from_checkpoint": [], "follower_courtesy_init": "checkpoint"}


def test_init_check_a_requires_exactly_the_three_new_ramp_tensors(tmp_path):
    before = ["driver_encoder.head.0.weight", "merge_yield.log_decel", "merge_yield.log_distance"]
    path, report = _report(tmp_path, before, g.RAMP_TENSORS)
    g.check_init(report, before, path, "a")
    _, bad = _report(tmp_path, before, g.RAMP_TENSORS[:2])
    with pytest.raises(Exception):
        g.check_init(bad, before, path, "a")
    with pytest.raises(Exception):
        g.check_init(report, [*before, g.RAMP_TENSORS[0]], path, "a")


def test_launch_subcommand_points_to_the_script():
    with pytest.raises(SystemExit, match="round16_gpuB.sh"):
        g.main(["launch"])


def test_install_repoints_the_13e_verdict_machinery():
    m = g.install()
    assert m.TAG == "r16" and m.RUN == "c4_wm_v10_round16" and m.check_init is g.check_init
    assert set(g.RAMP_TENSORS) <= set(m.YIELD_TENSORS)
    from wms.cli.assemble import build_model, build_physics
    cfg = g.check_configs()["a"]
    names = m.allowed_trainables(build_model(cfg, build_physics(cfg)), cfg, "a")
    assert set(g.RAMP_TENSORS) <= set(names)
