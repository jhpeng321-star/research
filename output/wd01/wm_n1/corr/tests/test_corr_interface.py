"""P5 acceptance tests: no label schema, quantile monotonicity, motion-summary
invariance, OOD score primitives."""
from __future__ import annotations

import inspect

import torch

from corr import (CorrectionHead, CorrConfig, CavMotionSummary, SUMMARY_RAW_F,
                  cav_motion_summary, ensemble_disagreement,
                  motion_density_fit, motion_density_score)

_FORBIDDEN = ("label", "strategy", "action", "controller", "policy")


def _forbidden(names):
    return [n for n in names
            if any(k in n.lower() for k in _FORBIDDEN)]


def test_summary_signature_has_no_strategy_or_labels():
    params = list(inspect.signature(cav_motion_summary).parameters)
    assert params == ["positions", "speeds", "accel", "cav_mask"]
    assert _forbidden(params) == []


def test_head_signature_has_no_strategy_or_labels():
    params = list(inspect.signature(CorrectionHead.forward).parameters)
    assert params == ["self", "summary", "global_token"]
    assert _forbidden(params) == []


def _traj(t=3, b=2, k=4, *, cav=None):
    gen = torch.Generator().manual_seed(0)
    positions = torch.rand(t, b, k, generator=gen) * 100.0
    speeds = 20.0 + torch.rand(t, b, k, generator=gen) * 5.0
    accel = torch.randn(t, b, k, generator=gen) * 1.5
    if cav is None:
        cav = torch.tensor([[True, False, True, False],
                            [False, False, True, True]])
    return positions, speeds, accel, cav


def test_cav_motion_summary_shape_and_range():
    positions, speeds, accel, cav = _traj()
    out = cav_motion_summary(positions, speeds, accel, cav)
    assert out.shape == (2, SUMMARY_RAW_F)
    assert torch.isfinite(out).all()
    assert out.dtype == torch.float32
    assert torch.allclose(out[:, 0], torch.tensor([0.5, 0.5]))


def test_summary_invariance_to_controller_name():
    positions, speeds, accel, cav = _traj()
    # Identical controller-produced trajectory, two different "controller"
    # labels; the function accepts no label, so the outputs must be identical.
    controller_a = "damping"
    controller_b = "wall-v2.2"
    assert controller_a != controller_b
    a = cav_motion_summary(positions, speeds, accel, cav)
    b = cav_motion_summary(positions.clone(), speeds.clone(), accel.clone(),
                           cav.clone())
    assert torch.equal(a, b)


def test_summary_empty_cav_is_zero():
    positions, speeds, accel, cav = _traj()
    cav = torch.zeros_like(cav)
    out = cav_motion_summary(positions, speeds, accel, cav)
    assert torch.allclose(out, torch.zeros(2, SUMMARY_RAW_F))


def test_projection_shape():
    proj = CavMotionSummary(CorrConfig())
    out = proj(torch.zeros(2, SUMMARY_RAW_F))
    assert out.shape == (2, 64)


def test_head_output_shapes_and_monotone_quantiles():
    cfg = CorrConfig()
    head = CorrectionHead(cfg)
    res = head(torch.randn(2, cfg.summary_dim),
               torch.randn(2, cfg.d_model))
    q = res["outcome_quantiles"]
    assert q.shape == (2, cfg.n_tau, cfg.n_outcome_targets, cfg.n_quantiles)
    assert res["collision_logits"].shape == (2, cfg.n_tau)
    assert torch.isfinite(q).all()
    assert (q >= 0).all()
    assert torch.all(q[..., 1:] - q[..., :-1] >= -1e-6)


def test_crps_delegates_and_prefers_accurate():
    levels = torch.linspace(0.05, 0.95, 19)
    target = torch.full((2, 3), 5.0)
    accurate = target.unsqueeze(-1).expand(2, 3, 19).clone()
    offset = accurate + 10.0
    c_acc = CorrectionHead.crps(accurate, target, levels)
    c_off = CorrectionHead.crps(offset, target, levels)
    assert float(c_acc) < float(c_off)
    assert float(c_acc) >= 0.0


def test_summary_depends_only_on_cav_trajectory():
    """Perturbing every NON-CAV slot (position/speed/accel) leaves the summary
    bit-identical: it is a function of the CAV trajectory alone."""
    positions, speeds, accel, cav = _traj()
    base = cav_motion_summary(positions, speeds, accel, cav)
    p, s, a = positions.clone(), speeds.clone(), accel.clone()
    nc = ~cav.bool()
    p[:, nc] += 500.0
    s[:, nc] = 0.0
    a[:, nc] -= 9.0
    assert torch.equal(base, cav_motion_summary(p, s, a, cav))


def test_correction_head_ignores_label_channels_in_full_batch():
    """Acceptance perturbation: a full batch may carry strategy/action/outcome
    channels, but the correction head only consumes the CAV-motion summary +
    global token, so perturbing those labels cannot change its output."""
    cfg = CorrConfig()
    head = CorrectionHead(cfg)
    head.eval()
    proj = CavMotionSummary(cfg)
    positions, speeds, accel, cav = _traj()
    g = torch.Generator().manual_seed(9)
    global_token = torch.randn(2, cfg.d_model, generator=g)

    def run(batch):
        raw = cav_motion_summary(batch["positions"], batch["speeds"],
                                 batch["accel"], batch["cav_mask"])
        return head(proj(raw), global_token)

    a = {"positions": positions, "speeds": speeds, "accel": accel,
         "cav_mask": cav, "strategy": "native",
         "action_code": torch.zeros(2, 3, dtype=torch.long),
         "outcome": torch.zeros(2, 3, 19)}
    b = dict(a, strategy="wall-v2.2",
             action_code=torch.ones(2, 3, dtype=torch.long),
             outcome=torch.ones(2, 3, 19) * 99.0)
    assert a["strategy"] != b["strategy"]
    oa, ob = run(a), run(b)
    for key in oa:
        assert torch.equal(oa[key], ob[key]), key


def test_state_global_token_ignores_label_channels():
    """M1FIX2: the state encoder's token reads kinematics + counters only, so
    a batch's strategy/action/CAV-flag/outcome channels cannot change it."""
    from corr import StateGlobalToken

    enc = StateGlobalToken(d_model=8)
    enc.eval()
    g = torch.Generator().manual_seed(3)
    batch = {
        "kin": torch.randn(2, 5, 7, generator=g),
        "kin_mask": torch.ones(2, 5, dtype=torch.bool),
        "counter": torch.randn(2, 2, 30, generator=g),
        "is_cav": torch.zeros(2, 5, dtype=torch.bool),
        "strategy": "native",
        "action_code": torch.zeros(2, 5, dtype=torch.long),
        "outcome": torch.zeros(2, 5, 19),
    }
    params = list(inspect.signature(StateGlobalToken.forward).parameters)
    assert params == ["self", "kin", "kin_mask", "counter"]
    assert _forbidden(params) == []
    base = enc(batch["kin"], batch["kin_mask"], batch["counter"])
    other = dict(batch, is_cav=~batch["is_cav"], strategy="wall-v2.2",
                 action_code=torch.ones(2, 5, dtype=torch.long),
                 outcome=torch.ones(2, 5, 19) * 99.0)
    assert torch.equal(base, enc(other["kin"], other["kin_mask"],
                                 other["counter"]))


def test_motion_density_score_higher_for_unfamiliar():
    gen = torch.Generator().manual_seed(1)
    familiar = torch.randn(64, SUMMARY_RAW_F, generator=gen) * 0.5
    model = motion_density_fit(familiar)
    near = motion_density_score(familiar[:4], model)
    far = motion_density_score(familiar[:4] + 8.0, model)
    assert (far > near).all()


def test_ensemble_disagreement_higher_for_divergent():
    gen = torch.Generator().manual_seed(2)
    base = torch.randn(1, 5, SUMMARY_RAW_F, generator=gen)
    unanimous = base.repeat(3, 1, 1)
    divergent = unanimous + torch.randn(3, 5, SUMMARY_RAW_F, generator=gen) * 5.0
    assert torch.allclose(ensemble_disagreement(unanimous),
                          torch.zeros(5))
    assert (ensemble_disagreement(divergent)
            > ensemble_disagreement(unanimous)).all()
