"""contrastfix A6: ORIGINAL-t0 train/eval global-token parity (DRAFT).

Status: pending the coordinator's reviewed guard and explicit test
authorization.  Synthetic-only; never executed in this draft state.

The trainer feeds ``StateGlobalToken(kin, kin_mask, counter)`` with the
window's ORIGINAL t0 state; the evaluation outcome path must rebuild the
token from the EXPLICIT t0 channels the P3 engine freezes before any
controller/teacher step (``t0_kin``/``t0_mask``/``t0_counter`` on the
rollout).  These synthetic tests pin the repair:

* identical ORIGINAL t0 state + counter -> identical training/evaluation
  token, including NONZERO t0 acceleration and VARIED vehicle lengths
  (which the trajectory tensors cannot carry);
* the same initial state with DIFFERENT future plans -> the SAME token;
* an A6-ON checkpoint refuses missing t0 channels (kinematics, mask or
  counter) in the real attribute path AND the injected mapping path --
  never a first-future-acceleration / 5 m default / ones-mask /
  zero-counter substitute;
* an A6-OFF checkpoint keeps the legacy zero-token path.

No real data, no SUMO: all state is built in memory.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from corr import StateGlobalToken
from evalh.idea_a import IdeaANotLoadedError, IdeaAVariantPredictor
from model import constants as C
from train import idea_a as T


class _Cfg:
    def __init__(self, on=True):
        self.state_global_token = bool(on)


def _state(B=2, K=5, seed=0):
    """A t0 kinematic view with NONZERO accel and VARIED lengths."""
    g = torch.Generator().manual_seed(seed)
    kin = torch.randn(B, K, 7, generator=g)
    kin[..., 1:4] = 0.0
    kin[..., 2] = 1.0                                  # lane index 1
    kin[..., 5] = torch.tensor([0.9, -1.4, 0.3, -0.6, 1.1])[:K].repeat(B, 1)
    kin[..., 6] = torch.tensor([4.2, 6.1, 5.4, 4.8, 7.0])[:K].repeat(B, 1)
    kin_mask = torch.ones(B, K, dtype=torch.bool)
    counter = torch.randn(B, C.N_ENTRANCES, C.COUNTER_BINS, generator=g)
    return kin, kin_mask, counter


def _future(kin, *, variant=0, S=1, T_steps=3):
    """A FUTURE plan that differs from every other ``variant``.

    Positions/speeds/lanes/masks/accel start AFTER the first applied step;
    only the explicit t0 channels carry the ORIGINAL t0 state.
    """
    B, K = kin.shape[:2]
    g = torch.Generator().manual_seed(100 + int(variant))
    pos = torch.cumsum(torch.randn(S, T_steps, B, K, generator=g), dim=1) + 20.0
    spd = torch.rand(S, T_steps, B, K, generator=g) * 10.0 + 5.0
    lanes = torch.randint(0, 3, (S, T_steps, B, K), generator=g)
    masks = torch.rand(S, T_steps, B, K, generator=g) > 0.2
    accel = torch.randn(S, T_steps, B, K, generator=g)
    return pos, spd, lanes, masks, accel


def _rollouts_from_state(kin, kin_mask, counter, *, future=None, attach=True,
                         as_mapping=False, S=1, T_steps=3):
    """A synthetic rollout carrying the explicit ORIGINAL t0 channels.

    ``attach=False`` builds a legacy test double with no ``t0_*`` channels
    at all (the pre-fix shape).  ``as_mapping`` returns the injected
    plain-dict form instead of an attribute object.
    """
    if future is None:
        future = _future(kin)
    pos, spd, lanes, masks, accel = future
    payload = dict(positions=pos, speeds=spd, lanes=lanes, masks=masks,
                   accel=accel)
    if attach:
        payload.update(t0_kin=kin.clone(), t0_mask=kin_mask.clone(),
                       t0_counter=counter.clone())
    if as_mapping:
        return dict(payload)
    return SimpleNamespace(**payload)


def _trained_encoder(d_model=16, seed=3):
    torch.manual_seed(seed)
    enc = StateGlobalToken(d_model=d_model)
    enc.eval()
    return enc


def _trainer_with(enc):
    cfg = T.IdeaATrainConfig(out_dir="unused", num_workers=0,
                             state_global_token=True)
    bundle = T.IdeaABundle(state_enc=enc, config=cfg)
    return T.IdeaATrainer(cfg, corpus=None, bundle=bundle)


def _predictor_with(enc):
    return IdeaAVariantPredictor(
        bundle=SimpleNamespace(state_enc=enc, cfg=_Cfg(True)))


def test_eval_token_equals_training_token_with_counter():
    enc = _trained_encoder()
    kin, kin_mask, counter = _state()
    trainer = _trainer_with(enc)
    train_token = trainer.global_token(
        {"kin": kin, "kin_mask": kin_mask, "counter": counter})

    predictor = _predictor_with(enc)
    attr_token = predictor._state_global_token(
        torch.zeros(kin.shape[0], 16),
        _rollouts_from_state(kin, kin_mask, counter))
    map_token = predictor._state_global_token(
        torch.zeros(kin.shape[0], 16),
        _rollouts_from_state(kin, kin_mask, counter, as_mapping=True))

    assert torch.allclose(train_token, attr_token, atol=1e-6)
    assert torch.allclose(train_token, map_token, atol=1e-6)
    assert torch.allclose(attr_token, enc(kin, kin_mask, counter), atol=1e-6)
    assert bool(attr_token.abs().sum() > 0.0)


def test_token_uses_nonzero_t0_accel_not_the_first_future_step():
    """The OLD fallback fed ``accel[0]`` (first FUTURE step) as the t0 a."""
    enc = _trained_encoder()
    kin, kin_mask, counter = _state()
    future = _future(kin)
    future_accel0 = future[4][0, 0]                    # [B,K] future step 0
    assert not torch.allclose(kin[..., 5], future_accel0)

    predictor = _predictor_with(enc)
    token = predictor._state_global_token(
        torch.zeros(kin.shape[0], 16),
        _rollouts_from_state(kin, kin_mask, counter, future=future))
    assert torch.allclose(token, enc(kin, kin_mask, counter), atol=1e-6)

    fallback_kin = kin.clone()
    fallback_kin[..., 5] = future_accel0               # what the old path fed
    assert not torch.allclose(token, enc(fallback_kin, kin_mask, counter),
                              atol=1e-6)


def test_token_uses_varied_lengths_not_the_5m_default():
    """The OLD fallback substituted the 5 m default for every length."""
    enc = _trained_encoder()
    kin, kin_mask, counter = _state()
    assert not torch.allclose(kin[..., 6], torch.full_like(kin[..., 6],
                                                           C.NOMINAL_LENGTH_M))

    predictor = _predictor_with(enc)
    token = predictor._state_global_token(
        torch.zeros(kin.shape[0], 16),
        _rollouts_from_state(kin, kin_mask, counter))
    assert torch.allclose(token, enc(kin, kin_mask, counter), atol=1e-6)

    default_kin = kin.clone()
    default_kin[..., 6] = C.NOMINAL_LENGTH_M           # what the old path fed
    assert not torch.allclose(token, enc(default_kin, kin_mask, counter),
                              atol=1e-6)


def test_same_t0_state_with_different_future_plans_same_token():
    """Only the t0 channels drive the token; the future plan must not."""
    enc = _trained_encoder()
    kin, kin_mask, counter = _state()
    predictor = _predictor_with(enc)
    proj = torch.zeros(kin.shape[0], 16)
    token_a = predictor._state_global_token(
        proj, _rollouts_from_state(kin, kin_mask, counter,
                                   future=_future(kin, variant=1)))
    token_b = predictor._state_global_token(
        proj, _rollouts_from_state(kin, kin_mask, counter,
                                   future=_future(kin, variant=2)))
    assert torch.allclose(token_a, token_b, atol=1e-6)


def test_missing_t0_kin_mask_channels_fail_closed():
    """A6 ON: absent/partial t0 kin/mask channels are refused, both paths."""
    enc = _trained_encoder()
    kin, kin_mask, counter = _state()
    predictor = _predictor_with(enc)
    proj = torch.zeros(kin.shape[0], 16)

    legacy = _rollouts_from_state(kin, kin_mask, counter, attach=False)
    try:
        predictor._state_global_token(proj, legacy)
    except IdeaANotLoadedError as exc:
        assert "t0" in str(exc)
    else:                                              # pragma: no cover
        raise AssertionError("attribute rollout without t0 channels was "
                             "scored via the trajectory fallback")

    full_map = _rollouts_from_state(kin, kin_mask, counter, as_mapping=True)
    partial = {k: v for k, v in full_map.items() if k != "t0_mask"}
    try:
        predictor._state_global_token(proj, partial)
    except IdeaANotLoadedError:
        pass
    else:                                              # pragma: no cover
        raise AssertionError("mapping rollout without t0_mask did not refuse")

    mask_only = {k: v for k, v in full_map.items() if k != "t0_kin"}
    try:
        predictor._state_global_token(proj, mask_only)
    except IdeaANotLoadedError:
        pass
    else:                                              # pragma: no cover
        raise AssertionError("mapping rollout without t0_kin did not refuse")


def test_missing_counter_fails_closed_attribute_and_mapping():
    """A6 ON: a present-but-None or absent counter channel is refused."""
    enc = _trained_encoder()
    kin, kin_mask, counter = _state()
    predictor = _predictor_with(enc)
    proj = torch.zeros(kin.shape[0], 16)

    none_counter = _rollouts_from_state(kin, kin_mask, counter)
    none_counter.t0_counter = None
    try:
        predictor._state_global_token(proj, none_counter)
    except IdeaANotLoadedError as exc:
        assert "counter" in str(exc)
    else:                                              # pragma: no cover
        raise AssertionError("missing required counter was silently scored")

    none_map = dict(_rollouts_from_state(kin, kin_mask, counter,
                                         as_mapping=True),
                    t0_counter=None)
    try:
        predictor._state_global_token(proj, none_map)
    except IdeaANotLoadedError:
        pass
    else:                                              # pragma: no cover
        raise AssertionError("mapping rollout without counter did not refuse")

    absent_map = _rollouts_from_state(kin, kin_mask, counter, as_mapping=True)
    absent_map.pop("t0_counter")
    try:
        predictor._state_global_token(proj, absent_map)
    except IdeaANotLoadedError:
        pass
    else:                                              # pragma: no cover
        raise AssertionError("mapping rollout with no counter key did not "
                             "refuse")


def test_eval_token_changes_with_the_counter():
    enc = _trained_encoder()
    kin, kin_mask, counter = _state()
    predictor = _predictor_with(enc)
    proj = torch.zeros(kin.shape[0], 16)
    with_counter = predictor._state_global_token(
        proj, _rollouts_from_state(kin, kin_mask, counter))
    zero_counter = predictor._state_global_token(
        proj, _rollouts_from_state(kin, kin_mask,
                                   torch.zeros_like(counter)))
    assert not torch.allclose(with_counter, zero_counter)
    # the zero-counter evaluation equals the explicit ``None`` legacy token
    assert torch.allclose(zero_counter, enc(kin, kin_mask, None), atol=1e-6)


def test_flag_off_checkpoint_keeps_legacy_zero_token():
    """No encoder + flag OFF reproduces the pre-A6 zero token exactly."""
    kin, kin_mask, counter = _state()
    predictor = IdeaAVariantPredictor(
        bundle=SimpleNamespace(state_enc=None, cfg=_Cfg(False)))
    token = predictor._state_global_token(
        torch.zeros(kin.shape[0], 16),
        _rollouts_from_state(kin, kin_mask, counter, attach=False))
    assert bool((token == 0.0).all())


def test_rollout_counter_reports_channel_presence_not_value():
    """``_rollout_counter`` says WHETHER a channel exists, not its value."""
    from rollout2.state import Rollouts

    counter = torch.zeros(2, C.N_ENTRANCES, C.COUNTER_BINS)
    r = Rollouts(
        positions=torch.zeros(1, 2, 2, 3), speeds=torch.zeros(1, 2, 2, 3),
        lanes=torch.zeros(1, 2, 2, 3, dtype=torch.long),
        masks=torch.ones(1, 2, 2, 3), accel=torch.zeros(1, 2, 2, 3),
        arrivals=torch.zeros(1, 2, 2, C.N_ENTRANCES),
        insertions=torch.zeros(1, 2, 2, C.N_ENTRANCES),
        t0_counter=counter)
    got, present = IdeaAVariantPredictor._rollout_counter(r, None)
    assert present is True and got is counter
    r.t0_counter = None
    _, present = IdeaAVariantPredictor._rollout_counter(r, None)
    assert present is True                             # channel carried, no value
    bare = SimpleNamespace()
    got, present = IdeaAVariantPredictor._rollout_counter(bare, None)
    assert present is False and got is None
    got, present = IdeaAVariantPredictor._rollout_counter(
        {"t0_counter": counter}, None)
    assert present is True and got is counter
