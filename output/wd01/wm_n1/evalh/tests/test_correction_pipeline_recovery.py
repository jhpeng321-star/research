"""DRAFT synthetic correction integrity tests; NOT RUN, pending reviewed guard.

No fixtures or data files are opened. RealCorrBundle is instantiated with
synthetic weights: these tests verify wiring, not trained predictive quality.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from corr import CavMotionSummary, CorrConfig, CorrectionHead, StateGlobalToken
from corr.evaluation import trajectory_summary
from evalh.idea_a import IdeaANotLoadedError, IdeaAVariantPredictor
from features.idea_a import CAV_ACCEL_SCALE, CAV_TGT_SPEED_SCALE
from model import constants as C
from train.g4_real import RealCorrBundle
from train.idea_a import _cav_motion_summary
from wm_api.engine import outcome_head


def _fixture():
    # Slot 0 survives; slot 1 is HDV; slot 2 departs then is reused;
    # slot 3 begins empty then gets a new arrival.
    pos = torch.tensor([[[0., 100., 20., 0.]], [[6., 107., 24., 0.]],
                        [[13., 116., 999., 500.]], [[21., 125., -800., 800.]]])
    speed = torch.tensor([[[10., 14., 8., 0.]], [[12., 14., 8., 0.]],
                          [[14., 18., 50., 40.]], [[16., 18., 60., 70.]]])
    accel = torch.tensor([[[4., 0., 0., 0.]], [[4., 8., 50., 40.]],
                          [[4., 0., 60., 70.]]])
    mask = torch.tensor([[[1, 1, 1, 0]], [[1, 1, 1, 0]],
                         [[1, 1, 0, 1]], [[1, 1, 1, 1]]], dtype=torch.bool)
    ids = torch.tensor([[[10, 11, 12, -1]], [[10, 11, 12, -1]],
                        [[10, 11, 12, 13]], [[10, 11, 14, 13]]])
    kin = torch.zeros(1, 4, 7)
    kin[..., 0], kin[..., 4] = pos[0], speed[0]
    kin[..., 1] = 1.
    kin[..., 5] = torch.tensor([[2., -1., -2., 0.]])
    kin[..., 6] = torch.tensor([[4.2, 7.0, 5.5, 0.]])
    return dict(positions=pos[None], speeds=speed[None], accel=accel[None],
                masks=mask[None], slot_ids=ids[None], t0_kin=kin,
                t0_mask=mask[0].clone(), t0_slot_id=ids[0].clone(),
                t0_is_cav=torch.tensor([[1, 0, 1, 0]], dtype=torch.bool),
                t0_counter=torch.arange(60).reshape(1, 2, 30).float())


def _bundle(dtype=torch.float32, device="cpu", on=True):
    torch.manual_seed(92)
    cfg = CorrConfig(d_model=16)
    corr = RealCorrBundle(cfg).to(device=device, dtype=dtype).eval()
    enc = StateGlobalToken(d_model=16).to(device=device, dtype=dtype).eval() if on else None
    return SimpleNamespace(corr=corr, state_enc=enc,
                           cfg=SimpleNamespace(state_global_token=on))


def _raw(data):
    return trajectory_summary(
        data["positions"][0], data["speeds"][0], data["accel"][0], data["masks"][0],
        slot_ids=data["slot_ids"][0],
        **{key: data[key] for key in ("t0_kin", "t0_mask", "t0_is_cav", "t0_slot_id")})


def _api(bundle, data):
    return outcome_head(
        bundle, data["positions"][0], data["speeds"][0], data["accel"][0],
        data["masks"][0], slot_ids=data["slot_ids"][0],
        **{key: data[key] for key in ("t0_kin", "t0_mask", "t0_is_cav", "t0_slot_id", "t0_counter")})


def _rows():
    return [SimpleNamespace(tau_s=float(C.TAUS[0]))]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_composition_exact_train_eval_api_parity(dtype, monkeypatch):
    data, bundle = _fixture(), _bundle(dtype)
    # Independently reconstruct the trainer's actual canonical window path.
    alive = torch.cumprod(torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0],
                                      [1, 1, 0, 0], [1, 1, 0, 0]]), dim=0).bool()
    times = torch.arange(4).view(4, 1)
    fill = torch.cummax(torch.where(alive, times, 0), dim=0).values
    pos_ff = data["positions"][0, :, 0].gather(0, fill)
    plan = torch.zeros(4, 4, 8)
    plan[..., 0] = data["speeds"][0, :, 0] / float(CAV_TGT_SPEED_SCALE)
    plan[..., 1] = torch.cat([data["t0_kin"][..., 5], data["accel"][0, :, 0]], 0) / float(CAV_ACCEL_SCALE)
    plan_ff = plan.gather(0, fill[..., None].expand(-1, -1, 8))
    expected_raw = _cav_motion_summary(
        pos_ff[:, None], plan_ff[:, None],
        (alive & data["t0_is_cav"][0]).unsqueeze(1))
    assert torch.equal(_raw(data), expected_raw)
    # Explicit pins catch omitted t0 and displacement polluted by reuse.
    assert expected_raw[0, 0].item() == 0.5
    assert expected_raw[0, 9].item() == -2.0  # nonzero original acceleration
    assert expected_raw[0, 12].item() == 12.5  # (21 + 4) / 2
    with torch.no_grad():
        token = bundle.state_enc(data["t0_kin"].to(dtype), data["t0_mask"], data["t0_counter"].to(dtype))
        expected_q = bundle.corr(expected_raw.to(dtype), token)["outcome_quantiles"]
    seen = []
    hook = bundle.corr.register_forward_pre_hook(lambda module, args: seen.append(args))
    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation constructed an untrained projection")
    monkeypatch.setattr(CavMotionSummary, "__init__", forbidden)
    predictor = IdeaAVariantPredictor(bundle=bundle)
    q_eval = predictor._outcome(_rows(), SimpleNamespace(**data))
    q_map = predictor._outcome(_rows(), data)
    q_api, raw_api = _api(bundle, data)
    hook.remove()
    target = C.OUTCOME_TARGETS.index("delay_s")
    np.testing.assert_array_equal(q_eval[0], expected_q[0, 0, target].numpy())
    np.testing.assert_array_equal(q_map, q_eval)
    np.testing.assert_array_equal(q_api, expected_q[0].numpy())
    np.testing.assert_array_equal(raw_api, expected_raw[0].numpy())
    assert len(seen) == 3 and all(args[0].shape[-1] == 16 for args in seen)
    assert all(not x.requires_grad for args in seen for x in args)


def test_hdv_and_departed_reused_slots_cannot_change_summary():
    data = _fixture()
    expected = _raw(data)
    changed = deepcopy(data)
    for key in ("positions", "speeds"):
        changed[key][:, 1:, :, 1] += 10000.  # HDV future
        changed[key][:, 2:, :, 2:] -= 9000.  # departed CAV/new arrival
    changed["accel"][..., 1] += 4000.
    changed["accel"][:, 1:, :, 2:] -= 3000.
    assert torch.equal(_raw(changed), expected)
    changed["positions"][0, -1, 0, 0] += 3.
    assert not torch.equal(_raw(changed), expected)


def test_same_slot_id_after_absence_stays_dead_and_immediate_reuse_is_dead():
    data = _fixture()
    same_id = deepcopy(data)
    same_id["slot_ids"][0, -1, 0, 2] = 12
    assert torch.equal(_raw(same_id), _raw(data))
    immediate = deepcopy(data)
    immediate["masks"][0, 2, 0, 2] = True
    immediate["slot_ids"][0, 2:, 0, 2] = 15
    assert torch.equal(_raw(immediate), _raw(data))


def test_multiple_samples_score_coherent_identity_then_average_quantiles():
    first, bundle = _fixture(), _bundle()
    second = deepcopy(first)
    second["masks"][0, 1, 0, 0] = False
    second["slot_ids"][0, 2:, 0, 0] = 20
    second["positions"][0, 1:, 0, 0] = 1000.
    second["speeds"][0, 1:, 0, 0] = 100.
    assert not torch.equal(_raw(first), _raw(second))
    combined = deepcopy(first)
    for key in ("positions", "speeds", "accel", "masks", "slot_ids"):
        combined[key] = torch.cat([first[key], second[key]], dim=0)
    predictor = IdeaAVariantPredictor(bundle=bundle)
    expected = (predictor._outcome(_rows(), first)
                + predictor._outcome(_rows(), second)) / 2.
    seen = []
    hook = bundle.corr.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].clone()))
    actual = predictor._outcome(_rows(), combined)
    hook.remove()
    assert len(seen) == 2
    assert torch.equal(seen[0], _raw(first))
    assert torch.equal(seen[1], _raw(second))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("as_mapping", [False, True])
@pytest.mark.parametrize("missing", ["t0_kin", "t0_mask", "t0_counter", "t0_is_cav", "t0_slot_id", "slot_ids", "accel"])
def test_corrected_scoring_missing_channels_fails_closed(as_mapping, missing):
    data, bundle = _fixture(), _bundle()
    data.pop(missing)
    rollout = data if as_mapping else SimpleNamespace(**data)
    with pytest.raises(IdeaANotLoadedError):
        IdeaAVariantPredictor(bundle=bundle)._outcome(_rows(), rollout)


def test_unsupported_bare_head_and_missing_head_fail_closed():
    data = _fixture()
    for head in (None, CorrectionHead(CorrConfig(d_model=16))):
        with pytest.raises(IdeaANotLoadedError, match="trained raw16"):
            IdeaAVariantPredictor(bundle=SimpleNamespace(corr=head))._outcome(_rows(), data)


def test_a6_off_zeros_still_requires_explicit_cav_identity():
    data, bundle = _fixture(), _bundle(on=False)
    data.pop("t0_counter")
    predictor = IdeaAVariantPredictor(bundle=bundle)
    assert np.isfinite(predictor._outcome(_rows(), data)).all()
    assert not predictor._state_global_token(torch.zeros(1, 16), {}).any()
    data.pop("t0_is_cav")
    with pytest.raises(IdeaANotLoadedError, match="t0_is_cav"):
        predictor._outcome(_rows(), data)


def test_explicit_a6_off_ignores_stale_encoder_without_requiring_counter():
    data, bundle = _fixture(), _bundle(on=True)
    bundle.cfg.state_global_token = False
    data.pop("t0_counter")
    seen = []
    hook = bundle.state_enc.register_forward_pre_hook(
        lambda module, args: seen.append(args))
    predictor = IdeaAVariantPredictor(bundle=bundle)
    token = predictor._state_global_token(torch.zeros(1, 16), data)
    q_eval = predictor._outcome(_rows(), data)
    q_api, _ = outcome_head(
        bundle, data["positions"][0], data["speeds"][0], data["accel"][0],
        data["masks"][0], slot_ids=data["slot_ids"][0],
        **{key: data[key] for key in ("t0_kin", "t0_mask", "t0_is_cav", "t0_slot_id")})
    hook.remove()
    assert not seen and not token.any()
    np.testing.assert_array_equal(q_eval[0], q_api[0, C.OUTCOME_TARGETS.index("delay_s")])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_encoder_and_correction_have_independent_dtype_device_placement(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    data, bundle = _fixture(), _bundle(torch.float64, device)
    bundle.state_enc.to(device="cpu", dtype=torch.float32)
    predictor = IdeaAVariantPredictor(bundle=bundle)
    seen = []
    encoder_seen = []
    hook = bundle.corr.register_forward_pre_hook(lambda module, args: seen.append(args))
    enc_hook = bundle.state_enc.register_forward_pre_hook(
        lambda module, args: encoder_seen.append(args))
    q = predictor._outcome(_rows(), data)
    q_api, _ = _api(bundle, data)
    hook.remove()
    enc_hook.remove()
    assert np.isfinite(q).all() and np.isfinite(q_api).all()
    assert all(x.device.type == device and x.dtype == torch.float64
               for args in seen for x in args)
    assert all(args[0].device.type == "cpu" and args[0].dtype == torch.float32
               and args[1].dtype == torch.bool and args[2].dtype == torch.float32
               for args in encoder_seen)
    reference = torch.zeros(1, 16, device=device, dtype=torch.float64)
    token = predictor._state_global_token(reference, data)
    changed = deepcopy(data)
    changed["positions"][:, 1:] += 300.
    changed["speeds"][:, 1:] += 20.
    assert torch.equal(token, predictor._state_global_token(reference, changed))
    assert not token.requires_grad
