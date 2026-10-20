"""R5INPUTS (2026-10-20): ``model.lane_v5_extra_features`` extra merge inputs.

Covers: the OFF path stays exactly the legacy feature vector / network width; the extra
columns are finite and causal; hand-built opening vs closing gaps, next-behind / next-ahead
neighbours, missing neighbours and predicted alignment times; the extra accumulators carry
across chunks and match the teacher-forced recomputation; and an old checkpoint loaded with
the switch ON is an exact identity (ZERO-expanded first trunk layer, identity scaling).
"""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest
import torch

from .test_gaprule import _kin
from .test_lanefix import _phys
from .test_lanev5 import _c4, _model, V5
from wms.config.load import load_train_config
from wms.model import lane_v5 as lv5
from wms.model.checkpoint import load_checkpoint, save_checkpoint
from wms.model.spec import ModelSpec
from wms.train.settings import TrainSettings

EXTRA = lv5.EXTRA_FEATURE_NAMES


def _scene(rows):
    return _kin(rows)


def _present(kin):
    return torch.ones(kin.shape[:2], dtype=torch.bool)


def _scene_kin(accel):
    """Ego ramp car (s=90, v=15) between lane-1 follower (80) and leader (120).

    Returns the kinematic tensor; follower / leader / next-neighbour accelerations are set
    by the caller through ``accel`` = ``(follower, next_behind, leader, next_ahead)``.
    """
    kin = _scene([(90.0, 2, 15.0), (120.0, 1, 20.0), (80.0, 1, 10.0),
                  (50.0, 1, 10.0), (160.0, 1, 22.0)])
    kin[0, 0, 5] = -2.0
    for slot, value in zip((2, 3, 1, 4), accel):
        kin[0, slot, 5] = value
    return kin


def _extra(kin, present, phys):
    acc = lv5.initial_acc(present, kin, phys, None, None, 1)
    raw, _, inst = lv5.step_features(acc, kin, present, phys, None, extra=True)
    return raw, inst


def _col(raw, name):
    """Scalar extra column of the ego slot (row 0, slot 0)."""
    return raw[0, 0, lv5.N_FEATURES + EXTRA.index(name)]


# --------------------------------------------------------------- switch off / width
def test_off_path_is_the_legacy_vector_and_extra_only_appends():
    phys = _phys()
    kin = _scene_kin((-1.0, 0.1, 0.2, 0.0))
    present = _present(kin)
    acc0 = lv5.initial_acc(present, kin, phys, None, None, 1)
    base, base_acc = lv5.sequence_features(kin.unsqueeze(1), present.unsqueeze(1), phys,
                                           torch.zeros_like(present, dtype=torch.float32)
                                           .unsqueeze(1), acc0)
    extra, extra_acc = lv5.sequence_features(kin.unsqueeze(1), present.unsqueeze(1), phys,
                                             torch.zeros_like(present, dtype=torch.float32)
                                             .unsqueeze(1), acc0, extra=True)
    assert base.shape[-1] == lv5.N_FEATURES
    assert extra.shape[-1] == lv5.N_FEATURES_EXTRA == lv5.N_FEATURES + len(EXTRA)
    assert torch.equal(base, extra[..., :lv5.N_FEATURES])
    n_base_acc = lv5._A["seen"] + 1
    assert torch.equal(base_acc[..., :n_base_acc], extra_acc[..., :n_base_acc])
    assert torch.equal(base_acc[..., n_base_acc:], torch.zeros_like(base_acc[..., n_base_acc:]))
    assert torch.equal(base_acc[..., lv5.ACC_NAMES.index("seen")], torch.ones(1, 1, 5))
    assert torch.isfinite(extra).all() and torch.isfinite(extra_acc).all()

    torch.manual_seed(0)
    off = lv5.LaneV5Net()
    torch.manual_seed(0)
    on = lv5.LaneV5Net(extra_features=True)
    assert off.trunk[0].in_features == lv5.N_FEATURES
    assert on.trunk[0].in_features == lv5.N_FEATURES_EXTRA
    assert on.feature_mu.shape == on.feature_sd.shape == (lv5.N_FEATURES_EXTRA,)
    assert torch.equal(on.feature_mu[-len(EXTRA):], torch.zeros(len(EXTRA)))
    assert torch.equal(on.feature_sd[-len(EXTRA):], torch.ones(len(EXTRA)))


def test_features_are_causal():
    phys = _phys()
    kin = _scene_kin((-1.0, 0.1, 0.2, 0.0))
    present = _present(kin)
    states = kin.unsqueeze(1).repeat(1, 6, 1, 1)
    states[:, 1:, 0, 0] += torch.arange(1, 6).view(1, 5).float() * 2.0
    age = torch.zeros(present.shape[0], 6, *present.shape[1:])
    acc0 = lv5.initial_acc(present, kin, phys, None, None, 1)
    f1, a1 = lv5.sequence_features(states, present.unsqueeze(1).repeat(1, 6, 1), phys,
                                   age, acc0, extra=True)
    states2 = states.clone()
    states2[:, 4:] += 100.0
    f2, a2 = lv5.sequence_features(states2, present.unsqueeze(1).repeat(1, 6, 1), phys,
                                   age, acc0, extra=True)
    assert torch.equal(f1[:, :4], f2[:, :4]) and torch.equal(a1[:, :4], a2[:, :4])


# --------------------------------------------------------------- hand-built semantics
def test_opening_and_closing_gap_and_next_neighbours():
    phys = _phys()
    open_kin = _scene_kin((-1.0, 0.1, 0.2, 0.0))            # follower 10 m/s < ego 15
    close_kin = _scene_kin((1.0, 0.1, 0.2, 0.0))
    close_kin[0, 2, 4] = 25.0                               # follower 25 m/s > ego
    p = _present(open_kin)
    raw_o, inst_o = _extra(open_kin, p, phys)
    raw_c, inst_c = _extra(close_kin, p, phys)

    assert float(_col(raw_o, "tfa")) == pytest.approx(-1.0)
    assert float(_col(raw_o, "tfa_ema")) == pytest.approx(0.3 * -1.0)
    assert float(_col(raw_o, "tfg_rate")) > 0.0            # opening
    assert float(_col(raw_c, "tfg_rate")) < 0.0            # closing
    assert float(inst_o["nb_gap"][0, 0]) == pytest.approx(35.0)
    assert float(_col(raw_o, "nbv")) == pytest.approx(-5.0)
    assert float(_col(raw_o, "nba")) == pytest.approx(0.1)
    assert float(_col(raw_o, "nbh")) == 1.0
    assert float(inst_o["na_gap"][0, 0]) == pytest.approx(65.0)
    assert float(_col(raw_o, "nav")) == pytest.approx(7.0)
    assert float(_col(raw_o, "naa")) == pytest.approx(0.0)
    assert float(_col(raw_o, "nah")) == 1.0
    # Alignment with the gap behind the ego needs braking; maximum acceleration cannot.
    assert 0.0 < float(_col(raw_o, "tta_cur")) < lv5.ALIGN_HORIZON_S
    assert float(_col(raw_o, "tta_max")) == pytest.approx(lv5.ALIGN_HORIZON_S)
    assert float(_col(raw_o, "tle_s")) == 0.0      # already past the acceleration-lane end


def test_missing_neighbours_have_neutral_semantics():
    phys = _phys()
    kin = _scene([(90.0, 2, 15.0)])
    p = _present(kin)
    raw, inst = _extra(kin, p, phys)
    for name in ("tfa", "tfa_ema", "tfg_rate", "tfg_trend", "tlg_rate", "tlg_trend",
                 "nbg", "nbv", "nba", "nbh", "nag", "nav", "naa", "nah",
                 "tta_cur", "tta_max"):
        assert float(_col(raw, name)) == 0.0, name
    assert torch.isfinite(raw).all()
    assert not bool(inst["nb_has"].any()) and not bool(inst["na_has"].any())
    # A leader with no follower: the follower-side features stay neutral, the leader side works.
    kin2 = _scene([(90.0, 2, 15.0), (120.0, 1, 20.0)])
    raw2, _ = _extra(kin2, _present(kin2), phys)
    assert float(_col(raw2, "tfg_rate")) == 0.0 and float(_col(raw2, "nbh")) == 0.0
    assert float(_col(raw2, "tlg_rate")) == pytest.approx(5.0)


def test_gap_trend_uses_the_two_second_ema_state():
    phys = _phys()
    kin = _scene_kin((-1.0, 0.1, 0.2, 0.0))
    present = _present(kin)
    states = kin.unsqueeze(1).repeat(1, 3, 1, 1)
    # Follower falls back: the follower gap opens over time.
    states[:, 1, 2, 0] -= 5.0
    states[:, 2, 2, 0] -= 10.0
    age = torch.zeros(1, 3, *present.shape[1:])
    acc0 = lv5.initial_acc(present, kin, phys, None, None, 1)
    feats, accs = lv5.sequence_features(states, present.unsqueeze(1).repeat(1, 3, 1),
                                        phys, age, acc0, extra=True)
    tfg = feats[0, :, 0, lv5.N_FEATURES + EXTRA.index("tfg_rate")]
    trend = feats[0, :, 0, lv5.N_FEATURES + EXTRA.index("tfg_trend")]
    assert float(tfg[0]) == pytest.approx(5.0)
    assert float(trend[0]) == pytest.approx(0.0, abs=1e-6)      # first sample: no EMA yet
    assert float(trend[1]) > 0.0 and float(trend[2]) > 0.0
    alpha = 1.0 - torch.exp(torch.tensor(-phys.dt_s / lv5.GAP_TREND_TAU_S))
    e2 = accs[0, 1, 0, lv5.ACC_NAMES.index("tfg_e2")]
    gap = 5.0 + 5.0
    expected = (gap - float(e2)) / (lv5.GAP_TREND_TAU_S * float(1.0 - alpha))
    assert float(trend[1]) == pytest.approx(expected, rel=1e-5)


# --------------------------------------------------------------- carry / recurrence
def test_extra_accumulators_carry_across_chunks_and_match_teacher_forcing():
    phys = _phys()
    kin = _scene_kin((-1.0, 0.1, 0.2, 0.0))
    present = _present(kin)
    states = kin.unsqueeze(1).repeat(1, 6, 1, 1)
    states[:, 1:, 2, 0] -= torch.arange(1, 6).view(1, 5).float()
    pres = present.unsqueeze(1).repeat(1, 6, 1)
    age = torch.zeros(1, 6, *present.shape[1:])
    acc0 = lv5.initial_acc(present, kin, phys, None, None, 1)
    feats, accs, inst = lv5.sequence_features(states, pres, phys, age, acc0, extra=True,
                                              return_instant=True)
    acc = acc0
    for t in range(6):
        raw, acc, _ = lv5.step_features(acc, states[:, t], present, phys, age[:, t],
                                        extra=True)
        assert torch.equal(raw, feats[:, t]) and torch.equal(acc, accs[:, t])
    # Shared merge-negotiation definition: tfa_ema(t) = 0.7 tfa_ema(t-1) + 0.3 a_follower(t)
    prev = 0.0
    for t in range(6):
        prev = 0.7 * prev + 0.3 * float(inst["tfa"][0, t, 0])
        got = accs[0, t, 0, lv5.ACC_NAMES.index("tfa_ema")]
        assert float(got) == pytest.approx(prev, rel=1e-6, abs=1e-6)
    # Continuation from a carried state equals the full sequence.
    with torch.no_grad():
        kin2 = torch.stack([states[:, 2], states[:, 3]], 1)
        f2, a2 = lv5.sequence_features(kin2, pres[:, 2:4], phys, age[:, 2:4], accs[:, 1],
                                       extra=True)
    assert torch.equal(f2, feats[:, 2:4]) and torch.equal(a2, accs[:, 2:4])


# --------------------------------------------------------------- checkpoint identity
def _v5_config_model():
    cfg = load_train_config("configs/c4_wm_v5_lane.toml", ["runtime.device=cpu"])
    from wms.cli.assemble import build_model as assemble_model, build_physics
    physics = build_physics(cfg, lambda name: name)
    return assemble_model(cfg, physics), cfg, physics


def test_old_checkpoint_loads_with_extra_inputs_as_exact_identity(tmp_path):
    model, cfg, physics = _v5_config_model()
    with torch.no_grad():
        # Non-identity scaling, so the identity extension is observable.
        model.lane_v5.feature_mu.fill_(0.5)
        model.lane_v5.feature_sd.fill_(2.0)
    path = tmp_path / "wm_s.pt"
    save_checkpoint(path, model, cfg)
    (tmp_path / "physics.json").write_text(json.dumps(asdict(physics)), encoding="utf-8")

    off = load_checkpoint(path).model
    assert off.lane_v5.extra_features is False
    assert off.lane_v5.trunk[0].in_features == lv5.N_FEATURES

    loaded = load_checkpoint(path, model_config={"lane_v5_extra_features": True}).model
    assert loaded.lane_v5.extra_features is True
    assert loaded.lane_v5.trunk[0].in_features == lv5.N_FEATURES_EXTRA
    assert torch.equal(loaded.lane_v5.feature_mu[:lv5.N_FEATURES],
                       model.lane_v5.feature_mu)
    assert torch.equal(loaded.lane_v5.feature_mu[lv5.N_FEATURES:],
                       torch.zeros(len(EXTRA)))
    assert torch.equal(loaded.lane_v5.feature_sd[:lv5.N_FEATURES],
                       model.lane_v5.feature_sd)
    assert torch.equal(loaded.lane_v5.feature_sd[lv5.N_FEATURES:], torch.ones(len(EXTRA)))

    torch.manual_seed(3)
    raw = torch.randn(5, lv5.N_FEATURES)
    extra_raw = torch.cat([raw, torch.randn(5, len(EXTRA)) * 20.0], -1)
    with torch.no_grad():
        a = model.lane_v5(raw)
        b = loaded.lane_v5(extra_raw)
        c = off.lane_v5(raw)
    assert torch.equal(a, c)
    assert torch.equal(a, b)          # zero new weights: arbitrary extra inputs do not act
    assert torch.equal(loaded.lane_v5.heads.weight, model.lane_v5.heads.weight)


# --------------------------------------------------------------- training / rollout
def test_extra_model_rollout_and_teacher_forced_rows():
    from .test_lanefix import _initial
    from .test_wmlane import _batch
    from wms.train.losses.lane_event import lane_v5_batch_rows

    _, flat, settings = _c4([*V5, "model.lane_v5_extra_features=true"])
    model, _ = _model(flat)
    model.physics = _phys()
    model.eval()
    b = _batch()
    with torch.no_grad():
        roll = model.rollout(_initial(b), n_steps=3, cav_teacher=b["cav_teacher"],
                             driver_history=b["driver_hist"],
                             driver_history_mask=b["driver_hist_mask"])
    assert roll["state"]["lane_v5_acc"].shape[-1] == lv5.ACC_F
    assert torch.isfinite(roll["state"]["lane_v5_acc"]).all()

    rows = lane_v5_batch_rows(model, b, settings)
    assert rows["features"].shape[-1] == lv5.N_FEATURES_EXTRA
    assert torch.isfinite(rows["features"]).all()
    with torch.no_grad():
        logits = model.lane_v5(rows["features"])
    assert logits.shape == (rows["features"].shape[0], len(lv5.TRANSITIONS))


# --------------------------------------------------------------- config plumbing
def test_switch_plumbing_and_validation():
    _, flat, settings = _c4([*V5, "model.lane_v5_extra_features=true"])
    assert flat["lane_v5_extra_features"] is True
    assert TrainSettings.from_flat(flat).lane_v5_extra_features is True
    assert ModelSpec.from_mapping(flat).lane_v5_extra_features is True
    model, _ = _model(flat)
    assert model.lane_v5.extra_features is True
    with pytest.raises(Exception):
        _c4(["model.lane_v5_extra_features=true"])
    assert settings.lane_v5_extra_features is True
