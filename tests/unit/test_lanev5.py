"""LANEV5 (2026-10-14): per-transition lane network (``model.lane_model = "hazard_v5"``).

Covers: switch off = no new module / keys / flat keys (the bit-identity golden tests of
test_wmlane / test_lanefix run on the same tree), one head per ROAD transition picked by the
car's lane, the physics mask (never unmasked, also under sampling), no driver embedding /
settings in the network, finite causal features (no future leakage), accumulators carried
across truncated-BPTT chunks and equal to the teacher-forced recomputation, gated decisions
(seeded, closed heads = argmax), end-of-lane forcing, the event loss on every recorded step
(decoupled gradients), the pre-fit file round trip, checkpoint round trip and config plumbing.
"""
from __future__ import annotations

import json
import math
import types
from dataclasses import asdict

import pytest
import torch

from .test_gaprule import _kin
from .test_lanefix import _initial, _phys
from .test_wmlane import _batch, _c4, _model
from wms.config.load import load_train_config
from wms.model import lane_v5 as lv5
from wms.model.checkpoint import load_checkpoint, save_checkpoint
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.spec import ModelSpec
from wms.physics.mobil import MASKED_LOGIT, mobil_merge_logits
from wms.train.losses.compose import build_loss
from wms.train.losses.lane_event import lane_v5_batch_rows
from wms.train.settings import TrainSettings
from wms.train.trainer import WmSTrainer
from wms.train.unroll import UnrollRunner

V5 = ['model.lane_model="hazard_v5"', 'model.lane_gap_rule="speed"', "loss.lane_event.weight=3.0",
      "loss.lane_hazard_decouple=true"]
V5_FLAT_KEYS = ("lane_v5_gates", "lane_v5_head_ap", "lane_v5_pretrain")


def _v5_model(overrides=(), seed=0):
    _, flat, settings = _c4([*V5, *overrides])
    model, _ = _model(flat, seed=seed)
    model.physics = _phys()
    return model, settings


def _rollout(model, b, n=12, **kw):
    with torch.no_grad():
        return model.rollout(_initial(b), n_steps=n, cav_teacher=b["cav_teacher"],
                             driver_history=b["driver_hist"],
                             driver_history_mask=b["driver_hist_mask"], **kw)


def _sc(net, x):
    return net.cap * math.tanh(x / net.cap)


def _single_cars(rows):
    """``[B, 1, 7]``: one car per batch row (s, lane, v)."""
    return torch.cat([_kin([r]) for r in rows], 0)


# --------------------------------------------------------------- switch off
def test_switch_off_builds_no_lane_v5_and_omits_flat_keys():
    for over in ((), ['model.lane_model="hazard"']):
        _, flat, _ = _c4(list(over))
        model, _ = _model(flat)
        assert getattr(model, "lane_v5", None) is None and not model._tracks_lane_v5
        assert not any(k.startswith("lane_v5") for k in model.state_dict())
        assert not any(k in flat for k in V5_FLAT_KEYS)
    b = _batch()
    roll = _rollout(model, b, n=3)
    assert "lane_v5_acc" not in roll["state"]


# --------------------------------------------------------------- heads / mask
def test_one_head_per_road_transition_picked_by_lane():
    phys = _phys()
    kin = _single_cars([(500.0, 1, 20.0), (500.0, 0, 20.0), (30.0, 2, 10.0)])
    present = torch.ones(3, 1, dtype=torch.bool)
    net = lv5.LaneV5Net()
    with torch.no_grad():
        net.heads.bias.copy_(torch.tensor([1.5, -2.5, 3.5]))
    base = mobil_merge_logits(kin, present, phys)
    acc = lv5.initial_acc(present, kin, phys, None, None, 1)
    raw, _, inst = lv5.step_features(acc, kin, present, phys, None)
    logits, head_logit, acts = lv5.lane_logits_v5(net, raw, inst, present, base, phys)
    assert acts.all()
    o, r, m = (_sc(net, x) for x in (1.5, -2.5, 3.5))      # soft cap of the head output
    assert head_logit[:, 0].tolist() == pytest.approx([o, r, m])
    # overtake 1 -> 0, return 0 -> 1, merge 2 -> 1; the current lane stays 0.
    assert logits[0, 0].tolist() == pytest.approx([o, 0.0, MASKED_LOGIT])
    assert logits[1, 0].tolist() == pytest.approx([0.0, r, MASKED_LOGIT])
    assert logits[2, 0].tolist() == pytest.approx([MASKED_LOGIT, m, 0.0])
    # offsets shift exactly their head
    logits2, _, _ = lv5.lane_logits_v5(net, raw, inst, present, base, phys, (1.0, 0.0, -1.0))
    assert float(logits2[0, 0, 0]) == pytest.approx(o + 1)
    assert float(logits2[1, 0, 1]) == pytest.approx(r)
    assert float(logits2[2, 0, 1]) == pytest.approx(m - 1)


def test_mask_is_never_lifted_and_cavs_keep_physics():
    phys = _phys()
    # lane-2 car before the merge zone (ramp entry not allowed), a blocked overtake (car
    # alongside in lane 0) and a CAV with a feasible overtake.
    cases = ((_kin([(-50.0, 2, 10.0)]), True), (_kin([(500.0, 1, 20.0), (502.0, 0, 20.0)]), True),
             (_kin([(500.0, 1, 20.0)]), False))
    net = lv5.LaneV5Net()
    with torch.no_grad():
        net.heads.bias.fill_(9.0)
    for k, learned in cases:
        present = torch.ones(k.shape[:2], dtype=torch.bool)
        base = mobil_merge_logits(k, present, phys)
        acc = lv5.initial_acc(present, k, phys, None, None, 1)
        raw, _, inst = lv5.step_features(acc, k, present, phys, None)
        lg, _, acts = lv5.lane_logits_v5(net, raw, inst, present & learned, base, phys)
        if not learned:
            assert torch.equal(lg, base) and not acts.any()          # CAV: physics only
            assert float(base[0, 0, 0]) > MASKED_LOGIT
        else:
            assert float(lg[0, 0].max()) == 0.0 and not acts[0, 0]   # stays masked
            assert int((lg[0, 0] <= MASKED_LOGIT).sum()) == 2


def test_sampling_with_extreme_heads_never_takes_a_masked_change():
    model, _ = _v5_model(['model.lane_decision="sample"'])
    model.eval()
    with torch.no_grad():
        model.lane_v5.heads.bias.fill_(9.0)
    b = _batch()
    roll = _rollout(model, b, lane_seed=5)
    lanes = torch.cat([b["states"][:, :1, :, 1:4].argmax(-1), roll["lanes"]], 1)
    hdv = (b["present"][:, 0] & ~b["is_cav"][:, 0]).unsqueeze(1)
    lanes = torch.where(hdv, lanes, torch.zeros_like(lanes))
    assert (lanes[:, 1:] - lanes[:, :-1]).abs().max() <= 1
    assert (lanes[:, 1:] != lanes[:, :-1]).any()                     # extreme heads do move
    assert not ((lanes[:, :-1] == 1) & (lanes[:, 1:] == 2)).any()     # never INTO the ramp


def test_no_driver_embedding_or_settings_enter_the_network():
    model, _ = _v5_model()
    assert model.lane_v5.trunk[0].in_features == lv5.N_FEATURES
    assert not any("driver" in name or "embed" in name for name in lv5.FEATURE_NAMES)
    b = _batch()
    state = _initial(b)
    k = state["present"].shape
    p1 = {"v0": torch.full(k, 25.0)}
    p2 = {"v0": torch.full(k, 33.0)}
    e1, e2 = torch.zeros(*k, model.driver_embed_dim), torch.randn(*k, model.driver_embed_dim)
    with torch.no_grad():
        model.lane_v5.heads.weight.normal_()
        a = model.lane_event_logits(state, driver_params=p1, driver_embedding=e1)
        c = model.lane_event_logits(state, driver_params=p2, driver_embedding=e2)
    assert torch.equal(a, c)


# --------------------------------------------------------------- features
def test_features_finite_and_causal():
    phys = _phys()
    b = _batch()
    states, present = b["states"][:, :12].clone(), b["present"][:, :12]
    age = torch.full(present.shape, 999.0)
    acc0 = lv5.initial_acc(present[:, 0], states[:, 0], phys, b["driver_hist"],
                           b["driver_hist_mask"], 1)
    f1, a1 = lv5.sequence_features(states, present, phys, age, acc0)
    assert f1.shape[-1] == lv5.N_FEATURES and torch.isfinite(f1).all()
    assert torch.isfinite(a1).all() and a1.shape[-1] == lv5.ACC_F
    states2 = states.clone()
    states2[:, 7:, :, 0] += 13.0
    states2[:, 7:, :, 4] *= 0.3
    f2, a2 = lv5.sequence_features(states2, present, phys, age, acc0)
    assert torch.equal(f1[:, :7], f2[:, :7]) and torch.equal(a1[:, :7], a2[:, :7])
    assert not torch.equal(f1[:, 7:], f2[:, 7:])


def test_initial_accumulators_from_history():
    phys = _phys()
    kin = _kin([(500.0, 1, 10.0)])
    present = torch.ones(1, 1, dtype=torch.bool)
    h = 6
    hist = torch.zeros(1, 1, h, 9)
    hist[..., 0] = torch.tensor([20.0, 30.0, 12.0, 11.0, 10.0, 10.0]) / 30.0
    hist[..., 3] = 1.0                    # lane 1
    hist[..., 5] = 0.2                    # 20 m gap
    hist[..., 7] = -5.0 / 30.0            # leader 5 m/s slower
    hist[..., 8] = 1.0
    hist[..., 1, 8] = 0.0                 # no leader seen at sample 1: the blocked run restarts
    mask = torch.ones(1, 1, h, dtype=torch.bool)
    acc = lv5.initial_acc(present, kin, phys, hist, mask, 2)
    get = lambda name: float(acc[0, 0, lv5.ACC_NAMES.index(name)])  # noqa: E731
    assert get("v0h") == pytest.approx(30.0)
    # samples 2..4 (leader 7, 6, 5 m/s < v0 30 - 2) are the trailing blocked run -> 3 x 2 steps
    assert get("blk") == pytest.approx(6.0)
    assert get("seen") == 1.0 and get("vs5_0") == -1.0
    none = lv5.initial_acc(present, kin, phys, None, None, 2)
    assert float(none[0, 0, lv5.ACC_NAMES.index("seen")]) == 0.0


def test_accumulators_carried_across_chunks_and_match_teacher_forcing():
    model, _ = _v5_model()
    model.eval()
    b = _batch()
    with torch.no_grad():
        model.lane_v5.heads.bias.fill_(0.5)        # some changes happen
    full = _rollout(model, b, n=12)
    params = {name: full["driver_parameters"][..., i] for i, name in enumerate(PARAMETER_NAMES)}
    kw = dict(driver_history=b["driver_hist"], driver_history_mask=b["driver_hist_mask"],
              driver_params=params, driver_embedding=full["driver_embedding"])
    with torch.no_grad():
        first = model.rollout(_initial(b), n_steps=6, cav_teacher=b["cav_teacher"][:, :7], **kw)
        second = model.rollout(first["state"], n_steps=6, k0=6,
                               cav_teacher=b["cav_teacher"][:, 6:],
                               history=first["history"], history_mask=first["history_mask"], **kw)
    assert torch.equal(full["state"]["lane_v5_acc"], second["state"]["lane_v5_acc"])
    assert torch.equal(full["lanes"][:, 6:], second["lanes"])
    assert torch.equal(full["positions"][:, 6:], second["positions"])
    # step-by-step rollout updates == the teacher-forced sequence recomputation
    phys = model.physics
    states, present = b["states"][:, :8], b["present"][:, :8]
    age = torch.full(present.shape, 999.0)
    acc0 = model.initial_lane_v5_acc(present[:, 0], states[:, 0], b["driver_hist"],
                                     b["driver_hist_mask"])
    feats, accs = lv5.sequence_features(states, present, phys, age, acc0)
    acc = acc0
    for t in range(8):
        raw, acc, _ = lv5.step_features(acc, states[:, t], present[:, t], phys, age[:, t])
        assert torch.allclose(raw, feats[:, t]) and torch.equal(acc, accs[:, t])


# --------------------------------------------------------------- decisions
def test_gated_decisions_seeded_and_closed_heads_are_argmax():
    model, _ = _v5_model(['model.lane_decision="gated"'])
    model.eval()
    b = _batch()
    with torch.no_grad():
        model.lane_v5.heads.bias.copy_(torch.tensor([-0.5, -0.5, -0.5]))   # p ~ 0.38 < 0.5
        model.lane_v5.head_ap.copy_(torch.tensor([0.9, 0.05, -1.0]))       # only overtake open
    assert model.lane_v5_head_ap_effective() == pytest.approx((0.9, 0.05, -1.0))
    assert model._samples_lanes
    r1, r2 = _rollout(model, b, lane_seed=11), _rollout(model, b, lane_seed=11)
    assert torch.equal(r1["lanes"], r2["lanes"])
    lanes0 = b["states"][:, 0, :, 1:4].argmax(-1)
    seq = torch.cat([lanes0.unsqueeze(1), r1["lanes"]], 1)
    learned = b["present"][:, 0] & ~b["is_cav"][:, 0]
    moved = (seq[:, 1:] != seq[:, :-1]) & learned.unsqueeze(1)
    # closed heads (return 0->1, merge 2->1 outside the forced window): argmax -> no change
    assert not (moved & (seq[:, :-1] == 0)).any()
    draws = [_rollout(model, b, lane_seed=s)["lanes"] for s in range(6)]
    assert any(not torch.equal(draws[0], d) for d in draws[1:])
    # config AP overrides the stored one; all heads closed -> identical to argmax
    model.lane_v5_head_ap = (0.0, 0.0, 0.0)
    gated = _rollout(model, b, lane_seed=11)
    model.lane_decision = "argmax"
    assert torch.equal(gated["lanes"], _rollout(model, b)["lanes"])


def test_end_of_lane_merge_is_forced():
    phys = _phys()
    net = lv5.LaneV5Net()
    with torch.no_grad():
        net.heads.bias.fill_(-9.0)
    s1 = phys.merge_zone_s1_m
    kin = _single_cars([(s1 - 2.0, 2, 1.0), (s1 - 20.0, 2, 1.0)])
    present = torch.ones(2, 1, dtype=torch.bool)
    base = mobil_merge_logits(kin, present, phys)
    acc = lv5.initial_acc(present, kin, phys, None, None, 1)
    raw, _, inst = lv5.step_features(acc, kin, present, phys, None)
    lg, _, acts = lv5.lane_logits_v5(net, raw, inst, present, base, phys)
    assert lg[0, 0, 1] >= lv5.FORCED_MERGE_LOGIT and not acts[0, 0]
    assert float(lg[1, 0, 1]) == pytest.approx(_sc(net, -9.0), abs=1e-4) and acts[1, 0]


# --------------------------------------------------------------- loss
def test_event_loss_uses_every_recorded_step_and_is_decoupled():
    model, settings = _v5_model(["loss.lane_event.weight=0.0"])
    model.lane_hazard_detach = True
    model.train()
    runner = UnrollRunner(model, settings, build_loss(settings, physics=model.physics),
                          projection=None, device=torch.device("cpu"))
    runner.run(_batch(), unroll_steps=6, backward=True, truncation_steps=3)
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0
               for p in model.lane_v5.parameters())
    model, settings = _v5_model()
    model.lane_hazard_detach = True
    model.train()
    composer = build_loss(settings, physics=model.physics)
    runner = UnrollRunner(model, settings, composer, projection=None, device=torch.device("cpu"))
    rec = runner.run(_batch(), unroll_steps=6, backward=True, truncation_steps=3)
    term = next(t for t in composer.unroll_terms if getattr(t, "name", "") == "lane_event") \
        if hasattr(composer, "unroll_terms") else None
    assert rec["lane_event"] > 0
    assert sum(float(p.grad.abs().sum()) for p in model.lane_v5.parameters()) > 0
    if term is not None:
        assert term.last_stats["steps"] == 12          # the whole 13-tick window, not 6


# --------------------------------------------------------------- pre-fit / checkpoint
def test_pretrain_tool_round_trip(tmp_path):
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "lanev5_pretrain", pathlib.Path(__file__).resolve().parents[2] / "tools" /
        "lanev5_pretrain.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    model, settings = _v5_model()
    b = _batch()
    b["states"][1, 6:, 2, 1:4] = torch.tensor([1.0, 0.0, 0.0])    # one recorded overtake/return
    rows = lane_v5_batch_rows(model, b, settings)
    assert rows["features"].shape[-1] == lv5.N_FEATURES and rows["stats"]["rows"] > 0
    data = {"features": rows["features"].numpy(), "head": rows["head"].numpy(),
            "label": rows["label"].numpy(), "weight": rows["label"].numpy() * 0 + 1}
    tool.set_scaling(model.lane_v5, data["features"], data["weight"])
    hist = tool.fit(model.lane_v5, data, epochs=2, lr=1e-2, seed=0, log=lambda *_: None)
    assert len(hist) == 2
    with torch.no_grad():
        model.lane_v5.head_ap.copy_(torch.tensor([0.3, 0.2, 0.6]))
    path = tmp_path / "pre.pt"
    lv5.save_pretrain(model.lane_v5, path, {"rows": 1})
    fresh, settings2 = _v5_model([f'model.lane_v5_pretrain="{path.as_posix()}"'], seed=3)
    WmSTrainer._apply_model_overrides(types.SimpleNamespace(settings=settings2, model=fresh))
    x = rows["features"][:50]
    assert torch.equal(fresh.lane_v5(x), model.lane_v5(x))
    assert fresh.lane_v5.head_ap.tolist() == pytest.approx([0.3, 0.2, 0.6])
    assert fresh.lane_v5_pretrain_report["head_ap"] == pytest.approx([0.3, 0.2, 0.6])
    m = tool.head_metrics(tool.predict(model.lane_v5, data["features"], data["head"]),
                          data["label"], data["weight"], data["head"])
    assert set(m) == set(lv5.TRANSITIONS)


def test_metric_helpers():
    import importlib.util
    import pathlib

    import numpy as np
    spec = importlib.util.spec_from_file_location(
        "lanev5_pretrain", pathlib.Path(__file__).resolve().parents[2] / "tools" /
        "lanev5_pretrain.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    y = np.array([0, 0, 1, 1], float)
    w = np.ones(4)
    assert tool.weighted_auc(np.array([0.1, 0.2, 0.8, 0.9]), y, w) == pytest.approx(1.0)
    assert tool.weighted_ap(np.array([0.1, 0.2, 0.8, 0.9]), y, w) == pytest.approx(1.0)
    assert tool.weighted_auc(np.array([0.9, 0.8, 0.2, 0.1]), y, w) == pytest.approx(0.0)
    assert tool.weighted_ap(np.array([0.9, 0.1, 0.8, 0.2]), y, w) == pytest.approx(
        (1 / 2 + 2 / 3) / 2)


def test_checkpoint_round_trip_keeps_v5(tmp_path):
    cfg = load_train_config("configs/c4_wm_v5_lane.toml", ["runtime.device=cpu"])
    from wms.cli.assemble import build_model as assemble_model, build_physics
    physics = build_physics(cfg, lambda n: n)
    model = assemble_model(cfg, physics)
    with torch.no_grad():
        model.lane_v5.feature_mu.fill_(0.5)
        model.lane_v5.head_ap.copy_(torch.tensor([0.3, 0.2, 0.6]))
    path = tmp_path / "wm_s.pt"
    save_checkpoint(path, model, cfg)
    (tmp_path / "physics.json").write_text(json.dumps(asdict(physics)), encoding="utf-8")
    loaded = load_checkpoint(path).model
    assert loaded.lane_model == "hazard_v5" and loaded.lane_decision == "gated"
    assert loaded.lane_hazard is None and loaded.lane_v5_gates == (0.25, 0.10, 0.5)
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


# --------------------------------------------------------------- plumbing
def test_plumbing_and_validation():
    _, flat, settings = _c4([*V5, 'model.lane_decision="gated"', "model.lane_v5_gates=[0.2,0.1,0.4]",
                             "model.lane_v5_head_ap=[0.3,-1.0,0.7]"])
    assert flat["lane_model"] == "hazard_v5" and flat["lane_decision"] == "gated"
    assert tuple(flat["lane_v5_gates"]) == (0.2, 0.1, 0.4)
    spec = ModelSpec.from_mapping(flat)
    assert spec.lane_v5_head_ap == (0.3, -1.0, 0.7)
    model, _ = _model(flat)
    assert model.lane_v5_head_ap_effective() == pytest.approx((0.3, -1.0, 0.7))
    assert TrainSettings.from_flat(flat).lane_v5_gates == (0.2, 0.1, 0.4)
    with pytest.raises(Exception):
        _c4(['model.lane_model="hazard"', 'model.lane_decision="gated"'])
    with pytest.raises(Exception):
        _c4(["model.lane_v5_gates=[0.2,0.1,0.4]"])
    with pytest.raises(ValueError):
        ModelSpec(lane_decision="gated")
    with pytest.raises(ValueError):
        ModelSpec(rollout_mode="hybrid", lane_model="hazard_v5", lane_v5_gates=(0.1, 0.2))


def test_v5_lane_config_differs_from_v4_only_in_the_lane_keys():
    a = load_train_config("configs/c4_wm_v4.toml", []).to_legacy_flat()
    b = load_train_config("configs/c4_wm_v5_lane.toml", []).to_legacy_flat()
    diff = {k: b.get(k) for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"lane_model": "hazard_v5", "lane_decision": "gated",
                    "lane_hazard_features": None, "merge_hazard_zone": None,
                    "lane_event_weight": 10.0, "out_dir": "runs/active/c4_wm_v5_lane"}


def test_fast_feasibility_equals_the_physics_mask():
    from .test_gaprule import fuzz_scene
    from wms.physics.config import PhysicsConfig
    b = _batch()
    scenes = [fuzz_scene(seed=s, b=6, k=40)[:2] for s in range(4)]
    slow = b["states"][:, 3].clone()
    slow[..., 4] *= 0.05
    scenes += [(b["states"][:, 3], b["present"][:, 3]), (slow, b["present"][:, 3])]
    jam = _kin([(60.0, 2, 0.0), (66.0, 1, 0.0), (58.0, 1, 2.0), (10.0, 2, 0.5),
                (14.0, 1, 3.0), (2.0, 1, 3.0), (300.0, 0, 30.0), (290.0, 1, 25.0)])
    scenes.append((jam, torch.ones(1, 8, dtype=torch.bool)))
    checked = 0
    for relax in (False, True):
        for rule in ("learned_discretionary", "mobil", "forced_merge_only"):
            phys = PhysicsConfig(sumo_fidelity=True, lane_rule=rule, lane_gap_rule="speed",
                                 gap_stopped_relax=relax)
            assert lv5.fast_mask_applies(phys)
            for kin, present in scenes:
                present = present.bool()
                inst = lv5.instant_features(kin, present, phys, None)
                base = mobil_merge_logits(kin, present, phys)
                ref = base.gather(-1, inst["tq"].unsqueeze(-1)).squeeze(-1) > MASKED_LOGIT
                assert torch.equal(inst["target_feasible"], ref & present)
                checked += int(ref[present].sum())
    assert checked > 50
    assert not lv5.fast_mask_applies(PhysicsConfig(lane_gap_rule="fixed"))


def test_recorded_rows_fast_and_physics_mask_paths_agree():
    model, settings = _v5_model(["model.gap_stopped_relax=true"])
    model.physics = _phys(gap_stopped_relax=True)
    b = _batch()
    b["states"][:, :, :, 4] *= 0.1                       # slow traffic: masks bind
    b["states"][1, 6:, 2, 1:4] = torch.tensor([1.0, 0.0, 0.0])
    assert model.lane_v5_mask_fn() is None
    fast = lane_v5_batch_rows(model, b, settings)
    model.lane_v5_mask_fn = lambda: model.lane_v5_base   # force the physics-module path
    slow = lane_v5_batch_rows(model, b, settings)
    assert fast["stats"] == slow["stats"]
    assert torch.equal(fast["features"], slow["features"])
    assert torch.equal(fast["label"], slow["label"]) and torch.equal(fast["head"], slow["head"])


def _tool():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "lanev5_pretrain", pathlib.Path(__file__).resolve().parents[2] / "tools" /
        "lanev5_pretrain.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def _toy_rows(n=3000, seed=0):
    import numpy as np
    g = np.random.default_rng(seed)
    x = g.normal(size=(n, lv5.N_FEATURES)).astype(np.float32)
    head = g.integers(0, 3, n)
    y = (g.random(n) < 1 / (1 + np.exp(-(x[:, 0] * 2 - 3)))).astype(np.float32)
    return {"features": x, "head": head, "label": y, "weight": np.ones(n, np.float32)}


def test_dropout_default_off_and_keys_unchanged():
    net = lv5.LaneV5Net()
    keys = list(net.state_dict())
    x = torch.randn(64, lv5.N_FEATURES)
    net.train()
    ref = net(x)
    with torch.no_grad():
        net.heads.weight.normal_()
    ref = net(x)
    assert net.dropout == 0.0 and torch.equal(net(x), ref)        # off: deterministic
    net.dropout = 0.5
    assert not torch.equal(net(x), ref)                           # train mode: dropped
    net.eval()
    assert torch.equal(net(x), ref)                               # eval: never dropped
    assert list(net.state_dict()) == keys


def test_episode_split_keeps_episodes_together():
    tool = _tool()
    keys = [("world", f"w{i // 4}") for i in range(200)]
    fit_idx, es_idx = tool.split_by_episode(keys, 0.2, 0)
    assert sorted(fit_idx + es_idx) == list(range(200)) and fit_idx and es_idx
    assert not {keys[i] for i in fit_idx} & {keys[i] for i in es_idx}
    assert tool.split_by_episode(keys, 0.2, 0) == (fit_idx, es_idx)
    assert tool.split_by_episode(keys, 0.2, 1) != (fit_idx, es_idx)


def test_early_stopping_restores_best_and_budget_stops():
    import time as _time
    tool = _tool()
    data, early = _toy_rows(seed=0), _toy_rows(n=800, seed=1)
    net = lv5.LaneV5Net()
    tool.set_scaling(net, data["features"], data["weight"])
    rep = {}
    tool.fit(net, data, epochs=30, lr=3e-2, seed=0, early=early, patience=2, minibatch=256,
             log=lambda *_: None, report=rep)
    assert rep["optimizer"] == "Adam" and rep["best_epoch"] <= rep["stop_epoch"]
    if rep["stopped_by"] == "early_stopping":
        assert rep["stop_epoch"] - rep["best_epoch"] == 2
    assert rep["es_criterion"] == "normalised"
    assert tool.summed_head_logloss(net, early, "normalised") == pytest.approx(
        rep["best_es_logloss"], rel=1e-6)
    assert min(rep["es_logloss"]) == pytest.approx(rep["best_es_logloss"])
    assert net.dropout == 0.0 and not net.training
    rep2 = {}
    tool.fit(lv5.LaneV5Net(), data, epochs=5, lr=1e-2, seed=0, weight_decay=0.01, dropout=0.2,
             deadline=_time.time() - 1, log=lambda *_: None, report=rep2)
    assert rep2["stop_epoch"] == 1 and rep2["stopped_by"] == "max_minutes"
    assert rep2["optimizer"] == "AdamW" and rep2["best_epoch"] is None


def test_normalised_criterion_counts_heads_equally():
    import numpy as np
    tool = _tool()
    data = _toy_rows(n=600, seed=2)
    net = lv5.LaneV5Net()
    with torch.no_grad():
        net.heads.weight.zero_()
    # constant predictor at each head's own rate -> every head scores exactly 1
    rates = [float(data["label"][data["head"] == h].mean()) for h in range(3)]
    with torch.no_grad():
        net.heads.bias.copy_(torch.tensor([float(np.log(r / (1 - r))) for r in rates]))
    assert tool.summed_head_logloss(net, data, "normalised") == pytest.approx(3.0, rel=1e-3)
    assert tool.summed_head_logloss(net, data, "raw") != pytest.approx(3.0, rel=1e-3)
    with pytest.raises(ValueError):
        tool.summed_head_logloss(net, data, "bogus")
