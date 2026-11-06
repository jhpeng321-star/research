"""V10LANEOPT (2026-10-19): staged mainline lane changes (``physics.mainline_lane_mode``).

Covers the switch plumbing (default 'free' = bit-identical), the courtesy rollout restriction
(merges fully learned; only the learned lane-1 -> lane-0 overtake of HDVs inside the approach
zone upstream of the merge draws; returns and outside-zone overtakes are frozen at the physics
prior) and the v10loop lane-mode curriculum (courtesy for iterations 1-2, switch to free only on
a lane_calib within +-20 % of SUMO, robust to a missing file).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from .test_gaprule import _kin
from .test_lanev5 import V5, _v5_model
from .test_wmlane import _c4
from wms.config.load import load_train_config
from wms.config.train import TrainConfig
from wms.model import lane_v5 as lv5
from wms.physics.config import MAINLINE_LANE_MODES, PhysicsConfig
from wms.physics.mobil import MASKED_LOGIT
from wms.train.settings import TrainSettings

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("loop_v10_laneopt", REPO / "scripts" / "c4" / "loop_v10.py")
LV = importlib.util.module_from_spec(_spec)
sys.modules["loop_v10_laneopt"] = LV
_spec.loader.exec_module(LV)
LOOP_CFG = LV.load_config(REPO / "configs" / "c4_dreamer_loop_v10loop.json")
_smoke_spec = importlib.util.spec_from_file_location(
    "laneopt_smoke", REPO / "scripts" / "c4" / "laneopt_smoke.py")
SM = importlib.util.module_from_spec(_smoke_spec)
sys.modules["laneopt_smoke"] = SM
_smoke_spec.loader.exec_module(SM)


def _learnable(mode: str, zone: float = 300.0) -> PhysicsConfig:
    return PhysicsConfig(sumo_fidelity=True, lane_rule="learned_discretionary",
                         lane_gap_rule="speed", lane_learnable=True,
                         mainline_lane_mode=mode, mainline_courtesy_zone_m=zone)


def _courtesy_scene(model, zone: float = 300.0):
    """One lane-1 HDV inside / outside the zone, a lane-0 HDV and an in-zone ramp car."""
    end = model.physics.merge_zone_s1_m
    kin = _kin([(end - 100.0, 1, 12.0),    # inside the courtesy zone -> overtake allowed
                (end - (zone + 50.0), 1, 12.0),   # outside the zone -> frozen
                (end - 500.0, 0, 12.0),    # return 0 -> 1 -> always frozen (no body conflict)
                (end - 30.0, 2, 8.0)])     # ramp car in the merge zone -> merge allowed
    present = torch.ones(1, 4, dtype=torch.bool)
    learned = torch.ones(1, 4, dtype=torch.bool)
    return kin, present, learned


def _forward(model, kin, present, learned):
    lane_base = model.lane_v5_base(kin, present)
    return model._lane_v5_forward(kin, present, learned, lane_base, None, None, None, None)


# --------------------------------------------------------------- switch plumbing
def test_switch_defaults_off_free_and_plumbing():
    assert TrainConfig().physics.mainline_lane_mode == "free"
    assert TrainConfig().physics.mainline_courtesy_zone_m == 300.0
    assert "mainline_lane_mode" not in TrainConfig().to_legacy_flat()
    assert "mainline_courtesy_zone_m" not in TrainConfig().to_legacy_flat()
    assert TrainSettings().mainline_lane_mode == "free"
    assert MAINLINE_LANE_MODES == ("free", "courtesy")

    _, flat, settings = _c4([*V5, "physics.lane_learnable=true",
                             "physics.mainline_lane_mode=courtesy",
                             "physics.mainline_courtesy_zone_m=250"])
    assert flat["mainline_lane_mode"] == "courtesy"
    assert flat["mainline_courtesy_zone_m"] == 250.0
    assert settings.mainline_lane_mode == "courtesy"
    assert load_train_config("c4_wm", ["physics.mainline_lane_mode=courtesy"]
                             ).physics.mainline_lane_mode == "courtesy"
    # override + inference paths accept the mode; the mode unknown value is refused.
    cfg = load_train_config("c4_wm", ['physics.overrides={mainline_lane_mode="courtesy"}'])
    assert cfg.physics.overrides["mainline_lane_mode"] == "courtesy"
    cfg = load_train_config("c4_wm", ['physics.inference={mainline_lane_mode="courtesy"}'])
    assert cfg.physics.inference["mainline_lane_mode"] == "courtesy"
    with pytest.raises(ValueError):
        PhysicsConfig(mainline_lane_mode="staged")
    with pytest.raises(ValueError):
        PhysicsConfig(mainline_courtesy_zone_m=-1.0)


def test_free_mode_is_bit_identical():
    model, _ = _v5_model()
    default = PhysicsConfig(sumo_fidelity=True, lane_rule="learned_discretionary",
                            lane_gap_rule="speed", lane_learnable=True)   # default 'free'
    explicit = _learnable("free")
    kin, present, learned = _courtesy_scene(model)
    model.physics = default
    ref_logits, _, ref_sample, _ = _forward(model, kin, present, learned)
    model.physics = explicit
    got_logits, _, got_sample, _ = _forward(model, kin, present, learned)
    assert torch.equal(ref_logits, got_logits)
    assert torch.equal(ref_sample, got_sample)
    assert bool(got_sample.all())


# --------------------------------------------------------------- courtesy restriction
def test_courtesy_freezes_the_mainline_but_keeps_merges():
    model, _ = _v5_model()
    with torch.no_grad():
        model.lane_v5.heads.bias.fill_(9.0)     # strong learned drive to change
    model.physics = _learnable("courtesy")
    kin, present, learned = _courtesy_scene(model)
    logits, _, sample, _ = _forward(model, kin, present, learned)
    inside, outside, ret, ramp = (0, 1, 2, 3)
    # only the in-zone overtake and the ramp merge draw
    assert bool(sample[0, inside]) and bool(sample[0, ramp])
    assert not bool(sample[0, outside]) and not bool(sample[0, ret])
    end = model.physics.merge_zone_s1_m
    # blocked mainline targets are masked; allowed heads keep their learned logit
    logits = logits.detach()
    assert float(logits[0, outside, 0]) <= MASKED_LOGIT + 1e-6
    assert float(logits[0, inside, 0]) > MASKED_LOGIT + 1e-6
    assert float(logits[0, ret, 1]) <= MASKED_LOGIT + 1e-6
    assert float(logits[0, ramp, 1]) > MASKED_LOGIT + 1e-6
    assert abs(float(kin[0, inside, 0]) - (end - 100.0)) < 1e-4


def test_courtesy_zone_is_configurable_and_cav_rows_never_draw():
    model, _ = _v5_model()
    with torch.no_grad():
        model.lane_v5.heads.bias.fill_(9.0)
    model.physics = _learnable("courtesy", zone=50.0)
    kin, present, _ = _courtesy_scene(model, zone=50.0)
    learned = torch.ones(1, 4, dtype=torch.bool)
    logits, _, sample, _ = _forward(model, kin, present, learned)
    # the 100 m-upstream car is now OUTSIDE the 50 m zone -> frozen
    assert not bool(sample[0, 0]) and float(logits.detach()[0, 0, 0]) <= MASKED_LOGIT + 1e-6
    learned_cav = learned.clone()
    learned_cav[0, 0] = False                      # a CAV slot keeps the physics prior
    _, _, sample_cav, _ = _forward(model, kin, present, learned_cav)
    assert not bool(sample_cav[0, 0])


def test_recorded_rows_training_is_mode_independent():
    """The lane-head loss trains all heads on all-risk rows in BOTH modes (only the draw is
    restricted), so the free stage starts calibrated."""
    model, _ = _v5_model(["loss.lane_event.weight=0.0"])
    from .test_wmlane import _batch
    b = _batch()
    b["states"][1, 6:, 2, 1:4] = torch.tensor([1.0, 0.0, 0.0])   # a recorded change
    free = model.physics
    model.physics = _learnable("free")
    rows_free = lv5.recorded_rows(b["states"], b["present"], b["is_cav"], model.physics, None,
                                  torch.full((b["states"].shape[0], b["states"].shape[1] - 1,
                                              b["states"].shape[2]), 999.0),
                                  model.initial_lane_v5_acc(b["present"][:, 0], b["states"][:, 0],
                                                            None, None))
    model.physics = _learnable("courtesy")
    rows_cour = lv5.recorded_rows(b["states"], b["present"], b["is_cav"], model.physics, None,
                                  torch.full((b["states"].shape[0], b["states"].shape[1] - 1,
                                              b["states"].shape[2]), 999.0),
                                  model.initial_lane_v5_acc(b["present"][:, 0], b["states"][:, 0],
                                                            None, None))
    assert rows_free["stats"] == rows_cour["stats"]
    assert rows_free["head"].tolist() == rows_cour["head"].tolist()
    assert rows_free["label"].tolist() == rows_cour["label"].tolist()


# --------------------------------------------------------------- loop curriculum
def _calib(heads: dict):
    return {"schema": "c4_lane_calib/1", "starts": 4,
            "heads": {name: dict(values) for name, values in heads.items()}}


def test_curriculum_defaults_and_plan_pass_through(tmp_path):
    cur = LV.lane_mode_curriculum(LOOP_CFG)
    assert cur["enabled"] is True and cur["courtesy_iterations"] == [1, 2]
    assert cur["tolerance"] == 0.2 and cur["gating_heads"] == ["overtake", "return"]
    assert LV.lane_mode_for_iteration(LOOP_CFG, 0)["mode"] == "free"
    assert LV.lane_mode_for_iteration(LOOP_CFG, 1)["mode"] == "courtesy"
    assert LV.lane_mode_for_iteration(LOOP_CFG, 2)["mode"] == "courtesy"
    # no calibration on disk -> the post-courtesy iterations stay 'courtesy'
    assert LV.lane_mode_for_iteration(LOOP_CFG, 3)["mode"] == "courtesy"
    paths = LV._paths(tmp_path, 1, LOOP_CFG)
    stages = LV.plan_iteration(LOOP_CFG, 1, paths, LV.allocate_seeds(LOOP_CFG, n=2), 200)
    wm = next(s for s in stages if s["name"] == "wm_finetune")
    assert "physics.mainline_lane_mode=courtesy" in wm["cmd"]
    for key in LOOP_CFG["wm"]["extra_set"]:
        assert key in wm["cmd"], key


def test_curriculum_switches_to_free_only_within_tolerance():
    within = _calib({"overtake": {"wm": 110.0, "sumo": 100.0},
                     "return": {"wm": 90.0, "sumo": 100.0},
                     "merge": {"wm": 500.0, "sumo": 100.0}})
    outside = _calib({"overtake": {"wm": 130.0, "sumo": 100.0},
                      "return": {"wm": 100.0, "sumo": 100.0}})
    assert LV.lane_mode_for_iteration(LOOP_CFG, 3, within)["mode"] == "free"
    assert LV.lane_mode_for_iteration(LOOP_CFG, 3, outside)["mode"] == "courtesy"
    # only the gating heads matter: a wild merge count does not block the switch
    assert LV.lane_mode_for_iteration(LOOP_CFG, 3, within)["calib"]["calibrated"] is True
    # explicit rel_error / ratio fields are honoured
    rel = _calib({"overtake": {"rel_error": 0.2}, "return": {"ratio": 0.8}})
    assert LV.lane_mode_for_iteration(LOOP_CFG, 3, rel)["mode"] == "free"
    bad = _calib({"overtake": {"rel_error": 0.21}, "return": {"rel_error": 0.0}})
    assert LV.lane_mode_for_iteration(LOOP_CFG, 3, bad)["mode"] == "courtesy"
    # a malformed / missing record for a gating head stays conservative
    assert LV.lane_mode_for_iteration(LOOP_CFG, 3, {"heads": {"overtake": {}}})["mode"] == "courtesy"
    assert LV.lane_mode_for_iteration(LOOP_CFG, 3, None)["mode"] == "courtesy"


def test_curriculum_reads_the_wm_run_lane_calib(tmp_path):
    paths = LV._paths(tmp_path, 3, LOOP_CFG)
    calib_path = LV.lane_calib_path(paths, LOOP_CFG)
    assert calib_path.name == "lane_calib.json"
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    calib_path.write_text(json.dumps(_calib({"overtake": {"wm": 105.0, "sumo": 100.0},
                                             "return": {"wm": 95.0, "sumo": 100.0}})),
                          encoding="utf-8")
    assert LV.read_lane_calib(calib_path)["heads"]["overtake"]["wm"] == 105.0
    mode = LV.lane_mode_for_iteration(LOOP_CFG, 3, LV.read_lane_calib(calib_path))
    assert mode["mode"] == "free"
    # malformed JSON -> None (robust), and the plan keeps the courtesy default
    calib_path.write_text("{not json", encoding="utf-8")
    assert LV.read_lane_calib(calib_path) is None
    stages = LV.plan_iteration(LOOP_CFG, 3, paths, LV.allocate_seeds(LOOP_CFG, n=2), 200)
    wm = next(s for s in stages if s["name"] == "wm_finetune")
    assert "physics.mainline_lane_mode=courtesy" in wm["cmd"]


def test_curriculum_disabled_keeps_current_behaviour():
    cfg = json.loads(json.dumps(LOOP_CFG))
    cfg["lane_mode_curriculum"]["enabled"] = False
    assert LV.lane_mode_for_iteration(cfg, 1)["mode"] == "free"
    assert LV.lane_mode_for_iteration(cfg, 3, None)["mode"] == "free"
    paths = LV._paths(Path("."), 1, cfg)
    stages = LV.plan_iteration(cfg, 1, paths, LV.allocate_seeds(cfg, n=2), 200)
    wm = next(s for s in stages if s["name"] == "wm_finetune")
    assert "physics.mainline_lane_mode=free" in wm["cmd"]


# --------------------------------------------------------------- smoke script
def test_smoke_script_restricts_courtesy_end_to_end():
    model = SM.build_smoke_model("configs/c4_wm_v10loop_iter0.toml", bias=2.0)
    batch = SM.smoke_batch(2, 15, 20, seed=11)
    free = SM.run_arm(model, batch, "free", 20, 1234)
    courtesy = SM.run_arm(model, batch, "courtesy", 20, 1234)
    assert courtesy["return"] == 0 and courtesy["overtake"] == 0
    assert courtesy["merge"] >= 1 and courtesy["merge"] == free["merge"]   # merges kept
    assert courtesy["courtesy"] >= 1                                        # courtesy kept
    assert free["return"] >= 1 and free["overtake"] >= 1                   # disabled in courtesy
    assert free["overtake_total"] > courtesy["overtake_total"]
