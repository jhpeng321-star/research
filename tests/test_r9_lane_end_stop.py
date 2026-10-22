"""R9LANE: ``physics.ramp_lane_end_stop`` - SUMO never moves a car past its lane end."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from wms.cli.assemble import build_physics
from wms.config.load import load_train_config
from wms.model import ModelSpec, WmSModel
from wms.model import merge_pivot as pivot
from wms.physics.config import PhysicsConfig
from wms.physics.state import state_kin

ROOT = Path(__file__).resolve().parents[1]
END = PhysicsConfig().merge_zone_s1_m


def make(flag):
    torch.manual_seed(19)
    physics = replace(PhysicsConfig(), ramp_lane_end_stop=flag)
    return WmSModel(ModelSpec(rollout_mode="driver_params", lane_model="hazard_v5",
                              lane_decision="sample", merge_decision="first_passage",
                              merge_st_width=1.0), physics=physics).eval()


def scene(ego_s, ego_v, ego_lane=2, others=(END - 60.0, END - 90.0)):
    s = torch.tensor([[ego_s, *others]])
    lanes = torch.tensor([[ego_lane, 1, 1]])
    return {"s_m": s, "v_mps": torch.tensor([[ego_v, 5.0, 5.0]]), "a_mps2": torch.zeros_like(s),
            "length_m": torch.full_like(s, 5.0), "lane": lanes,
            "present": torch.ones_like(s, dtype=torch.bool),
            "is_cav": torch.zeros_like(s, dtype=torch.bool)}


def forward_with(accel):
    def fixed(s, **kwargs):
        lane = s["lane"]
        logits = torch.full((*lane.shape, 3), -20.0)
        logits.scatter_(-1, lane.unsqueeze(-1), 20.0)
        return {"accel_mps2": torch.full_like(s["s_m"], accel), "lane_logits": logits,
                "lane_feasible": torch.ones_like(logits, dtype=torch.bool),
                "hdv_mask": s["present"] & ~s["is_cav"],
                "driver_parameters": torch.ones((*lane.shape, 9)),
                "lane_age": torch.zeros_like(lane), "lane_v5_acc": kwargs.get("lane_v5_acc"),
                "lane_sample": torch.zeros_like(s["present"])}
    return fixed


def step(m, s, accel=0.0, cross=False):
    # deterministic willingness: +inf latent -> the merge happens iff the car is eligible;
    # -inf -> the car never merges (it stays on the ramp lane)
    pend = [patch.object(m, "forward", side_effect=forward_with(accel)),
            patch.object(m.merge_pivot, "forward", return_value=torch.full_like(
                s["s_m"], float("inf") if cross else -float("inf")))]
    for p in pend:
        p.start()
    try:
        return m.step(s, lane_uniform=torch.full_like(s["s_m"], 0.5))
    finally:
        for p in pend:
            p.stop()


def helper(s, v, lane=2, accel=0.0, shift=None, present=True, dt=0.5):
    state = scene(s, v, lane)
    state["present"][0, 0] = present
    kin = state_kin(state)
    moved = pivot.advance_longitudinal(kin, torch.full_like(state["s_m"], accel), dt)
    out = pivot.ramp_lane_end_stop(kin, moved[..., 0], moved[..., 4], state["present"],
                                   PhysicsConfig(), shift)
    return moved, out


def test_end_is_the_network_lane_end_not_a_typed_constant():
    assert END == pytest.approx(75.44)


def test_stopped_car_at_the_lane_end_stays():
    moved, (s, v, stopped) = helper(END, 0.0, accel=1.0)
    assert moved[0, 0, 0] > END  # the unconstrained update creeps past the end
    assert bool(stopped[0, 0]) and s[0, 0] == torch.tensor(END) and v[0, 0] == 0.0


def test_approaching_car_stops_exactly_at_the_end():
    moved, (s, v, stopped) = helper(END - 1.0, 5.0)
    assert moved[0, 0, 0] > END
    assert bool(stopped[0, 0]) and s[0, 0].item() == torch.tensor(END).item() and v[0, 0] == 0.0


def test_cars_that_do_not_reach_the_end_other_lanes_and_absent_slots_are_untouched():
    moved, (s, v, stopped) = helper(END - 10.0, 5.0)
    assert not stopped.any() and torch.equal(s, moved[..., 0]) and torch.equal(v, moved[..., 4])
    moved, (s, v, stopped) = helper(END - 1.0, 5.0, lane=1)  # mainline car past x=END is normal
    assert not stopped.any() and torch.equal(s, moved[..., 0])
    moved, (s, v, stopped) = helper(END - 1.0, 5.0, present=False)
    assert not stopped.any() and torch.equal(s, moved[..., 0])


def test_route_coordinate_shift_is_honoured():
    moved, (s, v, stopped) = helper(END - 3.0 - 1.0, 5.0, shift=torch.full((1, 3), 3.0))
    assert bool(stopped[0, 0]) and (s[0, 0] + 3.0).item() == pytest.approx(END, abs=1e-5)


def test_flag_off_is_bit_identical_to_the_unconstrained_step():
    assert PhysicsConfig().ramp_lane_end_stop is False
    for pos, v in ((END, 0.0), (END - 1.0, 5.0), (END - 30.0, 8.0)):
        off = step(make(False), scene(pos, v))
        base_model = make(False)
        assert not base_model.physics.ramp_lane_end_stop
        ref = step(base_model, scene(pos, v))
        for key, value in off.items():
            if torch.is_tensor(value):
                assert torch.equal(value, ref[key]), key
    off = step(make(False), scene(END - 1.0, 5.0))
    assert off["next_s_m"][0, 0] > END  # legacy behaviour: the car is carried past the end


def test_step_holds_a_staying_car_at_the_end_and_does_not_change_other_slots():
    on, off = step(make(True), scene(END - 1.0, 5.0)), step(make(False), scene(END - 1.0, 5.0))
    assert on["next_s_m"][0, 0].item() == torch.tensor(END).item() and on["next_v_mps"][0, 0] == 0.0
    assert on["next_lane"][0, 0] == 2
    assert torch.equal(on["next_s_m"][0, 1:], off["next_s_m"][0, 1:])
    assert torch.equal(on["next_v_mps"][0, 1:], off["next_v_mps"][0, 1:])
    stopped = step(make(True), scene(END, 0.0), accel=1.0)
    assert stopped["next_s_m"][0, 0].item() == torch.tensor(END).item()
    assert stopped["next_v_mps"][0, 0] == 0.0


def test_merging_car_is_decided_at_the_end_but_keeps_its_free_motion():
    free = step(make(False), scene(END - 1.0, 5.0), cross=True)
    assert free["next_lane"][0, 0] == 2  # legacy: carried outside the zone, no merge possible
    on = step(make(True), scene(END - 1.0, 5.0), cross=True)
    assert on["next_lane"][0, 0] == 1  # held at the end it is inside the zone and merges
    assert on["next_s_m"][0, 0].item() == pytest.approx(free["next_s_m"][0, 0].item())
    assert on["next_v_mps"][0, 0].item() == pytest.approx(free["next_v_mps"][0, 0].item())


def test_decision_geometry_sees_the_pinned_front_and_is_idempotent():
    m_on, m_off = make(True), make(False)
    s = scene(END - 1.0, 5.0)
    accel = torch.zeros_like(s["s_m"])
    on = m_on.pivot_decision_geometry(s, accel)
    off = m_off.pivot_decision_geometry(s, accel)
    assert bool(on["ramp"][0, 0]) and not bool(off["ramp"][0, 0])
    assert on["s"][0, 0].item() == torch.tensor(END).item()
    again = m_on.pivot_decision_geometry(s, accel, next_s=on["s"], next_v=on["v"])
    assert torch.equal(again["s"], on["s"]) and torch.equal(again["eligible"], on["eligible"])


def test_round9_preset_pins_the_flag_and_the_fitter_physics_json():
    physics = build_physics(load_train_config(str(ROOT / "configs/c4_wm_v10round9.toml"), []))
    assert physics.ramp_lane_end_stop is True
    assert physics.merge_zone_s1_m == pytest.approx(75.44)
    stored = json.loads((ROOT / "configs/r9_physics_round9.json").read_text(encoding="utf-8"))
    assert stored == asdict(physics) and stored["ramp_lane_end_stop"] is True
    previous = build_physics(load_train_config(str(ROOT / "configs/c4_wm_v10round8.toml"), []))
    assert previous.ramp_lane_end_stop is False


def test_flag_must_be_boolean():
    with pytest.raises(ValueError):
        PhysicsConfig(ramp_lane_end_stop=1)
