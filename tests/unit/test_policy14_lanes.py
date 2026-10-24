"""Secure-gap lane vetoes, CPU/synthetic states only; no model, data or SUMO access."""
from dataclasses import FrozenInstanceError, asdict, replace
import sys
from types import ModuleType

import pytest
import torch

from wms.control.lanes import (
    LANE_LEFT, LANE_RIGHT, LANE_STAY, LaneRules, apply_lane_choices, lane_feasibility,
)

GS_HOOK = "wms.control.lanes:secure_gap_gs"


def state(s, lane, v=None, cav=None):
    k = len(s)
    return {"s_m": torch.tensor([s], dtype=torch.float32),
            "v_mps": torch.tensor([v if v is not None else [10.0] * k]),
            "lane": torch.tensor([lane]), "present": torch.ones(1, k, dtype=torch.bool),
            "is_cav": torch.tensor([cav if cav is not None else [True] * k]),
            "length_m": torch.full((1, k), 5.0)}


def inferred(x, headway, braking):
    return {**x, "secure_gap_T_s": torch.tensor([headway]),
            "secure_gap_b_mps2": torch.tensor([braking])}


@pytest.fixture
def register_hook(monkeypatch):
    """Expose a callable by the same lazy module:function API as an external provider."""
    module = ModuleType("policy14_test_secure_gap")
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def register(hook):
        module.veto = hook
        return LaneRules(secure_gap_hook=f"{module.__name__}:veto")

    return register


def test_hook_serialization_and_legacy_numeric_mapping():
    rules = LaneRules(secure_gap_hook="unmerged_13e.module:gap_veto")
    assert LaneRules.from_mapping(asdict(rules)) == rules  # no import at construction/load
    assert LaneRules.from_mapping({"secure_gap_hook": GS_HOOK}) == replace(
        LaneRules(), secure_gap_hook=GS_HOOK)
    legacy = LaneRules.from_mapping({"gap_base_m": "6", "gap_time_s": 1,
                                     "closing_time_s": 1, "secure_gap_hook": GS_HOOK})
    assert legacy.rear == (6.0, 1.0, 1.0)
    assert LaneRules.from_mapping(asdict(legacy)) == legacy
    assert LaneRules.from_mapping({"gap_base_m": "6"}).secure_gap_hook == ""
    with pytest.raises(FrozenInstanceError):
        rules.secure_gap_hook = ""


@pytest.mark.parametrize("hook", [None, 1, True, "module", "module:", ":function",
                                  "module:func:extra", "module:func()"])
def test_hook_field_rejects_non_string_or_malformed_paths(hook):
    with pytest.raises(ValueError, match="secure_gap_hook"):
        LaneRules.from_mapping({"secure_gap_hook": hook})


def test_empty_hook_preserves_default_mask_and_execution_without_import(monkeypatch):
    import wms.control.lanes as lanes

    def no_import(*args):
        pytest.fail("empty hook must not import a provider")

    monkeypatch.setattr(lanes.importlib, "import_module", no_import)
    x = state([-500.0, -800.0, -100.0, 30.0], [0, 1, 2, 2])
    mask = lane_feasibility(x)
    assert mask.tolist() == [[[True, False, True], [True, True, False],
                              [True, False, False], [True, True, False]]]
    explicit = LaneRules.from_mapping(asdict(LaneRules()))
    assert torch.equal(mask, lane_feasibility(x, rules=explicit))
    new_lane, changed = apply_lane_choices(x, torch.tensor([[2, 1, 1, 1]]), mask)
    assert new_lane.tolist() == [[1, 0, 2, 1]]
    assert changed.tolist() == [[True, True, False, True]]


def test_hook_masks_both_directions_but_stay_remains_feasible(register_hook):
    x = state([-500.0, -800.0], [0, 1])
    rules = register_hook(lambda state, other_lane, target: torch.zeros_like(target, dtype=torch.bool))
    assert lane_feasibility(x)[0, 0, LANE_RIGHT]
    assert lane_feasibility(x)[0, 1, LANE_LEFT]
    mask = lane_feasibility(x, rules=rules)
    assert mask.tolist() == [[[True, False, False], [True, False, False]]]
    # Even a stale mask from before the veto cannot authorize a change at execution.
    lane, changed = apply_lane_choices(x, torch.tensor([[2, 1]]), lane_feasibility(x), rules)
    assert torch.equal(lane, x["lane"]) and not changed.any()


def test_hook_cannot_unmask_legacy_geometry_dwell_or_gap_veto(register_hook):
    rules = register_hook(lambda state, other_lane, target: torch.ones_like(target, dtype=torch.bool))
    x = state([-500.0, -499.0, -100.0, -800.0], [0, 1, 2, 1], cav=[True, False, True, True])
    age = torch.tensor([[3.0, 3.0, 3.0, 0.0]])
    assert torch.equal(lane_feasibility(x, age, rules), lane_feasibility(x, age))
    # Legacy simultaneous conflict remains blocked even when the hook allows all.
    x = state([-500.0, -492.0], [0, 0])
    mask = lane_feasibility(x, rules=rules)
    assert mask[..., LANE_RIGHT].all()
    lane, changed = apply_lane_choices(x, torch.full((1, 2), LANE_RIGHT), mask, rules)
    assert torch.equal(lane, x["lane"]) and not changed.any()


def test_hook_rechecks_proposed_lanes_and_respects_acting_mask(register_hook):
    x = state([-500.0, -450.0], [0, 0])
    calls = []

    def veto(state, other_lane, target):
        calls.append((other_lane.clone(), target.clone()))
        # Each alone is safe; both occupying target lane 1 are vetoed.
        return ~((other_lane == 1).all(-1, keepdim=True).expand_as(target))

    rules = register_hook(veto)
    mask = lane_feasibility(x, rules=rules)
    assert mask[..., LANE_RIGHT].all()
    lane, changed = apply_lane_choices(x, torch.full((1, 2), LANE_RIGHT), mask, rules)
    assert torch.equal(lane, x["lane"]) and not changed.any()
    assert calls[-1][0].tolist() == [[1, 1]]
    assert calls[-1][1].tolist() == [[1, 1]]
    lane, changed = apply_lane_choices(x, torch.full((1, 2), LANE_RIGHT), mask, rules,
                                       acting=torch.tensor([[True, False]]))
    assert lane.tolist() == [[1, 0]] and changed.tolist() == [[True, False]]
    assert calls[-1][0].tolist() == [[1, 0]]


@pytest.mark.parametrize("result", [True, torch.tensor([[1.0]]), torch.tensor([True]),
                                    torch.ones(1, 2, dtype=torch.bool)])
def test_invalid_hook_outputs_fail_instead_of_broadcasting(register_hook, result):
    rules = register_hook(lambda state, other_lane, target: result)
    with pytest.raises(ValueError, match=r"bool \[B,K\]"):
        lane_feasibility(state([-500.0], [0]), rules=rules)


def test_missing_or_noncallable_provider_fails_only_when_enabled(register_hook):
    x = state([-500.0], [0])
    with pytest.raises(ValueError, match="unavailable"):
        lane_feasibility(x, rules=LaneRules(secure_gap_hook="unmerged_policy14_provider:veto"))
    with pytest.raises(ValueError, match="not callable"):
        lane_feasibility(x, rules=register_hook(1))


@pytest.mark.parametrize("ego_lane,choice", [(0, LANE_RIGHT), (1, LANE_LEFT)])
@pytest.mark.parametrize("side", ["front", "rear"])
def test_gs_uses_each_followers_inferred_T_and_b_for_both_directions(ego_lane, choice, side):
    # Bumper gap 105 m. Legacy closing rule requires <=85.5 m; gs needs
    # 2*15 + (15^2/(2*1) - 5^2/(2*4)) = 139.375 m for the new follower.
    if side == "front":
        x = inferred(state([-500.0, -390.0], [ego_lane, 1 - ego_lane],
                           v=[15.0, 5.0], cav=[True, False]), [2.0, 1.0], [1.0, 4.0])
    else:
        x = inferred(state([-500.0, -610.0], [ego_lane, 1 - ego_lane],
                           v=[5.0, 15.0], cav=[True, False]), [1.0, 2.0], [4.0, 1.0])
    assert lane_feasibility(x)[0, 0, choice]
    rules = LaneRules(secure_gap_hook=GS_HOOK)
    assert not lane_feasibility(x, rules=rules)[0, 0, choice]
    # Increase bumper gap past the analytically calculated 139.375 m boundary.
    x["s_m"][0, 1] = -355.5 if side == "front" else -644.5
    assert lane_feasibility(x, rules=rules)[0, 0, choice]


def test_gs_rechecks_simultaneous_changes_that_legacy_mask_allows():
    x = inferred(state([-500.0, -450.0], [0, 0]), [6.0, 6.0], [2.0, 2.0])
    rules = LaneRules(secure_gap_hook=GS_HOOK)
    mask = lane_feasibility(x, rules=rules)
    assert mask[..., LANE_RIGHT].all()  # Empty destination before simultaneous proposals.
    choices = torch.full((1, 2), LANE_RIGHT)
    assert apply_lane_choices(x, choices, lane_feasibility(x))[1].all()
    lane, changed = apply_lane_choices(x, choices, mask, rules)
    assert torch.equal(lane, x["lane"]) and not changed.any()
    lane, changed = apply_lane_choices(x, torch.tensor([[LANE_RIGHT, LANE_STAY]]), mask, rules)
    assert lane.tolist() == [[1, 0]] and changed.tolist() == [[True, False]]


@pytest.mark.parametrize("key", ["secure_gap_T_s", "secure_gap_b_mps2"])
@pytest.mark.parametrize("bad", [None, torch.tensor([[0.0]]), torch.tensor([[-1.0]]),
                                 torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]]),
                                 torch.tensor([1.0]), torch.tensor([[1]])])
def test_gs_requires_explicit_valid_inferred_parameters_without_hidden_fallback(key, bad):
    x = inferred(state([-500.0], [0]), [1.2], [2.0])
    if bad is None:
        del x[key]
    else:
        x[key] = bad
    # Even an empty target lane must not silently qualify a missing parameter path.
    with pytest.raises(ValueError, match=key):
        lane_feasibility(x, rules=LaneRules(secure_gap_hook=GS_HOOK))
    assert lane_feasibility(x)[0, 0, LANE_RIGHT]  # Default needs no inferred parameters.


def test_gs_ignores_absent_padding_and_does_not_mutate_state_or_need_driver_truth():
    x = inferred(state([-500.0, -390.0, 0.0], [0, 1, -1], cav=[True, False, False]),
                 [1.2, 1.3, float("nan")], [2.0, 2.5, float("nan")])
    x["present"][0, 2] = False
    before = {k: value.clone() for k, value in x.items()}
    rules = LaneRules(secure_gap_hook=GS_HOOK)
    mask = lane_feasibility(x, rules=rules)
    assert mask[..., LANE_STAY].all() and mask[0, 0, LANE_RIGHT]
    apply_lane_choices(x, torch.tensor([[LANE_RIGHT, LANE_STAY, LANE_STAY]]), mask, rules)
    for key in x:
        torch.testing.assert_close(x[key], before[key], equal_nan=True)


def consequence_state(s, lane, v=None, cav=None, *, b=2., headway=1.2):
    from wms.model.driver_encoder import PARAMETER_NAMES
    x = state(s, lane, v, cav)
    defaults = dict(v0=30., time_headway=headway, min_gap=2., accel=1.2, decel=b)
    x["inferred_driver_parameters"] = torch.tensor(
        [defaults.get(name, 1.) for name in PARAMETER_NAMES]).expand(1, len(s), -1).clone()
    x["inferred_idm_delta"] = 4
    return x


def consequence_rules(**kwargs):
    # Isolate the consequence threshold from independently calibrated legacy gaps.
    return LaneRules(consequence_mask=True, gap_base_m=0., gap_time_s=0., closing_time_s=0.,
                     rear_gap_base_m=0., rear_gap_time_s=0., rear_closing_time_s=0., **kwargs)


@pytest.mark.parametrize("ego_lane,choice", [(0, LANE_RIGHT), (1, LANE_LEFT), (2, LANE_LEFT)])
def test_consequence_uses_new_followers_learned_b_and_unclipped_idm(ego_lane, choice):
    from wms.model.driver_encoder import PARAMETER_NAMES
    ego_s = 25. if ego_lane == 2 else -500.
    x = consequence_state([ego_s, ego_s - 35.], [ego_lane, 1 if ego_lane == 2 else 1 - ego_lane],
                          v=[10., 15.], cav=[True, False], b=.8)
    rules = consequence_rules()
    assert not lane_feasibility(x, rules=rules)[0, 0, choice]
    # Change only the new follower's learned comfortable deceleration.
    x["inferred_driver_parameters"][0, 1, PARAMETER_NAMES.index("decel")] = 4.
    assert lane_feasibility(x, rules=rules)[0, 0, choice]


@pytest.mark.parametrize("other_s", [-500., -497., -503.])
def test_consequence_forbids_leader_and_follower_overlap_even_with_stale_mask(other_s):
    x = consequence_state([-500., other_s], [0, 1], cav=[True, False])
    stale = torch.ones(1, 2, 3, dtype=torch.bool)
    new_lane, changed = apply_lane_choices(x, torch.tensor([[LANE_RIGHT, LANE_STAY]]),
                                          stale, consequence_rules())
    assert not changed.any() and torch.equal(new_lane, x["lane"])


def test_consequence_rechecks_final_simultaneous_proposals():
    x = consequence_state([-500., -450.], [0, 0], headway=9.)
    rules = consequence_rules()
    mask = lane_feasibility(x, rules=rules)
    assert mask[..., LANE_RIGHT].all()  # Destination empty before proposals.
    lane, changed = apply_lane_choices(x, torch.full((1, 2), LANE_RIGHT), mask, rules)
    assert changed.tolist() == [[True, False]] and lane.tolist() == [[1, 0]]


def test_consequence_propagates_cancelled_leaver_back_into_destination_gap():
    x = consequence_state([-500., -497., -493.], [0, 1, 0], cav=[True, True, False])
    stale = torch.ones(1, 3, 3, dtype=torch.bool)
    lane, changed = apply_lane_choices(x, torch.tensor([[LANE_RIGHT, LANE_LEFT, LANE_STAY]]),
                                       stale, consequence_rules())
    assert not changed.any() and torch.equal(lane, x["lane"])


def test_consequence_missing_parameters_fails_closed_and_padding_is_ignored():
    x = state([-500.], [0])
    assert lane_feasibility(x)[0, 0, LANE_RIGHT]
    with pytest.raises(ValueError, match="inferred_driver_parameters"):
        lane_feasibility(x, rules=consequence_rules())
    x = consequence_state([-500., 0.], [0, -1])
    x["present"][0, 1] = False
    x["inferred_driver_parameters"][0, 1] = float("nan")
    assert lane_feasibility(x, rules=consequence_rules())[0, 0, LANE_RIGHT]
    x["inferred_driver_parameters"][0, 0, 4] = float("nan")
    with pytest.raises(ValueError, match="decel"):
        lane_feasibility(x, rules=consequence_rules())


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_consequence_flag_is_strict_boolean(value):
    with pytest.raises(ValueError, match="consequence_mask"):
        LaneRules.from_mapping({"consequence_mask": value})


def test_consequence_roundtrip_and_optional_second_tier():
    x = consequence_state([-500.], [0])
    rules = consequence_rules()
    assert LaneRules.from_mapping(asdict(rules)) == rules
    assert rules.secure_gap_hook == "" and lane_feasibility(x, rules=rules)[0, 0, LANE_RIGHT]
    with pytest.raises(ValueError, match="secure_gap_T_s"):
        lane_feasibility(x, rules=replace(rules, secure_gap_hook=GS_HOOK))
