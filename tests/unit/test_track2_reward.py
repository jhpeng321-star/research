"""P2: real-outcome reward locality, sign, registration and layer-only learning."""
from collections.abc import Mapping
import copy
from dataclasses import replace
import json

import pytest

from tools.coopmwm import merge_features as mf
from tools.coopmwm import track2_reward as tr


def config(**changes):
    # Test fixture registration, NOT a production weight recommendation.
    return dict(track2_reward=True, track2_reward_weight=0.5,
                track2_reward_provenance="unit-test fixture registration", **changes)


def vehicle(vid, s=0.0, speed=0.0, lane=1):
    return mf.ObservedVehicle(vid, s, lane, speed, 5.0)


def packet(vid="ego", s=0.0, speed=tr.SPEED_LIMIT_MPS, *, time=11.0,
           rows=(), lanes=(0, 1, 2)):
    return mf.SensorReport(time, vehicle(vid, s, speed), lanes, tuple(rows))


def reward(own, reports=None, *, partners=(), now=11.0):
    return tr.neighbourhood_reward(own, "ego", reports, now_s=now,
                                   partner_ids=partners, weight=0.5)


def row(vid, s=0.0, speed=0.0, lane=1, **extras):
    return dict(vid=vid, s_m=s, v_mps=speed, lane=lane, length_m=5.0, **extras)


def episode(*, post=True):
    frames = [[row("ego", speed=tr.SPEED_LIMIT_MPS), row("hdv", 10, 0)],
              [row("ego", speed=tr.SPEED_LIMIT_MPS / 2), row("hdv", 10, 0)]]
    steps = [dict(time_s=10 + i, rows=rows, reward=-98765.0 - i,
                  sample=dict(context=dict(members=["ego"]), value=0.0,
                              coop_layer={}, logp=0.0)) for i, rows in enumerate(frames)]
    if post:
        steps[-1].update(post_rows=[row("ego", speed=tr.SPEED_LIMIT_MPS),
                                   row("hdv", 10, tr.SPEED_LIMIT_MPS)], post_time_s=12.0)
    return dict(steps=steps, totals={"team_delay_vs": 123456789})


def test_constants_match_public_observation_contract():
    from wms.control import contract
    assert tr.SPEED_LIMIT_MPS == contract.SPEED_LIMIT_MPS
    assert tr.CONTROL_DT_S == contract.CONTROL_DT_S
    assert mf.SENSOR_RANGE_M == contract.SENSOR_RANGE_M


@pytest.mark.parametrize("weight", [None, True, 0.0, -1, float("nan"), float("inf"), "1"])
def test_registered_positive_finite_weight_is_required(weight):
    cfg = config()
    cfg["track2_reward_weight"] = weight
    with pytest.raises(ValueError, match="track2_reward_weight"):
        tr.settings(cfg)


@pytest.mark.parametrize("provenance", [None, "", "  ", 1])
def test_provenance_is_required(provenance):
    cfg = config()
    cfg["track2_reward_provenance"] = provenance
    with pytest.raises(ValueError, match="provenance"):
        tr.settings(cfg)


@pytest.mark.parametrize("arms", [["A", "B", "R"], ["B"], ["R"], ["B", "B"], "BR", [{}]])
def test_cannot_enable_A_or_give_B_and_R_different_weights(arms):
    with pytest.raises(ValueError, match="exactly B and R"):
        tr.settings(config(track2_reward_arms=arms))


class Unreadable:
    def __getattribute__(self, name):
        raise AssertionError(f"inactive reward read input {name}")


def test_default_off_and_A_are_identity_even_with_unreadable_receipts():
    episodes = Unreadable()
    for cfg, arm in [({}, "B"), ({"coop_reward": True}, "R"),
                     ({"track2_reward": False, "track2_reward_weight": "bad"}, "B"),
                     ({"track2_reward": True}, "A")]:
        out, receipt = tr.training_episodes(cfg, arm, episodes)
        assert out is episodes and receipt is None
        assert tr.layer_records(cfg, arm, episodes) == ([], None)
    with pytest.raises(ValueError, match="boolean"):
        tr.settings({"track2_reward": 1})


def test_formula_is_negative_delay_with_self_excluded_and_no_automation_filter():
    own = packet(speed=tr.SPEED_LIMIT_MPS / 2, rows=(
        vehicle("cav", 50, tr.SPEED_LIMIT_MPS), vehicle("hdv", -50, 0),
    ))
    out = reward(own)
    assert out["own_reward"] == -0.5
    assert out["neighbourhood_reward"] == -0.5
    assert out["team_addition"] == -0.25
    assert out["reward"] == -0.75
    assert out["neighbour_ids"] == ["cav", "hdv"]
    assert out["neighbour_timestamps_s"] == {"cav": 11, "hdv": 11}
    faster = reward(replace(own, vehicles=(vehicle("cav", 50, tr.SPEED_LIMIT_MPS),
                                           vehicle("hdv", -50, tr.SPEED_LIMIT_MPS))))
    assert faster["reward"] > out["reward"]


def test_empty_and_above_limit_neighbourhoods_have_zero_team_term():
    out = reward(packet(speed=tr.SPEED_LIMIT_MPS / 2))
    assert out["reward"] == -0.5 and out["empty_neighbourhood"]
    assert out["neighbourhood_reward"] == 0.0
    assert reward(packet(rows=(vehicle("fast", 0, 2 * tr.SPEED_LIMIT_MPS),)))["reward"] == 0.0


def test_local_range_and_lane_coverage_reject_unobserved_traffic():
    own = packet(lanes=(1,), rows=(vehicle("boundary", 100, 0),
                                   vehicle("far", 100.001, 0), vehicle("uncovered", 0, 0, 2)))
    out = reward(own)
    assert out["neighbour_ids"] == ["boundary"]
    assert out["reward"] == -0.5


@pytest.mark.parametrize("time", [8, 9.9, 10.1, 11, 12, float("nan")])
def test_remote_outcomes_only_from_exact_t_minus_one_packets(time):
    remote = packet("p", 50, 0, time=time)
    assert reward(packet(), {"p": remote}, partners=("p",)) == {
        **reward(packet()), "unavailable_partner_reports": 1}


def test_fractional_timestamps_remote_range_cap_selection_and_deduplication():
    own = packet(time=11.25, rows=(vehicle("shared", 20, tr.SPEED_LIMIT_MPS),))
    reports = {f"p{i}": packet(f"p{i}", i * 10, 0, time=10.25,
                               rows=(vehicle("shared", 20, 0),)) for i in range(1, 7)}
    reports["p1"] = replace(reports["p1"], vehicles=(
        vehicle("shared", 20, 0), vehicle("relayed_hdv", 110, 0),
        vehicle("outside_publisher", 111, 0)))
    out = reward(own, reports, partners=tuple(reports) + ("p1", "ego"), now=11.25)
    assert out["partner_reports"] == 4 and out["unavailable_partner_reports"] == 2
    assert out["neighbour_ids"] == ["p1", "p2", "p3", "p4", "relayed_hdv", "shared"]
    assert out["neighbour_timestamps_s"]["shared"] == 11.25  # direct wins
    assert out["neighbour_timestamps_s"]["relayed_hdv"] == 10.25
    assert out == reward(own, dict(reversed(list(reports.items()))),
                         partners=tuple(reversed(reports)), now=11.25)


def test_remote_publisher_identity_membership_radio_range_and_tie_precedence():
    own = packet()
    p = packet("p", 100, 0, time=10)
    assert reward(own, {"p": p})["neighbours"] == 0
    assert reward(own, {"p": p}, partners=("p",))["neighbour_ids"] == ["p"]
    for bad in (replace(p, source=vehicle("impostor", 50)),
                replace(p, source=vehicle("p", 100.001))):
        assert reward(own, {"p": bad}, partners=("p",))["neighbours"] == 0
    a = packet("a", 10, time=10, rows=(vehicle("shared", 20, 0),))
    b = packet("b", -10, time=10, rows=(vehicle("shared", 20, tr.SPEED_LIMIT_MPS),))
    out = reward(own, {"b": b, "a": a}, partners=("b", "a"))
    assert out["neighbourhood_reward"] == pytest.approx(-1 / 3)


@pytest.mark.parametrize("bad", [None, {}, packet(time=10),
    packet(rows=(vehicle("bad", speed=float("nan")),)),
    packet(rows=(vehicle("bad", speed=-1),)),
    packet(rows=(vehicle("same"), vehicle("same")))])
def test_missing_or_malformed_own_sensing_is_not_fabricated_zero(bad):
    with pytest.raises(ValueError, match="current own SUMO sensing"):
        reward(bad)


class SelectedOnly(Mapping):
    def __init__(self, p):
        self.p = p

    def __getitem__(self, key):
        assert key == "p"
        return self.p

    def __iter__(self):
        raise AssertionError("global report board scan")

    def __len__(self):
        return 1


def test_reward_never_scans_report_board_or_uses_unselected_global_fields():
    p = packet("p", 50, 0, time=10)
    expected = reward(packet(), {"p": p}, partners=("p",))
    assert reward(packet(), SelectedOnly(p), partners=("p",)) == expected
    assert reward(packet(), {"p": p, "unselected": Unreadable()}, partners=("p",)) == expected
    assert reward(packet())["partner_reports"] == 0  # no cached packets


def test_episode_adapter_uses_post_action_outcomes_not_global_reward_or_prestep():
    original = [episode()]
    before = copy.deepcopy(original)
    shaped, receipt = tr.training_episodes(config(), "B", original)
    assert original == before
    assert shaped is not original and shaped[0] is not original[0]
    assert [s["reward"] for s in shaped[0]["steps"]] == [-1.0, 0.0]
    for raw, out in zip(original[0]["steps"], shaped[0]["steps"]):
        assert out["sample"] is raw["sample"]
        assert out["base_reward"] == raw["reward"]
        assert out["rows"] is raw["rows"]
    assert receipt["own_reward_sum"] == -0.5
    assert receipt["team_addition_sum"] == -0.5
    assert receipt["reward_sum"] == -1.0
    assert receipt["cav_outcomes"] == 2
    assert not receipt["wm_used"] and not receipt["global_reward_used"]
    assert receipt["terminal_fallback_steps"] == 0
    json.dumps(receipt, allow_nan=False)


def test_B_and_R_share_registration_and_identical_reward_transforms():
    b, br = tr.training_episodes(config(), "B", [episode()])
    r, rr = tr.training_episodes(config(track2_reward_arms=["R", "B"]), "R", [episode()])
    assert b == r
    assert {k: v for k, v in br.items() if k != "arm"} == {
        k: v for k, v in rr.items() if k != "arm"}


def test_global_reward_and_out_of_range_outcomes_cannot_change_layer_rewards():
    raw = episode()
    changed = copy.deepcopy(raw)
    changed["totals"] = Unreadable()
    for step in changed["steps"]:
        step["reward"] = Unreadable()
        step["rows"].append(dict(vid="far", s_m=1000, v_mps=Unreadable()))
    changed["steps"][-1]["post_rows"].append(dict(vid="far", s_m=1000, v_mps=Unreadable()))
    baseline, _ = tr.training_episodes(config(), "B", [raw])
    actual, _ = tr.training_episodes(config(), "B", [changed])
    assert [s["reward"] for s in actual[0]["steps"]] == [s["reward"] for s in baseline[0]["steps"]]


def test_joint_reward_is_equal_CAV_mean_including_empty_neighbourhoods():
    raw = episode()
    for step in raw["steps"]:
        step["sample"]["context"]["members"].append("alone")
        step["rows"].append(row("alone", 1000, tr.SPEED_LIMIT_MPS))
    raw["steps"][-1]["post_rows"].append(row("alone", 1000, tr.SPEED_LIMIT_MPS))
    shaped, receipt = tr.training_episodes(config(), "B", [raw])
    assert shaped[0]["steps"][0]["reward"] == -0.5
    assert receipt["empty_neighbourhoods"] == 2
    assert receipt["cav_outcomes"] == 4


def test_explicit_packet_adapter_adds_only_valid_delayed_V2V_evidence():
    raw = episode()
    for step in raw["steps"]:
        now = step["time_s"] + 1
        step["reward_observations"] = dict(ego=dict(
            own=packet(time=now), partner_ids=("p",), partner_reports=dict(
                p=packet("p", 50, 0, time=now - 1))))
        step["rows"] = Unreadable()  # explicit local packets, never the global frame
    shaped, receipt = tr.training_episodes(config(), "R", [raw])
    assert [s["reward"] for s in shaped[0]["steps"]] == [-0.5, -0.5]
    assert receipt["partner_reports"] == 2


def test_terminal_missing_gapped_or_departed_ego_outcomes_fail_closed():
    with pytest.raises(ValueError, match="terminal post-action"):
        tr.training_episodes(config(), "B", [episode(post=False)])
    raw = episode()
    raw["steps"][1]["time_s"] = 99
    with pytest.raises(ValueError, match="one control tick"):
        tr.training_episodes(config(), "B", [raw])
    raw = episode()
    raw["steps"][-1]["post_rows"] = [row("hdv")]
    with pytest.raises(ValueError, match="missing post-action observable ego"):
        tr.training_episodes(config(), "B", [raw])


def test_empty_CAV_steps_are_counted_and_reward_is_never_applied_twice():
    raw = episode(post=False)
    for step in raw["steps"]:
        step["sample"]["context"]["members"] = []
    shaped, receipt = tr.training_episodes(config(), "B", [raw])
    assert receipt["empty_cav_steps"] == 2
    assert all(s["reward"] == 0.0 for s in shaped[0]["steps"])
    with pytest.raises(ValueError, match="already applied"):
        tr.training_episodes(config(), "B", shaped)


def test_layer_GAE_matches_existing_discount_and_excludes_non_PPO_records():
    from tools.coopmwm.accel_learner import advantages
    raw = episode()
    shaped, _ = tr.training_episodes(config(), "B", [raw])
    records, receipt = tr.layer_records(config(), "B", [raw])
    assert [r["group_advantage"] for r in records] == [a for a, _ in advantages(shaped[0]["steps"])]
    assert records[0]["sample"] is raw["steps"][0]["sample"]
    assert receipt["layer_records"] == 2
    raw["steps"][0]["sample"]["behaviour"] = dict(executed=True)
    raw["steps"][1]["sample"]["deterministic"] = True
    assert tr.layer_records(config(), "B", [raw])[0] == []


def test_real_reward_records_update_only_the_cooperation_layer():
    import torch
    from tools.coopmwm.acceleration import Policy
    from tools.coopmwm.coop_layer import CooperationLayer, LayerConfig

    torch.manual_seed(24)
    base = Policy()
    layer = CooperationLayer(base, LayerConfig(
        feature_dim=29, hidden_dim=8, learning_rate=0.001,
        kl_budget=0.045, clip_ratio=0.2, max_grad_norm=0.5, enabled=True))
    context = dict(x=[[0.1] * 29], rsu=[0.1] * 12, members=["ego"])
    raw = episode()
    for i, step in enumerate(raw["steps"]):
        step["sample"] = layer.sample(context, torch.Generator().manual_seed(50 + i))[0]
    before_base = {k: v.clone() for k, v in layer.base.state_dict().items()}
    before_layer = [p.clone() for p in layer.layer_parameters()]
    records, _ = tr.layer_records(config(), "B", [raw])
    metrics = layer.update(records, epochs=1)
    assert metrics[0]["accepted"]
    assert any(not torch.equal(a, b) for a, b in zip(before_layer, layer.layer_parameters()))
    assert all(torch.equal(v, layer.base.state_dict()[k]) for k, v in before_base.items())
    assert all(p.grad is None and not p.requires_grad for p in layer.base.parameters())
