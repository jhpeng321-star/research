"""C4 policy v2: per-CAV lane head, lane masks, REINFORCE, counterfactual
difference rewards (individual / close groups / all, common random numbers) and SUMO lane commands.

CPU only; the SUMO test skips without ``sumo``."""
from __future__ import annotations

import math
import shutil
from types import SimpleNamespace

import pytest
import torch

from wms.control.contract import ACCEL_MAX_MPS2, NATIVE_CAV_ACCEL_MAX_MPS2, PHYSICS_DT_S, RewardWeights
from wms.control.counterfactual import (KIND_ALL, KIND_BASE, KIND_FLIP, KIND_INDIVIDUAL,
                                        CounterfactualConfig, build_variants, close_groups,
                                        decompose, group_rich, idm_native_accel, row_coefficients)
from wms.control.dreamer import DreamerConfig, DreamerTrainer
from wms.control.lanes import (LANE_LEFT, LANE_RIGHT, LANE_STAY, LaneDwell, LaneRules,
                               apply_lane_choices, lane_feasibility)
from wms.control.policy import CavPolicy, PolicyOutput, RsuGroupEncoder, TeamCritic

HAS_SUMO = shutil.which("sumo") is not None


def state(s, lane, v=None, cav=None, present=None):
    k = len(s)
    v = [20.0] * k if v is None else v
    cav = [True] * k if cav is None else cav
    present = [True] * k if present is None else present
    return {"s_m": torch.tensor([s], dtype=torch.float32),
            "v_mps": torch.tensor([v], dtype=torch.float32),
            "a_mps2": torch.zeros(1, k), "lane": torch.tensor([lane]),
            "present": torch.tensor([present]), "is_cav": torch.tensor([cav]) & torch.tensor([present]),
            "length_m": torch.full((1, k), 5.0)}


# ------------------------------------------------------------------ lane mask
def test_lane_mask_geometry_never_ramp_and_never_outside_mainline():
    x = state([-500.0, -800.0, -100.0, 30.0], [0, 1, 2, 2], cav=[True, True, True, True])
    mask = lane_feasibility(x, None, LaneRules())[0]
    assert mask[:, LANE_STAY].all()
    assert mask[0].tolist() == [True, False, True]       # lane 0: no left (outside 0..1)
    assert mask[1].tolist() == [True, True, False]       # lane 1: never right into the ramp lane
    assert mask[2].tolist() == [True, False, False]      # ramp upstream of the merge zone
    assert mask[3].tolist() == [True, True, False]       # ramp lane inside the zone: merge now


def test_lane_rules_front_rear_parameters_and_legacy_mapping():
    r = LaneRules()
    assert r.rear == (r.rear_gap_base_m, r.rear_gap_time_s, r.rear_closing_time_s)
    assert (r.gap_base_m, r.gap_time_s, r.closing_time_s) == (1.0, 1.2, 4.5)
    # A mapping without rear_* keys is a legacy (symmetric) payload; with them it is taken as given.
    old = LaneRules.from_mapping({"gap_base_m": 6.0, "gap_time_s": 1.0, "closing_time_s": 1.0})
    assert old.rear == (6.0, 1.0, 1.0)
    new = LaneRules.from_mapping({"gap_base_m": 2.0, "rear_gap_base_m": 4.0, "rear_gap_time_s": 1.5,
                                  "rear_closing_time_s": 2.0})
    assert new.rear == (4.0, 1.5, 2.0) and new.gap_base_m == 2.0
    assert LaneRules.from_mapping(None).rear == r.rear
    import dataclasses
    assert LaneRules.from_mapping(dataclasses.asdict(old)).rear == (6.0, 1.0, 1.0)   # checkpoint round trip
    with pytest.raises(ValueError):
        LaneRules(rear_gap_time_s=-1.0)


def test_lane_mask_gaps_speed_term_dwell_edge_guard_and_hdv():
    rules = LaneRules()
    v = 20.0
    need_front = rules.gap_base_m + rules.gap_time_s * v       # equal speeds: no closing term
    # Target-lane vehicle ahead with bumper gap need_front - 0.5 (blocked) / + 0.5 (feasible).
    for extra, ok in ((-0.5, False), (0.5, True)):
        x = state([-500.0, -500.0 + 5.0 + need_front + extra], [0, 1], cav=[True, False])
        assert bool(lane_feasibility(x, None, rules)[0, 0, LANE_RIGHT]) is ok
    # Rear vehicle faster than the ego: closing term (calibrated defaults: 3 + 1.25*25 + 6*5 = 64.25 m).
    x = state([-500.0, -500.0 - 5.0 - 60.0], [0, 1], v=[20.0, 25.0], cav=[True, False])
    assert not bool(lane_feasibility(x, None, rules)[0, 0, LANE_RIGHT])
    x["s_m"][0, 1] = -500.0 - 5.0 - 68.0
    assert bool(lane_feasibility(x, None, rules)[0, 0, LANE_RIGHT])
    # Legacy symmetric rule (rear_* = None): 6 + 25 + 5 = 36 m.
    legacy = LaneRules(gap_base_m=6.0, gap_time_s=1.0, closing_time_s=1.0, rear_gap_base_m=None,
                       rear_gap_time_s=None, rear_closing_time_s=None)
    assert legacy.rear == (6.0, 1.0, 1.0)
    x["s_m"][0, 1] = -500.0 - 5.0 - 30.0
    assert not bool(lane_feasibility(x, None, legacy)[0, 0, LANE_RIGHT])
    x["s_m"][0, 1] = -500.0 - 5.0 - 40.0
    assert bool(lane_feasibility(x, None, legacy)[0, 0, LANE_RIGHT])
    assert not bool(lane_feasibility(x, None, rules)[0, 0, LANE_RIGHT])    # new default is stricter behind
    # Alongside (overlapping) vehicle blocks.
    x = state([-500.0, -501.0], [0, 1], cav=[True, False])
    assert not bool(lane_feasibility(x, None, rules)[0, 0, LANE_RIGHT])
    # Dwell clock.
    x = state([-500.0], [0])
    assert not bool(lane_feasibility(x, torch.tensor([[1.0]]), rules)[0, 0, LANE_RIGHT])
    assert bool(lane_feasibility(x, torch.tensor([[3.0]]), rules)[0, 0, LANE_RIGHT])
    # Edge guard before s = 0 (SUMO edge boundary): reach = v*1 + 10 = 30 m.
    assert not bool(lane_feasibility(state([-25.0], [0]), None, rules)[0, 0, LANE_RIGHT])
    assert bool(lane_feasibility(state([-35.0], [0]), None, rules)[0, 0, LANE_RIGHT])
    # HDV slot: stay only.
    assert lane_feasibility(state([-500.0], [0], cav=[False]), None, rules)[0, 0].tolist() == [
        True, False, False]


def test_apply_lane_choices_cancels_conflicting_simultaneous_changes():
    rules = LaneRules()
    # Two CAVs in lane 0, 8 m apart (3 m bumper gap); lane 1 empty: each alone may move right,
    # together they would sit 3 m apart in lane 1 -> both cancelled.
    x = state([-300.0, -292.0], [0, 0])
    mask = lane_feasibility(x, None, rules)
    assert mask[0, :, LANE_RIGHT].all()
    lane, changed = apply_lane_choices(x, torch.tensor([[LANE_RIGHT, LANE_RIGHT]]), mask, rules)
    assert lane.tolist() == [[0, 0]] and not changed.any()
    lane, changed = apply_lane_choices(x, torch.tensor([[LANE_RIGHT, LANE_STAY]]), mask, rules)
    assert lane.tolist() == [[1, 0]] and changed.tolist() == [[True, False]]
    # Infeasible choice (left of lane 0) is never executed; non-acting CAVs never move.
    lane, _ = apply_lane_choices(x, torch.tensor([[LANE_LEFT, LANE_RIGHT]]), mask, rules,
                                 acting=torch.tensor([[True, False]]))
    assert lane.tolist() == [[0, 0]]


def test_lane_dwell_clock():
    dwell = LaneDwell(3.0, 1.0)
    x = state([-500.0], [0])
    assert dwell.step(x).item() == 3.0                 # unknown history: allowed
    x["lane"] = torch.tensor([[1]])
    assert dwell.step(x).item() == 0.0                 # changed
    assert dwell.step(x).item() == 1.0
    x["lane"] = torch.tensor([[-1]])                   # junction transit keeps the last lane
    assert dwell.step(x).item() == 2.0
    x["lane"] = torch.tensor([[1]])
    assert dwell.step(x).item() == 3.0
    seeded = LaneDwell(3.0, 0.5)
    assert seeded.step({**x, "lane_dwell_s": torch.tensor([[0.5]])}).item() == 0.5


# ------------------------------------------------------------------ policy lane head
def _obs_and_signal(x):
    from wms.control.obs import build_observations
    from wms.control.rsu_field import build_rsu_field

    obs = build_observations(x)
    signal = RsuGroupEncoder()(obs, x["s_m"], build_rsu_field(x))
    return obs, signal


def test_policy_lane_head_masks_infeasible_choices_and_keeps_legacy_contract():
    torch.manual_seed(0)
    policy = CavPolicy()
    x = state([-500.0, -800.0, -900.0], [0, 1, 1], cav=[True, True, False])
    obs, signal = _obs_and_signal(x)
    legacy = policy(obs, signal)
    assert isinstance(legacy, tuple) and len(legacy) == 2
    mask = lane_feasibility(x, None, LaneRules())
    for _ in range(50):
        out = policy(obs, signal, lane_mask=mask)
        assert isinstance(out, PolicyOutput)
        assert mask.gather(-1, out.lane_choice[..., None]).all()
        assert out.lane_choice[0, 2] == LANE_STAY and out.log_prob_lane[0, 2] == 0
    # Uniform start (zero-initialised head) over the feasible choices: entropy = log 2.
    torch.testing.assert_close(out.entropy_lane[0, :2], torch.full((2,), math.log(2.0)))
    # Only "stay" feasible -> no lane log-probability (no REINFORCE signal).
    only_stay = torch.zeros_like(mask)
    out = policy(obs, signal, lane_mask=only_stay)
    assert (out.lane_choice == LANE_STAY).all() and (out.log_prob_lane == 0).all()
    assert (out.lane_options[0, :2] == 1).all()
    # Explicit noise reproduces the draw exactly (common random numbers).
    from wms.control.policy import draw_policy_noise
    noise = draw_policy_noise((1, 3), generator=torch.Generator().manual_seed(4))
    a = policy(obs, signal, lane_mask=mask, noise=noise)
    b = policy(obs, signal, lane_mask=mask, noise=noise)
    torch.testing.assert_close(a.accel, b.accel)
    assert torch.equal(a.lane_choice, b.lane_choice)
    # Accel-only actor (c4_dreamer/1): always stays.
    old = CavPolicy(lane_head=False)
    out = old(obs, signal, lane_mask=mask)
    assert (out.lane_choice == 0).all() and (out.log_prob_lane == 0).all()


W_ALL_ONLY = {"individual": 0.0, "group": 0.0, "coop": 0.0, "team": 0.0, "all": 1.0}


def _lanes(log_prob, active):
    return {"lane_log_prob": log_prob, "lane_entropy": torch.zeros_like(log_prob),
            "lane_active": active, "lane_acting": active, "lane_changed": torch.zeros_like(active)}


def test_reinforce_sign_and_difference_reward_advantage():
    """Positive advantage -> minimising the objective INCREASES log pi(choice); negative decreases."""
    trainer = SimpleNamespace(config=DreamerConfig(lane_entropy_weight=0.0))
    for advantage_sign in (1.0, -1.0):
        log_prob = torch.tensor([[[[-0.7]]]], requires_grad=True)       # [E,H,N,K]
        imagined = {"n_base": 1, "variants": None,
                    "returns": torch.tensor([[[5.0 + advantage_sign]]]),
                    "values": torch.tensor([[[5.0], [0.0]]]),
                    "lanes": _lanes(log_prob, torch.ones(1, 1, 1, 1, dtype=torch.bool))}
        loss, extra = DreamerTrainer.actor_objective(trainer, imagined, 1.0, W_ALL_ONLY)
        loss.backward()
        assert math.copysign(1.0, -float(log_prob.grad)) == advantage_sign
        assert "lane_pg_loss" in extra
    # Individual row: advantage = R_i - R_base (difference reward, CRN), not R - V.
    x = state([-500.0, -300.0], [0, 1])
    variants = build_variants(x, CounterfactualConfig(enabled=True, n_individual=1, n_group=0))
    assert variants.row_kind.tolist() == [KIND_ALL, KIND_BASE, KIND_INDIVIDUAL]
    log_prob = torch.full((1, 1, 3, 2), -0.7, requires_grad=True)
    active = torch.zeros(1, 1, 3, 2, dtype=torch.bool)
    slot = [key[1] for key in variants.individual_rows][0]
    active[0, 0, 2, slot] = True
    returns = torch.tensor([[[0.0, 10.0, 7.0]]])           # all, base, individual: D_i = -3
    imagined = {"n_base": 1, "variants": variants, "returns": returns,
                "values": torch.zeros(1, 2, 3), "lanes": _lanes(log_prob, active)}
    loss, extra = DreamerTrainer.actor_objective(
        trainer, imagined, 1.0, {**W_ALL_ONLY, "individual": 1.0, "all": 0.0})
    loss.backward()
    assert float(log_prob.grad[0, 0, 2, slot]) == pytest.approx(3.0)   # -(c * A), c = 1, A = -3
    assert extra["D_i"] == pytest.approx(-3.0)


# ------------------------------------------------------------------ counterfactual groups
def test_close_groups_chain_within_radius_and_ignore_singletons():
    s = torch.tensor([0.0, 100.0, 260.0, 300.0, 1000.0, 40.0])
    cav = torch.tensor([True, True, True, True, True, False])
    assert close_groups(s, cav, 150.0) == [[0, 1], [2, 3]]
    assert close_groups(s, cav, 200.0) == [[0, 1, 2, 3]]          # chained (0-100-260-300)


def test_build_variants_rows_masks_disjoint_groups_cover_members_and_flips():
    x = state([-900.0, -850.0, -800.0, -750.0, -100.0, 400.0, 430.0], [0, 1, 0, 1, 0, 0, 1],
              cav=[True, True, True, True, False, True, True])
    cfg = CounterfactualConfig(enabled=True, n_individual=2, n_group=2, n_flip=2, group_max_size=3)
    v = build_variants({key: value.expand(2, -1).clone() for key, value in x.items()}, cfg,
                       torch.Generator().manual_seed(0))
    assert v.n_starts == 2 and v.row_kind[:2].tolist() == [KIND_ALL, KIND_ALL]
    assert v.act_mask[0].tolist() == [True, True, True, True, False, True, True]
    per_start = {}
    for group in v.groups:
        start, members = group["start"], group["members"]
        per_start.setdefault(start, []).append(set(members))
        assert members == sorted(members) and len(members) <= 3        # contiguous window, capped
        assert v.act_mask[group["row"]].nonzero().flatten().tolist() == members
        assert all((start, m) in v.individual_rows for m in members)  # C_G computable
        assert len(group["flips"]) == min(2, len(members))
        for flipped, row in group["flips"].items():                   # G without i
            assert v.act_mask[row].nonzero().flatten().tolist() == [m for m in members if m != flipped]
        assert not v.act_mask[v.base_row[start]].any()
    for groups in per_start.values():
        assert len(groups) == 2 and not set.intersection(*groups)     # disjoint groups
    pair = next(g for g in v.groups if len(g["members"]) == 2)        # pairs reuse solo rows
    i, j = pair["members"]
    assert pair["flips"][i] == v.individual_rows[(pair["start"], j)]
    assert KIND_FLIP in v.row_kind.tolist()
    for (start, slot), row in v.individual_rows.items():
        assert v.act_mask[row].nonzero().flatten().tolist() == [slot]
        assert int(v.row_start[row]) == start
    assert build_variants(x, CounterfactualConfig(enabled=False)).n_rows == 1


def test_cooperation_decomposition_arithmetic():
    """C_G = D_G - sum D_i, M_i = D_G - D_G\\i, I_i = M_i - D_i on hand-set returns."""
    x = state([-900.0, -850.0, -800.0], [0, 1, 0])
    v = build_variants(x, CounterfactualConfig(enabled=True, n_group=1, n_flip=1),
                       torch.Generator().manual_seed(1))
    group = v.groups[0]
    assert group["members"] == [0, 1, 2]
    (flipped, flip_row), = group["flips"].items()
    j = torch.zeros(v.n_rows)
    j[0], j[v.base_row[0]] = 16.0, 10.0                    # D_all = 6
    for slot, value in zip((0, 1, 2), (11.0, 12.0, 10.5)):  # D_i = 1, 2, 0.5
        j[v.individual_rows[(0, slot)]] = value
    j[group["row"]] = 15.0                                   # D_G = 5
    j[flip_row] = 13.0                                       # D_G\i = 3 -> M_i = 2
    parts = decompose(j, v)
    d_i = {0: 1.0, 1: 2.0, 2: 0.5}[flipped]
    assert parts["B"] == [10.0] and parts["D_all"] == [6.0] and parts["D_G"] == [5.0]
    assert parts["C_G"] == [pytest.approx(5.0 - 3.5)]
    assert parts["M_i"] == [pytest.approx(2.0)] and parts["I_i"] == [pytest.approx(2.0 - d_i)]


def test_objective_coefficients_and_anti_gaming_guard():
    with pytest.raises(ValueError, match="anti-gaming"):
        CounterfactualConfig(w_individual=[1.0, 0.4], w_coop=[0.0, 0.5])
    with pytest.raises(ValueError, match="cover_group_members"):
        CounterfactualConfig(cover_group_members=False)
    with pytest.raises(ValueError, match="n_flip"):
        CounterfactualConfig(n_flip=0)
    x = state([-900.0, -850.0, -800.0, -300.0, -280.0], [0, 1, 0, 0, 1])
    cfg = CounterfactualConfig(enabled=True, n_group=2, n_flip=1)
    v = build_variants(x, cfg, torch.Generator().manual_seed(2))
    weights = {"individual": 0.5, "group": 1.0, "coop": 0.5, "team": 0.5, "all": 1.0}
    c = row_coefficients(v, weights, v.n_rows, 1)
    assert c[0] == pytest.approx(1.0)
    for row in v.individual_rows.values():                  # net w_ind - w_coop >= 0
        assert float(c[row]) >= -1e-9
    big = next(g for g in v.groups if len(g["members"]) == 3)
    pair = next(g for g in v.groups if len(g["members"]) == 2)
    assert float(c[big["row"]]) == pytest.approx(1.0 + 0.5 + 0.5)        # w_group + w_coop + w_team
    assert float(c[pair["row"]]) == pytest.approx(1.0 + 0.5)              # pairs: no team term
    (flip_row,) = big["flips"].values()
    assert float(c[flip_row]) == pytest.approx(-0.5)
    # Exactly the original objective without counterfactuals.
    assert row_coefficients(None, W_ALL_ONLY, 4, 4).tolist() == [0.25] * 4


def test_curriculum_starts_individual_and_ramps_to_cooperation():
    cfg = CounterfactualConfig(enabled=True)
    first, mid = cfg.weights(0), cfg.weights(1500)
    assert first == {"individual": 1.0, "group": 0.5, "coop": 0.0, "team": 0.0, "all": 0.25}
    assert mid == {"individual": 0.5, "group": 1.0, "coop": 0.5, "team": 0.5, "all": 1.0}
    assert cfg.weights(750)["coop"] == pytest.approx(0.25) and cfg.weights(10_000) == mid
    assert CounterfactualConfig().weights(0) == W_ALL_ONLY


def test_group_rich_predicate_and_sampler_oversampling():
    from wms.control.start_states import StartStateSampler

    cfg = CounterfactualConfig(enabled=True)
    slow = state([-300.0, -250.0, -280.0], [0, 0, 1], v=[5.0, 5.0, 5.0], cav=[True, True, False])
    fast = state([-300.0, -250.0, -280.0], [0, 0, 1], v=[30.0, 30.0, 30.0], cav=[True, True, False])
    far = state([-1500.0, -1450.0], [0, 0], v=[5.0, 5.0])
    assert group_rich(slow, cfg) and not group_rich(fast, cfg) and not group_rich(far, cfg)

    def record(x):
        kin = torch.zeros(1, x["s_m"].shape[1], 7)
        kin[0, :, 0], kin[0, :, 4] = x["s_m"][0], x["v_mps"][0]
        kin[0, torch.arange(x["s_m"].shape[1]), 1 + x["lane"][0]] = 1.0
        kin[0, :, 6] = 5.0
        return {"states": kin, "present": x["present"], "is_cav": x["is_cav"],
                "meta": {"world_id": "w1m2700p20t0", "split": "TRAIN"}}

    data = [record(fast)] * 9 + [record(slow)]
    dataset = type("D", (), {"records": data, "receipt": {"split": "TRAIN"},
                             "__len__": lambda self: len(data), "__getitem__": lambda self, i: data[i]})()
    sampler = StartStateSampler(dataset, seed=0)
    batch = sampler.sample_where(4, predicate=lambda st: group_rich(st, cfg), fraction=0.5, max_tries=50)
    flags = [m["group_rich"] for m in batch.metadata]
    assert flags[:2] == [True, True]                         # oversampled (base rate 10 %)


def test_idm_native_brakes_behind_slow_leader_and_cruises_free():
    x = state([-500.0, -480.0, -100.0], [0, 0, 1], v=[25.0, 5.0, 31.29], cav=[True, False, True])
    a = idm_native_accel(x)[0]
    assert a[0] == pytest.approx(-3.0) and abs(float(a[2])) < 1e-3


# ------------------------------------------------------------------ trainer end to end (fake WM)
class LaneWorldModel(torch.nn.Module):
    """Fake WM: CAVs take the callback motion (incl. lane), HDVs hold speed."""

    def __init__(self):
        super().__init__()
        self.physics = SimpleNamespace(dt_s=PHYSICS_DT_S)

    def rollout(self, x, *, n_steps, cav_controller, **kwargs):
        current = dict(x)
        traces = {key: [] for key in ("positions", "speeds", "lanes", "masks", "accel")}
        for _ in range(n_steps):
            command = cav_controller(current)
            cav = current["present"] & current["is_cav"]
            nxt = {**current,
                   "s_m": torch.where(cav, command[..., 0], current["s_m"] + PHYSICS_DT_S * current["v_mps"]),
                   "lane": torch.where(cav, command[..., 1].long(), current["lane"]),
                   "v_mps": torch.where(cav, command[..., 2], current["v_mps"]),
                   "a_mps2": torch.where(cav, command[..., 3], torch.zeros_like(command[..., 3]))}
            for out, name in (("positions", "s_m"), ("speeds", "v_mps"), ("lanes", "lane"),
                              ("masks", "present"), ("accel", "a_mps2")):
                traces[out].append(nxt[name])
            current = nxt
        return {**{key: torch.stack(seq, 1) for key, seq in traces.items()}, "state": current}


class Starts:
    def __init__(self, x):
        self.x = x

    def sample(self, batch_size, device="cpu"):
        from wms.control.start_states import StartBatch
        return StartBatch({k: v.expand(batch_size, -1).clone() for k, v in self.x.items()}, {})


def _real_trainer(lanes=True, counterfactual=True, hidden=16, accel_max=ACCEL_MAX_MPS2, **cfg):
    # accel_max: the scenario was tuned with the legacy 1.5 actor cap (ACCELCAP_NOTES.md).
    from wms.control.bridge import make_cav_controller
    from wms.control.obs import build_observations
    from wms.control.reward import team_reward
    from wms.control.safety import safe_accel
    from wms.control.train_cli import GroupSignal

    x = state([-900.0, -860.0, -820.0, -700.0, -880.0, -790.0], [0, 1, 0, 1, 1, 1],
              v=[25.0, 24.0, 26.0, 22.0, 20.0, 23.0], cav=[True, True, True, False, False, True])
    settings = dict(batch_size=2, horizon=3, checkpoint_every=0, validation_every=0,
                    actor_lr=1e-3, critic_lr=1e-3)
    settings.update(cfg)
    return DreamerTrainer(
        [LaneWorldModel()], CavPolicy(hidden_dim=hidden, accel_max=accel_max), GroupSignal(RsuGroupEncoder(hidden_dim=hidden)),
        TeamCritic(hidden_dim=hidden), Starts(x), observation_builder=build_observations,
        reward_fn=team_reward, safety=safe_accel, controller_factory=make_cav_controller,
        config=DreamerConfig(**settings), reward_weights=RewardWeights(),
        lane_rules=LaneRules() if lanes else None,
        counterfactual=CounterfactualConfig(enabled=counterfactual, n_individual=2, n_group=2,
                                            n_flip=2, group_max_size=4))


def test_counterfactual_and_flip_one_rows_use_common_random_numbers():
    torch.manual_seed(5)
    trainer = _real_trainer(lanes=False)
    batch = trainer.start_states.sample(2)
    variants = build_variants(batch.state, trainer.counterfactual)
    assert any(len(g["members"]) >= 3 for g in variants.groups)
    imagined = trainer.imagine(batch, variants=variants)
    actions = imagined["actions"][0, 0]                      # [N,K] executed accel, first decision
    for (start, slot), row in variants.individual_rows.items():
        # Same start, same noise: CAV `slot` proposes the same action alone as with all CAVs.
        torch.testing.assert_close(actions[row, slot], actions[start, slot])
    for group in variants.groups:
        for flipped, row in group["flips"].items():
            for member in group["members"]:
                if member != flipped:                        # G\i members: same draws as in G
                    torch.testing.assert_close(actions[row, member], actions[group["row"], member])
    for start, row in variants.base_row.items():
        native = idm_native_accel({k: v[start:start + 1] for k, v in batch.state.items()})
        from wms.control.safety import safe_accel
        expected = safe_accel({k: v[start:start + 1] for k, v in batch.state.items()}, native,
                              accel_max=NATIVE_CAV_ACCEL_MAX_MPS2)[0]   # native CAV ceiling
        cav = batch.state["is_cav"][start]
        torch.testing.assert_close(imagined["actions"][0, 0, row][cav], expected[cav])


def test_update_logs_cooperation_terms_by_size_and_penetration():
    torch.manual_seed(6)
    trainer = _real_trainer()
    trainer.update_count = 1500                               # cooperation-heavy curriculum
    metrics = trainer.update()
    for key in ("lane_pg_loss", "lane_entropy", "lane_change_rate", "B", "D_i", "D_G", "D_all",
                "C_G", "M_i", "I_i", "coop_positive_fraction", "by_group_size", "by_penetration",
                "cf_weights", "cf_rows"):
        assert key in metrics, key
    assert metrics["n_C_G"] >= 1 and math.isfinite(metrics["C_G"]) and math.isfinite(metrics["M_i"])
    assert metrics["cf_weights"]["coop"] == pytest.approx(0.5)
    assert set(metrics["by_penetration"]) <= {"share~p02", "share~p05", "share~p10", "share~p20",
                                              "share~p40"}
    assert metrics["lane_decisions_with_options"] > 0
    assert trainer.policy.lane_head[-1].weight.grad.abs().sum() > 0
    assert math.isfinite(metrics["actor_grad_norm"]) and metrics["actor_grad_norm"] > 0
    detail = trainer.validate_detail(trainer.start_states)
    assert detail["coop_gain_val"] is not None and detail["score"] == detail["D_all"]


def test_imagined_lane_change_is_applied_at_decision_tick_and_held():
    from wms.control.bridge import make_cav_controller
    from wms.control.safety import safe_accel

    def always_right(obs, signal, *, lane_mask=None, **kwargs):
        zero = torch.zeros(obs.cav_mask.shape)
        choice = torch.full(obs.cav_mask.shape, LANE_RIGHT, dtype=torch.long)
        return PolicyOutput(zero, choice, zero, zero, zero, zero, torch.ones_like(choice))

    x = state([-500.0], [0])
    controller = make_cav_controller(always_right, lambda *a: torch.zeros(1, 1, 8), safe_accel,
                                     lane_rules=LaneRules())
    out = controller({**x, "physics_step_index": 0})
    assert out[0, 0, 1] == 1                                   # changed at the decision tick
    nxt = {**x, "lane": torch.tensor([[1]]), "physics_step_index": 1}
    assert controller(nxt)[0, 0, 1] == 1                       # held (no change on the odd tick)


def test_reports_carry_previous_command_and_lane_intent():
    from wms.control.bridge import make_cav_controller
    from wms.control.contract import REPORT_FEATURES
    from wms.control.safety import safe_accel

    seen = []

    def policy(obs, signal, *, lane_mask=None, **kwargs):
        seen.append(obs.reports.clone())
        accel = torch.where(obs.cav_mask, torch.full(obs.cav_mask.shape, -1.0), torch.zeros(obs.cav_mask.shape))
        choice = torch.tensor([[LANE_RIGHT, LANE_STAY]])
        return PolicyOutput(accel, choice, accel, accel, accel, accel, torch.ones_like(choice))

    x = state([-500.0, -800.0], [0, 1])
    controller = make_cav_controller(policy, lambda *a: torch.zeros(1, 2, 8), safe_accel,
                                     lane_rules=LaneRules())
    controller({**x, "physics_step_index": 0})
    controller({**x, "lane": torch.tensor([[1, 1]]), "physics_step_index": 1})
    controller({**x, "lane": torch.tensor([[1, 1]]), "physics_step_index": 2})
    cmd, intent = REPORT_FEATURES.index("cmd_accel_mps2"), REPORT_FEATURES.index("lane_intent")
    assert seen[0][0, 1, 0, cmd] == 0 and seen[0][0, 1, 0, intent] == 0     # nothing commanded yet
    assert seen[1][0, 1, 0, cmd] == pytest.approx(-1.0)                      # CAV 1 hears CAV 0
    assert seen[1][0, 1, 0, intent] == 1.0 and seen[1][0, 0, 0, intent] == 0.0


# ------------------------------------------------------------------ checkpoints / deployment
def test_checkpoint_v3_round_trip_executor_lanes_and_legacy_upgrade(tmp_path, monkeypatch):
    from wms.control import policy as policy_module
    from wms.control.contract import LEGACY_F_REPORT

    torch.manual_seed(7)
    trainer = _real_trainer(hidden=64)          # deployment loader uses the default sizes
    monkeypatch.setattr("wms.control.dreamer.checked_path", lambda p, **k: __import__("pathlib").Path(p))
    path = tmp_path / "p.pt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    assert payload["format"] == "c4_dreamer/3" and payload["lane_head"] is True
    with torch.no_grad():
        trainer.policy.lane_head[-1].bias.copy_(torch.tensor([-5.0, 5.0, 5.0]))
    trainer.save_checkpoint(path)
    exe = policy_module.load_policy_fn(str(path))
    assert exe.has_lane_head
    x = state([-500.0, -900.0, -300.0], [0, 1, 1], cav=[True, True, False])
    accel, lanes = exe(x, with_lanes=True)
    assert lanes.tolist() == [[1, 0, -1]] and math.isnan(accel[0, 2])
    moved = {**x, "lane": torch.tensor([[1, 0, 1]])}                # SUMO executed both changes
    assert exe(moved, with_lanes=True)[1].tolist() == [[-1, -1, -1]]   # dwell: 1 s after the change
    exe.reset()
    assert exe(x).shape == (1, 3)
    # A /2 actor (11 report features) loads with zero weights on the new features: same output.
    state_dict = {k: v.clone() for k, v in trainer.policy.state_dict().items()}
    state_dict["report_encoder.0.weight"] = state_dict["report_encoder.0.weight"][:, :LEGACY_F_REPORT]
    torch.save({**payload, "format": "c4_dreamer/2", "policy": state_dict}, tmp_path / "old.pt")
    old = policy_module.load_policy_fn(str(tmp_path / "old.pt"))
    weight = old.policy.report_encoder[0].weight
    assert weight.shape[-1] == LEGACY_F_REPORT + 2 and not weight[:, LEGACY_F_REPORT:].any()


# ------------------------------------------------------------------ SUMO lane commands
class FakeTraci:
    def __init__(self):
        self.changes, self.modes = [], {}
        self.vehicle = SimpleNamespace(changeLane=lambda vid, idx, dur: self.changes.append((vid, idx, dur)),
                                       getLaneChangeMode=lambda vid: 1621,
                                       setLaneChangeMode=lambda vid, m: self.modes.__setitem__(vid, m))
        self.lane = SimpleNamespace(getLength=lambda lane_id: 1000.0)


def test_env_lane_command_mapping_without_sumo():
    from traffic_shepherd.c4.env import LC_MODE_POLICY, CavSumoEnv

    env = object.__new__(CavSumoEnv)
    env.api = FakeTraci()
    env._lane_lengths, env._lc_original = {}, {}
    env._table = {
        "up_left": dict(road_id="1191554636", lane_id="1191554636_1", lane_position_m=100.0, speed_mps=20.0),
        "merge_left": dict(road_id="1487606179", lane_id="1487606179_2", lane_position_m=10.0, speed_mps=20.0),
        "ramp": dict(road_id="1487606179", lane_id="1487606179_0", lane_position_m=20.0, speed_mps=20.0),
        "main_right": dict(road_id="1191554636", lane_id="1191554636_0", lane_position_m=100.0, speed_mps=20.0),
        "near_end": dict(road_id="1191554636", lane_id="1191554636_1", lane_position_m=975.0, speed_mps=20.0),
        "junction": dict(road_id=":J_0", lane_id=":J_0_0", lane_position_m=1.0, speed_mps=20.0),
        "hdv": dict(road_id="1191554636", lane_id="1191554636_1", lane_position_m=50.0, speed_mps=20.0),
    }
    targets = {"up_left": 1, "merge_left": 1, "ramp": 1, "main_right": 2, "near_end": 1,
               "junction": 0, "hdv": 1}
    policy = set(targets) - {"hdv"}
    env._lane_control(policy, set(env._table))
    assert env.api.modes == {vid: LC_MODE_POLICY for vid in policy}     # HDV untouched
    issued, counts = env._issue_lane_commands(targets, policy)
    assert sorted(env.api.changes) == [("merge_left", 1, 1.0), ("ramp", 1, 1.0), ("up_left", 0, 1.0)]
    assert counts == {"issued": 3, "refused_not_allowed": 1, "deferred_edge_end": 1,
                      "refused_junction": 1, "refused_not_policy": 1}
    assert issued["up_left"] == (1, 0)
    env._lane_control({"ramp"}, set(env._table))                      # released: restored
    assert env.api.modes["up_left"] == 1621 and env.api.modes["ramp"] == LC_MODE_POLICY


@pytest.mark.skipif(not HAS_SUMO, reason="SUMO not installed")
def test_env_executes_lane_commands_in_sumo(tmp_path):
    import numpy as np

    from traffic_shepherd.c4.env import LC_MODE_POLICY, CavSumoEnv

    env = CavSumoEnv(k_slots=192, physics_log=False, scenario_root=tmp_path / "scn")
    x = env.reset(seed=4107, demand_vph=2700, penetration=0.4, warmup_s=120, horizon_s=10,
                  controller="direct", episode_dir=tmp_path / "ep")
    totals = {}
    modes_ok = True
    try:
        while True:
            cav = (x["present"] & x["is_cav"])[0].numpy()
            lane = x["lane"][0].numpy()
            s = x["s_m"][0].numpy()
            accel = np.where(cav, 0.0, np.nan)[None]
            target = np.full((1, 192), -1)
            mainline = cav & ((lane == 0) | (lane == 1)) & (s < -200)
            target[0, mainline] = 1 - lane[mainline]                 # swap mainline lanes
            x, info = env.step(accel, lanes=target)
            for key, value in info["lane_commands"].items():
                totals[key] = totals.get(key, 0) + value
            if not info["done"]:
                ids = env.slot_vehicle_ids()
                for slot in np.flatnonzero(cav).tolist():
                    vid = ids[slot]
                    if vid in env._lc_original and vid in env.api.vehicle.getIDList():
                        modes_ok &= env.api.vehicle.getLaneChangeMode(vid) == LC_MODE_POLICY
            else:
                break
    finally:
        env.close()
    assert modes_ok and totals.get("issued", 0) > 0 and totals.get("completed", 0) > 0
    assert info["summary"]["collisions"] == 0
    from traffic_shepherd.c4.recorder import load_episode
    ep = load_episode(tmp_path / "ep")
    assert (ep["lane_command_target"] >= 0).sum() == totals["issued"]
