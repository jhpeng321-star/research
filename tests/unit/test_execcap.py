"""EXECCAP (EXECCAP_NOTES.md): SUMO-consistent execution limit of command-driven CAVs in WM-S.

In SUMO (speedMode 27) a commanded CAV realises ~min(command, its own vType IDM acceleration).
With ``model.cav_exec_cap = "idm"`` every command executor of imagination (bridge, coop-imagine
callable, trainer inline controller, direct action timelines) realises
``min(safe command, physics-core IDM accel of the slot, (v0 - v)/dt, accel setting)``; "off"
(default) leaves every path bit-identical.  CPU only, tiny randomly initialised WM-S.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from wms.control.bridge import make_cav_controller
from wms.control.contract import PHYSICS_DT_S, RewardWeights
from wms.control.coop_imagine import CallableCavController
from wms.control.counterfactual import CounterfactualConfig
from wms.control.dreamer import DreamerConfig, DreamerTrainer
from wms.control.lanes import LaneRules
from wms.control.obs import build_observations
from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic
from wms.control.reward import team_reward
from wms.control.safety import safe_accel
from wms.control.train_cli import GroupSignal
from wms.model import cav as cav_motion
from wms.model.controllers import DIRECT_ID
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.residual_net import HdvConfig
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel

from .test_c4_srcwait_imagination import Starts


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _world_model():
    torch.manual_seed(11)
    model = WmSModel(ModelSpec(rollout_mode="hybrid", history_hidden=16, k_hidden=8),
                     hdv=HdvConfig(d_model=16, n_heads=2, n_layers=2, ffn=32))
    return model.eval()


def _state(s, v, lane, cav):
    k = len(s)
    return {"s_m": torch.tensor([s], dtype=torch.float32),
            "v_mps": torch.tensor([v], dtype=torch.float32), "a_mps2": torch.zeros(1, k),
            "lane": torch.tensor([lane], dtype=torch.long),
            "present": torch.ones(1, k, dtype=torch.bool), "is_cav": torch.tensor([cav]),
            "length_m": torch.full((1, k), 5.0)}


def _params(model, k, **overrides):
    """Explicit per-slot driver settings [1, K] (defaults of the physics core, overridable)."""
    defaults = model._default_driver_parameters(torch.zeros(1, k, 7), 1, k)[0, 0]
    out = {name: torch.full((1, k), float(defaults[i])) for i, name in enumerate(PARAMETER_NAMES)}
    for name, value in overrides.items():
        out[name] = torch.as_tensor(value, dtype=torch.float32).reshape(1, -1).expand(1, k).clone()
    return out


def _constant_policy(value):
    def policy(obs, signal):
        return torch.full(obs.cav_mask.shape, float(value)), torch.zeros(obs.cav_mask.shape)
    policy.accel_max = 2.6
    return policy


def _group(obs, s, field=None):
    return torch.zeros(s.shape[0], 4)


def _bridge(value):
    return make_cav_controller(_constant_policy(value), _group, safe_accel, accel_max=2.6)


def _step(model, state, controller, params, mode):
    model.cav_exec_cap = mode
    try:
        return model.step(dict(state), k_steps=0, driver_params=params, cav_controller=controller)
    finally:
        model.cav_exec_cap = "off"


def _cav_accel(out, slot=0):
    return float(out["accel_mps2"][0, slot].detach())


# ------------------------------------------------------------------ switch / helpers
def test_modes_are_validated_and_default_is_off():
    model = _world_model()
    assert model.cav_exec_cap == "off"
    assert cav_motion.set_exec_cap(model, "IDM") == "idm" and model.cav_exec_cap == "idm"
    with pytest.raises(ValueError):
        cav_motion.check_exec_cap_mode("sumo")
    with pytest.raises(ValueError):
        DreamerConfig(cav_exec_cap="bogus")
    assert DreamerConfig().cav_exec_cap == "off"
    accel = torch.tensor([[1.0, -2.0]])
    assert cav_motion.apply_exec_cap({}, accel) is accel           # no cap key: identity


def test_off_never_computes_the_cap_nor_passes_it_to_callbacks(monkeypatch):
    model = _world_model()

    def boom(*args, **kwargs):
        raise AssertionError("cap computed with cav_exec_cap='off'")

    monkeypatch.setattr(WmSModel, "_exec_cap", boom)
    seen = []
    bridge = _bridge(2.6)

    def spy(state):
        seen.append(cav_motion.EXEC_CAP_KEY in state)
        return bridge(state)

    x = _state((-600.0, -560.0, -500.0), (15.0, 10.0, 20.0), (1, 1, 0), (True, False, True))
    roll = model.rollout(dict(x), n_steps=6, cav_controller=spy)
    assert seen and not any(seen)
    assert torch.isfinite(roll["positions"]).all()


def test_off_rollout_is_bit_identical_after_toggling_and_with_the_legacy_executor():
    """Off: same rollout (a) before/after an 'idm' run on the same model and (b) as a hand-written
    legacy executor (safety -> direct_command_motion, no cap) -- the pre-EXECCAP code path."""
    model = _world_model()
    x = _state((-600.0, -560.0, -500.0, -470.0), (15.0, 10.0, 20.0, 25.0), (1, 1, 0, 0),
               (True, False, True, False))

    def run():
        return model.rollout(dict(x), n_steps=8, cav_controller=_bridge(2.6))

    first = run()
    model.cav_exec_cap = "idm"
    capped = run()
    model.cav_exec_cap = "off"
    again = run()
    for key in ("positions", "speeds", "accel", "lanes", "masks"):
        assert torch.equal(first[key], again[key]), key
    assert not torch.equal(first["positions"], capped["positions"])   # the switch does act

    def legacy(state):
        present = state["present"].bool()
        mask = present & state["is_cav"].bool() & (state["lane"] >= 0)
        cmd = torch.where(mask, torch.full_like(state["s_m"], 2.6), torch.zeros_like(state["s_m"]))
        executed = safe_accel(state, cmd, accel_max=2.6)
        v = torch.where(present, state["v_mps"], torch.zeros_like(state["v_mps"]))
        s = torch.where(present, state["s_m"], torch.zeros_like(state["s_m"]))
        ns, nv, a, _ = cav_motion.direct_command_motion(
            s, v, torch.stack((executed, v + 2.6 * PHYSICS_DT_S), -1), PHYSICS_DT_S, -3.0, 2.6)
        return torch.stack((ns, state["lane"].to(ns.dtype), nv, a), -1)

    reference = model.rollout(dict(x), n_steps=8, cav_controller=legacy)
    for key in ("positions", "speeds", "accel"):
        assert torch.equal(first[key], reference[key]), key


# ------------------------------------------------------------------ on: physics of the cap
def test_command_behind_slow_leader_realises_at_most_the_idm_accel():
    model = _world_model()
    # CAV at 15 m/s, slow HDV leader 60 m ahead (gap 55 m) at 10 m/s: safety passes +2.6
    # (stopping distance is fine) but the CAV's own IDM wants much less.
    x = _state((-600.0, -540.0), (15.0, 10.0), (1, 1), (True, False))
    params = _params(model, 2)
    idm = float(model.physics_accel(x, params)[0, 0])
    safe = float(safe_accel(x, torch.tensor([[2.6, 0.0]]), accel_max=2.6)[0, 0])
    assert 0.0 < idm < safe - 0.5
    off = _step(model, x, _bridge(2.6), params, "off")
    on = _step(model, x, _bridge(2.6), params, "idm")
    assert _cav_accel(off) == pytest.approx(safe)
    assert _cav_accel(on) <= idm + 1e-6
    assert _cav_accel(on) == pytest.approx(idm)
    # the next state is recomputed from the executed acceleration
    assert float(on["next_v_mps"][0, 0].detach()) == pytest.approx(15.0 + idm * PHYSICS_DT_S)
    assert float(on["next_s_m"][0, 0].detach()) == pytest.approx(
        -600.0 + 0.5 * (15.0 + 15.0 + idm * PHYSICS_DT_S) * PHYSICS_DT_S)


def test_free_road_realises_min_of_command_accel_setting_and_approach_to_v0():
    model = _world_model()
    x = _state((-600.0,), (10.0,), (1,), (True,))
    # low accel setting: the vType-like limit (and IDM free term below it) bind
    params = _params(model, 1, accel=1.0, v0=30.0)
    on = _step(model, x, _bridge(2.6), params, "idm")
    free_term = 1.0 * (1.0 - (10.0 / 30.0) ** 4)        # (open road: a far virtual leader)
    expected = float(model.physics_accel(x, params)[0, 0])
    assert expected <= free_term + 1e-6
    assert _cav_accel(on) == pytest.approx(expected, rel=1e-5) and _cav_accel(on) <= 1.0
    # a small command stays the binding term
    small = _step(model, x, _bridge(0.2), params, "idm")
    assert _cav_accel(small) == pytest.approx(0.2, abs=1e-6)
    # above the desired speed the approach to v0 binds (SUMO max-speed): (v0 - v)/dt
    fast = _state((-600.0,), (12.0,), (1,), (True,))
    over = _step(model, fast, _bridge(2.6), _params(model, 1, v0=11.0), "idm")
    cap = float(cav_motion.exec_accel_cap(
        model.physics_accel(fast, _params(model, 1, v0=11.0)), fast["v_mps"], PHYSICS_DT_S,
        11.0, float(_params(model, 1)["accel"][0, 0]))[0, 0])
    assert cap == pytest.approx((11.0 - 12.0) / PHYSICS_DT_S)
    assert _cav_accel(over) == pytest.approx(max(cap, -3.0))


def test_braking_commands_pass_through():
    model = _world_model()
    x = _state((-600.0,), (20.0,), (1,), (True,))
    params = _params(model, 1)
    off = _step(model, x, _bridge(-2.0), params, "off")
    on = _step(model, x, _bridge(-2.0), params, "idm")
    assert _cav_accel(on) == _cav_accel(off) == pytest.approx(-2.0)


def test_recorded_teachers_are_unaffected():
    model = _world_model()
    x = _state((-600.0, -540.0, -500.0), (15.0, 10.0, 22.0), (1, 1, 0), (True, False, True))
    steps = 6
    teacher = torch.zeros(1, steps + 1, 3, 4)
    for i in range(steps + 1):   # teachers accelerate hard: +2.5 m/s^2 (would be capped)
        v = x["v_mps"][0] + 2.5 * PHYSICS_DT_S * i
        teacher[0, i, :, 0] = x["s_m"][0] + (x["v_mps"][0] + v) * 0.5 * PHYSICS_DT_S * i
        teacher[0, i, :, 1] = x["lane"][0].float()
        teacher[0, i, :, 2] = v
        teacher[0, i, :, 3] = 2.5
    out = {}
    for mode in ("off", "idm"):
        model.cav_exec_cap = mode
        out[mode] = model.rollout(dict(x), n_steps=steps, cav_teacher=teacher)
    model.cav_exec_cap = "off"
    for key in ("positions", "speeds", "accel"):
        assert torch.equal(out["off"][key], out["idm"][key]), key


def test_direct_action_timeline_is_capped():
    model = _world_model()
    x = _state((-600.0, -540.0), (15.0, 10.0), (1, 1), (True, False))
    params = _params(model, 2)
    action_id = torch.tensor([[DIRECT_ID, 0]])
    command = torch.tensor([[[2.6, 40.0], [0.0, 0.0]]])
    teacher = torch.zeros(1, 2, 4)
    res = {}
    for mode in ("off", "idm"):
        model.cav_exec_cap = mode
        res[mode] = model.step(dict(x), k_steps=0, driver_params=params, action_id=action_id,
                               action_command=command, cav_teacher=teacher)
    model.cav_exec_cap = "off"
    idm = float(model.physics_accel(x, params)[0, 0])
    assert float(res["off"]["accel_mps2"][0, 0]) == pytest.approx(2.6)
    assert float(res["idm"]["accel_mps2"][0, 0]) == pytest.approx(idm)


def test_coop_imagine_callable_is_capped():
    model = _world_model()
    x = _state((-600.0, -540.0), (15.0, 10.0), (1, 1), (True, False))
    params = _params(model, 2)

    def fn(full):
        return torch.full((1, full["s_m"].shape[1]), 2.6)

    res = {}
    for mode in ("off", "idm"):
        ctrl = CallableCavController([fn], torch.arange(2), 2, torch.tensor([[True, False]]),
                                     accel_max=2.6)
        res[mode] = _step(model, x, ctrl, params, mode)
    idm = float(model.physics_accel(x, params)[0, 0])
    assert _cav_accel(res["off"]) == pytest.approx(2.6)
    assert _cav_accel(res["idm"]) == pytest.approx(idm)


# ------------------------------------------------------------------ gradients
def test_gradient_flows_to_braking_commands_and_is_zero_where_the_cap_binds():
    v = torch.tensor([[15.0, 15.0]], requires_grad=True)
    cap = 0.8 - 0.05 * v                         # stand-in for the IDM ceiling (depends on state)
    command = torch.tensor([[-1.5, 2.6]], requires_grad=True)   # braking, binding
    state = {cav_motion.EXEC_CAP_KEY: cap}
    executed = cav_motion.apply_exec_cap(state, command)
    executed.sum().backward()
    assert command.grad.tolist() == [[1.0, 0.0]]
    assert v.grad[0, 1].item() == pytest.approx(-0.05) and v.grad[0, 0].item() == 0.0


def test_policy_gradient_through_capped_imagination_is_finite_and_nonzero():
    model = _world_model()
    model.requires_grad_(False)
    x = _state((-600.0, -540.0, -500.0), (15.0, 10.0, 25.0), (1, 1, 0), (True, False, True))
    params = _params(model, 3)
    gain = torch.tensor(1.0, requires_grad=True)

    def policy(obs, signal):
        # first CAV brakes (non-binding), second wants +2.6 on a free lane at speed (binding)
        base = torch.tensor([[-1.0, 0.0, 2.6]])
        return gain * base, torch.zeros_like(base)
    policy.accel_max = 2.6
    model.cav_exec_cap = "idm"
    roll = model.rollout(dict(x), n_steps=4, driver_params=params,
                         cav_controller=make_cav_controller(policy, _group, safe_accel,
                                                            accel_max=2.6))
    model.cav_exec_cap = "off"
    roll["speeds"][:, -1, 0].sum().backward()
    assert gain.grad is not None and torch.isfinite(gain.grad) and gain.grad.item() < 0


# ------------------------------------------------------------------ wiring
def _trainer(model, starts, *, cap="off", counterfactual=True):
    torch.manual_seed(3)
    settings = dict(batch_size=2, horizon=3, checkpoint_every=0, validation_every=0,
                    actor_lr=1e-3, critic_lr=1e-3, cav_exec_cap=cap)
    return DreamerTrainer(
        [copy.deepcopy(model)], CavPolicy(hidden_dim=16), GroupSignal(RsuGroupEncoder(hidden_dim=16)),
        TeamCritic(hidden_dim=16), starts, observation_builder=build_observations,
        reward_fn=team_reward, safety=safe_accel, controller_factory=make_cav_controller,
        config=DreamerConfig(**settings), reward_weights=RewardWeights(), lane_rules=LaneRules(),
        counterfactual=CounterfactualConfig(enabled=counterfactual, n_individual=2, n_group=2,
                                            n_flip=2, group_max_size=4))


def test_trainer_switch_is_applied_during_imagination_only(monkeypatch):
    model = _world_model()
    seen = []
    original = WmSModel._exec_cap

    def spy(self, *args, **kwargs):
        seen.append(self.cav_exec_cap)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(WmSModel, "_exec_cap", spy)
    trainer = _trainer(model, Starts(True), cap="idm")
    torch.manual_seed(7)
    metrics = trainer.update()
    assert seen and set(seen) == {"idm"}
    assert trainer.world_models[0].cav_exec_cap == "off"          # restored after the rollout
    assert all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values()
               if isinstance(v, (int, float)))
    seen.clear()
    off = _trainer(model, Starts(True), cap="off")
    torch.manual_seed(7)
    off.update()
    assert not seen


def test_trainer_inline_controller_is_capped(monkeypatch):
    """Without a controller factory the trainer's own integrator executes the commands; with the
    switch on it applies the ceiling to the policy CAVs (the actions it records are the executed)."""
    model = _world_model()
    calls = []
    original = cav_motion.apply_exec_cap

    def spy(state, accel, mask=None):
        calls.append(cav_motion.EXEC_CAP_KEY in state)
        return original(state, accel, mask)

    monkeypatch.setattr(cav_motion, "apply_exec_cap", spy)
    imagined = {}
    for cap in ("off", "idm"):
        calls.clear()
        torch.manual_seed(3)
        trainer = DreamerTrainer(
            [copy.deepcopy(model)], CavPolicy(hidden_dim=16),
            GroupSignal(RsuGroupEncoder(hidden_dim=16)), TeamCritic(hidden_dim=16),
            Starts(True, with_entries=False), observation_builder=build_observations,
            reward_fn=team_reward, safety=safe_accel, controller_factory=None,
            config=DreamerConfig(batch_size=2, horizon=3, checkpoint_every=0, validation_every=0,
                                 imagine_entries=False, cav_exec_cap=cap),
            reward_weights=RewardWeights())
        with torch.no_grad():
            torch.manual_seed(5)
            imagined[cap] = trainer.imagine(trainer.start_states.sample(2), deterministic=True)
        assert calls and all(c == (cap == "idm") for c in calls)
    a_off, a_on = imagined["off"]["actions"], imagined["idm"]["actions"]
    assert torch.isfinite(a_on).all()
    # first decision (same start state, same deterministic command): executed <= uncapped
    assert (a_on[:, 0] <= a_off[:, 0] + 1e-5).all()


def test_load_backend_reads_the_process_default(monkeypatch):
    import wms.model.checkpoint as checkpoint_module
    from wms.eval import backend

    def fake(path, **kwargs):
        return SimpleNamespace(model=_world_model(), legacy_config={}, projection=None)

    monkeypatch.setattr(checkpoint_module, "load_checkpoint", fake)
    monkeypatch.delenv(cav_motion.EXEC_CAP_ENV, raising=False)
    assert backend.load_backend("x.pt").model.cav_exec_cap == "off"
    monkeypatch.setenv(cav_motion.EXEC_CAP_ENV, "idm")
    assert backend.load_backend("x.pt").model.cav_exec_cap == "idm"
    assert backend.load_backend("x.pt", cav_exec_cap="off").model.cav_exec_cap == "off"
    monkeypatch.setenv(cav_motion.EXEC_CAP_ENV, "nonsense")
    with pytest.raises(ValueError):
        backend.load_backend("x.pt")


def test_final_policy_config_enables_the_cap_and_others_do_not():
    from pathlib import Path

    from wms.control.train_cli import _read_config
    repo = Path(__file__).resolve().parents[2]
    final = _read_config(repo / "configs" / "c4_policy_final.toml")
    # INFRAV5: the final policy uses fixed CAV vType settings ("idm_normal:textbook")
    assert DreamerConfig(**final["trainer"]).cav_exec_cap == "idm_normal:textbook"
    for name in ("c4_policy.toml", "c4_policy_base_p4.toml"):
        path = repo / "configs" / name
        if path.exists():
            assert DreamerConfig(**_read_config(path).get("trainer", {})).cav_exec_cap == "off"
