"""Offline native-init invariants. No simulator, traffic data, or evaluation seeds."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wms.control.bridge import make_cav_controller
from wms.control.counterfactual import native_fn
from wms.control.nativeinit import (NativeResidualCommand, compose_native, panel_gate,
                                    select_update, smoke_guard, execute_sumo_residual,
                                    verify_collection_gate, pair_merge_streams)
from wms.control.obs import build_observations
from wms.control.policy import (CavPolicy, PolicyExecutor, RsuGroupEncoder,
                                load_policy_components, load_policy_fn)
from wms.control.safety import safe_accel
from wms.control.train_cli import GroupSignal
from wms.model import ModelSpec, WmSModel
from wms.physics.config import PhysicsConfig


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def scene():
    s = torch.tensor([[30., 5., 90., -400., -420., 0.]])
    return dict(s_m=s, v_mps=s.new_tensor([[5., 2., 8., 12., 9., 0.]]),
                a_mps2=torch.zeros_like(s), length_m=torch.full_like(s, 5.),
                lane=torch.tensor([[2, 1, 1, 0, 0, -1]]),
                present=torch.tensor([[True, True, True, True, True, False]]),
                is_cav=torch.tensor([[True, False, False, True, False, False]]))


def networks(residual=True):
    torch.manual_seed(101)
    p = CavPolicy(lane_head=False, residual_on_native=residual, accel_max=1.2)
    return p, GroupSignal(RsuGroupEncoder())


def test_zero_residual_is_exact_even_with_stochastic_requests_and_gradients_are_residual_only():
    p, enc = networks()
    st = scene()
    obs = build_observations(st)
    signal = enc(obs, st["s_m"], None)
    out, logp = p(obs, signal, deterministic=False)
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.equal(logp, torch.zeros_like(logp))
    out.sum().backward()
    trained = {n for n, v in p.named_parameters() if v.grad is not None and v.grad.abs().sum() > 0}
    assert trained == {"residual_head.weight", "residual_head.bias"}
    assert all(v.grad is None for v in enc.parameters())
    assert {n for n, v in p.named_parameters() if v.requires_grad} == trained
    optim = torch.optim.Adam([v for v in p.parameters() if v.requires_grad], lr=1e-3)
    optim.step()
    assert p(obs, signal)[0].abs().max() > 0


@pytest.mark.parametrize("kind", ["idm", "head"])
@pytest.mark.parametrize("law", ["strict", "hybrid"])
def test_initial_full_wm_rollout_is_bit_identical_to_native_arm(kind, law):
    torch.manual_seed(17)
    model = WmSModel(ModelSpec(rollout_mode="driver_params", lane_model="hazard_v5",
                              merge_decision="first_passage", lane_decision="sample",
                              merge_passage_law=law),
                     physics=PhysicsConfig(secure_gap_veto=True)).eval().requires_grad_(False)
    p, enc = networks()
    state = {k: v.expand(2, -1).clone() for k, v in scene().items()}
    pair_merge_streams(state, torch.tensor([0, 0]), 1, 965149)
    active = torch.zeros_like(state["is_cav"])
    active[0] = state["is_cav"][0]
    c = make_cav_controller(p, enc, safe_accel, act_mask=active,
                            native=native_fn(kind, a_max=1.2), native_accel_max=1.2,
                            native_lane_change=True, native_ramp_merge="wm_law", symmetric_exec_cap=True)
    assert torch.equal(c.native_lane_mask(state)[0], c.native_lane_mask(state)[1])
    with torch.no_grad():
        for tick in range(20):
            out = model.step(state, k_steps=tick, cav_controller=c,
                             lane_uniform=torch.full_like(state["s_m"], .5))
            for key in ("next_s_m", "next_v_mps", "next_lane", "accel_mps2"):
                assert torch.equal(out[key][0], out[key][1]), (tick, kind, key)
            state.update(s_m=out["next_s_m"], v_mps=out["next_v_mps"],
                         lane=out["next_lane"], a_mps2=out["accel_mps2"])


def test_checkpoint_loader_preserves_action_semantics_and_executor_zero(tmp_path):
    p, enc = networks()
    path = tmp_path / "policy.pt"
    torch.save({"format": "c4_dreamer/3", "policy": p.state_dict(),
                "group_encoder": enc.state_dict(), "lane_control": "native",
                "residual_on_native": True}, path)
    loaded, group, rules, _ = load_policy_components(path)
    assert rules is None
    with pytest.raises(FileNotFoundError):
        load_policy_fn(path)
    from traffic_shepherd.c4.policy_pairs import load_pair_policy_factory
    with pytest.raises(FileNotFoundError):
        load_pair_policy_factory(str(path), None, None)
    exe = PolicyExecutor(loaded, group)
    command, lanes = exe(scene(), with_lanes=True)
    assert lanes is None and isinstance(command, NativeResidualCommand)
    assert (command.residual[np.isfinite(command.residual)] == 0).all()
    with pytest.raises(TypeError, match="not absolute"):
        np.asarray(command)
    with pytest.raises(ValueError, match="noise"):
        PolicyExecutor(loaded, group, extra_accel_noise=.1)


def load_before(name):
    import sys
    root = os.environ.get("NATIVEINIT_BASELINE")
    assert root, "baseline source required for bit-identity test"
    spec = importlib.util.spec_from_file_location("wms.control._before_" + name,
                                                Path(root) / "src/wms/control" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("stochastic", [False, True])
def test_flag_off_actor_and_executor_bit_identity(stochastic):
    before = load_before("policy")
    actors = []
    for cls in (before.CavPolicy, CavPolicy):
        torch.manual_seed(29)
        actors.append(cls())
    assert actors[0].state_dict().keys() == actors[1].state_dict().keys()
    assert all(torch.equal(v, actors[1].state_dict()[k]) for k, v in actors[0].state_dict().items())
    st = scene()
    obs = build_observations(st)
    group = torch.zeros(1, 6, 8)
    outputs = []
    for p in actors:
        torch.manual_seed(73)
        outputs.append(p(obs, group, deterministic=not stochastic,
                         lane_mask=torch.ones(1, 6, 3, dtype=torch.bool)))
    assert all(torch.equal(a, b) for a, b in zip(*outputs))
    encoder = RsuGroupEncoder()
    old, new = before.PolicyExecutor(actors[0], encoder), PolicyExecutor(actors[1], encoder)
    a, al = old(st, with_lanes=True)
    b, bl = new(st, with_lanes=True)
    assert np.array_equal(a, b, equal_nan=True) and np.array_equal(al, bl)


def test_flag_off_bridge_bit_identity():
    before = load_before("bridge")
    p, enc = networks(False)
    masks = torch.zeros_like(scene()["is_cav"])
    traces = []
    for make in (before.make_cav_controller, make_cav_controller):
        c = make(p, enc, safe_accel, act_mask=masks, native=native_fn("idm", a_max=1.2),
                 symmetric_exec_cap=True)
        state = copy.deepcopy(scene())
        result = []
        for tick in range(6):
            torch.manual_seed(tick)
            out = c({**state, "physics_step_index": tick})
            result.append(out)
            state.update(s_m=out[..., 0], lane=out[..., 1].long(), v_mps=out[..., 2])
        traces.append(torch.stack(result))
    assert torch.equal(*traces)


def test_flag_off_training_metrics_optimizer_and_all_tensors_are_bit_identical(monkeypatch):
    from . import test_c4_dreamer as fixture
    from wms.control.dreamer import DreamerTrainer
    before = load_before("dreamer")
    states, metrics = [], []
    for cls in (before.DreamerTrainer, DreamerTrainer):
        fixture._seed(123)
        monkeypatch.setattr(fixture, "DreamerTrainer", cls)
        trainer = fixture._trainer()
        rows = [trainer.update() for _ in range(3)]
        metrics.append([{k: v for k, v in row.items() if k != "wall_time_s"} for row in rows])
        states.append({"policy": trainer.policy.state_dict(), "group": trainer.group_encoder.state_dict(),
                       "critic": trainer.critic.state_dict(), "target": trainer.target_critic.state_dict(),
                       "actor_optimizer": trainer.actor_optimizer.state_dict(),
                       "critic_optimizer": trainer.critic_optimizer.state_dict(),
                       "torch_rng": torch.random.get_rng_state()})
    fixture._assert_equal(metrics[0], metrics[1])
    fixture._assert_equal(states[0], states[1])


def test_executor_composes_engine_native_at_each_physics_tick_and_zero_is_noop():
    class Vehicle:
        speed = 5.
        def __init__(self): self.calls = []
        def getSpeed(self, vid): return self.speed if vid == "cav" else 3.
        def getLeader(self, vid): return ("leader", 10.)
        def getMinGap(self, vid): return 2.
        def getDecel(self, vid): return 2.
        def getAccel(self, vid): return 1.2
        def getAllowedSpeed(self, vid): return 30.
        def getFollowSpeed(self, vid, speed, gap, lv, lb, lid):
            assert (gap, lv, lb, lid) == (12., 3., 2., "leader")
            self.calls.append(("native", speed))
            return speed + .2
        def setSpeed(self, *args): self.calls.append(("release", *args))
        def setAcceleration(self, *args): self.calls.append(("command", *args))
    vehicle = Vehicle()
    api = SimpleNamespace(vehicle=vehicle)
    assert execute_sumo_residual(api, {"cav": 0.}, -3., 1.2) == {}
    assert vehicle.calls == []
    for speed in (5., 7.):
        vehicle.speed = speed
        assert execute_sumo_residual(api, {"cav": .25}, -3., 1.2)["cav"] == pytest.approx(.65)
    assert [v for k, *v in vehicle.calls if k == "native"] == [[5.], [7.]]
    assert all(call[1] == "cav" for call in vehicle.calls if call[0] in ("command", "release"))


def test_actual_environment_tick_preserves_zero_native_and_records_nonzero_as_direct():
    from collections import Counter
    from traffic_shepherd.c4.env import CavSumoEnv
    class Api:
        def __init__(self):
            self.t, self.v, self.override = 0., 5., None
            self.vehicle = self
            self.simulation = self
            self.calls, self.trace = [], []
        def getIDList(self): return ["cav"]
        def getArrivedIDList(self): return []
        def getSpeed(self, vid): return self.v
        def getLeader(self, vid): return None
        def getAccel(self, vid): return 1.2
        def getDecel(self, vid): return 2.
        def getAllowedSpeed(self, vid): return 30.
        def getFollowSpeed(self, vid, speed, gap, *rest):
            assert np.isinf(gap)
            return speed + .2
        def setSpeed(self, vid, speed): self.override = None
        def setAcceleration(self, vid, command, duration):
            assert duration == .5
            self.calls.append((self.t, command))
            self.override = command
        def simulationStep(self):
            a = .4 if self.override is None else self.override
            self.v += .5 * a
            self.t += .5
            self.trace.append((self.t, self.v))
            self.override = None
    class Env:
        _control_tick = CavSumoEnv._control_tick
        _accel_by_vehicle = CavSumoEnv._accel_by_vehicle
        @property
        def now(self): return self.api.t
    def make():
        e = Env()
        e.api = Api()
        e.mode, e.warmup_s, e.k = "direct", 0., 1
        e.cav_ids, e._previous_live, e._retired_cavs = {"cav"}, {"cav"}, set()
        e._table, e._lc_original = {}, {}
        e.sensor = SimpleNamespace(bind=lambda table: None, capture=lambda *a, **k: None)
        e.slow_sensor, e.fleet, e.pi_session, e.recorder = None, None, None, None
        e._direct_pi, e.direct_accel_max_mps2 = False, 1.2
        e.frame = SimpleNamespace(present=np.array([True]), is_cav=np.array([True]), vid_index=[0])
        e.builder = SimpleNamespace(vehicle_ids=["cav"], pool=SimpleNamespace(slot_of=lambda v: 0))
        e._lane_control = lambda *a: None
        def ownership(frames, **kw):
            command = kw["wall_commands"].get("cav")
            return [SimpleNamespace(vehicle_id="cav", owner="native" if command is None else "wall",
                    fallback_reason=None, issued_target_speed_mps=None if command is None else e.api.v + command,
                    issued_acceleration_mps2=command)]
        e.ownership = SimpleNamespace(step=ownership)
        e._after_physics = lambda *a: Counter(source_wait_vehicle_seconds=0.)
        e.command_counts, e.fallback_counts, e.lane_command_counts = Counter(), Counter(), Counter()
        e.totals, e.window_totals = Counter(), Counter()
        return e
    native, zero, residual = make(), make(), make()
    native._control_tick(None)
    zero._control_tick(NativeResidualCommand(np.array([[0.]]), -3., 1.2))
    assert native.api.trace == zero.api.trace
    assert native.api.calls == zero.api.calls == []
    assert native._last_actions == zero._last_actions
    residual._control_tick(NativeResidualCommand(np.array([[.25]]), -3., 1.2))
    assert [t for t, _ in residual.api.calls] == [0., .5]
    assert all(a == pytest.approx(.65) for _, a in residual.api.calls)
    assert residual.command_counts["wall"] == 1
    assert residual.api.v > native.api.v


def test_native_envelope_preserves_native_outside_policy_bounds_and_has_gradient():
    native = torch.tensor([-5., -.2, 2.])
    r = torch.zeros(3, requires_grad=True)
    out = compose_native(native, r, -3., 1.2)
    assert torch.equal(native, out)
    out.sum().backward()
    assert r.grad[1] == 1.


def test_residual_credit_subset_releases_unselected_cavs_without_absolute_cast():
    command = NativeResidualCommand(np.array([[.2, -.4, np.nan]]), -3., 1.2)
    subset = command.subset(np.array([False, True, False]))
    assert subset.residual.tolist() == [[0., -.4, 0.]]
    assert command.residual[0, 0] == .2


def test_paired_merge_streams_cover_entrant_slots_and_preserve_saved_carry_and_rng():
    state = {"s_m": torch.zeros(4, 100)}
    before = torch.random.get_rng_state().clone()
    pair_merge_streams(state, torch.tensor([0, 1, 0, 1]), 2, 965149)
    keys = state["merge_pivot_rng_seed"]
    assert torch.equal(keys[0], keys[2]) and torch.equal(keys[1], keys[3])
    assert not torch.equal(keys[0], keys[1])
    pair_merge_streams(state, torch.tensor([0, 1, 0, 1]), 2, 965150)
    assert state["merge_pivot_rng_seed"] is keys
    assert torch.equal(before, torch.random.get_rng_state())


def test_paired_gate_catches_harm_hidden_by_own_start_and_selects_earliest_minimum(tmp_path):
    native = [100.] * 16
    assert not panel_gate([199.] * 16, native, update=10)["passed"]
    curve = [panel_gate([100.] * 16, native, update=0)]
    assert select_update(curve) is None
    curve += [panel_gate([99.] * 16, native, update=u) for u in (10, 20)]
    assert select_update(curve) == 10
    assert not panel_gate([101.] + [90.] * 15, native, update=30)["passed"]
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"checkpoint")
    receipt = {**curve[1], "checkpoint_sha256": hashlib.sha256(policy.read_bytes()).hexdigest()}
    policy.with_name("native_gate_result.json").write_text(json.dumps(receipt))
    assert verify_collection_gate({"checkpoint": str(policy)})["passed"]
    policy.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="stale"):
        verify_collection_gate({"checkpoint": str(policy)})


def test_smoke_harm_and_uncertainty_stop_but_exact_native_passes():
    assert smoke_guard([10.] * 4)["verdict"] == "harmful"
    assert smoke_guard([-10., 10., -10., 10.])["verdict"] == "inconclusive"
    assert smoke_guard([0.] * 4)["passed"]
    assert smoke_guard([-1.] * 4)["passed"]
    with pytest.raises(ValueError):
        smoke_guard([1.])


@pytest.mark.parametrize("harmful", [True, False])
def test_smoke_barrier_dispatches_and_charges_only_four_worlds_before_verdict(tmp_path, monkeypatch, harmful):
    from .test_fastloop import fixture, fast, loop
    cfg = fixture(tmp_path)
    cfg["policy"]["native_init"] = True
    cfg["collect"].update(worlds_per_cycle=6, branch_points=2)
    root = Path(cfg["loop_root"])
    state = loop.load_state(cfg)
    dispatched = []
    def world(loop_, cfg_, state_, ph, root_, item, box, attempt):
        dispatched.append(item["index"])
        if item["index"] >= 4:
            assert loop.read_json(root / "worlds_c00.json")["native_smoke"]["passed"]
        artifact = root / ("artifact_" + str(item["seed"]))
        (artifact / "pairs").mkdir(parents=True)
        entries, hashes = [], {}
        for start in range(2):
            for arm in ("B", "POLICY"):
                rel = f"summaries/{start}_{arm}/summary.json"
                path = artifact / rel
                path.parent.mkdir(parents=True)
                delta = (10 if harmful else -10) if arm == "POLICY" else 0
                path.write_text(json.dumps({"ledger": {"J_vehicle_seconds": 100 + delta}}))
                hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
                entries.append({"source": "@artifact/" + str(Path(rel).parent),
                                "pair_key": f"{item['seed']}@{start}", "branch": {"kind": arm}})
        manifest = artifact / "pairs/C4_EPISODES_MANIFEST.json"
        manifest.write_text(json.dumps({"episodes": entries}))
        hashes["pairs/C4_EPISODES_MANIFEST.json"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        (artifact / "HASHES.json").write_text(json.dumps(hashes))
        return {"artifact": str(artifact), "case": item, "collect": {"branch_collisions": 0,
                                                                    "branch_teleports": 0}}
    monkeypatch.setattr(fast, "_world", world)
    per_world = loop.collect_sim_s(cfg) // 6
    if harmful:
        with pytest.raises(loop.Refused, match="harmful"):
            fast.collect(loop, cfg, state, {"attempt": 1}, root)
        assert sorted(dispatched) == list(range(4))
        assert state["sim_s_charged"] == per_world * 4
        with pytest.raises(loop.Refused, match="scientific stop"):
            fast.collect(loop, cfg, state, {"attempt": 2}, root)
        assert len(dispatched) == 4
    else:
        result = fast.collect(loop, cfg, state, {"attempt": 1}, root)
        assert len(result["worlds"]) == 6
        assert state["sim_s_charged"] == per_world * 6
    charges = [json.loads(line) for line in (root / "seed_ledger.jsonl").read_text().splitlines()
               if '"sim_s_charged"' in line]
    assert len(charges) == (4 if harmful else 6)
