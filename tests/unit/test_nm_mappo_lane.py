"""N7 S-MAPPOWZL lane-capable MAPPO: config deltas, fresh lane head on u0, LCEXEC lane provider and
writer bindings (fake gate/writer), launcher option, smoke gate, sanity lane arm.  CPU only."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import torch

from wms.control.lanes import LaneRules
from wms.control.mappo.agent import ActorStep, build_networks, export_payload
from wms.control.mappo.config_helpers import build_trainer
from wms.control.mappo.fakeenv import FakeCavEnv
from wms.core.toml import loads

ROOT = Path(__file__).resolve().parents[2]
SPEED = ROOT / "configs" / "c4_mappo_speed_wz3to2.toml"
LANE = ROOT / "configs" / "c4_mappo_lane_wz3to2.toml"


def _load(name):
    key = f"nm_mappo_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "tools" / "nm_mappo" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


L = _load("launch")
S = _load("sanity_eval")


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _flat(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict) and k not in ("kwargs", "env_kwargs"):
            yield from _flat(v, prefix + (k,))
        else:
            yield prefix + (k,), v


def stub_cfg(tmp_path, src=LANE) -> Path:
    path = tmp_path / src.name
    path.write_text(re.sub(r'(?m)^api = "[^"]*"', 'api = ""', src.read_text(encoding="utf-8"), count=1),
                    encoding="utf-8")
    return path


# ------------------------------------------------------------------ fakes for the LCEXEC interface
def fake_lane_actions():
    from tools.nm_mappo.lane_actions import provisional
    return provisional()


def fake_writer(env, targets, policy_cavs, statuses=None):
    env.written = dict(targets)
    from collections import Counter
    return dict(targets), Counter(issued=len(targets))


# ------------------------------------------------------------------ config
def test_lane_config_equals_speed_config_except_lane_deltas():
    a, b = dict(_flat(loads(SPEED.read_text(encoding="utf-8")))), dict(_flat(loads(LANE.read_text(encoding="utf-8"))))
    deltas = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    assert deltas == {("policy", "lane_control"), ("policy", "lane_actions"), ("env", "kwargs")}
    assert b[("policy", "lane_control")] == "learned"
    assert b[("policy", "lane_actions")] == "tools.nm_mappo.lane_actions:lcexec"
    assert b[("env", "kwargs")] == {**a[("env", "kwargs")], "lane_writer": "lcexec"}
    assert b[("mappo", "lane_entropy_coef")] == 0.01 and b[("rollout", "iterations")] == 168


# ------------------------------------------------------------------ fresh lane head on u0
def _speed_export(tmp_path) -> Path:
    p, e, _ = build_networks(accel_max=2.6, lane_control="native")
    path = tmp_path / "u0.pt"
    torch.save(export_payload(p, e, LaneRules()), path)
    return path


def _lane_trainer(tmp_path, name):
    cfg = {"policy": {"lane_control": "learned", "accel_max_mps2": 2.6}, "env": {"name": "fake"},
           "mappo": {"epochs": 1, "minibatch_frames": 4},
           "rollout": {"iterations": 1, "episodes_per_iteration": 1, "horizon_s": 4., "warmup_s": 0.}}
    return build_trainer(cfg, workers=1, out_dir=tmp_path / name)


def test_fresh_lane_head_init_from_speed_only_export(tmp_path):
    u0 = _speed_export(tmp_path)
    src = torch.load(u0, weights_only=False)["policy"]
    with pytest.raises(ValueError, match="lane_head_init"):
        _lane_trainer(tmp_path, "a").init_from_checkpoint(u0)
    t1, t2, t3 = (_lane_trainer(tmp_path, n) for n in ("b", "c", "d"))
    r1 = t1.init_from_checkpoint(u0, lane_head_init="fresh", lane_head_seed=7)
    r2 = t2.init_from_checkpoint(u0, lane_head_init="fresh", lane_head_seed=7)
    r3 = t3.init_from_checkpoint(u0, lane_head_init="fresh", lane_head_seed=8)
    state = t1.policy.state_dict()
    assert all(torch.equal(state[k], v) for k, v in src.items())          # u0 weights strict
    assert r1["lane_head_fresh"]["keys"] == sorted(k for k in state if k.startswith("lane_head."))
    assert r1["lane_head_fresh"]["sha256"] == r2["lane_head_fresh"]["sha256"] != r3["lane_head_fresh"]["sha256"]
    assert not state["lane_head.2.weight"].any() and not state["lane_head.2.bias"].any()
    assert t1.policy.has_lane_head and t1.policy.accel_max == 2.6


# ------------------------------------------------------------------ lane provider (LCEXEC gate)
def _state(seed, k=64, junction=2):
    env = FakeCavEnv(k_slots=k, min_vehicles=k // 2)
    st = env.reset(seed=seed, demand_vph=3200, penetration=0.6, warmup_s=0, horizon_s=5)
    gen = torch.Generator().manual_seed(seed)
    st["lane"] = torch.where(st["present"], torch.randint(0, 3, st["lane"].shape, generator=gen), st["lane"])
    st["s_m"] = torch.where(st["present"], -300 + 400 * torch.rand(st["s_m"].shape, generator=gen), st["s_m"])
    idx = torch.nonzero(st["present"][0]).flatten()[:junction]
    st["lane"][0, idx] = -1                                         # junction transits block nearby
    return st


def test_vector_gate_parity_with_per_ego_reference_and_binding():
    from tools.nm_mappo import lane_actions as LA
    rules = LA.provisional_rules()
    vec = LA.VectorLaneGate(rules)
    res = LA.parity(vec, LA.per_ego_reference(rules), [_state(s) for s in range(12)])
    assert res["checked"] > 300 and res["mismatched"] == 0, res
    st = _state(99, junction=0)
    st["present"][0, :] = False
    for k, (lane, s_m) in enumerate(((0, -250.0), (1, -150.0), (2, -50.0), (2, -20.0), (1, 74.0))):
        st["present"][0, k], st["is_cav"][0, k] = True, True
        st["lane"][0, k], st["s_m"][0, k], st["v_mps"][0, k] = lane, s_m, 20.0
    m = vec.feasibility(st, None, LaneRules())[0]
    assert m[0].tolist() == [True, False, True]       # far -> adjacent (right)
    assert m[1].tolist() == [True, True, True]        # adjacent -> far, and INTO the closing lane
    assert m[2].tolist() == [True, True, False]       # closing -> adjacent
    assert m[3].tolist() == [True, True, False]       # closing -> adjacent (front gap 89 m)
    assert m[4].tolist() == [True, False, False]      # adjacent within the drop edge guard: no change


def test_actorstep_uses_provider_and_trainer_plumbs_it(tmp_path):
    p, e, _ = build_networks(accel_max=2.6, lane_control="learned")
    env = FakeCavEnv(k_slots=24)
    state = env.reset(seed=3, demand_vph=2400, penetration=0.6, warmup_s=0, horizon_s=5)
    out = ActorStep(p, e, LaneRules(), lane_actions=fake_lane_actions())(state, deterministic=False,
                                                                         generator=torch.Generator().manual_seed(0))
    changed = out["lane_changed"][0]
    assert not (changed & ~out["lane_mask"][0, :, 1:].any(-1)).any()
    cfg = {"policy": {"lane_control": "learned", "lane_actions": "tests.unit.test_nm_mappo_lane:fake_lane_actions"},
           "env": {"name": "fake"}, "mappo": {"epochs": 1, "minibatch_frames": 4},
           "rollout": {"iterations": 1, "episodes_per_iteration": 2, "horizon_s": 4., "warmup_s": 0.}}
    trainer = build_trainer(cfg, workers=1, out_dir=tmp_path / "run")
    assert trainer._spec()["lane_actions"] == "tests.unit.test_nm_mappo_lane:fake_lane_actions"
    assert trainer.run()["iterations_completed"] == 1


# ------------------------------------------------------------------ LCEXEC lane writer binding
def test_lane_writer_binding_delegates_and_unbound_raises():
    from tools.nm_mappo import n1_adapter as A
    with pytest.raises(RuntimeError, match="LCEXEC lane writer"):
        A.resolve_writer("tools.coopmwm.lc_runtime:no_such_writer")

    class Base:
        def _issue_lane_commands(self, targets, policy_cavs, statuses=None):
            raise AssertionError("frozen writer must not be used")

    env = A.lane_writer_class(Base, fake_writer)()
    issued, counts = env._issue_lane_commands({"main.1": 2}, {"main.1"})
    assert issued == {"main.1": 2} and counts["issued"] == 1 and env.written == {"main.1": 2}
    assert A.resolve_writer("tests.unit.test_nm_mappo_lane:fake_writer") is fake_writer


# ------------------------------------------------------------------ launcher
def test_plan_lane_option_and_smoke_gate(tmp_path):
    cfg = stub_cfg(tmp_path)
    plan = L.make_plan(option="mappowzl168_u0fresh", host="cpuQ", run_dir=tmp_path, init="u0.pt",
                       python="py", sumo_home=None, allow_stub=True, config=cfg, workers=46,
                       phase_cap_sim_s=7_000_000, other_charges_sim_s=32_400)
    argv = plan["argv"]["production"]
    assert argv[argv.index("--init-lane-head"):argv.index("--init-lane-head") + 4] == [
        "--init-lane-head", "fresh", "--init-lane-head-seed", "20261106"]
    assert "--init-lane-head" in plan["argv"]["smoke"]
    assert plan["lane_control"] == "learned" and plan["init"]["lane_head"] == "fresh"
    assert plan["lane_actions"] == "tools.nm_mappo.lane_actions:lcexec"
    with pytest.raises(SystemExit, match="does not match"):
        L.make_plan(option="mappowz168_u0", host="cpuQ", run_dir=tmp_path, init="u0.pt", python="py",
                    sumo_home=None, allow_stub=True, config=cfg)
    run = tmp_path / "run"
    (run / "smoke").mkdir(parents=True)
    L.write_json_new(run / "PLAN.json", plan)
    (run / "smoke/RUN_SUMMARY.json").write_text(json.dumps({"status": "completed"}))
    row = {"type": "iter", "episodes": 2, "failed_episodes": 0, "collisions": 0, "teleports": 0,
           "lane_commands_issued": 7, "policy_lane_change_rate": 0.01}
    (run / "smoke/log.jsonl").write_text(json.dumps(row) + "\n")
    assert L.smoke_passed(run) == (True, "PASS")          # lane-on: lane commands are expected
    (run / "smoke/log.jsonl").write_text(json.dumps({**row, "collisions": 1}) + "\n")
    assert L.smoke_passed(run)[0] is False


# ------------------------------------------------------------------ sanity lane arm (fake env)
def test_sanity_lane_arm_on_fake_env(tmp_path):
    cfg = stub_cfg(tmp_path)
    text = cfg.read_text(encoding="utf-8").replace('"tools.nm_mappo.lane_actions:lcexec"',
                                                  '"tests.unit.test_nm_mappo_lane:fake_lane_actions"')
    cfg.write_text(text, encoding="utf-8")
    p, e, _ = build_networks(accel_max=2.6, lane_control="learned")
    wzl = tmp_path / "wzl.pt"
    torch.save(export_payload(p, e, LaneRules()), wzl)
    out = tmp_path / "san"
    assert S.main(["--config", str(cfg), "--fake", "--policy", f"wzl={wzl}", "--out", str(out),
                   "--horizon-s", "3", "--warmup-s", "0"]) == 0
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    assert {r["lane_control"] for r in rows if r["arm"] == "wzl"} == {"learned"}
    summary = json.loads((out / "SUMMARY.json").read_text())
    assert summary["flags"]["S1_safety"] and summary["failed_rows"] == 0


# ------------------------------------------------------------------ LCEXEC final exports (subprocess: real N1 map)
_LCEXEC_PROBE = r"""
import json, math, torch
from tools.nm_mappo import n1_adapter as A, lane_actions as LA
A.install("wz3to2")
from tools.coopmwm import lc_families, lc_runtime
from wms.control.mappo.fakeenv import FakeCavEnv
vec = LA.lcexec()                                     # binds lc_runtime.mappo_lane_rules
assert isinstance(vec.rules, LA.GateRules) and "LCEXEC" in vec.rules.source
states = []
for seed in range(24):
    env = FakeCavEnv(k_slots=160, min_vehicles=90)
    st = env.reset(seed=seed, demand_vph=3200, penetration=0.6, warmup_s=0, horizon_s=5)
    g = torch.Generator().manual_seed(seed)
    s = -300 + 420 * torch.rand(st["s_m"].shape, generator=g)
    lane = torch.randint(0, 3, st["lane"].shape, generator=g)
    lane = torch.where((lane == 2) & (s >= 75.44), torch.ones_like(lane), lane)   # closing lane ends at the drop
    if seed % 3 == 0:
        lane[0, torch.nonzero(st["present"][0]).flatten()[:2]] = -1               # junction rows block nearby
    st["s_m"] = torch.where(st["present"], s, st["s_m"])
    st["lane"] = torch.where(st["present"], lane, st["lane"])
    st["v_mps"] = torch.where(st["present"], 5 + 25 * torch.rand(st["v_mps"].shape, generator=g), st["v_mps"])
    states.append(st)
res = LA.parity(vec, lc_families.mappo_lane_gate, states)
writer = A.resolve_writer()
from traffic_shepherd.c4 import env as E
inst = object.__new__(A.lane_writer_class(E.CavSumoEnv, writer))
glob = lc_runtime._frozen_globals(inst)
print(json.dumps({**res, "writer": writer.__name__, "frozen_writer_found": "LANE_TABLE" in glob,
                  "transitions": [[t.source, t.target, t.guard] for t in vec.rules.transitions]}))
"""


def test_lcexec_final_exports_parity_and_writer(tmp_path):
    import os
    import subprocess
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]))
    env.pop("COOPMWM_MAP_SPEC", None)
    out = subprocess.run([sys.executable, "-c", _LCEXEC_PROBE], cwd=ROOT, env=env, capture_output=True,
                         text=True, timeout=1200)
    assert out.returncode == 0, out.stderr[-3000:]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["checked"] > 2000 and res["mismatched"] == 0, res
    assert res["writer"] == "mappo_lane_writer" and res["frozen_writer_found"]
    assert sorted(map(tuple, res["transitions"])) == [(0, 1, True), (1, 0, True), (1, 2, True), (2, 1, False)]
