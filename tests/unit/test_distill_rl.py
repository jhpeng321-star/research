"""DIST-RL STUDENT_RL: config = lane-on MAPPO recipe + exactly the MAPPO_CT [rollout] deltas, opt-in seed
block guard, student actor check, launcher student option, and a fake-physics one-update dry run from a
lane-on student export (lane head loaded, critic / Lagrange fresh).  CPU only, zero SUMO."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from wms.control.lanes import LaneRules
from wms.control.mappo.agent import build_networks, export_payload
from wms.core.toml import loads

ROOT = Path(__file__).resolve().parents[2]
LANE = ROOT / "configs" / "c4_mappo_lane_wz3to2.toml"
CT = ROOT / "configs" / "c4_mappo_ct_d2800p20.toml"
TEMPLATE = ROOT / "configs" / "c4_distill_rl_d2800p20.toml"
BLOCK = "999000-999099"                     # CPU-test placeholder block (never simulated)


def _load(name):
    key = f"nm_mappo_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "tools" / "nm_mappo" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


L = _load("launch")
D = _load("distill_rl")


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


def _deltas(a_path, b_text):
    a = dict(_flat(loads(Path(a_path).read_text(encoding="utf-8"))))
    b = dict(_flat(loads(b_text)))
    return a, b, {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}


def _rendered(tmp_path, stub=False) -> Path:
    out = D.render(BLOCK, tmp_path / "cfg" / "c4_distill_rl.toml")
    path = Path(out["config"])
    if stub:
        path.write_text(re.sub(r'(?m)^api = "[^"]*"', 'api = ""', path.read_text(encoding="utf-8"), count=1),
                        encoding="utf-8")
    return path


# ------------------------------------------------------------------ config
def test_config_is_lane_recipe_plus_exactly_the_mappo_ct_rollout_deltas(tmp_path):
    text = _rendered(tmp_path).read_text(encoding="utf-8")
    lane, b, d_lane = _deltas(LANE, text)
    assert d_lane == {("rollout", k) for k in ("iterations", "episodes_per_iteration", "penetrations",
                                               "demands_vph", "seeds", "max_sumo_seconds", "keep_every")}
    ct, _, d_ct = _deltas(CT, text)
    # vs MAPPO_CT: only the lane-on switch (head, LCEXEC action set, LCEXEC writer) and the seed block
    assert d_ct == {("policy", "lane_control"), ("policy", "lane_actions"), ("env", "kwargs"),
                    ("rollout", "seeds")}
    assert b[("rollout", "seeds")] == BLOCK and ct[("rollout", "seeds")] == "989200-989587"
    for key in [k for k in b if k[0] in ("mappo", "reward_weights", "lanes")] + [
            ("rollout", "horizon_s"), ("rollout", "warmup_s"), ("rollout", "seed"), ("policy", "accel_max_mps2")]:
        assert b[key] == ct[key], key
    assert b[("reward_weights", "fairness")] == 1.0 and b[("mappo", "target_kl")] == 0.03
    with pytest.raises(FileExistsError):                 # render never overwrites
        D.render(BLOCK, tmp_path / "cfg" / "c4_distill_rl.toml")


# ------------------------------------------------------------------ seed guard extension
@pytest.mark.parametrize("bad", ["989300-989310", "993100-993100", "964050-964060", "962000-962010",
                                 "977500-977501", "991000-991010", "990600-990601", "989000-989050",
                                 "12-14", "999000-999000,999000-999001", "100000-120000"])
def test_extension_refuses_banned_or_bad_blocks(bad):
    from tools.coopmwm.map_scenarios import parse_extra_train_ranges
    with pytest.raises(ValueError):
        parse_extra_train_ranges(bad)
    with pytest.raises((ValueError, SystemExit)):
        D.render(bad, Path(os.devnull + "_never"))


def test_extension_is_opt_in_per_process():
    code = ("import json;from tools.coopmwm import map_scenarios as m;"
            "print(json.dumps([m.seed_scope(999992), m.seed_scope(989300), m.seed_scope(989650),"
            " m.world(999993, 2800, 0.2)['repeat'] if m.seed_scope(999993) else None]))")
    env = {k: v for k, v in os.environ.items() if k != "COOPMWM_EXTRA_TRAIN_RANGES"}

    def run(e):
        return json.loads(subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, env=e))

    assert run(env) == [None, "train", "heldout", None]
    assert run({**env, "COOPMWM_EXTRA_TRAIN_RANGES": "999992-999993"}) == ["train", "train", "heldout", 501]


# ------------------------------------------------------------------ student actor
def _student(tmp_path, *, lane=True, critic=False, accel=2.6) -> Path:
    torch.manual_seed(3)
    p, e, c = build_networks(accel_max=accel, lane_control="learned" if lane else "native")
    with torch.no_grad():
        for prm in p.lane_head.parameters() if lane else ():
            prm.add_(0.1)                                   # a BC-trained (non-zero) lane head
    payload = export_payload(p, e, LaneRules(), trainer_tag="distill_bc")
    if critic:
        payload["critic"] = c.state_dict()
    path = tmp_path / f"student_{lane}_{critic}_{accel}.pt"
    torch.save(payload, path)
    return path


def test_student_check(tmp_path):
    good = _student(tmp_path)
    rec = D.check_student(good, D.sha256_file(good))
    assert rec["ok"] and rec["lane_head_keys"] > 0 and rec["accel_bounds"][1] == 2.6
    with pytest.raises(SystemExit, match="pinned"):
        D.check_student(good, "0" * 64)
    with pytest.raises(SystemExit, match="lane"):
        D.check_student(_student(tmp_path, lane=False))
    with pytest.raises(SystemExit, match="critic"):
        D.check_student(_student(tmp_path, critic=True))
    with pytest.raises(SystemExit, match="accel_max"):
        D.check_student(_student(tmp_path, accel=1.5))


# ------------------------------------------------------------------ launcher
def test_plan_student_option(tmp_path, monkeypatch):
    monkeypatch.delenv("COOPMWM_EXTRA_TRAIN_RANGES", raising=False)
    cfg = _rendered(tmp_path, stub=True)
    st = _student(tmp_path)
    sha = D.sha256_file(st)
    plan = L.make_plan(option="rl12x8_student", host="cpuQ", run_dir=tmp_path, init=str(st), python="py",
                       sumo_home=None, allow_stub=True, config=cfg, workers=8, init_sha256=sha,
                       extra_train_ranges=BLOCK, phase_cap_sim_s=150_000, nice=10, disk_floor_gb=25,
                       mem_floor_gb=30)
    prod = plan["argv"]["production"]
    assert prod[prod.index("--init-from") + 1] == str(st) and "--init-lane-head" not in prod
    assert prod[prod.index("--workers"):] == ["--workers", "8", "--iterations", "12",
                                              "--episodes-per-iteration", "8", "--max-sumo-seconds", "57600"]
    assert "--init-lane-head" not in plan["argv"]["smoke"]
    assert plan["init"] == {"kind": "student", "path": str(st), "sha256": sha, "lane_head": None,
                            "lane_head_seed": None}
    assert plan["env"]["COOPMWM_EXTRA_TRAIN_RANGES"] == BLOCK and plan["lane_control"] == "learned"
    assert plan["budget_sim_s"]["planned_total"] == 87_060 and plan["per_iteration_sim_s"] == 7_200
    assert plan["mem_floor_bytes"] == 30_000_000_000 and plan["disk_floor_bytes"] == 25_000_000_000
    monkeypatch.setattr(L, "mem_available_bytes", lambda: 29_000_000_000)
    assert any("MemAvailable" in r for r in L.phase_guard(plan, tmp_path))
    monkeypatch.setattr(L, "mem_available_bytes", lambda: 31_000_000_000)
    assert not any("MemAvailable" in r for r in L.phase_guard(plan, tmp_path))
    with pytest.raises(SystemExit, match="init-sha256"):
        L.make_plan(option="rl12x8_student", host="cpuQ", run_dir=tmp_path, init=str(st), python="py",
                    sumo_home=None, allow_stub=True, config=cfg, extra_train_ranges=BLOCK)
    with pytest.raises(SystemExit, match="sha256"):
        L.make_plan(option="rl12x8_student", host="cpuQ", run_dir=tmp_path, init=str(st), python="py",
                    sumo_home=None, allow_stub=True, config=cfg, init_sha256="0" * 64,
                    extra_train_ranges=BLOCK)
    # the check stage refuses a drifted student
    run = tmp_path / "run"
    run.mkdir()
    L.write_json_new(run / "PLAN.json", {**plan, "init": {**plan["init"], "sha256": "f" * 64}})
    with pytest.raises(SystemExit, match="pinned ffffffff"):
        L.main(["check", "--run-dir", str(run)])


def test_fake_physics_one_update_from_student(tmp_path):
    """What `launch.py check` runs: the exact rendered config (REAL N1 descriptor: the LCEXEC lane rules
    need the lane-drop map), CPU fake env, ONE update, strict student load; fresh process with the PLAN
    env (the seed-block extension is read at map import, as in the SUMO stages)."""
    cfg = _rendered(tmp_path)
    st = _student(tmp_path)
    src = torch.load(st, weights_only=False)
    out = tmp_path / "dry"
    env = {**os.environ, "COOPMWM_EXTRA_TRAIN_RANGES": BLOCK,
           "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
    rc = subprocess.call([sys.executable, "-m", "wms.control.mappo.train", "--config", str(cfg), "--smoke",
                          "--workers", "1", "--out", str(out), "--init-from", str(st), "--iterations", "1",
                          "--update-threads", "1"], cwd=ROOT, env=env)
    assert rc == 0
    rows = [json.loads(s) for s in (out / "log.jsonl").read_text().splitlines()]
    meta = rows[0]["meta"]
    init = meta["init_from"]
    assert init["kind"] == "exported_actor" and init["critic"] == "fresh"
    assert init["lane_head_removed_keys"] == [] and init["lane_head_fresh"] is None
    assert init["lagrange_note"].startswith("exported actor has no multiplier")
    assert meta["map"]["name"] == "wz3to2"
    its = [r for r in rows if r.get("type") == "iter"]
    assert len(its) == 1 and its[0]["episodes"] == 2 and not any(r.get("type") == "failure" for r in rows)
    final = torch.load(out / "policy_latest.pt", weights_only=False)
    assert final.get("lane_control", "learned") == "learned"
    moved = [k for k in src["policy"] if not torch.equal(src["policy"][k], final["policy"][k])]
    assert moved, "one PPO update must move the student"
    assert set(final["policy"]) == set(src["policy"])


def test_bind_final_lane_on_export(tmp_path):
    cfg = _rendered(tmp_path, stub=True)
    st = _student(tmp_path)
    run = tmp_path / "run"
    (run / "production").mkdir(parents=True)
    plan = L.make_plan(option="rl12x8_student", host="cpuQ", run_dir=run, init=str(st), python="py",
                       sumo_home=None, allow_stub=True, config=cfg, workers=8,
                       init_sha256=D.sha256_file(st), extra_train_ranges=BLOCK)
    L.write_json_new(run / "PLAN.json", plan)
    (run / "production" / "policy_00012.pt").write_bytes(st.read_bytes())
    assert L.main(["bind", "--run-dir", str(run), "--iteration", "12"]) == 0
    receipt = json.loads((run / "BOUND_00012.json").read_text())
    assert receipt["lane_control"] == "learned" and receipt["preregistered_snapshot"] is True
    (run / "production" / "policy_00008.pt").write_bytes(_student(tmp_path, lane=False).read_bytes())
    with pytest.raises(SystemExit, match="lane-on"):
        L.main(["bind", "--run-dir", str(run), "--iteration", "8"])
