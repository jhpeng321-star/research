"""N7 wz3to2 speed-only MAPPO: config parity, map binding, launcher gates, quote, sanity eval.

CPU only (fake env, fake N1 descriptor); zero SUMO, no ssh."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wms.control.lanes import LaneRules
from wms.control.mappo.agent import build_networks, export_payload
from wms.control.mappo.config_helpers import build_trainer, merge_override
from wms.control.mappo import mapbind as M
from wms.control.mappo.fakeenv import FakeCavEnv
from wms.core.toml import loads

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "c4_mappo_speed_wz3to2.toml"
ORIGINAL = ROOT / "tools" / "nm_mappo" / "ORIGINAL_SPEED_CONFIG.json"


def _load(name):
    key = f"nm_mappo_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, ROOT / "tools" / "nm_mappo" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


L = _load("launch")
Q = _load("quote")
S = _load("sanity_eval")


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class FakeMapEnv(FakeCavEnv):
    """Fake env standing in for N1's map-aware CavSumoEnv (accepts and records env kwargs)."""

    def __init__(self, k_slots: int = 24, scenario_root=None, **kwargs):
        super().__init__(k_slots=k_slots)
        self.scenario_root, self.map_kwargs = scenario_root, kwargs


def fake_n1_loader(name):
    """Stand-in for the N1 descriptor API (attribute style, alias names)."""
    assert name == "wz3to2"
    return SimpleNamespace(
        name="wz3to2", env_name="tests.unit.test_nm_mappo.FakeMapEnv",
        env_kwargs={"k_slots": 24}, seeds_train="989200-989599", heldout_seed_range=(989600, 989699),
        demand_cells_vph=[2800, 3200], penetration_cells=[0.3, 0.6], spec_sha256="ab" * 32)


def wz_cfg(api: str = "") -> dict:
    cfg = loads(CONFIG.read_text(encoding="utf-8"))
    cfg["map"]["api"] = api
    return cfg


def write_cfg(tmp_path, api: str) -> Path:
    import re
    text = re.sub(r'(?m)^api = "[^"]*"', f'api = "{api}"', CONFIG.read_text(encoding="utf-8"), count=1)
    path = tmp_path / "wz.toml"
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------ config parity
DECLARED = {("map",), ("env", "name"), ("env", "cleanup_scenarios"), ("policy", "accel_max_mps2"),
            ("rollout", "seeds"), ("rollout", "demands_vph"), ("rollout", "penetrations"),
            ("rollout", "iterations"), ("rollout", "max_sumo_seconds"), ("rollout", "keep_every")}


def _flat(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict) and k not in ("kwargs", "env_kwargs"):
            yield from _flat(v, prefix + (k,))
        else:
            yield prefix + (k,), v


def test_config_equals_original_except_declared_deltas():
    orig = json.loads(ORIGINAL.read_text())["speed_config"]
    new = wz_cfg()
    a, b = dict(_flat(orig)), dict(_flat(new))
    deltas = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    deltas = {k[:1] if k[0] == "map" else k for k in deltas}
    assert deltas == DECLARED, sorted(deltas ^ DECLARED)
    # the deltas reproduce the original command-line values
    assert new["rollout"]["iterations"] == 168 and new["rollout"]["max_sumo_seconds"] == 168 * 46 * 600
    assert new["rollout"]["demands_vph"] == [3200] and new["rollout"]["penetrations"] == [0.6]
    assert new["rollout"]["keep_every"] == 6 and all(s % 6 == 0 for s in (24, 42, 84, 168))
    assert loads(CONFIG.read_text(encoding="utf-8"))["map"]["api"] == "tools.nm_mappo.n1_adapter:descriptor"
    assert new["policy"] == {"lane_control": "native", "accel_max_mps2": 2.6}
    prov = json.loads(ORIGINAL.read_text())["_provenance"]
    assert "--iterations 169" in prov["original_command"]
    assert "--max-sumo-seconds 4664400" in prov["original_command"]
    assert prov["original_toml_sha256"].startswith("a7fc8e09")


# ------------------------------------------------------------------ map binding
def test_stub_binding_cells_and_seed_guards():
    b = M.resolve_binding({"name": "wz3to2"})
    assert b.is_stub and b.train_seeds == tuple(range(989200, 989600))
    assert b.heldout_seeds == tuple(range(989600, 989700))
    assert b.check_seeds("989200-989599")[0] == 989200
    for bad, match in (("989600", "HELD-OUT"), ("964000", "forbidden"), ("962123", "forbidden"),
                       ("977001", "forbidden"), ("4107-4148", "outside"), ("989199", "outside")):
        with pytest.raises(M.MapBindingError, match=match):
            b.check_seeds(bad)
    with pytest.raises(M.MapBindingError, match="unknown"):
        M.resolve_binding({"name": "wz3to2", "seeds": "x"})
    with pytest.raises(M.MapBindingError, match="no stub"):
        M.resolve_binding({"name": "weave"})


def test_stub_refuses_sumo_env_but_allows_fake():
    with pytest.raises(M.MapBindingError, match="STUB"):
        M.apply_map(wz_cfg())
    cfg = merge_override(wz_cfg(), wz_cfg()["smoke"])
    resolved, binding = M.apply_map(cfg)
    assert resolved["env"]["name"] == "fake"
    assert resolved["rollout"]["demands_vph"] == [3200] and resolved["rollout"]["penetrations"] == [0.6]
    sentinel = merge_override(wz_cfg(), {**wz_cfg()["smoke"], "demands_vph": "map", "penetrations": "map"})
    resolved, binding = M.apply_map(sentinel)
    assert resolved["rollout"]["demands_vph"] == [2400, 2800, 3200] == list(binding.demands_vph)
    assert resolved["rollout"]["penetrations"] == [0.2, 0.4, 0.6]
    with pytest.raises(M.MapBindingError, match="unregistered cells"):
        M.apply_map(merge_override(wz_cfg(), {"env_name": "fake", "penetrations": [0.05]}))
    bad = wz_cfg()
    bad["env"]["name"] = "traffic_shepherd.c4.env.CavSumoEnv"     # bypassing the descriptor
    with pytest.raises(M.MapBindingError, match="must be 'map'"):
        M.apply_map(bad)
    leak = merge_override(wz_cfg(), {"env_name": "fake", "seeds": "989590-989610"})
    with pytest.raises(M.MapBindingError, match="HELD-OUT"):
        M.apply_map(leak)


def test_n1_descriptor_binding_resolves_env_and_cells():
    api = "tests.unit.test_nm_mappo:fake_n1_loader"
    resolved, b = M.apply_map(wz_cfg(api))
    assert not b.is_stub and b.api == api
    assert resolved["env"]["name"] == "tests.unit.test_nm_mappo.FakeMapEnv"
    assert resolved["env"]["kwargs"] == {"ttc": False, "physics_log": False,
                                         "population": "textbook", "k_slots": 24}
    assert resolved["rollout"]["demands_vph"] == [3200]
    assert resolved["rollout"]["penetrations"] == [0.6]
    assert b.record()["heldout_seeds"] == "989600-989699"
    with pytest.raises(M.MapBindingError, match="name"):
        M.binding_from_descriptor({"name": "other"}, name="wz3to2")
    with pytest.raises(M.MapBindingError, match="overlap"):
        M.binding_from_descriptor({"name": "wz3to2", "train_seeds": "1-10", "heldout_seeds": "5",
                                   "demands_vph": [1], "penetrations": [0.1]}, name="wz3to2")


def test_no_map_table_is_unchanged(tmp_path):
    cfg = {"policy": {"lane_control": "native"}, "env": {"name": "fake"},
           "rollout": {"iterations": 1, "episodes_per_iteration": 1, "horizon_s": 3., "warmup_s": 0.}}
    trainer = build_trainer(cfg, workers=1, out_dir=tmp_path)
    assert "map" not in trainer.meta and trainer.cfg.rollout.seeds == "4107-4148"


def test_trainer_runs_through_descriptor_env_on_train_seeds_only(tmp_path):
    cfg = wz_cfg("tests.unit.test_nm_mappo:fake_n1_loader")
    cfg = merge_override(cfg, {"iterations": 1, "episodes_per_iteration": 3, "horizon_s": 4.0,
                               "warmup_s": 0.0, "epochs": 1, "minibatch_frames": 4,
                               "max_sumo_seconds": 100.0, "update_threads": 1})
    trainer = build_trainer(cfg, workers=1, out_dir=tmp_path / "run")
    assert trainer.env_spec["env_name"] == "tests.unit.test_nm_mappo.FakeMapEnv"
    assert trainer.env_spec["env_kwargs"]["population"] == "textbook"
    assert trainer.meta["map"]["api"].endswith("fake_n1_loader")
    assert trainer.run()["iterations_completed"] == 1
    rows = [json.loads(s) for s in (tmp_path / "run/log.jsonl").read_text().splitlines()]
    eps = [r for r in rows if r["type"] == "episode"]
    assert len(eps) == 3 and all(989200 <= r["seed"] <= 989599 for r in eps)
    assert all(r["demand_vph"] == 3200 and r["penetration"] == 0.6 for r in eps)
    it = [r for r in rows if r["type"] == "iter"][0]
    assert it["lane_commands_issued"] == 0 and it["policy_lane_change_rate"] == 0
    payload = torch.load(tmp_path / "run/policy_latest.pt", weights_only=False)
    assert payload["lane_control"] == "native"


# ------------------------------------------------------------------ quote
def test_quote_numbers():
    full = Q.option_quote("full169_u0")
    assert full["controlled_s"] == 4_664_400 and full["planned_sim_s"] == 6_996_600
    assert full["hard_cap_sim_s"] == 2 * 6_996_600 + 1_320
    assert Q.option_quote("ft42_u0")["planned_sim_s"] == 1_738_800
    assert Q.option_quote("ft24_u0")["planned_sim_s"] == 993_600
    # calibration reproduces the measured original: 169 it on cpuQ <= 7.83 h
    assert abs(full["wall_h"]["cpuQ"][0] - 7.83) < 0.01
    w = full["wall_h"]
    assert w["cpuV"][0] < w["cpuQ"][0] < w["cpuV_shared12"][0]
    assert all(lo < hi for lo, hi in w.values())
    assert Q.sanity_eval_quote()["sim_s"] == 32_400
    prod = Q.option_quote("mappowz168_u0")
    assert prod["planned_sim_s"] == 6_955_200 and prod["controlled_s"] == 4_636_800
    assert prod["planned_sim_s"] + Q.SMOKE_SIM_S + 32_400 <= 7_000_000
    # generic worker counts reproduce the named host rows
    assert Q.wall_h_for_workers(169, 30) == w["cpuQ"]
    assert Q.wall_h_for_workers(169, 46) == w["cpuV"]
    assert Q.wall_h_for_workers(169, 12) == w["cpuV_shared12"]


# ------------------------------------------------------------------ launcher
def _fake_u0(tmp_path) -> Path:
    p, e, _ = build_networks(accel_max=2.6, lane_control="native")
    path = tmp_path / "u0.pt"
    torch.save(export_payload(p, e, LaneRules()), path)
    return path


def test_plan_refuses_stub_and_keeps_original_argv(tmp_path):
    stub = write_cfg(tmp_path, "")
    with pytest.raises(SystemExit, match="STUB"):
        L.make_plan(option="full169_u0", host="cpuQ", run_dir=tmp_path, init="u0.pt",
                    python="py", sumo_home=None, config=stub)
    plan = L.make_plan(option="full169_u0", host="cpuQ", run_dir=tmp_path, init="u0.pt",
                       python="py", sumo_home=None, allow_stub=True, config=stub)
    assert plan["sumo_allowed"] is False
    argv = plan["argv"]["production"]
    tail = argv[argv.index("--init-from"):]
    assert tail == ["--init-from", "u0.pt", "--update-threads", "16", "--workers", "30",
                    "--iterations", "169", "--episodes-per-iteration", "46",
                    "--max-sumo-seconds", "4664400"]
    smoke = plan["argv"]["smoke"]
    assert smoke[smoke.index("--workers"):] == ["--workers", "2", "--iterations", "1",
                                                "--episodes-per-iteration", "2", "--horizon-s", "30",
                                                "--max-sumo-seconds", "60"]
    scratch = L.make_plan(option="full169_scratch", host="cpuV", run_dir=tmp_path, init=None,
                          python="py", sumo_home=None, allow_stub=True, config=stub)
    assert "--init-from" not in scratch["argv"]["production"]
    assert scratch["argv"]["production"][scratch["argv"]["production"].index("--workers") + 1] == "46"
    shared = L.make_plan(option="full169_u0", host="cpuV", run_dir=tmp_path, init="u0.pt",
                         python="py", sumo_home=None, workers=12, allow_stub=True, config=stub)
    assert shared["quote_wall_h"] == Q.option_quote("full169_u0")["wall_h"]["cpuV_shared12"]
    assert shared["wall_cap_s"] >= 1.5 * 14.7 * 3600 - 1
    ft = L.make_plan(option="ft42_u0", host="cpuV", run_dir=tmp_path, init="u0.pt",
                     python="py", sumo_home=None, allow_stub=True, config=stub)
    assert ft["max_sumo_seconds_controlled"] == 1_159_200 and ft["snapshots_preregistered"] == [24, 42]
    with pytest.raises(SystemExit, match="--init"):
        L.make_plan(option="ft42_u0", host="cpuV", run_dir=tmp_path, init=None, python="py",
                    sumo_home=None, allow_stub=True, config=stub)
    prod = L.make_plan(option="mappowz168_u0", host="cpuQ", run_dir=tmp_path, init="u0.pt",
                       python="py", sumo_home=None, workers=46, allow_stub=True, config=stub,
                       wall_cap_h=60, phase_cap_sim_s=7_000_000, other_charges_sim_s=32_400)
    assert prod["wall_cap_s"] == 60 * 3600 and prod["phase_cap_sim_s"] == 7_000_000
    argv = prod["argv"]["production"]
    assert argv[argv.index("--iterations") + 1] == "168" and argv[argv.index("--workers") + 1] == "46"
    assert argv[argv.index("--max-sumo-seconds") + 1] == "4636800"
    with pytest.raises(SystemExit, match="phase cap"):
        L.make_plan(option="full169_u0", host="cpuQ", run_dir=tmp_path, init="u0.pt", python="py",
                    sumo_home=None, allow_stub=True, config=stub, phase_cap_sim_s=7_000_000,
                    other_charges_sim_s=32_400)


def test_registration_and_smoke_gates(tmp_path):
    cfg = write_cfg(tmp_path, "tests.unit.test_nm_mappo:fake_n1_loader")
    run = tmp_path / "run"
    run.mkdir()
    u0 = _fake_u0(tmp_path)
    plan = L.make_plan(option="ft24_u0", host="cpuQ", run_dir=run, init=str(u0), python="py",
                       sumo_home=None, config=cfg)
    assert plan["sumo_allowed"] is True
    L.write_json_new(run / "PLAN.json", plan)
    with pytest.raises(SystemExit, match="REGISTERED"):
        L.verify_plan(run)
    reg = tmp_path / "REGISTRATION.md"
    reg.write_text("block without the hash\n")
    with pytest.raises(SystemExit, match="does not contain"):
        L.main(["ack", "--run-dir", str(run), "--registration", str(reg)])
    digest = L.sha256_file(run / "PLAN.json")
    reg.write_text(f"N7 MAPPO-WZ plan sha256 {digest} registered before SUMO\n")
    assert L.main(["ack", "--run-dir", str(run), "--registration", str(reg)]) == 0
    with pytest.raises(SystemExit, match="d9042cc5"):   # fake u0 bytes != registered u0
        L.verify_plan(run)
    assert L.verify_plan(run, check_init=False)["plan_sha256"] == digest
    # production refuses without a passed smoke; smoke dry-run prints the trainer argv
    with pytest.raises(SystemExit, match="smoke gate"):
        L.main(["run", "--run-dir", str(run), "--dry-run"])
    assert L.main(["smoke", "--run-dir", str(run), "--dry-run"]) == 0
    smoke = run / "smoke"
    smoke.mkdir()
    (smoke / "RUN_SUMMARY.json").write_text(json.dumps({"status": "completed"}))
    row = {"type": "iter", "episodes": 2, "failed_episodes": 0, "collisions": 0, "teleports": 0,
           "lane_commands_issued": 0, "policy_lane_change_rate": 0.0}
    (smoke / "log.jsonl").write_text(json.dumps(row) + "\n")
    assert L.smoke_passed(run) == (True, "PASS")
    (smoke / "log.jsonl").write_text(json.dumps({**row, "collisions": 1}) + "\n")
    assert L.smoke_passed(run)[0] is False
    episode = {"type": "episode", "dropped_vehicle_steps": 3}
    (smoke / "log.jsonl").write_text(json.dumps(episode) + "\n" + json.dumps(row) + "\n")
    assert "overflow" in L.smoke_passed(run)[1]
    (smoke / "log.jsonl").write_text(json.dumps(row) + "\n")
    assert L.main(["run", "--run-dir", str(run), "--dry-run"]) == 0
    # config drift after registration is refused
    cfg.write_text(cfg.read_text() + "\n# drift\n")
    with pytest.raises(SystemExit, match="config changed"):
        L.verify_plan(run, check_init=False)


def test_check_refuses_wrong_u0_and_fake_smoke_warm_starts(tmp_path):
    from wms.control.mappo import train
    u0 = _fake_u0(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    stub = write_cfg(tmp_path, "")
    L.write_json_new(run / "PLAN.json", L.make_plan(
        option="ft24_u0", host="cpuV", run_dir=run, init=str(u0), python="py", sumo_home=None,
        allow_stub=True, config=stub))
    with pytest.raises(SystemExit, match="d9042cc5"):
        L.main(["check", "--run-dir", str(run)])
    # what `check` runs for the real u0: the exact config, fake env, strict speed-only warm start
    assert train.main(["--config", str(stub), "--smoke", "--workers", "1",
                       "--out", str(tmp_path / "smoke"), "--init-from", str(u0),
                       "--update-threads", "1"]) == 0
    rows = [json.loads(s) for s in (tmp_path / "smoke/log.jsonl").read_text().splitlines()]
    init = rows[0]["meta"]["init_from"]
    assert init["kind"] == "exported_actor" and init["lane_head_removed_keys"] == []
    assert rows[0]["meta"]["map"]["name"] == "wz3to2"


def test_resume_argv_drops_init_and_requires_checkpoint():
    argv = ["py", "-m", "x", "--init-from", "u0.pt", "--workers", "30"]
    assert L.resume_argv(argv) == ["py", "-m", "x", "--workers", "30", "--resume"]


# ------------------------------------------------------------------ sanity evaluation
def test_sanity_eval_fake_env(tmp_path):
    u0 = _fake_u0(tmp_path)
    p, e, _ = build_networks(accel_max=2.6, lane_control="native")
    wz = tmp_path / "wz.pt"
    torch.save(export_payload(p, e, LaneRules()), wz)
    out = tmp_path / "san"
    stub = write_cfg(tmp_path, "")       # never install the real N1 map in the pytest process
    rc = S.main(["--config", str(stub), "--fake", "--policy", f"u0={u0}", "--policy", f"wz={wz}",
                 "--out", str(out), "--horizon-s", "3", "--warmup-s", "0", "--allow-any-u0"])
    assert rc == 0
    summary = json.loads((out / "SUMMARY.json").read_text())
    assert summary["complete_worlds"] == 12 and summary["failed_rows"] == 0
    assert summary["flags"]["S1_safety"] and summary["flags"]["S2_finite"]
    assert set(summary["paired"]) == {"u0-native", "wz-native", "wz-u0"}
    rows = [json.loads(s) for s in (out / "rows.jsonl").read_text().splitlines()]
    assert {r["seed"] for r in rows} == set(range(989588, 989600))
    assert all(r["lane_commands_issued"] == 0 for r in rows)
    plan = json.loads((out / "PLAN.json").read_text())
    assert plan["planned_sim_s"] == 12 * 3 * 3.0
    with pytest.raises(SystemExit, match="d9042cc5"):
        S.main(["--config", str(stub), "--fake", "--policy", f"u0={u0}", "--out", str(tmp_path / "x"),
                "--horizon-s", "3", "--warmup-s", "0"])


def test_sanity_eval_sumo_mode_needs_matching_registration(tmp_path):
    cfg = write_cfg(tmp_path, "tests.unit.test_nm_mappo:fake_n1_loader")
    (tmp_path / "stub").mkdir()
    with pytest.raises(M.MapBindingError, match="STUB"):
        S.main(["--config", str(write_cfg(tmp_path / "stub", "")), "--print-panel"])
    u0 = _fake_u0(tmp_path)
    base = ["--config", str(cfg), "--policy", f"u0={u0}", "--horizon-s", "3", "--warmup-s", "0",
            "--allow-any-u0"]
    with pytest.raises(SystemExit, match="--registered"):
        S.main(base + ["--out", str(tmp_path / "a")])
    spec, binding, cells = S.resolve(loads(cfg.read_text()), fake=False)
    assert cells == [(0.6, 3200)]
    panel = S.make_panel(binding, cells, warmup_s=0.0, horizon_s=3.0)
    assert len(panel) == 12 and [w["seed"] for w in panel] == list(range(989588, 989600))
    assert all(w["map"] == "wz3to2" for w in panel)
    receipt = tmp_path / "SANITY_REGISTERED.json"
    receipt.write_text(json.dumps({"panel_sha256": "0" * 64, "budget_sim_s": 1e9}))
    with pytest.raises(SystemExit, match="panel_sha256"):
        S.main(base + ["--out", str(tmp_path / "b"), "--registered", str(receipt)])
    receipt.write_text(json.dumps({"panel_sha256": S.panel_sha256(panel), "budget_sim_s": 72.0}))
    assert S.main(base + ["--out", str(tmp_path / "c"), "--registered", str(receipt)]) == 0
    # later arm on the same panel, summarised with the earlier rows (no second native arm)
    wz = tmp_path / "wz.pt"
    p, e, _ = build_networks(accel_max=2.6, lane_control="native")
    torch.save(export_payload(p, e, LaneRules()), wz)
    receipt.write_text(json.dumps({"panel_sha256": S.panel_sha256(panel), "budget_sim_s": 36.0}))
    later = ["--config", str(cfg), "--policy", f"wz={wz}", "--horizon-s", "3", "--warmup-s", "0",
             "--skip-native", "--prior-rows", str(tmp_path / "c/rows.jsonl"),
             "--out", str(tmp_path / "d"), "--registered", str(receipt)]
    assert S.main(later) == 0
    summary = json.loads((tmp_path / "d/SUMMARY.json").read_text())
    assert set(summary["paired"]) == {"u0-native", "wz-native", "wz-u0"}
    assert summary["complete_worlds"] == 12


# ------------------------------------------------------------------ phase guard
def test_phase_guard_disk_and_cap(tmp_path):
    run = tmp_path / "run"
    (run / "production").mkdir(parents=True)
    (run / "smoke").mkdir()
    plan = {"disk_floor_bytes": 0, "phase_cap_sim_s": 7_000_000, "other_charges_sim_s": 32_400,
            "per_iteration_sim_s": 41_400, "episodes_per_iteration": 46, "iterations": 168}
    (run / "smoke/status.json").write_text(json.dumps({"sumo_seconds_simulated": 660.0,
                                                       "failed_episodes": 0}))
    assert L.phase_guard(plan, run) == []           # 660 + 32,400 + 168 x 41,400 = 6,988,260
    status = {"iteration": 10, "sumo_seconds_simulated": 414_000.0, "failed_episodes": 13}
    (run / "production/status.json").write_text(json.dumps(status))
    assert L.phase_guard(plan, run) == []           # 13 failures (11,700) still fit
    (run / "production/status.json").write_text(json.dumps({**status, "failed_episodes": 14}))
    assert "phase cap" in L.phase_guard(plan, run)[0]
    assert "disk" in L.phase_guard({**plan, "disk_floor_bytes": 1 << 62}, run)[0]


# ------------------------------------------------------------------ real N1 binding (subprocess)
_N1_PROBE = r"""
import json, sys
from wms.control.mappo import train
from wms.control.mappo.mapbind import resolve_binding
b = resolve_binding({"name": "wz3to2", "api": "tools.nm_mappo.n1_adapter:descriptor"})
import traffic_shepherd.c4.env as E, wms.control.mappo.agent as A, wms.control.rsu_field as R
rec = b.record()
print(json.dumps({"stub": b.is_stub, "cells": [rec["demands_vph"], rec["penetrations"]],
                  "train": rec["train_seeds"], "held": rec["heldout_seeds"],
                  "k": rec["extra"]["k_slots"], "net": rec["extra"]["network_file_sha256"],
                  "env": E.CavSumoEnv.MAP_NAME, "field": A.build_rsu_field is R.build_rsu_field,
                  "masked": R.build_rsu_field.__module__}))
sys.exit(train.main(sys.argv[1:]))
"""


def test_real_n1_binding_and_fake_smoke_in_subprocess(tmp_path):
    import os
    import subprocess
    u0 = _fake_u0(tmp_path)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]))
    env.pop("COOPMWM_MAP_SPEC", None)
    out = subprocess.run([sys.executable, "-c", _N1_PROBE, "--config", str(CONFIG), "--smoke",
                          "--workers", "1", "--update-threads", "1", "--init-from", str(u0),
                          "--out", str(tmp_path / "smoke")], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    probe = json.loads(out.stdout.splitlines()[0])
    assert probe == {"stub": False, "cells": [[2400, 2800, 3200], [0.2, 0.4, 0.6]],
                     "train": "989200-989599", "held": "989600-989699", "k": 512,
                     "net": "b62778986347e8c21bca90d7ddcc8d9faeb288290a4dbdc3f53ee64ae8c21dfa",
                     "env": "wz3to2", "field": True, "masked": "tools.coopmwm.map_observations"}
    rows = [json.loads(x) for x in (tmp_path / "smoke/log.jsonl").read_text().splitlines()]
    assert rows[0]["meta"]["map"]["extra"]["install_record"]["rsu_field_rebound"]
    assert rows[0]["rollout"]["demands_vph"] == [3200] and rows[0]["rollout"]["penetrations"] == [0.6]
    assert [r for r in rows if r["type"] == "iter"]
