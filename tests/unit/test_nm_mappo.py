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
        demand_cells_vph=[2500, 2800], penetration_cells=[0.1, 0.3], spec_sha256="ab" * 32)


def wz_cfg(api: str = "") -> dict:
    cfg = loads(CONFIG.read_text(encoding="utf-8"))
    cfg["map"]["api"] = api
    return cfg


def write_cfg(tmp_path, api: str) -> Path:
    text = CONFIG.read_text(encoding="utf-8").replace('api = ""', f'api = "{api}"', 1)
    path = tmp_path / "wz.toml"
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------ config parity
DECLARED = {("map",), ("env", "name"), ("env", "cleanup_scenarios"), ("policy", "accel_max_mps2"),
            ("rollout", "seeds"), ("rollout", "demands_vph"), ("rollout", "penetrations"),
            ("rollout", "iterations"), ("rollout", "max_sumo_seconds")}


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
    assert new["rollout"]["iterations"] == 169 and new["rollout"]["max_sumo_seconds"] == 4664400.0
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
    assert resolved["rollout"]["demands_vph"] == list(binding.demands_vph)
    assert resolved["rollout"]["penetrations"] == list(binding.penetrations)
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
    assert resolved["rollout"]["demands_vph"] == [2500, 2800]
    assert resolved["rollout"]["penetrations"] == [0.1, 0.3]
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
    assert all(r["demand_vph"] in (2500, 2800) and r["penetration"] in (0.1, 0.3) for r in eps)
    it = [r for r in rows if r["type"] == "iter"][0]
    assert it["lane_commands_issued"] == 0 and it["policy_lane_change_rate"] == 0
    payload = torch.load(tmp_path / "run/policy_00001.pt", weights_only=False)
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
    with pytest.raises(SystemExit, match="STUB"):
        L.make_plan(option="full169_u0", host="cpuQ", run_dir=tmp_path, init="u0.pt",
                    python="py", sumo_home=None)
    plan = L.make_plan(option="full169_u0", host="cpuQ", run_dir=tmp_path, init="u0.pt",
                       python="py", sumo_home=None, allow_stub=True)
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
                          python="py", sumo_home=None, allow_stub=True)
    assert "--init-from" not in scratch["argv"]["production"]
    assert scratch["argv"]["production"][scratch["argv"]["production"].index("--workers") + 1] == "46"
    shared = L.make_plan(option="full169_u0", host="cpuV", run_dir=tmp_path, init="u0.pt",
                         python="py", sumo_home=None, workers=12, allow_stub=True)
    assert shared["quote_wall_h"] == Q.option_quote("full169_u0")["wall_h"]["cpuV_shared12"]
    assert shared["wall_cap_s"] >= 1.5 * 14.7 * 3600 - 1
    ft = L.make_plan(option="ft42_u0", host="cpuV", run_dir=tmp_path, init="u0.pt",
                     python="py", sumo_home=None, allow_stub=True)
    assert ft["max_sumo_seconds_controlled"] == 1_159_200 and ft["snapshots_preregistered"] == [24, 42]
    with pytest.raises(SystemExit, match="--init"):
        L.make_plan(option="ft42_u0", host="cpuV", run_dir=tmp_path, init=None, python="py",
                    sumo_home=None, allow_stub=True)


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
    L.write_json_new(run / "PLAN.json", L.make_plan(
        option="ft24_u0", host="cpuV", run_dir=run, init=str(u0), python="py", sumo_home=None,
        allow_stub=True))
    with pytest.raises(SystemExit, match="d9042cc5"):
        L.main(["check", "--run-dir", str(run)])
    # what `check` runs for the real u0: the exact config, fake env, strict speed-only warm start
    assert train.main(["--config", str(CONFIG), "--smoke", "--workers", "1",
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
    rc = S.main(["--fake", "--policy", f"u0={u0}", "--policy", f"wz={wz}", "--out", str(out),
                 "--horizon-s", "3", "--warmup-s", "0", "--allow-any-u0"])
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
        S.main(["--fake", "--policy", f"u0={u0}", "--out", str(tmp_path / "x"),
                "--horizon-s", "3", "--warmup-s", "0"])


def test_sanity_eval_sumo_mode_needs_matching_registration(tmp_path):
    cfg = write_cfg(tmp_path, "tests.unit.test_nm_mappo:fake_n1_loader")
    with pytest.raises(M.MapBindingError, match="STUB"):
        S.main(["--config", str(CONFIG), "--print-panel"])
    u0 = _fake_u0(tmp_path)
    base = ["--config", str(cfg), "--policy", f"u0={u0}", "--horizon-s", "3", "--warmup-s", "0",
            "--allow-any-u0"]
    with pytest.raises(SystemExit, match="--registered"):
        S.main(base + ["--out", str(tmp_path / "a")])
    spec, binding = S.resolve(loads(cfg.read_text()), fake=False)
    panel = S.make_panel(binding, warmup_s=0.0, horizon_s=3.0)
    assert len(panel) == 4 and all(989200 <= w["seed"] <= 989599 for w in panel)
    receipt = tmp_path / "SANITY_REGISTERED.json"
    receipt.write_text(json.dumps({"panel_sha256": "0" * 64, "budget_sim_s": 1e9}))
    with pytest.raises(SystemExit, match="panel_sha256"):
        S.main(base + ["--out", str(tmp_path / "b"), "--registered", str(receipt)])
    receipt.write_text(json.dumps({"panel_sha256": S.panel_sha256(panel), "budget_sim_s": 24.0}))
    assert S.main(base + ["--out", str(tmp_path / "c"), "--registered", str(receipt)]) == 0
