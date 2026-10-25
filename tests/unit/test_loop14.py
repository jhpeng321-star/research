"""LOOP14 orchestrator, pair collection helpers, real-return export and acceptance rule (CPU, no SUMO)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


loop = _module("loop14_orchestrator", "scripts/c4/loop14.py")


# ------------------------------------------------------------------ fixtures
def _registration(tmp_path, **over):
    reg = {"id": "LOOP14-TEST", "status": "registered", "split": "TRAIN", "acknowledged_by": "test",
           "seed_block": [968000, 968009], "exclude": [[968002, 968003]],
           "budget": {"max_cycles": 2, "policy_updates_per_cycle": 10, "sim_s_per_cycle": 10_000,
                      "sim_s_total": 30_000, "gpu_h_policy_per_cycle": 1, "gpu_h_wm_per_cycle": 1,
                      "wm_steps_per_cycle": 40},
           "stop": {"consecutive_wm_rejects": 2, "max_branch_incidents_per_cycle": 0}}
    reg.update(over)
    path = tmp_path / "REG.json"
    path.write_text(json.dumps(reg), encoding="utf-8")
    return path


def _config(tmp_path, reg, **over):
    wm = tmp_path / "wm0" / "wm_s.pt"
    wm.parent.mkdir(exist_ok=True)
    wm.write_bytes(b"wm0")
    (tmp_path / "wm0" / "cfg.toml").write_text("x = 1\n", encoding="utf-8")
    cfg = {"loop_id": "LOOP14-TEST", "loop_root": str(tmp_path / "loop"), "registration": str(reg),
           "hosts": {"gpu": {"python": sys.executable, "code_root": str(ROOT)},
                     "cpu": {"python": sys.executable, "code_root": str(ROOT)}},
           "guards": {"min_free_gb": 0.0, "gpu_query": [sys.executable, "-c", "pass"]},
           "policy": {"config": "configs/c4_policy14.toml", "updates_per_cycle": 2, "seed": 965140,
                      "device": "cpu", "timeout_h": 0.5, "initial_real_returns": str(tmp_path / "r.npz")},
           "collect": {"work_root": str(tmp_path / "cpu"), "worlds_per_cycle": 3, "workers": 1, "warmup": 60,
                       "branch_first": 120, "branch_cadence": 60, "branch_points": 1, "branch_horizon": 300,
                       "demands": "2400", "penetrations": "0.2", "pen_weights": "1", "timeout_h": 0.5},
           "convert": {"val_frac": 0.0, "mb_per_episode": 8},
           "wm": {"config": "configs/c4_wm_loop14_ft.toml", "initial_wm": str(wm),
                  "initial_wm_config": str(tmp_path / "wm0" / "cfg.toml"), "device": "cpu", "timeout_h": 0.5},
           "check": {"roots": [], "onestep_windows": 2, "rollout_windows": 2, "pair_groups": 2, "timeout_h": 0.5}}
    for key, value in over.items():
        cfg[key] = value
    path = tmp_path / "loop.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _main(*args):
    return loop.main([str(a) for a in args])


def _state(tmp_path):
    return json.loads((tmp_path / "loop" / loop.STATE).read_text(encoding="utf-8"))


def _ledger(tmp_path, name="cycle_ledger.jsonl"):
    return [json.loads(x) for x in (tmp_path / "loop" / name).read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------ registration / budget
def test_registration_requires_ack_and_fresh_train_block(tmp_path):
    cfg = _config(tmp_path, _registration(tmp_path, acknowledged_by=""))
    assert _main("init", "--config", cfg) == 2
    cfg = _config(tmp_path, _registration(tmp_path, seed_block=[963990, 964010]))
    assert _main("init", "--config", cfg) == 2
    cfg = _config(tmp_path, _registration(tmp_path, status="draft"))
    assert _main("init", "--config", cfg) == 2
    assert not (tmp_path / "loop").exists()


def test_budget_refuses_oversized_collection(tmp_path):
    reg = _registration(tmp_path)
    assert loop.collect_sim_s(json.loads(_config(tmp_path, reg).read_text())) == 3 * (121 + 600)
    cfg = _config(tmp_path, reg, collect={**json.loads(_config(tmp_path, reg).read_text())["collect"],
                                          "worlds_per_cycle": 50})
    assert _main("init", "--config", cfg) == 2


def test_restricted_paths_refused(tmp_path):
    cfg = _config(tmp_path, _registration(tmp_path))
    doc = json.loads(cfg.read_text())
    doc["check"]["roots"] = ["/data/holdout_x"]
    cfg.write_text(json.dumps(doc))
    assert _main("init", "--config", cfg) == 2


# ------------------------------------------------------------------ phase order / guards
def test_init_order_and_gpu_busy(tmp_path):
    reg = _registration(tmp_path)
    cfg = _config(tmp_path, reg)
    assert _main("init", "--config", cfg) == 0
    assert _main("init", "--config", cfg) == 2                  # never re-initialised
    assert _main("run", "--config", cfg, "collect") == 2         # phase order
    doc = json.loads(cfg.read_text())
    doc["policy"]["device"] = "cuda"
    doc["guards"]["gpu_query"] = [sys.executable, "-c", "print('1234, python, 20000 MiB')"]
    cfg.write_text(json.dumps(doc))
    assert _main("run", "--config", cfg, "policy") == 5
    st = _state(tmp_path)
    assert st["cycles"][0]["phases"]["policy"]["status"] == "pending"
    assert st["cycles"][0]["phases"]["policy"]["attempt"] == 0
    assert _ledger(tmp_path)[-1]["event"] == "gpu_busy"
    doc["guards"]["gpu_query"] = [sys.executable, "-c", "import sys; sys.exit(9)"]   # query failure
    cfg.write_text(json.dumps(doc))
    assert _main("run", "--config", cfg, "policy") == 2
    # registration edited after init -> refused
    r = json.loads(reg.read_text())
    r["budget"]["max_cycles"] = 3
    reg.write_text(json.dumps(r))
    assert _main("status", "--config", cfg) == 2


def test_disk_guard_refuses(tmp_path):
    cfg = _config(tmp_path, _registration(tmp_path))
    assert _main("init", "--config", cfg) == 0
    doc = json.loads(cfg.read_text())
    doc["guards"]["min_free_gb"] = 1e9
    cfg.write_text(json.dumps(doc))
    assert _main("run", "--config", cfg, "policy") == 2
    assert _ledger(tmp_path)[-1]["event"] == "refused"


def test_failed_phase_needs_retry_and_new_attempt_dir(tmp_path, monkeypatch):
    cfg = _config(tmp_path, _registration(tmp_path))
    assert _main("init", "--config", cfg) == 0
    calls = []

    def fake_run(h, argv, *, env, log, timeout_s, cwd=None):
        out = Path(argv[argv.index("--output-dir") + 1])
        calls.append(out)
        if len(calls) == 1:
            return 1                                              # first attempt fails
        out.mkdir(parents=True)
        (out / "policy_latest.pt").write_bytes(b"p")
        (out / "RUN_SUMMARY.json").write_text(json.dumps({"status": "completed", "updates": [{"actor_loss": 1.0}]}))
        return 0

    monkeypatch.setattr(loop, "run_cmd", fake_run)
    assert _main("run", "--config", cfg, "policy") == 3
    assert _state(tmp_path)["cycles"][0]["phases"]["policy"]["status"] == "failed"
    assert _main("run", "--config", cfg, "policy") == 2          # never silently retried
    assert _main("retry", "--config", cfg, "policy") == 0
    assert calls[0].name == "c00" and calls[1].name == "c00.a2"
    st = _state(tmp_path)
    assert st["policy"]["checkpoint"].endswith("policy_latest.pt")
    assert st["cycles"][0]["phases"]["policy"]["status"] == "done"


def test_seed_allocation_recorded_before_collection(tmp_path, monkeypatch):
    cfg = _config(tmp_path, _registration(tmp_path))
    pol = tmp_path / "pol.pt"
    pol.write_bytes(b"p")
    assert _main("init", "--config", cfg, "--adopt-policy", pol) == 0
    seen = {}

    def fake_run(h, argv, *, env, log, timeout_s, cwd=None):
        if "traffic_shepherd.c4.collect" in argv:
            seen["ledger"] = (tmp_path / "loop" / "seed_ledger.jsonl").read_text()
            seen["seeds"] = argv[argv.index("--seeds") + 1]
            return 1
        return 0

    monkeypatch.setattr(loop, "run_cmd", fake_run)
    monkeypatch.setattr(loop, "sync", lambda *a, **k: None)
    assert _main("run", "--config", cfg, "collect") == 3
    assert seen["seeds"] == "968000,968001,968004"               # 968002-968003 excluded
    assert json.loads(seen["ledger"])["seeds"] == [968000, 968001, 968004]
    st = _state(tmp_path)
    assert st["sim_s_charged"] == 3 * 721 and st["seeds_used"] == [968000, 968001, 968004]
    assert _main("retry", "--config", cfg, "collect") == 3
    assert _state(tmp_path)["seeds_used"][3:] == [968005, 968006, 968007]   # never re-allocated


def test_cycle_decisions_accept_reject_and_stop(tmp_path, monkeypatch):
    cfg = _config(tmp_path, _registration(tmp_path))
    pol = tmp_path / "pol.pt"
    pol.write_bytes(b"p")
    assert _main("init", "--config", cfg, "--adopt-policy", pol) == 0
    verdicts = iter(["REJECT", "REJECT"])

    def receipt(name):
        def run(cfg_, state, ph, root):
            k = state["cycles"][-1]["cycle"]
            if name == "policy":
                state["policy"] = {"checkpoint": str(pol), "sha256": "p", "source": "x"}
                return {}
            if name == "collect":
                return {"branch_collisions": 0, "branch_teleports": 0}
            if name == "convert":
                return {"pairs": "P", "returns": f"R{k}.npz"}
            if name == "wm_ft":
                return {"candidate": f"cand{k}.pt", "candidate_sha256": f"c{k}"}
            decision = next(verdicts) if k > 0 else "ACCEPT"
            if decision == "ACCEPT":
                state["accepted_wm"] = {"checkpoint": f"cand{k}.pt", "config": "ft", "sha256": f"c{k}",
                                        "source": f"cycle {k}"}
                state["consecutive_rejects"] = 0
            else:
                state["consecutive_rejects"] += 1
            return {"decision": decision}
        return run

    monkeypatch.setattr(loop, "RUNNERS", {n: receipt(n) for n in loop.PHASES})
    for _ in range(3):                                           # adopted policy: collect, convert, wm_ft
        assert _main("next", "--config", cfg) == 0
    assert _main("next", "--config", cfg) == 0                   # check: ACCEPT
    st = _state(tmp_path)
    assert st["accepted_wm"]["sha256"] == "c0" and st["real_returns"] == "R0.npz"
    assert st["cycles"][-1]["cycle"] == 1
    for _ in range(4):
        assert _main("next", "--config", cfg) == 0
    assert _main("next", "--config", cfg) == 4                   # REJECT keeps the previous WM
    st = _state(tmp_path)
    assert st["accepted_wm"]["sha256"] == "c0"
    assert "max_cycles" in st["stopped"]
    assert _main("next", "--config", cfg) == 2                   # stopped loop refuses
    events = [e.get("event") for e in _ledger(tmp_path)]
    assert events.count("cycle_decision") == 2


def test_collection_incident_stops_loop(tmp_path, monkeypatch):
    cfg = _config(tmp_path, _registration(tmp_path))
    pol = tmp_path / "pol.pt"
    pol.write_bytes(b"p")
    assert _main("init", "--config", cfg, "--adopt-policy", pol) == 0
    monkeypatch.setattr(loop, "RUNNERS", {**loop.RUNNERS, "collect": lambda *a: {"branch_collisions": 1}})
    assert _main("next", "--config", cfg) == 0
    assert "collisions" in _state(tmp_path)["stopped"]
    assert _main("next", "--config", cfg) == 2


# ------------------------------------------------------------------ pair collection helpers
def test_decision_times_and_registered_seeds(tmp_path):
    from traffic_shepherd.c4.policy_pairs import check_registered_seeds, decision_times
    assert decision_times(300, 360, 300, 2) == [360, 660]
    with pytest.raises(ValueError):
        decision_times(300, 200, 60, 1)
    reg = _registration(tmp_path)
    check_registered_seeds([968000, 968004], str(reg))
    for bad in ([968002], [968010], [964100]):
        with pytest.raises(ValueError):
            check_registered_seeds(bad, str(reg))
    with pytest.raises(ValueError):
        check_registered_seeds([968000], None)


def test_entry_backlog_uses_schedule_and_inserted_only():
    from traffic_shepherd.c4.policy_pairs import entry_backlog_at, requested_by

    class Req(SimpleNamespace):
        def __getattr__(self, name):          # any other attribute (e.g. driver labels) is refused
            raise AssertionError(f"read {name}")

    reqs = [Req(vehicle_id="a", depart_s=10.0, route="mainline_route"),
            Req(vehicle_id="b", depart_s=20.0, route="ramp_route"),
            Req(vehicle_id="c", depart_s=30.0, route="ramp_route"),
            Req(vehicle_id="d", depart_s=99.0, route="mainline_route")]
    env = SimpleNamespace(requests=reqs, ledger=SimpleNamespace(inserted={"a"}))
    assert entry_backlog_at(env, 30.0) == [0, 2]
    assert requested_by(env, 30.0) == 3


# ------------------------------------------------------------------ real returns
def _pair_source(root: Path, *, j=1234.5, schema="c4_loop14_pair_summary/1", collisions=0):
    root.mkdir(parents=True)
    entries = []
    for kind in ("POLICY", "B"):
        raw = root / f"raw_{kind}"
        raw.mkdir()
        (raw / "summary.json").write_text(json.dumps({
            "schema": schema, "seed": 968000, "warmup_s": 120.0, "horizon_s": 300.0, "collisions": collisions,
            "teleports": 0, "dropped_vehicle_steps": 0, "requested_population": 300,
            "start_entry_backlog": [3, 1],
            "ledger": {"speed_deficit_vehicle_seconds": j / 2, "source_wait_vehicle_seconds": j / 2}}))
        ep = root / "episodes" / kind
        ep.mkdir(parents=True)
        np.save(ep / "time_s.npy", np.arange(60.0, 420.5, 0.5))
        kin = np.zeros((721, 4, 7), np.float32)
        kin[:, :, 1] = 1.0
        kin[:, :, 6] = 4.5
        np.save(ep / "veh_kin.npy", kin)
        np.save(ep / "veh_mask.npy", np.ones((721, 4), np.uint8))
        np.save(ep / "is_cav.npy", np.zeros((721, 4), np.uint8))
        entries.append({"split_role": "train", "seed": 968000, "dir": f"episodes/{kind}", "source": str(raw),
                        "pair_key": "w968000@t120", "branch": {"kind": kind}})
    (root / "C4_EPISODES_MANIFEST.json").write_text(json.dumps({"episodes": entries}))


def test_loop14_returns_export_contract(tmp_path, monkeypatch):
    from wms.control.loop14_returns import export_pairs
    from wms.control.real_returns14 import NPZRealReturnProvider
    monkeypatch.setenv("C4_ALLOWED_ROOTS", str(tmp_path))
    _pair_source(tmp_path / "src")
    receipt = export_pairs(tmp_path / "src", tmp_path / "out.npz", policy_checkpoint="sha")
    assert receipt["targets"] == 1 and receipt["source_kind"] == "current_policy"
    with np.load(tmp_path / "out.npz") as z:
        assert z["returns"].tolist() == [-1234.5]
        assert z["state__entry_backlog"].tolist() == [[3.0, 1.0]]
        assert str(z["policy_checkpoint"]) == "sha"
    NPZRealReturnProvider(tmp_path / "out.npz", seed=0)
    native = export_pairs(tmp_path / "src", tmp_path / "native.npz", policy_checkpoint="", kind="B")
    assert native["source_kind"] == "native"
    with pytest.raises(FileExistsError):
        export_pairs(tmp_path / "src", tmp_path / "out.npz", policy_checkpoint="sha")
    _pair_source(tmp_path / "bad", schema="c4_genrand_branch_summary/1")
    with pytest.raises(ValueError):
        export_pairs(tmp_path / "bad", tmp_path / "bad.npz", policy_checkpoint="sha")
    _pair_source(tmp_path / "crash", collisions=1)
    with pytest.raises(ValueError, match="no eligible"):
        export_pairs(tmp_path / "crash", tmp_path / "crash.npz", policy_checkpoint="sha")


# ------------------------------------------------------------------ acceptance rule
def test_acceptance_rule_is_the_13e_stop_rule_plus_safety():
    accept = _module("loop14_accept", "tools/loop14_checks/accept.py")
    one = lambda c, r: {"mae": {"candidate": {"mae": c}, "reference": {"mae": r}}}  # noqa: E731

    def roll(share_c, share_r, **c):
        base = {"hard_brake_share": share_r, "blowup_runs": 0, "overlap_pair_steps": 2, "ttc_lt_1_5_share": 0.01}
        return {"candidate": {**base, "hard_brake_share": share_c, **c}, "reference": base}

    assert accept.decide(one(0.50, 0.45), roll(0.010, 0.006)) == []
    assert accept.decide(one(0.56, 0.45), roll(0.010, 0.006))           # MAE + 0.11
    assert accept.decide(one(0.45, 0.45), roll(0.013, 0.006))           # share > 2x
    assert accept.decide(one(0.45, 0.45), roll(0.0, 0.0, blowup_runs=1))
    assert accept.decide(one(0.45, 0.45), roll(0.0, 0.0, overlap_pair_steps=3))
    assert accept.decide(one(0.45, 0.45), roll(0.0, 0.0, ttc_lt_1_5_share=0.03))
    with pytest.raises(ValueError):
        accept.decide(one(float("nan"), 0.45), roll(0.0, 0.0))


def test_lane1_hard_brake_definition():
    accept = _module("loop14_accept", "tools/loop14_checks/accept.py")
    v = np.array([[20.0, 20.0, 20.0], [18.0, 19.0, 20.0], [16.0, 18.0, 20.0]])
    lane = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]], float)
    mask = np.ones_like(v, bool)
    n, hard = accept.lane1_hard_brake({"speed": v, "lane": lane, "mask": mask,
                                       "is_cav": np.array([False, True, False])})
    assert (n, hard) == (2, 2)          # HDV 0: -4 m/s^2 twice; CAV 1 and lane-0 HDV 2 excluded


def test_fine_tune_config_recipe():
    from wms.config.load import load_train_config
    c = load_train_config(ROOT / "configs/c4_wm_loop14_ft.toml", ['model.init_from="x.pt"'])
    assert c.physics.lane_change_secure_gap and c.physics.emergency_braking and c.physics.secure_gap_veto
    assert c.model.mainline_merge_yield_learnable and not c.model.follower_courtesy_init_zero
    frozen = set(c.optim.freeze_prefixes)
    assert {"driver_encoder", "driver_updater", "merge_core", "merge_pivot", "lane_v5", "temporal"} <= frozen
    assert not {"merge_yield", "residual_net", "follower_courtesy"} & frozen
    shares = {s.name: s.mix_ratio for s in c.data.c4_sources}
    assert shares["loop14_pairs"] == 0.15 and abs(sum(shares.values()) - 1.0) < 1e-6
    assert [s.name for s in c.data.c4_sources if s.paired] == ["pairs300"]
    assert c.data.replay.low_share_oversample == 1.69
    assert c.model.init_from == "x.pt"
