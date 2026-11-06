"""Scheduling/data-integrity regressions: CPU only; never starts SUMO or a loop waiter."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/c4"))
import loop14 as loop  # noqa: E402
import loop14_fast as fast  # noqa: E402
from tools.fastloop.world import digest  # noqa: E402
from tools.main14.verdict_snapshot import choose_three  # noqa: E402


def fixture(tmp_path):
    spec = importlib.util.spec_from_file_location("fastloop_fixtures", ROOT / "tests/unit/test_loop14.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    path = m._config(tmp_path, m._registration(tmp_path))
    cfg = loop.read_json(path)
    cfg["hosts"].update(cpuP=dict(cfg["hosts"]["cpu"]), cpuQ=dict(cfg["hosts"]["cpu"]))
    cfg["fastloop"] = {"two_box": True, "spot_resume": True, "boxes": ["cpuP", "cpuQ"], "workers_per_box": 2}
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"p")
    loop.init(cfg, SimpleNamespace(adopt_policy=str(policy)))
    return cfg


def artifact(path):
    path.mkdir(parents=True)
    (path / "data.json").write_text("{}")
    (path / "HASHES.json").write_text(json.dumps({"data.json": digest(path / "data.json")}))


def test_flag_off_policy_command_and_environment_are_identical(tmp_path):
    cfg = fixture(tmp_path)
    state = loop.load_state(cfg)
    cfg.pop("fastloop")
    before = loop.policy_cmd(cfg, state, tmp_path)
    cfg["fastloop"] = {flag: False for flag in fast.FLAGS}
    assert loop.policy_cmd(cfg, state, tmp_path) == before
    cfg["fastloop"]["speed2_policy"] = True
    argv, env = loop.policy_cmd(cfg, state, tmp_path)
    assert argv == before[0]
    assert {k: v for k, v in env.items() if not k.startswith("WMS_SPEED_")} == before[1]
    assert env["WMS_SPEED_PIVOT_DIAG_OFF"] == "1"
    assert env["WMS_SPEED_GRAD_AUDIT_EVERY"] == "10"


def test_shards_preserve_one_box_seed_demand_and_penetration():
    from traffic_shepherd.c4.collect import plan_cases
    cfg = {"collect": {"penetrations": "0.05,0.1,0.2,0.4", "pen_weights": "1,1,2,2", "demands": "1900,2100,2400,2700"}}
    seeds = [968103 + i for i in range(13)]
    expected = plan_cases(seeds, [1900, 2100, 2400, 2700], [.05, .1, .2, .4], "weighted-cycle", "1,1,2,2")
    actual = fast.cases(cfg, seeds)
    assert [(c["seed"], int(c["demand"]), float(c["penetration"])) for c in actual] == expected
    assert [c["box"] for c in actual] == ["cpuP", "cpuQ"] * 6 + ["cpuP"]


def test_spot_retry_keeps_successes_and_replays_same_world_on_peer(tmp_path, monkeypatch):
    cfg = fixture(tmp_path)
    root = Path(cfg["loop_root"])
    monkeypatch.setattr(loop, "fast_module", lambda: fast)
    calls = []

    def world(loop_, cfg_, state, ph, root_, item, box, attempt):
        calls.append((item["seed"], box, attempt))
        ledger = [json.loads(s) for s in (root / "seed_ledger.jsonl").read_text().splitlines()]
        assert any(r.get("seed") == item["seed"] and r.get("attempt") == attempt for r in ledger)
        if item["index"] == 1 and attempt == 1:
            raise ConnectionError("spot reclaimed")
        out = root / f"artifact_{item['seed']}_{attempt}"
        artifact(out)
        return {"artifact": str(out), "case": item, "box": box, "attempt": attempt,
                "collect": {"branch_collisions": 0, "branch_teleports": 0}}

    monkeypatch.setattr(fast, "_world", world)
    assert loop.run_phase(cfg, "collect") == 3
    state = loop.load_state(cfg)
    assert state["sim_s_charged"] == 3 * 721
    assert state["seeds_used"] == [968000, 968001, 968004]
    # A coordinator crash may leave older state/journal counters: ledger still owns every charge.
    state["sim_s_charged"] = 0
    loop.save_state(cfg, state)
    journal = loop.read_json(root / "worlds_c00.json")
    journal.update(charged=0, attempts={})
    loop.write_atomic(root / "worlds_c00.json", journal)
    assert loop.retry(cfg, "collect") == 0
    state = loop.load_state(cfg)
    assert state["sim_s_charged"] == 4 * 721
    assert state["seeds_used"] == [968000, 968001, 968004]
    assert calls.count((968000, "cpuP", 1)) == 1
    assert (968001, "cpuP", 2) in calls
    assert state["cycles"][0]["phases"]["collect"]["receipt"]["seeds"] == [968000, 968001, 968004]


def test_artifact_hash_corruption_and_path_escape_fail_closed(tmp_path):
    out = tmp_path / "a"
    artifact(out)
    assert fast.artifact_valid(out)
    (out / "data.json").write_text("corrupt")
    assert not fast.artifact_valid(out)
    (out / "HASHES.json").write_text(json.dumps({"../escape": "sha"}))
    assert not fast.artifact_valid(out)


def test_retry_budget_is_not_waived_for_reclaim(tmp_path, monkeypatch):
    cfg = fixture(tmp_path)
    root = Path(cfg["loop_root"])
    state = loop.load_state(cfg)
    reg = loop.read_json(cfg["registration"])
    reg["budget"]["sim_s_per_cycle"] = 3 * 721
    Path(cfg["registration"]).write_text(json.dumps(reg))
    state["registration_sha256"] = loop.sha256(cfg["registration"])
    loop.save_state(cfg, state)
    monkeypatch.setattr(loop, "fast_module", lambda: fast)
    monkeypatch.setattr(fast, "_world", lambda *a: (_ for _ in ()).throw(ConnectionError("lost")))
    assert loop.run_phase(cfg, "collect") == 3
    assert loop.retry(cfg, "collect") == 2
    assert loop.load_state(cfg)["sim_s_charged"] == 3 * 721
    assert len(loop.read_json(root / "worlds_c00.json")["attempts"]) == 3


@pytest.mark.parametrize("accepted", [True, False])
def test_overlap_pins_policy_wm_even_if_current_wm_changes(tmp_path, monkeypatch, accepted):
    cfg = fixture(tmp_path)
    state = loop.load_state(cfg)
    returns = tmp_path / "returns.npz"
    returns.write_bytes(b"returns")
    state["real_returns"] = str(returns)
    root = Path(cfg["loop_root"])
    original_wm = copy.deepcopy(state["accepted_wm"])

    def policy(cfg_, snapshot, ph, root_):
        assert snapshot["cycles"][-1]["cycle"] == 1
        ckpt = root / "overlap.pt"
        ckpt.write_bytes(b"next policy")
        snapshot["policy"] = {"checkpoint": str(ckpt), "sha256": loop.sha256(ckpt),
                              "trained_on_wm_sha256": snapshot["accepted_wm"]["sha256"]}
        return {"exit": 0}

    monkeypatch.setattr(loop, "run_policy", policy)
    fast.overlap(loop, cfg, state, root)
    if accepted:
        candidate = tmp_path / "new.pt"
        candidate.write_bytes(b"new WM")
        state["accepted_wm"] = {"checkpoint": str(candidate), "sha256": loop.sha256(candidate), "config": "ft.toml"}
    state["cycles"].append(loop.new_cycle(1))
    fast.promote(loop, cfg, state, root)
    assert state["policy_wm"] == original_wm
    assert state["policy"]["trained_on_wm_sha256"] == original_wm["sha256"]
    receipt = state["cycles"][-1]["phases"]["policy"]["receipt"]
    assert receipt["dependencies"]["real_returns_sha256"] == loop.sha256(returns)
    assert receipt["accepted_wm_at_promotion"] == state["accepted_wm"]


def qualification(ratio):
    return {"core": {"sign:a": "PASS"}, "table": {"a": {"mag:a": "PASS"}},
            "fid": {"all_pairs": {"ratio": ratio}}}


def test_three_model_rule_uses_reference_fails_and_newer_ties():
    docs = {"13e_b2": qualification(1.), "14b": qualification(.7), "16": qualification(.7)}
    assert choose_three(docs)["base"] == "16"
    docs["16"]["core"]["sign:new"] = "FAIL"
    assert choose_three(docs)["base"] == "14b"
    docs["14b"]["fid"]["all_pairs"]["ratio"] = 1.1
    assert choose_three(docs)["base"] == "13e_b2"
    del docs["16"]
    with pytest.raises(KeyError):
        choose_three(docs)


def test_worker_cap_and_boolean_flags_fail_closed(tmp_path):
    cfg = fixture(tmp_path)
    cfg["fastloop"]["workers_per_box"] = 15
    with pytest.raises(loop.Refused):
        fast.validate(cfg, loop.Refused)
    cfg["fastloop"]["two_box"] = "true"
    with pytest.raises(loop.Refused):
        fast.validate(cfg, loop.Refused)


def test_wm_phase_does_not_inherit_policy_or_numerical_speed_flags(tmp_path, monkeypatch):
    cfg = fixture(tmp_path)
    cfg["fastloop"]["speed2_policy"] = True
    cfg["wm"]["env"] = {"WMS_SPEED_COMPACT_SLOTS": "1", "WMS_SPEED_PIVOT_DIAG_OFF": "1"}
    state = loop.load_state(cfg)
    state["cycles"][-1]["phases"]["convert"]["receipt"] = {"pairs": "pairs"}
    observed = []

    def command(h, argv, **kwargs):
        observed.append(kwargs["env"])
        if "--mode" in argv:
            out = Path(argv[argv.index("--out") + 1])
            out.mkdir(parents=True)
            (out / "wm_s.pt").write_bytes(b"candidate")
            (out / "physics.json").write_text("{}")
            (out / "train_log.jsonl").write_text('{"loss":1.0}\n')
        return 0

    monkeypatch.setattr(loop, "run_cmd", command)
    monkeypatch.setattr(loop, "validate_steps", lambda *a: 40)
    loop.run_wm_ft(cfg, state, {"attempt": 1}, Path(cfg["loop_root"]))
    for env in observed:
        assert env["WMS_SPEED_COMPACT_SLOTS"] == "0"
        assert env["WMS_SPEED_PIVOT_DIAG_OFF"] == "0"
        assert env["WMS_SPEED_BATCH_REPLAY"] == "0"
        assert env["WMS_SPEED_GRAD_AUDIT_EVERY"] == "1"


def test_one_world_cpu_dry_run_has_real_conversion_and_zero_sumo(tmp_path):
    from tools.fastloop.dry_run import run
    report = run(tmp_path / "dry")
    assert report["status"] == "PASS"
    assert report["actual_sumo_sim_s"] == 0
    assert report["receipt"]["targets"] == 1
    assert report["next_phase"] == "wm_ft"


def test_two_world_merge_preserves_receipts_and_unique_pair_groups(tmp_path):
    import numpy as np
    from tools.fastloop.dry_run import run
    report = run(tmp_path / "dry", worlds=2)
    assert report["receipt"]["pair_groups"] == 2
    manifest = loop.read_json(Path(report["receipt"]["pairs"]) / "C4_EPISODES_MANIFEST.json")
    assert manifest["n_train"] == 4 and manifest["n_val"] == 0
    assert len({row["dir"] for row in manifest["episodes"]}) == 4
    for row in manifest["episodes"]:
        assert Path(row["source"], "summary.json").is_file()
        assert Path(report["receipt"]["pairs"], row["dir"], "veh_kin.npy").is_file()
    with np.load(report["receipt"]["returns"], allow_pickle=False) as z:
        assert z["seeds"].tolist() == [968000, 968001]
        assert z["returns"].tolist() == [-100., -100.]


def test_cpu_acceptance_error_and_changed_panel_cannot_adopt_candidate(tmp_path, monkeypatch):
    from tools.fastloop.dry_run import run
    run(tmp_path / "dry")
    cfg = loop.read_json(tmp_path / "dry/loop.json")
    # dry_run uses an in-memory fastloop config; restore the same completed mechanical state.
    cfg["hosts"].update(cpuP=dict(cfg["hosts"]["cpu"]), cpuQ=dict(cfg["hosts"]["cpu"]))
    cfg["fastloop"] = {"cpu_checks": True, "check_roots": [str(tmp_path)], "check_env": {}}
    cfg["collect"]["worlds_per_cycle"] = 1
    cfg["collect"].update(warmup=10, branch_first=10, branch_horizon=300)
    state = loop.load_state(cfg)
    root = Path(cfg["loop_root"])
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    state["cycles"][-1]["phases"]["wm_ft"]["receipt"] = {"candidate": str(candidate), "candidate_sha256": loop.sha256(candidate)}
    reference = copy.deepcopy(state["accepted_wm"])

    def accept(h, argv, **kwargs):
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True)
        loop.write_atomic(out / "ACCEPT.json", {"decision": "ACCEPT"})
        panel = Path(argv[argv.index("--panel") + 1])
        panel.write_text("changed" if ".a2" in str(out) else "fixed")
        assert int(argv[argv.index("--workers") + 1]) <= 14
        return 2 if ".a1" in str(out) else 0

    monkeypatch.setattr(loop, "run_cmd", accept)
    verdict = fast.check(loop, cfg, state, {"attempt": 1}, root)
    assert verdict["decision"] == "REJECT"
    assert state["accepted_wm"] == reference
    with pytest.raises(RuntimeError, match="panel changed"):
        fast.check(loop, cfg, state, {"attempt": 2}, root)
    assert state["accepted_wm"] == reference


@pytest.mark.parametrize("enabled", [False, True])
def test_prepare_only_writes_new_hashed_request_and_explicit_retry_reserve(tmp_path, monkeypatch, enabled):
    from tools.fastloop import prepare as prep
    spec = importlib.util.spec_from_file_location("prepare_fixtures", ROOT / "tests/unit/test_loop14.py")
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    old = tmp_path / "old"
    old.mkdir()
    reg_path = fixtures._registration(old)
    reg = loop.read_json(reg_path)
    reg["sources"] = {}
    loop.write_atomic(reg_path, reg)
    cfg_path = fixtures._config(old, reg_path)
    cfg = loop.read_json(cfg_path)
    panel = old / "gate_panel.json"
    panel.write_text("frozen gate")
    cfg["policy"].update(extra_args=["--gate-panel", str(panel), "--horizon-gate", "current.json"], allowed_roots=[])
    loop.write_atomic(cfg_path, cfg)
    arm_path = old / "ARMED.json"
    loop.write_atomic(arm_path, {"config": str(cfg_path), "registration": str(reg_path)})
    code = tmp_path / "code"
    (code / "configs").mkdir(parents=True)
    (code / "configs/c4_fastloop_options.json").write_text((ROOT / "configs/c4_fastloop_options.json").read_text())
    loop.write_atomic(code / "FASTLOOP_SOURCE.json", {"release": True, "code_sha": "f" * 40, "files": {}})
    contract = loop.read_json(ROOT / "tools/fastloop/CPU_CONTRACT.template.json")
    contract["scenario_fixture"] = {"cpuP": str(tmp_path / "fixture"), "cpuQ": str(tmp_path / "fixture")}
    contract["base_models"]["16"] = {"checkpoint": str(tmp_path / "wm16.pt"), "config": str(tmp_path / "cfg16.json")}
    contract["check_roots"] = [str(tmp_path / "train")]
    monkeypatch.setattr(prep, "ROOT", code)
    monkeypatch.setattr(prep, "verify_cpu_contract", lambda *a: {"verified": True})
    out = tmp_path / "new_arm"
    result = prep.prepare(arm_path, contract, out, "2026-10-26T18:00:00+08:00", enabled, enabled)
    assert result["armed"] is False and result["launched"] is False
    assert not (out / "waiter.pid").exists() and not (out / "loop" / loop.STATE).exists()
    request = loop.read_json(out / "ARM_REQUEST.json")
    assert request["config_sha256"] == loop.sha256(out / "config.json")
    assert request["registration_sha256"] == loop.sha256(out / "REGISTRATION.json")
    actual_cfg = loop.read_json(out / "config.json")
    assert all(actual_cfg["fastloop"][flag] is enabled for flag in fast.FLAGS)
    amended = loop.read_json(out / "REGISTRATION.json")
    assert amended["stop"] == reg["stop"] and amended["seed_block"] == reg["seed_block"]
    assert amended["budget"]["sim_s_per_cycle"] == (2 * loop.collect_sim_s(cfg) if enabled else reg["budget"]["sim_s_per_cycle"])


def test_driver_overlaps_cpu_collection_but_serializes_gpu_and_preserves_old_policy_wm(tmp_path, monkeypatch):
    import threading
    cfg = fixture(tmp_path)
    cfg["fastloop"]["overlap_policy"] = True
    Path(cfg["policy"]["initial_real_returns"]).write_bytes(b"initial critic data")
    root = Path(cfg["loop_root"])
    arm = tmp_path / "arm"
    arm.mkdir()
    original = loop.load_state(cfg)["accepted_wm"]["sha256"]
    started, finished = threading.Event(), threading.Event()
    events = []

    def policy(cfg_, snapshot, ph, root_):
        events.append("policy1_start")
        started.set()
        checkpoint = root / "policy1.pt"
        checkpoint.write_bytes(b"overlapped policy")
        snapshot["policy"] = {"checkpoint": str(checkpoint), "sha256": loop.sha256(checkpoint),
                              "trained_on_wm_sha256": snapshot["accepted_wm"]["sha256"]}
        finished.set()
        return {"updates": 2}

    def phase(name):
        def run(cfg_, state, ph, root_):
            k = state["cycles"][-1]["cycle"]
            events.append(f"{name}{k}")
            if name == "collect":
                assert started.wait(5), "CPU collection should overlap policy1"
                if k == 1:
                    assert state["policy"]["trained_on_wm_sha256"] == original
                    assert state["policy_wm"]["sha256"] == original
                    assert state["accepted_wm"]["sha256"] != original
                return {"branch_collisions": 0, "branch_teleports": 0}
            if name == "convert":
                target = root / f"returns{k}.npz"
                target.write_bytes(b"fresh current-policy returns")
                return {"returns": str(target), "pairs": "mechanics"}
            if name == "wm_ft":
                assert finished.is_set(), "GPU WM-FT must wait for overlapped GPU policy"
                target = root / f"candidate{k}.pt"
                target.write_bytes(f"new WM {k}".encode())
                return {"candidate": str(target), "candidate_sha256": loop.sha256(target)}
            ft = state["cycles"][-1]["phases"]["wm_ft"]["receipt"]
            state["accepted_wm"] = {"checkpoint": ft["candidate"], "sha256": ft["candidate_sha256"], "config": "ft.toml"}
            return {"decision": "ACCEPT"}
        return run

    monkeypatch.setattr(loop, "run_policy", policy)
    monkeypatch.setattr(loop, "fast_module", lambda: fast)
    monkeypatch.setattr(loop, "RUNNERS", {name: phase(name) for name in ("collect", "convert", "wm_ft", "check")})
    waiter = SimpleNamespace(prepare_horizon=lambda *a: None, retire_pair_cache=lambda *a: None)
    fast.drive(loop, waiter, cfg, arm)
    assert events.count("policy1_start") == 1
    assert events.index("policy1_start") < events.index("wm_ft0")
    assert "max_cycles" in loop.load_state(cfg)["stopped"]
