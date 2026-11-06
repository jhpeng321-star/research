"""Fail-closed base selection and the integrated arm's cross-cycle/config contracts."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.main14.verdict_snapshot import choose_base, snapshot  # noqa: E402


def load_loop():
    spec = importlib.util.spec_from_file_location("main14_test_loop", ROOT / "scripts/c4/loop14.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_waiter():
    sys.path.insert(0, str(ROOT / "scripts/c4"))
    spec = importlib.util.spec_from_file_location("main14_test_waiter", ROOT / "scripts/c4/main14_autolaunch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qualification(ratio=1.0):
    return {"core": {"sign:pulse_m/follower": "FAIL"},
            "support": {"mag:pulse_m/follower": "PASS"},
            "table": {"set1": {"sign:gap_open/follower": "INCONCLUSIVE"}},
            "per_set": {"set1": {"pulse_m/follower@5": {"sign_verdict": "FAIL", "mag_verdict": "PASS"}}},
            "fid": {"all_pairs": {"ratio": ratio}}}


def test_base_rule_accepts_same_fails_and_equal_strict_ratio():
    ref = qualification()
    assert choose_base(ref, copy.deepcopy(ref))["base"] == "14b"


@pytest.mark.parametrize("location", ["core", "support", "table", "per_set"])
def test_new_reaction_fail_forces_exact_fallback(location):
    ref, cand = qualification(), qualification(0.5)
    if location == "core":
        cand["core"]["sign:new/follower"] = "FAIL"
    elif location == "support":
        cand["support"]["mag:pulse_m/follower"] = "FAIL"
    elif location == "table":
        cand["table"]["set1"]["sign:gap_open/follower"] = "FAIL"
    else:
        cand["per_set"]["set1"]["pulse_m/follower@5"]["mag_verdict"] = "FAIL"
    assert choose_base(ref, cand)["base"] == "13e_b2"


def test_strict_ttc_worse_forces_fallback_and_missing_items_refuse():
    assert choose_base(qualification(1), qualification(1.00001))["base"] == "13e_b2"
    cand = qualification()
    cand.pop("per_set")
    with pytest.raises(ValueError, match="missing"):
        choose_base(qualification(), cand)
    with pytest.raises(ValueError, match="nonfinite"):
        choose_base(qualification(), qualification(float("nan")))


def test_snapshot_does_not_treat_fidelity_alone_as_qualification(tmp_path):
    p = tmp_path / "autopilot_r10/runs/r14b/score/r14b"
    p.mkdir(parents=True)
    (p / "fid.json").write_text(json.dumps({"ratio": 0.5}))
    assert snapshot(tmp_path)["ready"] is False


def test_chain_requires_both_exact_pid_exit_and_summary(tmp_path, monkeypatch):
    waiter = load_waiter()
    summary = tmp_path / "RUN_SUMMARY.json"
    monkeypatch.setattr(waiter, "proc_identity", lambda pid: "123")
    arm = {"round14b_start_ticks": "123"}
    assert not waiter.chain_finished(arm, summary)
    summary.write_text("{}")
    assert not waiter.chain_finished(arm, summary)
    for identity in (None, "456"):  # exited/zombie, or PID reused by another process
        monkeypatch.setattr(waiter, "proc_identity", lambda pid: identity)
        assert waiter.chain_finished(arm, summary)
    summary.unlink()
    assert not waiter.chain_finished(arm, summary)


def test_scoring_deadline_cannot_be_satisfied_by_late_or_partial_verdict():
    waiter = load_waiter()
    assert waiter.timely_verdict({"ready": True, "verdict_mtime": 100}, 100)
    assert not waiter.timely_verdict({"ready": True, "verdict_mtime": 101}, 100)
    assert not waiter.timely_verdict({"ready": False}, 100)


def test_source_mutation_refuses_before_launch(tmp_path, monkeypatch):
    waiter = load_waiter()
    monkeypatch.setattr(waiter, "ROOT", tmp_path)
    p = tmp_path / "source.py"
    p.write_text("before")
    arm = {"source_files": {"source.py": waiter.digest(p)}, "registration": str(p),
           "registration_sha256": waiter.digest(p), "config": str(p), "config_sha256": waiter.digest(p)}
    waiter.verify_source(arm)
    p.write_text("after")
    with pytest.raises(RuntimeError, match="source changed"):
        waiter.verify_source(arm)


def test_registered_integrated_budget_and_policy_arguments(tmp_path):
    loop = load_loop()
    cfg = loop.read_json(ROOT / "configs/c4_loop14_main14_gpur.json")
    reg = loop.read_json(ROOT / "configs/c4_main14_registration.json")
    assert loop.collect_sim_s(cfg) == 29133
    assert 4 * loop.collect_sim_s(cfg) <= reg["budget"]["sim_s_total"]
    assert cfg["collect"]["workers"] <= 16
    assert reg["exclude"] == [[968100, 968102]]
    state = {"cycles": [loop.new_cycle(1)], "accepted_wm": {"checkpoint": "wm.pt", "config": "wm.json"},
             "real_returns": "R.npz", "real_credit": "REAL_CREDIT.json", "policy": {"checkpoint": "policy.pt"}}
    argv, _env = loop.policy_cmd(cfg, state, tmp_path)
    assert argv[argv.index("--max-updates") + 1] == "150"
    assert "--gate-panel" in argv and "--horizon-gate" in argv and "--real-credit" in argv


def test_fixed_start_split_is_used_across_optimisation_seeds(monkeypatch):
    from wms.control import c4_starts, train_cli
    from wms.config.train import TrainConfig
    from wms.config import schema
    import wms.data.registry

    monkeypatch.setattr(train_cli, "checked_path", lambda p: Path(p))
    monkeypatch.setattr(train_cli, "_read_config", lambda p: {})
    source = SimpleNamespace(data=SimpleNamespace(c4_base=True))
    monkeypatch.setattr(schema, "from_mapping", lambda cls, raw: source if cls is TrainConfig else None)
    monkeypatch.setattr(wms.data.registry, "resolve", lambda p: p)
    monkeypatch.setattr(c4_starts, "build_c4_start_windows", lambda *a, **k: SimpleNamespace())
    observed = []
    monkeypatch.setattr(train_cli, "split_start_states", lambda ds, **kwargs: observed.append(kwargs["seed"]))
    for seed in (965140, 965141, 965142, 965143):
        train_cli._load_starts({"split_seed": 965140}, [SimpleNamespace(history_s=10)], seed)
    assert observed == [965140] * 4


def test_full_gated_horizon_curriculum_and_24gb_settings():
    from wms.control.train_cli import _read_config
    raw = _read_config(ROOT / "configs/c4_policy14_main14.toml")
    assert raw["horizon"]["horizons_s"] == [10, 30, 60]
    assert raw["horizon"]["curriculum_updates"] == [0, 150, 300]
    assert raw["trainer"]["batch_size"] == 1 and raw["trainer"]["checkpoint_imagination"]
    assert raw["uncertainty"]["future_samples"] == 2
    assert raw["counterfactual"]["w_coop"][-1] <= raw["counterfactual"]["w_individual"][-1]
