"""Launcher gate tests: registration, model lock, authorization, budget."""
from __future__ import annotations

import json

import pytest

from inc1 import constants as C
from inc1 import gates as G


def _lock(**overrides):
    lock = {
        "schema": C.MODEL_LOCK_SCHEMA,
        "registration_lf_sha256": C.REGISTRATION_LF_SHA,
        "member_1_sha256": "a" * 64,
        "pack_manifest_sha256": "b" * 64,
        "code_commit": "c221984",
        "partA_v2g_param_sha256": "c" * 64,
        "zero_shot": {"event_tokens_nonempty": 0, "obstacle_insertions": 0},
        "template": False,
    }
    lock.update(overrides)
    return lock


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_real_registration_passes():
    assert G.check_registration() == C.REGISTRATION_LF_SHA


def test_tampered_registration_refused(tmp_path):
    fake = tmp_path / "REG.md"
    fake.write_text("not the registration\n", encoding="utf-8")
    with pytest.raises(G.GateError):
        G.check_registration(fake)


def test_missing_model_lock_refused(tmp_path):
    with pytest.raises(G.GateError):
        G.check_model_lock(tmp_path / "absent.json")


def test_template_lock_refused(tmp_path):
    path = _write(tmp_path, "lock.json", _lock(template=True))
    with pytest.raises(G.GateError):
        G.check_model_lock(path)


def test_nonzero_zero_shot_refused(tmp_path):
    path = _write(tmp_path, "lock.json", _lock(
        zero_shot={"event_tokens_nonempty": 1, "obstacle_insertions": 0}))
    with pytest.raises(G.GateError):
        G.check_model_lock(path)


def test_valid_lock_passes_and_hashes(tmp_path):
    path = _write(tmp_path, "lock.json", _lock())
    lock = G.check_model_lock(path)
    assert lock["code_commit"] == "c221984"
    # the lock is JSON text: its gate sha is LF-normalised (CRLF-safe)
    assert lock["_sha256"] == C.lf_sha256(path)


def test_authorization_must_bind_registration_and_lock(tmp_path):
    lock = G.check_model_lock(_write(tmp_path, "lock.json", _lock()))
    wrong = _write(tmp_path, "auth_wrong.json", {
        "schema": C.AUTHORIZATION_SCHEMA,
        "registration_lf_sha256": "0" * 64,
        "model_lock_path": str(lock["_path"]),
        "model_lock_sha256": lock["_sha256"],
        "budget_total_sim_s": C.BUDGET_TOTAL_S,
    })
    with pytest.raises(G.GateError):
        G.check_authorization(wrong, model_lock=lock)
    tampered = _write(tmp_path, "auth_tampered.json", {
        "schema": C.AUTHORIZATION_SCHEMA,
        "registration_lf_sha256": C.REGISTRATION_LF_SHA,
        "model_lock_path": str(lock["_path"]),
        "model_lock_sha256": "f" * 64,
        "budget_total_sim_s": C.BUDGET_TOTAL_S,
    })
    with pytest.raises(G.GateError):
        G.check_authorization(tampered, model_lock=lock)
    good = _write(tmp_path, "auth.json", {
        "schema": C.AUTHORIZATION_SCHEMA,
        "registration_lf_sha256": C.REGISTRATION_LF_SHA,
        "model_lock_path": str(lock["_path"]),
        "model_lock_sha256": lock["_sha256"],
        "budget_total_sim_s": C.BUDGET_TOTAL_S,
    })
    auth = G.check_authorization(good, model_lock=lock)
    assert auth["_path"] == str(good)


def test_generators_manifest_gate_binds_the_shared_package():
    """A2.6: the launcher gate verifies the live R5/R6 generator package."""
    record = G.check_generators_manifest()
    assert record["generator_module_sha256"]
    assert record["generators_manifest"].endswith("INC1_GENERATORS.json")


def test_generators_manifest_gate_refuses_a_missing_manifest(tmp_path):
    with pytest.raises(G.GateError):
        G.check_generators_manifest(tmp_path / "absent.json")


def test_budget_refusal_and_charge(tmp_path):
    counter = G.BudgetCounter(tmp_path / "budget.jsonl", limit_sim_s=100.0)
    counter.charge(item="warmup", sim_s=60.0)
    assert counter.committed_sim_s() == 60.0
    with pytest.raises(G.GateError):
        counter.check(50.0)
    counter.charge(item="arm", sim_s=30.0)
    assert counter.check(10.0)["remaining_sim_s"] == pytest.approx(10.0)


def test_complete_run_never_retried(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "RUN_RECEIPT.json").write_text("{}", encoding="utf-8")
    with pytest.raises(G.GateError):
        G.refuse_retry_of_complete(run_dir)


def test_crashed_run_restarts_once(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert G.crashed_restart_allowed(run_dir) is True
    G.mark_crashed_restart(run_dir, detail="first crash")
    assert G.crashed_restart_allowed(run_dir) is False
