"""CV27 predecessor, roster and oracle tests; no simulation."""
import json
from pathlib import Path

import pytest

from output.controller_verification_v27 import prepare_registration as prep

ROOT = Path(__file__).resolve().parents[1]
V25 = ROOT / "output/controller_verification_v25/independent_review.py"
DERIVED = ROOT / "output/controller_verification_v27/independent_review.py"
V25_SHA = "4357c0b31cacf5173f934cecebd56d39450b6bc40b45758db6a057b9204d67cb"


def canonical():
    return prep.build_registration(ROOT)


def _mutate_prior(monkeypatch, mutate):
    original = Path.read_text
    target = (prep.ROOT / prep.PRIOR_LEDGER).resolve()

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.resolve() == target:
            data = json.loads(text)
            mutate(data)
            return json.dumps(data)
        return text

    monkeypatch.setattr(Path, "read_text", changed)


def test_predecessor_stop_removal_is_rejected(monkeypatch):
    _mutate_prior(monkeypatch, lambda d: d["campaigns"][prep.PRIOR_CAMPAIGN].update(stopped=False))
    with pytest.raises(ValueError, match="not the pinned stopped state"):
        prep.build_registration()


def test_predecessor_reservation_mutation_is_rejected(monkeypatch):
    def flip(d):
        for r in d["reservations"]:
            if r.get("status") == "failed":
                r["status"] = "complete"
    _mutate_prior(monkeypatch, flip)
    with pytest.raises(ValueError, match="4 complete"):
        prep.build_registration()


def test_predecessor_detail_mutation_is_rejected(monkeypatch):
    def detail(d):
        for r in d["reservations"]:
            if r.get("status") == "failed":
                r["detail"] = "OSError('other')"
    _mutate_prior(monkeypatch, detail)
    with pytest.raises(ValueError, match="not the pinned state"):
        prep.build_registration()


def test_predecessor_receipt_frontier_mutation_is_rejected(monkeypatch):
    original = Path.read_text
    target = (ROOT / "runs/active/controller_verification_v25/jobs/7012m2700-native"
              / "terminal_receipt.json").resolve()

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.resolve() == target:
            data = json.loads(text)
            data["actual_simulated_seconds"] = 1.0
            return json.dumps(data)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="pinned failed_or_interrupted"):
        prep.build_registration()


def test_roster_exact():
    reg = canonical()
    assert reg["campaign"] == "cv27_7012_completion"
    assert reg["cases"] == ["7012m2700-retain", "7012m2700-native", "7012m2700-pi-external-u"]
    assert reg["budget"]["ceilings"]["total_seconds"] == 7200
    assert reg["cumulative_potential_seconds"] == 240000
    assert reg["prior_budget_stopped"] is True
    assert reg["prior_reserved_seconds"] == 12000
    assert reg["prior_reached_seconds"] == 10328.0
    assert {j["seed"] for j in reg["jobs"].values()} == {7012}
    assert {j["loading"] for j in reg["jobs"].values()} == {2700}


def test_v25_oracle_is_unchanged():
    import hashlib
    assert hashlib.sha256(V25.read_bytes()).hexdigest() == V25_SHA


def test_derived_oracle_registers_only_7012():
    from output.controller_verification_v27 import independent_review as oracle
    assert oracle.REGISTERED_CELLS == {(7012, 2700)}
    assert (7011, 2700) not in oracle.REGISTERED_CELLS


def test_derived_oracle_body_revert_proof():
    def split(text):
        end = text.index('"""', text.index('"""') + 3) + 3
        return text[:end], text[end:]

    _, v25_body = split(V25.read_text(encoding="utf-8"))
    _, derived_body = split(DERIVED.read_text(encoding="utf-8"))
    reverted = derived_body.replace(
        "#: Registered CV-27 (seed, mainline_vph) cells, mirroring the registration.\n"
        "REGISTERED_CELLS = {(7012, 2700)}\n",
        "#: Registered CV-25 (seed, mainline_vph) cells, mirroring the registration.\n"
        "REGISTERED_CELLS = {(s, 2700) for s in range(7011, 7013)}\n")
    reverted = reverted.replace(
        '        "registered CV27 scope",', '        "registered CV25 scope",')
    assert reverted == v25_body


def test_preserved_out_of_scope_run_fails_closed_on_scope():
    run = ROOT / "runs/active/controller_verification_v25/jobs/7011m2700-retain/run"
    cv25_reg = ROOT / "output/controller_verification_v25/REGISTRATION.json"
    if not cv25_reg.is_file() or not (run / "manifest.json").is_file():
        pytest.skip("preserved CV25 evidence absent")
    from output.controller_verification_v27 import independent_review as oracle
    expected = json.loads(
        cv25_reg.read_text(encoding="utf-8"))["jobs"]["7011m2700-retain"]["expected"]
    result = oracle.verify_run(str(run), expected)
    assert result["integrity_pass"] is False
    assert result["failures"] == ["registered CV27 scope"]


def test_collision_failure_mode_is_charge_free_and_stop_free(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cv27_runner", ROOT / "output/controller_verification_v27/runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    reg = canonical()
    reg["output_root"] = str((tmp_path / "jobs").resolve())
    reg["ledger_path"] = str((tmp_path / "ledger.json").resolve())
    for job in reg["jobs"].values():
        out = tmp_path / "jobs" / Path(job["output_dir"]).name
        job["output_dir"] = str(out.resolve())
        job["run_dir"] = str((out / "run").resolve())
    monkeypatch.setattr(
        runner, "_validate_registration",
        lambda _registration, _root, job, _sumo_binary=None: reg["jobs"][job])
    out = Path(reg["jobs"]["7012m2700-retain"]["output_dir"])
    out.mkdir(parents=True)
    monkeypatch.setattr(
        runner, "BudgetGuard",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("guard constructed")))
    with pytest.raises(FileExistsError):
        runner.run_job(reg, job="7012m2700-retain", root=ROOT)
    assert not (tmp_path / "ledger.json").exists()
