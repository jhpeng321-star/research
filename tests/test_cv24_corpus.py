"""CV24 predecessor, roster and oracle tests; no simulation."""
import json
from pathlib import Path

import pytest

from output.controller_verification_v24 import prepare_registration as prep

ROOT = Path(__file__).resolve().parents[1]
V23 = ROOT / "output/controller_verification_v23/independent_review.py"
DERIVED = ROOT / "output/controller_verification_v24/independent_review.py"
V23_SHA = "0d2937b3ebc3946dc211fdfd79a7503550ee5af3c23ea7fb040e02590286ea9f"
PRESERVED_RUN = ROOT / "runs/active/controller_verification_v23/jobs/7010m2700-retain"


def canonical():
    return prep.build_registration(ROOT)


@pytest.mark.parametrize(
    "field,value",
    [("amount_s", 1200), ("completed_simulated_s", 100.0), ("status", "failed")],
)
def test_predecessor_complete_reservation_mutation_is_rejected(monkeypatch, field, value):
    original = Path.read_text
    target = (prep.ROOT / prep.PRIOR_LEDGER).resolve()

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.resolve() == target:
            data = json.loads(text)
            data["reservations"][0][field] = value
            return json.dumps(data)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="4 complete"):
        prep.build_registration()


def test_predecessor_failed_reservation_mutation_is_rejected(monkeypatch):
    original = Path.read_text
    target = (prep.ROOT / prep.PRIOR_LEDGER).resolve()

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.resolve() == target:
            data = json.loads(text)
            data["reservations"][3]["completed_simulated_s"] = 2400.0
            return json.dumps(data)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="not the pinned state"):
        prep.build_registration()


def test_predecessor_stop_removal_is_rejected(monkeypatch):
    original = Path.read_text
    target = (prep.ROOT / prep.PRIOR_LEDGER).resolve()

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.resolve() == target:
            data = json.loads(text)
            data["campaigns"][prep.PRIOR_CAMPAIGN]["stopped"] = False
            return json.dumps(data)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="not the pinned stopped state"):
        prep.build_registration()


def test_roster_exact():
    reg = canonical()
    assert reg["campaign"] == "cv24_corpus_completion"
    assert len(reg["cases"]) == 6
    assert reg["cases"][0] == "7011m2700-retain"
    assert reg["cases"][-1] == "7012m2700-pi-external-u"
    assert reg["budget"]["ceilings"]["total_seconds"] == 14400
    assert reg["cumulative_potential_seconds"] == 218400
    assert reg["prior_budget_stopped"] is True
    assert reg["prior_reserved_seconds"] == 12000
    assert reg["prior_reached_seconds"] == 9948.5
    assert {j["seed"] for j in reg["jobs"].values()} == {7011, 7012}
    assert {j["loading"] for j in reg["jobs"].values()} == {2700}


def test_v23_oracle_is_unchanged():
    import hashlib
    assert hashlib.sha256(V23.read_bytes()).hexdigest() == V23_SHA


def test_derived_oracle_differs_only_in_scope():
    def split(text):
        end = text.index('"""', text.index('"""') + 3) + 3
        return text[:end], text[end:]

    _, v23_body = split(V23.read_text(encoding="utf-8"))
    _, derived_body = split(DERIVED.read_text(encoding="utf-8"))
    reverted = derived_body.replace(
        "#: Registered CV-24 (seed, mainline_vph) cells, mirroring the registration.\n"
        "REGISTERED_CELLS = {(s, 2700) for s in range(7011, 7013)}\n",
        "#: Registered CV-23 (seed, mainline_vph) cells, mirroring the registration.\n"
        "REGISTERED_CELLS = {(s, 2700) for s in range(7010, 7013)}\n")
    reverted = reverted.replace(
        '        "registered CV24 scope",', '        "registered CV23 scope",')
    assert reverted == v23_body


def test_registered_cells_exact():
    from output.controller_verification_v24 import independent_review as oracle
    assert oracle.REGISTERED_CELLS == {(s, 2700) for s in range(7011, 7013)}
    assert (7010, 2700) not in oracle.REGISTERED_CELLS


def test_preserved_out_of_scope_run_fails_closed_on_scope():
    run = PRESERVED_RUN / "run"
    cv23_reg = ROOT / "output/controller_verification_v23/REGISTRATION.json"
    if not cv23_reg.is_file() or not (run / "manifest.json").is_file():
        pytest.skip("preserved CV23 evidence absent")
    from output.controller_verification_v24 import independent_review as oracle
    expected = json.loads(
        cv23_reg.read_text(encoding="utf-8"))["jobs"]["7010m2700-retain"]["expected"]
    result = oracle.verify_run(str(run), expected)
    assert result["integrity_pass"] is False
    assert result["failures"] == ["registered CV24 scope"]


def test_collision_failure_mode_is_charge_free_and_stop_free(tmp_path, monkeypatch):
    """Concurrent-lane registration: a job whose directory already exists
    fails before constructing the guard (no charge, no campaign stop)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cv24_runner", ROOT / "output/controller_verification_v24/runner.py")
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
    out = Path(reg["jobs"]["7011m2700-retain"]["output_dir"])
    out.mkdir(parents=True)
    monkeypatch.setattr(
        runner, "BudgetGuard",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("guard constructed")))
    with pytest.raises(FileExistsError):
        runner.run_job(reg, job="7011m2700-retain", root=ROOT)
    assert not (tmp_path / "ledger.json").exists()
