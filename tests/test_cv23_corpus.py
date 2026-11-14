"""CV23 predecessor, roster and oracle tests; no simulation."""
import json
from pathlib import Path

import pytest

from output.controller_verification_v23 import prepare_registration as prep

ROOT = Path(__file__).resolve().parents[1]
V22 = ROOT / "output/controller_verification_v22/independent_review.py"
DERIVED = ROOT / "output/controller_verification_v23/independent_review.py"
V22_SHA = "57907b9cf5916a05aad86d397008f19371e469086f25f983ce4b6b0b1e073ce8"
PRESERVED_RUN = ROOT / "runs/active/controller_verification_v21/jobs/7008m2700-retain"
CV22_REGISTRATION = ROOT / "output/controller_verification_v22/REGISTRATION.json"


def canonical():
    return prep.build_registration(ROOT)


def test_predecessor_shape_is_rejected_on_mutation(monkeypatch):
    original = Path.read_text
    target = (prep.ROOT / prep.PRIOR_LEDGER).resolve()

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.resolve() == target:
            data = json.loads(text)
            data["reservations"][0]["completed_simulated_s"] = 2400.0
            return json.dumps(data)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="not the pinned state"):
        prep.build_registration()


def test_predecessor_status_flip_is_rejected(monkeypatch):
    original = Path.read_text
    target = (prep.ROOT / prep.PRIOR_LEDGER).resolve()

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.resolve() == target:
            data = json.loads(text)
            data["reservations"][0]["status"] = "complete"
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
    assert reg["campaign"] == "cv23_corpus_final"
    assert len(reg["cases"]) == 9
    assert reg["cases"][0] == "7010m2700-retain"
    assert reg["cases"][-1] == "7012m2700-pi-external-u"
    assert reg["budget"]["ceilings"]["total_seconds"] == 21600
    assert reg["cumulative_potential_seconds"] == 204000
    assert reg["prior_budget_stopped"] is True
    assert reg["prior_reserved_seconds"] == 2400
    assert reg["prior_reached_seconds"] == 0.0
    assert {j["seed"] for j in reg["jobs"].values()} == {7010, 7011, 7012}


def test_v22_oracle_is_unchanged():
    import hashlib
    assert hashlib.sha256(V22.read_bytes()).hexdigest() == V22_SHA


def test_derived_oracle_body_identical_to_v22():
    """No semantic change: only the docstring and scope message differ."""
    def split(text):
        end = text.index('"""', text.index('"""') + 3) + 3
        return text[:end], text[end:]

    _, v22_body = split(V22.read_text(encoding="utf-8"))
    _, derived_body = split(DERIVED.read_text(encoding="utf-8"))
    reverted = derived_body.replace(
        "#: Registered CV-23 (seed, mainline_vph) cells, mirroring the registration.",
        "#: Registered CV-22 (seed, mainline_vph) cells, mirroring the registration.")
    reverted = reverted.replace(
        '        "registered CV23 scope",', '        "registered CV22 scope",')
    assert reverted == v22_body


def test_registered_cells_exact():
    from output.controller_verification_v23 import independent_review as oracle
    assert oracle.REGISTERED_CELLS == {(s, 2700) for s in range(7010, 7013)}


def test_preserved_out_of_scope_run_fails_closed_on_scope():
    run = PRESERVED_RUN / "run"
    cv21_reg = ROOT / "output/controller_verification_v21/REGISTRATION.json"
    if not cv21_reg.is_file() or not (run / "manifest.json").is_file():
        pytest.skip("preserved CV21 evidence absent")
    from output.controller_verification_v23 import independent_review as oracle
    expected = json.loads(
        cv21_reg.read_text(encoding="utf-8"))["jobs"]["7008m2700-retain"]["expected"]
    result = oracle.verify_run(str(run), expected)
    assert result["integrity_pass"] is False
    assert result["failures"] == ["registered CV23 scope"]
