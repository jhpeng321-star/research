"""CV22 predecessor, roster and oracle tests; no simulation."""
import json
from pathlib import Path

import pytest

from output.controller_verification_v22 import prepare_registration as prep

ROOT = Path(__file__).resolve().parents[1]
V21 = ROOT / "output/controller_verification_v21/independent_review.py"
DERIVED = ROOT / "output/controller_verification_v22/independent_review.py"
V21_SHA = "d4043ec88e51464be09d1daa502d7260aa0ea3e4dfe67ef00193df520d108625"
PRESERVED_RUN = ROOT / "runs/active/controller_verification_v21/jobs/7008m2700-retain"
CV21_REGISTRATION = ROOT / "output/controller_verification_v21/REGISTRATION.json"


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


def test_predecessor_failed_reservation_mutation_is_rejected(monkeypatch):
    original = Path.read_text
    target = (prep.ROOT / prep.PRIOR_LEDGER).resolve()

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.resolve() == target:
            data = json.loads(text)
            data["reservations"][4]["completed_simulated_s"] = 0.0
            return json.dumps(data)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="pinned state"):
        prep.build_registration()


def test_roster_exact():
    reg = canonical()
    assert reg["campaign"] == "cv22_corpus_final"
    assert len(reg["cases"]) == 9
    assert reg["cases"][0] == "7010m2700-retain"
    assert reg["cases"][-1] == "7012m2700-pi-external-u"
    assert reg["budget"]["ceilings"]["total_seconds"] == 21600
    assert reg["cumulative_potential_seconds"] == 182400
    assert reg["prior_budget_stopped"] is True
    assert reg["prior_reserved_seconds"] == 12000
    assert reg["prior_reached_seconds"] == 9717.0
    assert {j["loading"] for j in reg["jobs"].values()} == {2700}
    assert {j["seed"] for j in reg["jobs"].values()} == {7010, 7011, 7012}


def test_v21_oracle_is_unchanged():
    import hashlib
    assert hashlib.sha256(V21.read_bytes()).hexdigest() == V21_SHA


def test_derived_oracle_differs_only_in_scope():
    def split(text):
        end = text.index('"""', text.index('"""') + 3) + 3
        return text[:end], text[end:]

    _, v21_body = split(V21.read_text(encoding="utf-8"))
    _, derived_body = split(DERIVED.read_text(encoding="utf-8"))
    reverted = derived_body.replace(
        "\n#: Registered CV-22 (seed, mainline_vph) cells, mirroring the registration.\n"
        "REGISTERED_CELLS = {(s, 2700) for s in range(7010, 7013)}\n",
        "\n#: Registered CV-21 (seed, mainline_vph) cells, mirroring the registration.\n"
        "REGISTERED_CELLS = {(s, 2700) for s in range(7008, 7013)}\n")
    reverted = reverted.replace(
        '        "registered CV22 scope",', '        "registered CV21 scope",')
    assert reverted == v21_body


def test_registered_cells_exact():
    from output.controller_verification_v22 import independent_review as oracle
    assert oracle.REGISTERED_CELLS == {(s, 2700) for s in range(7010, 7013)}
    assert (7008, 2700) not in oracle.REGISTERED_CELLS
    assert (7009, 2700) not in oracle.REGISTERED_CELLS


def test_preserved_out_of_scope_run_fails_closed_on_scope():
    run = PRESERVED_RUN / "run"
    if not CV21_REGISTRATION.is_file() or not (run / "manifest.json").is_file():
        pytest.skip("preserved CV21 evidence absent")
    reg = json.loads(CV21_REGISTRATION.read_text(encoding="utf-8"))
    from output.controller_verification_v22 import independent_review as oracle
    result = oracle.verify_run(str(run), reg["jobs"]["7008m2700-retain"]["expected"])
    assert result["integrity_pass"] is False
    assert result["failures"] == ["registered CV22 scope"]
