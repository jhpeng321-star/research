"""CV21 predecessor, roster and oracle tests; no simulation."""
import importlib.util
import json
from pathlib import Path

import pytest

from output.controller_verification_v21 import prepare_registration as prep

ROOT = Path(__file__).resolve().parents[1]
V20 = ROOT / "output/controller_verification_v20/independent_review.py"
DERIVED = ROOT / "output/controller_verification_v21/independent_review.py"
V20_SHA = "3a6d7745d072c833bb767ae2c38f0cec8c25b69c17a1abeabae78271d87406e6"
PRESERVED_RUN = ROOT / "runs/active/controller_verification_v20/jobs/7001m2000-retain"
CV20_REGISTRATION = ROOT / "output/controller_verification_v20/REGISTRATION.json"


def canonical():
    SPEC = importlib.util.spec_from_file_location(
        "cv20p", ROOT / "output/controller_verification_v20/prepare_registration.py")
    _mod = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(_mod)
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
    with pytest.raises(ValueError, match="21 complete"):
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
            data["reservations"][21]["status"] = "complete"
            return json.dumps(data)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="21 complete"):
        prep.build_registration()


def test_predecessor_interrupted_receipt_absence_is_rejected(monkeypatch):
    real_is_file = Path.is_file
    target = (prep.ROOT / "runs/active/controller_verification_v20/jobs/7008m2700-retain"
              / "terminal_receipt.json").resolve()

    def changed(self, *args, **kwargs):
        if Path(self).resolve() == target:
            return False
        return real_is_file(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", changed)
    with pytest.raises(ValueError, match="interrupted receipt is missing"):
        prep.build_registration()


def test_roster_exact():
    reg = canonical()
    assert reg["campaign"] == "cv21_corpus_completion"
    assert len(reg["cases"]) == 15
    assert reg["cases"][0] == "7008m2700-retain"
    assert reg["cases"][-1] == "7012m2700-pi-external-u"
    assert reg["budget"]["ceilings"]["total_seconds"] == 36000
    assert reg["cumulative_potential_seconds"] == 160800
    assert reg["prior_budget_stopped"] is True
    assert reg["prior_reserved_seconds"] == 52800
    assert {j["loading"] for j in reg["jobs"].values()} == {2700}
    assert {j["seed"] for j in reg["jobs"].values()} == set(range(7008, 7013))


def test_v20_oracle_is_unchanged():
    import hashlib
    assert hashlib.sha256(V20.read_bytes()).hexdigest() == V20_SHA


def test_derived_oracle_differs_only_in_scope():
    def split(text):
        end = text.index('"""', text.index('"""') + 3) + 3
        return text[:end], text[end:]

    _, v20_body = split(V20.read_text(encoding="utf-8"))
    _, derived_body = split(DERIVED.read_text(encoding="utf-8"))
    reverted = derived_body.replace(
        "\n#: Registered CV-21 (seed, mainline_vph) cells, mirroring the registration.\n"
        "REGISTERED_CELLS = {(s, 2700) for s in range(7008, 7013)}\n",
        "\n#: Registered CV-20 (seed, mainline_vph) cells, mirroring the registration.\n"
        "REGISTERED_CELLS = {(s, 2000) for s in range(7001, 7007)}"
        " | {(s, 2700) for s in range(7007, 7013)}\n")
    reverted = reverted.replace(
        '        "registered CV21 scope",', '        "registered CV20 scope",')
    assert reverted == v20_body


def test_registered_cells_exact():
    from output.controller_verification_v21 import independent_review as oracle
    assert oracle.REGISTERED_CELLS == {(s, 2700) for s in range(7008, 7013)}
    assert (4107, 2000) not in oracle.REGISTERED_CELLS
    assert (7001, 2000) not in oracle.REGISTERED_CELLS


def test_preserved_out_of_scope_run_fails_closed_on_scope():
    run = PRESERVED_RUN / "run"
    if not CV20_REGISTRATION.is_file() or not (run / "manifest.json").is_file():
        pytest.skip("preserved CV20 evidence absent")
    reg = json.loads(CV20_REGISTRATION.read_text(encoding="utf-8"))
    from output.controller_verification_v21 import independent_review as oracle
    result = oracle.verify_run(str(run), reg["jobs"]["7001m2000-retain"]["expected"])
    assert result["integrity_pass"] is False
    assert result["failures"] == ["registered CV21 scope"]
