"""CV28 predecessor, roster and oracle tests; no simulation."""
import json
from pathlib import Path

import pytest

from output.controller_verification_v28 import prepare_registration as prep

ROOT = Path(__file__).resolve().parents[1]
V27 = ROOT / "output/controller_verification_v27/independent_review.py"
DERIVED = ROOT / "output/controller_verification_v28/independent_review.py"
V27_SHA = "5aac22182ceb231e605c0f55034de477063ea3810b9f5b440e3393a0f94d7ba8"


def canonical():
    return prep.build_registration(ROOT)


def test_predecessor_complete_shape_rejected_on_mutation(monkeypatch):
    original = Path.read_text
    target = (prep.ROOT / prep.PRIOR_LEDGER).resolve()

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.resolve() == target:
            data = json.loads(text)
            data["reservations"][0]["status"] = "failed"
            return json.dumps(data)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="3 complete"):
        prep.build_registration()


def test_predecessor_stop_injection_is_rejected(monkeypatch):
    original = Path.read_text
    target = (prep.ROOT / prep.PRIOR_LEDGER).resolve()

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.resolve() == target:
            data = json.loads(text)
            data["campaigns"][prep.PRIOR_CAMPAIGN] = {"stopped": True}
            return json.dumps(data)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="stopped"):
        prep.build_registration()


def test_predecessor_accepted_review_absence_is_rejected(monkeypatch):
    real_is_file = Path.is_file
    target = (ROOT / "runs/active/controller_verification_v27/jobs/7012m2700-native"
              / "accepted_review.json").resolve()

    def changed(self, *args, **kwargs):
        if Path(self).resolve() == target:
            return False
        return real_is_file(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", changed)
    with pytest.raises(ValueError, match="accepted review missing"):
        prep.build_registration()


def test_roster_exact():
    reg = canonical()
    assert reg["campaign"] == "cv28_validation_holdout"
    assert len(reg["cases"]) == 6
    assert reg["cases"][0] == "7101m2000-retain"
    assert reg["cases"][-1] == "7102m2700-pi-external-u"
    assert reg["budget"]["ceilings"]["total_seconds"] == 14400
    assert reg["cumulative_potential_seconds"] == 254400
    assert reg["prior_budget_stopped"] is False
    assert reg["prior_reserved_seconds"] == 7200
    assert {j["seed"] for j in reg["jobs"].values()} == {7101, 7102}
    assert {j["loading"] for j in reg["jobs"].values()} == {2000, 2700}


def test_v27_oracle_is_unchanged():
    import hashlib
    assert hashlib.sha256(V27.read_bytes()).hexdigest() == V27_SHA


def test_derived_oracle_differs_only_in_scope():
    def split(text):
        end = text.index('"""', text.index('"""') + 3) + 3
        return text[:end], text[end:]

    _, v27_body = split(V27.read_text(encoding="utf-8"))
    _, derived_body = split(DERIVED.read_text(encoding="utf-8"))
    reverted = derived_body.replace(
        "#: Registered CV-28 (seed, mainline_vph) cells, mirroring the registration.\n"
        "REGISTERED_CELLS = {(7101, 2000), (7102, 2700)}\n",
        "#: Registered CV-27 (seed, mainline_vph) cells, mirroring the registration.\n"
        "REGISTERED_CELLS = {(7012, 2700)}\n")
    reverted = reverted.replace(
        '        "registered CV28 scope",', '        "registered CV27 scope",')
    assert reverted == v27_body


def test_registered_cells_exact():
    from output.controller_verification_v28 import independent_review as oracle
    assert oracle.REGISTERED_CELLS == {(7101, 2000), (7102, 2700)}
    assert (7012, 2700) not in oracle.REGISTERED_CELLS


def test_preserved_out_of_scope_run_fails_closed_on_scope():
    run = ROOT / "runs/active/controller_verification_v27/jobs/7012m2700-retain/run"
    cv27_reg = ROOT / "output/controller_verification_v27/REGISTRATION.json"
    if not cv27_reg.is_file() or not (run / "manifest.json").is_file():
        pytest.skip("preserved CV27 evidence absent")
    from output.controller_verification_v28 import independent_review as oracle
    expected = json.loads(cv27_reg.read_text(encoding="utf-8"))["jobs"]["7012m2700-retain"]["expected"]
    result = oracle.verify_run(str(run), expected)
    assert result["integrity_pass"] is False
    assert result["failures"] == ["registered CV28 scope"]
