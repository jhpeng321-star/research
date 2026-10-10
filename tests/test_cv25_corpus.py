"""CV25 predecessor, roster and oracle tests; no simulation."""
import json
from pathlib import Path

import pytest

from output.controller_verification_v25 import prepare_registration as prep

ROOT = Path(__file__).resolve().parents[1]
V24 = ROOT / "output/controller_verification_v24/independent_review.py"
DERIVED = ROOT / "output/controller_verification_v25/independent_review.py"
V24_SHA = "148294946b459e584cd02bec3cfa2973f5caea7522cca00f257ac3d584585f82"


def canonical():
    return prep.build_registration(ROOT)


def test_predecessor_wrong_reservation_count_is_rejected(monkeypatch):
    original = Path.read_text
    target = (prep.ROOT / prep.PRIOR_LEDGER).resolve()

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.resolve() == target:
            data = json.loads(text)
            data["reservations"].append(dict(id="x", campaign=prep.PRIOR_CAMPAIGN,
                                           status="failed", amount_s=2400))
            return json.dumps(data)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="exactly 2"):
        prep.build_registration()


def test_predecessor_failed_detail_mutation_is_rejected(monkeypatch):
    original = Path.read_text
    target = (prep.ROOT / prep.PRIOR_LEDGER).resolve()

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.resolve() == target:
            data = json.loads(text)
            for r in data["reservations"]:
                if r.get("status") == "failed":
                    r["detail"] = "OSError('other')"
            return json.dumps(data)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="not the pinned state"):
        prep.build_registration()


def test_predecessor_receipt_frontier_mutation_is_rejected(monkeypatch):
    original = Path.read_text
    target = (prep.ROOT / "runs/active/controller_verification_v24/jobs/7012m2700-retain"
              / "terminal_receipt.json").resolve()

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.resolve() == target:
            data = json.loads(text)
            data["actual_simulated_seconds"] = 999.0
            return json.dumps(data)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="pinned failed_or_interrupted"):
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
    assert reg["campaign"] == "cv25_corpus_finish"
    assert len(reg["cases"]) == 6
    assert reg["cases"][0] == "7011m2700-retain"
    assert reg["cases"][-1] == "7012m2700-pi-external-u"
    assert reg["budget"]["ceilings"]["total_seconds"] == 14400
    assert reg["cumulative_potential_seconds"] == 232800
    assert reg["prior_budget_stopped"] is True
    assert reg["prior_reserved_seconds"] == 4800
    assert reg["prior_reached_seconds"] == 2842.5
    assert {j["seed"] for j in reg["jobs"].values()} == {7011, 7012}


def test_v24_oracle_is_unchanged():
    import hashlib
    assert hashlib.sha256(V24.read_bytes()).hexdigest() == V24_SHA


def test_derived_oracle_body_identical_to_v24():
    def split(text):
        end = text.index('"""', text.index('"""') + 3) + 3
        return text[:end], text[end:]

    _, v24_body = split(V24.read_text(encoding="utf-8"))
    _, derived_body = split(DERIVED.read_text(encoding="utf-8"))
    reverted = derived_body.replace(
        "#: Registered CV-25 (seed, mainline_vph) cells, mirroring the registration.",
        "#: Registered CV-24 (seed, mainline_vph) cells, mirroring the registration.")
    reverted = reverted.replace(
        '        "registered CV25 scope",', '        "registered CV24 scope",')
    assert reverted == v24_body


def test_registered_cells_exact():
    from output.controller_verification_v25 import independent_review as oracle
    assert oracle.REGISTERED_CELLS == {(s, 2700) for s in range(7011, 7013)}
