"""CV20 predecessor contract rejection without changing preserved evidence."""
import json
from pathlib import Path
import pytest
from output.controller_verification_v20 import prepare_registration as prep


@pytest.mark.parametrize(
    "field,value",
    [("amount_s", 1200), ("completed_simulated_s", 100.0), ("status", "failed")],
)
def test_predecessor_reservation_mutation_is_rejected(monkeypatch, field, value):
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
    with pytest.raises(ValueError, match="CV19"):
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


def test_predecessor_wrong_reservation_count_is_rejected(monkeypatch):
    original = Path.read_text
    target = (prep.ROOT / prep.PRIOR_LEDGER).resolve()

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.resolve() == target:
            data = json.loads(text)
            data["reservations"] = data["reservations"][:3]
            return json.dumps(data)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="six complete"):
        prep.build_registration()


def test_missing_predecessor_accepted_review_is_rejected(monkeypatch):
    real_is_file = Path.is_file
    target = (prep.ROOT / "runs/active/controller_verification_v19/jobs/4107-retain"
              / "accepted_review.json").resolve()

    def changed(self, *args, **kwargs):
        if Path(self).resolve() == target:
            return False
        return real_is_file(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", changed)
    with pytest.raises(ValueError, match="accepted reviews"):
        prep.build_registration()
