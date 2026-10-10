"""CV19 predecessor contract rejection without changing preserved evidence."""
import json
from pathlib import Path
import pytest
from output.controller_verification_v19 import prepare_registration as prep


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
    with pytest.raises(ValueError, match="CV18"):
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


def test_predecessor_receipt_flipping_is_rejected(monkeypatch):
    original = Path.read_text
    receipt_target = (prep.ROOT
                      / "runs/active/controller_verification_v18/jobs/4107-retain/terminal_receipt.json").resolve()

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.resolve() == receipt_target:
            data = json.loads(text)
            data["review_status"] = "accepted"
            return json.dumps(data)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="pinned completed/failed-review"):
        prep.build_registration()


def test_predecessor_receipt_absence_is_rejected(monkeypatch):
    real_is_file = Path.is_file
    receipt_target = (prep.ROOT
                      / "runs/active/controller_verification_v18/jobs/4107-retain/terminal_receipt.json").resolve()

    def changed(self, *args, **kwargs):
        if Path(self).resolve() == receipt_target:
            return False
        return real_is_file(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", changed)
    with pytest.raises(ValueError, match="failed-review receipt"):
        prep.build_registration()
