import json
import sys
from pathlib import Path

import pytest

WM_N1 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WM_N1))
from lpwo import bounded_cpu as C, guard as G, prepare_pilot_bindings as P  # noqa: E402


def test_cpu_wrapper_limits_and_refuses_existing_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "WORKSPACE", tmp_path)
    monkeypatch.chdir(tmp_path)
    for arguments in (["--limit-bytes", "6000000000"], []):
        receipt = tmp_path / "receipt.json"
        receipt.write_text("{}")
        with pytest.raises(SystemExit):
            C.main(["--receipt", str(receipt), *arguments, "--", "-m", "pytest", "--version"])
        assert receipt.read_text() == "{}"


def test_explicit_bindings_use_only_train_pilot_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "WORKSPACE", tmp_path)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": "holdout_exclusions/1", "excluded": []}))
    tool = tmp_path / "tool.py"
    tool.write_text("def assert_final_holdout_eligible(worlds, data):\n    assert data is not None\n")
    index = tmp_path / "INDEX.json"
    base = {"key": "chunk4/synthetic__wall", "world": "synthetic", "policy": "wall",
            "split": "TRAIN", "root_rel": "runs/campaign_walls_pilot_chunk4"}
    index.write_text(json.dumps({"runs": [base, {**base, "split": "HOLDOUT"},
        {**base, "policy": "native"}, {**base, "root_rel": "runs/campaign_walls_pilot_chunk5"}]}))
    output = tmp_path / "bindings.json"
    assert P.main(["--index", str(index), "--out", str(output),
                   "--campaign-root", str(tmp_path / "campaign_walls_pilot_chunk4"),
                   "--guard-spec", str(spec), "--guard-tool", str(tool),
                   "--allowed-root", str(tmp_path)]) == 0
    assert list(json.loads(output.read_text())) == [base["key"]]
