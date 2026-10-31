"""Regression: a complete other-candidate quartet must never score ES-WM."""
import copy
from pathlib import Path

import pytest

from tools.wmfixmerge.es_verdict_plan import isolated_plan
from tools.wmfixmerge.producer_b import verify_candidate_binding


@pytest.mark.parametrize("field", ["candidate_sha256", "candidate_config_sha256", "base_wm_sha256"])
def test_other_checkpoint_config_or_reference_is_rejected(field):
    identity = dict(candidate_sha256="es", candidate_config_sha256="config", base_wm_sha256="wm2")
    binding = dict(candidate_sha256="es", config_sha256="config", reference_sha256="wm2")
    verify_candidate_binding(identity, binding)
    identity[field] = "other"
    with pytest.raises(ValueError, match=field):
        verify_candidate_binding(identity, binding)


def test_isolation_preserves_every_other_verdict_binding():
    import json
    tree = Path(__file__).resolve().parents[2]
    plan = json.loads((tree / "tools/wmfixmerge/verdict_plan.json").read_text())
    original = copy.deepcopy(plan)
    result = isolated_plan(plan, Path("/private/tree"), Path("/private/run"), lambda _: "newhash")
    assert plan == original
    assert result["stages"]["A"] == plan["stages"]["A"]
    assert result["stages"]["C1"] == plan["stages"]["C1"]
    assert result["stages"]["SAFETY"] == plan["stages"]["SAFETY"]
    producer = result["stages"]["B"]["producer"]
    assert producer["args"][producer["args"].index("--candidate-dir") + 1] == str(Path("/private/run/quartet_candidate"))
    for name in ("producer_b.py", "quartet_candidate.py"):
        assert producer["files"][str(Path("/private/tree/tools/wmfixmerge") / name)] == "newhash"
    for key in plan.keys() - {"stages"}:
        assert result[key] == plan[key]
    old = plan["stages"]["B"]["producer"]
    for path, sha in old["files"].items():
        if Path(path).name not in {"producer_b.py", "quartet_candidate.py"}:
            assert producer["files"][path] == sha
