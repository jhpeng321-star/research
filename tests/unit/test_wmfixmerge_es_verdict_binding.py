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


def test_missing_registered_inputs_block_before_any_a_rollout(tmp_path):
    from tools.wmfixmerge.data import digest
    from tools.wmfixmerge.verdict import run
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    producer = dict(script=str(artifact), files={str(artifact): digest(artifact)})
    plan = dict(schema="wmfix_verdict_plan/1", accept_roots=[str(tmp_path)],
                accept_panel=str(tmp_path / "missing_accept.json"),
                stages=dict(A={}, SAFETY={},
                    B=dict(adapter="bound_external", producer=dict(producer,
                        args=["--candidate-dir", str(tmp_path / "other_candidate")])),
                    C1=dict(adapter="bound_external", producer=dict(producer,
                        args=["--cohort", str(tmp_path / "missing_cohort.json"),
                              "--runner", str(tmp_path / "missing_runner.json")]))))
    for key in ("reference", "base_config", "controllers", "corrected_report", "harness_contract",
                "validation_real", "zero_reference"):
        plan[key] = str(artifact)
    result = run(plan, artifact, artifact, tmp_path, execute=True)
    assert result["status"] == "BLOCKED" and result["executed_stages"] == []
    assert set(result["blockers"]) == {"missing accept_panel", "missing B--candidate-dir",
                                       "missing C1--cohort", "missing C1--runner"}
    assert not (tmp_path / "RUN.json").exists()
