import json
from pathlib import Path

import pytest

from tools.wmfixmerge.data import digest
from tools.wmfixmerge.verdict import bound_panel, preflight, run
from tools.wmfixmerge.verdict_metrics import c1, metric, speedhead, validation


def test_default_plan_explicitly_blocks_unbound_adapters_and_missing_data(tmp_path):
    plan = json.loads(Path("tools/wmfixmerge/verdict_plan.json").read_text())
    result = run(plan, "missing.pt", "missing.json", tmp_path, execute=True)
    assert result["status"] == "BLOCKED" and result["executed_stages"] == []
    assert any("B exact frozen producer" in x for x in result["blockers"])
    assert any("C1 exact frozen producer" in x for x in result["blockers"])
    assert not (tmp_path / "RUN.json").exists()


def test_persistent_run_receipt_never_duplicates_dispatch(tmp_path, monkeypatch):
    import tools.wmfixmerge.verdict as driver
    monkeypatch.setattr(driver, "preflight", lambda *args: [])
    (tmp_path / "RUN.json").write_text('{"pid":123}')
    result = run({}, "missing", "missing", tmp_path, execute=True)
    assert result["status"] == "BLOCKED" and not result["executed_stages"]


def test_missing_stage_and_zero_confident_gains_cannot_pass():
    assert preflight({}, "x", "y")
    assert metric([dict(real=1., base_draws=[1.] * 8)], "base")["confident_gain_accuracy"] is None
    zero = metric([dict(real=0., base_draws=[-1.] * 8)], "base")
    assert zero["confident_gain_count"] == 1 and zero["confident_gain_nonzero_count"] == 0
    assert zero["confident_gain_coverage"] == 1 and zero["confident_gain_accuracy"] is None
    with pytest.raises(ValueError):
        validation(dict(schema="wmfix_validation/1", K=8, mix="ON", native_lanes=True, rows=[]))


def speedhead_fixture():
    absolute, rows = [], []
    for world in range(963880, 963888):
        for h in (300, 600):
            for a in ("CTRL0", "MAPPO", "PI", "CIRCLES"):
                delta = 0 if a == "CTRL0" else (world - 963884) * (1 + ["MAPPO", "PI", "CIRCLES"].index(a))
                absolute.append(dict(world=str(world), arm=a, horizon=h, real=100. + delta,
                    base_draws=[100. - delta] * 8, candidate_draws=[100. + delta] * 8,
                    components=dict(real=dict(road=100. + delta, source=0),
                        base=dict(road=[100. - delta] * 8, source=[0.] * 8),
                        candidate=dict(road=[100. + delta] * 8, source=[0.] * 8))))
                if a != "CTRL0":
                    rows.append(dict(world=str(world), arm=a, horizon=h, real=float(delta),
                                     base_draws=[-float(delta)] * 8, candidate_draws=[float(delta)] * 8))
    return dict(schema="wmfix_speedhead/1", K=8, mix="ON", native_lanes=True,
                matched_t300_history_queue=True, closed_loop_all_cavs=True, rows=rows, absolute=absolute)


def test_speedhead_checks_all_world_directions_magnitudes_and_ctrl0_accounting():
    doc = speedhead_fixture()
    assert speedhead(doc)["passed"]
    doc["absolute"][0]["candidate_draws"][0] += 1
    with pytest.raises(ValueError, match="sum to J"):
        speedhead(doc)
    doc = speedhead_fixture()
    doc["rows"].pop()
    with pytest.raises(ValueError, match="missing/duplicate"):
        speedhead(doc)


def test_validation_actual_bootstrap_gate_and_complete_grid():
    rows = [dict(world=str(w), pair_key=str(w) + "@t300", arm=a, horizon=h,
                 real=float((w - 18) * h), base_draws=[float(-(w - 18) * h)] * 8,
                 candidate_draws=[float((w - 18) * h)] * 8)
            for w in range(36) for a in ("ours_c1", "coopoff_c1", "mappo")
            for h in (60, 90, 120, 180, 300)]
    doc = dict(schema="wmfix_validation/1", K=8, mix="ON", native_lanes=True, rows=rows)
    report = validation(doc)
    assert report["passed"] and report["pooled"]["delta_ci95"]["mae"][1] < 0
    assert report["pooled"]["candidate"]["confident_gain_accuracy"] == 1
    assert report["pooled"]["base"]["confident_gain_accuracy"] == 0
    assert report["H_G"] is None
    doc["rows"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        validation(doc)


def test_c1_missing_cohort_and_regression_block():
    doc = dict(schema="wmfix_c1_regression/1", reference_label="C1", canonical_cohort_sha256="a" * 64,
               canonical_worlds={"PI": ["a"], "unseen_wall": ["b"]}, rows=[
                   dict(world="a", cohort="PI", base_error=1., candidate_error=.9, base_blowups=0, candidate_blowups=0),
                   dict(world="b", cohort="unseen_wall", base_error=1., candidate_error=.9, base_blowups=0, candidate_blowups=0)])
    assert c1(doc)["passed"]
    doc["rows"][1]["candidate_blowups"] = 1
    assert not c1(doc)["passed"]
    doc["rows"].pop()
    with pytest.raises(ValueError):
        c1(doc)


def test_external_result_must_bind_models_config_and_producer(tmp_path):
    p = tmp_path / "result.json"
    p.write_text(json.dumps(dict(binding={"candidate": "wrong"}, producer={"script": "frozen.py"})))
    spec = dict(receipt=str(p), sha256=digest(p), producer={"script": "frozen.py"})
    with pytest.raises(ValueError, match="binding mismatch"):
        bound_panel(spec, {}, stage="B", out=tmp_path, candidate="x", config="y", reference="z", device="cpu")


def test_zero_exit_without_result_is_not_a_pass(tmp_path):
    script = tmp_path / "producer.py"
    script.write_text("pass\n")
    spec = dict(producer=dict(script=str(script), files={str(script): digest(script)}, args=["--out", "{out}"]))
    with pytest.raises(FileNotFoundError):
        bound_panel(spec, {}, stage="B", out=tmp_path, candidate="x", config="y", reference="z", device="cpu")


def test_failed_safety_prevents_all_stage_adoption(tmp_path, monkeypatch):
    import tools.wmfixmerge.verdict as driver
    import tools.wmfixmerge.verdict_metrics as metrics
    import tools.wmfixmerge.verdict_panels as panels
    for name in ("candidate.pt", "config.json", "reference.pt", "report.md"):
        (tmp_path / name).write_text("test plumbing only")
    monkeypatch.setattr(driver, "preflight", lambda *args: [])
    monkeypatch.setattr(driver, "bound_panel", lambda *args, **kwargs: {})
    monkeypatch.setattr(panels, "validation", lambda *args: {})
    monkeypatch.setattr(panels, "safety", lambda *args: {"decision": "REJECT"})
    for name in ("validation", "speedhead", "c1"):
        monkeypatch.setattr(metrics, name, lambda *args: {"passed": True})
    plan = dict(reference=str(tmp_path / "reference.pt"), corrected_report=str(tmp_path / "report.md"),
                stages={"B": {}, "C1": {}})
    result = run(plan, tmp_path / "candidate.pt", tmp_path / "config.json", tmp_path, execute=True)
    assert result["status"] == "REJECT" and not result["passed"]
    assert result["panels"]["SAFETY"]["passed"] is False
