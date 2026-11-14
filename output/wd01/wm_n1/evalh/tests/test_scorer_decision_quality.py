"""Decision-quality (regret) tests: synthetic fixtures only.

No CAL / HOLDOUT / chunk-5 / J4B / INC1 data is opened here; every row,
prediction and fixed-best artifact is generated under ``tmp_path``.  The tests
pin the four required identities (oracle-perfect -> zero regret, constant
predictor -> fixed-best, random baseline exact) plus the fail-closed counts,
the SHA pin and the default-off CLI path.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from evalh.protocol import N_QUANTILES, OutcomeRow
from evalh.scorer import decision_quality as DQ
from evalh.scorer.cli import ScorerCliRefusal, run_score

TAU = 60.0


def _q(value):
    return np.full(N_QUANTILES, float(value))


def _row(world, arm, y, *, tau=TAU, t0=0.0, dec=0.0, hb=0.0):
    return OutcomeRow(world=world, t0_s=t0, decision_t_s=dec, tau_s=tau,
                      policy=arm, y_delay_s=y, hard_brake_s=hb)


# ------------------------------------------------------------- core tests --
def test_oracle_perfect_predictor_zero_regret():
    rows = [_row("w1", "native", 5.0), _row("w1", "pi", 2.0)]
    q = np.stack([_q(r.y_delay_s) for r in rows])
    points, stats = DQ.decision_points_from_rows(rows, q, cost="delay_s")
    assert stats.n_decision_points == 1 and stats.n_rows_used == 2
    doc = DQ.evaluate(points, fixed_best=DQ.build_fixed_best(points),
                      n_boot=50, seed=1)
    cell = doc["cells"]["delay_s"]["60"]
    assert cell["wm_mean_regret"] == pytest.approx(0.0)
    assert cell["wm_risk_regret"] == pytest.approx(0.0)
    assert cell["wm_picks_oracle_fraction"] == pytest.approx(1.0)
    assert doc["primary_claim"]["per_tau"]["60"]["holds"] is False  # tie


def test_constant_predictor_equals_fixed_best():
    specs = [("w1", 10.0, 4.0), ("w2", 12.0, 5.0)]
    mean = {"native": float(np.mean([s[1] for s in specs])),
            "pi": float(np.mean([s[2] for s in specs]))}
    rows, q = [], []
    for world, y_native, y_pi in specs:
        for arm, y in (("native", y_native), ("pi", y_pi)):
            rows.append(_row(world, arm, y))
            q.append(_q(mean[arm]))
    points, _ = DQ.decision_points_from_rows(rows, np.stack(q), cost="delay_s")
    fb = DQ.build_fixed_best(points)
    assert fb["cells"]["delay_s"]["60"]["best_arm"] == "pi"
    doc = DQ.evaluate(points, fixed_best=fb, n_boot=50, seed=1)
    cell = doc["cells"]["delay_s"]["60"]
    assert cell["wm_mean_regret"] == pytest.approx(cell["fixed_best_regret"])
    assert cell["wm_minus_fixed_best_mean"] == pytest.approx(0.0)
    assert cell["wm_picks_oracle_fraction"] == pytest.approx(1.0)


def test_random_baseline_exact():
    # realized native 6, pi 2, damping 4 -> mean arm cost 4, oracle 2,
    # so the exact uniform-choice expected regret is 2.
    rows = [_row("w1", "native", 6.0), _row("w1", "pi", 2.0),
            _row("w1", "damping", 4.0)]
    q = np.stack([_q(0.0), _q(0.0), _q(0.0)])
    points, _ = DQ.decision_points_from_rows(rows, q, cost="delay_s")
    doc = DQ.evaluate(points, fixed_best=DQ.build_fixed_best(points),
                      n_boot=50, seed=1)
    cell = doc["cells"]["delay_s"]["60"]
    assert cell["random_expected_regret"] == pytest.approx(2.0)
    assert cell["fixed_best_regret"] == pytest.approx(0.0)


def test_fail_closed_duplicate_arm_identity():
    rows = [_row("w1", "native", 5.0), _row("w1", "pi", 2.0),
            _row("w1", "pi", 3.0)]
    q = np.stack([_q(0.0)] * 3)
    points, stats = DQ.decision_points_from_rows(rows, q, cost="delay_s")
    assert points == []
    assert stats.n_skipped_identity == 1
    assert stats.n_decision_points == 0


def test_fail_closed_insufficient_and_unknown_and_nonfinite():
    rows = [_row("w1", "native", 5.0), _row("w1", "trajectory", 1.0)]
    q = np.stack([_q(0.0), _q(0.0)])
    points, stats = DQ.decision_points_from_rows(rows, q, cost="delay_s")
    assert stats.n_arms_unknown == 1
    assert stats.n_skipped_insufficient_arms == 1 and points == []

    bad = [_row("w1", "native", 5.0), _row("w1", "pi", float("nan"))]
    points, stats = DQ.decision_points_from_rows(
        bad, np.stack([_q(0.0), _q(0.0)]), cost="delay_s")
    assert stats.n_arms_nonfinite == 1
    assert stats.n_skipped_insufficient_arms == 1 and points == []


def test_tau_filter_counts():
    rows = [_row("w1", "native", 5.0, tau=15.0),
            _row("w1", "pi", 2.0, tau=15.0)]
    _, stats = DQ.decision_points_from_rows(rows, cost="delay_s")
    assert stats.n_skipped_tau == 2 and stats.n_decision_points == 0


def test_cluster_key_is_physical_world():
    rows = [_row("w1t0", "native", 5.0, t0=0.0),
            _row("w1t0", "pi", 2.0, t0=0.0),
            _row("w1t60", "native", 7.0, t0=60.0),
            _row("w1t60", "pi", 1.0, t0=60.0)]
    q = np.stack([_q(0.0)] * 4)
    points, _ = DQ.decision_points_from_rows(rows, q, cost="delay_s")
    doc = DQ.evaluate(points, fixed_best=DQ.build_fixed_best(points),
                      n_boot=50, seed=1)
    assert doc["cells"]["delay_s"]["60"]["n_points"] == 2
    assert doc["cells"]["delay_s"]["60"]["n_clusters"] == 1


def test_fixed_best_artifact_sha_pin(tmp_path):
    rows = [_row("w1", "native", 5.0), _row("w1", "pi", 2.0)]
    points, _ = DQ.decision_points_from_rows(rows, None, cost="delay_s")
    doc = DQ.build_fixed_best(points, train_source={"path": "synthetic"})
    path = tmp_path / "fb.json"
    sha = DQ.write_fixed_best(doc, path)
    loaded, got = DQ.load_fixed_best(path, expected_sha256=sha)
    assert got == sha and loaded == doc
    with pytest.raises(DQ.DecisionQualityRefusal, match="sha256"):
        DQ.load_fixed_best(path, expected_sha256="0" * 64)
    with pytest.raises(DQ.DecisionQualityRefusal, match="already exists"):
        DQ.write_fixed_best(doc, path)


def test_evaluate_refuses_missing_fixed_best_cell():
    rows = [_row("w1", "native", 5.0), _row("w1", "pi", 2.0)]
    q = np.stack([_q(0.0)] * 2)
    points, _ = DQ.decision_points_from_rows(rows, q, cost="delay_s")
    empty = dict(schema=DQ.SCHEMA_FIXED_BEST, cells={})
    with pytest.raises(DQ.DecisionQualityRefusal, match="no cell"):
        DQ.evaluate(points, fixed_best=empty, n_boot=10)


def test_wm_refuses_labels_only_point():
    rows = [_row("w1", "native", 5.0), _row("w1", "pi", 2.0)]
    points, _ = DQ.decision_points_from_rows(rows, None, cost="delay_s")
    with pytest.raises(DQ.DecisionQualityRefusal, match="no predicted"):
        DQ.wm_arm(points[0])


def test_render_table_has_expected_header():
    rows = [_row("w1", "native", 5.0), _row("w1", "pi", 2.0)]
    q = np.stack([_q(0.0)] * 2)
    points, _ = DQ.decision_points_from_rows(rows, q, cost="delay_s")
    doc = DQ.evaluate(points, fixed_best=DQ.build_fixed_best(points),
                      n_boot=10)
    table = DQ.render_table(doc)
    assert table.startswith("| cost | tau |")
    assert "| delay_s | 60 |" in table


def test_canonical_arm_handles_kind_and_policy():
    assert DQ.canonical_arm("damping_fleet") == "damping"
    assert DQ.canonical_arm("switching_mix") == "switching"
    assert DQ.canonical_arm("pi-external-u") == "pi"


def _dict_row(world, source, arm_kind, y, *, tau=TAU):
    return dict(world=world, source=source, arm_kind=arm_kind, policy=arm_kind,
                t0_s=0.0, decision_t_s=0.0, tau_s=tau, y_delay_s=y,
                hard_brake_s=0.0)


def test_future_arms_opt_in_and_source_exclusion():
    rows = [
        _dict_row("w1", "swpi", "switching", 3.0),
        _dict_row("w1", "native", "native_fleet", 5.0),
        _dict_row("w1", "base", "damping_fleet", 1.0),
    ]
    allowed = DQ.ALLOWED_ARMS + DQ.FUTURE_ARMS
    points, stats = DQ.decision_points_from_rows(rows, allowed_arms=allowed)
    assert stats.n_rows_excluded_arm == 1                    # base rule-(b)
    assert len(points) == 1
    assert {a.arm for a in points[0].arms} == {"native", "switching"}


def test_duplicate_after_source_exclusion_is_identity_conflict():
    rows = [
        _dict_row("w1", "native", "native_fleet", 5.0),
        _dict_row("w1", "other", "native_fleet", 6.0),
    ]
    points, stats = DQ.decision_points_from_rows(rows)
    assert points == [] and stats.n_skipped_identity == 1



# --------------------------------------------------------------- CLI tests --
def _dq_rows():
    rows = []
    for tau in (60.0, 120.0):
        rows.append(_row("w1", "native", 100.0, tau=tau))
        rows.append(_row("w1", "pi", 130.0, tau=tau))
    return rows


class _FakeA:
    name = "A:teacher"
    summary_source = "training_packed_plan"

    def predict_quantiles(self, rows):
        return np.stack([_q(r.y_delay_s) for r in rows])


class _FakeSource:
    def __init__(self, spec):
        self.spec = spec
        self.half = "cal_tune"
        self.selected = [object()]


class _FakeBound:
    def __init__(self, sha):
        self.bundle = object()
        self.sha256 = sha


def _member(tmp_path):
    p = tmp_path / "member_1.pt"
    p.write_bytes(b"decision-quality-test-member")
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def _fixed_best(tmp_path):
    rows = _dq_rows()
    pts = []
    for cost in DQ.COSTS:
        cost_points, _ = DQ.decision_points_from_rows(rows, None, cost=cost)
        pts.extend(cost_points)
    doc = DQ.build_fixed_best(pts, train_source={"path": "synthetic"})
    path = tmp_path / "fb.json"
    return path, DQ.write_fixed_best(doc, path)


def _args(tmp_path, member, out, **extra):
    base = dict(
        command="score", release_spec=str(tmp_path / "release.json"),
        release_sha="a" * 64, checkpoint=str(member), out=str(out),
        cav_motion="teacher", comparators="zero")
    base.update(extra)
    return SimpleNamespace(**base)


def _run(tmp_path, member, sha, out, args):
    return run_score(
        args,
        source_factory=lambda spec_path, expected_release_sha256: _FakeSource(
            {"checkpoint_member_sha256": sha}),
        checkpoint_loader=lambda path, **kw: _FakeBound(sha),
        rows_loader=lambda source: (_dq_rows(), {"n_items": 1}),
        a_factory=lambda bundle: _FakeA(),
        git_head="b" * 40)


def test_cli_decision_quality_flag_end_to_end(tmp_path):
    member, sha = _member(tmp_path)
    fb_path, fb_sha = _fixed_best(tmp_path)
    out = tmp_path / "out"
    args = _args(tmp_path, member, out, decision_quality=True,
                 dq_fixed_best=str(fb_path), dq_fixed_best_sha256=fb_sha,
                 dq_n_boot=50)
    _run(tmp_path, member, sha, out, args)
    assert (out / "DECISION_QUALITY.json").is_file()
    results = json.loads((out / "RESULTS.json").read_text("utf-8"))
    assert results["decision_quality"]["schema"] == DQ.SCHEMA
    assert results["decision_quality_table"].startswith("| cost | tau |")
    assert results["inputs"]["decision_quality"]["fixed_best_sha256"] == fb_sha
    dq = json.loads((out / "DECISION_QUALITY.json").read_text("utf-8"))
    assert dq["cells"]["delay_s"]["60"]["n_points"] == 1


def test_cli_decision_quality_default_off(tmp_path):
    member, sha = _member(tmp_path)
    out = tmp_path / "out"
    _run(tmp_path, member, sha, out, _args(tmp_path, member, out))
    assert not (out / "DECISION_QUALITY.json").exists()
    results = json.loads((out / "RESULTS.json").read_text("utf-8"))
    assert "decision_quality" not in results


def test_cli_decision_quality_requires_fixed_best(tmp_path):
    member, sha = _member(tmp_path)
    out = tmp_path / "out"
    args = _args(tmp_path, member, out, decision_quality=True)
    with pytest.raises(ScorerCliRefusal, match="dq-fixed-best"):
        _run(tmp_path, member, sha, out, args)
