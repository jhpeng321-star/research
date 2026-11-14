"""E2 decision-quality arm-set tests (synthetic rows only)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evalh.protocol import N_QUANTILES, OutcomeRow
from evalh.scorer import arm_contrasts as AC
from evalh.scorer import decision_quality as DQ
from evalh.scorer.cli import (_dq_allowed_arms, _dq_dedup_key, _dq_normalizer,
                              _dq_pool_arms)


def _q(value):
    return np.full(N_QUANTILES, float(value))


def _row(world, arm, *, tau=60.0, t0=0.0, dec=0.0, delay=100.0):
    return dict(world=world, source=arm, arm_kind=arm, t0_s=t0,
                decision_t_s=dec, tau_s=tau, y_delay_s=delay,
                hard_brake_s=0.0)


def test_e2_allowed_arms_registry():
    assert set(AC.E2_ALLOWED_ARMS) == {
        "native", "pi", "fixed1", "fixed2", "switching1", "switching2",
        "switching3", "w-lo-early", "w-lo-late", "w-mid-early", "w-mid-late",
        "arm0", "arm1", "arm2", "arm3"}
    # default DQ registry is unchanged
    assert DQ.ALLOWED_ARMS == ("native", "damping", "pi", "wall")


def test_decision_points_with_e2_arms():
    rows = [_row("w1", "native", delay=100.0),
            _row("w1", "fixed1", delay=130.0),
            _row("w1", "w-lo-early", delay=90.0),
            _row("w1", "arm2", delay=110.0)]
    points, stats = DQ.decision_points_from_rows(
        rows, taus=(60.0,), allowed_arms=AC.E2_ALLOWED_ARMS)
    assert len(points) == 1
    assert {a.arm for a in points[0].arms} == {
        "native", "fixed1", "w-lo-early", "arm2"}
    assert stats.n_arms_unknown == 0
    assert points[0].arm("arm2") is not None


def test_default_registry_drops_e2_arms():
    rows = [_row("w1", "native", delay=100.0),
            _row("w1", "fixed1", delay=130.0),
            _row("w1", "arm2", delay=110.0)]
    points, stats = DQ.decision_points_from_rows(rows, taus=(60.0,))
    assert points == []
    assert stats.n_arms_unknown == 2


def test_dq_allowed_arms_flag_default_and_e2():
    assert _dq_allowed_arms(SimpleNamespace()) == DQ.ALLOWED_ARMS
    assert _dq_allowed_arms(SimpleNamespace(dq_arms="registered")) == DQ.ALLOWED_ARMS
    assert _dq_allowed_arms(SimpleNamespace(dq_arms="e2")) == \
        AC.E2_DQ_CANDIDATE_ARMS
    # every trajectory member is its own candidate; 'trajectory' is not one
    allowed = _dq_allowed_arms(SimpleNamespace(dq_arms="e2"))
    for arm in ("arm0", "arm1", "arm2", "arm3"):
        assert arm in allowed
    assert "trajectory" not in allowed
    assert _dq_normalizer(SimpleNamespace(dq_arms="e2")) is DQ._e2_candidate_arm
    assert _dq_pool_arms(SimpleNamespace(dq_arms="e2")) == tuple(
        sorted(DQ.E2_POOL_ARMS))
    assert _dq_dedup_key(SimpleNamespace(dq_arms="e2")) is DQ.e2_dedup_key
    assert _dq_normalizer(SimpleNamespace()) is None


def test_e2_candidate_arm_keeps_trajectory_members_distinct():
    assert DQ._e2_candidate_arm({"source": "arm0"}) == "arm0"
    assert DQ._e2_candidate_arm({"source": "arm2"}) == "arm2"
    assert DQ._e2_candidate_arm({"source": "fixed1"}) == "fixed_mix"
    assert DQ._e2_candidate_arm({"arm_kind": "switching"}) == "switching_mix"
    assert DQ._e2_candidate_arm({"policy": "wall"}) == "wall"
    assert DQ._e2_candidate_arm({"policy": "native"}) == "native"
    # a bare pooled token never becomes a candidate
    assert DQ._e2_candidate_arm({"source": "trajectory"}) == "trajectory"
    # historical type helper still maps members to their family
    assert DQ._row_arm_type({"source": "arm0"}) == "trajectory"


def _outcome(world, arm, policy, delay, *, tau=60.0, run_key=None):
    return OutcomeRow(world=world, t0_s=0.0, decision_t_s=0.0, tau_s=tau,
                      policy=policy, y_delay_s=delay,
                      features={"arm": arm, "pack_item_position": 0,
                                "run_key": run_key})


def _e2_points(rows, q=None, *, taus=(60.0,), allowed=None):
    return DQ.decision_points_from_rows(
        rows, q, taus=taus,
        allowed_arms=allowed or AC.E2_DQ_CANDIDATE_ARMS,
        normalizer=DQ._e2_candidate_arm, pool_arms=DQ.E2_POOL_ARMS,
        dedup_key=DQ.e2_dedup_key)


def test_rules_1_and_2_trajectory_members_and_baselines_own_candidates():
    rows = [
        _outcome("w1", "native", "native", 100.0),
        _outcome("w1", "pi", "pi", 130.0),
        _outcome("w1", "arm0", "trajectory", 105.0),
        _outcome("w1", "arm1", "trajectory", 95.0),
        _outcome("w1", "arm2", "trajectory", 120.0),
        _outcome("w1", "arm3", "trajectory", 80.0),
    ]
    points, stats = _e2_points(rows)
    assert len(points) == 1 and stats.n_skipped_identity == 0
    arms = {a.arm: a.realized for a in points[0].arms}
    assert arms == {"native": 100.0, "pi": 130.0, "arm0": 105.0,
                    "arm1": 95.0, "arm2": 120.0, "arm3": 80.0}
    assert "trajectory" not in arms


def test_rule_3_random_mix_families_pooled_mean_realized_and_predicted():
    rows = [
        _outcome("w1", "native", "native", 100.0),
        _outcome("w1", "fixed1", "fixed", 120.0),
        _outcome("w1", "fixed2", "fixed", 80.0),
        _outcome("w1", "switching1", "switching", 110.0),
        _outcome("w1", "switching2", "switching", 90.0),
        _outcome("w1", "switching3", "switching", 70.0),
        _outcome("w1", "w-lo-early", "wall", 60.0),
        _outcome("w1", "w-mid-late", "wall", 100.0),
    ]
    q = np.stack([_q(0.0), _q(2.0), _q(4.0), _q(1.0), _q(3.0), _q(5.0),
                  _q(8.0), _q(0.0)])
    points, _ = _e2_points(rows, q)
    arms = {a.arm: a for a in points[0].arms}
    assert arms["fixed_mix"].realized == pytest.approx(100.0)
    assert arms["switching_mix"].realized == pytest.approx(90.0)
    assert arms["wall"].realized == pytest.approx(80.0)
    assert arms["fixed_mix"].predicted == pytest.approx(np.full(N_QUANTILES, 3.0))
    assert arms["switching_mix"].predicted == pytest.approx(
        np.full(N_QUANTILES, 3.0))
    assert arms["wall"].predicted == pytest.approx(np.full(N_QUANTILES, 4.0))


def test_rule_4_duplicate_native_pi_dedup_prefers_truearms_and_records():
    rows = [
        _outcome("w1", "native", "native", 100.0,
                 run_key="chunk2/w1__native"),
        _outcome("w1", "native", "native", 111.0,
                 run_key="truearms_chunk2/w1__native"),   # preferred
        _outcome("w1", "pi", "pi", 90.0, run_key="chunk2/w1__pi"),
        _outcome("w1", "pi", "pi", 99.0,
                 run_key="truearms_chunk3/w1__pi"),       # preferred
        _outcome("w1", "arm0", "trajectory", 80.0),
    ]
    points, stats = _e2_points(rows)
    assert len(points) == 1 and stats.n_skipped_identity == 0
    arms = {a.arm: a.realized for a in points[0].arms}
    assert arms["native"] == pytest.approx(111.0)   # truearms, not mean 105.5
    assert arms["pi"] == pytest.approx(99.0)         # truearms, not mean 94.5
    assert stats.n_dedup_truearms == 2 and stats.n_dedup_released == 0
    assert stats.dedup_choices == {"native": 1, "pi": 1}


def test_rule_4_duplicate_without_dedup_key_is_identity_conflict():
    rows = [
        _outcome("w1", "native", "native", 100.0),
        _outcome("w1", "native", "native", 110.0),
        _outcome("w1", "arm0", "trajectory", 80.0),
    ]
    # same e2 candidate/pool config but with no dedup_key: native duplicates
    # are an identity conflict, not averaged and not silently kept.
    points, stats = DQ.decision_points_from_rows(
        rows, taus=(60.0,), allowed_arms=AC.E2_DQ_CANDIDATE_ARMS,
        normalizer=DQ._e2_candidate_arm, pool_arms=DQ.E2_POOL_ARMS,
        dedup_key=None)
    assert points == [] and stats.n_skipped_identity == 1


def test_rule_5_fixed_best_type_level_trajectory_is_mean_over_members():
    rows = [
        _outcome("w1", "arm0", "trajectory", 10.0),
        _outcome("w1", "arm1", "trajectory", 20.0),
        _outcome("w1", "arm2", "trajectory", 30.0),
        _outcome("w1", "arm3", "trajectory", 40.0),
    ]
    q = np.stack([_q(10.0), _q(20.0), _q(30.0), _q(40.0)])
    points, _ = _e2_points(rows, q)
    point = points[0]
    # candidates are the four separate arms
    assert {a.arm for a in point.arms} == {"arm0", "arm1", "arm2", "arm3"}
    ranking = ["trajectory", "native"]
    assert DQ.fixed_best_arm(point, ranking) == "trajectory"
    # fixed-best trajectory realized = mean over the decision's trajectory arms
    assert DQ.fixed_best_realized(point, ranking) == pytest.approx(25.0)
    # oracle = best SINGLE candidate; random = uniform over candidates present
    assert DQ.oracle_arm(point) == "arm0"
    doc = DQ.evaluate(points, fixed_best=dict(
        schema=DQ.SCHEMA_FIXED_BEST,
        cells={"delay_s": {"60": {"ranking": ranking}}}), n_boot=20, seed=1)
    cell = doc["cells"]["delay_s"]["60"]
    assert cell["fixed_best_regret"] == pytest.approx(25.0 - 10.0)
    assert cell["random_expected_regret"] == pytest.approx(25.0 - 10.0)


def test_two_decision_points_many_arms():
    rows = []
    for w in ("w1", "w2"):
        for i, arm in enumerate(AC.E2_ALLOWED_ARMS):
            rows.append(_row(w, arm, tau=60.0, t0=0.0, delay=100.0 + i))
    points, stats = DQ.decision_points_from_rows(
        rows, taus=(60.0,), allowed_arms=AC.E2_ALLOWED_ARMS)
    assert len(points) == 2
    assert all(len(p.arms) == len(AC.E2_ALLOWED_ARMS) for p in points)
    assert stats.n_decision_points == 2
