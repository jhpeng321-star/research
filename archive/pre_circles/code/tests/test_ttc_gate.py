"""Tests for the benefit gate (ttc_gate, 2026-08-06).

Seed-variance finding: the policy helps only where the baseline is chaotic
(corr(no-op TTC, d_ttc) = -0.55); on calm corridors its own metering injects
waves. The gate remaps fairness/recover/vacate to coast while the corridor-TTC
EMA is below ``ttc_gate_lo``; protect (safety) and coast are exempt.
"""

from tests.test_shepherd_actions import _make_engine


def _engine(**overrides):
    mc = {
        "ttc_gate": True,
        "ttc_gate_lo": 1.0,
        "ttc_gate_modes": ("fair_balance", "recover", "vacate_drop"),
    }
    mc.update(overrides)
    return _make_engine(mode_config=mc)


def test_off_no_gate():
    engine = _engine(ttc_gate=False)
    engine._corridor_ttc_ema = 0.0
    assert engine._ttc_gate_blocks("fair_balance") is False


def test_calm_blocks_fair_balance():
    engine = _engine()
    engine._corridor_ttc_ema = 0.2  # < lo 1.0
    assert engine._ttc_gate_blocks("fair_balance") is True
    assert engine._ttc_gate_blocks("recover") is True
    assert engine._ttc_gate_blocks("vacate_drop") is True


def test_disturbed_allows():
    engine = _engine()
    engine._corridor_ttc_ema = 5.0  # >= lo
    assert engine._ttc_gate_blocks("fair_balance") is False


def test_protect_exempt_even_when_calm():
    engine = _engine()
    engine._corridor_ttc_ema = 0.0
    assert engine._ttc_gate_blocks("guard") is False  # safety never gated


def test_coast_always_legal():
    engine = _engine()
    engine._corridor_ttc_ema = 0.0
    assert engine._ttc_gate_blocks("coast") is False


def test_threshold_boundary():
    engine = _engine(ttc_gate_lo=3.0)
    engine._corridor_ttc_ema = 3.0
    assert engine._ttc_gate_blocks("fair_balance") is False  # >= lo
    engine._corridor_ttc_ema = 2.99
    assert engine._ttc_gate_blocks("fair_balance") is True  # < lo
