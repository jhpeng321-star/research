"""Pre-registered S-WMFIX-PILOT decision rule (MAIN 10:40, option b)."""
import pytest

from tools.wmfixmerge.pilot import EVAL_STEPS, HORIZONS, RERUNS, decide


def runs(mae120, mae300):
    return {120: [mae120] * (1 + RERUNS), 300: [mae300] * (1 + RERUNS)}


def evals(s120, s300):
    return {s: {120: v120, 300: v300} for s, (v120, v300) in zip(EVAL_STEPS, zip(s120, s300))}


def test_pass_when_both_horizons_improve_beyond_two_sd():
    # Identical reruns -> SD 0; any strict improvement passes only when both improve.
    d = decide(runs(283.0, 2185.0), evals((200.0, 190.0, 180.0), (1500.0, 1400.0, 1300.0)))
    assert d["passed"] is True
    assert d["baseline"] == {120: 283.0, 300: 2185.0}
    assert d["improvement"] == {120: 103.0, 300: 885.0}
    assert d["best_of_steps"] == {120: 180.0, 300: 1300.0}


def test_fail_when_one_horizon_worsens():
    # Best-of-steps 120 s is 290 (worse than baseline 283); 300 s improves.
    d = decide(runs(283.0, 2185.0), evals((290.0, 295.0, 291.0), (1500.0, 1400.0, 1300.0)))
    assert d["passed"] is False
    assert d["improvement"][120] < 0


def test_noise_requires_two_sd_margin_strict():
    noisy = {120: [283.0, 293.0, 273.0, 288.0], 300: [2185.0, 2185.0, 2185.0, 2185.0]}
    mean120 = sum(noisy[120]) / 4  # 284.25; sample SD 8.5396 -> 2 SD 17.079
    target = {10: {120: mean120 - 9.9, 300: 1300.0},
              20: {120: mean120 - 30.0, 300: 1400.0},
              30: {120: mean120 - 5.0, 300: 1500.0}}
    d = decide(noisy, target)
    assert d["improvement"][120] == pytest.approx(30.0)  # best of steps
    assert d["passed"] is True  # 30.0 > 17.079
    edge = {s: {120: mean120 - 2 * d["sample_sd"][120], 300: v[300]}
            for s, v in target.items()}
    assert decide(noisy, edge)["passed"] is False  # exactly 2 SD: strict > required


def test_nonfinite_or_unfrozen_fail_closed():
    good = decide(runs(283.0, 2185.0), evals((180.0, 190.0, 200.0), (1300.0, 1400.0, 1500.0)))
    assert good["passed"] is True
    assert decide(runs(283.0, 2185.0), evals((180.0, 190.0, 200.0), (1300.0, 1400.0, 1500.0)),
                  finite_gradients=False)["passed"] is False
    assert decide(runs(283.0, 2185.0), evals((180.0, 190.0, 200.0), (1300.0, 1400.0, 1500.0)),
                  frozen_bitwise=False)["passed"] is False


def test_protocol_constants_are_registered():
    assert (PILOT_STEPS := 30) == 30
    assert EVAL_STEPS == (10, 20, 30)
    assert RERUNS == 3
    assert HORIZONS == (120, 300)
