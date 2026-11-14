"""The J-PULSE hook: retrain config + the registered OOF diagnostic
(registration s.13) on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from partB_v04b import pulse_hook as PH
from partB_v04b import reg


def test_pulse_arm_labels():
    assert PH._pulse_arm_label("__pulse_damping60") == ("damping", 60.0)
    assert PH._pulse_arm_label("__pulse_pi240") == ("pi", 240.0)
    with pytest.raises(ValueError):
        PH._pulse_arm_label("__native")
    with pytest.raises(ValueError):
        PH._pulse_arm_label("__pulse_lidar90")


def test_retrain_config_is_a_disclosed_new_version():
    cfg = PH.pulse_retrain_config(dict(first_fit="WD-01-PB-V04B"))
    assert cfg["schema"] == "wd01_partB_v04b_pulse_retrain/1"
    assert cfg["disclosed_new_version"] is True
    assert cfg["silent_refit"] is False
    assert cfg["tuning_grid"] == [list(c) for c in reg.GRID]
    assert "never a selection input" in cfg["chunk5_usage"]
    assert cfg["base_registration_lf_sha256"] == reg.reg_sha()


def _pulse_pairs(n_worlds=8, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for wi in range(n_worlds):
        w = f"wp{wi:02d}"
        for arm in ("__pulse_damping60", "__pulse_damping120",
                    "__pulse_pi120", "__pulse_pi240"):
            for tau in (120.0, 300.0, 570.0):
                rows.append(dict(contrast="pulse-native", world=w,
                                 key=f"chunk4/{w}", arm_source=arm,
                                 tau=tau, demand=2700, penetration=0.1))
    df = pd.DataFrame(rows)
    for tau in (120.0, 300.0, 570.0):
        col = f"dy_{reg.GATING_HEAD}_{int(tau)}"
        df[col] = [(1.0 if "damping" in a else -1.0) * tau / 10.0
                   + rng.normal(0, 1.0) for a in df["arm_source"]]
    return df


def test_oof_diagnostic_stratifies_and_reports_only(_none=None):
    pairs = _pulse_pairs()
    calls = []

    def factory(train_rows):
        calls.append(len(train_rows))

        def pred(g):                 # damping rows: dY>0; pi rows: dY<0
            sign = np.where(g["arm_source"].str.contains("damping"),
                            1.0, -1.0)
            return {q: sign * 100.0 + (q - 0.5) for q in reg.QUANTILES}
        return pred
    recs = PH.pulse_oof_diagnostic(pairs, factory, n_folds=4)
    assert calls and all(c < len(pairs) for c in calls)
    by = {(r["controller"], r["length_s"]) for r in recs}
    assert by == {("damping", 60.0), ("damping", 120.0),
                  ("pi", 120.0), ("pi", 240.0)}
    assert all(r["diagnostic_only"] for r in recs)
    assert {r["tau"] for r in recs} == set(PH.PULSE_TAUS)
    # truth table: the sign predictor is right on every eligible row
    accs = [r["sign_acc"] for r in recs if r["n_eligible"]]
    assert accs and all(a == 1.0 for a in accs)


def test_oof_diagnostic_needs_enough_worlds():
    with pytest.raises(ValueError, match="worlds"):
        PH.pulse_oof_diagnostic(_pulse_pairs(n_worlds=2),
                                lambda tr: (lambda g: {}))
