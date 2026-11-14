"""E2 arm-contrast tests (synthetic rows only; no CAL/real data)."""
from __future__ import annotations

import numpy as np
import pytest

from evalh.protocol import N_QUANTILES
from evalh.scorer import arm_contrasts as AC


def _rec(world, source, tau, delay, *, t0=0.0, t=0.0, avail=1.0,
         split="TRAIN"):
    return dict(world=world, source=source, t0_s=t0, t=t, tau=tau,
                tau_available=avail, total_delay_ref_s=delay, split=split)


def _labels():
    ref = [_rec(w, "native", tau, 100.0)
           for w in ("w1", "w2") for tau in (60.0, 120.0)]
    treat = []
    for w in ("w1", "w2"):
        for tau in (60.0, 120.0):
            treat.append(_rec(w, "fixed1", tau, 130.0))
            treat.append(_rec(w, "fixed2", tau, 150.0))
            treat.append(_rec(w, "switching1", tau, 170.0))
            treat.append(_rec(w, "w-lo-early", tau, 190.0))
            treat.append(_rec(w, "arm0", tau, 210.0))
    return treat, ref


def test_registry_shape():
    assert AC.GROUP_MEMBERS["fixed_mix"] == ("fixed1", "fixed2")
    assert AC.GROUP_MEMBERS["trajectory"] == ("arm0", "arm1", "arm2", "arm3")
    assert "w-lo-early" in AC.GROUP_MEMBERS["wall"]
    assert AC.group_contrast("fixed_mix") == "fixed_mix-native"
    assert AC.member_contrast("arm2") == "arm2-native"
    assert AC.contrast_members("wall-native") == AC.GROUP_MEMBERS["wall"]
    assert AC.contrast_members("fixed1-native") == ("fixed1",)
    assert "native" in AC.E2_ALLOWED_ARMS and "pi" in AC.E2_ALLOWED_ARMS
    with pytest.raises(AC.ArmContrastRefusal):
        AC.contrast_members("nope-native")


def test_cells_pooled_and_separate():
    treat, ref = _labels()
    cells, items = AC.build_arm_cells(treat, ref, taus=(60.0, 120.0))
    # fixed_mix pools fixed1+fixed2: dV = [30, 50] at each tau across two worlds
    assert set(cells) == {"fixed1-native", "fixed2-native",
                          "fixed_mix-native", "switching1-native",
                          "switching_mix-native", "w-lo-early-native",
                          "wall-native", "arm0-native", "trajectory-native"}
    pooled = cells["fixed_mix-native"]["60"]
    assert pooled["n_pairs"] == 4 and pooled["n_worlds"] == 2
    assert pooled["quantiles"][-1] == pytest.approx(50.0)
    assert cells["fixed1-native"]["60"]["quantiles"][-1] == pytest.approx(30.0)
    assert "fixed_mix-native" in items
    assert {"fixed1", "fixed2", "native"} <= {
        it["source"] for it in items["fixed_mix-native"]}


def test_missing_reference_is_omitted_not_zero():
    treat, ref = _labels()
    cells, _ = AC.build_arm_cells(
        treat + [_rec("w9", "fixed1", 60.0, 999.0)], ref, taus=(60.0,))
    # w9 has no native reference -> not paired; no zero substitution
    assert cells["fixed1-native"]["60"]["n_pairs"] == 2


def test_t0_mapper_and_tau_availability_and_split():
    ref = [_rec("w1", "native", 60.0, 100.0)]
    treat = [_rec("w1", "fixed1", 60.0, 130.0),
             _rec("w1", "fixed1", 60.0, 777.0, t=30.0),          # t != t0
             _rec("w1", "fixed1", 60.0, 888.0, avail=0.0),       # unavailable
             _rec("w1", "fixed1", 99.0, 999.0)]                  # tau unregistered
    cells, _ = AC.build_arm_cells(treat, ref, taus=(60.0,))
    assert cells["fixed1-native"]["60"]["n_pairs"] == 1
    assert cells["fixed1-native"]["60"]["quantiles"][-1] == pytest.approx(30.0)


def test_duplicate_treatment_row_refused():
    ref = [_rec("w1", "native", 60.0, 100.0)]
    treat = [_rec("w1", "fixed1", 60.0, 130.0),
             _rec("w1", "fixed1", 60.0, 131.0)]
    with pytest.raises(AC.ArmContrastRefusal):
        AC.build_arm_cells(treat, ref, taus=(60.0,))


def test_duplicate_reference_row_refused():
    ref = [_rec("w1", "native", 60.0, 100.0),
           _rec("w1", "native", 60.0, 101.0)]
    treat = [_rec("w1", "fixed1", 60.0, 130.0)]
    with pytest.raises(AC.ArmContrastRefusal):
        AC.build_arm_cells(treat, ref, taus=(60.0,))


def test_quantiles_are_nineteen():
    treat, ref = _labels()
    cells, _ = AC.build_arm_cells(treat, ref, taus=(60.0, 120.0))
    q = np.asarray(cells["trajectory-native"]["120"]["quantiles"])
    assert q.shape == (N_QUANTILES,)


def test_e2_metadata_lists_all_contrasts():
    meta = AC.e2_metadata()
    assert meta["reference_policy"] == "native"
    assert set(meta["group_contrasts"]) == set(AC.GROUP_CONTRASTS)
    assert set(meta["member_contrasts"]) == set(AC.MEMBER_CONTRASTS)
