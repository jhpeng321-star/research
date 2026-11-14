"""The v0.4b design matrix: deployable strict/detector state + v0.4a
action tokens + policy/pi + arm family + the planned-schedule
descriptor; prohibited inputs never appear (registration s.3)."""
from __future__ import annotations

import numpy as np
import pytest

from partB_v04b import reg
from partB_v04b import descriptors_v04b as DE
from partB_v04b import pairs_v04b as PB


@pytest.mark.parametrize("variant,width", [
    ("deployable_strict", 14 + 5 + 5 + 3 + 5 + 19 * 4 + 7),
    ("deployable_detector", 20 + 5 + 5 + 3 + 5 + 19 * 4 + 7)])
def test_feature_names_match_matrix(rows3, pairs3, variant, width):
    names = DE.feature_names(variant)
    assert len(names) == width
    tau_g = PB.complete_pairs(pairs3, 60.0)
    X = DE.pair_matrix(tau_g, rows3, variant)
    assert X.shape == (len(tau_g), width)


def test_strict_block_is_the_strict_variant(rows3, pairs3):
    """The HEADLINE state block carries the strict demand[6] + context[8]
    with the entrance queues zeroed -- never the detector columns."""
    g = PB.complete_pairs(pairs3, 60.0)
    Xs = DE.pair_matrix(g, rows3, "deployable_strict")
    Xd = DE.pair_matrix(g, rows3, "deployable_detector")
    # detector state block = [det demand 12 | ctx 8]; strict = [strict
    # demand 6 | ctx 8]; the strict demand equals detector demand cols
    # 6..11 (the CAV subset) -- the SourceObs contract
    assert np.allclose(Xs[:, 0:6], Xd[:, 6:12])
    assert not np.allclose(Xs[:, 0:6], Xd[:, 0:6])
    assert (Xd[:, 18:20] != 0).any()          # detector carries queues
    assert (Xs[:, 12:14] == 0).all()          # strict: queues zeroed



def test_arm_family_and_schedule_block(pairs3, rows3):
    g = PB.complete_pairs(pairs3, 60.0)
    X = DE.pair_matrix(g, rows3, "deployable_strict")
    names = DE.feature_names("deployable_strict")
    fam = X[:, [names.index(f"fam_{f}") for f in reg.FAMILIES]]
    wall_rows = g["contrast"] == reg.HEADLINE_CONTRAST
    assert (fam[wall_rows.to_numpy(),
                reg.FAMILIES.index("wall")] == 1.0).all()
    assert fam.sum(axis=1).max() == 1.0       # one-hot
    # the planned schedule of a wall arm carries its wall speed/hold
    hold = X[wall_rows.to_numpy(), names.index("wall_hold")]
    expect = g[wall_rows]["arm_source"].map(
        {"W-lo-early": 0.1, "W-lo-late": 0.2, "W-mid-early": 0.4,
         "W-mid-late": 0.1}).to_numpy()
    assert np.allclose(hold, expect)


def test_prohibited_inputs_never_appear():
    for variant in DE.VARIANTS:
        joined = " ".join(DE.feature_names(variant)).lower()
        for bad in ("penetration", "pre_fix", "speed_mode", "world",
                    "policy_id"):
            assert bad not in joined


def test_unknown_variant_is_loud(rows3, pairs3):
    g = PB.complete_pairs(pairs3, 60.0)
    with pytest.raises(ValueError):
        DE.pair_matrix(g, rows3, "oracle")
