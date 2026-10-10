"""The pair builder: same world + t0 only; the saved-state contract;
complete horizons; strata (registration s.2/s.4/s.12)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from partB_v04b import reg
from partB_v04b import pairs_v04b as PB
from partB_v04b.pairs_v04b import PairContractError


def test_pairs_carry_all_registered_contrasts(pairs3):
    got = set(pairs3["contrast"])
    assert got == set(reg.CONTRAST_SOURCES)
    assert set(pairs3.loc[pairs3["contrast"] ==
                          reg.HEADLINE_CONTRAST, "arm_source"]) == \
        set(reg.WALL_ARMS)


def test_pairs_share_world_and_t0_only(pairs3, rows3):
    """A pair exists only between same-(world, t0) rows; the merge keys
    guarantee no cross-world or cross-t0 pair can exist."""
    assert (pairs3["t0"] == 300.0).all()
    for (w, arm), g in pairs3.groupby(["world", "arm_source"]):
        assert g["tau"].nunique() == len(reg.TAUS)
        # the arm row and the native row come from the SAME world
        nat = rows3[(rows3["world"] == w) &
                    (rows3["source"] == reg.NATIVE_SOURCE)]
        assert len(nat) == 1
        # a shifted-t0 arm row cannot pair: each world holds ONE t0 and
        # the merge keys are (world, t0)
        rows = rows3.copy()
        k = rows.index[rows["source"] == "base"][0]
        rows.loc[k, "t0_s"] = 300.5
        with pytest.raises(PairContractError, match=">1 t0"):
            PB.build_pairs(rows)
        # two worlds with DIFFERENT t0 values both pair at their own t0
        rows = rows3.copy()
        k = rows.index[rows["world"] == rows3["world"].iloc[0]]
        rows.loc[k, "t0_s"] = 240.0
        p2 = PB.build_pairs(rows)
        t1 = rows3["world"].iloc[0]
        assert (p2[p2["world"] == t1]["t0"] == 240.0).all()
        assert (p2[p2["world"] != t1]["t0"] == 300.0).all()


def test_state_id_mismatch_refuses_the_world(rows3):
    rows = rows3.copy()
    k = rows.index[rows["source"] == reg.NATIVE_SOURCE][0]
    rows.loc[k, "state_id"] = "different_state"
    with pytest.raises(PairContractError, match="saved_state_id"):
        PB.build_pairs(rows)


def test_duplicate_native_row_refuses(rows3):
    rows = pd.concat([rows3, rows3.iloc[[0]]], ignore_index=True)
    with pytest.raises(PairContractError, match="native"):
        PB.build_pairs(rows)


def test_dy_is_arm_minus_native_with_known_signs(pairs3):
    """The synthetic truth table: wall arms lower delay (dY < 0), aux
    arms carry their registered synthetic offsets."""
    for tau in reg.TAUS:
        g = PB.complete_pairs(pairs3, tau)
        wall = g[g["contrast"] == reg.HEADLINE_CONTRAST]
        assert (wall[f"dy_delay_{int(tau)}"] < 0).all()
        for _, r in wall.iterrows():
            expect = -2.0 * {"W-lo-early": 60.0, "W-lo-late": 120.0,
                             "W-mid-early": 240.0,
                             "W-mid-late": 60.0}[r["arm_source"]]
            assert r[f"dy_delay_{int(tau)}"] == pytest.approx(expect)
        dam = g[g["contrast"] == "damping-native"]
        assert np.allclose(dam[f"dy_delay_{int(tau)}"], -30.0)


def test_truncated_horizons_are_dropped(rows3):
    rows = rows3.copy()
    k = rows.index[rows["source"] == "W-lo-early"][0]
    rows.loc[k, "tau_avail_30"] = 20.0     # tau 30 not available
    p = PB.build_pairs(rows)
    one = p[(p["world"] == rows.loc[k, "world"]) &
            (p["arm_source"] == "W-lo-early") &
            np.isclose(p["tau"], 30.0)]
    assert one[f"dy_delay_30"].isna().all()
    full = PB.complete_pairs(p, 30.0)
    assert not ((full["world"] == rows.loc[k, "world"]) &
                (full["arm_source"] == "W-lo-early")).any()


def test_strata_and_representation_counts(pairs3):
    wall = pairs3[pairs3["contrast"] == reg.HEADLINE_CONTRAST]
    holds = wall["arm_source"].map(
        {"W-lo-early": 60.0, "W-lo-late": 120.0,
         "W-mid-early": 240.0, "W-mid-late": 60.0})
    expect = holds.map(lambda h: "le60" if h <= 60 else "mid")
    assert (wall["dur_stratum"] == expect).all()
    # aux whole-window arms (fleet damping) are 'whole'
    dam = pairs3[pairs3["contrast"] == "damping-native"]
    assert (dam["dur_stratum"] == "whole").all()
    rep = PB.representation_counts(wall)
    one = wall.drop_duplicates(subset=["world", "arm_source"])
    assert rep["n"] == len(one)
    # W-lo-early (hold 60 s, requested at t0) differs ONLY inside the
    # first 60 s: one first-60-only pair per world -- the registered
    # s.12.1 item 4 representation count
    assert rep["n_first60_only"] == one["world"].nunique()


def test_parquet_round_trip(rows3, pairs3, tmp_path):
    from partB_v04b import pairs_v04b as PB
    a = tmp_path / "rows.parquet"
    b = tmp_path / "pairs.parquet"
    PB.to_parquet_ready(rows3).to_parquet(a, index=False)
    PB.to_parquet_ready(pairs3).to_parquet(b, index=False)
    rows2 = pd.read_parquet(a)
    pairs2 = pd.read_parquet(b)
    p = PB.build_pairs(rows2)
    assert len(p) == len(PB.build_pairs(rows3))
    assert np.allclose(
        np.stack(pairs2["arm_desc"].map(np.asarray)),
        np.stack(pairs3["arm_desc"].map(np.asarray)))

