"""A9 step 5: overlay identity + t0 row pick are fail-closed."""
import pandas as pd
import pytest

from dspeed import overlay


def _frame(**kw):
    base = dict(source=["arm0"], policy=["trajectory"], chunk_id=["chunk2"],
                split=["TRAIN"], t0_s=[100.0])
    base.update(kw)
    return pd.DataFrame(base)


def test_identity_accepts_matching():
    overlay._identity_ok(_frame(), dict(arm="arm0", chunk="chunk2",
                                        t0_s=100.0))


@pytest.mark.parametrize("col,val", [
    ("source", "arm1"), ("policy", "native"), ("chunk_id", "chunk3"),
    ("split", "CALIBRATION"), ("t0_s", 101.0)])
def test_identity_rejects_mismatch(col, val):
    with pytest.raises(Exception):
        overlay._identity_ok(_frame(**{col: [val]}),
                             dict(arm="arm0", chunk="chunk2", t0_s=100.0))


def test_t0_pick_rejects_duplicate_tau():
    df = pd.DataFrame({"t": [100.0, 100.0], "tau": [30.0, 30.0]})
    with pytest.raises(Exception):
        overlay._t0_pick(df, 100.0)


def test_t0_pick_keeps_only_t0_rows():
    df = pd.DataFrame({"t": [100.0, 130.0], "tau": [30.0, 30.0]})
    picked = overlay._t0_pick(df, 100.0)
    assert list(picked) == [30.0]
    assert float(picked[30.0]["t"]) == 100.0
