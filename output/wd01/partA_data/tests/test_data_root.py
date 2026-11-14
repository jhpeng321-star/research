"""Part 3k: WD01_DATA_ROOT data-root override tests.

The default (env unset) must keep the historical in-repo layout exactly
(partA_data/data, ../part1); setting WD01_DATA_ROOT points pdl/loader at
<root>/partA_data/data and <root>/part1.  A synthetic npz under a tmp dir
must load through loader.load_root when the override is set.
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pdl as PD  # noqa: E402


@pytest.fixture()
def restore_env():
    old = os.environ.get("WD01_DATA_ROOT")
    yield
    if old is None:
        os.environ.pop("WD01_DATA_ROOT", None)
    else:
        os.environ["WD01_DATA_ROOT"] = old
    importlib.reload(PD)


def test_default_layout_unchanged(restore_env):
    os.environ.pop("WD01_DATA_ROOT", None)
    importlib.reload(PD)
    here = Path(PD.__file__).resolve().parent
    assert PD.DATA == here / "data"
    assert PD.PART1 == here.parent / "part1"
    assert PD.DATA_ROOT_ENV == "WD01_DATA_ROOT"


def test_override_repoints_data_and_part1(restore_env, tmp_path):
    os.environ["WD01_DATA_ROOT"] = str(tmp_path)
    importlib.reload(PD)
    assert PD.DATA == tmp_path / "partA_data" / "data"
    assert PD.PART1 == tmp_path / "part1"


def test_override_loads_synthetic_root(restore_env, tmp_path):
    os.environ["WD01_DATA_ROOT"] = str(tmp_path)
    importlib.reload(PD)
    d = tmp_path / "partA_data" / "data"
    d.mkdir(parents=True)
    z = dict(
        vid=np.array([0, 0, 1], dtype=np.int32),
        tick=np.array([0, 1, 0], dtype=np.int16),
        lane=np.array([1, 1, 0], dtype=np.int8),
        s=np.array([100.0, 105.0, 50.0]),
        v=np.array([10.0, 10.0, 20.0], dtype=np.float32),
        a=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        is_cav=np.array([0, 0, 1], dtype=np.int8),
        in_junction=np.array([0, 0, 0], dtype=np.int8),
        type_label=np.array([1, 1, 1], dtype=np.int8),
        vid_table=np.array(["a", "b"], dtype="U8"),
        lc_vid=np.array([], dtype=np.int32),
        lc_tick=np.array([], dtype=np.int16),
        lc_from=np.array([], dtype=np.int8),
        lc_to=np.array([], dtype=np.int8),
        lc_s=np.array([], dtype=np.float64),
        b_vid=np.array([], dtype=np.int32),
        b_tick=np.array([], dtype=np.int16),
        b_lane=np.array([], dtype=np.int8),
        b_s=np.array([], dtype=np.float64),
        b_v=np.array([], dtype=np.float32),
        b_departed=np.array([], dtype=np.int8),
        e_vid=np.array([], dtype=np.int32),
        e_tick=np.array([], dtype=np.int16),
        e_s=np.array([], dtype=np.float64),
        e_arrived=np.array([], dtype=np.int8),
        ar_tick=np.array([], dtype=np.int16),
        ar_entrance=np.array([], dtype=np.int8),
        ar_v=np.array([], dtype=np.float32),
        ar_cav=np.array([], dtype=np.int8),
        ar_type=np.array([], dtype=np.int8),
    )
    np.savez(d / "root_synth3k.npz", **z)
    import loader as LD
    out = LD.load_root("synth3k")
    assert set(out["vid"].tolist()) == {0, 1}
    assert out["v"].shape == (3,)
    # the same file is invisible without the override (default layout)
    os.environ.pop("WD01_DATA_ROOT", None)
    importlib.reload(PD)
    assert not (Path(PD.DATA) / "root_synth3k.npz").exists()


def test_engine_weights_dir_follows_override(restore_env, tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "part3"))
    try:
        import engine as E
    except Exception:
        pytest.skip("engine imports unavailable")
    os.environ.pop("WD01_DATA_ROOT", None)
    here = Path(E.__file__).resolve().parent
    assert E._p3d_weights_dir() == here.parent / "part3d" / "weights" / \
        "run3d"
    os.environ["WD01_DATA_ROOT"] = str(tmp_path)
    assert E._p3d_weights_dir() == tmp_path / "part3d" / "weights" / "run3d"
