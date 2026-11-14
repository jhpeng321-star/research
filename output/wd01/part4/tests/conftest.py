"""Shared fixtures for the part-4 tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PART4 = Path(__file__).resolve().parents[1]
if str(PART4) not in sys.path:
    sys.path.insert(0, str(PART4))
os.environ.setdefault("WD_DEVICE", "cpu")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from delay.geometry import make_geometry  # noqa: E402

sys.path.insert(0, str(PART4.parent))        # output/wd01 (wdpaths)
import wdpaths  # noqa: E402

PARTA_DATA = str(Path(wdpaths.remap(os.environ.get(
    "WD01_PARTA_DATA",
    str(Path(wdpaths.RD_WT_ROOT) / "wd01" / "output" / "wd01" /
        "partA_data")))))


@pytest.fixture(scope="session")
def real_data_available():
    return Path(PARTA_DATA, "MANIFEST.json").exists()


@pytest.fixture()
def synth_geom():
    """Round-number synthetic section.

    lane 0 "main_left":  E1 [0,100] v=10, E2 [100,140] v=20
    lane 1 "main_right": E1 [0,90]  v=10, E2 [90,130]  v=20
    lane 2 "accel":      E3 [0,40]  v=5
    lane changes happen within an edge; E1 joins 0<->1 (offsets differ by
    10 m), E2 likewise.
    """
    return make_geometry(
        {0: [(0, 100, 10, "E1"), (100, 140, 20, "E2")],
         1: [(0, 90, 10, "E1"), (90, 130, 20, "E2")],
         2: [(0, 40, 5, "E3")]},
        names={0: "main_left", 1: "main_right", 2: "accel"},
        default_speed=10)


def trace(vid, t, lane, s, vtype=-1):
    from delay.core import VehicleTrace
    t = np.asarray(t, float)
    lane = np.asarray(lane, int)
    s = np.asarray(s, float)
    if lane.ndim == 0:
        lane = np.full(len(t), int(lane))
    if s.ndim == 0:
        s = np.full(len(t), float(s))
    return VehicleTrace(vid=str(vid), t=t, lane=lane, s=s,
                        vehicle_type=vtype)
