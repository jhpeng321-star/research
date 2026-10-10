"""TRAIN-only fixed-best V2 builder tests (synthetic labels only)."""
from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from evalh.scorer import build_fixed_best_v2 as BV
from evalh.scorer import decision_quality as DQ


def _row(world="w1", source="fixed1", *, t0=0.0, t=0.0, tau=60.0, avail=1.0,
         delay=100.0, brake=1.0, split="TRAIN", pre_fix=False):
    return dict(world=world, source=source, arm=source, arm_kind="fixed",
                policy="fixed_mix", split=split, t0_s=t0, t=t, tau=tau,
                tau_available=avail, total_delay_ref_s=delay,
                hard_brake_s=brake, pre_fix=pre_fix)


def test_arm_type_maps_members_and_groups():
    assert BV._arm_type({"source": "arm0"}) == "trajectory"
    assert BV._arm_type({"arm_kind": "wall"}) == "wall"
    assert BV._arm_type({"policy": "pi_fleet"}) == "pi"
    assert BV._arm_type({"source": "base"}) == "damping"


def test_decision_points_pool_arm_types():
    rows = [
        _row(source="native", delay=100.0),
        _row(source="fixed1", delay=120.0),
        _row(source="fixed2", delay=110.0),
        _row(source="arm0", delay=90.0),
    ]
    points = BV._decision_points(rows, costs=("delay_s",), taus=(60.0,))
    assert len(points) == 1
    arms = {a.arm for a in points[0].arms}
    assert arms == {"native", "fixed_mix", "trajectory"}


def _write_sidecar(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_writes_schema_and_sha(tmp_path):
    base = tmp_path / "base.parquet"
    cov = tmp_path / "cov.parquet"
    traj = tmp_path / "traj.parquet"
    sha_b = _write_sidecar(base, [
        _row(source="native", delay=100.0),
        _row(source="pi", delay=105.0),
    ])
    sha_c = _write_sidecar(cov, [
        _row(source="fixed1", delay=120.0),
        _row(source="w-lo-early", delay=95.0),
    ])
    sha_t = _write_sidecar(traj, [
        _row(source="arm0", delay=90.0),
    ])
    doc = BV.build([base, cov, traj], shas=[sha_b, sha_c, sha_t],
                   costs=("delay_s",), taus=(60.0,))
    cell = doc["cells"]["delay_s"]["60"]
    assert doc["schema"] == DQ.SCHEMA_FIXED_BEST
    assert set(cell["arm_mean"]) == {"native", "pi", "fixed_mix", "wall",
                                     "trajectory"}
    assert cell["best_arm"] == "trajectory"
    assert doc["train_source"]["split"] == "TRAIN"


def test_build_refuses_bad_pin(tmp_path):
    base = tmp_path / "base.parquet"
    sha_b = _write_sidecar(base, [_row(source="native"), _row(source="pi")])
    with pytest.raises(BV.FixedBestV2Refusal):
        BV.build([base], shas=["0" * 64])
    assert sha_b  # keep the variable meaningful
