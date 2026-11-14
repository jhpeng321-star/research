"""E2 TRAIN_CLIMATOLOGY_V2 build tests (synthetic sidecars; no real labels)."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from evalh.scorer import build_climatology as BC
from evalh.scorer import build_climatology_v2 as V2

_BASE_COLS = list(BC._COLUMNS)
_ARM_COLS = list(V2.ARM_COLUMNS)


def _base_rows():
    rows = []
    for w in ("w1", "w2"):
        for tau in (60.0, 120.0):
            rows.append(dict(world=w, source="native", arm_kind="native_fleet",
                             policy="native_fleet", split="TRAIN", t0_s=0.0,
                             t=0.0, tau=tau, tau_available=1.0,
                             total_delay_ref_s=100.0))
            rows.append(dict(world=w, source="pi", arm_kind="pi_fleet",
                             policy="pi_fleet", split="TRAIN", t0_s=0.0, t=0.0,
                             tau=tau, tau_available=1.0,
                             total_delay_ref_s=130.0))
    return rows


def _arm_rows(source, delay, *, worlds=("w2",), pre_fix=False):
    out = []
    for w in worlds:
        for tau in (60.0, 120.0):
            out.append(dict(world=w, source=source, split="TRAIN", t0_s=0.0,
                            t=0.0, tau=tau, tau_available=1.0,
                            total_delay_ref_s=delay, pre_fix=pre_fix))
    return out


def _write(path, rows, cols):
    path.mkdir(parents=True, exist_ok=True)
    f = path / ("base_train_outcomes.parquet" if cols is _BASE_COLS
                else "overlay_outcomes.parquet")
    pd.DataFrame(rows, columns=cols).to_parquet(f, index=False)
    return f


def _fixture(tmp_path):
    base_dir = tmp_path / "cov"
    base = _write(base_dir, _base_rows(), _BASE_COLS)
    cover_rows = (_arm_rows("fixed1", 140.0, pre_fix=True)
                  + _arm_rows("fixed2", 150.0)
                  + _arm_rows("switching1", 160.0)
                  + _arm_rows("w-lo-early", 170.0)
                  + _arm_rows("fixed1", 145.0))
    coverage = _write(base_dir, cover_rows, _ARM_COLS)
    traj_dir = tmp_path / "traj"
    traj = _write(traj_dir, _arm_rows("arm0", 210.0), _ARM_COLS)
    return base, coverage, traj


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_v2_adds_arms_and_reproduces_pi(tmp_path):
    base, coverage, traj = _fixture(tmp_path)
    frozen = BC.compute_climatology(base.parent, expected_sha256=None, filename=base.name)
    doc = V2.build_document(
        base, coverage, traj, base_sha256=_sha(base),
        coverage_sha256=_sha(coverage), trajectory_sha256=_sha(traj))
    # pi-native reproduced exactly by the V1 code path
    assert doc["cells"]["pi-native"] == frozen["cells"]["pi-native"]
    # E2 arm contrasts present, pooled + separate
    for name in ("fixed_mix-native", "fixed1-native", "fixed2-native",
                 "switching_mix-native", "wall-native", "w-lo-early-native",
                 "trajectory-native", "arm0-native"):
        assert name in doc["cells"], name
    assert doc["contrasts"] == sorted(doc["cells"])
    assert doc["e2"]["reference_policy"] == "native"
    # pre_fix corrected row (145) wins over the pre-fix row (140)
    assert doc["cells"]["fixed1-native"]["60"]["quantiles"][-1] == pytest.approx(
        45.0)


def test_parity_detects_difference(tmp_path):
    base, coverage, traj = _fixture(tmp_path)
    doc = V2.build_document(
        base, coverage, traj, base_sha256=_sha(base),
        coverage_sha256=_sha(coverage), trajectory_sha256=_sha(traj))
    frozen_doc = BC.compute_climatology(base.parent, expected_sha256=None, filename=base.name)
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(frozen_doc), encoding="utf-8")
    assert V2.pi_parity(doc, frozen_path)["identical"] is True
    frozen_doc["cells"]["pi-native"]["60"]["quantiles"][0] += 1.0
    frozen_path.write_text(json.dumps(frozen_doc), encoding="utf-8")
    parity = V2.pi_parity(doc, frozen_path)
    assert parity["identical"] is False and parity["detail"]


def test_run_build_refuses_parity_diff_and_existing(tmp_path):
    base, coverage, traj = _fixture(tmp_path)
    frozen_doc = BC.compute_climatology(base.parent, expected_sha256=None, filename=base.name)
    frozen_doc["cells"]["pi-native"]["60"]["quantiles"][0] += 1.0
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(frozen_doc), encoding="utf-8")
    out = tmp_path / "v2.json"
    with pytest.raises(V2.ClimatologyV2Refusal):
        V2.run_build(base, coverage, traj, out, base_sha256=_sha(base),
                     coverage_sha256=_sha(coverage),
                     trajectory_sha256=_sha(traj), frozen_path=frozen)
    assert not out.exists()
    # allow-parity-diff records the difference instead of refusing
    V2.run_build(base, coverage, traj, out, base_sha256=_sha(base),
                 coverage_sha256=_sha(coverage), trajectory_sha256=_sha(traj),
                 frozen_path=frozen, allow_parity_diff=True)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["pi_native_parity"]["identical"] is False
    assert (out.with_name(out.name + ".sha256")).is_file()
    # never overwrite
    with pytest.raises(V2.ClimatologyV2Refusal):
        V2.run_build(base, coverage, traj, out, base_sha256=_sha(base),
                     coverage_sha256=_sha(coverage),
                     trajectory_sha256=_sha(traj))


def test_sha_pin_refused(tmp_path):
    base, coverage, traj = _fixture(tmp_path)
    with pytest.raises(V2.ClimatologyV2Refusal):
        V2.build_document(base, coverage, traj, base_sha256="0" * 64,
                          coverage_sha256=_sha(coverage),
                          trajectory_sha256=_sha(traj))


def test_non_train_split_refused(tmp_path):
    base, coverage, traj = _fixture(tmp_path)
    rows = _arm_rows("fixed1", 140.0)
    for r in rows:
        r["split"] = "CALIBRATION"
    pd.DataFrame(rows, columns=_ARM_COLS).to_parquet(coverage, index=False)
    with pytest.raises(V2.ClimatologyV2Refusal):
        V2.build_document(base, coverage, traj, base_sha256=_sha(base),
                          coverage_sha256=_sha(coverage),
                          trajectory_sha256=_sha(traj))


def test_sha256sums_pin(tmp_path):
    d = tmp_path / "sealed"
    d.mkdir()
    (d / "x.parquet").write_bytes(b"data")
    sha = hashlib.sha256(b"data").hexdigest()
    (d / "SHA256SUMS").write_text(f"{sha}  x.parquet\n", encoding="utf-8")
    assert V2.sha256sums_pin(d, "x.parquet") == sha
    with pytest.raises(V2.ClimatologyV2Refusal):
        V2.sha256sums_pin(d, "missing.parquet")
