"""tools/c4_compare.py pairing and bootstrap logic (no SUMO)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "c4_compare", Path(__file__).resolve().parents[2] / "tools" / "c4_compare.py")
cc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cc)


def _write(root: Path, label: str, tag: str, seed, dem, pen, deficit, srcwait, collisions=0):
    d = root / label / "episodes" / tag / f"s{seed}_m{dem}_p{int(pen * 1000):04d}"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(dict(
        seed=seed, demand_vph=dem, penetration=pen, collisions=collisions, teleports=0,
        window=dict(speed_deficit_vehicle_seconds=deficit, source_wait_vehicle_seconds=srcwait,
                    arrivals=100, braking_vehicle_seconds=10, ttc_below_3_vehicle_seconds=5))))


def test_parse_controllers_unique_labels():
    out = cc.parse_controllers(["native", "a=x/policy.pt", "b=y/policy.pt"])
    assert out == [("native", "native"), ("a", "x/policy.pt"), ("b", "y/policy.pt")]
    with pytest.raises(SystemExit):
        cc.parse_controllers(["a=x/policy.pt", "a=y/policy.pt"])
    with pytest.raises(SystemExit):
        cc.parse_controllers(["native", "native"])
    with pytest.raises(SystemExit):
        cc.parse_controllers(["bogus"])
    with pytest.raises(SystemExit):
        cc.parse_controllers(["a=runs/holdout/policy.pt"])


def test_total_wait_and_pairing_intersection(tmp_path):
    for seed, pen in [(1, 0.1), (2, 0.1), (3, 0.2)]:
        _write(tmp_path, "native", "native", seed, 2700, pen, 100.0, 50.0)
    for seed, pen in [(1, 0.1), (2, 0.1)]:               # missing (3, 0.2)
        _write(tmp_path, "pol", "policy_latest", seed, 2700, pen, 80.0, 40.0)
    data = cc.load_all(tmp_path)
    assert data["native"][(1, 2700, 0.1)]["total_wait"] == 150.0
    assert cc.paired_keys(data) == [(1, 2700, 0.1), (2, 2700, 0.1)]
    rep = cc.build_report(data)
    d = rep["diffs"]["pol"]["total_wait"]
    assert d["n"] == 2 and d["mean"] == pytest.approx(-30.0)
    assert d["lo"] == d["hi"] == pytest.approx(-30.0)
    assert rep["total_wait_percent_vs_native"]["pol"] == pytest.approx(-20.0)
    assert set(rep["by_penetration"]) == {"0.1"}
    assert set(rep["by_demand"]) == {"2700"}
    assert "pol" in cc.render_md(rep)
    assert len(json.dumps(rep)) < 1_000_000


def test_colliding_file_names_distinct_labels(tmp_path):
    _write(tmp_path, "native", "native", 1, 2400, 0.2, 10, 5)
    _write(tmp_path, "a", "policy_latest", 1, 2400, 0.2, 9, 5)
    _write(tmp_path, "b", "policy_latest", 1, 2400, 0.2, 7, 5)
    rep = cc.build_report(cc.load_all(tmp_path))
    assert rep["diffs"]["a"]["total_wait"]["mean"] == -1.0
    assert rep["diffs"]["b"]["total_wait"]["mean"] == -3.0


def test_bootstrap_deterministic_and_covers_mean():
    x = np.arange(20, dtype=float) - 5.0
    a, b = cc.bootstrap_ci(x), cc.bootstrap_ci(x)
    assert a == b
    assert a["mean"] == pytest.approx(x.mean())
    assert a["lo"] < a["mean"] < a["hi"]
    # manual reproduction of the resampling scheme
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 20, size=(2000, 20))
    lo, hi = np.quantile(x[idx].mean(axis=1), [0.025, 0.975])
    assert a["lo"] == pytest.approx(lo) and a["hi"] == pytest.approx(hi)
    assert cc.bootstrap_ci([3.0])["lo"] is None
    assert cc.bootstrap_ci([])["mean"] is None


def test_stochastic_flag_env(tmp_path, monkeypatch):
    seen = []

    class R:
        returncode = 0

    monkeypatch.setattr(cc.subprocess, "run", lambda cmd, env: seen.append((cmd, env)) or R())
    base = ["run", "--controllers", "native", "--seeds", "4117", "--out", str(tmp_path / "o")]
    monkeypatch.setenv("C4_POLICY_STOCHASTIC", "1")
    cc.main(base)
    assert "C4_POLICY_STOCHASTIC" not in seen[-1][1] and seen[-1][1]["OMP_NUM_THREADS"] == "1"
    assert "--no-physics-log" in seen[-1][0]
    cc.main(base + ["--stochastic"])
    assert seen[-1][1]["C4_POLICY_STOCHASTIC"] == "1"
