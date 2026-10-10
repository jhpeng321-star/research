"""CLI: version + the synthetic smoke pipeline (no real data)."""
from __future__ import annotations

import json

from combine.cli import _synthetic_pipeline, main


def test_version(capsys):
    assert main(["version"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["package"] == "wm_n1.combine"


def test_synthetic_pipeline_writes_outputs(tmp_path):
    out = tmp_path / "out"
    summary = _synthetic_pipeline(out, seed=0, n_train=8, n_cal=4, k=5)
    for name in ("summary.json", "PLANNER_VARIANT.json", "oof_tree.parquet",
                 "oof_tree.npz", "abstain_curve.csv"):
        assert (out / name).exists(), name
    assert summary["n_oof_rows"] == 8 * 3 * 2
    assert 0.0 <= summary["pool_weight"] <= 1.0
    assert summary["planner_variant"] == "C"
    assert len(summary["planner_variant_sha"]) == 64
