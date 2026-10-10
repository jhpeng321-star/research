"""End-to-end: CLI outputs, size math, SIZE_ESTIMATE.md sections."""
import json

import pandas as pd

import build_inventory as cli
import inventory_lib as inv
from fixtures import ALL_SPECS, build_fixture_root


def test_estimate_sizes_math(tmp_path):
    root = build_fixture_root(tmp_path)
    guard = inv.make_guard(root)
    rows, _h, _p = inv.build_rows(root, guard, specs=ALL_SPECS)
    micro = {"zstd_ratio": 0.5, "ticks_per_second": 1000.0}
    est = inv.estimate_sizes(rows, micro, workers=4)
    assert est["total_ticks"] > 0
    assert est["veh_bytes"] == est["total_ticks"] * 128 * 16 * 2
    assert est["mask_bytes"] == est["total_ticks"] * 128 * 2
    assert est["wall_hours_4_workers"] == est["total_ticks"] / (1000.0 * 4) / 3600.0


def test_cli_end_to_end(tmp_path, monkeypatch):
    root = build_fixture_root(tmp_path)
    monkeypatch.setattr(inv, "PLAN_SPECS", ALL_SPECS)
    out = tmp_path / "out"
    rc = cli.main(["--data-root", str(root), "--out", str(out),
                   "--micro-lines", "10", "--workers", "4"])
    assert rc == 0
    for name in ("inventory.parquet", "pairs.parquet", "SIZE_ESTIMATE.md",
                 "opened_paths.json", "holdout_names.json", "refused_runs.json",
                 "summary.json"):
        assert (out / name).is_file(), name

    df = pd.read_parquet(out / "inventory.parquet")
    assert len(df) > 0
    assert "policy" in df.columns and "split" in df.columns
    assert (df["split"] == "HOLDOUT").sum() == 0

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["holdout_runs_by_name"] >= 1
    assert summary["pairs"] > 0

    md = (out / "SIZE_ESTIMATE.md").read_text(encoding="utf-8")
    for heading in ("Runs enumerated", "Policy pairs", "Micro-sample",
                    "memmap size estimate", "Refused / incomplete",
                    "HOLDOUT accounting", "Opened-path audit"):
        assert heading in md, heading

    opened = json.loads((out / "opened_paths.json").read_text(encoding="utf-8"))
    assert all("wEXm2000p02t100" not in p for p in opened["paths"])
