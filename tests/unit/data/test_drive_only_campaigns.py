"""Drive-only raw campaign trees resolve through the data registry (env / data_cache), not runs/."""
from __future__ import annotations

import json

from wms.data import lpwo, window_lists

CAMPAIGN = "campaign_stage1_chunk3"
ENV = "WMS_DATA_CAMPAIGN_STAGE1_CHUNK3"


def _schedule(root, run):
    path = root / run / "run" / "switch_schedule.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"run": run, "schedule": {"arm": "x", "events": []}}),
                    encoding="utf-8")
    return path


def test_drive_only_root_honours_the_environment_override(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path / "fetched"))
    assert window_lists.drive_only_root(CAMPAIGN) == tmp_path / "fetched"


def test_drive_only_root_defaults_below_data_cache(monkeypatch):
    monkeypatch.delenv("WMS_DATA_RAW_CHUNK1_P40", raising=False)
    root = window_lists.drive_only_root("raw_chunk1_p40")
    assert root.name == "raw_chunk1_p40" and root.parent.name == "data_cache"
    assert window_lists.CHUNK1_P40_OVERRIDE_ROOT.name == "raw_chunk1_p40"


def test_resolve_schedule_finds_the_fetched_campaign_only(tmp_path, monkeypatch):
    fetched = tmp_path / "fetched"
    expected = _schedule(fetched, "w1__base")
    monkeypatch.setenv(ENV, str(fetched))
    empty = (tmp_path / "nothing",)
    found = window_lists.resolve_schedule(
        "w1__base", f"runs/{CAMPAIGN}", {}, data_roots=empty)
    assert found == expected
    # a different campaign is never served from the fetched chunk-3 tree
    assert window_lists.resolve_schedule(
        "w1__base", "runs/campaign_stage1_chunk2", {}, data_roots=empty) is None


def test_run_dir_of_prefers_the_in_tree_run_then_the_fetched_one(tmp_path, monkeypatch):
    fetched = tmp_path / "fetched"
    (fetched / "w1__base").mkdir(parents=True)
    monkeypatch.setenv(ENV, str(fetched))
    row = {"root_rel": f"runs/{CAMPAIGN}", "run": "w1__base"}
    root = tmp_path / "tree"
    assert lpwo.run_dir_of(row, root) == fetched / "w1__base"
    in_tree = root / "runs" / CAMPAIGN / "w1__base"
    in_tree.mkdir(parents=True)
    assert lpwo.run_dir_of(row, root) == in_tree
    # a campaign that is not Drive-only keeps the plain in-tree location
    other = {"root_rel": "runs/campaign_stage1_chunk2", "run": "w1__base"}
    assert lpwo.run_dir_of(other, root) == root / "runs" / "campaign_stage1_chunk2" / "w1__base"
