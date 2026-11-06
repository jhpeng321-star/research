"""scripts/c4/wm_delay_check.py: the A6 delay-accuracy tool (pure helpers + CLI wiring).

No torch, no GPU, no checkpoint: the module is loaded via ``importlib.util`` and only its
torch-free helpers (region delay, paired statistics, selection, path guard, CLI parser) run.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
_spec = importlib.util.spec_from_file_location(
    "wm_delay_check", REPO / "scripts" / "c4" / "wm_delay_check.py")
WD = importlib.util.module_from_spec(_spec)
sys.modules["wm_delay_check"] = WD
_spec.loader.exec_module(WD)


# --------------------------------------------------------------------------- regions
def test_region_of_covers_the_four_zones():
    assert WD.region_of(-100.0, 0) == "up_near"
    assert WD.region_of(-100.0, 2) == "ramp"
    assert WD.region_of(10.0, 1) == "merge"
    assert WD.region_of(200.0, 1) == "downstream"
    assert WD.region_of(600.0, 1) is None          # off-road (s >= 553.33)
    assert WD.region_of(-900.0, 1) is None         # beyond the up_near zone
    assert WD.region_of(float("nan"), 1) is None


# --------------------------------------------------------------------------- delay level
def _frame(t, rows):
    return {"time_s": float(t), "vehicles": rows}


def _v(vid, s, lane, v, is_cav=False, present=True):
    return {"id": vid, "s": s, "lane": lane, "v": v, "is_cav": is_cav, "present": present}


def test_frame_delay_sums_shortfall_on_the_one_second_grid():
    rows = [_v("m", 10.0, 1, 0.0), _v("r", 10.0, 2, WD.FREE_SPEED_MPS)]
    frames = [_frame(t, rows) for t in range(61)]
    out = WD.frame_delay(frames, 0.0, (60.0,))
    assert out["60"]["merge"] == pytest.approx(60.0)
    assert out["60"]["ramp"] == pytest.approx(0.0)
    assert out["60"]["total"] == pytest.approx(60.0)
    assert out["60"]["ramp_queue"] == pytest.approx(0.0)


def test_frame_delay_counts_the_entrance_queue():
    rows = [_v("q", -900.0, 1, 0.0), _v("e", -500.0, 2, 0.0)]
    frames = [_frame(t, rows) for t in range(61)]
    out = WD.frame_delay(frames, 0.0, (60.0,))
    assert out["60"]["total"] == pytest.approx(0.0)      # both off-road
    assert out["60"]["ramp_queue"] == pytest.approx(120.0)   # two stopped queued per second


def test_frame_delay_is_zero_without_frames():
    out = WD.frame_delay([], 0.0, (60.0,))
    assert out["60"]["total"] == 0.0 and out["60"]["ramp_queue"] == 0.0


class _FakePredictor:
    """Returns the same frames for every arm (enough to exercise roster selection)."""

    def __init__(self, rows):
        self.rows = rows

    def __call__(self, record, controllers, horizon, sample):
        return [_frame(t, self.rows) for t in range(int(horizon) + 1)]


def _roster_record():
    return {"world": "w", "source_arm": "native", "t0_s": 0.0,
            "state": {"time_s": 0.0, "vehicles": [_v("m", 10.0, 1, 20.0)]},
            "recorded": {"native": [_frame(t, [_v("m", 10.0, 1, 20.0)])
                                    for t in range(61)]}}


def test_level_roster_override_selects_the_matched_fleet(monkeypatch):
    import copy
    rows = [_v("m", 10.0, 1, 20.0), _v("__entry_1", 20.0, 1, 0.0)]
    fake = _FakePredictor(rows)
    rec = _roster_record()
    roster = WD.evaluate_start(fake, copy.deepcopy(rec), (60.0,), 1, 60.0, roster_only=True)
    everyone = WD.evaluate_start(fake, copy.deepcopy(rec), (60.0,), 1, 60.0, roster_only=False)
    roster_total = roster["arms"]["native"]["wm"]["60"]["total"]
    all_total = everyone["arms"]["native"]["wm"]["60"]["total"]
    assert all_total - roster_total == pytest.approx(60.0)

    monkeypatch.setattr(WD, "ROSTER_ONLY", True)
    inherited = WD.evaluate_start(fake, copy.deepcopy(rec), (60.0,), 1, 60.0)
    assert inherited["arms"]["native"]["wm"]["60"]["total"] == pytest.approx(roster_total)
    monkeypatch.setattr(WD, "ROSTER_ONLY", False)
    inherited = WD.evaluate_start(fake, copy.deepcopy(rec), (60.0,), 1, 60.0)
    assert inherited["arms"]["native"]["wm"]["60"]["total"] == pytest.approx(all_total)


# --------------------------------------------------------------------------- near CAV |ds|
def test_near_cav_abs_ds_only_counts_hdvs_near_a_cav():
    t0 = _frame(0.0, [_v("cav", 100.0, 1, 20.0, is_cav=True), _v("h1", 150.0, 1, 20.0),
                      _v("h2", 300.0, 1, 20.0)])
    recorded = _frame(10.0, [_v("cav", 300.0, 1, 20.0, is_cav=True), _v("h1", 350.0, 1, 20.0),
                             _v("h2", 500.0, 1, 20.0)])
    predicted = _frame(10.0, [_v("cav", 300.0, 1, 20.0, is_cav=True), _v("h1", 355.0, 1, 20.0),
                              _v("h2", 480.0, 1, 20.0)])
    out = WD.near_cav_abs_ds([predicted], [recorded], t0, (10.0,))
    assert out["10"] == pytest.approx(5.0)   # only h1 is near the CAV at t0


# --------------------------------------------------------------------------- paired stats
def test_sign_agreement_spearman_mae():
    assert WD.sign_agreement([1.0, -1.0, 2.0], [1.0, 1.0, -2.0]) == pytest.approx(1 / 3)
    assert WD.sign_agreement([0.0], [0.0]) == 1.0
    assert WD.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert WD.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert WD.spearman([1], [1]) is None
    assert WD.spearman([1, 1, 1], [1, 2, 3]) is None
    assert WD.mean_abs_error([1, 2], [3, 4]) == pytest.approx(2.0)
    assert WD.log_ratio(2.0, 1.0) == pytest.approx(WD.math.log(2.0))
    assert WD.log_ratio(0.0, 1.0) is None


def test_ranks_handle_ties():
    assert WD._ranks([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]


# --------------------------------------------------------------------------- selection
def _rec(i):
    return {"world": f"w{i:03d}", "key": f"k{i:03d}", "source_arm": "wall", "t0_s": float(i)}


def test_select_evenly_is_deterministic():
    records = [_rec(i) for i in range(100)]
    chosen = WD.select_evenly(records, 48)
    assert len(chosen) == 48
    assert [WD.selection_key(r) for r in chosen] == \
        [WD.selection_key(r) for r in WD.select_evenly(records, 48)]
    assert WD.selection_key(chosen[0]) == WD.selection_key(sorted(records, key=WD.selection_key)[0])
    assert WD.select_evenly(records[:3], 48) == sorted(records[:3], key=WD.selection_key)


def test_selection_round_trip(tmp_path):
    path = tmp_path / "sel.json"
    WD.write_selection(path, [_rec(0), _rec(1)], "states")
    assert [r["key"] for r in WD.load_selection(path, "states")] == ["k000", "k001"]
    ppath = tmp_path / "psel.json"
    WD.write_selection(ppath, [{"pair_key": "p0"}], "pairs")
    assert WD.load_selection(ppath, "pairs")[0]["pair_key"] == "p0"


# --------------------------------------------------------------------------- paired branches
def _pair_record(pair_key, arm, frames):
    return {"world": "w", "key": pair_key, "pair_key": pair_key, "source_arm": arm,
            "t0_s": 300.0, "state": {"time_s": 300.0, "vehicles": [_v("m", 10.0, 1, 20.0)]},
            "recorded": {arm: frames}}


def test_group_pairs_merges_arms_and_picks_the_native_reference():
    base = _pair_record("p0", "B", [_frame(300.0, [_v("m", 10.0, 1, 20.0)])])
    action = _pair_record("p0", "SEED:0", [_frame(300.0, [_v("m", 12.0, 1, 18.0)])])
    units = WD.group_pairs([base, action])
    assert len(units) == 1
    unit = units[0]
    assert unit["pair_key"] == "p0" and unit["ref_arm"] == "B" and unit["action_arm"] == "SEED:0"
    assert set(unit["recorded"]) == {"B", "SEED:0"}


def test_group_pairs_reads_a_multi_arm_record():
    record = _pair_record("p1", "native", [])
    record["recorded"] = {"native": [_frame(300.0, [_v("m", 10.0, 1, 20.0)])],
                          "wall": [_frame(300.0, [_v("m", 9.0, 1, 19.0)])]}
    units = WD.group_pairs([record])
    assert len(units) == 1 and units[0]["ref_arm"] == "native"


def test_pair_key_is_stable():
    record = _pair_record("p0", "native", [])
    record["recorded"] = {"native": [_frame(300.0, [_v("m", 10.0, 1, 20.0)])],
                          "wall": [_frame(300.0, [_v("m", 9.0, 1, 19.0)])]}
    unit = WD.group_pairs([record])[0]
    assert WD.pair_key(unit) == ("p0", "native", "wall", 300.0)


# --------------------------------------------------------------------------- rollback + CLI
def test_rollback_block_reads_the_fixed_horizons():
    level = {"120": {"log_ratio_total": 0.2}}
    pairs = {"60": {"sign_agreement": 0.7}}
    near = {"10": 3.0, "30": 5.0}
    block = WD.rollback_block(level, pairs, near)
    assert block["pair_sign_agreement"] == 0.7
    assert block["delay_level_log_ratio"] == 0.2
    assert block["near_cav_abs_ds_10"] == 3.0
    assert block["pair_sign_agreement_horizon_s"] == 60
    assert block["delay_level_log_ratio_horizon_s"] == 120


def test_restricted_paths_refused():
    with pytest.raises(SystemExit):
        WD.refuse_restricted(str(Path("data") / "holdout" / "x.json"))
    assert WD.refuse_restricted("data_cache/fixed_train_val_states.json").endswith(".json")


def test_cli_parses_and_requires_wm_and_out():
    ap = WD.parser()
    args = ap.parse_args(["--wm", "x.pt", "--states", "s.json", "--out", "o.json"])
    assert args.wm == ["x.pt"] and args.out == "o.json"
    assert args.device == "cpu" and args.k == 8 and args.entries == "queue"
    assert args.horizons == "60,120,300"
    assert args.level_roster_only is None and args.pair_roster_only is None
    split = ap.parse_args(["--wm", "x.pt", "--out", "o.json",
                           "--level-roster-only", "1", "--pair-roster-only", "0"])
    assert split.level_roster_only == 1 and split.pair_roster_only == 0
    multi = ap.parse_args(["--wm", "a.pt", "--wm", "b.pt", "--out", "o.json"])
    assert multi.wm == ["a.pt", "b.pt"]
    with pytest.raises(SystemExit):
        ap.parse_args(["--states", "s.json", "--out", "o.json"])
    with pytest.raises(SystemExit):
        ap.parse_args(["--wm", "x.pt"])


def test_markdown_summary_lists_sections(tmp_path):
    results = {"schema": WD.SCHEMA, "horizons": ["60", "120"],
               "k": 8, "entries": "queue", "n_starts": 2, "n_pairs": 1,
               "wms": {"wm.pt": {"delay_level": {"60": {"wm_total": 1.0, "sumo_total": 1.0,
                                                        "log_ratio_total": 0.0}},
                                 "paired": {"60": {"sign_agreement": 0.5, "spearman": None,
                                                   "mean_abs_error": 0.1}}}},
               "rollback": {"pair_sign_agreement": 0.5}}
    text = WD.render_markdown(results)
    assert "Rollback metrics" in text and "sign agreement" in text and "wm.pt" in text
    assert WD.parse_horizons("60,120,300") == (60.0, 120.0, 300.0)
