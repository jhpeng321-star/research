"""PL-02b tests: the chunk-parameterised FINAL renderer (report_chunk_final).

Hermetic except where the COPIED chunk-1 result package inside this
workspace is the fixture (the chunk-1 per-run cache is NOT part of this
workspace, so the byte-identity proof of the chunk-1 path runs over the
recorded per-root records instead of a full re-render).
"""

import argparse
import csv
import json
from pathlib import Path

import pytest

pytest.importorskip(
    "output.pl02a_offline_v1.report_chunk_final",
    reason="PL-02a FINAL report module (output/pl02a_offline_v1) "
    "lives in the planner worktree only",
)
from output.pl02a_offline_v1 import report_chunk_final as rcf  # noqa: E402
from output.pl02a_offline_v1.collisions import CollisionPair, RunReport  # noqa: E402
from output.pl02a_offline_v1.report_chunk1_final import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_OUT,
    DEFAULT_PLAN,
    DEFAULT_RUNS_ROOT,
    DEFAULT_SUMMARY,
    DEFAULT_TALLY,
    P40_PREFIX_NOTE,
    root_note,
)

_CHUNK1_RESULTS = Path(rcf.__file__).parent / "results_chunk1_final" / "RESULTS.json"
_CHUNK1_ROOTS_CSV = Path(rcf.__file__).parent / "results_chunk1_final" / "roots.csv"


# ------------------------------------------------------------ chunk presets
def test_chunk1_preset_matches_original_defaults():
    spec = rcf.CHUNKS[1]
    assert spec.runs_root == DEFAULT_RUNS_ROOT
    assert spec.plan == DEFAULT_PLAN
    assert spec.tally == DEFAULT_TALLY
    assert spec.chunk_summary == DEFAULT_SUMMARY
    assert spec.out == DEFAULT_OUT
    assert spec.cache == DEFAULT_CACHE
    assert spec.penetration_note == P40_PREFIX_NOTE


def test_chunk2_preset_is_no_tally_post_fix():
    spec = rcf.CHUNKS[2]
    assert spec.tally is None  # no owner tally: own-scan mode
    assert spec.chunk_summary is None
    assert spec.speed_mode_label == "speedMode 27 (post-fix)"
    assert "campaign_stage1_chunk2" in str(spec.runs_root)
    assert spec.cache / "campaign_stage1_chunk2" == spec.cache / spec.runs_root.name
    assert spec.reference_results is not None  # chunk-1 NOT-POOLED reference
    assert spec.penetration_note == ""


def test_resolve_args_overrides_and_empty_forces_none():
    base = {"chunk": 2, "runs_root": None, "plan": None, "out": None, "cache": None,
            "tally": None, "chunk_summary": None}
    spec, runs_root, _plan, _out, cache, tally, summary = rcf.resolve_args(
        argparse.Namespace(**base)
    )
    assert spec.chunk == 2
    assert tally is None and summary is None  # chunk-2 preset: none
    assert cache == DEFAULT_CACHE and runs_root == rcf.CHUNKS[2].runs_root
    # an explicitly-EMPTY --tally forces none even on the chunk-1 preset
    spec1, _r, _p, _o, _c, tally1, _s = rcf.resolve_args(
        argparse.Namespace(**{**base, "chunk": 1, "tally": ""})
    )
    assert spec1.chunk == 1 and tally1 is None
    with pytest.raises(SystemExit):
        rcf.resolve_args(argparse.Namespace(**{**base, "chunk": None}))


# ------------------------------------------- chunk-1 byte-stability (roots.csv)
@pytest.mark.skipif(not _CHUNK1_RESULTS.is_file(), reason="copied chunk-1 results absent")
def test_chunk1_roots_csv_byte_identical(tmp_path):
    """With chunk-1 arguments the renderer reproduces
    results_chunk1_final/roots.csv BYTE-IDENTICALLY.

    The full chunk-1 re-render needs the chunk-1 per-run cache, which is
    NOT part of this workspace (the pl12 worktree ships without it) -- so the
    proof runs over the recorded per-root records: RESULTS.json['per_root']
    is exactly the record list build_report handed to the roots.csv writer,
    and this renderer's writer is that same code path with a parameterised
    note source."""
    results = json.loads(_CHUNK1_RESULTS.read_text(encoding="utf-8"))
    records = results["per_root"]
    pre_t0 = set(results["inclusion"]["pre_t0_base_collision_roots"])
    out = tmp_path / "roots.csv"
    rcf.write_roots_csv_generic(out, records, pre_t0, root_note)
    assert out.read_bytes() == _CHUNK1_ROOTS_CSV.read_bytes()


@pytest.mark.skipif(not _CHUNK1_RESULTS.is_file(), reason="copied chunk-1 results absent")
def test_root_note_generic_equivalent_to_original_on_chunk1_records():
    results = json.loads(_CHUNK1_RESULTS.read_text(encoding="utf-8"))
    pre_t0 = set(results["inclusion"]["pre_t0_base_collision_roots"])
    for record in results["per_root"]:
        assert rcf.root_note_generic(record, pre_t0, P40_PREFIX_NOTE) == root_note(
            record, pre_t0
        )
        # chunk-2 mode (no pre-fix note) only ever DROPS the p40 note
        dropped = "; ".join(
            segment
            for segment in root_note(record, pre_t0).split("; ")
            if segment != P40_PREFIX_NOTE
        )
        assert rcf.root_note_generic(record, pre_t0, "") == dropped


# ------------------------------------------------- interrupted-dir exclusion
def test_classify_unplanned_excludes_interrupted_dirs():
    names = [
        "w8495m2700p05t360__fixed1.interrupted_20260927T075813Z",
        "w8495m2700p05t360__pi.interrupted_20260927T075813Z",
        "stray_unplanned_run__base",
        "w8481m2700p02t480__base",  # not unplanned in reality, but a clean name
    ]
    interrupted, other = rcf.classify_unplanned(names)
    assert interrupted == [
        "w8495m2700p05t360__fixed1.interrupted_20260927T075813Z",
        "w8495m2700p05t360__pi.interrupted_20260927T075813Z",
    ]
    assert other == ["stray_unplanned_run__base", "w8481m2700p02t480__base"]
    assert rcf.classify_unplanned([]) == ([], [])


# ----------------------------------------------------- HOLDOUT row exclusion
def _record(root: str, split: str) -> dict:
    return {
        "root": root,
        "meta": {"split": split, "penetration": 0.02, "loading_veh_h": 2700},
        "window_s": 600,
        "usable": True,
        "native_delay_s": 100.0,
        "arms": {"pi": {"delay_s": 90.0, "saving_vs_native_s": 10.0}},
        "best_safe_arm": "pi",
        "best_safe_delay_s": 90.0,
        "provider_signs_agree": True,
    }


def test_holdout_rows_never_reach_roots_csv(tmp_path):
    records = [
        _record("wtrain__base", "TRAIN"),
        _record("wcalib__base", "CALIBRATION"),
        _record("wholdout__base", "HOLDOUT"),
    ]
    # production chain: build_report refuses to write ANY output while an
    # out-of-scope record is present (the guard runs before the writer)
    with pytest.raises(SystemExit, match="scope guard"):
        rcf.assert_no_holdout_records(records, ("TRAIN", "CALIBRATION"))
    in_scope = [r for r in records if str(r["meta"]["split"]) in ("TRAIN", "CALIBRATION")]
    rcf.assert_no_holdout_records(in_scope, ("TRAIN", "CALIBRATION"))
    out = tmp_path / "roots.csv"
    rcf.write_roots_csv_generic(out, in_scope, set(), lambda r, f: "")
    rows = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert sorted(r["split"] for r in rows) == ["CALIBRATION", "TRAIN"]
    assert not any("holdout" in r["root"].lower() for r in rows)


def test_holdout_record_fails_loudly():
    with pytest.raises(SystemExit, match="scope guard"):
        rcf.assert_no_holdout_records([_record("wholdout__base", "HOLDOUT")], ["TRAIN"])
    # in-scope records pass the guard untouched
    rcf.assert_no_holdout_records([_record("wtrain__base", "TRAIN")], ["TRAIN", "CALIBRATION"])


def _pair(root: str, run: str, t_s: float, classification: str) -> CollisionPair:
    return CollisionPair(
        run=run,
        root=root,
        arm=run.rsplit("__", 1)[-1],
        is_base=run.endswith("__base"),
        receipt_status="complete",
        t_s=t_s,
        t0_s=480.0,
        classification=classification,
        follower_id="f1",
        follower_type="cautious_car",
        follower_speed_mps=1.0,
        follower_accel_mps2=0.0,
        leader_id="l1",
        leader_type="normal_car",
        leader_speed_mps=0.0,
        lane_id="lane",
        follower_position_m=1.0,
        pair_kind="human-behind-CAV",
    )


def test_no_tally_collision_section_holdout_count_only():
    worlds = {"wtrain": {"split": "TRAIN"}, "wholdout": {"split": "HOLDOUT"}}
    pairs = [
        _pair("wtrain", "wtrain__pi", 500.0, "in_window"),
        _pair("wholdout", "wholdout__native", 501.0, "in_window"),
    ]
    section, block = rcf.render_scan_collision_sections(
        pairs,
        reports=[],
        worlds=worlds,
        splits=("TRAIN", "CALIBRATION"),
        counts={"n_roots_planned": 2},
        interrupted=["w__base.interrupted_X"],
        unplanned_other=[],
        failures=[],
        speed_mode_label=rcf.CHUNK2_SPEED_MODE_LABEL,
        chunk=2,
    )
    assert "HOLDOUT collision pairs (count only): 1" in section
    assert block["holdout_collision_pairs_count_only"] == 1
    assert block["pairs_in_scope"] == 1
    # the HOLDOUT run/root names appear NOWHERE in the section
    assert "wholdout" not in section.lower()
    assert block["interrupted_run_dirs_ignored"] == ["w__base.interrupted_X"]


def test_no_tally_scan_zero_count_statement_and_loud_failure():
    worlds = {"wtrain": {"split": "TRAIN"}}
    section, block = rcf.render_scan_collision_sections(
        [],
        reports=[],
        worlds=worlds,
        splits=("TRAIN", "CALIBRATION"),
        counts={"n_roots_planned": 1},
        interrupted=[],
        unplanned_other=[],
        failures=[],
        speed_mode_label=rcf.CHUNK2_SPEED_MODE_LABEL,
        chunk=2,
    )
    assert "collision-pair count (speedMode 27 (post-fix)): 0 in scope + 0 HOLDOUT" in section
    assert "receipts and physics AGREE" in section
    assert block["mode"].startswith("no-tally physics scan")
    assert block["receipt_physics_agreement_ok"] is True
    with pytest.raises(SystemExit, match="DISAGREEMENT"):
        rcf.render_scan_collision_sections(
            [],
            reports=[],
            worlds=worlds,
            splits=("TRAIN",),
            counts={},
            interrupted=[],
            unplanned_other=[],
            failures=["some_run: receipt status 'failed' (expected complete)"],
            speed_mode_label=rcf.CHUNK2_SPEED_MODE_LABEL,
            chunk=2,
        )


# --------------------------------------------- receipt/physics agreement
def _write_plan(tmp_path: Path) -> Path:
    plan = {
        "worlds": [
            {
                "world": "wA",
                "t0_s": 60.0,
                "base_arm": {
                    "run": "wA__base",
                    "duration_s": 600.0,
                    "saves_state": {"at_s": 60.0},
                },
                "branches": [
                    {
                        "arm": arm,
                        "run": f"wA__{arm}",
                        "duration_s": 600.0,
                        "restores_state": {"at_s": 60.0, "from_run": "wA__base"},
                    }
                    for arm in ("native", "pi", "fixed1")
                ],
            }
        ]
    }
    path = tmp_path / "WORLD_PLAN.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _report(run: str, status="complete", last_t=660.0, match=None) -> RunReport:
    return RunReport(
        run=run,
        root="wA",
        arm=run.rsplit("__", 1)[-1],
        is_base=True,
        receipt_status=status,
        n_pairs=0,
        n_in_window=0,
        n_pre_t0=0,
        summary_involvements=None if match is None else 0,
        involvements_match=match,
        physics_last_t=last_t,
    )


def test_receipt_physics_agreement_checks(tmp_path):
    plan = _write_plan(tmp_path)  # planned end = 660.0
    ok = rcf.receipt_physics_failures([_report("wA__base")], plan)
    assert ok == []
    failures = rcf.receipt_physics_failures(
        [
            _report("wA__base", status="failed"),
            _report("wA__native", last_t=500.0),
            _report("wA__pi", last_t=None),
            _report("wA__fixed1", match=False),
        ],
        plan,
    )
    joined = "\n".join(failures)
    assert "receipt status 'failed'" in joined
    assert "short of the planned 660" in joined
    assert "no physics rows" in joined
    assert "collision_involvements=0 does not match" in joined
    # an involvements_match of None (no summary.json) is NOT a disagreement
    assert rcf.receipt_physics_failures([_report("wA__base", match=None)], plan) == []


# ---------------------------------------------------- headline additions
def test_pi_vs_native_and_median_native_delay():
    good = _record("w1__base", "TRAIN")
    good["arms"]["pi"].update({"safe_planner": True})
    bad = _record("w2__base", "TRAIN")
    bad["arms"]["pi"].update({"delay_s": 120.0, "saving_vs_native_s": -20.0})
    bad["arms"]["pi"].update({"safe_planner": False})
    no_pi = _record("w3__base", "TRAIN")
    no_pi["arms"] = {}
    summary = rcf.pi_vs_native_summary([good, bad, no_pi])
    assert summary["n_roots_with_pi"] == 2
    assert summary["fraction_pi_beats_native_safely"] == pytest.approx(0.5)
    # 0-filled: the no-improvement root counts 0
    assert summary["median_pi_saving_0filled_s"] == pytest.approx(5.0)
    assert rcf.median_native_delay_s([good, bad]) == pytest.approx(100.0)
    assert rcf.median_native_delay_s([]) is None
