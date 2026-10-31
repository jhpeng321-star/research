"""select() admission tests for the wmdata_fit role and the unchanged FIT allowlist."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.wmfixmerge.data import SEEDS, WMDATA_SEEDS, select  # noqa: E402


def write_corpus(tmp, roles, rows):
    root = tmp / "corpus"
    root.mkdir(parents=True)
    (root / "SPLIT.json").write_text(json.dumps({"worlds": roles}))
    (root / "C4_EPISODES_MANIFEST.json").write_text(json.dumps({"episodes": rows}))
    for row in rows:
        (root / row["dir"]).mkdir(parents=True)
    return root


def fit_row(seed, arm, kind):
    world = f"w{seed}m2400p20t300"
    return {"seed": seed, "arm": arm, "world": world, "dir": f"episodes/{world}__{arm}",
            "warmup_s": 300, "pair_key": f"{world}@t300", "branch": {"kind": kind}}


def test_wmdata_world_selected_with_all_five_arms(tmp_path):
    rows = [fit_row(969400, "native", "B")] + [fit_row(969400, a, "POLICY")
                                               for a in ("ctrl0", "mappo", "pi", "circles")]
    root = write_corpus(tmp_path, {"969400": "wmdata_fit"}, rows)
    entries = select(root)
    assert {r["arm"] for r in entries} == {"native", "ctrl0", "mappo", "pi", "circles"}


def test_wmdata_partial_world_accepted_for_canary_verification(tmp_path):
    rows = [fit_row(969400, "native", "B"), fit_row(969400, "ctrl0", "POLICY")]
    root = write_corpus(tmp_path, {"969400": "wmdata_fit"}, rows)
    assert {r["arm"] for r in select(root)} == {"native", "ctrl0"}


def test_wmdata_seed_outside_block_refused(tmp_path):
    rows = [fit_row(969399, "native", "B"), fit_row(969399, "ctrl0", "POLICY")]
    root = write_corpus(tmp_path, {"969399": "wmdata_fit"}, rows)
    try:
        select(root)
    except ValueError as exc:
        assert "outside the registered block" in str(exc)
    else:
        raise AssertionError("unregistered wmdata seed admitted")


def test_wmdata_unregistered_arm_refused(tmp_path):
    rows = [fit_row(969400, "native", "B"), fit_row(969400, "ours_c1", "POLICY")]
    root = write_corpus(tmp_path, {"969400": "wmdata_fit"}, rows)
    try:
        select(root)
    except ValueError as exc:
        assert "unregistered wmdata_fit arm" in str(exc)
    else:
        raise AssertionError("unregistered wmdata arm admitted")


def test_wmdata_mappo_is_not_skipped(tmp_path):
    rows = [fit_row(969401, "native", "B"), fit_row(969401, "mappo", "POLICY")]
    root = write_corpus(tmp_path, {"969401": "wmdata_fit"}, rows)
    assert {r["arm"] for r in select(root)} == {"native", "mappo"}


def test_fit_path_unchanged(tmp_path):
    seed = SEEDS[0]
    rows = []
    for s in SEEDS:
        rows += [fit_row(s, "native", "B"), fit_row(s, "ours_c1", "POLICY"),
                 fit_row(s, "mappo", "POLICY")]
    root = write_corpus(tmp_path, {str(s): "fit" for s in SEEDS}, rows)
    entries = select(root)
    assert {r["arm"] for r in entries} == {"native", "ours_c1"}  # mappo stays diagnostic
    assert {int(r["seed"]) for r in entries} == set(SEEDS)
    (root / "SPLIT.json").write_text(json.dumps({"worlds": {str(seed): "validation",
                                                            **{str(s): "fit" for s in SEEDS[1:]}}}))
    try:
        select(root)
    except ValueError as exc:
        assert "is not FIT" in str(exc)
    else:
        raise AssertionError("non-fit role admitted for the 12-seed allowlist")


def test_fit_path_still_requires_all_twelve(tmp_path):
    seed = SEEDS[0]
    rows = [fit_row(seed, "native", "B"), fit_row(seed, "ours_c1", "POLICY")]
    root = write_corpus(tmp_path, {str(s): "fit" for s in SEEDS}, rows)
    try:
        select(root)
    except ValueError as exc:
        assert "registered TRAIN seeds missing" in str(exc)
    else:
        raise AssertionError("incomplete FIT corpus accepted")


def test_fit_arm_restriction_unchanged(tmp_path):
    rows = []
    for s in SEEDS:
        rows += [fit_row(s, "native", "B"), fit_row(s, "pi", "POLICY")]
    root = write_corpus(tmp_path, {str(s): "fit" for s in SEEDS}, rows)
    try:
        select(root)
    except ValueError as exc:
        assert "unregistered TRAIN arm" in str(exc)
    else:
        raise AssertionError("WMDATA-only arm admitted through the fit path")


def test_wmdata_block_disjoint_from_fit():
    assert not (set(WMDATA_SEEDS) & set(SEEDS))
