"""P7 Idea A held-out dry-run filters (A3.4 tests i/ii/iii, gate G6).

Builds a synthetic packed corpus with J4 / J4B, PI and wall-speed arms and
proves that each held-out config builds train/eval splits with the excluded
arms ABSENT from the training view.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import train.idea_a as ia
from train.dataset import MemmapCorpus
from train.tests import packsynth


def _runs():
    return [
        dict(chunk="chunk4", world="J4", arm="native", policy="native",
             split="TRAIN", t0_s=100.0),
        dict(chunk="chunk4", world="J4", arm="pi", policy="pi",
             split="TRAIN", t0_s=100.0),
        dict(chunk="chunk4", batch="walls_pilot_chunk4", world="J4",
             arm="W-lo-early", policy="wall", split="TRAIN", t0_s=100.0),
        dict(chunk="chunk4", batch="walls_pilot_chunk4", world="J4",
             arm="W-mid-early", policy="wall", split="TRAIN", t0_s=100.0),
        dict(chunk="chunk4", world="J4", arm="base", policy="damping",
             split="TRAIN", t0_s=100.0),
        dict(chunk="chunk4", world="J4B", arm="native", policy="native",
             split="TRAIN", t0_s=200.0),
        dict(chunk="chunk4", world="J4B", arm="pi", policy="pi",
             split="TRAIN", t0_s=200.0),
        dict(chunk="chunk2", world="J4", arm="native", policy="native",
             split="CALIBRATION", t0_s=300.0),
    ]


@pytest.fixture()
def corpus(tmp_path):
    # MemmapCorpus requires a TRAIN-only pack unless mixed-split identity
    # sidecars are present; this fixture has no such sidecars.
    packsynth.write_pack(tmp_path / "pack", [
        run for run in _runs() if run["split"] == "TRAIN"])
    return MemmapCorpus(tmp_path / "pack", split="TRAIN")


def _arms(items):
    return sorted((it["world"], it["arm"]) for it in items)


def test_held_out_i_j4_to_j4b(corpus):
    view = ia.held_out_corpus_view(corpus, "i_j4_to_j4b")
    arms = set(view.arms())
    assert not any(world == "J4B" for world, _ in arms)   # absent from train
    assert ("J4", "native") in arms and ("J4", "pi") in arms
    assert set(view.eval_arms()) == {("J4B", "native"), ("J4B", "pi")}
    assert len(view) == 5 and len(view) + len(view.eval_arms()) == 7
    for i in range(len(view)):                # the view serves corpus items
        assert view[i]["world"] in ("J4",)


def test_held_out_ii_leave_pi_out(corpus):
    view = ia.held_out_corpus_view(corpus, "ii_leave_pi_out")
    arms = set(view.arms())
    assert not any(arm == "pi" for _, arm in arms)
    assert set(view.eval_arms()) == {("J4", "pi"), ("J4B", "pi")}
    assert len(view) == 5


def test_held_out_iii_leave_top_wall_speed_out(corpus):
    view = ia.held_out_corpus_view(corpus, "iii_leave_top_wall_speed_out")
    arms = set(view.arms())
    assert ("J4", "W-lo-early") in arms       # lower factor stays
    assert ("J4", "W-mid-early") not in arms  # frozen top factor excluded
    assert view.eval_arms() == [("J4", "W-mid-early")]
    assert ia.wall_speed_factor({"arm": "wall_v31"}) == 31.0
    assert ia.wall_speed_factor({"arm": "wall-27mph"}) == 27.0
    assert ia.wall_speed_factor({"wall_speed_factor": 24}) == 24.0
    explicit = ia.held_out_corpus_view(corpus, "iii_leave_top_wall_speed_out",
                                       factor=0.60)
    assert ("J4", "W-lo-early") not in set(explicit.arms())
    assert explicit.eval_arms() == [("J4", "W-lo-early")]


def test_held_out_iii_without_wall_arms_refused():
    with pytest.raises(ia.IdeaAConfigError):
        ia.held_out_split([{"world": "J4", "arm": "native",
                            "policy": "native"}],
                          "iii_leave_top_wall_speed_out")


def test_held_out_dry_run_gate_receipt(corpus):
    report = ia.held_out_dry_run(corpus)
    assert sorted(report) == sorted(ia.HELD_OUT_TESTS)
    assert report["i_j4_to_j4b"]["n_eval"] == 2
    assert report["ii_leave_pi_out"]["n_eval"] == 2
    assert report["iii_leave_top_wall_speed_out"]["excluded_arms"] == \
        [("J4", "W-mid-early")]
    for test in ia.HELD_OUT_TESTS:
        assert report[test]["n_train"] + report[test]["n_eval"] == 7
    with pytest.raises(ia.IdeaAConfigError):
        ia.held_out_split([], "nope")
    with pytest.raises(ia.IdeaAConfigError):
        ia.held_out_dry_run(object())


def test_held_out_split_on_raw_items_disjoint():
    items = [{k: v for k, v in run.items()} for run in _runs()
             if run["split"] == "TRAIN"]
    for test in ia.HELD_OUT_TESTS:
        split = ia.held_out_split(items, test)
        assert (sorted(_arms(split["train"]) + _arms(split["eval"]))
                == sorted(_arms(items)))
        assert not (set(_arms(split["train"])) & set(_arms(split["eval"])))
        assert split["train"] is not split["eval"]
        for it in split["train"]:
            assert it["split"] == "TRAIN"


def test_real_j4_batch_selects_only_registered_mid_arms():
    """The real pack uses ordinary world IDs; batch identifies the J4 law."""
    world = "w8623m2600p02t540"
    other = "w8659m2700p10t420"
    items = [
        dict(batch="chunk4", chunk="chunk4", world=world,
             arm="native", policy="native", split="TRAIN"),
        dict(batch="walls_pilot_chunk4", chunk="chunk4", world=world,
             arm="W-mid-early", policy="wall", split="TRAIN"),
        dict(batch="walls_pilot_chunk4", chunk="chunk4", world=other,
             arm="W-mid-late", policy="wall", split="TRAIN"),
        dict(batch="walls_pilot_chunk4", chunk="chunk4", world=other,
             arm="W-lo-early", policy="wall", split="TRAIN"),
        dict(batch="chunk4", chunk="chunk4", world=other,
             arm="W-mid-early", policy="wall", split="TRAIN"),
    ]
    result = ia.held_out_split(items, "iii_leave_top_wall_speed_out")
    assert result["eval"] == [items[1], items[2]]
    assert result["train"] == [items[0], items[3], items[4]]
    assert not ({id(it) for it in result["train"]} &
                {id(it) for it in result["eval"]})
    assert [it["world"] for it in result["eval"]] == [world, other]


@pytest.mark.parametrize("bad", [
    {"chunk": "chunk3"},
    {"split": "CALIBRATION"},
    {"policy": "native"},
    {"arm": "wall_v31"},
    {"wall_speed_factor": 0.60},
    {"world": ""},
])
def test_j4_batch_contradictory_metadata_refused(bad):
    item = dict(batch="walls_pilot_chunk4", chunk="chunk4",
                world="w8623m2600p02t540", arm="W-mid-early",
                policy="wall", split="TRAIN")
    item.update(bad)
    with pytest.raises(ia.IdeaAConfigError):
        ia.held_out_split([item], "iii_leave_top_wall_speed_out")


@pytest.mark.parametrize("missing", ["chunk", "split"])
def test_j4_batch_missing_provenance_refused(missing):
    item = dict(batch="walls_pilot_chunk4", chunk="chunk4",
                world="w8623m2600p02t540", arm="W-mid-early",
                policy="wall", split="TRAIN")
    del item[missing]
    with pytest.raises(ia.IdeaAConfigError):
        ia.held_out_split([item], "iii_leave_top_wall_speed_out")


def test_j4_mid_absent_refused_instead_of_selecting_lo():
    item = dict(batch="walls_pilot_chunk4", chunk="chunk4",
                world="w8659m2700p10t420", arm="W-lo-early",
                policy="wall", split="TRAIN")
    with pytest.raises(ia.IdeaAConfigError, match="registered factor"):
        ia.held_out_split([item], "iii_leave_top_wall_speed_out")


def test_j4_alias_without_batch_refused():
    item = dict(world="w8623m2600p02t540", arm="W-mid-early", policy="wall",
                split="TRAIN")
    with pytest.raises(ia.IdeaAConfigError, match="batch identity"):
        ia.held_out_split([item], "iii_leave_top_wall_speed_out")


def _synthetic_inventory_rows():
    """Inventory schema uses raw chunk '4', unlike packed chunk 'chunk4'."""
    world = "w8623m2600p02t540"
    return [
        dict(batch="walls_pilot_chunk4", chunk="4", world=world,
             split="TRAIN", arm="W-mid-early", policy="wall", run="j4-mid"),
        dict(batch="walls_pilot_chunk4", chunk="4", world=world,
             split="TRAIN", arm="W-lo-late", policy="wall", run="j4-lo"),
        dict(batch="chunk4", chunk="4", world=world,
             split="TRAIN", arm="native", policy="native", run="native"),
        dict(batch="chunk4", chunk="4", world=world,
             split="TRAIN", arm="pi", policy="pi", run="pi"),
        dict(batch="chunk4", chunk="4", world=world,
             split="TRAIN", arm="W-mid-early", policy="wall",
             run="not-j4-wall"),
        dict(batch="chunk4", chunk="4", world="w8624m2600p02t540",
             split="CALIBRATION", arm="pi", policy="pi", run="cal-pi"),
    ]


def test_g6_inventory_preserves_provenance_and_scopes_j4b(monkeypatch,
                                                             tmp_path):
    inventory = tmp_path / "synthetic-inventory.parquet"
    inventory.write_bytes(b"monkeypatched reader; no real parquet")
    rows = _synthetic_inventory_rows()

    def fake_read(path):
        assert Path(path) == inventory
        return pd.DataFrame(rows)

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    items = ia._heldout_inventory_items(inventory)
    assert len(items) == 5              # the CAL row is absent
    assert items[0]["world"] == rows[0]["world"]
    assert items[0]["batch"] == "walls_pilot_chunk4"
    assert items[0]["chunk"] == "4"     # actual inventory spelling
    assert items[0]["policy"] == "wall"
    assert items[0]["wall_speed_factor"] == 0.85
    assert not any(it.get("planned") for it in items)

    report = ia.heldout_split_report(inventory)
    splits = report["splits"]
    assert any(name.startswith("J4B::") for name in
               splits["heldout-i-j4b"]["eval"])
    for name in ("heldout-ii-leave-pi", "heldout-iii-leave-highest-wall-speed"):
        assert not any(value.startswith("J4B::") for value in
                       splits[name]["train"])
    assert splits["heldout-iii-leave-highest-wall-speed"]["eval"] == [
        f"{rows[0]['world']}::walls_pilot_chunk4::W-mid-early"]
    assert f"{rows[0]['world']}::chunk4::W-mid-early" in \
        splits["heldout-iii-leave-highest-wall-speed"]["train"]


@pytest.mark.parametrize("wrong", [
    {"chunk": "3"}, {"policy": "native"},
    {"arm": "wall_v31"}, {"world": ""},
])
def test_g6_inventory_rejects_contradictory_pilot_metadata(monkeypatch,
                                                             wrong):
    rows = _synthetic_inventory_rows()
    rows[0].update(wrong)
    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame(rows))
    with pytest.raises(ia.IdeaAConfigError):
        ia._heldout_inventory_items(Path("synthetic-only.parquet"))


@pytest.mark.parametrize("missing", ["chunk", "policy", "batch"])
def test_g6_inventory_requires_provenance_columns(monkeypatch, missing):
    rows = _synthetic_inventory_rows()
    for row in rows:
        del row[missing]
    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame(rows))
    with pytest.raises(ia.IdeaAConfigError, match="missing provenance"):
        ia._heldout_inventory_items(Path("synthetic-only.parquet"))


@pytest.mark.parametrize("empty", ["chunk", "policy", "world"])
def test_g6_inventory_requires_provenance_values(monkeypatch, empty):
    rows = _synthetic_inventory_rows()
    rows[0][empty] = None
    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame(rows))
    with pytest.raises(ia.IdeaAConfigError, match="empty provenance"):
        ia._heldout_inventory_items(Path("synthetic-only.parquet"))
