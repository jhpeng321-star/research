"""A9 composite identity: pack-local IDs re-project to one shared space."""
from __future__ import annotations

import os

import numpy as np
import pytest

import train.idea_a as ia


def test_remap_canonical_ids_aligns_pack_local_spaces():
    # pack A local ids: 0 -> (chunk,wa,v0), 1 -> (chunk,wa,v1)
    a_map = [("chunk", "wa", "v0"), ("chunk", "wa", "v1")]
    # pack B local ids: 0 -> (chunk,wa,v1), 1 -> (chunk,wa,v2)
    b_map = [("chunk", "wa", "v1"), ("chunk", "wa", "v2")]
    lookup = {k: i for i, k in enumerate(sorted(set(a_map) | set(b_map)))}
    a = ia.remap_canonical_ids(np.asarray([0, 1, -1]), a_map, lookup)
    b = ia.remap_canonical_ids(np.asarray([-1, 0, 1]), b_map, lookup)
    # (chunk,wa,v1) is local 1 in A and local 0 in B -> SAME shared id.
    assert a.tolist() == [lookup[("chunk", "wa", "v0")],
                          lookup[("chunk", "wa", "v1")], -1]
    assert b[1] == a[1]
    assert b[0] == -1 and b[2] == lookup[("chunk", "wa", "v2")]


def test_extra_identity_dirs_must_match_extra_packs(monkeypatch):
    corpus = ia.IdeaACorpus.__new__(ia.IdeaACorpus)
    corpus._is_multi_pack = True
    corpus._extra_pack_dirs = ["A", "B"]
    with pytest.raises(ia.IdeaAConfigError):
        corpus._resolve_extra_identity_dirs(["only-one"])
    monkeypatch.setenv("WM_N1_EXTRA_TRAIN_IDENTITIES",
                       "p1" + os.pathsep + "p2")
    assert corpus._resolve_extra_identity_dirs([]) == ["p1", "p2"]
    assert corpus._resolve_extra_identity_dirs(["one", "two"]) == ["one", "two"]


def test_component_position_uses_unshifted_per_pack_gpos():
    corpus = ia.IdeaACorpus.__new__(ia.IdeaACorpus)
    corpus._is_multi_pack = True
    corpus.pack_dir = "P"
    assert corpus._component_position(
        dict(_pack=1, _manifest_gpos=7)) == (1, 7)
    with pytest.raises(ia.IdeaAConfigError):
        corpus._component_position(dict(_pack=1, _manifest_gpos=None))


def test_single_pack_component_position_uses_manifest_index():
    corpus = ia.IdeaACorpus.__new__(ia.IdeaACorpus)
    corpus._is_multi_pack = False
    corpus.pack_dir = "P"
    item = dict(chunk="c", world="w", arm="a", t0_s=1.0, tick=2, plan_row=3)
    corpus._manifest_pos = {ia.IdeaACorpus._item_position_key(item): 11}
    assert corpus._component_position(item) == (0, 11)
    corpus._manifest_pos = {}
    with pytest.raises(ia.IdeaAConfigError):
        corpus._component_position(item)
