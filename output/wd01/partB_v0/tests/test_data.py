"""Data-loader contract tests: canonical dataset (read-only) + synthetic.

Canonical counts are hardcoded from DATA_CARD_V0.md; the canonical tree
is never written to (the WD_FORBID_D hook in output/wd01/conftest.py
keeps every path on E:).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest_load import (CARD_MANIFEST_SHA, CARD_NATIVE_T570_TRAIN,
                           CARD_ROWS, CARD_SOURCES, CARD_TD)
from partB_v0.data import POLICY_OF_SOURCE, V0Dataset, load_manifest


def test_manifest_sha_and_verification(canonical_manifest):
    assert canonical_manifest.sha256 == CARD_MANIFEST_SHA
    assert canonical_manifest.files_verified
    assert len(canonical_manifest) == 222


def test_no_holdout_selected_or_requestable(canonical_manifest):
    assert len(canonical_manifest.entries) == 222
    assert all(e["split"] != "HOLDOUT"
               for e in canonical_manifest.entries.values())
    with pytest.raises(ValueError, match="HOLDOUT"):
        load_manifest(canonical_manifest.root, splits=("HOLDOUT",),
                      verify_files=False)


def test_row_counts_match_data_card(canonical_ds):
    r = canonical_ds.rows
    got = r.groupby(["chunk_id", "split", "source"]).size().to_dict()
    for key, n in CARD_ROWS.items():
        assert got[key] == n, key
    assert len(r) == sum(CARD_ROWS.values()) == 57854


def test_td_counts_match_data_card(canonical_ds):
    t = canonical_ds.td
    got = t.groupby(["chunk_id", "split"]).size().to_dict()
    for key, n in CARD_TD.items():
        assert got[key] == n, key
    assert len(t) == sum(CARD_TD.values()) == 78500


def test_source_counts_match_data_card(canonical_manifest):
    got = {}
    for e in canonical_manifest.entries.values():
        got[(e["chunk_id"], e["split"])] = \
            got.get((e["chunk_id"], e["split"]), 0) + 1
    for key, n in CARD_SOURCES.items():
        assert got[key] == n, key


def test_native_train_tau570_counts(canonical_ds):
    """The thin long-tau native rows the plan flags (141 + 147)."""
    r = canonical_ds.rows
    sel = r[(r["split"] == "TRAIN") & (r["source"] == "native")
            & np.isclose(r["tau"], 570.0)]
    got = sel.groupby("chunk_id").size().to_dict()
    assert got == CARD_NATIVE_T570_TRAIN


def test_policy_exposure_and_flags(canonical_ds):
    r = canonical_ds.rows
    m = r.groupby("source")["policy"].unique().to_dict()
    assert set(m["base"]) == {POLICY_OF_SOURCE["base"]}
    assert set(m["native"]) == {POLICY_OF_SOURCE["native"]}
    assert set(r["pre_fix"].unique()) == {True, False}   # chunk1 vs chunk2
    assert set(r["chunk_id"].unique()) == {"chunk1", "chunk2"}
    assert set(r["policy_id"].unique()) == {0, 1}


def test_row_weights_equalise_per_tau_policy(canonical_ds):
    r = canonical_ds.rows
    sums = r.groupby(["tau", "policy"])["row_weight"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-9)


def test_gather_and_td_batches(canonical_ds):
    b = canonical_ds.gather(np.arange(16))
    assert b["lane_bins"].shape == (16, 3, 320, 3)
    assert b["context"].shape[1] == 9          # 8 featurize + pre_fix
    assert b["tau"].shape == (16,)
    bt, bn, d30 = canonical_ds.gather_td(np.arange(16))
    assert bt["lane_bins"].shape == bn["lane_bins"].shape == \
        (16, 3, 320, 3)
    assert (bn["tau"] == bt["tau"] - 30.0).all()
    assert (d30 > 0).all()


def test_gather_tau_policy_override(canonical_ds):
    """The same decision-time position queried under both policies."""
    r = canonical_ds.rows
    base = r[(r["split"] == "CALIBRATION") & (r["source"] == "base")
             & np.isclose(r["tau"], 570.0)].iloc[0]
    g = [int(base["t_idx"])]
    for pol in (0, 1):
        b = canonical_ds.gather_global(g, [570.0], [pol],
                                       [float(base["pre_fix"])])
        assert b["policy"][0] == pol
        assert b["tau"][0] == 570.0


def test_superseded_excluded(tiny_dataset):
    """A MANIFEST entry marked superseded_by is skipped by default."""
    from conftest import make_tiny_dataset
    import json
    root, sha = tiny_dataset
    man = json.loads((root / "MANIFEST.json").read_text("utf-8"))
    key = sorted(man["worlds"])[0]
    man["worlds"][key]["superseded_by"] = "chunk1/some_rebuild__base"
    (root / "MANIFEST.json").write_text(json.dumps(man), encoding="utf-8")
    m_ex = load_manifest(root, verify_files=False, log=lambda *_: None)
    assert key not in m_ex.entries
    assert m_ex.n_superseded_skipped == 1
    m_in = load_manifest(root, verify_files=False,
                         exclude_superseded=False, log=lambda *_: None)
    assert key in m_in.entries
    assert make_tiny_dataset and sha    # sha unused here


def test_superseded_files_absent_tolerated_when_excluded(tiny_dataset):
    """v0.3 bundle contract: superseded sources' files are OMITTED (the
    MANIFEST entry stays for provenance).  Default load must verify and
    train fine without them; explicitly INCLUDING them fails with a
    clear error naming the missing file."""
    import json
    root, _ = tiny_dataset
    man = json.loads((root / "MANIFEST.json").read_text("utf-8"))
    key = sorted(man["worlds"])[0]
    man["worlds"][key]["superseded_by"] = "chunk1_fix27"
    (root / "MANIFEST.json").write_text(json.dumps(man), encoding="utf-8")
    for rel in man["worlds"][key]["files"]:
        p = root / rel
        assert p.exists()
        p.unlink()

    # default (exclude): verification passes without the absent files
    m = load_manifest(root, verify_files=True, log=lambda *_: None)
    assert key not in m.entries and m.files_verified
    assert m.n_superseded_skipped == 1
    ds = V0Dataset(m, log=lambda *_: None)
    assert len(ds.rows) > 0

    # explicit inclusion: clear error, not a bare FileNotFoundError
    with pytest.raises(ValueError, match="missing.*superseded_by"):
        load_manifest(root, verify_files=True, exclude_superseded=False,
                      log=lambda *_: None)


def test_sha_verification_catches_corruption(tiny_dataset):
    root, _ = tiny_dataset
    man = load_manifest(root, verify_files=True, log=lambda *_: None)
    target = root / sorted(next(iter(man.entries.values()))["files"])[0]
    raw = target.read_bytes()
    target.write_bytes(raw + b"x")             # corrupt one byte
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_manifest(root, verify_files=True, log=lambda *_: None)


def test_synthetic_dataset_loads(tiny_dataset):
    root, sha = tiny_dataset
    man = load_manifest(root, verify_files=True, log=lambda *_: None)
    assert man.sha256 == sha
    ds = V0Dataset(man, log=lambda *_: None)
    assert len(ds.rows) > 0 and len(ds.td) > 0
    # CRN pair present: base at t0 with tau=570, native at t0+10
    r = ds.rows
    b0 = r[np.isclose(r.t, r.t0_s) & np.isclose(r.tau, 570.0)
           & (r.source == "base")]
    n0 = r[np.isclose(r.t, r.t0_s + 10.0) & np.isclose(r.tau, 570.0)
           & (r.source == "native")]
    assert len(b0) == 5 and len(n0) == 5
