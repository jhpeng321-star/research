"""PackedEvalSource: real EvalSource rows over packed shards (Mismatch 5)."""
import json

import numpy as np
import pytest

import synth
from corpus import pack
from corpus import source as SRC


def _pack(tmp_path):
    data = tmp_path / "shards"
    synth.write_shards(data)
    out = tmp_path / "pack"
    pack.pack_shards(data, out, workers=1)
    return out


def test_outcome_rows_protocol_shape(tmp_path):
    out = _pack(tmp_path)
    src = SRC.PackedEvalSource(out)
    rows = src.outcome_rows("train_oof")
    assert rows
    r = rows[0]
    assert type(r).__name__ == "OutcomeRow"
    assert r.world in ("w1", "w2") and r.policy
    assert r.tau_s in synth.TAUS
    assert np.isfinite(r.y_delay_s)
    assert src.sign_threshold >= 1.0
    cal = src.outcome_rows("cal")
    assert {x.world for x in cal} == {"c1"}


def test_rollout_rows_protocol_shape(tmp_path):
    out = _pack(tmp_path)
    src = SRC.PackedEvalSource(out)
    rows = src.rollout_rows("train_oof")
    assert rows
    r = rows[0]
    assert type(r).__name__ == "RolloutRow"
    h = len(src.horizons_s)
    assert r.true_pos.shape[0] == h and r.true_speed.shape[0] == h
    assert r.true_pos.shape == r.true_speed.shape
    assert np.isfinite(r.true_pos).all()


def test_final_test_refused(tmp_path):
    out = _pack(tmp_path)
    src = SRC.PackedEvalSource(out)
    with pytest.raises(SRC.PackedSourceRefusal):
        src.outcome_rows("final_test")


def test_sealed_pack_refused(tmp_path):
    out = tmp_path / "pack"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({
        "schema": "wm_n1_pack/1", "k_max": 4, "arrays": {},
        "outcomes": {"file": "outcomes.parquet"},
        "items": [{"world": "hnd1", "arm": "native", "chunk": "chunk4",
                   "split": "HOLDOUT", "t0_s": 0.0, "pair_key": "x",
                   "tick": 0, "plan_row": 0, "outcome_rows": []}]}),
        encoding="utf-8")
    with pytest.raises(SRC.PackedSourceRefusal):
        SRC.PackedEvalSource(out)
