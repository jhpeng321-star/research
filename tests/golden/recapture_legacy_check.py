"""Prove the CURRENT wms/wms_eval code reproduces the stored golden artifacts.

Slow (~15-25 min on 2 CPU threads): it re-captures everything into a temporary
directory and compares with tests/golden/data.  Restrict with
``GOLDEN_PARTS=selection,batches`` (comma list of selection,batches,forward,
losses,train,eval).  Run ``pytest tests/golden -q`` twice to confirm determinism.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import capture_golden as cg  # noqa: E402
import golden_compare as gc  # noqa: E402

PARTS = tuple(p for p in os.environ.get("GOLDEN_PARTS", ",".join(cg.ALL_PARTS)).split(",") if p)
pytestmark = pytest.mark.skipif(not (gc.DATA / "meta.json").exists(),
                                reason="golden data not captured")

BATCH_FILES = [cg.batch_variant_name(u, h) for u in (10, 60) for h in (True, False)]


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    out = tmp_path_factory.mktemp("golden_fresh")
    cg.capture(out, PARTS)
    return out


def _need(part):
    if part not in PARTS:
        pytest.skip(f"part {part} not selected")


def _json(fresh, name, strict=True):
    errs = gc.compare_json(gc.load_json(fresh / name), name, strict_hashes=strict)
    assert not errs, "\n".join(errs[:20])


def _npz(fresh, name):
    errs = gc.compare_npz(gc.load_npz(fresh / name), name)
    assert not errs, "\n".join(errs[:20])


def test_selection_counts_and_keys(fresh):
    _need("selection")
    _json(fresh, "selection.json")
    stored = gc.load_json("selection.json")
    for split, n3 in (("train", 294), ("val", 47)):
        rows = stored["kept_keys"][split]
        assert sum(1 for r in rows if r[0] == "pack_v3") == n3


@pytest.mark.parametrize("tag", BATCH_FILES)
def test_decoded_batches(fresh, tag):
    _need("batches")
    _npz(fresh, f"{tag}.npz")
    _json(fresh, f"{tag}.json")


@pytest.mark.parametrize("name", ["forward_c1", "forward_r3"])
def test_model_forward(fresh, name):
    _need("forward")
    _npz(fresh, f"{name}.npz")
    _json(fresh, f"{name}.json")


def test_r3_init_state(fresh):
    _need("forward")
    _npz(fresh, "r3_init_state.npz")


def test_losses_and_gradients(fresh):
    _need("losses")
    _json(fresh, "losses_r3.json")


def test_three_training_steps(fresh):
    _need("train")
    _json(fresh, "train3_r3.json")
    _npz(fresh, "train3_final_params.npz")


def test_eval_rollouts(fresh):
    _need("eval")
    _json(fresh, "eval_rollouts.json")
    _npz(fresh, "eval_pred_frames.npz")
