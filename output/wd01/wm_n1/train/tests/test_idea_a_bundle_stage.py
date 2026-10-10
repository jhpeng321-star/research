"""P7 bundle stage-list additions: ``idea-a-held-out`` + ``idea-a-ensemble``.

INTERFACES_V2 assigns the bundle stage-list additions to P7.  Both stages are
CPU-only, idempotent with the existing queue, and never train anything.
"""
from __future__ import annotations

import json

import pytest

from bundle import queue
from test_idea_a_heldout import _runs
from train.tests import packsynth

_V1_STAGES = ["verify-upload", "convert+verify", "timing-gate", "approval",
              "member-1", "members-2-5"]


def _seed_state(work_dir, stages):
    work_dir.mkdir(parents=True, exist_ok=True)
    state = {"schema": "wm_n1_bundle_state/1",
             "completed": {s: {"stage": s, "ok": True, "completed_utc": "seed"}
                           for s in stages}}
    (work_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")


def _mk_members(work_dir, members):
    mdir = work_dir / "members"
    mdir.mkdir(parents=True, exist_ok=True)
    for m in members:
        (mdir / f"member_{m}.pt").write_bytes(b"ckpt")


def test_stage_order_and_default_stop_unchanged(tmp_path):
    assert queue.STAGE_ORDER[:6] == tuple(_V1_STAGES)
    assert queue.STAGE_ORDER[6:] == ("idea-a-ensemble", "idea-a-held-out")


def test_stage_idea_a_held_out_runs_on_packed_corpus(tmp_path):
    packsynth.write_pack(tmp_path / "pack", _runs())
    _seed_state(tmp_path, _V1_STAGES + ["idea-a-ensemble"])
    receipt = queue.run_stage("idea-a-held-out", work_dir=tmp_path,
                              device="cpu")
    assert receipt["ok"] is True
    from train.idea_a import HELD_OUT_TESTS

    assert set(receipt["tests"]) == set(HELD_OUT_TESTS)
    assert receipt["tests"]["ii_leave_pi_out"]["n_eval"] == 2
    # idempotent: a completed stage is returned, not re-run
    again = queue.run_stage("idea-a-held-out", work_dir=tmp_path,
                            device="cpu")
    assert again.get("skipped") is True


def test_stage_idea_a_ensemble_full_and_partial(tmp_path):
    _seed_state(tmp_path, _V1_STAGES)
    _mk_members(tmp_path, [1, 2, 3, 4, 5])
    receipt = queue.run_stage("idea-a-ensemble", work_dir=tmp_path,
                              device="cpu")
    assert receipt["ok"] is True and receipt["ensemble_k"] == 5
    assert receipt["final_test_eligible"] is True
    assert receipt["members"] == [1, 2, 3, 4, 5]

    partial = tmp_path / "partial"
    _seed_state(partial, _V1_STAGES)
    _mk_members(partial, [1, 2, 3, 4])
    with pytest.raises(queue.BundleStageError):
        queue.run_stage("idea-a-ensemble", work_dir=partial, device="cpu")
    state = json.loads((partial / "state.json").read_text(encoding="utf-8"))
    assert "idea-a-ensemble" not in state["completed"]


def test_stage_idea_a_ensemble_without_checkpoints_refused(tmp_path):
    _seed_state(tmp_path, _V1_STAGES)
    with pytest.raises(queue.BundleStageError):
        queue.run_stage("idea-a-ensemble", work_dir=tmp_path, device="cpu")


def test_new_stages_require_predecessors(tmp_path):
    with pytest.raises(queue.BundleOrderError):
        queue.run_stage("idea-a-held-out", work_dir=tmp_path, device="cpu")
    _seed_state(tmp_path, _V1_STAGES[:-1])      # members-2-5 missing
    with pytest.raises(queue.BundleOrderError):
        queue.run_stage("idea-a-ensemble", work_dir=tmp_path, device="cpu")
