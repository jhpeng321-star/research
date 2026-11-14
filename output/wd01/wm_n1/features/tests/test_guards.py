"""Split guard, forbidden-file open guard, and deployment-honesty greps."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from features import featurizer as F
from features import splits as S
from synth_features import MAIN_RIGHT, plan_row, row, veh, write_run

FEATURES_DIR = Path(__file__).resolve().parents[1]


def _plan(tmp_path, worlds):
    p = tmp_path / "WORLD_PLAN.json"
    p.write_text(json.dumps({"worlds": worlds}), encoding="utf-8")
    return p


def test_holdout_refused_from_split_guard(tmp_path):
    plan = _plan(tmp_path, [dict(world="wHold", split="HOLDOUT", t0_s=300.0)])
    with pytest.raises(S.SplitRefusal):
        S.decide_split("chunk4", "wHold", plan_path=plan)


def test_train_and_calibration_allowed(tmp_path):
    plan = _plan(tmp_path, [dict(world="wT", split="TRAIN", t0_s=300.0),
                            dict(world="wC", split="CALIBRATION", t0_s=300.0)])
    assert S.decide_split("chunk4", "wT", plan_path=plan) == "TRAIN"
    assert S.decide_split("chunk4", "wC", plan_path=plan) == "CALIBRATION"


def test_excluded_root_refused(tmp_path):
    plan = _plan(tmp_path, [dict(world="wE", split="TRAIN", t0_s=300.0)])
    ex = {"excluded": [{"world": "wE"}]}
    with pytest.raises(S.SplitRefusal):
        S.decide_split("chunk4", "wE", plan_path=plan, exclusions=ex)


def test_chunk5_gate(tmp_path):
    with pytest.raises(S.SplitRefusal):
        S.decide_split("chunk5", "wFresh")
    with pytest.raises(S.SplitRefusal):
        S.decide_split("chunk5", "wFresh", final_test=True,
                       registration_sha256="deadbeef")
    assert S.decide_split("chunk5", "wFresh", final_test=True,
                          registration_sha256=S.registration_sha()) == "TEST"


def test_chunk5_gate_binds_frozen_chunk5_registration(tmp_path):
    """The gate binds the chunk-5 registration (not the WM-N1 one), CRLF-safe."""
    assert "REGISTRATION_CHUNK5_TEST.md" in S.REGISTRATION_CHUNK5.name
    assert S.registration_sha() == S.CHUNK5_REGISTRATION_LF_SHA256
    assert S.decide_split("chunk5", "wFresh", final_test=True,
                          registration_sha256=S.CHUNK5_REGISTRATION_LF_SHA256) == "TEST"
    # a CRLF copy of the LF-canonical text hashes to the same value
    lf = (S.REGISTRATION_CHUNK5.read_bytes()
          .replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    crlf = tmp_path / "crlf.md"
    crlf.write_bytes(lf.replace(b"\n", b"\r\n"))
    assert S.lf_sha256_file(crlf) == S.CHUNK5_REGISTRATION_LF_SHA256
    # the WM-N1 registration value is a different, reported constant
    assert S.wm_n1_registration_sha() != S.CHUNK5_REGISTRATION_LF_SHA256


def test_split_decided_before_any_run_open(tmp_path, monkeypatch):
    """A HOLDOUT refusal must fire without opening a run file."""
    plan = _plan(tmp_path, [dict(world="wHold", split="HOLDOUT", t0_s=300.0)])
    empty_runs = tmp_path / "runs"
    empty_runs.mkdir()
    with pytest.raises(S.SplitRefusal):
        # runs root is empty: if the guard opened a run it would FileNotFound
        S.decide_split("chunk4", "wHold", plan_path=plan)
        F.build_shard("chunk4", "wHold", "base", empty_runs / "wHold__base",
                      split="TRAIN", plan_row=plan_row(300.0))


def test_no_forbidden_reads_functional(tmp_path):
    rows = [row(0.5, {"mainline.0": veh(MAIN_RIGHT, 10, 30)}),
            row(1.0, {"mainline.0": veh(MAIN_RIGHT, 10.5, 30)})]
    root = write_run(tmp_path / "base", rows)      # contains decoy files
    assert (root / "run" / "dispatcher_rows.jsonl").is_file()
    assert (root / "run" / "insertions.jsonl").is_file()
    S.OpenAudit.reset()
    S.OpenAudit.start()
    try:
        F.build_shard("chunk4", "wsynth", "base", root, split="TRAIN",
                      plan_row=plan_row(0.0), outcomes_root=None)
    finally:
        S.OpenAudit.stop()
    opened = S.OpenAudit.opened_names()
    assert "dispatcher_rows.jsonl" not in opened
    assert "insertions.jsonl" not in opened
    assert "physics_steps.jsonl" in opened


def test_static_no_forbidden_filenames():
    hits = []
    for p in FEATURES_DIR.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        for needle in ("dispatcher_rows", "insertions.jsonl"):
            if needle in text:
                hits.append(f"{p.name}:{needle}")
    assert not hits, hits
