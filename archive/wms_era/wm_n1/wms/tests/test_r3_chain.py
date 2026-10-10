"""ROUND-3 chain contract checks (read-only string/format assertions)."""
from __future__ import annotations

from pathlib import Path

CHAIN = Path(__file__).resolve().parents[5] / "_out" / "train_r3_chain.sh"


def _text() -> str:
    return CHAIN.read_text(encoding="utf-8")


def test_chain_exists_and_is_lf_only():
    assert CHAIN.is_file()
    assert b"\r" not in CHAIN.read_bytes()
    assert _text().startswith("#!/usr/bin/env bash")


def test_chain_keeps_guards_and_box_opt_outs():
    text = _text()
    for token in ("SMOKE_CONTENTION_GUARD", "RSS_GUARD_BYTES", "MIN_FREE_OVERRIDE",
                  "guarded_train", "host_check", "plan_steps", "gate_stages",
                  "MIN_FREE_BYTES"):
        assert token in text, token


def test_chain_defines_both_presets_with_the_r3_braking():
    text = _text()
    assert "R3_PRESET=${R3_PRESET:-R3_FULL}" in text
    assert "R3_FULL)" in text and "R3_CTRL)" in text
    assert "${ASYM_SPEED_LOSS:=2}" in text and "${DECEL_WEIGHT:=5}" in text
    assert "${PROPAGATION_LOSS:=1}" in text
    assert "${ASYM_SPEED_LOSS:=1}" in text and "${DECEL_WEIGHT:=1}" in text
    assert "${PROPAGATION_LOSS:=0}" in text
    assert '--decel-threshold' in text and "${DECEL_THRESHOLD:-3}" in text


def test_chain_has_the_r3_data_and_flags():
    text = _text()
    for token in ("_out/lpwo_v3/windows_train_noPI_noWall.json",
                  "_out/lpwo_v3/windows_val_noPI_noWall.json",
                  "_out/lpwo_v3/excluded_runs_PI_wall_v3.json",
                  "_out/sidecars_packv3_v2/identities",
                  "_out/sidecars_packv3_v2/outcomes",
                  "SWITCH_MIX_RATIO=${SWITCH_MIX_RATIO:-0.3}",
                  "EVENT_OVERSAMPLE=${EVENT_OVERSAMPLE:-1}",
                  "START_SHIFT_PROB=${START_SHIFT_PROB:-0.75}",
                  "--switch-mix-ratio", "--event-oversample", "--start-shift-prob",
                  "--merge-cal", "--merge-cal-params"):
        assert token in text, token


def test_chain_exposes_jam_and_pass_through_hooks():
    text = _text()
    for token in ("JAM_OVERSAMPLE=${JAM_OVERSAMPLE:-1}",
                  "JAM_SPEED_MPS=${JAM_SPEED_MPS:-5}",
                  "--jam-oversample", "--jam-speed-mps", "--jam-cache-dir",
                  "R3_RESIDUAL_SOFT_BOUND_MPS2", "--residual-soft-bound-mps2",
                  "R3_RESIDUAL_MAG_WEIGHT", "--residual-mag-weight",
                  "R3_LANE_RATE_WEIGHT", "--lane-rate-weight",
                  "R3_LANE_RATE_TARGET", "--lane-rate-target",
                  "R3_SELECT_BY_LONG_HORIZON", "--select-by-long-horizon",
                  "R3_HISTORY_S", "--history-s",
                  "R3_HISTORY_STRIDE_S", "--history-stride-s",
                  "ENTRIES_V2=${ENTRIES_V2:-0}", "--entries-v2",
                  "WM_S_STAGE_CKPT_COPY_DIR"):
        assert token in text, token


def test_chain_required_terms_follow_enabled_terms():
    text = _text()
    # traffic/stability/conservation are appended only when their weight is on.
    assert "if float(traffic) == 1:" in text
    assert "if float(stability) > 0:" in text
    assert "if float(conservation) > 0:" in text
    assert "terms = ['position', 'speed', 'lane']" in text
    assert '--require-terms "$TERMS"' in text
