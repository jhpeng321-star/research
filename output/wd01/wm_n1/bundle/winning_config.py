"""``WINNING_CONFIG.json`` schema + writer (Amendment 4 A4.2).

The member-1-first improvement loop runs at most 3 rounds of at most 4 a-priori
knob configurations (round 0 is the registered default) and selects ONE winner
by the CAL-tune mean paired dV CRPS.  This module writes the self-describing
record that ``APPROVAL_A_M2_5.json`` binds:

* ``knobs`` -- the four A4.2 knobs (learning-rate multiplier, epochs
  multiplier, multi-step rollout-loss weight, paired-contrast-loss weight);
* ``rounds`` -- the per-round configs and their CAL-tune scores (losers
  included, A4.3);
* ``selected`` -- the winning config id, and ``selection_metric``;
* the frozen ``amendment_4_lf_sha256`` / ``cal_split_lf_sha256`` bindings;
* an internal canonical ``sha256`` so drift is detectable.

The *raw-bytes* sha256 of the written file is what ``require_a_m2_5`` binds
(``winning_config_sha256``); :func:`write_approval_a_m2_5` produces that owner
file from a written winner and the completed member-1 checkpoint.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from evalh import cal_split

WINNING_CONFIG_SCHEMA = "wm_n1_winning_config/1"
WINNING_CONFIG_FILENAME = "WINNING_CONFIG.json"

#: A4.2 a-priori knob grids (multiples of the registered default).
LEARNING_RATE_GRID = (0.5, 1.0, 2.0)
EPOCHS_GRID = (1.0, 1.5)
ROLLOUT_LOSS_WEIGHT_GRID = (0.5, 1.0, 2.0)
CONTRAST_LOSS_WEIGHT_GRID = (0.5, 1.0, 2.0)

KNOB_GRIDS = {
    "learning_rate": LEARNING_RATE_GRID,
    "epochs": EPOCHS_GRID,
    "rollout_loss_weight": ROLLOUT_LOSS_WEIGHT_GRID,
    "contrast_loss_weight": CONTRAST_LOSS_WEIGHT_GRID,
}
KNOB_KEYS = tuple(KNOB_GRIDS)

SELECTION_METRIC = ("CAL-tune mean paired dV CRPS over the gating contrasts "
                    "at tau in {60, 120, 300}, subject to 10-90 coverage in "
                    "[0.75, 0.85] (amendment A4.2)")

MAX_ROUNDS = 3
MAX_CONFIGS_PER_ROUND = 4


class WinningConfigError(RuntimeError):
    """The winning-config record is malformed or drifted."""


def _raw_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(record: dict) -> str:
    """sha256 of the canonical JSON of ``record`` minus its own sha256 field."""
    payload = {k: v for k, v in record.items() if k != "sha256"}
    blob = json.dumps(payload, sort_keys=True, indent=1)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _validate_knobs(knobs: dict) -> dict:
    if not isinstance(knobs, dict):
        raise WinningConfigError("knobs must be an object")
    out = {}
    for key in KNOB_KEYS:
        if key not in knobs:
            raise WinningConfigError(f"knobs missing {key!r}")
        val = float(knobs[key])
        if val not in tuple(float(g) for g in KNOB_GRIDS[key]):
            raise WinningConfigError(
                f"knob {key}={val!r} is not in the A4.2 grid "
                f"{KNOB_GRIDS[key]}")
        out[key] = val
    extra = sorted(set(knobs) - set(KNOB_KEYS))
    if extra:
        raise WinningConfigError(
            f"unknown knob(s) {extra}; A4.2 freezes exactly {list(KNOB_KEYS)}")
    return out


def _validate_rounds(rounds) -> list:
    if not isinstance(rounds, list) or not rounds:
        raise WinningConfigError("rounds must be a non-empty list")
    if len(rounds) > MAX_ROUNDS:
        raise WinningConfigError(
            f"{len(rounds)} rounds > A4.2 cap of {MAX_ROUNDS}")
    out = []
    for rec in rounds:
        if not isinstance(rec, dict):
            raise WinningConfigError("each round must be an object")
        n = int(rec.get("round", -1))
        configs = list(rec.get("configs") or [])
        scores = rec.get("cal_tune_scores") or {}
        if n < 0:
            raise WinningConfigError(f"round {rec!r} has no non-negative round")
        if not configs:
            raise WinningConfigError(f"round {n} lists no configs")
        if len(configs) > MAX_CONFIGS_PER_ROUND:
            raise WinningConfigError(
                f"round {n} has {len(configs)} configs > A4.2 cap of "
                f"{MAX_CONFIGS_PER_ROUND}")
        out.append(dict(rec, round=n, configs=configs,
                        cal_tune_scores=scores))
    return out


def build_winning_config(*, selected: str, knobs: dict, rounds,
                         cal_tune_scores=None,
                         amendment_4_lf_sha256: str = None,
                         cal_split_lf_sha256: str = None,
                         selection_metric: str = SELECTION_METRIC,
                         seed: int = 20261001, rounds_log=None) -> dict:
    """Build the record (with its internal ``sha256``), validating A4.2 grids.

    ``cal_tune_scores`` is the flat winner score map; ``rounds`` carries the
    per-round log (losers included).  ``rounds_log`` is an accepted alias for
    the round log used by some callers.  The winner must be one of the
    configs named in the rounds.
    """
    knobs = _validate_knobs(knobs)
    rr = _validate_rounds(rounds if rounds is not None else rounds_log)
    known = {c for rec in rr for c in rec["configs"]}
    if str(selected) not in known:
        raise WinningConfigError(
            f"selected {selected!r} is not among the logged configs {sorted(known)}")
    record = {
        "schema": WINNING_CONFIG_SCHEMA,
        "amendment_4_lf_sha256": (amendment_4_lf_sha256
                                  or cal_split.LF_SHA_AMENDMENT_4),
        "cal_split_lf_sha256": (cal_split_lf_sha256
                                or cal_split.LF_SHA_CAL_SPLIT_A4),
        "seed": int(seed),
        "knobs": knobs,
        "rounds": rr,
        "cal_tune_scores": dict(cal_tune_scores or {}),
        "selected": str(selected),
        "selection_metric": str(selection_metric),
    }
    record["sha256"] = canonical_sha(record)
    return record


def write_winning_config(path, record: dict) -> str:
    """Write ``record`` (recomputing its internal sha) and return its raw sha."""
    out = {k: v for k, v in record.items() if k != "sha256"}
    out["sha256"] = canonical_sha(out)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, sort_keys=True, indent=1), encoding="utf-8")
    return _raw_sha(p)


def verify_winning_config(path) -> dict:
    """Re-read a written winner; raise on schema/grid/sha drift."""
    p = Path(path)
    if not p.is_file():
        raise WinningConfigError(f"winning config not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                                # noqa: BLE001
        raise WinningConfigError(f"winning config is not JSON: {p}: {e}") from e
    if not isinstance(data, dict):
        raise WinningConfigError(f"winning config is not an object: {p}")
    if data.get("schema") != WINNING_CONFIG_SCHEMA:
        raise WinningConfigError(
            f"winning config schema {data.get('schema')!r} != "
            f"{WINNING_CONFIG_SCHEMA!r}")
    got = canonical_sha(data)
    if got != data.get("sha256"):
        raise WinningConfigError(
            f"winning config sha256 {data.get('sha256')} != recomputed {got}")
    _validate_knobs(data.get("knobs") or {})
    _validate_rounds(data.get("rounds") or [])
    return data


def write_approval_a_m2_5(path, *, winning_config_path, m1_checkpoint_path,
                          owner: str = "owner", approved_utc: str = None,
                          stage: str = "members-2-5") -> dict:
    """Write ``APPROVAL_A_M2_5.json`` bound to the winner + member-1 ckpt.

    Binds the registration/amendment-1/amendment-4 + amendment-6 (state
    global token) + CAL-split LF-shas together with the raw-bytes sha256 of
    both the completed member-1 checkpoint and the written
    ``WINNING_CONFIG.json``.  This is the canonical way to produce the A_M2_5
    owner file, so the winner schema is used rather than restated.
    """
    from . import approval as A

    rec = verify_winning_config(winning_config_path)
    if rec.get("amendment_4_lf_sha256") != A.LF_SHA_AMENDMENT_4:
        raise WinningConfigError(
            "winning config amendment_4_lf_sha256 does not match the frozen "
            f"Amendment 4 LF-sha256 {A.LF_SHA_AMENDMENT_4}")
    if rec.get("cal_split_lf_sha256") != A.LF_SHA_CAL_SPLIT_A4:
        raise WinningConfigError(
            "winning config cal_split_lf_sha256 does not match the frozen "
            f"CAL split LF-sha256 {A.LF_SHA_CAL_SPLIT_A4}")
    m1 = Path(m1_checkpoint_path)
    if not m1.is_file():
        raise WinningConfigError(f"member-1 checkpoint not found: {m1}")
    body = {
        "schema": A.APPROVAL_A_M2_5_SCHEMA,
        "approved": True,
        "owner": owner,
        "approved_utc": approved_utc or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "stage": stage,
        "registration_lf_sha256": A.LF_SHA_WM_N1,
        "amendment_lf_sha256": A.LF_SHA_AMENDMENT_1,
        "amendment_4_lf_sha256": A.LF_SHA_AMENDMENT_4,
        "amendment_6_lf_sha256": A.LF_SHA_AMENDMENT_6,
        "cal_split_lf_sha256": A.LF_SHA_CAL_SPLIT_A4,
        "m1_checkpoint_sha256": _raw_sha(m1),
        "winning_config_sha256": _raw_sha(winning_config_path),
        "winning_config_selected": rec.get("selected"),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body, sort_keys=True, indent=1), encoding="utf-8")
    return body


__all__ = [
    "WINNING_CONFIG_SCHEMA", "WINNING_CONFIG_FILENAME", "KNOB_GRIDS",
    "KNOB_KEYS", "LEARNING_RATE_GRID", "EPOCHS_GRID",
    "ROLLOUT_LOSS_WEIGHT_GRID", "CONTRAST_LOSS_WEIGHT_GRID",
    "SELECTION_METRIC", "MAX_ROUNDS", "MAX_CONFIGS_PER_ROUND",
    "WinningConfigError", "canonical_sha", "build_winning_config",
    "write_winning_config", "verify_winning_config", "write_approval_a_m2_5",
]
