"""WM-S P5 unseen-controller (held-out-strategy) split builder (T8).

T8 of ``runs/active/WM-S-PREP-20261005`` prepares the held-out-strategy
training-window exclusions for the WM-S next-state model.  The design mirrors
the frozen held-out protocol of the WM-N1 Idea A model
(``REGISTRATION_WM_N1_AMENDMENT_3_IDEA_A.md`` A3.4 and
``WM_N1_IDEA_A_DESIGN.md`` section 11):

``i_unseen_wall_law_v22``
    train on the J4 wall panel (wall law v2.1 only) and evaluate the unseen
    wall law v2.2 (J4B).  J4B outcomes are NEVER read here: the evaluation set
    is materialised by mirroring the released J4 panel metadata and tagging it
    with the v2.2 law.  Reading J4B outcomes is a separate, registered step.
``ii_leave_pi_out``
    every PI arm is removed from training.
``iii_leave_factor_mid_out``
    the J4 wall arms of the a-priori highest speed factor ``mid`` (0.85) are
    removed (``HIGHEST_J4_WALL_SPEED_FACTOR``).

A held-out split is expressed as a list of ``(run, arm)`` keys to EXCLUDE from
the WM-S training windows plus an evaluation-set definition.  Evaluation is
defined on two released splits: the excluded arms that are present in the
TRAIN split (development) and, for the registered comparison, the same family
restricted to CAL-tune (``CAL_SPLIT_A4.json``).  Test (i) is the documented
exception: its registered evaluation is the sealed J4B (v2.2) panel.

Input records are window-index entries; DICT records must carry at least
``run`` (or ``world``+``arm``) and ``arm``, and may carry ``world``, ``split``,
``policy``, ``batch``, ``chunk``, ``wall_law``, ``wall_speed_factor`` and
``t0_s``.  The loader is field-alias tolerant so it can consume the window
index built by ws-p2data (T5) as soon as it lands.

This module reads metadata only.  It never opens CAL/HOLDOUT/chunk-5/J4B/INC1
outcome files and never starts SUMO.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

P5_SPLITS_SCHEMA = "wm_s_p5_splits/1"

#: The three registered held-out tests, in registration order.
HELD_OUT_TESTS = (
    "i_unseen_wall_law_v22",
    "ii_leave_pi_out",
    "iii_leave_factor_mid_out",
)

#: Frozen J4/J4B identities (mirrors output/wd01/wm_n1/train/idea_a.py).
J4_WALL_BATCH = "walls_pilot_chunk4"
J4_WALL_LAW = "wall-v2.1-closure"
J4B_WALL_LAW = "wall-v2.2"
J4_WALL_FACTORS = {"mid": 0.85, "lo": 0.60}
HIGHEST_J4_WALL_SPEED_FACTOR = "mid"

_SEALED_NEVER_READ = ("J4B", "CAL", "HOLDOUT", "chunk5", "INC1")

_J4_ARM_RE = re.compile(r"w[^a-z0-9]*(mid|lo)(?:[-_]|$)", re.IGNORECASE)
_WALL_FACTOR_RE = re.compile(r"wall[^0-9]*v?([0-9]+(?:\.[0-9]+)?)")

_MISSING = object()


# ------------------------------------------------------------------ records --
def _get(rec, *names):
    """First non-empty value among ``names`` on a dict/attribute record."""
    for name in names:
        if isinstance(rec, dict):
            if name in rec:
                value = rec[name]
            else:
                continue
        else:
            value = getattr(rec, name, _MISSING)
            if value is _MISSING:
                continue
        if value is None:
            continue
        if isinstance(value, float) and value != value:  # NaN
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _text(rec, *names) -> str:
    value = _get(rec, *names)
    return str(value).strip() if value is not None else ""


def record_arm(rec) -> str:
    """The arm name (``arm`` then policy fallback); ``""`` if absent."""
    return _text(rec, "arm", "policy_arm", "policy", "arm_name")


def record_world(rec) -> str:
    return _text(rec, "world", "world_id", "scenario", "cell")


def record_run(rec) -> str:
    run = _text(rec, "run", "run_id", "key")
    if run:
        return run
    world, arm = record_world(rec), record_arm(rec)
    if world and arm:
        return f"{world}__{arm}"
    return world or arm


def record_split(rec) -> str:
    raw = _text(rec, "split", "dataset_split", "set").upper()
    if raw in ("CAL", "CALIBRATION", "CAL_TUNE", "CAL_TUNE_HALF", "CAL-CAL"):
        return "CAL"
    if raw in ("CAL_CAL",):
        return "CAL_CAL"
    if raw.startswith("TRAIN"):
        return "TRAIN"
    return raw


def record_policy(rec) -> str:
    return _text(rec, "policy", "controller", "action").lower()


def wall_action_selected(rec) -> bool:
    """True when the row's own metadata selects a wall action.

    An explicit non-wall policy (native / PI / damping / ...) wins over a
    wall-shaped arm name; only without any policy field does the wall-shaped
    arm name decide.
    """
    policy = record_policy(rec)
    value = policy if policy else record_arm(rec).lower()
    return bool(value) and (value == "wall"
                            or value.startswith(("wall", "w-", "w_")))


def has_frozen_j4_law(rec) -> bool:
    """Frozen wall-law identity check: v2.1, never the v2.2 (J4B) law.

    An explicit law field must identify the frozen wall law v2.1 (a v2.2 or
    J4B identity always fails).  Without a law field the frozen J4 campaign
    label itself carries the v2.1 identity.
    """
    if is_j4b(rec):
        return False
    law = _text(rec, "wall_law", "law", "law_version", "controller_law").lower()
    if not law:
        return True
    return "v2.1" in law or ("j4" in law and "j4b" not in law)


def is_j4(rec) -> bool:
    """True for a J4 (wall law v2.1) pilot wall arm.

    All three frozen J4 identities must agree: the campaign label (or the
    chunk-4 wall-arm shape), the wall action actually selected by the row's
    policy, and the frozen wall law v2.1 (never v2.2).  A native or
    otherwise non-wall row of ``walls_pilot_chunk4`` is NOT J4.
    """
    campaign = _text(rec, "batch", "campaign", "batch_name") == J4_WALL_BATCH
    arm = record_arm(rec)
    chunk = _text(rec, "chunk", "chunk_label").lower().replace("chunk", "")
    panel = campaign or (chunk in ("4",)
                         and _J4_ARM_RE.search(arm) is not None)
    return panel and wall_action_selected(rec) and has_frozen_j4_law(rec)


def is_wall(rec) -> bool:
    arm, policy = record_arm(rec).lower(), record_policy(rec)
    for value in (policy, arm):
        if value == "wall" or value.startswith(("wall_", "wall-", "wallv")) \
                or value.startswith("w-") or value.startswith("w_"):
            return True
    return False


def is_pi(rec) -> bool:
    arm, policy = record_arm(rec).lower(), record_policy(rec)
    for value in (policy, arm):
        if value == "pi" or value.startswith(("pi_", "pi-")):
            return True
    return False


def wall_factor_label(rec) -> Optional[str]:
    """The wall speed factor label (``mid`` / ``lo``) of a wall arm."""
    explicit = _get(rec, "wall_speed_factor", "speed_factor", "wall_speed")
    if explicit is not None:
        try:
            value = float(explicit)
        except (TypeError, ValueError):
            value = None
        if value is not None:
            for label, factor in J4_WALL_FACTORS.items():
                if abs(value - factor) < 1e-9:
                    return label
    if is_j4(rec):
        m = _J4_ARM_RE.search(record_arm(rec))
        if m:
            return m.group(1).lower()
    for key in ("arm", "policy"):
        m = _WALL_FACTOR_RE.search(_text(rec, key).lower())
        if m:
            try:
                value = float(m.group(1))
            except ValueError:
                continue
            for label, factor in J4_WALL_FACTORS.items():
                if abs(value - factor) < 1e-9:
                    return label
    return None


def wall_law_of(rec) -> Optional[str]:
    law = _text(rec, "wall_law", "law", "law_version", "controller_law")
    if law:
        return law
    if is_j4(rec):
        return J4_WALL_LAW
    if record_world(rec).upper().startswith("J4B"):
        return J4B_WALL_LAW
    return None


def is_j4b(rec) -> bool:
    if record_world(rec).upper().startswith("J4B"):
        return True
    law = (_text(rec, "wall_law", "law", "law_version") or "").lower()
    return "v2.2" in law or "j4b" in law


def record_key(rec) -> tuple:
    """The ``(run, arm)`` exclusion key of a window record."""
    return (record_run(rec), record_arm(rec))


# -------------------------------------------------------------------- rules --
def heldout_tests() -> list:
    """The three registered WM-S held-out tests (metadata description)."""
    return [
        {
            "test": "i_unseen_wall_law_v22",
            "kind": "unseen_law",
            "train_rule": "J4 wall panel only (wall law v2.1)",
            "eval_dev": "the excluded (non-J4) TRAIN arms",
            "eval_registered": "J4B wall law v2.2 panel (SEALED; metadata mirror)",
            "reads_sealed": True,
        },
        {
            "test": "ii_leave_pi_out",
            "kind": "leave_policy",
            "train_rule": "remove every PI arm",
            "eval_dev": "the PI arms present in TRAIN",
            "eval_registered": "the PI arms of CAL-tune",
            "reads_sealed": False,
        },
        {
            "test": "iii_leave_factor_mid_out",
            "kind": "leave_wall_factor",
            "train_rule": "remove the J4 wall arms of factor 'mid' (0.85)",
            "eval_dev": "the 'mid' wall arms present in TRAIN",
            "eval_registered": "the 'mid' wall arms of CAL-tune",
            "reads_sealed": False,
        },
    ]


def _excluded(test: str, rec) -> bool:
    if test == "i_unseen_wall_law_v22":
        return not is_j4(rec)
    if test == "ii_leave_pi_out":
        return is_pi(rec)
    if test == "iii_leave_factor_mid_out":
        return is_wall(rec) and wall_factor_label(rec) \
            == HIGHEST_J4_WALL_SPEED_FACTOR
    raise ValueError(f"unknown held-out test {test!r}; expected {HELD_OUT_TESTS}")


def _split_of_records(records: Iterable[dict]) -> dict:
    train, cal = [], []
    for rec in records:
        split = record_split(rec)
        if split == "TRAIN":
            train.append(rec)
        elif split.startswith("CAL"):
            cal.append(rec)
    return {"train": train, "cal": cal}


def _dedupe(records: Iterable[dict]) -> list:
    """Deterministic, de-duplicated ``(run, arm)`` rows with window counts."""
    rows: dict = {}
    for rec in records:
        run, arm = record_key(rec)
        key = (run, arm)
        row = rows.get(key)
        if row is None:
            rows[key] = row = {
                "run": run, "arm": arm, "world": record_world(rec),
                "policy": record_policy(rec), "n_windows": 0,
            }
        row["n_windows"] += 1
    return [rows[k] for k in sorted(rows)]


def _count_excluded_in_train(train_records, excluded_keys) -> list:
    """Leak check: excluded ``(run, arm)`` keys appearing in the train windows."""
    leaks = Counter()
    for rec in train_records:
        if record_key(rec) in excluded_keys:
            leaks[record_key(rec)] += 1
    return [{"run": k[0], "arm": k[1], "n_windows": n}
            for k, n in sorted(leaks.items())]


def _j4b_mirror(j4_records) -> list:
    """Mirror the J4 panel as the sealed J4B (v2.2) evaluation set.

    Built from released J4 metadata only; no J4B file is opened.  The world/arm
    identities are the same panel re-run under wall law v2.2.
    """
    seen = {}
    for rec in j4_records:
        key = record_key(rec)
        if key in seen:
            continue
        seen[key] = {
            "run": key[0], "arm": key[1], "world": record_world(rec),
            "policy": "wall", "wall_law": J4B_WALL_LAW,
            "campaign": "campaign_walls_j4b", "sealed": True,
            "n_windows": None,
        }
    return [seen[k] for k in sorted(seen)]


def build_exclusions(records, test: str, *, cal_tune_worlds=None) -> dict:
    """Build one held-out split over an iterable of window-index records.

    Returns the TRAIN-window ``(run, arm)`` exclusion list, the TRAIN and
    CAL-tune evaluation-set definitions and the counting verification.  The
    train/eval sets are disjoint by construction, so the excluded arms are
    ABSENT from training.
    """
    if test not in HELD_OUT_TESTS:
        raise ValueError(f"unknown held-out test {test!r}; expected {HELD_OUT_TESTS}")
    records = list(records)
    parts = _split_of_records(records)
    train = [r for r in parts["train"] if not _excluded(test, r)]
    excluded_train = [r for r in parts["train"] if _excluded(test, r)]
    excluded_all = [r for r in records if _excluded(test, r)]
    excluded_keys = {record_key(r) for r in excluded_all}

    cal = parts["cal"]
    if cal_tune_worlds is not None:
        tune = set(cal_tune_worlds)
        cal_tune = [r for r in cal if record_world(r) in tune]
    else:
        cal_tune = cal

    leaks = _count_excluded_in_train(train, excluded_keys)

    dev_excluded = excluded_train
    if test == "i_unseen_wall_law_v22":
        j4_train = [r for r in parts["train"] if is_j4(r)]
        registered = {
            "split": "J4B (sealed)",
            "note": ("J4B wall law v2.2 panel mirrored from the released J4 "
                     "metadata; J4B outcome files are never opened by this task"),
            "n_runs": len(_j4b_mirror(j4_train)),
            "runs": _j4b_mirror(j4_train),
        }
    else:
        registered = {
            "split": "CAL_TUNE",
            "note": "CAL-tune arms of the held-out family (CAL_SPLIT_A4.json)",
            "n_runs": len(_dedupe([r for r in cal_tune if _excluded(test, r)])),
            "runs": _dedupe([r for r in cal_tune if _excluded(test, r)]),
        }

    return {
        "test": test,
        "kind": next(t["kind"] for t in heldout_tests() if t["test"] == test),
        "train_exclude": _dedupe(excluded_train),
        "n_train_windows": len(train),
        "n_train_exclude_windows": len(excluded_train),
        "n_index_windows": len(records),
        "eval": {
            "dev": {
                "split": "TRAIN",
                "note": "excluded arms present in the TRAIN split (development)",
                "n_runs": len(_dedupe(dev_excluded)),
                "runs": _dedupe(dev_excluded),
            },
            "registered": registered,
        },
        "verification": {
            "excluded_keys": len(excluded_keys),
            "leak_count": len(leaks),
            "leaks": leaks,
            "absent_from_train": not leaks,
        },
    }


def materialize(records, *, cal_tune_worlds=None, source: str = "") -> dict:
    """Build all three held-out splits plus the aggregate verification."""
    records = list(records)
    splits = {t: build_exclusions(records, t, cal_tune_worlds=cal_tune_worlds)
              for t in HELD_OUT_TESTS}
    all_zero = all(not s["verification"]["leaks"] for s in splits.values())
    return {
        "schema": P5_SPLITS_SCHEMA,
        "task": "T8/WM-S-PREP-20261005",
        "source": source,
        "design": ("WM_N1_IDEA_A_DESIGN.md section 11 / "
                   "REGISTRATION_WM_N1_AMENDMENT_3 A3.4"),
        "cal_tune_worlds": list(cal_tune_worlds) if cal_tune_worlds else None,
        "n_index_windows": len(records),
        "splits": splits,
        "all_exclusions_absent_from_train": all_zero,
        "sealed_never_read": list(_SEALED_NEVER_READ),
    }


# -------------------------------------------------------------- index load --
_INDEX_LIST_KEYS = ("windows", "window_index", "records", "items", "runs",
                    "entries", "reviews")


def _extract(obj) -> list:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in _INDEX_LIST_KEYS:
            value = obj.get(key)
            if isinstance(value, list):
                return value
        # a mapping run-id -> record
        values = list(obj.values())
        if values and all(isinstance(v, dict) for v in values):
            return values
    raise ValueError("window index has no record list")


def load_index(path) -> list:
    """Load a window index from JSON, JSONL or Parquet (metadata only)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"window index not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd  # optional dependency
        df = pd.read_parquet(p)
        return _nan_to_none(df.to_dict("records"))
    if suffix in (".jsonl", ".ndjson"):
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return _extract(out) if isinstance(out, dict) else out
    obj = json.loads(p.read_text(encoding="utf-8"))
    return _extract(obj)


def _nan_to_none(records: list) -> list:
    out = []
    for rec in records:
        out.append({k: (None if isinstance(v, float) and v != v else v)
                    for k, v in rec.items()})
    return out


def load_cal_tune_worlds(path=None) -> Optional[list]:
    """CAL-tune world names via the frozen ``evalh.cal_split`` loader.

    Falls back to reading ``CAL_SPLIT_A4.json`` directly when the evalh
    package is not importable.  Metadata only; no outcome is read.
    """
    try:
        import sys
        wm_n1 = Path(__file__).resolve().parents[1]
        if str(wm_n1) not in sys.path:
            sys.path.insert(0, str(wm_n1))
        from evalh.cal_split import half_worlds  # type: ignore
        return list(half_worlds("cal_tune", path=path))
    except Exception:  # noqa: BLE001 - fall back to the raw split file
        split_path = Path(path) if path else \
            Path(__file__).resolve().parents[1] / "CAL_SPLIT_A4.json"
        if not split_path.is_file():
            return None
        data = json.loads(split_path.read_text(encoding="utf-8"))
        return [str(w) for w in data.get("cal_tune", [])]


__all__ = [
    "P5_SPLITS_SCHEMA", "HELD_OUT_TESTS", "J4_WALL_BATCH", "J4_WALL_LAW",
    "J4B_WALL_LAW", "J4_WALL_FACTORS", "HIGHEST_J4_WALL_SPEED_FACTOR",
    "is_j4", "is_j4b", "is_wall", "is_pi", "wall_factor_label", "wall_law_of",
    "wall_action_selected", "has_frozen_j4_law",
    "record_run", "record_arm", "record_world", "record_split", "record_key",
    "heldout_tests", "build_exclusions", "materialize", "load_index",
    "load_cal_tune_worlds",
]
