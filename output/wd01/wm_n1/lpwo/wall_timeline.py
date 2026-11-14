"""Wall ownership / timeline metadata of one wall-pilot run directory.

Reads only ``branch_record.json`` and ``wall_events.jsonl`` (a few kB each, never
``physics_steps.jsonl`` or any outcome row).  A run is classified by what the
recorded wall actually did to the CAVs:

* ``assembled_released`` / ``aborted_in_hold``   -> **engaged** (members were
  assembled side by side and held at wall speed; ownership + assembly + hold);
* ``aborted_in_assembly``                        -> **owned, not assembled**
  (members were wall-owned while assembling but the wall never formed);
* ``not_feasible``                               -> **not owned** (no request,
  no CAV was ever under wall ownership);
* anything unreadable or self-contradictory      -> ``unverifiable`` (fail closed:
  treated as owned for training exclusion, never as an engaged test wall).

All ``*_s`` times in the source are absolute simulation seconds; the ``*_rel_s``
fields are relative to the run ``t0_s`` (the pack window origin).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

ENGAGED_OUTCOMES = frozenset({"assembled_released", "aborted_in_hold"})
OWNED_NOT_ASSEMBLED = frozenset({"aborted_in_assembly"})
NOT_OWNED_OUTCOMES = frozenset({"not_feasible"})
KNOWN_OUTCOMES = ENGAGED_OUTCOMES | OWNED_NOT_ASSEMBLED | NOT_OWNED_OUTCOMES


def _rel(value, t0_s):
    return None if value is None else float(value) - float(t0_s)


def _read_events(path: Path) -> list[dict]:
    events = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # the per-second member-lane rows are the bulk of the file; keep the
            # kind only so one run stays tiny in memory
            if row.get("kind") == "wall_member_lanes":
                continue
            else:
                events.append(row)
    return events


def read_wall_timeline(run_dir: str | Path, guard=None) -> dict:
    """Timeline/ownership record for ``<run_dir>`` (the dir holding ``run/``)."""
    run_dir = Path(run_dir)
    inner = run_dir / "run"
    record_path = inner / "branch_record.json"
    events_path = inner / "wall_events.jsonl"
    base = {"run": run_dir.name, "run_dir": str(run_dir)}
    if guard is None:
        raise ValueError("explicit holdout guard required")
    for path in (run_dir, record_path, events_path):
        guard.check_path(path)
    try:
        branch = guard.read_json(record_path)
        guard.check_record({"split": "TRAIN", **branch,
                            "world": branch.get("root_id") or branch.get("world")})
        events = _read_events(events_path)
    except (OSError, ValueError) as exc:
        return {**base, "status": "unverifiable", "owned": True, "engaged": False,
                "reason": f"{type(exc).__name__}: {exc}"}

    outcome = str(branch.get("outcome", ""))
    has_request = any(e.get("kind") == "wall_request" for e in events)
    members = branch.get("members")
    t0_s = float(branch.get("t0_s", 0.0))
    status = outcome if outcome in KNOWN_OUTCOMES else "unverifiable"
    reason = None
    if status == "unverifiable":
        reason = f"unknown outcome {outcome!r}"
    elif status in NOT_OWNED_OUTCOMES and (has_request or members):
        status, reason = "unverifiable", "not_feasible outcome but a wall request/members exist"
    elif status not in NOT_OWNED_OUTCOMES and not (has_request and members):
        status, reason = "unverifiable", f"{outcome} outcome without wall_request/members"
    owned = status != "not_feasible"      # unverifiable is treated as owned
    engaged = status in ENGAGED_OUTCOMES

    handback = [e for e in events if e.get("kind") == "wall_handback"]
    handback_s = handback[0].get("timestamp_s") if handback else None
    assembled_s = branch.get("assembled_at_s")
    request_s = branch.get("request_s")
    released_s = branch.get("released_at_s")
    aborted_s = branch.get("aborted_at_s")
    owner_end = released_s if released_s is not None else (
        aborted_s if aborted_s is not None else handback_s)
    if status in ENGAGED_OUTCOMES | OWNED_NOT_ASSEMBLED:
        times = [request_s, owner_end] + ([assembled_s] if engaged else [])
        valid = all(isinstance(t, (int, float)) and math.isfinite(t) for t in times)
        valid = valid and request_s < owner_end
        valid = valid and (not engaged or request_s <= assembled_s < owner_end)
        valid = valid and (engaged or assembled_s is None)
        valid = valid and isinstance(members, list) and len(members) == len(set(members))
        if not valid:
            status, reason, engaged = "unverifiable", "incomplete/contradictory ownership times or members", False
    out = {
        **base,
        "status": status, "outcome": outcome, "owned": bool(owned),
        "engaged": bool(engaged), "reason": reason,
        "world": branch.get("root_id"), "arm": branch.get("branch_type"),
        "wall_id": branch.get("wall_id"), "timing": branch.get("timing"),
        "members": list(members) if members else None,
        "t0_s": t0_s,
        "request_s": request_s, "assembled_at_s": assembled_s,
        "released_at_s": released_s, "aborted_at_s": aborted_s,
        "handback_s": handback_s, "owner_end_s": owner_end,
        "request_rel_s": _rel(request_s, t0_s),
        "assembled_rel_s": _rel(assembled_s, t0_s),
        "released_rel_s": _rel(released_s, t0_s),
        "aborted_rel_s": _rel(aborted_s, t0_s),
        "owner_end_rel_s": _rel(owner_end, t0_s),
        "abort_reason": branch.get("abort_reason"),
        "requested_hold_s": branch.get("requested_hold_s"),
        "achieved_hold_s": branch.get("achieved_hold_s"),
        "wall_speed_mps": branch.get("wall_speed_mps"),
        "handback_to": (handback[0].get("handback") if handback else None),
        "n_events": len(events),
    }
    return out


def phase_at(timeline: dict, rel_s: float) -> Optional[str]:
    """Wall phase ``none|assembly|hold|handed_back`` at run-relative time."""
    if not timeline.get("owned") or timeline.get("request_rel_s") is None:
        return "none"
    start = timeline["request_rel_s"]
    end = timeline.get("owner_end_rel_s")
    if rel_s < start:
        return "none"
    if end is not None and rel_s >= end:
        return "handed_back"
    assembled = timeline.get("assembled_rel_s")
    if assembled is None or rel_s < assembled:
        return "assembly"
    return "hold"
