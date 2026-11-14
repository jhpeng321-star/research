"""Harness cross-tick state capture / restore for the planner (T-B).

Zero SUMO, zero traci import. A SUMO ``loadState`` restores the simulator
exactly, but the PYTHON side of the control loop carries state across
control ticks: ``CommandOwnership`` (``_owned`` / ``_last`` /
``_mode_configured`` / ``_time``), the loop state (``previous_live`` /
``retired_cavs``), the dispatcher session's committed assignments and
clocks, and the PI adapter's recurrence state. A restored process that
starts cold drives the previously commanded CAVs UNCOMMANDED for the first
control interval (the defect the T-A 4b check exposed).

Two producers feed :func:`restore_harness_state`:

- the campaign base run's saved-state manifest (``harness_state``:
  ``ownership`` / ``loop_state`` / ``roster`` ...) -- it carries no session
  or PI state (none existed in the base fleet), and
- the planner master's own snapshot (:func:`capture_harness_state`), which
  additionally carries ``session`` and ``pi`` blocks so a lookahead branch
  restored from the snapshot starts from the master's exact carried state.

All captured values are JSON-safe (NaN is encoded as ``None``).
"""

from __future__ import annotations

import math
from typing import Mapping, Optional

STATE_SCHEMA = "planner_harness_state/1"

#: CommandOwnership fields that cross a control tick (declared inventory; a
#: frozen-side rename fails LOUDLY here, never silently drops state).
OWNERSHIP_ATTRS = ("_owned", "_last", "_mode_configured", "_time", "cav_speed_mode")

#: MixedAssignmentSession bookkeeping that crosses a control tick.
SESSION_ATTRS = (
    "_committed",
    "_pending",
    "_pending_since",
    "_pending_horizon",
    "_last_exec",
    "_exec_horizon",
    "_commit_until",
    "_plan_id",
    "_horizon_s",
    "_last_plan_time",
    "_first_plan_applied",
    "_pi_engaged",
    "_last_tick",
)

_LAST_KEYS = (
    "vehicle_id",
    "timestamp_s",
    "owner",
    "network_output_mps2",
    "wall_request_mps2",
    "issued_acceleration_mps2",
    "issued_target_speed_mps",
    "fallback_reason",
    "released_to_native",
    "actor_input_si",
)


class HarnessStateError(RuntimeError):
    """A capture/restore contract violation (never silently worked around)."""


def _need(obj, name: str, what: str):
    if not hasattr(obj, name):
        raise HarnessStateError(
            f"{what} has no attribute {name!r}: the frozen side changed; the "
            "planner harness-state inventory must be updated (loud, never "
            "silently dropping carried state)"
        )
    return getattr(obj, name)


def _nan_to_none(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _none_to_nan(value):
    return float("nan") if value is None else float(value)


# ------------------------------------------------------------------ ownership
def capture_ownership(owner) -> dict:
    for name in OWNERSHIP_ATTRS:
        _need(owner, name, "CommandOwnership")
    last = {}
    for vehicle, record in owner._last.items():
        row = record.as_dict() if hasattr(record, "as_dict") else dict(record)
        actor = row.get("actor_input_si")
        last[str(vehicle)] = {
            key: (list(actor) if key == "actor_input_si" and actor else row.get(key))
            for key in _LAST_KEYS
        }
    return {
        "cav_speed_mode": int(owner.cav_speed_mode),
        "mode_configured": sorted(str(v) for v in owner._mode_configured),
        "owned": {str(v): str(o) for v, o in owner._owned.items()},
        "last": last,
        "time_s": None if owner._time is None else float(owner._time),
    }


def rebuild_ownership(owner, state: Mapping) -> None:
    """Inject captured ownership state into a fresh ``CommandOwnership``.

    The campaign's ``rebuild_ownership`` semantics (same attribute set);
    the captured ``cav_speed_mode`` is NOT enforced against the owner's:
    the campaign base manifests record 31 while the campaign fleet runs
    mode 27 (the owner rewrites 31 -> 27 on every live CAV at each tick),
    so the planner's mode-27 owner is the faithful continuation."""
    from traffic_shepherd.ownership import CommandRecord  # noqa: PLC0415

    for name in OWNERSHIP_ATTRS:
        _need(owner, name, "CommandOwnership")

    def _record(row: Mapping):
        actor = row.get("actor_input_si")
        return CommandRecord(
            vehicle_id=str(row["vehicle_id"]),
            timestamp_s=float(row["timestamp_s"]),
            owner=str(row["owner"]),
            network_output_mps2=row.get("network_output_mps2"),
            wall_request_mps2=row.get("wall_request_mps2"),
            issued_acceleration_mps2=row.get("issued_acceleration_mps2"),
            issued_target_speed_mps=row.get("issued_target_speed_mps"),
            fallback_reason=row.get("fallback_reason"),
            released_to_native=bool(row.get("released_to_native")),
            actor_input_si=tuple(actor) if actor else None,
        )

    owner._owned = {str(v): str(o) for v, o in (state.get("owned") or {}).items()}
    owner._last = {str(v): _record(r) for v, r in (state.get("last") or {}).items()}
    # The captured ``mode_configured`` set is deliberately NOT restored: it
    # only arms CommandOwnership's "mode changed outside its interface"
    # validator, and a fresh process after ``loadState`` does not carry the
    # python-side proof that each CAV's speedMode was set by this owner.
    # Starting empty makes the owner re-verify: any live CAV still at 31 is
    # (re)configured to 27 on the FIRST tick, before any physics step, and a
    # CAV already at 27 is untouched -- identical dynamics either way.
    owner._mode_configured = set()
    time_s = state.get("time_s")
    owner._time = None if time_s is None else float(time_s)


# -------------------------------------------------------------------- session
def capture_session(session) -> dict:
    for name in SESSION_ATTRS:
        _need(session, name, "MixedAssignmentSession")
    state = {name: _plain(getattr(session, name)) for name in SESSION_ATTRS}
    # Wall-group sessions are NOT carried (walls are exclusively owned by the
    # wall session for the whole episode and stay diagnostic mechanics): the
    # ids are recorded as evidence only; a branch stages any kept wall from
    # its plan / controller_state exactly as before this task.
    state["walls_active_not_carried"] = sorted(str(w) for w in (getattr(session, "_walls", None) or {}))
    return state


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(v) for v in value)
    return value


def restore_session(session, state: Mapping) -> None:
    for name in SESSION_ATTRS:
        _need(session, name, "MixedAssignmentSession")
        if name not in state:
            raise HarnessStateError(f"session state is missing {name!r}")
    roster = set(session.cav_ids)
    for name in ("_committed", "_pending", "_pending_since", "_pending_horizon",
                 "_last_exec", "_exec_horizon", "_commit_until"):
        table = dict(state[name])
        if set(table) != roster:
            raise HarnessStateError(
                f"session state {name!r} roster differs from the session roster"
            )
        setattr(session, name, table)
    for name in ("_plan_id", "_horizon_s", "_last_plan_time", "_first_plan_applied",
                 "_pi_engaged", "_last_tick"):
        setattr(session, name, state[name])


# ------------------------------------------------------------------------ PI
def capture_pi(session) -> Optional[dict]:
    pi_session = getattr(session, "pi_session", None)
    if pi_session is None:
        return None
    adapter = pi_session.adapter
    inner = adapter.pi
    if inner._pending or adapter._pending:
        raise HarnessStateError(
            "PI adapter has an uncommitted proposal mid-tick; capture only "
            "BETWEEN control ticks"
        )
    return {
        "initialized": sorted(str(v) for v in adapter._initialized),
        "tick": adapter._tick,
        "state": {
            str(v): [float(pair[0]), _nan_to_none(float(pair[1]))]
            for v, pair in inner._state.items()
        },
        "generation": {str(v): int(g) for v, g in inner._generation.items()},
        "stopped": bool(adapter.stopped),
    }


def restore_pi(session, state: Optional[Mapping]) -> None:
    pi_session = getattr(session, "pi_session", None)
    if state is None:
        return
    if pi_session is None:
        raise HarnessStateError("PI state was captured but the session has no PI session")
    adapter = pi_session.adapter
    inner = adapter.pi
    adapter._initialized = {str(v) for v in state.get("initialized", [])}
    adapter._tick = state.get("tick")
    inner._state = {
        str(v): (float(pair[0]), _none_to_nan(pair[1])) for v, pair in state["state"].items()
    }
    inner._generation = {str(v): int(g) for v, g in state["generation"].items()}
    adapter.stopped = bool(state.get("stopped", False))


# --------------------------------------------------------------------- harness
def capture_harness_state(control_write, session, roster) -> dict:
    """The full carried state of ONE control loop (call between ticks)."""
    return {
        "schema": STATE_SCHEMA,
        "roster": {"cav_ids": sorted(str(c) for c in roster)},
        "ownership": capture_ownership(control_write.owner),
        "loop_state": {
            "previous_live": sorted(control_write.loop_state.get("previous_live", [])),
            "retired_cavs": sorted(control_write.loop_state.get("retired_cavs", [])),
        },
        "session": capture_session(session),
        "pi": capture_pi(session),
    }


def restore_harness_state(control_write, session, state: Mapping) -> dict:
    """Restore into a fresh control loop; returns a small evidence dict.

    ``state`` is a base-run manifest ``harness_state`` (ownership +
    loop_state only) or a planner snapshot (plus ``session`` / ``pi``)."""
    if "ownership" not in state:
        raise HarnessStateError("harness state carries no 'ownership' block")
    rebuild_ownership(control_write.owner, state["ownership"])
    loop = state.get("loop_state") or {}
    control_write.loop_state = {
        "previous_live": sorted(str(v) for v in loop.get("previous_live", [])),
        "retired_cavs": sorted(str(v) for v in loop.get("retired_cavs", [])),
    }
    if state.get("session") is not None:
        restore_session(session, state["session"])
    restore_pi(session, state.get("pi"))
    return {
        "owned": len(state["ownership"].get("owned") or {}),
        "session_restored": state.get("session") is not None,
        "pi_restored": state.get("pi") is not None,
    }


def previous_lane_ids(physics_row: Optional[Mapping]) -> dict:
    """{vehicle: lane_id} from a saved physics row (lane_changed seeding)."""
    if not physics_row:
        return {}
    return {
        str(v): str(row.get("lane_id"))
        for v, row in (physics_row.get("vehicles") or {}).items()
        if isinstance(row, Mapping) and row.get("lane_id") is not None
    }


__all__ = [
    "HarnessStateError",
    "OWNERSHIP_ATTRS",
    "SESSION_ATTRS",
    "STATE_SCHEMA",
    "capture_harness_state",
    "capture_ownership",
    "capture_pi",
    "capture_session",
    "previous_lane_ids",
    "rebuild_ownership",
    "restore_harness_state",
    "restore_pi",
    "restore_session",
]
