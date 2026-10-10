"""Per-step bulk traffic snapshot via TraCI vehicle subscriptions.

Reading per-vehicle state one TraCI call at a time costs a full TCP round
trip per value; in dense traffic the observation/reward builders issued
thousands of those calls per env step. SUMO's subscription mechanism
returns the same values in a single ``getAllSubscriptionResults()`` call.

Equivalence (verified empirically on SUMO 1.27.1, see
``tools/probe_subscription_vars.py`` and ``tools/probe_leader_subscription.py``):
every subscribed variable below is bit-identical to the direct getter at the
same simulation time, including the parameterized ``VAR_LEADER`` (dist
parameter, matched against ``getLeader(vid, dist)`` for >5000 comparisons
with zero mismatches in identity and gap).

The snapshot is an *optional, ambient* data source: readers in
``env/traci_query.py``, ``env/features.py`` and ``env/rewards.py`` consult
:func:`get_active_snapshot` first and fall back to the original direct
TraCI calls when no snapshot is active (unit tests with mocked traci,
one-off scripts, custom reward functions). ``env.core`` activates the
snapshot for exactly the lifetime of a SUMO connection.

Vehicles that depart during step ``t`` are subscribed right after the step;
the subscribe command itself returns the vehicle's current values, so new
vehicles are fully covered from the same step. ``_complete_record`` fills any
variable still missing (defensive: partial records, subscribe races) with the
direct getters — same values either way.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set

import traci
import traci.constants as tc

# Look-ahead (m) used for the parameterized VAR_LEADER subscription. Must
# stay equal to env.traci_query.DEFAULT_LOOK_AHEAD; leader queries with any
# other look-ahead fall back to direct getLeader calls.
SUBSCRIPTION_LOOK_AHEAD = 200.0

SNAPSHOT_VARS: tuple = (
    tc.VAR_SPEED,
    tc.VAR_ACCELERATION,
    tc.VAR_ROAD_ID,
    tc.VAR_LANE_ID,
    tc.VAR_LANE_INDEX,
    tc.VAR_LANEPOSITION,
    tc.VAR_POSITION,
    tc.VAR_TIMELOSS,
    tc.VAR_TYPE,
    tc.VAR_SIGNALS,
    tc.VAR_LEADER,
)


class TrafficSnapshot:
    """Bulk per-step vehicle state for one SUMO connection.

    Lifecycle per episode: :meth:`reset` after (re)starting SUMO, then
    :meth:`update` exactly once after every ``traci.simulationStep``,
    before any observation / reward / action queries read from it.
    """

    def __init__(self, look_ahead: float = SUBSCRIPTION_LOOK_AHEAD):
        self.look_ahead = float(look_ahead)
        self._subscribed: Set[str] = set()
        self.data: Dict[str, Dict[int, Any]] = {}
        self.ids: List[str] = []
        self.id_set: Set[str] = set()
        self._step_memo: Dict[Any, Any] = {}
        self._static_memo: Dict[Any, Any] = {}

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        """Drop all state for a new SUMO connection / episode."""
        self._subscribed.clear()
        self.data = {}
        self.ids = []
        self.id_set = set()
        self._step_memo = {}
        self._static_memo = {}

    def update(self) -> None:
        """Refresh from the current simulation state (post-simulationStep)."""
        try:
            departed = traci.simulation.getDepartedIDList()
        except Exception:
            departed = ()
        for vid in departed:
            if vid not in self._subscribed:
                try:
                    traci.vehicle.subscribe(
                        vid,
                        list(SNAPSHOT_VARS),
                        parameters={tc.VAR_LEADER: ("d", self.look_ahead)},
                    )
                    self._subscribed.add(vid)
                except Exception:
                    continue

        try:
            results = traci.vehicle.getAllSubscriptionResults() or {}
        except Exception:
            results = {}
        try:
            ids = list(traci.vehicle.getIDList())
        except Exception:
            ids = []

        data: Dict[str, Dict[int, Any]] = {}
        for vid in ids:
            rec = results.get(vid)
            if rec is None or any(var not in rec for var in SNAPSHOT_VARS):
                # Partial record (defensive; subscribe races): complete it
                # with direct getters.
                rec = self._complete_record(vid, rec)
            else:
                # Copy: getAllSubscriptionResults returns live dicts that the
                # traci connection resets/replaces on the next step.
                rec = dict(rec)
            data[vid] = rec

        self.data = data
        self.ids = ids
        self.id_set = set(ids)
        self._step_memo = {}

    def _complete_record(self, veh_id: str, rec: Optional[Dict[int, Any]]) -> Dict[int, Any]:
        """Fill missing variables with direct getters (same values)."""
        out = dict(rec or {})
        getters = (
            (tc.VAR_SPEED, traci.vehicle.getSpeed),
            (tc.VAR_ACCELERATION, traci.vehicle.getAcceleration),
            (tc.VAR_ROAD_ID, traci.vehicle.getRoadID),
            (tc.VAR_LANE_ID, traci.vehicle.getLaneID),
            (tc.VAR_LANE_INDEX, traci.vehicle.getLaneIndex),
            (tc.VAR_LANEPOSITION, traci.vehicle.getLanePosition),
            (tc.VAR_POSITION, traci.vehicle.getPosition),
            (tc.VAR_TIMELOSS, traci.vehicle.getTimeLoss),
            (tc.VAR_TYPE, traci.vehicle.getTypeID),
            (tc.VAR_SIGNALS, traci.vehicle.getSignals),
            (tc.VAR_LEADER, lambda vid: traci.vehicle.getLeader(vid, self.look_ahead)),
        )
        for var, getter in getters:
            if var not in out:
                try:
                    out[var] = getter(veh_id)
                except Exception:
                    continue
        return out

    # -- generic per-step / static memoization ------------------------------
    def memo(self, key: Any, fn: Callable[[], Any], static: bool = False) -> Any:
        """Return ``fn()`` cached: per-step, or for the episode when ``static``.

        Used for values that cannot change within a step (edge occupancy,
        per-edge vehicle ID lists) or within an episode (lane lengths) so
        repeated per-CAV queries collapse into one TraCI call per step.
        """
        cache = self._static_memo if static else self._step_memo
        if key not in cache:
            cache[key] = fn()
        return cache[key]

    # -- typed accessors (None mirrors the direct getter raising) -----------
    def has(self, veh_id: str) -> bool:
        return veh_id in self.id_set

    def value(self, veh_id: str, var: int) -> Any:
        rec = self.data.get(veh_id)
        if rec is None:
            return None
        return rec.get(var)

    def speed(self, veh_id: str) -> Optional[float]:
        return self.value(veh_id, tc.VAR_SPEED)

    def accel(self, veh_id: str) -> Optional[float]:
        return self.value(veh_id, tc.VAR_ACCELERATION)

    def road(self, veh_id: str) -> Optional[str]:
        return self.value(veh_id, tc.VAR_ROAD_ID)

    def lane_id(self, veh_id: str) -> Optional[str]:
        return self.value(veh_id, tc.VAR_LANE_ID)

    def lane_index(self, veh_id: str) -> Optional[int]:
        return self.value(veh_id, tc.VAR_LANE_INDEX)

    def lane_pos(self, veh_id: str) -> Optional[float]:
        return self.value(veh_id, tc.VAR_LANEPOSITION)

    def position(self, veh_id: str):
        return self.value(veh_id, tc.VAR_POSITION)

    def time_loss(self, veh_id: str) -> Optional[float]:
        return self.value(veh_id, tc.VAR_TIMELOSS)

    def vtype(self, veh_id: str) -> Optional[str]:
        return self.value(veh_id, tc.VAR_TYPE)

    def signals(self, veh_id: str) -> Optional[int]:
        return self.value(veh_id, tc.VAR_SIGNALS)

    def leader(self, veh_id: str):
        """Subscription VAR_LEADER record: ``(leader_id, gap)`` / None-ish."""
        return self.value(veh_id, tc.VAR_LEADER)

    def ordered_ids(self) -> List[str]:
        """Vehicle IDs in the same order as traci.vehicle.getIDList()."""
        return self.ids


# ---------------------------------------------------------------------------
# Ambient active snapshot (single TraCI connection per process, matching the
# existing codebase assumption: traci itself is a module-level singleton).
# ---------------------------------------------------------------------------
_active_snapshot: Optional[TrafficSnapshot] = None


def set_active_snapshot(snapshot: Optional[TrafficSnapshot]) -> None:
    global _active_snapshot
    _active_snapshot = snapshot


def get_active_snapshot() -> Optional[TrafficSnapshot]:
    return _active_snapshot


def clear_active_snapshot() -> None:
    set_active_snapshot(None)
