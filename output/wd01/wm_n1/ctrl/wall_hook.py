"""P1 wall-law hook: how wall-v2.2 arrives and what it adds.

The in-tree J4 wall law is ``wall-v2.1-closure`` (partA_ctrl.emulators
``WALL_LAW_VERSION``, emulators.py:437).  Idea A must also run wall v2.2,
whose ONLY delta is the automatic MERGE-RELEASE rule (owner brief,
WM-N1 Idea A piece P1): during HOLD the wall is released when the LEAD
member's cumulative road coordinate reaches the merge start
``s_road = 2644.2 m`` -- the hold is capped at the merge start, it can
never keep members standing upstream of it.  Assembly pacing, safety
aborts and the bounded pacing law are the verified v2.1 mirrors, reused
unchanged (controllers are exact, never learned).

Road geometry (frozen network, task constants):

===== ============ ==========
road  role         length (m)
===== ============ ==========
1191554636  entry   2644.2  (to the merge start)
1487606179  merge     75.4
1487606178  exit     477.9
===== ============ ==========

``s_road_m(road_id, lane_position_m)`` maps a SUMO row onto that
cumulative coordinate; :data:`MERGE_START_S_ROAD_M` == 2644.2 is the
release threshold.  In the batched world (merge-relative ``s_m``,
``features.geometry`` convention) ``s_road = s_m + MERGE_START_S_ROAD_M``.

A landed law registers by name (``register_wall_law``); the built-in
v2.2 law below is registered by :func:`ensure_builtin_laws` (called at
``ctrl`` import; tests may ``clear_registry()`` and re-install).
"""
from __future__ import annotations

from typing import Dict, Optional, Protocol, runtime_checkable

import torch
from torch import Tensor

from .config import WALL_LAW_V21, WALL_LAW_V22

# ------------------------------------------------------------ v2.2 constants

#: release threshold: the lead member's cumulative road coordinate (m).
MERGE_START_S_ROAD_M = 2644.2

#: release comparisons fire at ``POINT - tol`` (frozen wall_group.py:239).
MERGE_RELEASE_TOL_M = 1e-9

#: SUMO road id -> cumulative s_road offset at the road's upstream end (m).
ROAD_OFFSETS_M: Dict[str, float] = {
    "1191554636": 0.0,       # entry, ends at the merge start
    "1487606179": 2644.2,    # merge, 75.4 m long
    "1487606178": 2719.6,    # exit, 477.9 m long
}

#: SUMO road id -> modelled road length (m), matching ROAD_OFFSETS_M.
ROAD_LENGTHS_M: Dict[str, float] = {
    "1191554636": 2644.2,
    "1487606179": 75.4,
    "1487606178": 477.9,
}


def s_road_m(road_id: str, lane_position_m: float) -> float:
    """Cumulative road coordinate of one vehicle row (float, metres).

    Raises ``KeyError`` for a road outside the frozen table (junction
    internal ``:``-lanes have no coordinate; callers skip those rows).
    """
    road = str(road_id)
    if road not in ROAD_OFFSETS_M:
        raise KeyError(f"road {road!r} is not in the frozen s_road table")
    return ROAD_OFFSETS_M[road] + float(lane_position_m)


class WallLawUnavailable(KeyError):
    """The requested wall law has not been registered yet."""


@runtime_checkable
class WallLaw(Protocol):
    """Runtime interface of a registered wall pacing law.

    All tensors are batched float64; slot convention matches the mover
    (``group < 0`` / absent = not a wall member).
    """

    name: str

    def gap_to_front(self, positions_m: Tensor, group: Tensor,
                     closure: Optional[Tensor] = None) -> Tensor:
        """Per-member distance behind the wall front (``[B]``/``[B,K]``)."""
        ...

    def pace(self, wall_speed_mps: Tensor, speed_mps: Tensor, *,
             dt_s: float = 1.0,
             gap_to_front_m: Optional[Tensor] = None) -> Tensor:
        """Bounded member acceleration (``[B]``/``[B,K]``)."""
        ...


_REGISTRY: Dict[str, WallLaw] = {}


def register_wall_law(law: WallLaw) -> WallLaw:
    """Register (or replace) a wall law by ``law.name``; returns it."""
    name = str(getattr(law, "name", ""))
    if not name:
        raise ValueError("wall law must expose a non-empty .name")
    _REGISTRY[name] = law
    return law


def resolve_wall_law(name: str) -> WallLaw:
    """Return the registered law, or raise :class:`WallLawUnavailable`."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:  # noqa: TRY003 - explicit named error
        raise WallLawUnavailable(
            f"wall law {name!r} is not registered; landed branches call "
            f"ctrl.wall_hook.register_wall_law(...) at import. "
            f"registered: {sorted(_REGISTRY)}") from exc


def wall_law_status() -> Dict[str, bool]:
    """``{name: registered}`` for the two known versions."""
    return {WALL_LAW_V21: WALL_LAW_V21 in _REGISTRY,
            WALL_LAW_V22: WALL_LAW_V22 in _REGISTRY}


def clear_registry() -> None:
    """Test helper: forget all registered laws."""
    _REGISTRY.clear()


# ------------------------------------------------------- built-in v2.2 law

class MergeReleaseWallLaw:
    """``wall-v2.2``: the v2.1 closure pacing + the merge-release rule.

    ``gap_to_front`` / ``pace`` delegate to the verified J4 mirrors
    (``partA_ctrl.wall_gap_to_front`` / ``wall_pace``), so a v2.2 wall in
    ASSEMBLY/HOLD is commanded exactly like v2.1.  The delta:

    * :meth:`release_mask` - a wall slot is released (handed back to
      native, target NaN) when its wall's LEAD member has reached
      :data:`MERGE_START_S_ROAD_M` **and** the wall is in HOLD.  The rule
      never fires during assembly ("merge_release ... in HOLD only") and
      caps the hold at the merge start.
    """

    name = WALL_LAW_V22

    # -- v2.1 mirrors (identical law, delegated read-only) ---------------
    def gap_to_front(self, positions_m: Tensor, group: Tensor,
                     closure: Optional[Tensor] = None) -> Tensor:
        from ._bridge import emulators

        return emulators().wall_gap_to_front(positions_m, group,
                                             closure=closure)

    def pace(self, wall_speed_mps: Tensor, speed_mps: Tensor, *,
             dt_s: float = 1.0,
             gap_to_front_m: Optional[Tensor] = None) -> Tensor:
        from ._bridge import emulators

        return emulators().wall_pace(wall_speed_mps, speed_mps,
                                     dt_s=dt_s,
                                     gap_to_front_m=gap_to_front_m)

    # -- the v2.2 delta ---------------------------------------------------
    @staticmethod
    def release_mask(s_road_m: Tensor, group: Tensor,
                     hold: Tensor) -> Tensor:
        """``True`` where the slot's wall is released this tick.

        Parameters: ``s_road_m`` float [..] cumulative road coordinate of
        every slot, ``group`` int64 [..] wall id (-1 none), ``hold`` bool
        [..] True while that slot's wall is in HOLD.  A slot is released
        when ``hold`` and its wall's lead (max ``s_road_m``) has reached
        ``MERGE_START_S_ROAD_M``.  Non-members and assembly slots are
        never released.  Deterministic, stateless, pure.
        """
        gid = group.to(torch.int64)
        flat = gid.reshape(-1)
        pos = s_road_m.to(torch.float64).reshape(-1)
        n = int(flat.max().item()) + 1 if flat.numel() else 0
        lead = torch.full((max(n, 1),), float("-inf"),
                          dtype=torch.float64, device=pos.device)
        members = flat >= 0
        if members.any():
            lead = lead.scatter_reduce(
                0, flat[members].clamp(min=0), pos[members], "amax",
                include_self=True)
        reached = torch.zeros_like(pos, dtype=torch.bool)
        reached[members] = (lead[flat[members]]
                            >= MERGE_START_S_ROAD_M - MERGE_RELEASE_TOL_M)
        hold_flat = hold.to(torch.bool).reshape(-1)
        out = reached & hold_flat
        return out.reshape(group.shape)


_BUILTIN = (MergeReleaseWallLaw(),)


def ensure_builtin_laws() -> None:
    """Register the built-in v2.2 merge-release law if absent."""
    if WALL_LAW_V22 not in _REGISTRY:
        register_wall_law(_BUILTIN[0])
