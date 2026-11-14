"""Read-only adapter over ``partB_v0data`` action + wall inputs (v0.4a rule).

CAV actions come ONLY from ``output/wd01/partB_v0data/actions_v04.py``:
the per-CAV token is the PLANNED action from ``switch_schedule.json`` (latest
schedule event at or before t); the ``kind == "command"`` owner rows supply
the token for a CAV with no schedule entry and are never used to relabel a
scheduled PI token (the owner-confirmed v0.4a rule: a PI-external-U CAV is
written through the damping writer, so realised owners would mislabel PI).

No file outside the two read-only partB_v0data modules is imported; this
package never reads the per-tick dispatcher action log or the
realised-insertion log (asserted by a static + functional open-guard test).
"""
from __future__ import annotations

import sys
from pathlib import Path

_WD01 = Path(__file__).resolve().parents[2]          # output/wd01
_PARTB = _WD01 / "partB_v0data"
if str(_PARTB) not in sys.path:
    sys.path.insert(0, str(_PARTB))

from actions_v04 import (  # noqa: E402
    ACTIONS_V04, ACTION_UNKNOWN, ActionTimeline)
from wall_tokens_v04 import WallEpisodeSet  # noqa: E402

__all__ = ["PlannedActionTimeline", "WallFeatureSource", "ACTIONS_V04",
           "ACTION_UNKNOWN", "ActionTimeline", "WallEpisodeSet"]


class PlannedActionTimeline:
    """Planned-schedule-authoritative per-CAV action token.

    ``fleet_policy`` (from the run manifest) applies to every vehicle when
    the run has no per-CAV schedule (the fleet base runs).  Otherwise the
    PLANNED schedule is authoritative; a CAV without a schedule entry falls
    back to the realised command-owner timeline, and with no evidence at all
    the token is ``unknown`` (never a silent "native").
    """

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self._fleet = ActionTimeline.from_run_dir(run_dir, command_files=())
        self._realised = ActionTimeline.from_run_dir(run_dir)
        self.fleet_policy = self._realised.fleet_policy

    def action_at(self, vid: str, t: float) -> str:
        if self.fleet_policy is not None:
            return self.fleet_policy
        planned = self._fleet.action_at(vid, t)
        if planned != ACTION_UNKNOWN:
            return planned
        return self._realised.action_at(vid, t)

    def action_slot(self, vid: str, t: float) -> int:
        return ACTIONS_V04.index(self.action_at(vid, t))


class WallFeatureSource:
    """Thin read-only wrapper carrying the wall episode set of one run."""

    def __init__(self, episode_set: WallEpisodeSet | None):
        self._set = episode_set

    @classmethod
    def from_run_dir(cls, run_dir):
        return cls(WallEpisodeSet.from_branch_dir(run_dir))

    @property
    def empty(self) -> bool:
        return self._set is None or len(self._set) == 0

    def walls_at(self, t, positions=None, cav_roster=None):
        if self._set is None:
            return []
        return self._set.walls_at(t, positions=positions, cav_roster=cav_roster)
