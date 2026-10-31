"""Undo recorded lane-origin shifts before a post-motion LC decision.

This only reconstructs a physical coordinate: it does not supply a recorded
motivation or outcome to the lane model. SUMO preserves positionOnLane when
changing lanes; the frozen WM upstream origins differ by1.45m.
"""
import numpy as np

from wms.physics.lc2013_geometry import UPSTREAM_POSITION_OFFSETS_M

SNAPSHOT_CONTRACT = 'post-motion s/v/a restored to PRE-change lane origins; teacher forced v2'


def lane_shift(s, source, destination, merge_start):
    offsets = np.asarray(UPSTREAM_POSITION_OFFSETS_M, dtype=np.float64)
    through = (source < 2) & (destination < 2)
    return np.where((s < merge_start) & through, offsets[source] - offsets[destination], 0.)


def postmotion_snapshot(before, after, physics):
    moved = np.array(before, copy=True)
    moved[:, [0, 4, 5]] = after[:, [0, 4, 5]]
    source, destination = before[:, 1:4].argmax(-1), after[:, 1:4].argmax(-1)
    moved[:, 0] -= lane_shift(after[:, 0], source, destination, physics.merge_zone_s0_m)
    return moved


def proposal_position(moved_source_s, source, destination, physics):
    return moved_source_s + lane_shift(moved_source_s, source, destination, physics.merge_zone_s0_m)
