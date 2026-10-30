"""Opt-in scorer ownership repair; longitudinal timelines remain untouched.

Only timeline callbacks are wrapped. Actual policy bridges retain their lane
requests, feasibility rules and native-lane masks (including residual policies).
The packed action timeline has no original-slot policy lane requests: such arms
remain diagnostic and must not be labelled native-lane qualification contrasts.
"""
import torch


class NativeTimeline:
    def __init__(self, recorded):
        self.recorded = recorded

    def native_lane_mask(self, state):
        return state['present'].bool() & state['is_cav'].bool() & (state['lane'] >= 0)

    def __call__(self, state):
        from wms.model.cav import WM_LANE_TARGET_KEY, WM_MERGE_KEY
        motion = self.recorded(state)
        mask = self.native_lane_mask(state)
        targets = state.get(WM_LANE_TARGET_KEY)
        if targets is None:
            raise ValueError('native timeline lanes require WM-computed native targets')
        # A callback crossing, if supplied, has priority over native delegation.
        crossing = motion[..., 1].long() != state['lane']
        native = mask & ~crossing
        lane = torch.where(native & (state['lane'] != 2), targets, motion[..., 1])
        state[WM_MERGE_KEY] = native & (state['lane'] == 2) & (lane == 2)
        return torch.stack((motion[..., 0], lane.to(motion.dtype),
                            motion[..., 2], motion[..., 3]), -1)


def repair(callback):
    """Repair originals and native entrant fallbacks; leave policy bridges intact."""
    from wms.train.cav_drive import TimelineControllers
    from wms.control.entrant import EntrantControllers, RowControllers
    if isinstance(callback, NativeTimeline):
        return callback
    if isinstance(callback, TimelineControllers):
        return NativeTimeline(callback)
    if isinstance(callback, EntrantControllers):
        callback.recorded = repair(callback.recorded)
        callback.feedback = repair(callback.feedback)
    elif isinstance(callback, RowControllers):
        callback.callbacks = [repair(c) for c in callback.callbacks]
    elif hasattr(callback, 'groups'):
        callback.groups = [(ix, repair(c)) for ix, c in callback.groups]
    return callback
