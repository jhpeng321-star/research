"""Probe-only capture of the actual WM decision scene, after entry admission."""
from contextlib import contextmanager
from functools import wraps

import torch

from wms.physics.state import state_kin, state_present


@contextmanager
def capture_rollout_geometry(model_cls):
    """Attach detached decision snapshots to each rollout without replaying geometry.

    Recorded targets have no generated entrants. Neither their lengths nor a
    shifted post-step presence mask can describe the scene after insertion.
    Keep this instrumentation scoped to the diagnostic, including on exceptions.
    """
    original_rollout = model_cls.rollout
    original_geometry = model_cls.pivot_decision_geometry
    active = []

    @wraps(original_geometry)
    def decision(self, state, *args, **kwargs):
        geo = original_geometry(self, state, *args, **kwargs)
        if active and active[-1][0] is self:
            kin = state_kin(state)
            present = state_present(state, kin)
            cav = state.get("is_cav", torch.zeros_like(present)).bool()
            active[-1][1].append({
                "kin": kin.detach().clone(), "present": present.detach().clone(),
                "is_cav": cav.detach().clone(),
                "geometry": {key: value.detach().clone() for key, value in geo.items()},
            })
        return geo

    @wraps(original_rollout)
    def rollout(self, *args, **kwargs):
        snapshots = []
        active.append((self, snapshots))
        try:
            result = original_rollout(self, *args, **kwargs)
            result["r9_decision_scenes"] = snapshots
            return result
        finally:
            active.pop()

    model_cls.rollout = rollout
    model_cls.pivot_decision_geometry = decision
    try:
        yield
    finally:
        model_cls.rollout = original_rollout
        model_cls.pivot_decision_geometry = original_geometry


def wm_diag(prep, ctx, b, j, k):
    """Describe the captured decision; never fabricate entrant metadata from targets."""
    snapshot = ctx.pred["r9_decision_scenes"][j]
    kin, present, cav = (snapshot[key][b] for key in ("kin", "present", "is_cav"))
    g = {key: value[b] for key, value in snapshot["geometry"].items()}
    rec = ctx.targets.states[b, ctx.c0 + j]
    lane = kin[:, 1:4].argmax(-1)
    ov = int(g["overlap_index"][k])
    out = dict(ramp=bool(g["ramp"][k]), scene_valid=bool(g["scene_valid"][k]),
               overlap=bool(g["overlap"][k]), cav=bool(cav[k]),
               phys=bool(g["physical_eligible"][k]), elig=bool(g["eligible"][k]),
               fixed_overlap=bool(g["fixed_overlap"][k]), partner=ov,
               ego_wm_pre=float(kin[k, 0]), ego_wm_moved=float(g["s"][k]),
               ego_len=float(kin[k, 6]), ego_rec_pre=float(rec[k, 0]),
               ego_wm_present=bool(present[k]),
               ego_rec_present=bool(prep["risk"][b, ctx.c0 + j, k]))
    if ov >= 0:
        out.update(p_wm_moved=float(g["s"][ov]), p_len=float(kin[ov, 6]),
                   p_lane=int(lane[ov]), p_rec_pre=float(rec[ov, 0]),
                   p_rec_lane=int(rec[ov, 1:4].argmax()),
                   p_present_wm=bool(present[ov]), p_is_cav=bool(cav[ov]),
                   p_wm_pre=float(kin[ov, 0]))
    observed = (torch.isfinite(kin).all(-1) & (kin[:, 1:4].sum(-1) > 0)
                & (kin[:, 6] > 0))
    out["n_unobserved_present"] = int((present & ~observed).sum())
    out["n_target_wm"] = int((present & (lane == 1)).sum())
    out["n_target_rec"] = int(((rec[:, 1:4].argmax(-1) == 1)
                                & (rec[:, 1:4].sum(-1) > 0)).sum())
    return out
