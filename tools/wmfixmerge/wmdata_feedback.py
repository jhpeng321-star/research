"""Explicit frozen entrant algorithms for owner-approved wmdata_fit arms."""


def feedback(specs, arm_names, model, timeline, original_slots, device, *, branches=None):
    import torch
    from tools.wmfixmerge.data import digest, safe
    from tools.wmrequal.grouped_feedback import Grouped
    from wms.control.entrant import policy_feedback
    from wms.data.action_timeline import CONTROLLERS
    from wms.train.cav_drive import TimelineControllers

    groups = []
    for arm in sorted(set(arm_names)):
        indices = [i for i, name in enumerate(arm_names) if name == arm]
        if arm in ("ctrl0", "mappo"):
            spec = dict(specs[arm])
            checkpoint, sha = safe(spec.pop("checkpoint")), spec.pop("sha256")
            if digest(checkpoint) != sha:
                raise ValueError("wmdata frozen policy changed: " + arm)
            callback = policy_feedback(checkpoint, device=device, **spec)
        else:
            algorithm = {"native": "native", "pi": "pi-external-u", "circles": "damping"}[arm]
            row = {k: v[indices].clone() if torch.is_tensor(v) else v for k, v in timeline.items()}
            row["controller"].fill_(CONTROLLERS.index(algorithm))
            callback = TimelineControllers(model, row)
        groups.append((indices, callback))
    return Grouped(groups)
