"""Observable-export adapter; no saved tape/export identity crosswalk."""
import numpy as np


def lane_arrays(start, raw):
    ids = np.asarray(raw['vid_index'])
    selected = start['state']['present'][0].cpu().numpy().astype(bool)
    known = ids[-1, selected]
    cav = start['state']['is_cav'][0, selected].cpu().numpy().astype(bool)
    # Ownership of past departed drivers is not known by this export. Keep
    # their bodies visible but never treat them as observation evidence/egos.
    is_cav = ~np.isin(ids, known[~cav])
    return dict(time_s=raw['time_s'], veh_kin=raw['kin'], veh_mask=raw['mask'],
                vid=ids, is_cav=is_cav)
