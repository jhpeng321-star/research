"""S-EXECPARITY3 lane-chain geometry: the WM corridor coordinate made monotone
along SUMO's physical lane chains (default OFF flag ``C4_SUMO_LEADER_RULE``).

The WM corridor coordinate ``s`` is continuous across the merge junctions while
the registered network's lanes are not: between the corridor segments sit three
internal junction connectors whose physical lengths are hidden from ``s``
(``assets/research/highway_merge_shepherd_long/osm.net.xml.gz``, net version
1.20).  SUMO's lane/link-based leader search DOES traverse them, so a leader
gap measured along the lane chain differs from the corridor-projected gap by
exactly the connector lengths:

* ramp -> accel lane across ``:1780739017_0_0`` (2.93 m): a ramp ego's SUMO
  leader gap is the frame gap + 2.93 m (the measured 00:20 anatomy);
* mainline entry -> merge edge across ``:1780739017_1_*`` (3.00 m per lane);
* merge edge -> exit edge across ``:13638027693_0_*`` (8.00 m per lane).

:func:`chain_s` adds those registered network attributes (nothing fitted,
nothing hand-tuned) to the corridor ``s`` so that within every frame lane the
chain order and the gaps are SUMO's link order for through traffic.
:func:`junction_chain_s` maps a junction-transit vehicle (on a connector, its
observable frame state being the carried last modelled lane/position) to the
ENTRY of that connector - the geometric lower bound of its true chain position,
conservative for an upstream ego (never overstates the gap).  Lanes 0/1/2 all
reach chain coordinate 0.0 at the merge junction and ENTRY+MERGE_S1 at the exit
junction, so that bound is lane-independent.

Pure geometry of the physics layer: no wms.control imports, no behavioural
parameters.  The flag :func:`leader_rule_enabled` (default OFF, read per call)
gates every use; with it off every consumer keeps its frozen legacy body
bit-for-bit.
"""
from __future__ import annotations

import os

#: Environment flag enabling the SUMO lane/link leader rule (default OFF).
LEADER_RULE_ENV = "C4_SUMO_LEADER_RULE"
_TRUTHY = ("1", "true", "yes", "on")

#: Length of the merge-edge lanes in the corridor frame (network attribute;
#: equals ``wms.control.contract.MERGE_S1_M``, asserted there).
MERGE_S1_M = 75.44
#: Internal junction-connector lane lengths of the registered merge network:
#: ``:1780739017_0_0`` (ramp -> merge-edge lane 2),
#: ``:1780739017_1_0`` / ``:1780739017_1_1`` (mainline entry -> merge edge),
#: ``:13638027693_0_0`` / ``:13638027693_0_1`` (merge edge -> exit edge).
MERGE_CONNECTOR_RAMP_LEN_M = 2.93
MERGE_CONNECTOR_ENTRY_LEN_M = 3.00
MERGE_CONNECTOR_EXIT_LEN_M = 8.00


def leader_rule_enabled(env=None) -> bool:
    """Whether the SUMO lane/link leader rule is enabled (read per call)."""
    source = os.environ if env is None else env
    return str(source.get(LEADER_RULE_ENV, "")).strip().lower() in _TRUTHY


def chain_s(lane, s_m):
    """Lane-chain coordinate ``c`` of a modelled (non-junction) vehicle.

    ``lane``: long tensor/array of frame lanes (0/1/2); ``s_m``: corridor
    coordinate.  Returns ``c`` with the same dtype as ``s_m``: the corridor
    ``s`` plus the connector lengths of the junctions the vehicle is
    downstream of (entry connector for ``s > 0`` on lanes 0/1, exit connector
    for ``s > MERGE_S1_M`` on lanes 0/1, ramp connector for ``s > 0`` on
    lane 2).  Unused for junction/absent rows (lane < 0), where the shift is
    immaterial.
    """
    import torch
    s = torch.as_tensor(s_m)
    lane_t = torch.as_tensor(lane).long()
    entry = (lane_t != 2) & (s > 0.)
    exit_ = (lane_t != 2) & (s > float(MERGE_S1_M))
    ramp = (lane_t == 2) & (s > 0.)
    return s + (entry.to(s) * float(MERGE_CONNECTOR_ENTRY_LEN_M)
                + exit_.to(s) * float(MERGE_CONNECTOR_EXIT_LEN_M)
                + ramp.to(s) * float(MERGE_CONNECTOR_RAMP_LEN_M))


def junction_chain_s(lane_carry, s_m):
    """Chain coordinate of a junction-transit vehicle (observable lower bound).

    A junction vehicle sits on an internal connector; its observable frame
    state is the carried last modelled (lane, s).  It is placed at the ENTRY of
    the connector ahead of that position - the geometric lower bound of its
    true chain position (conservative for an upstream ego: never overstates
    the gap).
    """
    import torch
    s = torch.as_tensor(s_m)
    return torch.where(s <= 0., torch.zeros_like(s),
                       torch.full_like(s, float(MERGE_CONNECTOR_ENTRY_LEN_M
                                                 + MERGE_S1_M)))
