"""S-EXECPARITY3 SUMO lane/link-based leader identification (default OFF).

The deployment executor takes its leader from SUMO ``getLeader``: the nearest
vehicle ahead on the ego's physical LANE, following the lane's downstream links
(including the merging connectors) across junctions. The legacy WM frame instead
projected every vehicle onto the corridor ``s`` coordinate and took the nearest
same-frame-lane vehicle ahead in ``s``.  Because ``s`` is continuous across the
merge junctions while the physical lanes are not, that projection manufactures
PHANTOM leaders (S-EXECPARITY2 decomposition: 5.14 % of replay rows, 95.6 % of
the pooled squared error; ramp/merging vehicles placed 0.1-2 m ahead of an ego
whose SUMO leader is ~9 m away or absent):

* an ego on the accel lane (frame lane 2, ``s >= 0``) must NOT see ramp
  vehicles (frame lane 2, ``s < 0``): they are upstream across the junction;
* a ramp ego's leader ACROSS the ramp connector is further away by the
  connector's physical length (the corridor coordinate hides it); the same
  holds on the mainline across the entry and exit junctions;
* junction-transit vehicles (no frame lane) are ON a connector: they lead the
  vehicles still on the connector's upstream lane, and nobody upstream of the
  junction leads them.

The rule implemented here restores SUMO's semantics from observable/geometric
information only (frame lane, ``s``, junction flag, optional ``lane_carry`` and
the registered network's connector lengths): every vehicle is mapped to a
monotone lane-chain coordinate ``c`` (corridor ``s`` plus the hidden connector
lengths of the junctions upstream of it), and the leader is the nearest vehicle
ahead in ``c`` on the same physical lane.  On this network each frame lane is a
strictly linear chain (mainline lanes: entry edge -> :1780739017_1_* -> merge
edge -> :13638027693_0_* -> exit edge; ramp: -> :1780739017_0_0 -> accel lane,
a dead end), so chain order == SUMO's link order for through traffic.  No truth
driver types, no vType files, no hand-set behavioural thresholds.

Default OFF: :func:`leader_gap_dv_sumo` then calls the frozen legacy
:func:`wms.control.native_cav.leader_gap_dv` unchanged (bitwise), and
``wms.physics.neighbours.lane_neighbour`` keeps its legacy body.
"""
from __future__ import annotations

import os

#: Environment flag enabling the SUMO lane/link leader rule (default OFF).
LEADER_RULE_ENV = "C4_SUMO_LEADER_RULE"
_TRUTHY = ("1", "true", "yes", "on")


def leader_rule_enabled(env=None) -> bool:
    """Whether the SUMO lane/link leader rule is enabled (read per call)."""
    source = os.environ if env is None else env
    return str(source.get(LEADER_RULE_ENV, "")).strip().lower() in _TRUTHY


def chain_s(lane, s_m):
    """Lane-chain coordinate ``c`` of a modelled (non-junction) vehicle.

    ``lane``: long tensor/array of frame lanes (0/1/2, anything else -> no
    shift); ``s_m``: corridor coordinate.  Returns ``c`` with the same dtype as
    ``s_m``: the corridor ``s`` plus the connector lengths of the junctions the
    vehicle is downstream of (entry connector for ``s > 0`` on lanes 0/1, exit
    connector for ``s > MERGE_S1_M`` on lanes 0/1, ramp connector for
    ``s > 0`` on lane 2).  Pure :mod:`wms.control.contract` geometry.
    """
    import torch
    from .contract import (MERGE_CONNECTOR_ENTRY_LEN_M, MERGE_CONNECTOR_EXIT_LEN_M,
                           MERGE_CONNECTOR_RAMP_LEN_M, MERGE_S1_M)
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
    true chain position (conservative for an upstream ego: never overstates the
    gap).  Lanes 0/1/2 all reach chain coordinate 0.0 at the merge junction and
    ENTRY+MERGE_S1 at the exit junction, so the bound is lane-independent.
    """
    import torch
    from .contract import (MERGE_CONNECTOR_ENTRY_LEN_M, MERGE_S1_M)
    s = torch.as_tensor(s_m)
    return torch.where(s <= 0., torch.zeros_like(s),
                       torch.full_like(s, float(MERGE_CONNECTOR_ENTRY_LEN_M
                                                 + MERGE_S1_M)))


def leader_gap_dv_sumo(state):
    """SUMO-rule leader (bumper gap, ego-minus-leader dv, presence) [B,K].

    Default OFF: returns :func:`wms.control.native_cav.leader_gap_dv(state)`
    bitwise.  ON: nearest-ahead in the lane-chain coordinate on the same
    physical lane; junction-transit slots participate with their carried lane
    (``state["lane_carry"]``, long; when absent they stay non-candidates as in
    the legacy rule) at the connector-entry bound.  Bumper-gap floor and
    no-leader conventions are the frozen legacy ones.
    """
    import torch
    from .native_cav import leader_gap_dv
    if not leader_rule_enabled():
        return leader_gap_dv(state)
    present = state["present"].bool()
    lane = state["lane"].long()
    junction = state.get("junction")
    junction = (torch.zeros_like(present) if junction is None
                else torch.as_tensor(junction).bool())
    carry = state.get("lane_carry")
    carry = (torch.full_like(lane, -1) if carry is None
             else torch.as_tensor(carry).long())
    modelled = present & (lane >= 0) & ~junction
    on_connector = present & junction & (carry >= 0)
    eff_lane = torch.where(on_connector, carry, lane)
    eff_s = torch.where(on_connector, junction_chain_s(carry, state["s_m"]),
                        chain_s(lane, state["s_m"]))
    valid = modelled | on_connector
    s = torch.where(valid, eff_s, torch.zeros_like(eff_s))
    v = torch.where(valid, state["v_mps"], torch.zeros_like(eff_s)).clamp_min(0.0)
    length = state.get("length_m")
    if length is None:
        from wms.core.constants import DEFAULT_LENGTH_M
        length = torch.full_like(eff_s, float(DEFAULT_LENGTH_M))
    length = torch.where(valid, torch.as_tensor(length, dtype=v.dtype),
                         torch.zeros_like(v))
    k = s.shape[-1]
    ds = s[:, None, :] - s[:, :, None]
    other = ~torch.eye(k, dtype=torch.bool, device=s.device)[None]
    ahead = valid[:, :, None] & valid[:, None, :] \
        & (eff_lane[:, :, None] == eff_lane[:, None, :]) & (ds >= 0) & other
    distance, index = torch.where(ahead, ds, torch.full_like(ds, float("inf"))).min(-1)
    has = torch.isfinite(distance)
    gap = (torch.where(has, distance, torch.zeros_like(distance))
           - length.gather(-1, index)).clamp_min(0.1)
    dv = v - v.gather(-1, index)
    return torch.where(has, gap, torch.zeros_like(gap)), dv, has
