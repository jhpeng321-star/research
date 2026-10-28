"""S-EXECPARITY3 SUMO lane/link-based leader identification (default OFF).

The deployment executor takes its leader from SUMO ``getLeader``: the nearest
vehicle ahead on the ego's physical LANE, following the lane's downstream links
(including the merging connectors) across junctions. The legacy WM frame instead
projected every vehicle onto the corridor ``s`` coordinate and took the nearest
same-frame-lane vehicle ahead in ``s``.  Because ``s`` is continuous across the
merge junctions while the physical lanes are not, that projection manufactures
PHANTOM leaders (S-EXECPARITY2 decomposition: 5.14 % of replay rows, 95.6 % of
the pooled squared error - in truth dominated by the hidden junction connectors:
ramp egos whose SUMO leader gap = frame gap + the 2.93 m ramp connector, and
mainline egos misled by the 3.00/8.00 m entry/exit connectors, plus mid-merge
vehicles at 0-2 m corridor projections whose SUMO leader is ~9 m away or
absent):

* an ego on the accel lane (frame lane 2, ``s >= 0``) must NOT see ramp
  vehicles (frame lane 2, ``s < 0``): they are upstream across the junction;
* a ramp ego's leader ACROSS the ramp connector is 2.93 m further than the
  corridor projection says; the same holds on the mainline across the entry
  (3.00 m) and exit (8.00 m) junctions;
* junction-transit vehicles (no frame lane) are ON a connector: they lead the
  vehicles still on the connector's upstream lane, and nobody on another lane
  leads from them.

The rule implemented here restores SUMO's semantics from observable/geometric
information only (frame lane, ``s``, junction flag, optional ``lane_carry`` and
the registered network's connector lengths - see
:mod:`wms.physics.lane_chain`): every vehicle is mapped to a monotone
lane-chain coordinate ``c`` and the leader is the nearest vehicle ahead in
``c`` on the same physical lane.  On this network each frame lane is a strictly
linear chain (mainline lanes: entry edge -> :1780739017_1_* -> merge edge ->
:13638027693_0_* -> exit edge; ramp: -> :1780739017_0_0 -> accel lane, a dead
end), so chain order == SUMO's link order for through traffic.  No truth driver
types, no vType files, no hand-set behavioural thresholds.

Default OFF: :func:`leader_gap_dv_sumo` then calls the frozen legacy
:func:`wms.control.native_cav.leader_gap_dv` unchanged (bitwise), and
``wms.physics.neighbours.lane_neighbour`` keeps its legacy body.

Measured on the 00:20 recordings (leader agreement vs the executor's logged
``getLeader``): mismatches fall 1,675/32,617 rows (5.14 %) -> 450 (1.38 %),
with zero previously-matched rows broken; the residual is dominated by
far leaders beyond SUMO's own brakeGap-based getLeader search budget (MSVehicle.cpp:
``dist == 0 -> brakeGap(speed) + minGap``) and junction-internal connector
rows - both unobservable in a WM frame and benign for the executed law.
"""
from __future__ import annotations

from wms.physics.lane_chain import (LEADER_RULE_ENV, chain_s, junction_chain_s,
                                    leader_rule_enabled)  # noqa: F401  (re-export)


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
