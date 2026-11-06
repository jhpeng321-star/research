"""LCEXEC: probe member role spec -> free_program MemberProgram (exact role semantics).

One function, :func:`member_program`, maps ONE probe member (``role`` + ``params`` exactly
as the probe Arm received them: SUMO global x, SUMO lane indices on the approach edge) to a
:class:`lc_program.MemberProgram` in the WM frame. The executor then reproduces the probe
Arm.control decisions for that role (release order, lcMode/tau activation, lane request
rule, speed law). Roles and their probe sources (verbatim semantics):

* ``hold`` (NEWMAP2 nm2_core / PLANGEN / PLANGEN2 / LOWCAV / LCMOVES): speed law to v_w,
  lcMode 513 at the first active second; releases hold_end, release_x (default X_B), anchor;
* ``pulse`` (dither): constant a for dur_s, no lcMode;
* ``tau`` (COOPKINDS ck_core / PLANGEN2): native longitudinal, tau + lcMode (lc_mode) on zone
  entry; releases hold_end, release_x, zone (before zone: inactive, after: release);
* ``merge_early``: native, lcMode 513, changeLane(target_lane) every second while in SUMO
  lane 0 and x >= trigger_x;
* ``zip_merge``: native, lcMode 512, releases merged (left SUMO lane 0 / approach) then
  fallback_x; changeLane(1) every second once x >= merge_x;
* ``zip_yield``: track the partner at -gap (kp, kv, dmax, amax), lcMode 513; release when
  the partner left SUMO lane 0 / approach;
* ``lane_to``: native, lcMode 513, changeLane(target_lane) while own SUMO lane < target and
  x < giveup_x;
* ``prog`` (LCMOVES lcm_core): optional lane change (lc_target, lc_from, lc_from_x,
  lc_giveup_x default X_B-300, lc_mode) + one speed action (hold / track / match / none),
  release_on_partner_merge, merger + fallback_x (default X_B-15); lcMode on action.

COOPKINDS/PLANGEN2 probe defaults are applied exactly as the probe did (hold_s absent =
never, release_x absent = never; hold role release_x absent = X_B).
"""
from __future__ import annotations

from .lc_program import (LC_MODE_POLICY, LC_MODE_TRACI_ONLY, X_B_PROBE, LaneReq, Lon, MemberProgram)

ROLES = ('hold', 'pulse', 'tau', 'merge_early', 'zip_merge', 'zip_yield', 'lane_to', 'prog')
COOPKINDS_ROLES = ('tau', 'merge_early', 'zip_merge', 'zip_yield', 'lane_to')


def _hold(p):
    return None if p.get('hold_s') is None else int(p['hold_s'])


def _rel(p, frame, default=None):
    x = p.get('release_x', default)
    return None if x is None else frame.s(x)


def member_program(role, params, frame, *, x_b=X_B_PROBE):
    """Exact MemberProgram of one probe member (``frame``: lc_program.Frame of the map)."""
    p = dict(params or {})
    start = int(p.get('start_s', 0))
    closing = frame.closing_lane
    if role == 'hold':
        anchor = p.get('anchor')
        releases = ('hold_end', 'release_s') + (('anchor',) if anchor is not None else ())
        return MemberProgram(
            lon=Lon('speed', v_w=float(p['v_w']), amax=float(p['amax']), dmax=float(p['dmax'])),
            start_el=start, hold_s=int(p['hold_s']), release_s=_rel(p, frame, x_b), anchor=anchor,
            anchor_release_s=None if anchor is None else frame.s(p['anchor_release_x']),
            lc_mode=LC_MODE_POLICY, mode_on='first_active', releases=releases, role='hold')
    if role == 'pulse':
        return MemberProgram(lon=Lon('accel', a=float(p['a'])), start_el=start, hold_s=int(p['dur_s']),
                             mode_on='never', releases=('hold_end',), role='pulse')
    common = dict(start_el=start, hold_s=_hold(p), release_s=_rel(p, frame), role=role)
    if role == 'tau':
        zone = p.get('zone')
        return MemberProgram(
            lon=Lon('native'), zone=None if zone is None else (frame.s(zone[0]), frame.s(zone[1])),
            tau=float(p['tau']), lc_mode=int(p.get('lc_mode', LC_MODE_POLICY)), mode_on='first_active',
            releases=('hold_end', 'release_s') + (('zone',) if zone is not None else ()), **common)
    if role == 'merge_early':
        return MemberProgram(
            lon=Lon('native'), lane=LaneReq(frame.wm(int(p.get('target_lane', 1))), 'from_lane',
                                            source_lane=closing, from_s=frame.s(p['trigger_x'])),
            lc_mode=LC_MODE_POLICY, mode_on='first_active', releases=('hold_end', 'release_s'), **common)
    if role == 'zip_merge':
        return MemberProgram(
            lon=Lon('native'), lane=LaneReq(frame.wm(1), 'from_lane', source_lane=closing,
                                            from_s=frame.s(p['merge_x'])),
            merger_lane=closing, fallback_s=frame.s(p['fallback_x']), lc_mode=LC_MODE_TRACI_ONLY,
            mode_on='first_active', releases=('hold_end', 'release_s', 'merged_zip', 'fallback'),
            mode_logged=True, **common)
    if role == 'zip_yield':
        return MemberProgram(
            lon=Lon('track', refs=(p['partner'],), off=-float(p['gap']), kp=float(p['kp']), kv=float(p['kv']),
                    dmax=float(p['dmax']), amax=float(p['amax'])),
            partner=p['partner'], partner_lane=closing, lc_mode=LC_MODE_POLICY, mode_on='first_active',
            releases=('hold_end', 'release_s', 'partner_left'), **common)
    if role == 'lane_to':
        return MemberProgram(
            lon=Lon('native'), lane=LaneReq(frame.wm(int(p['target_lane'])), 'right_of_target',
                                            giveup_s=frame.s(p['giveup_x'])),
            lc_mode=LC_MODE_POLICY, mode_on='first_active', releases=('hold_end', 'release_s'), **common)
    if role == 'prog':
        kind = p.get('speed')
        lon_kind = {'hold': 'speed', 'track': 'track', 'match': 'match', None: 'native'}[kind or None]
        lon = Lon(lon_kind, v_w=None if p.get('v_w') is None else float(p['v_w']),
                  amax=float(p.get('amax', 1.0)), dmax=float(p.get('dmax', 1.0)),
                  refs=tuple(p.get('track_ref') or ()), off=float(p.get('track_off', 0.0)),
                  kp=float(p.get('kp', 0.15)), kv=float(p.get('kv', 0.6)),
                  settle_gap=None if p.get('settle_gap') is None else float(p['settle_gap']),
                  from_el=int(p.get('sp_from', 0)))
        if lon_kind in ('speed', 'track', 'match') and 'dmax' not in p:
            raise ValueError('LCMOVES prog speed action without dmax (probe reads p["dmax"])')
        tl = p.get('lc_target')
        lane = None if tl is None else LaneReq(
            frame.wm(int(tl)), 'until_reached', from_el=int(p.get('lc_from', 0)),
            from_s=None if p.get('lc_from_x') is None else frame.s(p['lc_from_x']),
            giveup_s=frame.s(p.get('lc_giveup_x', x_b-300.0)))
        partner = p.get('release_on_partner_merge')
        merger = bool(p.get('merger'))
        releases = (('hold_end', 'release_s') + (('partner_left',) if partner is not None else ())
                    + (('merger', 'fallback') if merger else ()))
        return MemberProgram(
            lon=lon, lane=lane, partner=partner, partner_lane=closing if partner is not None else None,
            merger_lane=closing if merger else None,
            fallback_s=frame.s(p.get('fallback_x', x_b-15.0)) if merger else None,
            lc_mode=int(p.get('lc_mode', LC_MODE_POLICY)), mode_on='on_action', releases=releases,
            mode_logged=True, **common)
    raise ValueError(f'unknown probe role {role!r}')


__all__ = ['ROLES', 'COOPKINDS_ROLES', 'member_program']
