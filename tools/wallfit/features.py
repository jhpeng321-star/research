"""Observable-only features for (situation, joint move) pairs + the frozen WM's own outputs.

Inputs are the CAV-sensed observable rows (vid, s_m, v_mps, lane, is_cav, length_m) of the WM
query roster (or, when no WM query exists, the collection CONTEXT rows), the move
(family, members, onset, hold, depth), the WM's own predictions (mean synergy, saving vs policy,
best-solo saving, draws) and the WM latent belief (frozen driver-encoder outputs per row).
World configuration labels (cell / demand / penetration / variant / arm) are NEVER features.

Pure python + numpy (no torch) so the default-OFF ranking path never imports torch.
"""
import math
import re

import numpy as np

FAMILIES = ('wall', 'easing', 'ramp_assist', 'yield_lane', 'open_gap_merge', 'gap_open')
RAMP_END_S = 75.44            # contract RSU_RAMP_S_M upper bound (ramp lane ends / merge point)
BINS = ((-400., -200.), (-200., 0.), (0., 100.))
SLOG_SCALE = 10.              # veh-s
_ID_RE = re.compile(r'(?:mainline|ramp)\.\d+')
_TAIL_RE = re.compile(r'_o(\d+)_h(\d+)_d([0-9.]+)$')


# ------------------------------------------------------------------ moves

def parse_plan_id(plan_id):
    """Best-effort (family, members, onset, hold, depth) from probe/mining plan ids such as
    'g_wall_mainline.191_mainline.176_mainline.192_o0_h30_d1' or
    'a_yield_lane_mainline.132_ramp.26_o29_h20_d0.3'. Missing parts are None."""
    pid = str(plan_id or '')
    head = pid.split('__')[0]
    family = None
    for fam in sorted(FAMILIES, key=len, reverse=True):
        if re.search(r'(^|_)' + re.escape(fam) + r'_', head):
            family = fam
            break
    members = _ID_RE.findall(head)
    m = _TAIL_RE.search(head)
    onset, hold, depth = (int(m.group(1)), int(m.group(2)), float(m.group(3))) if m else (None, None, None)
    return dict(family=family, members=members, onset=onset, hold=hold, depth=depth)


def move_from_plan(plan):
    """Move dict from a probe/mining plan (probe ease plans or accepted-G5 g5_move plans)."""
    parsed = parse_plan_id(plan.get('id') or plan.get('plan_id'))
    family = plan.get('family') or parsed['family']
    members = list(plan.get('members') or parsed['members'])
    q = plan.get('probe') or plan.get('g5_move') or {}
    onset = q.get('onset', q.get('onset_s', parsed['onset']))
    hold = q.get('hold', q.get('hold_s', parsed['hold']))
    depth = q.get('depth', parsed['depth'])
    if hold is None and plan.get('duration_s') is not None and onset is not None:
        hold = int(plan['duration_s']) - int(onset)
    return normalize_move(dict(family=family, members=members, onset=onset, hold=hold, depth=depth))


def move_from_action(action, plan_id=None):
    """Move dict from a WM action (wm_move / wm_role_schedule). The probe labels every ease plan
    'ramp_assist' inside wm_move, so the plan id's family wins when it is known."""
    mv = action.get('wm_move') or {}
    rs = action.get('wm_role_schedule') or {}
    parsed = parse_plan_id(plan_id) if plan_id else dict(family=None)
    if mv:
        fam = parsed.get('family') or mv.get('family')
        return normalize_move(dict(family=fam, members=list(mv['members']), onset=mv.get('onset'),
                                   hold=mv.get('hold'), depth=mv.get('depth')))
    if rs:
        fam = parsed.get('family') or rs.get('family')
        return normalize_move(dict(family=fam, members=list(rs['members']), onset=rs.get('onset_s'),
                                   hold=rs.get('hold_s'), depth=rs.get('depth')))
    if plan_id:
        return normalize_move(parse_plan_id(plan_id))
    raise ValueError('action carries no wm_move / wm_role_schedule and no plan id')


def normalize_move(move):
    fam = move.get('family')
    if fam == 'g5_yield_lane':
        fam = 'yield_lane'
    return dict(family=fam, members=[str(v) for v in move.get('members') or []],
                onset=None if move.get('onset') is None else int(move['onset']),
                hold=None if move.get('hold') is None else int(move['hold']),
                depth=None if move.get('depth') is None else float(move['depth']))


def move_key(move):
    """Order-free identity of a joint move (plan-id prefixes such as 'lead_'/'g_' are ignored)."""
    d = move.get('depth')
    return '|'.join([str(move.get('family')), ','.join(sorted(move.get('members') or [])),
                     f"o{move.get('onset')}", f"h{move.get('hold')}",
                     'd' + ('None' if d is None else f'{float(d):g}')])


# ------------------------------------------------------------------ transforms

def slog(x, scale=SLOG_SCALE):
    return math.copysign(math.log1p(abs(float(x)) / scale), float(x))


def inv_slog(y, scale=SLOG_SCALE):
    return math.copysign(math.expm1(abs(float(y))) * scale, float(y))


# ------------------------------------------------------------------ observable situation + move

def _mean(xs, default=0.):
    return float(sum(xs) / len(xs)) if xs else float(default)


OBS_NAMES = (
    ['n_rows', 'n_cav', 'ramp_n', 'ramp_stopped', 'ramp_head_rel', 'ramp_mean_v', 'main_min_v_merge']
    + [f'lane{ln}_{a:g}_{b:g}_{k}' for ln in (0, 1) for a, b in BINS for k in ('n', 'v')]
    + [f'fam_{f}' for f in FAMILIES] + ['fam_other']
    + ['n_members', 'members_present', 'onset', 'hold', 'depth', 'impulse',
       'm_s_mean', 'm_s_min', 'm_s_max', 'm_spread', 'm_v_mean', 'm_v_min',
       'm_lane0', 'm_lane1', 'm_lane2', 'lead_gap', 'lead_dv', 'hdv_in_span',
       'up_n', 'up_v', 'm_cav_frac'])


def situation_features(rows, move):
    """Fixed-length observable feature vector (OBS_NAMES order) for one (situation, move)."""
    rows = [r for r in rows if r.get('s_m') is not None]
    by = {str(r['vid']): r for r in rows}
    ramp = [r for r in rows if int(r['lane']) == 2]
    main = [r for r in rows if int(r['lane']) in (0, 1)]
    head = max((float(r['s_m']) for r in ramp if float(r['s_m']) <= RAMP_END_S), default=RAMP_END_S)
    near = [float(r['v_mps']) for r in main if -200. <= float(r['s_m']) <= 100.]
    out = [len(rows) / 50., sum(bool(r['is_cav']) for r in rows) / 20., len(ramp) / 10.,
           sum(float(r['v_mps']) < 2. for r in ramp) / 10., (head - RAMP_END_S) / 100.,
           _mean([float(r['v_mps']) for r in ramp]) / 30., (min(near) if near else 30.) / 30.]
    for ln in (0, 1):
        for a, b in BINS:
            xs = [float(r['v_mps']) for r in main if int(r['lane']) == ln and a <= float(r['s_m']) < b]
            out += [len(xs) / 10., _mean(xs) / 30.]
    fam = move.get('family')
    out += [float(fam == f) for f in FAMILIES] + [float(fam not in FAMILIES)]
    members = list(move.get('members') or [])
    present = [by[v] for v in members if v in by]
    onset, hold, depth = (move.get('onset') or 0), (move.get('hold') or 0), (move.get('depth') or 0.)
    out += [len(members) / 6., len(present) / max(1, len(members)), onset / 30., hold / 30., float(depth),
            len(members) * hold * float(depth) / 100.]
    if present:
        s = [float(r['s_m']) - head for r in present]
        v = [float(r['v_mps']) for r in present]
        lanes = [int(r['lane']) for r in present]
        lead = max(present, key=lambda r: float(r['s_m']))
        ahead = [r for r in rows if int(r['lane']) == int(lead['lane']) and float(r['s_m']) > float(lead['s_m'])]
        if ahead:
            ldr = min(ahead, key=lambda r: float(r['s_m']))
            gap = float(ldr['s_m']) - float(lead['s_m']) - float(ldr.get('length_m') or 5.)
            dv = float(ldr['v_mps']) - float(lead['v_mps'])
        else:
            gap, dv = 200., 0.
        lo, hi = min(float(r['s_m']) for r in present), max(float(r['s_m']) for r in present)
        span = [r for r in main if not r['is_cav'] and lo <= float(r['s_m']) <= hi and str(r['vid']) not in members]
        up = [float(r['v_mps']) for r in main if lo - 200. <= float(r['s_m']) < lo and str(r['vid']) not in members]
        out += [_mean(s) / 100., min(s) / 100., max(s) / 100., (hi - lo) / 100., _mean(v) / 30., min(v) / 30.,
                lanes.count(0) / len(lanes), lanes.count(1) / len(lanes), lanes.count(2) / len(lanes),
                min(gap, 200.) / 100., dv / 10., len(span) / 10., len(up) / 10., _mean(up, 30.) / 30.,
                sum(bool(r['is_cav']) for r in present) / len(present)]
    else:
        out += [0.] * 15
    vec = np.asarray(out, dtype=np.float32)
    assert vec.shape == (len(OBS_NAMES),)
    return vec


# ------------------------------------------------------------------ frozen-WM outputs

WM_NAMES = ('wm_synergy', 'wm_saving_vs_policy', 'wm_best_solo_saving')
WM_EXTRA_NAMES = ('wm_synergy_draw_std', 'wm_h300')


def wm_features(wm):
    """(reg[3], extra[2], mask) from the WM's own prediction dict (None -> zeros, mask 0).
    reg is in slog units and is the residual base of the head (WM_NAMES order)."""
    if not wm or wm.get('mean_synergy') is None:
        return np.zeros(3, np.float32), np.zeros(2, np.float32), 0.
    reg = [slog(wm['mean_synergy']), slog(wm.get('saving_vs_policy') or 0.), slog(wm.get('best_solo_saving') or 0.)]
    draws = wm.get('draws') or []
    std = float(np.std(np.asarray(draws, float))) if len(draws) > 1 else 0.
    extra = [slog(std), float(int(wm.get('horizon_s') or 120) >= 300)]
    return np.asarray(reg, np.float32), np.asarray(extra, np.float32), 1.


# ------------------------------------------------------------------ WM latent belief

def latent_row_dim(latent):
    return int(np.asarray(latent['embedding']).shape[-1] + np.asarray(latent['values']).shape[-1]
               + np.asarray(latent['uncertainty']).shape[-1])


def latent_features(latent, rows, move, dim):
    """Pooled frozen-WM latent: mean over present members ++ mean over non-member rows within
    100 m of the members' span. Returns (vec[2*dim], mask). Missing latent -> zeros, mask 0."""
    if dim <= 0:
        return np.zeros(0, np.float32), 0.
    if not latent:
        return np.zeros(2 * dim, np.float32), 0.
    per = np.concatenate([np.asarray(latent['embedding'], np.float32), np.asarray(latent['values'], np.float32),
                          np.asarray(latent['uncertainty'], np.float32)], -1)
    if per.shape[-1] != dim:
        raise ValueError(f'latent width {per.shape[-1]} != head latent width {dim}')
    slot = {str(v): i for i, v in enumerate(latent['vids'])}
    members = [v for v in move.get('members') or [] if v in slot]
    if not members:
        return np.zeros(2 * dim, np.float32), 0.
    s = {str(r['vid']): float(r['s_m']) for r in rows}
    ms = [s[v] for v in members if v in s]
    lo, hi = (min(ms), max(ms)) if ms else (-1e9, 1e9)
    others = [v for v in slot if v not in members and v in s and lo - 100. <= s[v] <= hi + 100.]
    mvec = per[[slot[v] for v in members]].mean(0)
    ovec = per[[slot[v] for v in others]].mean(0) if others else np.zeros(dim, np.float32)
    return np.concatenate([mvec, ovec]).astype(np.float32), 1.


def example_inputs(example, latent=None, latent_dim=0):
    """All head inputs for one dataset pair (or one rank-time candidate)."""
    move = example['move']
    rows = example['situation']['rows']
    obs = situation_features(rows, move)
    reg, extra, wm_mask = wm_features(example.get('wm'))
    lat, lat_mask = latent_features(latent, rows, move, latent_dim)
    return dict(obs=np.concatenate([obs, extra]).astype(np.float32), wm=reg, wm_mask=np.float32(wm_mask),
                lat=lat, lat_mask=np.float32(lat_mask))


INPUT_NAMES = tuple(OBS_NAMES) + WM_EXTRA_NAMES
