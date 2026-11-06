"""Registered gradual-release wall macro (W2, default OFF via ``wall_release``).

COOP_PROBE_REPORT (2026-11-06): the same verified wall helps on some CRN seeds and loses
up to -505 veh-s on others; the large losses are attributed to the abrupt release (a
box offset that drops from -depth to 0 in one control second) launching a shockwave.
This module adds ONE new macro shape for the ``wall`` family: hold, then a monotone ramp
back to the base policy over ``release_s`` seconds.

Shape (per member, policy-relative offset at integer second t after the move start):
    [0, onset)                         0
    [onset, onset+hold)                -depth                       (unchanged box)
    [onset+hold, onset+hold+R)         -depth * f(k, R), k = t-onset-hold
    later                              0
with f_linear(k, R) = 1 - (k+1/2)/R and f_cosine(k, R) = (1 + cos(pi (k+1/2)/R))/2 --
both strictly inside (0, 1), monotone decreasing, mean 1/2 (midpoint sampling of the
continuous ramp on the one-second control grid). ``step`` (R = 0) is the legacy box and
keeps the legacy bytes (no release keys in ``Move.to_dict``).

Registered ranges: R in RELEASE_WINDOW_S (integer seconds), depth in the existing
variety depth window (validated by ``moves.Move``), onset/hold in the existing variety
windows. Walls still release at once if any member reaches the observable merge zone
(``moves.offsets`` state guard); proposals therefore only keep a ramp that completes
before the members' observable constant-speed zone arrival.

Executable representation: ``plan(move, policy_profile)`` = the G5 executable fields
(``group_lane.executable_fields``: profile = policy_profile + macro offsets, plus the
stored perturbation_profile) and an explicit ``wall_macro``; ``from_plan`` is the exact
binding (like ``group_lane.from_plan``), used by ``moves.from_plan`` for such plans.
"""
import math

import numpy as np

from . import moves
from .variety import VARIETY_REGISTERED

RELEASE_SHAPES = ('linear', 'cosine')
PROPOSAL_SHAPES = ('step',) + RELEASE_SHAPES
RELEASE_WINDOW_S = (5, 30)
WALL_RELEASE_REGISTERED = dict(
    release_shapes=RELEASE_SHAPES, proposal_shapes=PROPOSAL_SHAPES,
    release_window_s=RELEASE_WINDOW_S,
    depth_window=tuple(VARIETY_REGISTERED['variety_depth_window']),
    hold_window_s=tuple(VARIETY_REGISTERED['variety_hold_window_s']),
    onset_window_s=tuple(VARIETY_REGISTERED['variety_onset_window_s']),
    provenance='W2 owner 2026-11-06: gradual release after hold (probe losses attributed to abrupt release)')
DESCRIPTOR = ('release_frac_of_max', 'shape_step', 'shape_linear', 'shape_cosine')
DESCRIPTOR_DIM = len(DESCRIPTOR)


def release_fraction(shape, k, release_s):
    """Fraction of the hold depth still applied k seconds into a release of length R."""
    if shape not in RELEASE_SHAPES:
        raise ValueError(f'unregistered release shape {shape!r}')
    if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k < int(release_s):
        raise ValueError('release second outside [0, release_s)')
    x = (k+.5)/float(release_s)
    if shape == 'linear':
        return 1.-x
    return .5*(1.+math.cos(math.pi*x))


def validate(move):
    """Called by moves.Move for every non-default release (fail closed)."""
    shape, release = move.release_shape, move.release_s
    if move.family != 'wall':
        raise ValueError('only the wall family has a registered gradual release')
    if shape not in RELEASE_SHAPES:
        raise ValueError(f'release_shape must be one of {RELEASE_SHAPES} (step has release_s 0)')
    if not isinstance(release, int) or isinstance(release, bool) \
            or not RELEASE_WINDOW_S[0] <= release <= RELEASE_WINDOW_S[1]:
        raise ValueError(f'release_s must be an integer in the registered window {RELEASE_WINDOW_S} s')


def duration(move):
    """Seconds until the move's offsets are 0 for good (onset + hold + release)."""
    return int(move.onset)+int(move.hold)+int(getattr(move, 'release_s', 0))


def descriptor(move):
    """Trigger-head release descriptor; None (the 'none' slot / padding) is all zeros."""
    if move is None:
        return [0.]*DESCRIPTOR_DIM
    get = (lambda k, d: move.get(k, d)) if isinstance(move, dict) else (lambda k, d: getattr(move, k, d))
    shape, release = get('release_shape', 'step'), int(get('release_s', 0))
    return [release/float(RELEASE_WINDOW_S[1])] + [float(shape == s) for s in PROPOSAL_SHAPES]


def _arrival(rows):
    return min(((moves._ZONE_S0-float(r['s_m']))/float(r['v_mps'])
                if float(r['v_mps']) > 0 else math.inf) for r in rows)


def with_release(move, rows, rng):
    """Proposal-time shape draw for a feasible wall candidate (registered, B = R).

    Draws a shape uniformly from PROPOSAL_SHAPES and R uniformly from the registered
    window (always two draws, so the stream is shape-independent). The ramp must end
    before the members' observable constant-speed merge-zone arrival: R is first capped
    by the room left after the registered minimum hold, then the hold is shortened (never
    below its registered minimum) so onset + hold + R <= arrival. If less than the
    registered minimum ramp fits, the candidate stays the legacy step wall.
    """
    shape = PROPOSAL_SHAPES[int(rng.integers(len(PROPOSAL_SHAPES)))]
    release = int(rng.uniform(RELEASE_WINDOW_S[0], RELEASE_WINDOW_S[1]+1))
    release = min(release, RELEASE_WINDOW_S[1])
    if move is None or move.family != 'wall' or shape == 'step':
        return move
    by_vid = {r['vid']: r for r in rows}
    if any(v not in by_vid for v in move.members):
        return move
    arrival = _arrival([by_vid[v] for v in move.members])
    hold = move.hold
    if math.isfinite(arrival):
        # The ramp must finish before the zone. The hold may give way to the ramp, never
        # below the registered minimum hold; otherwise the candidate stays a step wall.
        release = min(release, int(math.floor(arrival-move.onset-moves._HOLD_LO)))
        if release < RELEASE_WINDOW_S[0]:
            return move
        hold = min(hold, int(math.floor(arrival-move.onset-release)))
    from dataclasses import replace
    return replace(move, hold=hold, release_s=release, release_shape=shape)


def plan(move, policy_profile, *, plan_id=None):
    """Executable verified-plan dict for a wall macro (step or gradual release)."""
    from .group_lane import executable_fields
    if not isinstance(move, moves.Move) or move.family != 'wall':
        raise ValueError('wall_release.plan needs a wall moves.Move')
    dur = duration(move)
    fields = executable_fields(move, dur, policy_profile)
    shape = move.release_shape if move.release_s else 'step'
    pid = plan_id or (f'wt_wall_{"_".join(move.members)}_o{move.onset}_h{move.hold}_d{move.depth:g}'
                      f'_{shape}{move.release_s}')
    return dict(id=pid, family='wall', members=list(move.members), duration_s=dur,
                wall_macro=move.to_dict(), **fields)


def from_plan(plan):
    """Exact executable binding of a ``wall_macro`` plan (never a lossy projection)."""
    from .group_lane import _FORBIDDEN_REPRESENTATIONS, _matrix, macro_offsets
    macro = plan.get('wall_macro')
    if not isinstance(macro, dict):
        raise ValueError('wall plan requires an explicit wall_macro')
    move = moves.Move.from_dict(macro)
    if move.family != 'wall' or list(move.members) != plan.get('members'):
        raise ValueError('wall_macro member/family binding mismatch')
    dur = plan.get('duration_s')
    if type(dur) is not int or dur != duration(move):
        raise ValueError('wall plan duration_s must equal onset + hold + release_s')
    for key in ('profile', 'policy_profile', 'perturbation_profile'):
        if key not in plan:
            raise ValueError(f'wall plan requires {key} (incomplete executable representation)')
    if any(key in plan for key in _FORBIDDEN_REPRESENTATIONS):
        raise ValueError('wall plan carries a conflicting knot representation')
    shape = (len(move.members), dur)
    profile = _matrix(plan['profile'], shape, 'profile')
    base = _matrix(plan['policy_profile'], shape, 'policy_profile')
    stored = _matrix(plan['perturbation_profile'], shape, 'perturbation_profile')
    wanted = macro_offsets(move, dur)
    if not np.allclose(profile-base, wanted, atol=1e-8, rtol=0) \
            or not np.allclose(stored, wanted, atol=1e-8, rtol=0):
        raise ValueError('wall executable profile does not equal policy reference plus macro offsets')
    from .plans import acceleration_at
    for t in range(dur):
        executed = acceleration_at(plan, t)
        if set(executed) != set(move.members) or any(
                abs(executed[v]-profile[i, t]) > 1e-8 for i, v in enumerate(move.members)):
            raise ValueError('wall executor does not run the bound executable profile')
    return move
