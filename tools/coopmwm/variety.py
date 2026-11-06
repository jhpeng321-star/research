"""Option ``search_families`` for the MPPI/niche plan search (default OFF, byte-identical).

Owner direction 2026-11-04: search for MORE VARIED cooperative plans (walls, gap
opening, ramp assist) WITHOUT forcing them: structured candidates only enter the
same candidate pool as the free MPPI/niche noise, capped to a registered share,
and the world-model score plus the exact SUMO verification decide what survives.
With no ``variety_*`` config key this module is never imported by ``mppi_search``.

1. Wider group selection (``decision_groups``): ADDITIVE groups, the registered
   chained groups stay untouched and first. (a) ``wall``: upstream multi-lane
   abreast CAV groups - the registered chaining/window rules re-run with
   ``chain_m = variety_wall_span_m`` and only components spanning at least
   ``variety_wall_min_lanes`` distinct lanes are kept; (b) ``gap_open``: ramp
   CAVs inside the observable merge zone with the mainline CAVs they would
   merge into. Geometry only uses the existing observable conventions (ramp
   lane = 2 = ``merge_lane``, mainline target lane = 1 = ``merge_target_lane``,
   the TRAIN observable merge-zone projection s_m in [-100, 75.44] from
   ``coopfree.selection``, the inherited mainline association window 100 m from
   ``coopbandit.protocol.group`` and the wall law span 75 m from
   ``src/wms/physics/wall_exec.MAX_WALL_SPAN_M``); every distance is a
   registered config value below. Duplicate member sets are dropped; groups
   overlap between kinds is possible but harmless (collection never executes
   guide plans; committed-plan exclusion still applies at the decision sites).
2. Structured candidate GENERATORS (``pool_candidates``): wall = abreast CAVs
   synchronise speed then decelerate together for a sampled hold then release
   (open-loop profile of the wall v2.1 pace law ``clip((target - v)/dt, lo, hi)``,
   integrated from the observable v0 exactly like ``centre_feasibility``);
   gap_open = a mainline CAV eases (decelerates) ahead of the predicted arrival
   of the ramp partner then recovers; ramp_assist = the ramp CAV and the
   mainline CAV ahead coordinate timing (mainline eases, ramp synchronises to
   the eased speed). Parameters (onset, hold, depth, members, duration) are
   sampled per candidate from the search's own child stream and every profile
   goes through the SAME ``build_plan`` bounds + feasibility clamp as free
   candidates; a candidate the clamp erases is marked ``policy_duplicate`` by
   the existing rule. Structured candidates TAKE ordinary population slots
   (total population, batch widths and WM cost are unchanged); at most
   ``floor(variety_structured_share * population)`` enter a search, so the free
   search dominates (default share 0.25). Nothing guarantees a structured
   candidate reaches the top-k/top-8: the WM score decides.
3. R fairness: ``mppi_search.sample_unscored`` draws structured candidates from
   the SAME family mix with random parameters under the SAME registered keys
   (share cap honoured by the same draw), so arm R searches the same direction
   blind; with no variety config its draws are byte-identical.
4. Family tags: every searched candidate gets ``variety_family`` in
   ``('free', 'wall', 'gap_open', 'ramp_assist')`` (free = MPPI/niche/library/
   centre/mappo/proposer seeds). The tag rides inside the plan dict into the
   SUMO branch specs and receipts, so bank reports can count verified/admitted
   plans per family (``family_counts``).
"""
import math
import numpy as np

from .map_descriptor import FRAME  # N1: model-frame geometry (map descriptor frame contract)

# Registered option config (all of it; unknown variety_* keys are rejected even
# when the option is off, mirroring coop_search.split_config). Documented
# distances: variety_wall_span_m mirrors the wall v2.1 executor's registered
# MAX_WALL_SPAN_M; variety_merge_zone_m is the TRAIN observable merge-zone
# projection (coopfree.selection features zone); variety_gap_link_m is the
# inherited observable mainline association window (coopbandit.protocol.group).
# Generator ranges are sampled uniformly per candidate; profiles are clamped by
# the existing bounds/feasibility clamp, never trusted.
VARIETY_REGISTERED = dict(
    variety_families=False,
    variety_structured_share=0.25,
    variety_wall_span_m=75.0,
    variety_wall_min_lanes=2,
    variety_merge_zone_m=FRAME.local_zone_s_m,
    variety_gap_link_m=100.,
    variety_ramp_lane=FRAME.merge_lane,           # ramp (merge map) / closing lane (lane drop)
    variety_mainline_lane=FRAME.main_lanes[1],
    variety_onset_window_s=(0., 30.),
    variety_hold_window_s=(20., 90.),
    variety_depth_window=(0.3, 1.0),
)
FAMILIES = ('free', 'wall', 'gap_open', 'ramp_assist')
STRUCTURED = ('wall', 'gap_open', 'ramp_assist')
CONTROL_DT_S = 1.0   # one-second MAPPO control interval (the plant's decision tick)


def split_config(cfg):
    """Pop every ``variety_*`` key from ``cfg`` (in place). Returns the resolved variety
    config, or None when the option is off (then cfg carries no variety key:
    byte-identical output). Mirrors ``coop_search.split_config``; every registered value
    is checked even while the option is off."""
    raw = {k: cfg.pop(k) for k in [k for k in cfg if k.startswith('variety_')]}
    unknown = set(raw)-set(VARIETY_REGISTERED)
    if unknown:
        raise ValueError(f'unregistered variety config keys {sorted(unknown)}')
    v = dict(VARIETY_REGISTERED, **raw)
    share = float(v['variety_structured_share'])
    if not 0. <= share <= 1:
        raise ValueError('variety_structured_share must be within [0, 1]')
    for key in ('variety_wall_span_m', 'variety_gap_link_m'):
        if not float(v[key]) > 0 or not math.isfinite(float(v[key])):
            raise ValueError(f'{key} must be finite and > 0')
    zone = tuple(float(x) for x in v['variety_merge_zone_m'])
    if len(zone) != 2 or not (math.isfinite(zone[0]) and math.isfinite(zone[1]) and zone[0] < zone[1]):
        raise ValueError('variety_merge_zone_m must be an ordered finite (s0, s1) pair')
    if int(v['variety_wall_min_lanes']) < 2:
        raise ValueError('variety_wall_min_lanes must be >= 2 (multi-lane abreast groups)')
    if int(v['variety_ramp_lane']) == int(v['variety_mainline_lane']):
        raise ValueError('variety_ramp_lane and variety_mainline_lane must differ')
    for key in ('variety_onset_window_s', 'variety_hold_window_s', 'variety_depth_window'):
        w = tuple(float(x) for x in v[key])
        if len(w) != 2 or not (all(math.isfinite(x) for x in w) and 0 <= w[0] <= w[1]):
            raise ValueError(f'{key} must be an ordered finite (lo, hi) window with lo >= 0')
    if not raw.get('variety_families'):
        return None
    cfg.update({k: v[k] for k in VARIETY_REGISTERED})
    return v


def plan_config(config=None):
    """Resolved variety config from a merged search-config dict (a COPY is split; the
    caller's dict keeps its keys). None when the option is off."""
    cfg = dict(config or {})
    return split_config(cfg)


# --------------------------------------------------------------------------- group selection

def wall_groups(rows, vcfg, exclude=()):
    """Upstream multi-lane abreast CAV groups: the registered chaining/window rules
    re-run with chain_m = variety_wall_span_m; only components spanning at least
    ``variety_wall_min_lanes`` distinct lanes qualify (wall candidates)."""
    from .mppi_search import group_components
    out = []
    for g in group_components(rows, chain_m=float(vcfg['variety_wall_span_m']), exclude=exclude):
        if len({int(r['lane']) for r in g['rows']}) >= int(vcfg['variety_wall_min_lanes']):
            out.append(dict(g, variety_kind='wall',
                            rule=dict(g['rule'], kind='wall',
                                      span_m=float(vcfg['variety_wall_span_m']),
                                      min_lanes=int(vcfg['variety_wall_min_lanes']))))
    return out


def gap_groups(rows, vcfg, exclude=()):
    """Gap-opening / ramp-assist candidates: ramp CAVs inside the observable merge
    zone together with the mainline CAVs within the registered link window they
    would merge into (roles: the mainline CAV ahead eases, the ramp CAV arrives)."""
    from .mppi_search import REGISTERED, _digest
    s0, s1 = (float(x) for x in vcfg['variety_merge_zone_m'])
    link = float(vcfg['variety_gap_link_m'])
    ramp_lane, main_lane = int(vcfg['variety_ramp_lane']), int(vcfg['variety_mainline_lane'])
    max_size, min_size = int(REGISTERED['max_size']), int(REGISTERED['min_size'])
    cavs = [dict(r) for r in rows if r['is_cav'] and r['vid'] not in set(exclude)]
    ramps = sorted((r for r in cavs if int(r['lane']) == ramp_lane and s0 <= r['s_m'] <= s1),
                   key=lambda r: (r['s_m'], r['vid']))
    mains = [r for r in cavs if int(r['lane']) == main_lane and s0 <= r['s_m'] <= s1]
    out = []
    for r in ramps:
        members_rows = sorted([r]+[p for p in mains if abs(p['s_m']-r['s_m']) <= link],
                              key=lambda x: (-x['s_m'], x['lane'], x['vid']))[:max_size]
        if len(members_rows) < min_size:
            continue
        members = [x['vid'] for x in members_rows]
        out.append(dict(id='G'+_digest(members)[:12], members=members, size=len(members_rows),
                        lanes=sorted({int(x['lane']) for x in members_rows}),
                        spread_m=float(members_rows[0]['s_m']-members_rows[-1]['s_m']),
                        leader_s_m=float(members_rows[0]['s_m']),
                        rows=[{k: x[k] for k in ('vid', 's_m', 'v_mps', 'lane', 'is_cav')} for x in members_rows],
                        rule=dict(chain_m=link, max_size=max_size, min_size=min_size),
                        variety_kind='gap_open'))
    return out


def decision_groups(rows, vcfg=None, exclude=()):
    """Decision-time groups: the registered ``group_components`` output first
    (unchanged, byte-identical when ``vcfg`` is None), then ADDITIONAL wall /
    gap_open groups (duplicate member sets dropped, deterministic order)."""
    from .mppi_search import group_components
    base = group_components(rows, exclude=exclude)
    if vcfg is None:
        return base
    taken = {frozenset(g['members']) for g in base}
    out = list(base)
    for extra in wall_groups(rows, vcfg, exclude)+gap_groups(rows, vcfg, exclude):
        key = frozenset(extra['members'])
        if key in taken:
            continue
        taken.add(key)
        out.append(extra)
    return out


# --------------------------------------------------------------------------- structured generators

def _rows_of(group, members):
    by = {r['vid']: r for r in group['rows']}
    return [by[v] for v in members]


def _lane_counts(group, vcfg):
    rows = group['rows']
    ramp = [r for r in rows if int(r['lane']) == int(vcfg['variety_ramp_lane'])]
    main = [r for r in rows if int(r['lane']) == int(vcfg['variety_mainline_lane'])]
    return ramp, main


def _pace_row(v0, targets, bounds, feasibility, index):
    """Open-loop one-member acceleration row pacing to per-second target speeds.

    The wall v2.1 law's pace rule a = clip((target - v)/dt, lo, hi), integrated
    from the observable v0 with the same actuator clip as ``centre_feasibility``
    (allowed-speed ceiling/floor) when resolved; bounds otherwise."""
    lo, hi = float(bounds[0]), float(bounds[1])
    allowed = None if feasibility is None else float(feasibility['allowed'][index])
    v, out = float(v0), np.empty(len(targets))
    for t, target in enumerate(targets):
        lo_t, hi_t = lo, hi
        if allowed is not None:
            lo_t, hi_t = max(lo, -v/CONTROL_DT_S), min(hi, (allowed-v)/CONTROL_DT_S)
        a = float(np.clip((float(target)-v)/CONTROL_DT_S, lo_t, hi_t))
        out[t] = a
        v = max(0., v+CONTROL_DT_S*a)
    return out


def _clamp_window(onset, hold, D):
    """At least one synchronise/ease second, one hold second, release inside D."""
    onset = int(np.clip(onset, 1, max(1, D-2)))
    hold = int(np.clip(hold, 1, D-onset))
    return onset, hold


def _ease_targets(v, D, onset, hold, depth):
    """Member's own speed, eased below by ``depth`` during [onset, onset+hold), then
    recovered (release) - the gap_open/ramp_assist ease-recover shape."""
    targets = np.full(D, float(v))
    onset, hold = _clamp_window(onset, hold, D)
    targets[onset:onset+hold] = float(v)-depth
    return targets


def _finish(family, group, P, rows, D, bounds, regime, feas):
    """build_plan (bounds + feasibility clamp, policy_duplicate rule, canonical
    re-grid) with the family tag. Origin stays the registered 'sampled'; the
    family distinction lives only in the additive ``variety_family`` key."""
    from .mppi_search import build_plan
    p = build_plan(group, P, None, int(D), bounds, 'sampled', regime, 'V'+family[0],
                   absolute=np.stack([np.asarray(r, float) for r in rows]), feasibility=feas)
    p['variety_family'] = family
    return p


def wall_candidate(group, P, bounds, regime, rng, vcfg, feas, durations):
    """WALL: abreast members synchronise to the group mean observable speed, then
    decelerate together at a sampled depth for a sampled hold, then release."""
    members = group['members']
    P = np.asarray(P, float)
    if len(members) < 2 or len({int(r['lane']) for r in group['rows']}) < int(vcfg['variety_wall_min_lanes']):
        return None
    D = int(durations[int(rng.choice(len(durations)))])
    v0 = [float(r['v_mps']) for r in _rows_of(group, members)]
    v_ref = float(np.mean(v0))
    depth = float(rng.uniform(*vcfg['variety_depth_window']))
    onset, hold = _clamp_window(rng.uniform(*vcfg['variety_onset_window_s']),
                                rng.uniform(*vcfg['variety_hold_window_s']), D)
    rows = []
    for i, v in enumerate(v0):
        targets = np.full(D, v_ref)                      # synchronise
        targets[onset:onset+hold] = v_ref-depth          # decelerate together
        targets[onset+hold:] = v                         # release to the member's own speed
        rows.append(_pace_row(v, targets, bounds, feas, i))
    return _finish('wall', group, P, rows, D, bounds, regime, feas)


def gap_open_candidate(group, P, bounds, regime, rng, vcfg, feas, durations):
    """GAP_OPEN: a mainline member eases ahead of the ramp partner's predicted
    arrival, then recovers; other members keep the policy forecast."""
    members = group['members']
    P = np.asarray(P, float)
    ramp, main = _lane_counts(group, vcfg)
    if not ramp or not main:
        return None
    D = int(durations[int(rng.choice(len(durations)))])
    depth = float(rng.uniform(*vcfg['variety_depth_window']))
    hold = int(rng.uniform(*vcfg['variety_hold_window_s']))
    eased = main[int(rng.integers(0, len(main)))]        # which mainline member eases
    merger = ramp[int(rng.integers(0, len(ramp)))]       # the ramp arrival it anticipates
    gap = float(eased['s_m'])-float(merger['s_m'])
    arrival = gap/max(float(merger['v_mps']), 1e-6)      # geometric predicted arrival (s)
    lead = float(rng.uniform(*vcfg['variety_onset_window_s']))
    onset, hold = _clamp_window(math.floor(arrival-lead) if math.isfinite(arrival) and arrival > 1
                                else lead, hold, D)
    rows = []
    for r in group['rows']:
        i = members.index(r['vid'])
        if r['vid'] == eased['vid']:
            targets = _ease_targets(float(r['v_mps']), D, onset, hold, depth)
            rows.append(_pace_row(float(r['v_mps']), targets, bounds, feas, i))
        else:
            rows.append(np.asarray(P[i, :D], float))     # policy continues (post-clamp)
    return _finish('gap_open', group, P, rows, D, bounds, regime, feas)


def ramp_assist_candidate(group, P, bounds, regime, rng, vcfg, feas, durations):
    """RAMP_ASSIST: the mainline CAV ahead eases and the ramp CAV synchronises to the
    eased speed in a sampled offset window (coordinated timing), then both recover."""
    members = group['members']
    P = np.asarray(P, float)
    ramp, main = _lane_counts(group, vcfg)
    if not ramp or not main:
        return None
    D = int(durations[int(rng.choice(len(durations)))])
    depth = float(rng.uniform(*vcfg['variety_depth_window']))
    hold = int(rng.uniform(*vcfg['variety_hold_window_s']))
    ahead = max(main, key=lambda r: (r['s_m'], r['vid']))   # the mainline CAV ahead
    merger = ramp[int(rng.integers(0, len(ramp)))]
    onset_m, hold_m = _clamp_window(rng.uniform(*vcfg['variety_onset_window_s']), hold, D)
    onset_r, hold_r = _clamp_window(onset_m+float(rng.uniform(*vcfg['variety_onset_window_s'])),
                                    hold, D)
    eased_v = float(ahead['v_mps'])-depth
    rows = []
    for r in group['rows']:
        i = members.index(r['vid'])
        v = float(r['v_mps'])
        if r['vid'] == ahead['vid']:
            targets = _ease_targets(v, D, onset_m, hold_m, depth)
        elif r['vid'] == merger['vid']:
            targets = np.full(D, v)                          # ramp member: approach timing
            targets[onset_r:onset_r+hold_r] = eased_v        # synchronise to the eased speed
        else:
            targets = None
        rows.append(_pace_row(v, targets, bounds, feas, i) if targets is not None
                    else np.asarray(P[i, :D], float))
    return _finish('ramp_assist', group, P, rows, D, bounds, regime, feas)


_GENERATORS = dict(wall=wall_candidate, gap_open=gap_open_candidate, ramp_assist=ramp_assist_candidate)


def pool_candidates(group, P, bounds, regime, rng, vcfg, feas, count, durations):
    """Up to ``count`` structured candidates (population slots they take are ordinary);
    families rotate so the mix stays balanced, an unavailable family's slot falls to
    the next (bounded attempts; the caller tops the pool up with free draws)."""
    out, tries = [], 0
    while len(out) < int(count) and tries < 8*int(count)+8:
        tries += 1
        # Rotation advances on failures too: an unavailable family's slot falls to the next.
        family = STRUCTURED[(tries-1) % len(STRUCTURED)]
        p = _GENERATORS[family](group, P, bounds, regime, rng, vcfg, feas, durations)
        if p is not None:
            out.append(p)
    return out


def random_mix_candidates(group, P, bounds, regime, rng, vcfg, feas, count, durations,
                          duration_s=None):
    """R's structured draws: the same families with random parameters, family chosen
    uniformly by the draw's own stream (no WM scoring; identical registered keys)."""
    if duration_s is not None and duration_s not in durations:
        raise ValueError('unregistered matched duration')
    out, tries = [], 0
    while len(out) < int(count) and tries < 8*int(count)+8:
        tries += 1
        family = STRUCTURED[int(rng.integers(0, len(STRUCTURED)))]
        D = [duration_s] if duration_s is not None else durations
        p = _GENERATORS[family](group, P, bounds, regime, rng, vcfg, feas, D)
        if p is not None:
            out.append(p)
    return out


# --------------------------------------------------------------------------- bank report

def family_counts(verification_root):
    """Per-family counts from SUMO verification receipts (read-only; the family tag
    rides inside each candidate's plan dict, so bank reports can count
    verified/admitted candidates per family without re-reading plans).

    verified: graded candidate branches; admitted: that candidate was the receipt's
    chosen plan with a positive grade (J_policy - J_chosen > 0)."""
    import json
    from pathlib import Path
    counts = {f: dict(verified=0, admitted=0) for f in FAMILIES}
    root = Path(verification_root)
    paths = sorted(root.glob('verifications/*.json')) or sorted(root.glob('*.json'))
    for path in paths:
        try:
            r = json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue
        chosen = r.get('chosen')
        positive = isinstance(r.get('grade_veh_s'), (int, float)) and r['grade_veh_s'] > 0
        for b in r.get('branches', []):
            i = b.get('index', 0)
            if not i or b.get('status') != 'graded' or not isinstance(b.get('plan'), dict):
                continue
            family = b['plan'].get('variety_family')
            if family not in FAMILIES:
                continue        # legacy receipts (option off): not counted per family
            counts[family]['verified'] += 1
            if chosen == i-1 and positive:
                counts[family]['admitted'] += 1
    return counts
