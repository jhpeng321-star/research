"""G2 observable merge-role graph, disjoint arbitration and one-shot events.

Only the initiating mainline CAV's already-authorized local rows/partner slots
enter an edge. No union of other egos' observations, outcome, or traffic truth
is used. Roles describe the existing gap_open/ramp_assist vocabulary; this does
not introduce per-role acceleration profiles or assert positive synergy.
"""
from dataclasses import dataclass
import math

import numpy as np

from . import moves

RULE = 'g2-observable-disjoint-merge-pairs-v1'
PROVENANCE = 'tracks_spec.md MAIN 23:15 owner item G2; CLEAN2_GROUP_COOP_RESEARCH rank 2'


def settings(cfg):
    """Common B/R opt-in; no arm-specific settings or unregistered thresholds."""
    raw = cfg.get('coordination_opportunities', False)
    if raw is False:
        return None
    if (not isinstance(raw, dict) or set(raw) != {'enabled', 'rule', 'provenance'}
            or raw['enabled'] is not True or raw['rule'] != RULE
            or not isinstance(raw['provenance'], str) or not raw['provenance'].strip()):
        raise ValueError('G2 requires enabled, registered rule and provenance')
    if type(cfg['trigger_candidates']) is not int or cfg['trigger_candidates'] < 2:
        raise ValueError('G2 needs at least two candidates for joint and singleton options')
    return dict(raw)


def canonical_rows(state):
    """Unify exact float32 encodings of registered zone endpoints, not padding.

    The observable encoder stores float32 positions. Accept the exact float64
    registration and its exact float32 image as the same boundary. Even the next
    float32 value outside remains outside. All other coordinates are unchanged.
    """
    rows = []
    for raw in state['rows']:
        moves._row_checked(raw)
        row = {k: raw[k] for k in moves._ROW_KEYS}
        for boundary in (moves._ZONE_S0, moves._ZONE_S1):
            if float(row['s_m']) == float(np.float32(boundary)):
                row['s_m'] = boundary
        rows.append(row)
    return rows


@dataclass(frozen=True)
class Opportunity:
    yielder: str
    merger: str
    remaining_s: float
    separation_m: float

    @property
    def members(self):
        return (self.yielder, self.merger)

    def to_dict(self):
        return dict(proposer=self.yielder, members=list(self.members),
                    roles={self.yielder: 'mainline_yielder', self.merger: 'ramp_merger'},
                    remaining_s=self.remaining_s if math.isfinite(self.remaining_s) else None,
                    separation_m=self.separation_m)


def _remaining_s(row):
    remaining = moves._ZONE_S1-float(row['s_m'])
    return remaining/float(row['v_mps']) if row['v_mps'] else (0. if remaining == 0 else math.inf)


def graph(obs_states, observable_cavs, eligible, *, occupied=(), command_reservations=None, now):
    """Directed local role edges, ordered by observable ramp remaining time.

    Zone, lanes, association link and sensing radius are inherited registrations.
    Membership in eligible is actuator availability only, never a feature source.
    Reservations are consulted exclusively in the proposer's local view.
    """
    eligible, occupied = set(eligible), set(occupied)
    edges = []
    for ego in sorted(eligible):
        rows = canonical_rows(obs_states[ego])
        by_vid = {r['vid']: r for r in rows}
        if len(by_vid) != len(rows) or obs_states[ego]['self'] != ego or ego not in by_vid:
            raise ValueError('G2 needs distinct observable rows including ego')
        own = by_vid[ego]
        unavailable = occupied | {v for v, expiry in (command_reservations or {}).get(ego, {}).items()
                                  if now < expiry}
        if (ego in unavailable or not own['is_cav'] or int(own['lane']) != moves._MAIN_LANE
                or not moves._ZONE_S0 <= float(own['s_m']) <= moves._ZONE_S1):
            continue
        visible = set(observable_cavs[ego]) & eligible
        for row in rows:
            partner = row['vid']
            distance = abs(float(row['s_m'])-float(own['s_m']))
            if (partner == ego or partner not in visible or partner in unavailable
                    or not row['is_cav'] or int(row['lane']) != moves._RAMP_LANE
                    or not moves._ZONE_S0 <= float(row['s_m']) <= moves._ZONE_S1
                    or distance > min(moves.V2V_RANGE_M,
                                      moves.VARIETY_REGISTERED['variety_gap_link_m'])):
                continue
            edges.append(Opportunity(ego, partner, _remaining_s(row), distance))
    return tuple(sorted(edges, key=lambda e: (e.remaining_s, e.separation_m, e.members)))


def disjoint(edges):
    """Deterministic bounded greedy matching; not an outcome/value optimizer."""
    used, selected = set(), []
    for edge in edges:
        if not used.intersection(edge.members):
            selected.append(edge)
            used.update(edge.members)
    return tuple(selected)


def candidates(state, rng, n):
    """Pair-specific conditional slate with joint, single and separate none.

    The role mainline ego initiates; both pair members must still be acquirable.
    First slots guarantee ramp_assist and ease_off; remaining slots cycle over
    these plus gap_open with balanced counts. Parameters stay in the registered
    library; onset cannot lie beyond observable constant-speed ramp-zone exit.
    No hold or global look-ahead is cropped. Owned/missing partners fall back to
    the original feasible-family path (the caller omits the pair hint then).
    """
    ego, partner = state['self'], state['coordination_partner']
    by_vid = {r['vid']: r for r in canonical_rows(state)}
    available = set(state['acquirable_cav_ids'])
    if n < 2 or not {ego, partner} <= available:
        raise ValueError('G2 pair slate requires free members and two slots')
    own, ramp = by_vid[ego], by_vid[partner]
    # Revalidate the edge from this local view, including roles/range/ownership.
    # Validate without inventing a second ego's observation.
    distance = abs(float(own['s_m'])-float(ramp['s_m']))
    if (not own['is_cav'] or not ramp['is_cav'] or int(own['lane']) != moves._MAIN_LANE
            or int(ramp['lane']) != moves._RAMP_LANE
            or not all(moves._ZONE_S0 <= float(r['s_m']) <= moves._ZONE_S1 for r in (own, ramp))
            or distance > min(moves.V2V_RANGE_M, moves.VARIETY_REGISTERED['variety_gap_link_m'])):
        raise ValueError('G2 invalid observable pair')
    families = ['ramp_assist', 'ease_off', 'gap_open']
    order = families[:2]
    while len(order) < n:
        order.extend(['gap_open'] if len(order) == 2 else rng.permutation(families).tolist())
    out = []
    eta = _remaining_s(ramp)
    for family in order[:n]:
        move = moves._candidate(family, own, [ramp], rng)
        if family != 'ease_off' and math.isfinite(eta):
            onset = min(move.onset, max(0, math.ceil(eta)-1))
            move = moves.Move(move.family, move.members, onset, move.hold, move.depth)
        out.append(move)
    stats = state.get('proposal_stats')
    if stats is not None:
        stats.update(family_checks=len(moves.STRUCTURED), infeasible_family_checks=1)
    return out


class ApproachEvents:
    """One extra event per observable role pair per episode, independent of cadence.

    Seen pairs are marked only when actually offered in a decision. Ownership
    prevents an event until free; disappearance/reappearance and expiry cannot
    repeatedly retrigger the same role pair. A new episode builds a new instance.
    """
    def __init__(self):
        self.seen = set()

    def pending(self, edges):
        return tuple(edge for edge in edges if edge.members not in self.seen)

    def record(self, edges):
        self.seen.update(edge.members for edge in edges)


def aggregate(receipts):
    rows = [row.get('coordination_opportunities') for row in receipts]
    if not rows or all(row is None for row in rows):
        return None
    if (any(row is None or row['rule'] != RULE for row in rows)
            or len({row['provenance'] for row in rows if row is not None}) != 1):
        raise ValueError('mixed G2 opportunity registrations')
    keys = ('decisions', 'offered_joint', 'graph_edges', 'disjoint_pairs', 'event_decisions')
    out = dict(rule=RULE, provenance=rows[0]['provenance'],
               **{key: sum(row[key] for row in rows) for key in keys})
    out['offered_joint_rate'] = out['offered_joint']/out['decisions'] if out['decisions'] else None
    return out
