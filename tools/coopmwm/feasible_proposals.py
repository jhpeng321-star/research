"""F2 registered feasible-family slates for the common B/R move API.

Rule f2-feasible-balanced-v1 (MAIN 2026-11-05 15:15): equal slots per
feasible non-none family, counts differing by at most one. Randomly permute
families each cycle so a short slate has no fixed-family preference. None is
the trigger's separate baseline slot. Uniform random teachers use this SAME
slate; there is no arm-specific proposal distribution or outcome-based tuning.

Geometry/parameters come exclusively from moves/VARIETY_REGISTERED. Runtime
supplies locally filtered acquirable ids after commitment expiry and every
earlier same-tick acquisition. Missing runtime ownership information defaults
to the supplied observable rows (offline bank slates have no commitments).
"""
from copy import deepcopy
from itertools import combinations

import numpy as np
import torch

from . import moves
from .trigger_policy import TriggerBatch

RULE = 'f2-feasible-balanced-v1'
PROVENANCE = 'tracks_spec.md MAIN 15:15 2026-11-05 owner item F2'


def candidates(obs_state, rng, n):
    """At most n balanced, currently acquirable moves from local rows only.

    ``acquirable_cav_ids`` is a subset of the ego's sensing/V2V CAV ids,
    including ego only when unowned. An owned ego has only the caller's none
    option. Optional ``proposal_stats`` receives family feasibility counts.
    All members (including gap_open's non-actuating partner) must be free.
    """
    if type(n) is not int or n < 0 or not isinstance(rng, np.random.Generator):
        raise ValueError('nonnegative integer budget and numpy Generator required')
    if 'coordination_partner' in obs_state:
        from .coordination_opportunities import candidates as pair_candidates
        return pair_candidates(obs_state, rng, n)
    rows = [moves._row_checked(r) for r in obs_state.get('rows', ())]
    by_vid = {r['vid']: r for r in rows}
    ego = obs_state.get('self')
    if len(by_vid) != len(rows) or ego not in by_vid or not by_vid[ego]['is_cav']:
        raise ValueError('distinct observable rows including the acting CAV required')
    own = by_vid[ego]
    available = set(obs_state.get('acquirable_cav_ids', by_vid))
    partners = moves._nearby([r for r in rows if r['is_cav'] and r['vid'] != ego
                             and r['vid'] in available], float(own['s_m']))
    from . import group_lane
    structured = group_lane.families(obs_state)[1:]
    def propose(family, group):
        if family in moves.LANE_FAMILIES:
            return group_lane.candidate(family, obs_state, own, group, rng)
        return moves._candidate(family, own, group, rng)
    pools = {}
    if n and ego in available:
        for family in structured:
            # Enumerate bounded local member sets. The nearest three alone may
            # miss an otherwise feasible wall in a different lane.
            subsets = ([partners] if family != 'wall' else
                (group for size in range(1, moves.MOVE_MAX_MEMBERS)
                 for group in combinations(partners, size)))
            options = [move for group in subsets
                       if (move := propose(family, group)) is not None]
            if options:
                pools[family] = options
    stats = obs_state.get('proposal_stats')
    if stats is not None:
        stats.update(family_checks=len(structured) if n else 0,
                     infeasible_family_checks=len(structured)-len(pools) if n else 0)
    out = []
    while pools and len(out) < n:
        for family in rng.permutation(list(pools)):
            options = pools[family]
            template = options[int(rng.integers(len(options)))]
            # Gap/ramp eligibility depends on the full local pool (notably the
            # ahead-most mainline role); only walls enumerate member subsets.
            group = ([by_vid[v] for v in template.members if v != ego]
                     if family == 'wall' else partners)
            out.append(propose(family, group))
            if len(out) == n:
                break
    return out


def decide_and_start(trigger, context, *, obs_states, observable_cavs, executor,
                     now, rng, generator, deterministic=False, command_reservations=None,
                     coordination=None, opportunity_pairs=None, event=False):
    """Acquire in stable id order (opt-in G2: role pairs first).

    Each slate is conditional on earlier acquisitions. PPO reuses these exact
    slates/masks/logps; it never regenerates an unconditional proposal table.
    command_reservations maps ego -> observable CAV id -> authoritative expiry.
    It includes the controller's live/practice plans; executor commitments are
    checked again after each acquisition. Neither source is traffic truth.
    G2 retains original context ordering and the same conditional PPO contract.
    """
    executor._advance(now)
    members = context['members']
    eligible = set(members)
    batches, accepted = {}, {}
    stats = dict(rule=RULE, provenance=PROVENANCE, decisions=len(members),
                 family_checks=0, infeasible_family_checks=0,
                 owned_cav_checks=0, controller_owned_cav_checks=0,
                 proposed_by_family=dict.fromkeys(trigger.families[1:], 0),
                 selected_non_none=0, rejected_non_none=0)
    order = sorted(members)
    pairs = ()
    if coordination is not None:
        from . import coordination_opportunities as opportunities
        edges = opportunities.graph(obs_states, observable_cavs, eligible,
            occupied={v for entry in executor.active for v in entry['move'].members},
            command_reservations=command_reservations, now=now)
        pairs = (opportunities.disjoint(edges)
            if opportunity_pairs is None else tuple(opportunity_pairs))
        pair_members = [v for pair in pairs for v in pair.members]
        if any(pair not in edges for pair in pairs) or len(set(pair_members)) != len(pair_members):
            raise ValueError('G2 requires disjoint currently observable/acquirable pairs')
        priority = [pair.yielder for pair in pairs]
        order = priority + [v for v in order if v not in priority]
        stats['coordination_opportunities'] = dict(rule=opportunities.RULE,
            provenance=coordination['provenance'], decisions=len(members), offered_joint=0,
            graph_edges=len(edges), disjoint_pairs=len(pairs), event_decisions=int(event),
            decision_order=order, pairs=[pair.to_dict() for pair in pairs])
    pair_by_ego = {pair.yielder: pair for pair in pairs}
    event_members = {v for pair in pairs for v in pair.members}
    for ego in order:
        index = members.index(ego)
        local_cavs = set(observable_cavs[ego]) & eligible
        controller_owned = {v for v, until in (command_reservations or {}).get(ego, {}).items()
                            if v in local_cavs and now < until}
        occupied = controller_owned | {v for entry in executor.active for v in entry['move'].members}
        stats['controller_owned_cav_checks'] += len(controller_owned)
        stats['owned_cav_checks'] += len(local_cavs & occupied)
        local_stats = {}
        state = dict(obs_states[ego], acquirable_cav_ids=[v for v in observable_cavs[ego]
                     if v in eligible and v not in occupied], proposal_stats=local_stats)
        if coordination is not None:
            state['rows'] = opportunities.canonical_rows(state)
            if event:
                # Preserve full context/return alignment; unrelated egos have
                # no action-choice gradient and cannot churn on pair events.
                allowed = (set(pair_by_ego[ego].members) if ego in pair_by_ego else
                           ({ego} if ego in event_members else set()))
                state['acquirable_cav_ids'] = [v for v in state['acquirable_cav_ids'] if v in allowed]
            pair = pair_by_ego.get(ego)
            if pair is not None and set(pair.members) <= set(state['acquirable_cav_ids']):
                state['coordination_partner'] = pair.merger
        local_context = dict(context, members=[ego], x=[context['x'][index]])
        batch = trigger.decide(local_context, obs_states={ego: state},
            observable_cavs={ego: observable_cavs[ego]}, rng=rng, generator=generator,
            deterministic=deterministic)
        batches[ego] = batch
        for key in ('family_checks', 'infeasible_family_checks'):
            stats[key] += local_stats.get(key, 0)
        for move in batch.tables[0]:
            if move is not None:
                stats['proposed_by_family'][move.family] += 1
        if coordination is not None:
            stats['coordination_opportunities']['offered_joint'] += int(any(
                m is not None and len(m.actors) >= 2 for m in batch.tables[0]))
        move = batch.selected()[ego]
        selected = move is not None and move.family != 'none'
        accepted[ego] = executor.start(move, ego=ego, observable_cav_ids=observable_cavs[ego],
            now=now, until=now+move.onset_s+move.hold_s) if selected else False
        stats['selected_non_none'] += int(selected)
        stats['rejected_non_none'] += int(selected and not accepted[ego])
    ordered = [batches[v] for v in members]
    batch = TriggerBatch(deepcopy(context), tuple(b.tables[0] for b in ordered),
        *(torch.cat([getattr(b, key) for b in ordered], dim=0)
          for key in ('descriptors', 'mask', 'actions', 'logp', 'values')))
    return batch, accepted, stats


def aggregate(receipts):
    """Update-level telemetry; None means no measured denominator, never zero.

    infeasible_rate = rejected non-none selections / non-none selections.
    family_infeasible_rate = unavailable family checks / all family checks.
    The latter includes geometry and ownership exclusions BEFORE selection.
    """
    receipts = list(receipts)
    if not receipts:
        return None
    if any(row['rule'] != RULE for row in receipts):
        raise ValueError('mixed proposal registrations')
    out = dict(rule=RULE, provenance=PROVENANCE,
        **{key: sum(row[key] for row in receipts) for key in
           ('decisions', 'family_checks', 'infeasible_family_checks',
            'selected_non_none', 'rejected_non_none')},
        proposed_by_family={f: sum(row['proposed_by_family'].get(f, 0) for row in receipts)
                              for f in dict.fromkeys(f for row in receipts for f in row['proposed_by_family'])})
    # Older F2 receipts predate the controller bridge: don't invent measured
    # zeros when mixing those episodes into an update.
    for key in ('owned_cav_checks', 'controller_owned_cav_checks'):
        out[key] = sum(row[key] for row in receipts) if all(key in row for row in receipts) else None
    for rate, num, den in (('infeasible_rate', 'rejected_non_none', 'selected_non_none'),
                          ('family_infeasible_rate', 'infeasible_family_checks', 'family_checks')):
        out[rate] = out[num]/out[den] if out[den] else None
    from .coordination_opportunities import aggregate as aggregate_opportunities
    coordination = aggregate_opportunities(receipts)
    if coordination is not None:
        out['coordination_opportunities'] = coordination
    return out
