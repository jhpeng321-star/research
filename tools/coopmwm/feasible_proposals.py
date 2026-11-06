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
    # W1 (opt-in): v2v wall protocol options from the ego's own state and the t-1
    # messages it received; an event decision offers only joins.
    wall = obs_state.get('v2v_wall')
    if wall is not None:
        from . import v2v_wall
        structured = (('wall_join',) if wall.get('only_join')
                      else tuple(structured)+v2v_wall.FAMILIES)
    release = obs_state.get('wall_release', False)
    if type(release) is not bool:
        raise ValueError('wall_release must be bool')
    # N2 (opt-in): lane-drop joint moves and protocol options from the ego's own state,
    # its acquirable observable partners and the t-1 messages it received.
    ld = obs_state.get('lane_drop')
    if ld is not None:
        from . import lane_drop
        event = bool(ld.get('only_join')) or (wall is not None and bool(wall.get('only_join')))
        structured = lane_drop.families(structured, ld, event)
    # N9 (opt-in): lane-drop family options appended after W1/N2. On an event decision
    # every present protocol is join-only (offers of non-triggering protocols cleared).
    families = obs_state.get('nm_families')
    if families is not None:
        from . import nm_families
        event = (bool(families.get('only_join')) or (wall is not None and bool(wall.get('only_join')))
                 or (ld is not None and bool(ld.get('only_join'))))
        if event:
            reduced = wall is not None or ld is not None
            structured = (tuple(structured) if reduced else ())+nm_families.join_families(families)
        else:
            structured = tuple(structured)+nm_families.slate_families(families)
    def propose(family, group):
        if family in moves.LANE_FAMILIES:
            return group_lane.candidate(family, obs_state, own, group, rng)
        move = moves._candidate(family, own, group, rng)
        if release and family == 'wall':
            # W2 (opt-in, B = R): registered gradual-release shape draw for this wall.
            from .wall_release import with_release
            move = with_release(move, [own]+list(group), rng)
        return move
    pools = {}
    if n and ego in available:
        for family in structured:
            if wall is not None and family in v2v_wall.FAMILIES:
                options = v2v_wall.pool(family, wall, own, rng)
                if options:
                    pools[family] = options
                continue
            if ld is not None and lane_drop.owns(family):
                options = lane_drop.pool(family, ld, own, partners, rng)
                if options:
                    pools[family] = options
                continue
            if families is not None and nm_families.is_family(family):
                options = nm_families.pool(family, families, own, rng)
                if options:
                    pools[family] = options
                continue
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
            if wall is not None and family in v2v_wall.FAMILIES:
                out.append(v2v_wall.redraw(template, wall, own, rng))
                if len(out) == n:
                    break
                continue
            if ld is not None and lane_drop.owns(family):
                option = lane_drop.redraw(template, ld, own, by_vid, rng)
                if option is not None:
                    out.append(option)
                if len(out) == n:
                    break
                continue
            if families is not None and nm_families.is_family(family):
                out.append(nm_families.redraw(template, families, own, rng))
                if len(out) == n:
                    break
                continue
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
                     coordination=None, opportunity_pairs=None, event=False,
                     wall=None, wall_events=None, lane_drop=None, lane_drop_events=None,
                     families=None, family_events=None):
    """Acquire in stable id order (opt-in G2: role pairs first).

    W1 (opt-in ``wall``): protocol options start through ``wall.start`` (atomic
    session ownership/cap + executor commitment). ``wall_events`` marks a
    non-periodic first-receipt tick: those egos get join-only slates, all others
    (outside G2 pairs) only ``none``. N9 (opt-in ``families``/``family_events``)
    follows the identical start/event contract through ``families.start``.

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
                 proposed_by_family=dict.fromkeys(getattr(trigger, 'option_families', trigger.families)[1:], 0),
                 selected_non_none=0, rejected_non_none=0)
    if wall is not None:
        stats['v2v_wall'] = dict(rule=wall.settings['rule'], event=wall_events is not None,
            event_cavs=sorted(wall_events or ()), offered_join=0, selected_initiate=0,
            accepted_initiate=0, selected_join=0, accepted_join=0)
    if lane_drop is not None:
        stats['lane_drop'] = dict(rule=lane_drop.settings['rule'], event=lane_drop_events is not None,
            event_cavs=sorted(lane_drop_events or ()), offered_join=0, selected_initiate=0,
            accepted_initiate=0, selected_join=0, accepted_join=0)
    ld_tick = lane_drop is not None and lane_drop_events is not None
    if families is not None:
        stats['nm_families'] = dict(rule=families.settings['rule'], event=family_events is not None,
            event_cavs=sorted(family_events or ()), offered_join=0, selected_initiate=0,
            accepted_initiate=0, selected_join=0, accepted_join=0)
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
        free = ego in state['acquirable_cav_ids']
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
        nm_tick = families is not None and family_events is not None
        if ld_tick or nm_tick:
            # N2/N9 event tick (non-periodic first receipt): only pending joins, others none.
            in_wall = (wall is not None and wall_events is not None and ego in wall_events
                       and ego not in event_members and 'v2v_wall' in state)
            in_ld = ld_tick and ego in lane_drop_events and ego not in event_members and 'lane_drop' in state
            in_nm = (nm_tick and ego in family_events and ego not in event_members
                     and 'nm_families' in state)
            if in_wall or in_ld or in_nm:
                state['acquirable_cav_ids'] = [ego] if free else []
                if 'v2v_wall' in state:
                    state['v2v_wall'] = dict(state['v2v_wall'], only_join=True,
                                             **({} if in_wall else dict(offers=())))
                if 'lane_drop' in state:
                    state['lane_drop'] = dict(state['lane_drop'], only_join=True,
                                              **({} if in_ld else dict(offers=())))
                if 'nm_families' in state:
                    state['nm_families'] = dict(state['nm_families'], only_join=True,
                                                **({} if in_nm else dict(offers=())))
            elif coordination is None:
                state['acquirable_cav_ids'] = []
        elif wall is not None and wall_events is not None:
            if ego in wall_events and ego not in event_members and 'v2v_wall' in state:
                state['acquirable_cav_ids'] = [ego] if free else []
                state['v2v_wall'] = dict(state['v2v_wall'], only_join=True)
                if 'lane_drop' in state:
                    state['lane_drop'] = dict(state['lane_drop'], only_join=True, offers=())
                if 'nm_families' in state:
                    state['nm_families'] = dict(state['nm_families'], only_join=True, offers=())
            elif coordination is None:
                state['acquirable_cav_ids'] = []
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
        if wall is not None:
            wall.mark_offered(ego, batch.tables[0])
            stats['v2v_wall']['offered_join'] += sum(m is not None and m.family == 'wall_join'
                                                     for m in batch.tables[0])
        if lane_drop is not None:
            lane_drop.mark_offered(ego, batch.tables[0])
            stats['lane_drop']['offered_join'] += sum(m is not None and m.family == 'lane_drop_join'
                                                      for m in batch.tables[0])
        if families is not None:
            families.mark_offered(ego, batch.tables[0])
            stats['nm_families']['offered_join'] += sum(
                m is not None and m.family in families.FAMILIES and m.family.endswith('_join')
                for m in batch.tables[0])
        if selected and wall is not None and move.family in wall.FAMILIES:
            accepted[ego] = wall.start(move, ego=ego, executor=executor,
                                       observable_cav_ids=observable_cavs[ego], now=now)
            kind = 'initiate' if move.family == 'wall_initiate' else 'join'
            stats['v2v_wall']['selected_'+kind] += 1
            stats['v2v_wall']['accepted_'+kind] += int(accepted[ego])
        elif selected and lane_drop is not None and move.family in lane_drop.FAMILIES:
            accepted[ego] = lane_drop.start(move, ego=ego, executor=executor,
                                            observable_cav_ids=observable_cavs[ego], now=now)
            kind = 'initiate' if move.family == 'lane_drop_initiate' else 'join'
            stats['lane_drop']['selected_'+kind] += 1
            stats['lane_drop']['accepted_'+kind] += int(accepted[ego])
        elif selected and families is not None and move.family in families.FAMILIES:
            accepted[ego] = families.start(move, ego=ego, executor=executor,
                                           observable_cav_ids=observable_cavs[ego], now=now)
            kind = 'join' if move.family.endswith('_join') else 'initiate'
            stats['nm_families']['selected_'+kind] += 1
            stats['nm_families']['accepted_'+kind] += int(accepted[ego])
        else:
            # A W2 gradual-release wall stays committed through its release ramp (0 otherwise).
            accepted[ego] = executor.start(move, ego=ego, observable_cav_ids=observable_cavs[ego],
                now=now, until=now+move.onset_s+move.hold_s+getattr(move, 'release_s', 0)) if selected else False
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
