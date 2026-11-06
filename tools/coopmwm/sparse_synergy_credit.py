"""G4: prospective sparse SUMO macro credit contracts.

Production capture, atomic accounting and the fixed-macro executor live in
g4_runtime. This module remains pure and independently validates their results.
Only complete measured base/group/ALL-solo bundles enter the initiating trigger
advantage. Global measured delay is an offline target, never a policy input.
"""
from copy import deepcopy
import hashlib
import json
import math

from . import moves
from .accel_learner import UNIT


SCHEMA = 'coopmwm-g4-sparse-synergy-v1'
REGISTRATION = dict(schema=SCHEMA, arms=['B', 'R'], sample_denominator=16,
    max_starts_per_arm=8, verification_seeds=[1, 2], episode_end_s=1200,
    budget_cap_sim_s=230400, normalization_veh_s=float(UNIT), weight=1.0,
    advantage_clip=1.0,
    provenance='MAIN 2026-11-05 23:15 G4; CLEAN2_GROUP_COOP_RESEARCH rank 4; '
        'prospective implementation registration, no tuning: 1/16 hash, 8 starts/arm, '
        '2 CRN seeds, full 1200s replay, <=4 actors; weight=1, inherited accel_learner.UNIT, '
        'symmetric clip +/-1; retains local total-outcome reward and F4',
    sampling='outcome-blind hash; match B/R counts per update; no replacement/retry',
    target='mean_seed(min_member J_solo - J_group); no solo zero floor',
    intervention='pre-action capture at start_s; fixed selected macro only on retained '
        'members through original expiry/release; all other commands and subsequent '
        'decisions common base; identical prefix, state, CRN and full window',
    unsampled='zero synergy term; no WM substitute')
CAPTURE_KEYS = ('snapshot_sha256', 'prefix_sha256', 'crn_sha256',
                'policy_sha256', 'base_policy_sha256')


def registration():
    return deepcopy(REGISTRATION)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


# ---------------------------------------------------------------- G4LD extension (default OFF)
# G4LD 2026-11-06 (LEARNAUDIT finding 3): the SAME registered estimator (group vs best of
# base/all-member solos, 2 CRN seeds, full-prefix replay, weight/UNIT/clip, hash sampling,
# B/R arity matching, cumulative caps) over on-policy N2 lane-drop moves and W1/N2 V2V
# sessions. Absent/False key = every G4 output byte-identical to the original registration.
EXTENSION_RULE = 'g4ld-lane-drop-v2v-session-synergy-credit-v1'
EXTENSION_KINDS = dict(closing_pair='n2_joint', abreast_wall='n2_joint', group_hold='n2_joint',
                       lane_drop_initiate='n2_session', wall_initiate='w1_session')
JOIN_FAMILY = dict(lane_drop_initiate='lane_drop_join', wall_initiate='wall_join')
# Further option families (e.g. N9 nm_families) must register a branch adapter here before
# they can be enabled; an enabled family without one fails closed (no WM/surrogate substitute).
ADAPTERS = {}


def extension_keys(*, provenance, families=tuple(EXTENSION_KINDS)):
    """Config value for ``g4_extension`` (one common B = R key; requires g4_synergy_credit)."""
    return dict(enabled=True, rule=EXTENSION_RULE, families=sorted(families), provenance=provenance)


def extension(cfg):
    """None when OFF (absent/False). Otherwise the validated common B = R extension slot."""
    slot = cfg.get('g4_extension', False)
    if slot is False:
        return None
    families = slot.get('families') if isinstance(slot, dict) else None
    if (not isinstance(slot, dict) or set(slot) != {'enabled', 'rule', 'families', 'provenance'}
            or slot['enabled'] is not True or slot['rule'] != EXTENSION_RULE
            or not isinstance(families, list) or not families or families != sorted(set(families))
            or any(f not in EXTENSION_KINDS and f not in ADAPTERS for f in families)
            or not isinstance(slot['provenance'], str) or not slot['provenance'].strip()):
        raise ValueError('G4LD extension requires the registered rule, a sorted non-empty family '
                         'list with branch representations, and provenance')
    if settings(cfg) is None:
        raise ValueError('G4LD extension requires the enabled G4 registration')
    return deepcopy(slot)


def members_of(row):
    """Physical macro members (joint move members, or every accepted session member)."""
    return list(row['g4ld']['members']) if 'g4ld' in row else list(row['move']['members'])


def _protocol_option(family, data):
    from . import lane_drop, v2v_wall
    if family in ('lane_drop_initiate', 'lane_drop_join'):
        return lane_drop.LaneDropOption.from_dict(data)
    return v2v_wall.WallOption.from_dict(data)


def _eligible(sample):
    return not sample.get('deterministic') and not sample.get('behaviour', {}).get('executed')


def _session_joins(steps, index, option, family):
    """Accepted joins of ONE session from the recorded slates, in decision order."""
    join, end = JOIN_FAMILY[family], steps[index]['time_s']+option.onset+option.hold
    out = []
    for step in steps[index+1:]:
        if step['time_s'] >= end:
            break
        trigger = step['sample'].get('track2_trigger')
        if trigger is None:
            continue
        for i, vid in enumerate(trigger['context']['members']):
            selected = trigger['tables'][i][trigger['actions'][i]]
            if (selected is None or selected['family'] != join or selected.get('wall_id') != option.wall_id
                    or not trigger.get('accepted', {}).get(vid, False)):
                continue
            joined = _protocol_option(join, selected)
            if (joined.members != (vid,) or joined.initiator != option.initiator
                    or step['time_s']+joined.onset+joined.hold != end
                    or vid == option.initiator or vid in [m['vid'] for m in out]):
                raise ValueError('G4LD recorded join does not match its session')
            out.append(dict(vid=vid, start_s=step['time_s'], option=joined.to_dict(),
                            eligible=_eligible(step['sample'])))
    return out


def _extension_row(episode, steps, index, ego, selected, trigger, members, pending):
    """One G4LD opportunity row (or None) for an accepted extension-family selection.

    Joint N2 move: one initiating decision (the proposer's). V2V session: the initiator's
    initiate plus every accepted join of that session; each member's OWN decision row is
    a credit recipient. Row identity (opportunity_id) uses the original five keys only.
    """
    family, start = selected['family'], steps[index]['time_s']
    if family in ADAPTERS:
        return ADAPTERS[family]['row'](episode, steps, index, ego, selected, trigger, members, pending)
    kind = EXTENSION_KINDS[family]
    if kind == 'n2_joint':
        if len(selected.get('members') or ()) < 2:
            return None   # single-member hold (e.g. a solo group_hold): not a joint start, as legacy G4
        from . import lane_drop
        move = lane_drop.LaneDropMove.from_dict(selected)
        if ego not in move.members or not set(move.members) <= set(members):
            raise ValueError('G4LD accepted joint lane-drop member mismatch')
        data = move.to_dict()
        schedule = [dict(vid=v, start_s=start) for v in move.members]
        credit = [dict(vid=ego, time_s=start)]
    else:
        option = _protocol_option(family, selected)
        if option.members != (ego,) or option.initiator != ego or option.wall_id != f'{ego}@{start}':
            raise ValueError('G4LD accepted session initiation mismatch')
        data = option.to_dict()
        joins = [] if pending else _session_joins(steps, index, option, family)
        schedule = [dict(vid=ego, start_s=start, option=data)]+[
            {k: j[k] for k in ('vid', 'start_s', 'option')} for j in joins]
        credit = [dict(vid=ego, time_s=start)]+[dict(vid=j['vid'], time_s=j['start_s'])
                                                for j in joins if j['eligible']]
        if not pending and len(schedule) < 2:
            return None   # nobody joined: a single-actor commitment, not a joint start
    row = dict(episode_id=episode['id'], time_s=start, proposer=ego, move=data,
               trigger_sha256=digest(trigger))
    row['opportunity_id'] = digest(row)
    row['g4ld'] = dict(kind=kind, family=family, members=[s['vid'] for s in schedule],
                       schedule=schedule, credit=credit)
    return row


def _identity(row):
    return {k: row[k] for k in ('episode_id', 'time_s', 'proposer', 'move', 'trigger_sha256')}


def _check_extension_start(start, ext):
    """Validate one G4LD start's macro, membership and credit recipients."""
    g = start['g4ld']
    family = g.get('family')
    if ext is None or family not in ext['families'] or start['move'].get('family') != family:
        raise ValueError('G4LD start without its enabled extension family')
    if family in ADAPTERS:
        return ADAPTERS[family]['check'](start)
    kind = EXTENSION_KINDS[family]
    members, schedule = g['members'], g['schedule']
    if (g['kind'] != kind or members != [s['vid'] for s in schedule]
            or len(set(members)) != len(members) or len(members) < 2
            or start['proposer'] != (members[0] if kind != 'n2_joint' else start['proposer'])
            or start['proposer'] not in members):
        raise ValueError('G4LD invalid macro membership')
    if kind == 'n2_joint':
        from . import lane_drop
        move = lane_drop.LaneDropMove.from_dict(start['move'])
        if (list(move.members) != members or g['credit'] != [dict(vid=start['proposer'], time_s=start['time_s'])]
                or any(s != dict(vid=v, start_s=start['time_s']) for s, v in zip(schedule, members))):
            raise ValueError('G4LD joint lane-drop schedule/credit mismatch')
        return
    end = None
    for i, s in enumerate(schedule):
        option = _protocol_option(family if i == 0 else JOIN_FAMILY[family], s['option'])
        stop = s['start_s']+option.onset+option.hold
        end = stop if end is None else end
        if (option.members != (s['vid'],) or option.initiator != members[0] or stop != end
                or type(s['start_s']) is not int or s['start_s'] < start['time_s']
                or (i == 0 and (s['start_s'] != start['time_s'] or s['option'] != start['move']))):
            raise ValueError('G4LD session schedule mismatch')
    times = {s['vid']: s['start_s'] for s in schedule}
    if (not g['credit'] or g['credit'][0] != dict(vid=members[0], time_s=start['time_s'])
            or any(times.get(c['vid']) != c['time_s'] for c in g['credit'])
            or len({c['vid'] for c in g['credit']}) != len(g['credit'])):
        raise ValueError('G4LD credit must go to each member own initiating decision')


def settings(cfg):
    slot = cfg.get('g4_synergy_credit', False)
    if slot is False:
        return None
    if (not isinstance(slot, dict) or set(slot) != {'enabled', 'registration', 'phase_id',
                                                   'first_update'}
            or slot['enabled'] is not True or slot['registration'] != REGISTRATION
            or not isinstance(slot['phase_id'], str) or not slot['phase_id'].strip()
            or type(slot['first_update']) is not int or slot['first_update'] < 1):
        raise ValueError('G4 requires fixed common registration and prospective successor phase')
    return deepcopy(slot)


# ---------------------------------------------------------------- budget-derived start cap (default OFF)
# MAIN 2026-11-06 (clean5a, via N4): replace the hand-set 8 starts/arm with the cap DERIVED from
# the registered G4 sim-s cap: per arity n, floor(budget_cap_sim_s / (2 arms x |verification seeds|
# x (n+2) branches x episode_end_s)) = 12 / 9 / 8 for n = 2 / 3 / 4 at 230400. Sampling, scale,
# clip, B/R arity matching and the cumulative sim-s cap are unchanged; absent key = the original 8.
START_CAP_RULE = 'g4-budget-derived-start-cap-v1'


def start_cap_keys(*, provenance):
    return dict(enabled=True, rule=START_CAP_RULE, provenance=provenance)


def start_cap(cfg):
    """None when OFF (absent/False); else the validated common B = R slot (requires G4)."""
    slot = cfg.get('g4_start_cap', False)
    if slot is False:
        return None
    if (not isinstance(slot, dict) or set(slot) != {'enabled', 'rule', 'provenance'}
            or slot['enabled'] is not True or slot['rule'] != START_CAP_RULE
            or not isinstance(slot['provenance'], str) or not slot['provenance'].strip()):
        raise ValueError('budget-derived G4 start cap requires the registered rule and provenance')
    if settings(cfg) is None:
        raise ValueError('budget-derived G4 start cap requires the enabled G4 registration')
    return deepcopy(slot)


def pair_sim_s(arity):
    """Full-prefix verification effort of one matched B/R start pair of this arity."""
    return (2*len(REGISTRATION['verification_seeds'])*(arity+2)*REGISTRATION['episode_end_s'])


def derived_caps():
    """{arity: starts per arm} derived from the registered cumulative sim-s cap."""
    return {n: REGISTRATION['budget_cap_sim_s']//pair_sim_s(n) for n in range(2, moves.MOVE_MAX_MEMBERS+1)}


def _arity_starts():
    return {str(n): 0 for n in range(2, moves.MOVE_MAX_MEMBERS+1)}


def initial_state(cfg):
    state = dict(binding=settings(cfg), last_update=0, reserved_sim_s=0,
                 starts={'B': 0, 'R': 0}, opportunity_ids=[])
    ext = extension(cfg)
    if ext is not None:
        state['extension'] = ext   # G4LD: bound into the reservation journal (absent when OFF)
    cap = start_cap(cfg)
    if cap is not None:            # budget-derived cap: journalled per-arity start counts
        state.update(start_cap=cap, arity_starts=_arity_starts())
    return state


def _max_starts(cap):
    return REGISTRATION['max_starts_per_arm'] if cap is None else max(derived_caps().values())


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def opportunities(episodes, ext=None, *, pending=False):
    """Accepted stochastic joint boundaries only, using exact recorded slates.

    gap_open has a context partner but one actuator; it is not a joint start.
    No outcomes, gains, rewards or measured physical-effect signs select starts.
    ``ext`` (G4LD; default None = the original families only) adds the enabled N2 joint
    lane-drop moves and W1/N2 V2V sessions; ``pending`` (capture time only) emits a session
    initiation before its joins are known, with the same opportunity_id as the final row.
    """
    found, seen = [], set()
    for episode in sorted(episodes, key=lambda e: e['id']):
        if episode['id'] in seen:
            raise ValueError('duplicate G4 episode')
        seen.add(episode['id'])
        if episode.get('practice'):
            continue
        steps = sorted(episode['steps'], key=lambda s: s['time_s'])
        for index, step in enumerate(steps):
            sample = step['sample']
            trigger = sample.get('track2_trigger')
            if (trigger is None or sample.get('deterministic')
                    or sample.get('behaviour', {}).get('executed')):
                continue
            members = trigger['context']['members']
            if (members != sample['context']['members'] or len(set(members)) != len(members)
                    or any(len(trigger[k]) != len(members) for k in ('tables', 'actions', 'mask'))):
                raise ValueError('G4 trigger alignment mismatch')
            for i, ego in enumerate(members):
                action = trigger['actions'][i]
                table = trigger['tables'][i]
                if (type(action) is not int or not 0 <= action < len(table)
                        or not trigger['mask'][i][action]):
                    raise ValueError('invalid G4 recorded action')
                selected = table[action]
                if selected is None or not trigger.get('accepted', {}).get(ego, False):
                    continue
                if ext is not None and selected['family'] in ext['families']:
                    row = _extension_row(episode, steps, index, ego, selected, trigger, members,
                                         pending)
                    if row is not None:
                        found.append(row)
                    continue
                if selected['family'] in ('wall_initiate', 'wall_join'):
                    continue  # W1 per-CAV protocol options are not single-proposer joint macros
                if selected['family'] in ('closing_pair', 'abreast_wall', 'group_hold',
                                          'lane_drop_initiate', 'lane_drop_join'):
                    continue  # N2 closed-loop lane-drop macros: G4LD extension only (opt-in)
                move = moves.Move.from_dict(selected)
                if len(move.actors) < 2:
                    continue
                if ego not in move.members or not set(move.members) <= set(members):
                    raise ValueError('G4 accepted joint move member mismatch')
                start = step['time_s']
                if type(start) is not int or not 0 <= start < REGISTRATION['episode_end_s']:
                    raise ValueError('invalid G4 boundary time')
                row = dict(episode_id=episode['id'], time_s=start, proposer=ego,
                    move=move.to_dict(), trigger_sha256=digest(trigger))
                row['opportunity_id'] = digest(row)
                found.append(row)
    if len({r['opportunity_id'] for r in found}) != len(found):
        raise ValueError('duplicate G4 macro boundary')
    return found


def _sampled(row, episode, cfg):
    # Deliberately excludes arm, selected family, policy, rewards and outcomes.
    key = [cfg['training_seed'], episode['world']['seed'], row['time_s'], row['proposer']]
    return int(digest(key), 16) % REGISTRATION['sample_denominator'] == 0


def _jobs(start, binding):
    members = members_of(start)
    choices = [('group', members), ('base', [])]+[(f'solo{i}', [m])
                                                 for i, m in enumerate(members)]
    jobs = []
    for seed in REGISTRATION['verification_seeds']:
        for label, retained in choices:
            job = dict(schema=SCHEMA, binding=binding, start=start, branch=label,
                retained_members=retained, verification_seed=seed,
                window=dict(start_s=start['time_s'], end_s=REGISTRATION['episode_end_s']),
                intervention=REGISTRATION['intervention'],
                metric='source-inclusive network delay J_veh_s; lower is better',
                sim_s=REGISTRATION['episode_end_s'])  # FULL prefix replay included
            jobs.append(dict(job, job_sha256=digest(job)))
    return jobs


def plan_update(cfg, update, episodes_by_arm, captures, state):
    """Return (paired job plan, next reservation state), with no file I/O.

    captures[arm][opportunity_id] binds an immutable PRE-decision capture via
    CAPTURE_KEYS plus stage='pre_action' and time_s. No snapshot payload is read.
    The caller must atomically persist state, reserve this complete quote in the
    existing run budget, and only then hand jobs to an authorized executor.
    Unused/failed reservations are never recycled. Pair scarcity is not topped
    up by outcome-selected starts or additional samples in the other arm.
    """
    binding = settings(cfg)
    if binding is None:
        raise ValueError('G4 planning requires explicit opt-in')
    ext = extension(cfg)
    cap = start_cap(cfg)
    if (set(episodes_by_arm) != {'B', 'R'} or state['binding'] != binding
            or state.get('extension') != ext or state.get('start_cap') != cap
            or (cap is not None and (set(state.get('arity_starts', {})) != set(_arity_starts())
                                     or any(not 0 <= v <= derived_caps()[int(n)]
                                            for n, v in state['arity_starts'].items())))
            or type(update) is not int or update < binding['first_update']
            or update <= state['last_update']
            or state['starts']['B'] != state['starts']['R']
            or not 0 <= state['starts']['B'] <= _max_starts(cap)
            or not 0 <= state['reserved_sim_s'] <= REGISTRATION['budget_cap_sim_s']):
        raise ValueError('G4 phase/update/reservation mismatch')
    candidates = {}
    for arm in ('B', 'R'):
        episodes = {e['id']: e for e in episodes_by_arm[arm]}
        registered_worlds = {w['seed'] for w in cfg['worlds_by_update'][str(update)]}
        for episode in episodes.values():
            if (episode['arm'] != arm or episode.get('g4_source') != 'ordinary_training'
                    or episode['world']['seed'] not in registered_worlds
                    or episode.get('actual_sim_s') != REGISTRATION['episode_end_s']
                    or episode.get('truncated')):
                raise ValueError('G4 requires complete registered ordinary TRAIN episodes')
        candidates[arm] = [r for r in opportunities(episodes_by_arm[arm], ext)
                           if _sampled(r, episodes[r['episode_id']], cfg)]
    # Matching arity matches branch counts AND full-prefix verification effort.
    selected = {'B': [], 'R': []}
    remaining = REGISTRATION['max_starts_per_arm']-state['starts']['B']
    budget = REGISTRATION['budget_cap_sim_s']-state['reserved_sim_s']
    taken = {}
    for arity in range(2, moves.MOVE_MAX_MEMBERS+1):
        pools = {a: [r for r in candidates[a] if len(members_of(r)) == arity]
                 for a in ('B', 'R')}
        if cap is None:
            count = min(remaining, len(pools['B']), len(pools['R']))
        else:
            # Arity-specific derived cap AND the remaining cumulative sim-s (cheapest arity
            # first, the existing deterministic order); never a quote above the cap.
            count = min(derived_caps()[arity]-state['arity_starts'][str(arity)],
                        len(pools['B']), len(pools['R']), budget//pair_sim_s(arity))
            budget -= count*pair_sim_s(arity)
            taken[str(arity)] = count
        for arm in ('B', 'R'):
            selected[arm] += pools[arm][:count]
        remaining -= count
    count = len(selected['B'])
    jobs, starts = [], []
    for arm in ('B', 'R'):
        for row in selected[arm]:
            if row['opportunity_id'] in state['opportunity_ids']:
                raise ValueError('G4 opportunity already reserved')
            capture = deepcopy(captures[arm][row['opportunity_id']])
            if (set(capture) != {*CAPTURE_KEYS, 'stage', 'time_s'}
                    or capture['stage'] != 'pre_action' or capture['time_s'] != row['time_s']
                    or not all(_sha(capture[k]) for k in CAPTURE_KEYS)):
                raise ValueError('G4 requires exact pre-action state/prefix/policy/CRN binding')
            start = dict(row, arm=arm, update=update, capture=capture)
            starts.append(start)
            jobs.extend(_jobs(start, binding))
    quote = sum(j['sim_s'] for j in jobs)
    if state['reserved_sim_s']+quote > REGISTRATION['budget_cap_sim_s']:
        raise ValueError('G4 cumulative full-prefix budget cap exceeded')
    after = deepcopy(state)
    after.update(last_update=update, reserved_sim_s=state['reserved_sim_s']+quote)
    for arm in ('B', 'R'):
        after['starts'][arm] += count
    if cap is not None:
        for n, value in taken.items():
            after['arity_starts'][n] += value
    after['opportunity_ids'] += [s['opportunity_id'] for s in starts]
    plan = dict(schema=SCHEMA, binding=binding, update=update, starts=starts, jobs=jobs,
        quote_sim_s=quote, reservation_before=deepcopy(state), reservation_after=after)
    if ext is not None:
        plan['extension'] = ext
    if cap is not None:
        plan['start_cap'] = cap
    return dict(plan, plan_sha256=digest(plan)), after


def measured_envelope(plan, results):
    """Validate all receipts before credit; partial bundles never become labels.

    Per-job receipts: job_sha256, source='sumo', status='graded', window,
    capture, verification_seed, retained_members, J_veh_s, actual_sim_s,
    safety (collisions/teleports/hard_brakes/ttc_events). An executor issues a
    receipt only after prefix/CRN/command ownership/full-window checks pass.
    """
    _validate_plan(plan)
    results = list(results)
    by_job = {}
    for result in results:
        key = result['job_sha256']
        if key in by_job:
            raise ValueError('duplicate G4 branch receipt')
        by_job[key] = result
    if set(by_job) != {j['job_sha256'] for j in plan['jobs']}:
        raise ValueError('G4 requires complete group/base/all-member-solo receipts')
    for job in plan['jobs']:
        result = by_job[job['job_sha256']]
        if (digest({k: v for k, v in job.items() if k != 'job_sha256'}) != job['job_sha256']
                or result.get('source') != 'sumo' or result.get('status') != 'graded'
                or any(result.get(k) != job[k] for k in
                       ('window', 'verification_seed', 'retained_members'))
                or result.get('capture') != job['start']['capture']
                or not _finite(result.get('J_veh_s'))
                or result.get('actual_sim_s') != job['sim_s']
                or set(result.get('safety', {})) != {'collisions', 'teleports', 'hard_brakes', 'ttc_events'}
                or any(not _finite(v) or v < 0 for v in result['safety'].values())):
            raise ValueError('G4 measured source/window/CRN/intervention/safety mismatch')
    rows = []
    for start in plan['starts']:
        seeds = []
        for seed in REGISTRATION['verification_seeds']:
            costs = {j['branch']: by_job[j['job_sha256']]['J_veh_s'] for j in plan['jobs']
                     if j['start'] == start and j['verification_seed'] == seed}
            solos = [costs[f'solo{i}'] for i in range(len(members_of(start)))]
            seeds.append(dict(verification_seed=seed, total_benefit_veh_s=costs['base']-costs['group'],
                best_solo_benefit_veh_s=costs['base']-min(solos),
                synergy_veh_s=min(solos)-costs['group']))
        synergy = sum(s['synergy_veh_s'] for s in seeds)/len(seeds)
        normalized = REGISTRATION['weight']*synergy/REGISTRATION['normalization_veh_s']
        limit = REGISTRATION['advantage_clip']
        if not _finite(normalized):
            raise ValueError('nonfinite G4 measured difference')
        rows.append(dict(start=start, seeds=seeds, synergy_veh_s=synergy,
                         advantage=max(-limit, min(limit, normalized))))
    return dict(schema=SCHEMA, plan=deepcopy(plan), results=deepcopy(list(results)), rows=rows)


def execute_plan(plan, *, reserve, run_branch):
    """Explicit executor seam: reserve the entire paired quote before any branch.

    reserve(plan) must atomically persist reservation_after and precharge the
    caller's existing ledger (including its broader remaining budget). It must
    reject duplicate plan IDs. Exceptions propagate; there are no retries,
    alternative scorers, partial labels, or recycled reservations. This function
    performs no physics itself; run_branch is the authorized SUMO worker adapter.
    """
    _validate_plan(plan)
    reserve(deepcopy(plan))
    return measured_envelope(plan, [run_branch(deepcopy(job)) for job in plan['jobs']])


def command_offsets(job, elapsed_s, *, observable_state=None):
    """Exact relative macro mask for a worker's existing single safety writer.

    Keep the ORIGINAL full move for release/role semantics, then mask its actor
    commands; do not reconstruct a singleton wall or change partner geometry.
    Background and future triggers use the bound common base. This contains no
    learned surrogate, residual-gradient target, or hidden state lookup.
    """
    if 'g4ld' in job['start']:
        raise ValueError('G4LD macros run through g4_runtime.ExtensionMacro, not legacy offsets')
    move = moves.Move.from_dict(job['start']['move'])
    offsets = moves.offsets(move, elapsed_s, state=observable_state)
    retained = set(job['retained_members'])
    return {v: value if v in retained else 0. for v, value in offsets.items()}


def _validate_plan(plan):
    body = {k: v for k, v in plan.items() if k != 'plan_sha256'}
    if plan.get('schema') != SCHEMA or digest(body) != plan.get('plan_sha256'):
        raise ValueError('G4 plan digest mismatch')
    binding = settings(dict(g4_synergy_credit=plan['binding']))
    if binding is None:
        raise ValueError('G4 executable plan requires enabled registration')
    update = plan['update']
    ext = plan.get('extension')
    if ext is not None and extension(dict(g4_synergy_credit=plan['binding'], g4_extension=ext)) != ext:
        raise ValueError('G4LD plan extension mismatch')
    cap = plan.get('start_cap')
    if cap is not None and start_cap(dict(g4_synergy_credit=plan['binding'], g4_start_cap=cap)) != cap:
        raise ValueError('G4 plan start-cap mismatch')
    for start in plan['starts']:
        capture = start['capture']
        row = _identity(start)
        if 'g4ld' in start:
            _check_extension_start(start, ext)
            actors, members = members_of(start), members_of(start)
        else:
            move = moves.Move.from_dict(start['move'])
            actors, members = move.actors, move.members
        if (start['arm'] not in ('B', 'R') or start['update'] != update
                or start['opportunity_id'] != digest(row)
                or len(actors) < 2 or start['proposer'] not in members
                or type(start['time_s']) is not int or not 0 <= start['time_s'] < 1200
                or capture.get('stage') != 'pre_action' or capture.get('time_s') != start['time_s']
                or not all(_sha(capture.get(k)) for k in CAPTURE_KEYS)):
            raise ValueError('G4 invalid macro/capture binding')
    expected = [j for s in plan['starts'] for j in _jobs(s, binding)]
    if plan['jobs'] != expected:
        raise ValueError('G4 exact full member/seed/intervention job set required')
    before, after = plan['reservation_before'], plan['reservation_after']
    if (before['binding'] != binding or after['binding'] != binding
            or before.get('extension') != ext or after.get('extension') != ext
            or before.get('start_cap') != cap or after.get('start_cap') != cap
            or type(update) is not int or before['last_update'] >= update
            or update < binding['first_update'] or after['last_update'] != update
            or after['reserved_sim_s'] != before['reserved_sim_s']+plan['quote_sim_s']
            or plan['quote_sim_s'] != sum(j['sim_s'] for j in plan['jobs'])
            or not 0 <= before['reserved_sim_s'] <= after['reserved_sim_s'] <= REGISTRATION['budget_cap_sim_s']):
        raise ValueError('G4 invalid cumulative reservation')
    counts = {a: sum(s['arm'] == a for s in plan['starts']) for a in ('B', 'R')}
    effort = {a: sum(j['sim_s'] for j in plan['jobs'] if j['start']['arm'] == a)
              for a in ('B', 'R')}
    new_ids = [s['opportunity_id'] for s in plan['starts']]
    if (counts['B'] != counts['R'] or effort['B'] != effort['R']
            or before['starts']['B'] != before['starts']['R']
            or len(set(new_ids)) != len(new_ids)
            or len(set(before['opportunity_ids'])) != sum(before['starts'].values())
            or set(new_ids) & set(before['opportunity_ids'])
            or after['opportunity_ids'] != before['opportunity_ids']+new_ids or any(
                before['starts'][a]+counts[a] != after['starts'][a]
                or not 0 <= before['starts'][a] <= after['starts'][a] <= _max_starts(cap)
                for a in ('B', 'R'))):
        raise ValueError('G4 B/R verification effort mismatch')
    if cap is not None:
        added = {str(n): sum(len(members_of(s)) == n for s in plan['starts'] if s['arm'] == 'B')
                 for n in range(2, moves.MOVE_MAX_MEMBERS+1)}
        if (set(before.get('arity_starts', {})) != set(added) or any(
                after['arity_starts'][n] != before['arity_starts'][n]+added[n]
                or not 0 <= after['arity_starts'][n] <= derived_caps()[int(n)] for n in added)):
            raise ValueError('G4 budget-derived arity start cap mismatch')


def trigger_terms(cfg, arm, update, episodes, envelope):
    """Offline PPO-edge validation. Return proposer-only additive advantages.

    Ordinary local-return advantages/critic targets and residual rewards remain
    intact. No inverse-probability multiplier, sign filtering or dense fallback.
    Validation happens before any optimizer is changed.
    """
    binding = None if arm == 'A' else settings(cfg)
    if binding is not None and update < binding['first_update']:
        binding = None
    if binding is None:
        if envelope is not None:
            raise ValueError('G4 envelope supplied to disabled/reference arm')
        return {}, None
    if envelope is None:
        raise ValueError('enabled G4 requires measured envelope (including empty plans)')
    plan = envelope['plan']
    if plan['binding'] != binding or plan['update'] != update:
        raise ValueError('G4 stale phase/update binding')
    checked = measured_envelope(plan, envelope['results'])
    if envelope.get('rows') != checked['rows']:
        raise ValueError('G4 credit arithmetic mismatch')
    ext = extension(cfg)
    if plan.get('extension') != ext:
        raise ValueError('G4LD stale extension binding')
    actual = {r['opportunity_id']: r for r in opportunities(episodes, ext)}
    by_episode = {e['id']: e for e in episodes}
    terms, rows = {}, []
    for row in checked['rows']:
        start = row['start']
        if start['arm'] != arm:
            continue
        expected = actual.get(start['opportunity_id'])
        if expected is None or any(start[k] != v for k, v in expected.items()):
            raise ValueError('G4 label is stale or not this exact accepted on-policy macro')
        episode = by_episode[start['episode_id']]
        if (episode.get('g4_source') != 'ordinary_training'
                or episode['world']['seed'] not in {w['seed'] for w in cfg['worlds_by_update'][str(update)]}
                or episode.get('actual_sim_s') != REGISTRATION['episode_end_s']
                or episode.get('truncated') or not _sampled(start, episode, cfg)):
            raise ValueError('G4 credit is not a prospectively sampled ordinary TRAIN start')
        if 'g4ld' in start:
            # Each member's OWN initiating decision (joint move: the proposer only).
            keys = [(start['episode_id'], c['time_s'], c['vid']) for c in start['g4ld']['credit']]
        else:
            keys = [(start['episode_id'], start['time_s'], start['proposer'])]
        for key in keys:
            if key in terms:
                raise ValueError('duplicate G4 initiating macro credit')
            terms[key] = row['advantage']
        rows.append(row)
    return terms, dict(enabled=True, registration=registration(), phase_id=binding['phase_id'],
        plan_sha256=plan['plan_sha256'], measured_starts=len(rows),
        eligible_starts=len(actual), unsampled_starts=len(actual)-len(rows),
        quote_sim_s=plan['quote_sim_s'],
        cumulative_reserved_sim_s=plan['reservation_after']['reserved_sim_s'], rows=rows)
