"""G4: prospective sparse SUMO macro credit; pure CPU contracts, no dispatch.

The owner of collection supplies pre-action captures and persists the returned
reservation state BEFORE dispatch. An authorized executor must implement the
job's exact fixed-macro intervention, not a whole-policy/diagnostic ablation.
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


def initial_state(cfg):
    return dict(binding=settings(cfg), last_update=0, reserved_sim_s=0,
                starts={'B': 0, 'R': 0}, opportunity_ids=[])


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def opportunities(episodes):
    """Accepted stochastic joint boundaries only, using exact recorded slates.

    gap_open has a context partner but one actuator; it is not a joint start.
    No outcomes, gains, rewards or measured physical-effect signs select starts.
    """
    found, seen = [], set()
    for episode in sorted(episodes, key=lambda e: e['id']):
        if episode['id'] in seen:
            raise ValueError('duplicate G4 episode')
        seen.add(episode['id'])
        if episode.get('practice'):
            continue
        for step in sorted(episode['steps'], key=lambda s: s['time_s']):
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
    members = start['move']['members']
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
    if (set(episodes_by_arm) != {'B', 'R'} or state['binding'] != binding
            or type(update) is not int or update < binding['first_update']
            or update <= state['last_update']
            or state['starts']['B'] != state['starts']['R']
            or not 0 <= state['starts']['B'] <= REGISTRATION['max_starts_per_arm']
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
        candidates[arm] = [r for r in opportunities(episodes_by_arm[arm])
                           if _sampled(r, episodes[r['episode_id']], cfg)]
    # Matching arity matches branch counts AND full-prefix verification effort.
    selected = {'B': [], 'R': []}
    remaining = REGISTRATION['max_starts_per_arm']-state['starts']['B']
    for arity in range(2, moves.MOVE_MAX_MEMBERS+1):
        pools = {a: [r for r in candidates[a] if len(r['move']['members']) == arity]
                 for a in ('B', 'R')}
        count = min(remaining, len(pools['B']), len(pools['R']))
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
    after['opportunity_ids'] += [s['opportunity_id'] for s in starts]
    plan = dict(schema=SCHEMA, binding=binding, update=update, starts=starts, jobs=jobs,
        quote_sim_s=quote, reservation_before=deepcopy(state), reservation_after=after)
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
            solos = [costs[f'solo{i}'] for i in range(len(start['move']['members']))]
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
    for start in plan['starts']:
        capture = start['capture']
        row = {k: start[k] for k in ('episode_id', 'time_s', 'proposer', 'move', 'trigger_sha256')}
        move = moves.Move.from_dict(start['move'])
        if (start['arm'] not in ('B', 'R') or start['update'] != update
                or start['opportunity_id'] != digest(row)
                or len(move.actors) < 2 or start['proposer'] not in move.members
                or type(start['time_s']) is not int or not 0 <= start['time_s'] < 1200
                or capture.get('stage') != 'pre_action' or capture.get('time_s') != start['time_s']
                or not all(_sha(capture.get(k)) for k in CAPTURE_KEYS)):
            raise ValueError('G4 invalid macro/capture binding')
    expected = [j for s in plan['starts'] for j in _jobs(s, binding)]
    if plan['jobs'] != expected:
        raise ValueError('G4 exact full member/seed/intervention job set required')
    before, after = plan['reservation_before'], plan['reservation_after']
    if (before['binding'] != binding or after['binding'] != binding
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
                or not 0 <= before['starts'][a] <= after['starts'][a] <= REGISTRATION['max_starts_per_arm']
                for a in ('B', 'R'))):
        raise ValueError('G4 B/R verification effort mismatch')


def trigger_terms(cfg, arm, update, episodes, envelope):
    """Offline PPO-edge validation. Return proposer-only additive advantages.

    Ordinary local-return advantages/critic targets and residual rewards remain
    intact. No inverse-probability multiplier, sign filtering or dense fallback.
    Validation happens before any optimizer is changed.
    """
    binding = None if arm == 'A' else settings(cfg)
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
    actual = {r['opportunity_id']: r for r in opportunities(episodes)}
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
        key = (start['episode_id'], start['time_s'], start['proposer'])
        if key in terms:
            raise ValueError('duplicate G4 initiating macro credit')
        terms[key] = row['advantage']
        rows.append(row)
    return terms, dict(enabled=True, registration=registration(), phase_id=binding['phase_id'],
        plan_sha256=plan['plan_sha256'], measured_starts=len(rows),
        eligible_starts=len(actual), unsampled_starts=len(actual)-len(rows),
        quote_sim_s=plan['quote_sim_s'],
        cumulative_reserved_sim_s=plan['reservation_after']['reserved_sim_s'], rows=rows)
