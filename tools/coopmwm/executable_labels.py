"""G1: CPU-only executable teacher binding, prospective jobs and admission.

No runner is imported or dispatched. Results must describe the *library macro*,
not its native ancestor. Inputs/results are ordinary TRAIN observable receipts.
The manifest retains every failed/negative candidate, not just admitted labels.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from . import moves
from .bank_safety import verdict as safety_verdict, RULE as SAFETY_RULE
from .bank_statistics import BOOT_B, BOOT_RNG_SEED, BLOCK_S
from .protocol import digest

SCHEMA = 'coopmwm/executable-labels/g1-v1'
PROVENANCE = 'tracks_spec.md MAIN 23:15 2026-11-05 item G1'
CAP_SIM_S = 230_400
EFFECT_EPS_MPS2 = 1e-6  # production clean2_receipts physical-effect threshold
SAFETY = ('collisions', 'teleports', 'hard_brake', 'ttc_below_1p5_events')
ENGINE = dict(macro='moves.Move + moves.offsets; full registered strength',
    ownership='CommandOwnership single writer; acquire all members atomically',
    background='common base policy for all non-selected members and all other vehicles',
    residual='identical base residual in every branch; only library offset differs',
    prefix='exact captured prefix and t0 command; intervention strictly after t0',
    noise='matched counter-based/member-keyed CRN; no mean-noise fallback',
    release='library release/expiry and existing final safety writer; no later macros')
STATISTICS = dict(B=BOOT_B, rng_seed=BOOT_RNG_SEED, block_s=BLOCK_S,
    interval='two-sided percentile CI95, lower quantile .025',
    selection='recompute best solo from ALL full-window solo totals in every paired draw',
    scope='within-context circular blocks, independently within each verification seed; not generalization',
    admission='pooled mean and EACH seed CI95 low > 0; safety and physical joint effect on each seed')


def seal(record, key):
    body = {k: deepcopy(v) for k, v in record.items() if k != key}
    return dict(body, **{key: digest(body)})


def check(record, key):
    if record.get(key) != digest({k: v for k, v in record.items() if k != key}):
        raise ValueError(f'G1 {key} mismatch')


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'G1 {name} must be an integer >= {minimum}')
    return value


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('G1 immutable SHA256 binding required')
    return value


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('G1 finite numeric evidence required')
    return float(value)


def _local(move, local, decision_s):
    """Validate local availability and exact library geometry, without sampling.

    local rows are the proposer's causal sensing/V2V view, never global rows.
    Replay binding and teacher sensor/partner inputs use this SAME t-1 view;
    decision_s remains the execution time, not an observation timestamp.
    Offered labels are checked geometrically even if bank CE inserted the label.
    """
    if _integer(local['observed_at_s'], 'observation time') != decision_s-1:
        raise ValueError('G1 requires causal t-1 observable visibility')
    rows = [moves._row_checked(r) for r in local['rows']]
    by_id = {r['vid']: r for r in rows}
    ego = local['self']
    if len(by_id) != len(rows) or ego not in move.members:
        raise ValueError('G1 distinct observable rows and member proposer required')
    if not set(move.members) <= set(by_id) or any(not by_id[v]['is_cav'] for v in move.members):
        raise ValueError('G1 every member must be an observable CAV')
    if not set(move.members) <= set(local['acquirable_cav_ids']):
        raise ValueError('G1 member owned or unavailable')
    own = by_id[ego]
    if any(abs(float(by_id[v]['s_m'])-float(own['s_m'])) > moves.V2V_RANGE_M for v in move.members):
        raise ValueError('G1 member outside local sensing/V2V range')
    if move.to_dict() not in local['offered_moves']:
        raise ValueError('G1 exact executable label not offered')
    if move.family == 'wall':
        chosen = [by_id[v] for v in move.members]
        arrival = min((moves._ZONE_S0-float(r['s_m']))/float(r['v_mps'])
                      if float(r['v_mps']) > 0 else math.inf for r in chosen)
        if (any(float(r['s_m']) >= moves._ZONE_S0 or
                abs(float(r['s_m'])-float(own['s_m'])) > moves.VARIETY_REGISTERED['variety_wall_span_m']
                for r in chosen) or len({int(r['lane']) for r in chosen}) < moves._WALL_MIN_LANES
                or move.onset+move.hold > arrival
                or tuple(r['vid'] for r in moves._leader_first(chosen)) != move.members):
            raise ValueError('G1 exact wall is not online geometry/arrival eligible')
    if move.family in ('gap_open', 'ramp_assist'):
        available = set(local['acquirable_cav_ids'])
        cavs = [r for r in rows if r['is_cav'] and r['vid'] in available]
        pools = moves._zone_pools(cavs, own)
        pools = {k: [r for r in rs if abs(float(r['s_m'])-float(own['s_m'])) <=
                     moves.VARIETY_REGISTERED['variety_gap_link_m']] for k, rs in pools.items()}
        ramps = [r for r in pools['ramps'] if r['vid'] in move.members]
        mains = [r for r in pools['mains'] if r['vid'] in move.members]
        if len(move.members) != 2 or len(ramps) != 1 or len(mains) != 1:
            raise ValueError('G1 ramp/mainline roles not locally feasible')
        if move.family == 'gap_open' and move.members[0] != mains[0]['vid']:
            raise ValueError('G1 gap opener must be first member')
        if move.family == 'ramp_assist' and (
                max(pools['mains'], key=lambda r: (float(r['s_m']), r['vid']))['vid'] != mains[0]['vid']
                or tuple(r['vid'] for r in moves._leader_first([ramps[0], mains[0]])) != move.members):
            raise ValueError('G1 ramp assist requires the ahead-most mainline CAV')
    return {v: ('context_only' if v not in move.actors else
                'mainline_gap_opener' if move.family == 'gap_open' else
                'ramp_synchronizer' if move.family == 'ramp_assist' and int(by_id[v]['lane']) == moves._RAMP_LANE
                else 'mainline_yielder' if move.family == 'ramp_assist' else 'upstream_holder'
                if move.family == 'wall' else 'easing_actor') for v in move.members}


def bind_lesson(*, arm, receipt, example, plan, snapshot_sha256, prefix_sha256,
                base_policy_sha256, runtime_sha256, local, decision_s, window, move=None):
    """Bind one exact F3 target at its actual decision time (no implicit wall label).

    A later/remaining-onset target is a DIFFERENT lesson and needs its own jobs.
    Source verification is only ancestry; it is never executable certification.
    """
    if arm not in ('B', 'R') or receipt['episode']['role'] != 'train' or receipt['episode']['arm'] != arm:
        raise ValueError('G1 only corresponding B/R ordinary TRAIN contexts')
    if type(receipt['episode']['world']['seed']) is not int or not 986000 <= receipt['episode']['world']['seed'] <= 986071:
        raise ValueError('G1 requires F3 registered TRAIN worlds')
    t0 = _integer(receipt['t0'], 't0')
    check({k: v for k, v in receipt.items() if not k.startswith('_')}, 'receipt_sha256')
    decision_s = _integer(decision_s, 'decision_s')
    start, end = window['start_s'], window['end_s']
    _integer(start, 'window start'); _integer(end, 'window end', 1)
    if window != receipt['window'] or start != t0 or not t0 < decision_s < end or end > 1200:
        raise ValueError('G1 full registered window and post-t0 decision required')
    move = moves.from_plan(plan) if move is None else moves.Move.from_dict(move.to_dict())
    if not set(move.members) <= set(plan['members']):
        raise ValueError('G1 macro members must come from source lesson')
    roles = _local(move, local, decision_s)
    if move.family != 'none' and decision_s+move.onset+move.hold > end:
        raise ValueError('G1 full macro must fit registered window; no shortened hold')
    native_members, native = moves._plan_offsets(plan)
    native_duration = int(native.shape[1])
    horizon = max(native_duration, decision_s-t0+move.onset+move.hold)
    padded = np.zeros((len(native_members), horizon))
    padded[:, :native_duration] = native
    executable = np.zeros_like(padded)
    for t in range(horizon):
        if t0+t >= decision_s:
            for vid, off in moves.offsets(move, t0+t-decision_s).items():
                executable[native_members.index(vid), t] = off
    return seal(dict(schema=SCHEMA, arm=arm, verification_id=receipt['verification_id'],
        example_id=(example or {}).get('example_id', receipt['verification_id']+'#s1'),
        source_receipt_sha256=_sha(receipt['receipt_sha256']), source_example_sha256=digest(example),
        source_plan_sha256=digest(plan), snapshot_sha256=_sha(snapshot_sha256),
        prefix_sha256=_sha(prefix_sha256), base_policy_sha256=_sha(base_policy_sha256),
        runtime_sha256=_sha(runtime_sha256), episode=deepcopy(receipt['episode']), t0=t0,
        decision_s=decision_s, window=deepcopy(window), macro=move.to_dict(), actors=list(move.actors),
        roles=roles, local=deepcopy(local), engine=ENGINE,
        fidelity=dict(native_members=native_members, native_duration_s=native_duration,
            squared_offset_error=float(np.square(padded-executable).sum()),
            exact_native_profile=bool(np.array_equal(padded, executable)),
            preference='2-actor native profile first; no outcome-based selection')),
        'lesson_sha256')


def job_spec(lesson, seeds):
    """Prospective full-prefix SUMO contract, deliberately NOT a legacy native-plan job.

    seeds: two {verification_seed, snapshot_sha256, rng_sha256} immutable restart
    bindings supplied by the authorized collector. State/prefix/base stay fixed.
    """
    check(lesson, 'lesson_sha256')
    if (len(seeds) != 2 or len({s['verification_seed'] for s in seeds}) != 2
            or len({s['rng_sha256'] for s in seeds}) != 2
            or len({s['snapshot_sha256'] for s in seeds}) != 2):
        raise ValueError('G1 exactly two distinct verification seeds required')
    macro = moves.Move.from_dict(lesson['macro'])
    t0 = _integer(lesson['t0'], 't0')
    decision = _integer(lesson['decision_s'], 'decision_s')
    end = _integer(lesson['window']['end_s'], 'window end', 1)
    if (lesson['arm'] not in ('B', 'R') or lesson['episode']['role'] != 'train'
            or lesson['episode']['arm'] != lesson['arm']
            or type(lesson['episode']['world']['seed']) is not int
            or not 986000 <= lesson['episode']['world']['seed'] <= 986071
            or lesson['window']['start_s'] != t0 or not t0 < decision < end or end > 1200
            or (macro.family != 'none' and decision+macro.onset+macro.hold > end)):
        raise ValueError('G1 lesson requires ordinary TRAIN context and full executable window')
    for key in ('source_receipt_sha256', 'source_example_sha256', 'source_plan_sha256',
                'snapshot_sha256', 'prefix_sha256', 'base_policy_sha256', 'runtime_sha256'):
        _sha(lesson[key])
    roles = _local(macro, lesson['local'], lesson['decision_s'])
    if lesson['schema'] != SCHEMA or lesson['engine'] != ENGINE or lesson['roles'] != roles or lesson['actors'] != list(macro.actors):
        raise ValueError('G1 lesson execution semantics differ')
    choices = [('group', list(macro.actors)), ('base', [])]+[(f'solo{i}', [v]) for i, v in enumerate(macro.members)]
    branches = []
    for seed in seeds:
        _integer(seed['verification_seed'], 'verification seed')
        _sha(seed['snapshot_sha256']); _sha(seed['rng_sha256'])
        common = dict(lesson_sha256=lesson['lesson_sha256'], seed=deepcopy(seed),
            prefix_sha256=lesson['prefix_sha256'], base_policy_sha256=lesson['base_policy_sha256'],
            runtime_sha256=lesson['runtime_sha256'], window=lesson['window'],
            decision_s=lesson['decision_s'], macro=lesson['macro'], engine=ENGINE,
            simulate_from_s=0, simulate_through_s=lesson['window']['end_s'])
        for name, selected in choices:
            branches.append(seal(dict(common, branch=name, selected_actors=selected,
                background_actors='base', prefix_match_required=True,
                crn_alignment_required_through_s=lesson['window']['end_s'],
                metric='source-inclusive team_delay_vs increments, every second of the full window',
                sim_s=lesson['window']['end_s']), 'branch_sha256'))
    return seal(dict(schema=SCHEMA, lesson=lesson, seeds=deepcopy(seeds), branches=branches,
        statistics=STATISTICS, safety_rule=SAFETY_RULE, sim_s=sum(b['sim_s'] for b in branches)), 'job_sha256')


def _interval(values):
    return [float(v) for v in np.quantile(values, [.025, .975])]


def _statistics(series_by_seed):
    """Pair ALL branch time indices and reselect best solo in each draw."""
    rng = np.random.default_rng(BOOT_RNG_SEED)
    pooled_s = np.zeros(BOOT_B); pooled_t = np.zeros(BOOT_B)
    per_seed = []
    for series in series_by_seed:
        # rows: group, base, solo0, ...; columns: exact registered seconds
        n = series.shape[1]
        length = max(1, min(BLOCK_S, n//2 if n >= 2 else 1))
        blocks = -(-n//length)
        starts = rng.integers(0, n, (BOOT_B, blocks))
        idx = ((starts[:, :, None]+np.arange(length)) % n).reshape(BOOT_B, -1)[:, :n]
        totals = series[:, idx].sum(axis=2)
        synergy = totals[2:].min(axis=0)-totals[0]
        benefit = totals[1]-totals[0]
        observed = series.sum(axis=1)
        pooled_s += synergy/len(series_by_seed); pooled_t += benefit/len(series_by_seed)
        per_seed.append(dict(J_group_veh_s=float(observed[0]), J_base_veh_s=float(observed[1]),
            J_solos_veh_s=[float(v) for v in observed[2:]],
            synergy_veh_s=float(observed[2:].min()-observed[0]),
            total_benefit_veh_s=float(observed[1]-observed[0]),
            synergy_ci95=_interval(synergy), total_benefit_ci95=_interval(benefit)))
    return dict(per_seed=per_seed, synergy_mean_veh_s=float(np.mean([s['synergy_veh_s'] for s in per_seed])),
        total_benefit_mean_veh_s=float(np.mean([s['total_benefit_veh_s'] for s in per_seed])),
        synergy_ci95=_interval(pooled_s), total_benefit_ci95=_interval(pooled_t), rule=STATISTICS)


def _physical(lesson, audit):
    """Check issued-command deltas, not just membership or nonzero requested offsets."""
    macro = moves.Move.from_dict(lesson['macro'])
    counts, seen = {}, set()
    for row in audit:
        t = _integer(row['time_s'], 'audit time')
        vid = row['vid']
        if (t, vid) in seen or vid not in macro.members or not lesson['decision_s'] <= t < lesson['window']['end_s']:
            raise ValueError('G1 duplicate/out-of-window/non-member actuation evidence')
        seen.add((t, vid))
        # state is the executor's observable clamp/release input, not simulator truth.
        state = row['observable_state']
        if macro.family == 'wall' and (not isinstance(state.get('s_m'), dict) or vid not in state['s_m']):
            raise ValueError('G1 wall effect requires observed release positions')
        expected = moves.offsets(macro, t-lesson['decision_s'], state)[vid]
        offset = _finite(row['library_move_offset_mps2'])
        raw = _finite(row['requested_raw'])-_finite(row['without_library_raw'])
        measured = _finite(row['issued_safe'])-_finite(row['without_library_issued_safe'])
        if (not math.isclose(offset, expected, abs_tol=1e-6, rel_tol=1e-6)
                or not math.isclose(raw, offset, abs_tol=1e-6, rel_tol=1e-6)
                or not math.isclose(measured, _finite(row['library_issued_delta_mps2']), abs_tol=1e-6, rel_tol=1e-6)):
            raise ValueError('G1 actual issued-command or library-offset effect mismatch')
        if vid in macro.actors and offset and abs(measured) > EFFECT_EPS_MPS2:
            counts.setdefault(t, set()).add(vid)
    joint = sorted(t for t, actors in counts.items() if len(actors) >= 2)
    return dict(joint_seconds=len(joint), joint_times_s=joint,
        physically_affected_actors=sorted(set().union(*counts.values()) if counts else set()),
        threshold_mps2=EFFECT_EPS_MPS2, passed=bool(joint))


def admit(job, results):
    """Recompute admission from exact bound branch receipts; retain all negatives.

    A malformed/incomplete/mismatched receipt is diagnostic, never a certificate.
    Results are sealed with result_sha256 and reference the complete branch spec
    hash. Each has window, prefix_matched, crn_aligned_through_s, delay_increments,
    safety counts, and (group only) library_actuation_audit.
    """
    results = list(results)
    check(job, 'job_sha256'); check(job['lesson'], 'lesson_sha256')
    # Rebuild instead of trusting edited branch selections, statistics or costs.
    if job != job_spec(job['lesson'], job['seeds']):
        raise ValueError('G1 job differs from registered executable verification contract')
    reasons, physical, safe, series_by_seed = [], [], [], []
    try:
        expected = {b['branch_sha256']: b for b in job['branches']}
        actual = {}
        for result in results:
            check(result, 'result_sha256')
            key = result['branch_sha256']
            if key not in expected or key in actual:
                raise ValueError('G1 unexpected or duplicate branch')
            spec = expected[key]
            if (result['status'] != 'graded' or result['window'] != spec['window']
                    or result['prefix_matched'] is not True
                    or result['crn_aligned_through_s'] != spec['window']['end_s']
                    or result['actual_sim_s'] != spec['sim_s']):
                raise ValueError('G1 ungraded, unmatched horizon/prefix/CRN or full-prefix accounting')
            values = [_finite(v) for v in result['delay_increments']]
            if len(values) != spec['window']['end_s']-spec['window']['start_s'] or any(v < 0 for v in values):
                raise ValueError('G1 complete source-inclusive per-second delay increments required')
            for k in SAFETY:
                _integer(result['safety'][k], 'safety '+k)
            actual[key] = result
        if set(actual) != set(expected):
            raise ValueError('G1 group, base and ALL member solos required on both seeds')
        width = len(job['lesson']['macro']['members'])+2
        for i in range(2):
            own = [actual[b['branch_sha256']] for b in job['branches'][i*width:(i+1)*width]]
            series_by_seed.append(np.array([r['delay_increments'] for r in own], dtype=float))
            effect = _physical(job['lesson'], own[0]['library_actuation_audit'])
            physical.append(effect)
            if not effect['passed']:
                reasons.append(f'seed {i+1}: no simultaneous physically executing joint actors')
            deltas = {k: own[0]['safety'][k]-own[1]['safety'][k] for k in SAFETY}
            safety = safety_verdict(deltas)
            safety['zero_collisions_teleports_all_branches'] = all(
                not r['safety']['collisions'] and not r['safety']['teleports'] for r in own)
            safe.append(dict(deltas=deltas, **safety))
            if not safety['new_combined_rule'] or not safety['zero_collisions_teleports_all_branches']:
                reasons.append(f'seed {i+1}: safety failed')
        stats = _statistics(series_by_seed)
        if stats['synergy_ci95'][0] <= 0 or any(s['synergy_ci95'][0] <= 0 for s in stats['per_seed']):
            reasons.append('executable group-over-best-solo CI95 low <= 0')
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        reasons.append('invalid/incomplete evidence: '+str(error))
        stats = None
    return seal(dict(schema=SCHEMA, job_sha256=job['job_sha256'],
        lesson_sha256=job['lesson']['lesson_sha256'], admitted=not reasons,
        status='admitted' if not reasons else 'diagnostic_only', reasons=reasons,
        statistics=stats, physical=physical, safety=safe,
        result_sha256=[r.get('result_sha256') for r in results], provenance=PROVENANCE), 'admission_sha256')


def budget_quote(jobs=None):
    """Conservative full-prefix cap, no speculative branch reuse or dispatch."""
    maximum = 8*2*2*(2+4)*1200
    quote = dict(schema=SCHEMA, provenance=PROVENANCE, contexts_per_arm=8, arms=['B', 'R'],
        verification_seeds=2, max_member_solos=4, full_episode_sim_s=1200,
        preferred_two_actor_sim_s=8*2*2*4*1200, worst_case_sim_s=maximum,
        cap_sim_s=CAP_SIM_S, includes_prefix_replay=True, branch_reuse_discount_sim_s=0,
        dispatched=False, executed_sim_s=0)
    if jobs is not None:
        jobs = list(jobs)
        arms = {a: [] for a in ('B', 'R')}
        for job in jobs:
            check(job, 'job_sha256')
            if job != job_spec(job['lesson'], job['seeds']):
                raise ValueError('G1 noncanonical job in quote')
            arms[job['lesson']['arm']].append(job)
        if len(arms['B']) != len(arms['R']) or any(len(v) > 8 for v in arms.values()):
            raise ValueError('G1 equal B/R effort, at most eight contexts per arm')
        # Match per-family/arity effort and verification seeds, not outcomes.
        def effort(rows):
            return sorted((j['lesson']['macro']['family'], len(j['lesson']['macro']['members']),
                j['lesson']['window']['end_s'], tuple(s['verification_seed'] for s in j['seeds'])) for j in rows)
        if effort(arms['B']) != effort(arms['R']):
            raise ValueError('G1 B/R family, arity, full-prefix length and seed effort must match')
        identities = [(j['lesson']['arm'], j['lesson']['snapshot_sha256'], j['lesson']['decision_s']) for j in jobs]
        if len(set(identities)) != len(identities):
            raise ValueError('G1 distinct contexts before repeats; duplicate context in quote')
        quote.update(planned_sim_s=sum(j['sim_s'] for j in jobs), branches=sum(len(j['branches']) for j in jobs),
            job_sha256=[j['job_sha256'] for j in jobs])
    if maximum > CAP_SIM_S or quote.get('planned_sim_s', 0) > CAP_SIM_S:
        raise ValueError('G1 verification quote exceeds 230400 sim-s')
    return quote


def prioritize(lessons):
    """Prospective preference only; never rank by measured synergy."""
    return sorted(lessons, key=lambda l: (len(l['actors']) != 2,
        len(l['fidelity']['native_members']) != 2, not l['fidelity']['exact_native_profile'], l['lesson_sha256']))


def manifest(entries):
    """entries contain job + raw results; retain negative and missing receipts."""
    entries = list(entries)
    budget = budget_quote([e['job'] for e in entries])
    rows = [dict(job=e['job'], results=e['results'], admission=admit(e['job'], e['results'])) for e in entries]
    return seal(dict(schema=SCHEMA, entries=rows, budget=budget,
        admitted=sum(r['admission']['admitted'] for r in rows), diagnostics=sum(not r['admission']['admitted'] for r in rows)),
        'manifest_sha256')


def read_manifest(path):
    # Existing F3 path guard rejects lexical AND resolved restricted paths.
    from .track2_backend import _path
    path = _path(path)
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    check(value, 'manifest_sha256')
    if value.get('schema') != SCHEMA or value != manifest(value['entries']):
        raise ValueError('G1 manifest admission/evidence changed')
    return value


def restart_binding(snapshot):
    """Canonical bindings from the already verified F3 restart, no path reads."""
    from . import sumo_verify as sv
    return dict(snapshot_sha256=sv.snapshot_sha(snapshot),
        prefix_sha256=digest(dict(prefixes=snapshot['prefixes'], samples=snapshot['samples'])),
        base_policy_sha256=_sha(snapshot['policy']['state_dict_sha256']))


def bank_certificate(value, *, arm, receipt, example, plan, snapshot_sha256,
                     prefix_sha256=None, base_policy_sha256=None, runtime_sha256=None):
    """Select an actually admitted exact-source binding, not a handoff boolean."""
    matches = []
    for entry in value['entries']:
        lesson = entry['job']['lesson']
        if (lesson['arm'] == arm and lesson['verification_id'] == receipt['verification_id']
                and lesson['source_receipt_sha256'] == receipt['receipt_sha256']
                and lesson['episode'] == receipt['episode'] and lesson['t0'] == receipt['t0']
                and lesson['window'] == receipt['window']
                and lesson['source_example_sha256'] == digest(example)
                and lesson['source_plan_sha256'] == digest(plan)
                and lesson['snapshot_sha256'] == snapshot_sha256
                and all(expected is None or lesson[key] == expected for key, expected in (
                    ('prefix_sha256', prefix_sha256), ('base_policy_sha256', base_policy_sha256),
                    ('runtime_sha256', runtime_sha256))) and entry['admission']['admitted']):
            matches.append(lesson)
    if len(matches) > 1:
        raise ValueError('G1 ambiguous executable lesson for source example')
    return matches[0] if matches else None


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('quote', 'certify'))
    parser.add_argument('--input', help='JSON {entries: [{job, results}]} for certify')
    args = parser.parse_args()
    if args.command == 'quote':
        value = budget_quote()
    else:
        if not args.input:
            parser.error('certify requires --input')
        from .track2_backend import _path
        raw = json.loads(_path(args.input).read_text(encoding='utf-8'))
        value = manifest(raw['entries'])
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
