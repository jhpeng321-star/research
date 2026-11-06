"""Clean3 mined-plan exposure: smoke proofs (a)-(d) and per-update usage reports.

Owner 2026-11-06 ~05:05: the policy must be SHOWN what the mined plans can do.
B and R use identical rules on their own admitted mined examples; A is untouched.
Nothing here changes training; it reads receipts, tapes and the bank.
"""
from collections import Counter
import json
import math
from pathlib import Path

from .protocol import digest

EXECUTABLE_G5 = ('g5_move', 'lane_actions', 'uses_lane_actions', 'profile', 'policy_profile',
                 'perturbation_profile')


def rows(cfg):
    path = cfg.get('mined_moves')
    if not path:
        return []
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def examples(native, arm):
    """Every admitted mined (move, seed) example the arm's native bank reader yields."""
    from . import bank_breadth
    wanted = {(r['verification_id'], r['plan_sha256']) for r in rows(native) if r['arm'] == arm}
    out = []
    def consume(*, cfg, receipt, example, plan, pairs, trace):
        key = (receipt['verification_id'], digest(plan))
        if '_g5_mined' not in cfg or key not in wanted:
            return
        try:
            move = bank_breadth.mined_move(plan)
            missing = [k for k in EXECUTABLE_G5 if k not in plan] if plan.get('g5_move') else []
        except ValueError as error:
            out.append(dict(verification_id=key[0], plan_sha256=key[1], executable=False,
                missing_fields=[str(error)], example_id=(example or {}).get('example_id', key[0]+'#s1')))
            return
        out.append(dict(verification_id=key[0], plan_sha256=key[1], family=move.family,
            members=len(move.members), example_id=(example or {}).get('example_id', key[0]+'#s1'),
            seed_index=int((example or {}).get('seed_index', 1)), executable=not missing,
            missing_fields=missing, imitation_pairs=len(pairs or ())))
    bank_breadth.visit(native, arm, consume)
    return out, wanted


def observability(found, contexts):
    """Per (example, seed): observable from some member's slots == has an imitation context."""
    rows = []
    for e in found:
        rows.append(dict(verification_id=e['verification_id'], example_id=e['example_id'],
                         seed_index=e.get('seed_index'), family=e.get('family'),
                         observable=(e['verification_id'], e['example_id']) in contexts))
    by_seed = {}
    for r in rows:
        key = str(r['seed_index'])
        by_seed.setdefault(key, dict(observable=0, non_observable=0))
        by_seed[key]['observable' if r['observable'] else 'non_observable'] += 1
    return dict(examples=rows, observable=sum(r['observable'] for r in rows),
                non_observable=sum(not r['observable'] for r in rows), by_seed=by_seed,
                rule='imitation targets only where a member own observable slots contain every macro member')


def tape_check(plan, tape, t0):
    """The demonstrated plan really executed: G5 lane effect + bound profile, else profile."""
    from .group_lane import verify_tape, from_plan
    if plan.get('g5_move'):
        verify_tape(plan, tape, t0)
        return dict(kind='g5_lane_and_profile', passed=True)
    from .plans import acceleration_at
    steps = {int(s['time_s']): s for s in tape['steps']}
    covered = 0
    for t in range(int(plan['duration_s'])):
        step = steps.get(int(t0)+t)
        if step is None:
            continue
        executed = (step.get('sample') or {}).get('absolute_accels') or {}
        for vid, value in acceleration_at(plan, t).items():
            if vid not in executed or abs(float(executed[vid])-float(value)) > 1e-8:
                raise ValueError('demonstration did not execute the verified profile')
        covered += 1
    if covered == 0:
        raise ValueError('demonstration tape does not cover the plan window')
    return dict(kind='profile', passed=True, covered_s=covered)


def demonstrations(episodes, artifacts):
    """Per practice episode: demonstrated mined plan and its actual-execution check."""
    from .accel_storage import read
    out = []
    for episode in episodes:
        practice = episode.get('practice') or {}
        demo = practice.get('demonstration')
        if not demo:
            continue
        row = dict(id=episode['id'], verification_id=practice['verification_id'],
                   example_id=practice['example_id'], seed_index=int(practice['seed_index']),
                   family=(practice.get('breadth') or {}).get('family'), plan_sha256=demo['plan_sha256'])
        try:
            plan = practice['plan']
            if digest(plan) != demo['plan_sha256']:
                raise ValueError('demonstration plan binding changed')
            tape = read(Path(artifacts)/'jobs'/episode['id']/'RESULT.json')
            row.update(executed=tape_check(plan, tape, int(practice['t0'])))
        except (ValueError, KeyError, TypeError, OSError) as error:
            row.update(executed=dict(passed=False, reason=str(error)[:300]))
        out.append(row)
    return out


def credit_rows(native, arm, demos):
    """G4-format measured synergy rows for demonstrated joint moves.

    Source: the mining verification of that exact plan (group vs each member solo,
    common full window, matched CRN), target formula mean_seed(min solo J - group J),
    normalised/clipped with the registered G4 UNIT/clip. Recorded per update; not an
    extra PPO term (the G4 PPO registration is unchanged).
    """
    from .sparse_synergy_credit import REGISTRATION
    by_key = {(r['verification_id'], r['plan_sha256']): r for r in rows(native) if r['arm'] == arm}
    out = []
    for demo in demos:
        if not demo['executed'].get('passed'):
            continue
        row = by_key.get((demo['verification_id'], demo['plan_sha256']), {})
        synergy = (row.get('verified_synergy') or {})
        values = [float(v) for v in (synergy.get('veh_s') or {}).values() if v is not None]
        lows = [float(v) for v in (synergy.get('lower95') or {}).values() if v is not None]
        if not values:
            continue
        mean = sum(values)/len(values)
        clip = REGISTRATION['advantage_clip']
        out.append(dict(verification_id=demo['verification_id'], plan_sha256=demo['plan_sha256'],
            family=demo['family'], episode=demo['id'], measured_synergy_veh_s=mean,
            synergy_lower95_veh_s=min(lows) if lows else None,
            normalized_advantage=max(-clip, min(clip, REGISTRATION['weight']*mean/REGISTRATION['normalization_veh_s'])),
            source='mining verification: group vs each member solo, full window, matched CRN',
            target=REGISTRATION['target'], used_in_ppo=False))
    return out


def usage(native, arm, episodes, learning, artifacts):
    """Per-update mined usage: imitation, practice starts, executed joint moves, fit, trigger rates."""
    mined = {(r['verification_id']) for r in rows(native) if r['arm'] == arm}
    bank = learning.get('bank_trigger') or {}
    contexts = bank.get('contexts') or []
    ce = bank.get('context_final_ce') or []
    imitation = Counter()
    fit = []
    for i, c in enumerate(contexts):
        if c['verification_id'] in mined:
            imitation[(c['family'], c['example_id'])] += 1
            if i < len(ce) and ce[i] is not None:
                fit.append(float(ce[i]))
    practice = Counter()
    for e in episodes:
        p = e.get('practice') or {}
        if p.get('verification_id') in mined:
            practice[((p.get('breadth') or {}).get('family'), p['example_id'])] += 1
    demos = demonstrations(episodes, artifacts)
    executed = Counter((d['family'], d['example_id']) for d in demos if d['executed'].get('passed'))
    selected, decisions = Counter(), 0
    for e in episodes:
        for s in e['steps']:
            trig = (s.get('sample') or {}).get('track2_trigger')
            if not trig:
                continue
            for i, table in enumerate(trig['tables']):
                decisions += 1
                action = trig['actions'][i]
                move = table[action] if 0 <= action < len(table) else None
                fam = (move or {}).get('family', 'none')
                if fam != 'none' and len((move or {}).get('members', ())) >= 2:
                    fam += ':joint'
                selected[fam] += 1
    key = lambda c: {f'{f}|{x}': n for (f, x), n in sorted(c.items())}
    try:
        found, _ = examples(native, arm)
        seen = {(c['verification_id'], c['example_id']) for c in contexts}
        obs = observability(found, seen)
    except (ValueError, KeyError, TypeError, OSError) as error:
        obs = dict(error=str(error)[:300])
    return dict(arm=arm, mined_examples=len(mined), observability=obs, imitation=key(imitation), practice_starts=key(practice),
        executed_joint_moves=key(executed), demonstrations=demos,
        imitation_fit_ce=(sum(fit)/len(fit) if fit else None), imitation_fit_contexts=len(fit),
        trigger_rate_by_family={f: n/decisions for f, n in sorted(selected.items())} if decisions else {},
        trigger_decisions=decisions, g4_credit=credit_rows(native, arm, demos))


def smoke_check(cfg, updates, artifacts):
    """Owner (a)-(d) for B's mined examples; R exposed symmetrically on its own examples."""
    from . import bank_breadth, launch_production as production
    native = production.backend_config(cfg, root=artifacts, smoke=True)
    arms = {u['arm']: u for u in updates}
    demo_families = set(native.get('demonstrate_families') or ())
    g5_admitted = {r['arm'] for r in rows(native) if r.get('g5_move')}
    out = {}
    for arm in ('B', 'R'):
        found, wanted = examples(native, arm)
        if arm == 'B' and not wanted:
            raise ValueError('owner rule: admitted mined rows required for B')
        pool = bank_breadth.practice_pool(native, arm)
        demo_ctx = {r['context_id']: r for r in pool if r.get('demonstrate')}
        info = dict(examples=len(found))
        if wanted:
            bad = [e for e in found if not e['executable']]
            if bad or {(e['verification_id'], e['plan_sha256']) for e in found} != wanted:
                raise ValueError('(a) every admitted mined move must load as an executable bank example: '+arm)
            contexts = {(c['verification_id'], c['example_id']) for c in
                        arms[arm]['learning']['bank_trigger'].get('contexts', [])}
            targets = {(e['verification_id'], e['example_id']) for e in found}
            # MAIN 10:2x amendment (option 1): imitation targets cover only the examples
            # observable from a member's own slots; every example must still load as an
            # executable bank example (above) and be practice-demonstrable (pool, below).
            info['observability'] = observability(found, contexts)
            pool_examples = {(r['verification_id'], r['example_id']) for r in pool if r.get('mined')}
            if not targets <= pool_examples:
                raise ValueError('(b) mined seed situations missing from the practice pool: '+arm)
            info['imitation_targets'] = len(targets & contexts)
        started = [e for e in arms[arm]['practice']['episodes']
                   if (e.get('breadth') or {}).get('context_id') in demo_ctx
                   and (arm != 'B' or demo_ctx[e['breadth']['context_id']].get('mined'))]
        if not started:
            raise ValueError('(b) smoke practice did not start from a '+('mined' if arm == 'B' else
                             'mined-family') + ' seed situation: '+arm)
        demos = arms[arm]['clean3_mined_usage']['demonstrations']
        passed = [d for d in demos if d['executed'].get('passed')]
        if not passed:
            raise ValueError('(c) no demonstrated joint move executed its verified profile: '+arm)
        if arm in g5_admitted and not any(d['executed'].get('kind') == 'g5_lane_and_profile' for d in passed):
            raise ValueError('(c) smoke must run a real G5 lane move through the bound runtime env: '+arm)
        credit = arms[arm]['clean3_mined_usage']['g4_credit']
        if wanted and not credit:
            raise ValueError('(d) G4-format measured synergy rows missing for demonstrated moves: '+arm)
        if demo_families and not {d['family'] for d in passed} & demo_families:
            raise ValueError('(c) demonstrated family must be a mined family: '+arm)
        info.update(practice_starts=len(started), demonstrations=[{k: d[k] for k in ('id', 'family', 'executed')}
                    for d in demos], g4_credit=len(credit))
        out[arm] = info
    b, r = arms['B']['practice'], arms['R']['practice']
    if b['practice'] != r['practice'] or [e['breadth']['family'] for e in b['episodes'] if e.get('breadth')] !=             [e['breadth']['family'] for e in r['episodes'] if e.get('breadth')]:
        raise ValueError('fairness: R practice counts/families must equal B')
    return out
