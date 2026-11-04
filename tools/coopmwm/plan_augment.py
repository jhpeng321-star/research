"""T2G: local, CPU-only augmentation of verified cooperative bank examples.

Default OFF. B and R call the same API, each with its OWN bank. No files are
opened and no physics is dispatched by ``augment``. Enabled settings must come
from the run registration; in particular there is no invented perturbation
radius, verification fraction, WM sample count or shortened look-ahead.

Source contract::

    {"confirmation": <sealed all8 confirmation>, "example_index": int,
     "parameters": {"onset_s": float, "hold_s": float, "depth": float},
     "context": {"ego_id": str, "t_s": float, "rows": [...],
                 "policy_traj": {cav_id: [base_accel, ...]}}}

``parameters`` are labels of the source plan (e.g. from moves.from_plan), NOT
fresh draws. The source's effective residual is time-warped around these labels,
depth-scaled and added to the recipient's own base policy trajectory. The full
plan duration stays fixed. At most one non-ego partner is replaced, by a t-1
visible CAV in the same observable lane (preserving the family's partner role).
Rows must be the acting CAV's sensing/V2V view, never a global vehicle table;
only whitelisted observable fields reach the scorer.

WM callback: ``score(context, plans, *, samples, horizon_s)`` -> mapping of
plan id to paired cost samples (LOWER is better), using the same belief and
sample worlds for every plan. Plans contain an explicit base and every solo.
Keep iff mean(base - group) > 0 AND mean(best-solo - group) > 0. Missing,
nonfinite or unpaired samples fail closed. These are WM labels, not SUMO labels.

Selected spot checks are pending and excluded from training until resolved.
``verify_all8`` is the explicit later-run bridge to existing charged machinery;
``resolve_spot_checks`` validates its sealed results. Neither is called by
``augment``. There is no CLI that could accidentally launch a job.
"""
import copy
import math

import numpy as np

from . import mppi_search as ms
from . import variety
from .protocol import digest


SCHEMA = 'coopmwm/plan-augment/v1'
ALL8 = 'all8_both_seeds'
WINDOWS = dict(onset_s='variety_onset_window_s', hold_s='variety_hold_window_s',
               depth='variety_depth_window')
SETTINGS = {'enabled', 'seed', 'attempts_per_plan', 'parameter_fraction',
            'partner_swap_probability', 'spot_check_fraction', 'wm_samples',
            'horizon_s', 'bounds', 'variety'}
OBS_FIELDS = ('vid', 'is_cav', 's_m', 'v_mps', 'lane', 'allowed_speed_mps', 'observed_at_s')


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _finite(value, name):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f'{name} must be finite')
    return x


def config(raw=None):
    """Validate registered settings; None / {enabled: False} is exactly OFF."""
    c = dict(raw or {})
    if set(c)-SETTINGS:
        raise ValueError(f'unregistered plan_augment keys {sorted(set(c)-SETTINGS)}')
    if not isinstance(c.get('enabled', False), bool):
        raise ValueError('enabled must be boolean')
    if not c.get('enabled', False):
        return None
    if SETTINGS-set(c):
        raise ValueError(f'enabled plan_augment requires registered keys {sorted(SETTINGS-set(c))}')
    for name, minimum in (('seed', 0), ('attempts_per_plan', 1), ('wm_samples', 2), ('horizon_s', 1)):
        c[name] = _integer(c[name], name, minimum)
    for name in ('parameter_fraction', 'partner_swap_probability', 'spot_check_fraction'):
        c[name] = _finite(c[name], name)
        if not 0 <= c[name] <= 1:
            raise ValueError(f'{name} must be in [0, 1]')
    if c['parameter_fraction'] == 0:
        raise ValueError('parameter_fraction must be positive')
    bounds = tuple(_finite(x, 'bounds') for x in c['bounds'])
    if len(bounds) != 2 or not bounds[0] < bounds[1] <= 1.2:
        raise ValueError('bounds must be ordered and respect the CAV acceleration cap 1.2')
    c['bounds'] = bounds
    v = dict(c['variety'])
    if set(v)-set(variety.VARIETY_REGISTERED):
        raise ValueError('unregistered variety keys')
    # Validate ranges even if the older search_families switch is off.
    v['variety_families'] = True
    c['variety'] = variety.plan_config(v)
    return c


def _sealed(record):
    from .exact_grade import verify_sealed
    verify_sealed(record, 'confirmation_sha256')
    if record.get('verify_mode') != ALL8 or record.get('bank_rule') != 'per_seed_v1':
        raise ValueError('a sealed all8 per_seed_v1 confirmation is required')


def _source(source, arm, c):
    record = source['confirmation']
    _sealed(record)
    if record.get('arm') != arm:
        raise ValueError('augmentation must use the arm\'s own bank')
    i = _integer(source['example_index'], 'example_index')
    ex = record['per_seed_v1']['examples'][i]
    if ex.get('admitted') is not True or ex.get('cooperative') is not True:
        raise ValueError('only verified admitted cooperative examples may be augmented')
    plan = ex['plan']
    members = plan['members']
    if not 2 <= len(members) <= 4 or len(set(members)) != len(members):
        raise ValueError('cooperative moves require 2..4 distinct CAVs')
    D = _integer(plan['duration_s'], 'duration_s', 1)
    if c['horizon_s'] < D:
        raise ValueError('registered full look-ahead cannot be shorter than the plan')
    params = {k: _finite(source['parameters'][k], k) for k in WINDOWS}
    for key, window in WINDOWS.items():
        lo, hi = c['variety'][window]
        if not lo <= params[key] <= hi:
            raise ValueError(f'source {key} outside registered variety range')
    if params['depth'] <= 0 or params['hold_s'] <= 0:
        raise ValueError('source depth and hold must be positive')
    raw = source['context']
    ego, now = raw['ego_id'], _finite(raw['t_s'], 't_s')
    if now != record['t0']:
        raise ValueError('source context time must match its verified decision')
    rows = []
    for r in raw['rows']:
        row = {k: r[k] for k in OBS_FIELDS}
        if row['is_cav'] is not True:
            continue  # HDVs can never become a commanded member
        for k in ('s_m', 'v_mps', 'allowed_speed_mps', 'observed_at_s'):
            row[k] = _finite(row[k], k)
        if row['observed_at_s'] > now-1:
            raise ValueError('only t-1 observable rows may be used')
        if row['v_mps'] < 0 or row['allowed_speed_mps'] <= 0:
            raise ValueError('invalid observable speed')
        rows.append(row)
    by = {r['vid']: r for r in rows}
    if len(by) != len(rows) or ego not in members or not set(members) <= set(by):
        raise ValueError('ego and every source member must be distinct visible CAVs')
    policies = {}
    for vid in by:
        p = np.asarray(raw['policy_traj'][vid], float)
        if p.ndim != 1 or len(p) < D or not np.isfinite(p).all():
            raise ValueError('each visible CAV needs a finite full-duration base trajectory')
        policies[vid] = p[:D].tolist()
    residual = np.asarray(plan['perturbation_profile'], float)
    profile = np.asarray(plan['profile'], float)
    if residual.shape != (len(members), D) or profile.shape != residual.shape:
        raise ValueError('source profile shape does not match its members and duration')
    if not np.isfinite(residual).all() or not np.isfinite(profile).all():
        raise ValueError('source profile must be finite')
    # The source residual must be relative to THIS source policy, not another seed/actor.
    base = np.asarray([policies[v] for v in members])
    if not np.allclose(profile-residual, base, rtol=0, atol=1e-8):
        raise ValueError('source context does not match the verified plan policy centre')
    context = dict(ego_id=ego, t_s=now, rows=rows, policy_traj=policies)
    return ex, plan, params, context


def _build(context, members, profile, source_plan, c, tag):
    D = source_plan['duration_s']
    P = np.asarray([context['policy_traj'][v] for v in members])
    feas = ms.centre_feasibility(context, members, P, c['bounds'], [D])
    if feas is None:
        raise ValueError('observable actuator feasibility is required')
    p = ms.build_plan(dict(members=members, id=source_plan.get('group_id')), P, None,
                      D, c['bounds'], 'sampled', source_plan['regime'], tag,
                      absolute=profile, feasibility=feas)
    if 'variety_family' in source_plan:
        p['variety_family'] = source_plan['variety_family']
    return p


def _perturb(plan, params, context, c, rng):
    new = {}
    for key, window in WINDOWS.items():
        lo, hi = (float(x) for x in c['variety'][window])
        radius = c['parameter_fraction']*(hi-lo)
        new[key] = float(rng.uniform(max(lo, params[key]-radius), min(hi, params[key]+radius)))
    if new['hold_s'] <= 0:
        return None
    old_members, members = list(plan['members']), list(plan['members'])
    by = {r['vid']: r for r in context['rows']}
    if rng.random() < c['partner_swap_probability']:
        swaps = [(i, v) for i, old in enumerate(members) if old != context['ego_id']
                 for v in sorted(by) if v not in members and by[v]['lane'] == by[old]['lane']]
        if swaps:
            i, v = swaps[int(rng.integers(len(swaps)))]
            members[i] = v
    D = plan['duration_s']
    t = np.arange(D, dtype=float)
    # Move the source onset and stretch its hold without truncating the WM horizon.
    mapped = params['onset_s']+(t-new['onset_s'])*params['hold_s']/new['hold_s']
    residual = np.stack([np.interp(mapped, t, row, left=0., right=0.)
                         for row in plan['perturbation_profile']])
    residual *= new['depth']/params['depth']
    P = np.asarray([context['policy_traj'][v] for v in members])
    candidate = _build(context, members, P+residual, plan, c, 'AUG')
    effective = np.asarray(candidate['profile'])-P
    if candidate.get('policy_duplicate') or not np.any(effective):
        return None
    if members == old_members and np.array_equal(candidate['profile'], plan['profile']):
        return None
    return candidate, new


def _screen(context, plan, c, score):
    members = plan['members']
    P = np.asarray([context['policy_traj'][v] for v in members])
    base = _build(context, members, P, plan, c, 'BASE')
    solos = []
    for i in range(len(members)):
        profile = P.copy()
        profile[i] = plan['profile'][i]
        solos.append(_build(context, members, profile, plan, c, f'SOLO{i}'))
    plans = [base, plan]+solos
    values = score(copy.deepcopy(context), copy.deepcopy(plans),
                   samples=c['wm_samples'], horizon_s=c['horizon_s'])
    try:
        costs = np.asarray([values[p['id']] for p in plans], float)
    except (KeyError, TypeError, ValueError):
        return None
    if costs.shape != (len(plans), c['wm_samples']) or not np.isfinite(costs).all():
        return None
    # Best solo is selected by expected gain on the SAME paired futures.
    best = min(costs[2:], key=lambda x: float(np.mean(x)))
    gain, synergy = float(np.mean(costs[0]-costs[1])), float(np.mean(best-costs[1]))
    if gain <= 0 or synergy <= 0:
        return None
    return dict(group_gain=gain, best_solo_gain=float(np.mean(costs[0]-best)),
                synergy=synergy, samples=c['wm_samples'], horizon_s=c['horizon_s'])


def augment(sources, *, arm, registration=None, score=None):
    """Return serializable examples, pending all8 requests and a CPU receipt.

    Off never consumes ``sources`` or calls a callback. This function is pure:
    no writes, subprocesses, SUMO, remote dispatch or model loading.
    """
    if arm not in ('B', 'R'):
        raise ValueError('plan augmentation is for B and R only')
    c = config(registration)
    out = dict(schema=SCHEMA, arm=arm, enabled=c is not None, examples=[], spot_checks=[],
               counts=dict(sources=0, proposed=0, wm_scored=0, wm_kept=0, spot_checks=0),
               registration=copy.deepcopy(c), sumo_dispatched=False)
    if c is None:
        out['receipt_sha256'] = digest(out)
        return out
    if not callable(score):
        raise ValueError('enabled augmentation needs a belief-only WM scorer')
    rng_seed, spot_seed = np.random.SeedSequence(c['seed']).spawn(2)
    rng, spot_rng = np.random.default_rng(rng_seed), np.random.default_rng(spot_seed)
    seen, retained = set(), []
    for source in sources:
        ex, plan, params, context = _source(source, arm, c)
        out['counts']['sources'] += 1
        for _ in range(c['attempts_per_plan']):
            out['counts']['proposed'] += 1
            draw = _perturb(plan, params, context, c, rng)
            if draw is None:
                continue
            candidate, new = draw
            identity = digest(dict(profile=candidate['profile'], members=candidate['members'],
                                   source=source['confirmation']['confirmation_sha256'],
                                   seed_index=ex['seed_index']))
            if identity in seen:
                continue
            seen.add(identity)
            out['counts']['wm_scored'] += 1
            wm = _screen(context, candidate, c, score)
            if wm is None:
                continue
            retained.append(dict(id='AUG'+identity[:24], arm=arm, plan=candidate, parameters=new,
                                 source_plan_id=plan['id'], source_seed_index=ex['seed_index'],
                                 source_t_s=context['t_s'],
                                 source_confirmation_sha256=source['confirmation']['confirmation_sha256'],
                                 source_example_index=source['example_index'], wm=wm,
                                 verification='wm_only', sumo_verified=False))
    n = int(math.ceil(c['spot_check_fraction']*len(retained)))
    selected = set(int(i) for i in spot_rng.permutation(len(retained))[:n])
    for i, example in enumerate(retained):
        if i in selected:
            example['verification'] = 'pending_all8'
            out['spot_checks'].append(example)
        else:
            out['examples'].append(example)
    out['counts'].update(wm_kept=len(retained), spot_checks=n)
    out['receipt_sha256'] = digest(out)
    return out


def resolve_spot_checks(batch, checker):
    """Run an explicitly supplied checker; accept only matching sealed all8 examples.

    The callback receives an immutable-by-copy request and returns an ordinary
    all8 confirmation. Existing all8 caching/charging owns dispatch idempotency.
    Return a new batch; do not relabel WM-only examples as SUMO-verified.
    """
    if batch.get('receipt_sha256') != digest({k: v for k, v in batch.items() if k != 'receipt_sha256'}):
        raise ValueError('augmentation receipt identity mismatch')
    out = copy.deepcopy(batch)
    out.pop('receipt_sha256')
    out['spot_check_results'] = []
    for request in out['spot_checks']:
        record = checker(copy.deepcopy(request))
        _sealed(record)
        if record.get('arm') != out['arm']:
            raise ValueError('spot-check arm mismatch')
        examples = [ex for ex in record['per_seed_v1']['examples']
                    if ex.get('admitted') is True and ex.get('cooperative') is True
                    and ex.get('plan') == request['plan']]
        for ex in examples:
            accepted = copy.deepcopy(request)
            accepted.update(id=request['id']+f"/seed{ex['seed_index']}", verification='all8',
                            sumo_verified=True, confirmation_sha256=record['confirmation_sha256'],
                            seed_index=ex['seed_index'], verified_example=copy.deepcopy(ex))
            out['examples'].append(accepted)
        out['spot_check_results'].append(dict(id=request['id'], accepted_examples=len(examples),
                                             confirmation_sha256=record['confirmation_sha256']))
    out['spot_checks'] = []
    out['counts']['spot_checks_resolved'] = len(out['spot_check_results'])
    out['counts']['sumo_verified_examples'] = sum(x['sumo_verified'] for x in out['examples'])
    out['receipt_sha256'] = digest(out)
    return out


def spot_check_quote(request, t0):
    """Conservative all8 quote, including both seeds, later harm and every solo."""
    from .verify_perseed import quote_decision
    plan = request['plan']
    return quote_decision(t0, [plan['duration_s']], [plan['members']], {1: [0], 2: [0]})


def verify_all8(request, *, snapshot, root, run_root, runner_for, phase,
                workers=2, health=None):
    """EXPLICIT later-run physics adapter; never called by augment or CPU build.

    Caller binds the source's OWN snapshot/seed/actor and the registered single
    ledger root. Both first-seed verification and all8 confirmation use existing
    health, runner admission, charge, safety and later-harm machinery unchanged.
    """
    from pathlib import Path
    from . import replay_bank as rb
    from . import verify_perseed as vp
    arm = request['arm']
    if arm not in ('B', 'R') or request['verification'] != 'pending_all8':
        raise ValueError('expected a pending B/R augmentation request')
    if snapshot['t0'] != request['source_t_s']:
        raise ValueError('spot-check snapshot time does not match the source')
    plans = [request['plan']]
    first_dir = Path(rb.bank_root(root, arm))/'augment_first_all8'
    receipt = vp._verify('augment_first_all8', snapshot, plans, first_dir, runner_for,
                         run_root, phase, 'wm' if arm == 'B' else 'random', workers,
                         rb.sv.TAIL_S, [], root, arm, health)
    return vp.confirm_all8(root, arm, receipt, snapshot, runner_for, run_root, phase,
                          workers=workers, candidates=plans, branches_dir=first_dir/'branches',
                          health=health, later_harm_seed1=True, later_harm_seed2=True)
