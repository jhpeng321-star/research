"""Conservative statistical bank views; original receipts stay immutable.

Two process-local rules, selected only through ``install(rb, bank_rule=...)``:

- ``'t1727'`` (MAIN17:27, legacy adapter): two-seed Student-t (df=1) CIs of the seed gains.
- ``'boot_v2'`` (default of install(), i.e. of the V2V reader): per-unit paired bootstrap.

Nothing changes when install() is not called (replay_bank.confirmations is untouched).
"""
import copy
import json
import math
from pathlib import Path


def replicated_ci(first,second):
    """Two-sided 95% Student-t mean CI over two independent CRN seeds (df=1).

    The extended window is NOT an independent replicate and is never a third
    observation. This deliberately exposes the large uncertainty at n=2.
    """
    if first is None or second is None:return None
    a,b=float(first),float(second)
    if not math.isfinite(a) or not math.isfinite(b):return None
    mean=(a+b)/2;half=12.706204736432095*abs(a-b)/2
    return [mean-half,mean+half]


def view(confirmation):
    c=copy.deepcopy(confirmation)
    ci=replicated_ci(c.get('first_seed_advantage_veh_s'),c.get('replication_advantage_veh_s'))
    c['legacy_admitted']=c.get('admitted',False)
    c['replicated_advantage_ci95']=ci
    c['admitted']=bool(c['legacy_admitted'] and ci is not None and ci[0]>0)
    co=c.get('cooperation') or {}
    co['legacy_cooperative']=co.get('cooperative')
    # A first-seed solo comparison cannot establish a replicated synergy CI.
    # Accept only independently recorded seed-level gains, never member-level
    # pseudo-replicates or a CI across unrelated bank states.
    rep=co.get('replication_cooperation_gain_veh_s')
    sci=replicated_ci(co.get('cooperation_gain_veh_s'),rep)
    co['synergy_ci95']=sci
    co['cooperative']=bool(c['admitted'] and co.get('complete') and sci is not None and sci[0]>0)
    co['statistical_reason']=('replicated synergy CI lower > 0' if co['cooperative'] else
        'replicated synergy CI unavailable or does not exclude zero')
    c['cooperation']=co
    c['statistical_rule']='two independent seed Student-t CI95 lower > 0; extended/safety constraints retained'
    return c


REQUIRED_KEYS=('first_seed_advantage_veh_s','replication_advantage_veh_s',
               'cooperation.replication_cooperation_gain_veh_s')


def _applicable(c,key):
    """Records whose producer would carry ``key``; failed confirmations / incomplete or
    single-CAV cooperation legitimately lack the replicated values."""
    if key=='first_seed_advantage_veh_s':return True
    if key=='replication_advantage_veh_s':
        return 'replication' in (c.get('checks') or {}) or 'extended_advantage_veh_s' in c
    co=c.get('cooperation') or {}
    return bool(co.get('complete')) and co.get('cooperation_gain_veh_s') is not None


def _has(c,key):
    for part in key.split('.'):
        if not isinstance(c,dict) or part not in c:return False
        c=c[part]
    return True


def check_keys(records,bank_rule='t1727'):
    """Schema guard: a statistical key absent from EVERY applicable record is a schema
    mismatch, not evidence; raise instead of silently rejecting the whole bank."""
    records=list(records)
    if bank_rule=='boot_v2':
        rows=[c for c in records if 'confirmation_branches_complete' not in (c.get('checks') or {})]
        for key in ('verification_id','replication.verification_id','checks.no_later_harm','checks.safety'):
            if rows and not any(_has(c,key) for c in rows):
                raise KeyError(f'bank_statistics boot_v2: key {key!r} missing from all applicable confirmations')
        return
    if bank_rule!='t1727':raise ValueError('unknown schema rule '+str(bank_rule))
    for key in REQUIRED_KEYS:
        rows=[c for c in records if _applicable(c,key)]
        if rows and not any(_has(c,key) for c in rows):
            raise KeyError(f'bank_statistics: key {key!r} missing from all {len(rows)} applicable '
                           'confirmation records; refusing to reject the whole bank silently')


def _install_t1727(rb):
    """Schema-checked legacy adapter; boot_v2 remains the registered default."""
    if getattr(rb.confirmations,'statistical_1727',False):return
    original=rb.confirmations
    def confirmations(root,arm):
        records=original(root,arm);check_keys(records.values())
        return {k:view(v) for k,v in records.items()}
    confirmations.statistical_1727=True
    rb.confirmations=confirmations


# ---------------------------------------------------------------- rule v2: paired unit bootstrap

PER_SEED = 'per_seed_v1'
BANK_RULES = ('boot_v2', 't1727', PER_SEED)
BOOT_B = 2000
BOOT_RNG_SEED = 20261103
ALPHA_ONE_SIDED = 0.05
BLOCK_S = 30
# Optional per-vehicle outcome in a branch RESULT.json: {vehicle_id: team delay veh-s inside
# [t0, t_end)} (road speed deficit + source wait). NOT produced by any current plant; when
# present on both branches of every pair it is used instead of the per-step fallback.
PER_VEHICLE_KEY = 'per_vehicle_window_delay_vs'
UNIT_PER_VEHICLE = 'per_vehicle'
UNIT_PER_STEP = 'per_step_block'
RULE_V2 = ('boot_v2: per-unit paired difference (policy-continues minus plan, same seed, same '
           'window; positive = plan helped); bootstrap within each seed (B=2000, RNG 20261103), pooled '
           'by summing the per-seed resampled totals; admit iff one-sided 95% lower bound of the '
           'pooled total > 0 AND the per-unit mean is > 0 on EACH seed AND the existing later-harm '
           'and composite safety checks pass. Units: vehicles if branch tapes carry '
           f'{PER_VEHICLE_KEY!r}, else seconds of the exact team_delay_vs increments resampled as '
           f'circular {BLOCK_S} s blocks. Synergy: best-solo branch minus group branch, same bootstrap; '
           'cooperative only when replicated on two seeds.')


class BankDataMissing(ValueError):
    """Required per-unit outcome data absent for every record: never silently reject all."""


class _Missing(Exception):
    pass


def paired_bootstrap(units_by_seed, B=BOOT_B, rng_seed=BOOT_RNG_SEED, block=None, alpha=ALPHA_ONE_SIDED):
    """Pooled total over seeds; each seed's units resampled with replacement independently.

    units_by_seed: per-seed 1-D arrays of paired per-unit differences (positive = better).
    block=None -> iid units (vehicles); block=L -> circular block bootstrap of a time series.
    Returns the one-sided (1-alpha) lower bound of the pooled total plus per-seed summaries."""
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    totals = np.zeros(int(B))
    per_seed = []
    for x in units_by_seed:
        x = np.asarray(x, float).reshape(-1)
        n = len(x)
        if n == 0:
            raise _Missing('a seed has zero units')
        if not np.all(np.isfinite(x)):
            raise _Missing('non-finite per-unit difference')
        if block is None:
            L = None
            idx = rng.integers(0, n, (int(B), n))
        else:
            L = int(max(1, min(int(block), n//2 if n >= 2 else 1)))
            nb = -(-n//L)
            starts = rng.integers(0, n, (int(B), nb))
            idx = ((starts[:, :, None]+np.arange(L)[None, None, :]) % n).reshape(int(B), nb*L)[:, :n]
        boot = x[idx].sum(1)
        totals += boot
        per_seed.append(dict(n_units=n, total_veh_s=float(x.sum()), mean_per_unit_veh_s=float(x.mean()),
                             lower95_one_sided_veh_s=float(np.quantile(boot, alpha)), block_len=L))
    return dict(lower95_one_sided_veh_s=float(np.quantile(totals, alpha)),
                upper95_one_sided_veh_s=float(np.quantile(totals, 1-alpha)),
                pooled_total_veh_s=float(sum(s['total_veh_s'] for s in per_seed)),
                per_seed=per_seed, B=int(B), rng_seed=int(rng_seed), alpha_one_sided=alpha,
                resampling=('iid vehicles within each seed' if block is None
                            else 'circular time blocks within each seed'))


def _read_result(branches_dir, vid, branch):
    from .accel_storage import read
    from .protocol import digest
    p = Path(branches_dir)/vid/branch['branch']/'RESULT.json'
    if not p.exists():
        raise _Missing(f'missing branch tape {p}')
    tape = read(p)
    if branch.get('result_sha256') and digest(tape) != branch['result_sha256']:
        raise ValueError(f'branch tape {p} differs from its sealed receipt')
    return tape


def _step_series(tape, t0, t_end):
    import numpy as np
    from .branch_eval import totals_at
    try:
        cum = [totals_at(tape, t)['team_delay_vs'] for t in range(int(t0), int(t_end)+1)]
    except (KeyError, ValueError, TypeError) as e:
        raise _Missing(f'per-step team_delay_vs fingerprints absent in [{t0},{t_end}]: {e!r}')
    return np.diff(np.asarray(cum, float))


def _stats(pairs):
    """pairs: [(ref tape, alt tape, window)] per seed -> bootstrap of (ref - alt) per unit.

    Per-vehicle units only if EVERY tape carries PER_VEHICLE_KEY; otherwise per-step blocks."""
    import numpy as np
    if all(PER_VEHICLE_KEY in ref and PER_VEHICLE_KEY in alt for ref, alt, _ in pairs):
        units = []
        for ref, alt, _ in pairs:
            a, b = ref[PER_VEHICLE_KEY], alt[PER_VEHICLE_KEY]
            ids = sorted(set(a) | set(b))
            units.append(np.array([float(a.get(i, 0.))-float(b.get(i, 0.)) for i in ids]))
        source, block = UNIT_PER_VEHICLE, None
    else:
        units = [_step_series(ref, w['start_s'], w['end_s'])-_step_series(alt, w['start_s'], w['end_s'])
                 for ref, alt, w in pairs]
        source, block = UNIT_PER_STEP, BLOCK_S
    out = paired_bootstrap(units, block=block)
    out['unit_source'] = source
    return out


def _receipt(root_dir, entry):
    from .exact_grade import verify_sealed
    p = Path(root_dir)/'verifications'/(entry['verification_id']+'.json')
    if not p.exists():
        raise _Missing(f'missing verification receipt {p}')
    r = json.loads(p.read_text())
    verify_sealed(r, 'receipt_sha256')
    if entry.get('receipt_sha256') and r['receipt_sha256'] != entry['receipt_sha256']:
        raise ValueError(f'receipt {p} differs from the confirmation link')
    return r


def _admission(c, st):
    checks = c.get('checks') or {}
    if checks.get('no_later_harm') is not True:
        return False, 'later-harm check failed (extended-window advantage <= 0)'
    if checks.get('safety') is not True:
        return False, 'composite safety check failed'
    means = [s['mean_per_unit_veh_s'] for s in st['per_seed']]
    if not all(m > 0 for m in means):
        return False, 'per-seed mean not positive on every seed (%s)' % ', '.join('%.4g' % m for m in means)
    if not st['lower95_one_sided_veh_s'] > 0:
        return False, 'pooled one-sided 95%% lower bound %.4g <= 0' % st['lower95_one_sided_veh_s']
    return True, 'pooled lower95 > 0, positive mean on both seeds, later-harm and safety pass'


def _best_solo(solo_root, solos):
    """(member, solo advantage, solo plan tape, window) of the best member-only solo on one seed."""
    rows = []
    for s in solos:
        r = _receipt(solo_root, s)
        if r['status'] not in ('accepted', 'rejected') or r.get('J_candidates_veh_s') is None:
            raise _Missing(f'solo {s["verification_id"]} ungraded')
        rows.append((r['J_policy_veh_s']-r['J_candidates_veh_s'][0], s['member'], r))
    adv, member, r = max(rows, key=lambda x: x[0])
    return member, adv, _read_result(Path(solo_root)/'branches', r['verification_id'], r['branches'][1]), r['window']


def paired_window(group,solo):
    """Legacy top-eight group tails can be longer than the chosen-plan solos.

    Use their exact shared recorded window; never compare unmatched totals or
    extend a missing trace. The minimum end is a data-availability boundary,
    not an outcome-selected horizon.
    """
    if group['start_s']!=solo['start_s']:
        raise ValueError('solo and group start times differ; synergy not paired')
    end=min(group['end_s'],solo['end_s'])
    if end<=group['start_s']:raise ValueError('empty paired synergy window')
    return dict(group,end_s=end)


def _cooperation(bank, c, group_tapes, admitted):
    """Synergy per unit = best-solo branch minus group branch (same seed); positive = group better.

    Solos on the first seed live in <bank>/confirm_solo (legacy); second-seed solos, when the
    live agent runs them, are linked as cooperation['replication_solo'] (same row shape as
    'solo') with receipts/branches under <bank>/confirm_solo_replication."""
    co = copy.deepcopy(c.get('cooperation') or {})
    co['legacy_cooperative'] = co.get('cooperative')
    co['needs_second_seed_solos'] = False
    first_solos, rep_solos = co.get('solo') or [], co.get('replication_solo') or []
    if (co.get('group_size') or 0) < 2:
        co.update(cooperative=False, cooperative_status='single_cav',
                  statistical_reason='single-CAV plan: cooperative=False by definition')
        return co
    if co.get('complete') is not True or not first_solos:
        co.update(cooperative=None if co.get('complete') is False else False, cooperative_status='incomplete',
                  statistical_reason='first-seed solo branches incomplete; never imputed')
        return co
    seeds = [(bank/'confirm_solo', first_solos, group_tapes[0])]
    if rep_solos:
        seeds.append((bank/'confirm_solo_replication', rep_solos, group_tapes[1]))
    pairs, best = [], []
    paired_windows=[]
    for d, solos, (grp, window) in seeds:
        member, adv, solo_tape, sw = _best_solo(d, solos)
        common=paired_window(window,sw)
        pairs.append((solo_tape, grp, common))
        paired_windows.append(dict(group_window=window,solo_window=sw,used=common))
        best.append(dict(member=member, solo_advantage_veh_s=adv))
    st = _stats(pairs)
    st['best_solo_per_seed'] = best
    st['paired_windows'] = paired_windows
    co['synergy_v2'] = st
    positive = st['lower95_one_sided_veh_s'] > 0 and all(s['mean_per_unit_veh_s'] > 0 for s in st['per_seed'])
    if len(seeds) == 2:
        co['cooperative'] = bool(admitted and positive)
        co['cooperative_status'] = 'replicated' if positive else 'not_cooperative'
        co['statistical_reason'] = ('two-seed synergy lower95 > 0 and positive on both seeds' if positive
                                    else 'two-seed synergy lower95 <= 0 or a seed mean <= 0')
        if positive and not admitted:
            co['statistical_reason'] += ' (entry not admitted: not counted)'
    else:
        # Single-seed evidence is never counted as replicated cooperation; the entry stays
        # usable for imitation (as admitted, without cooperative oversampling).
        co['cooperative'] = False
        co['cooperative_status'] = 'single_seed' if positive else 'not_cooperative'
        co['needs_second_seed_solos'] = bool(admitted and positive)
        co['statistical_reason'] = ('first-seed synergy lower95 > 0; second-seed solos required before counting'
                                    if positive else 'first-seed synergy lower95 <= 0 or mean <= 0')
    return co


def _not_evaluated(c, reason, status):
    v = copy.deepcopy(c)
    v.update(legacy_admitted=c.get('admitted', False), admitted=False, bank_rule='boot_v2',
             statistical_rule=RULE_V2, bank_rule_v2=dict(admitted=False, reason=reason))
    co = copy.deepcopy(c.get('cooperation') or {})
    if co:
        co.update(legacy_cooperative=co.get('cooperative'), cooperative=False,
                  cooperative_status=status, needs_second_seed_solos=False)
        v['cooperation'] = co
    return v


def view_v2(c, first_receipt, bank):
    """One confirmation under rule v2 (original never mutated). Raises _Missing on absent data."""
    if 'confirmation_branches_complete' in (c.get('checks') or {}):
        return _not_evaluated(c, 'confirmation branches incomplete; never imputed', 'not_admitted')
    if first_receipt is None:
        raise _Missing('first-seed receipt not found in the bank or its sources')
    rep_root = bank/'confirm_replication'
    rep = _receipt(rep_root, c['replication'])
    if rep['status'] not in ('accepted', 'rejected'):
        raise _Missing('replication receipt ungraded')
    first_dir = first_receipt['_branches']
    tapes = []
    for r, d, idx in ((first_receipt, first_dir, first_receipt['chosen']+1), (rep, rep_root/'branches', 1)):
        tapes.append((_read_result(d, r['verification_id'], r['branches'][0]),
                      _read_result(d, r['verification_id'], r['branches'][idx]), r['window']))
    st = _stats(tapes)
    ok, reason = _admission(c, st)
    out = copy.deepcopy(c)
    out.update(legacy_admitted=c.get('admitted', False), admitted=ok, bank_rule='boot_v2', statistical_rule=RULE_V2,
               bank_rule_v2=dict(admission=st, admitted=ok, reason=reason,
                                 per_seed_means_veh_s=[s['mean_per_unit_veh_s'] for s in st['per_seed']],
                                 per_seed_totals_veh_s=[s['total_veh_s'] for s in st['per_seed']],
                                 n_units=[s['n_units'] for s in st['per_seed']]))
    if c.get('cooperation') is not None:
        out['cooperation'] = _cooperation(bank, c, [(alt, w) for _, alt, w in tapes], ok)
    return out


_CACHE = {}


def confirmations_v2(original, rb, root, arm, sources=()):
    """{vid: v2 view}. Per-entry missing data -> not admitted with the reason; raises
    BankDataMissing when NO complete confirmation has per-unit data (never silently reject all)."""
    conf = original(root, arm)
    check_keys(conf.values(),bank_rule='boot_v2')
    bank = Path(rb.bank_root(root, arm))
    from .solo_seed2_links import merge
    conf = merge(bank,conf)
    def key(vid):
        return (str(root), arm, vid, conf[vid].get('confirmation_sha256'))
    need = [vid for vid in conf if key(vid) not in _CACHE]
    if need:
        receipts = {r['verification_id']: r for r in rb.bank_decisions(root, arm, sources=sources)}
        fresh, missing, attempted = {}, {}, 0
        for vid in need:
            c = conf[vid]
            attempted += 'confirmation_branches_complete' not in (c.get('checks') or {})
            try:
                fresh[vid] = view_v2(c, receipts.get(vid), bank)
            except _Missing as e:
                missing[vid] = str(e)
                fresh[vid] = _not_evaluated(c, 'per-unit data missing: '+str(e), 'data_missing')
        if attempted and len(missing) >= attempted:
            raise BankDataMissing(
                f'boot_v2: per-unit outcome data absent for all {attempted} complete confirmations of arm {arm}; '
                'required: first-seed decision receipt (bank or sources) and RESULT.json tapes with per-second '
                f'team_delay_vs fingerprints (or {PER_VEHICLE_KEY!r}) for the first-seed policy/chosen, '
                'replication policy/cand1 and solo cand1 branches. First missing: '
                + json.dumps(dict(list(missing.items())[:3])))
        for vid, v in fresh.items():
            _CACHE[key(vid)] = v
    return {vid: _CACHE[key(vid)] for vid in conf}


# ---------------------------------------------------------------- rule per_seed_v1 (owner 2026-11-03 21:45)

RULE_PS1 = ('per_seed_v1: a cooperative plan need not generalise across traffic seeds. Every (plan, seed) '
            'verification is its own example: admit it iff THAT seed\'s per-unit paired block bootstrap '
            f'(policy-continues minus plan, B={BOOT_B}, RNG {BOOT_RNG_SEED} re-seeded per example, circular '
            f'{BLOCK_S} s blocks or per-vehicle units) has one-sided 95% lower bound > 0 AND that seed\'s '
            'composite safety check passes (seed 1: extended-window deltas; seed 2: replication-window '
            'deltas) AND, on seed 1, the existing later-harm (extended-window) check passes. Seed 2 has no '
            'extended window: its later-harm is NOT MEASURED and disclosed per example (never imputed). '
            'Imitation pairs of an example come from THAT seed\'s plan-branch tape. Cooperative label '
            'per example: that seed\'s best-solo minus group per-unit bootstrap lower bound > 0 when '
            'that seed\'s solos exist, else no_solos (admitted, not cooperative). replicated = '
            'cooperative on both seeds (reporting only; weights unchanged).')
SEED_DIRS = {1: ('first-seed decision', 'confirm_solo', 'solo'),
             2: ('confirm_replication', 'confirm_solo_replication', 'replication_solo')}


def _seed_safety(c, seed_index):
    """(ok, source): the composite MAIN04:26 rule on THIS seed's recorded deltas."""
    deltas = (c.get('safety_deltas') or {}).get('extended' if seed_index == 1 else 'replication')
    if deltas is not None:
        from .bank_safety import verdict
        return bool(verdict(deltas)['new_combined_rule']), ('extended_window_seed1' if seed_index == 1
                                                          else 'replication_window_seed2')
    return c.get('checks', {}).get('safety') is True, 'plan_level_check (per-seed deltas absent)'


def _seed_cooperation(bank, c, seed_index, plan_tape, window, admitted, group_size):
    """Within-seed cooperation label of one example (never pooled across seeds)."""
    co = c.get('cooperation') or {}
    if group_size < 2:
        return dict(cooperative=False, cooperative_status='single_cav',
                    statistical_reason='single-CAV plan: cooperative=False by definition')
    _, solo_dir, key = SEED_DIRS[seed_index]
    solos = co.get(key) or []
    if seed_index == 1 and co.get('complete') is False:
        return dict(cooperative=False, cooperative_status='incomplete',
                    statistical_reason='first-seed solo branches incomplete; never imputed')
    if not solos:
        return dict(cooperative=False, cooperative_status='no_solos',
                    statistical_reason=f'no seed-{seed_index} solos recorded; admitted, not counted as cooperative')
    member, adv, solo_tape, sw = _best_solo(bank/solo_dir, solos)
    common = paired_window(window, sw)
    st = _stats([(solo_tape, plan_tape, common)])
    st.update(best_solo_member=member, best_solo_advantage_veh_s=adv,
              paired_window=dict(group_window=window, solo_window=sw, used=common))
    positive = st['lower95_one_sided_veh_s'] > 0
    reason = (f'seed-{seed_index} synergy lower95 > 0' if positive else f'seed-{seed_index} synergy lower95 <= 0')
    if positive and not admitted:
        reason += ' (example not admitted: not counted)'
    return dict(cooperative=bool(admitted and positive),
                cooperative_status='cooperative' if positive else 'not_cooperative',
                statistical_reason=reason, synergy=st)


def _seed_example(bank, c, seed_index, receipt, branches_dir, plan_index, advantage, group_size):
    """One (plan, seed) example; raises _Missing on absent per-unit data."""
    pol = _read_result(branches_dir, receipt['verification_id'], receipt['branches'][0])
    branch = receipt['branches'][plan_index]
    plan = _read_result(branches_dir, receipt['verification_id'], branch)
    window = receipt['window']
    st = _stats([(pol, plan, window)])
    checks = c.get('checks') or {}
    later = checks.get('no_later_harm') is True if seed_index == 1 else None
    safe, safety_source = _seed_safety(c, seed_index)
    lower = st['lower95_one_sided_veh_s']
    if later is False:
        ok, reason = False, 'seed-1 later-harm check failed (extended-window advantage <= 0)'
    elif not safe:
        ok, reason = False, f'seed-{seed_index} composite safety check failed ({safety_source})'
    elif not lower > 0:
        ok, reason = False, f'seed-{seed_index} one-sided 95%% lower bound %.4g <= 0' % lower
    else:
        ok = True
        reason = (f'seed-{seed_index} lower95 > 0, safety pass'
                  + (', later-harm pass' if later else ', later-harm NOT MEASURED on seed 2'))
    ex = dict(seed_index=seed_index, example_id=f'{c["verification_id"]}#s{seed_index}', admitted=ok,
              status='admitted' if ok else 'rejected', reason=reason, admission=st,
              lower95_one_sided_veh_s=lower, total_veh_s=st['pooled_total_veh_s'],
              mean_per_unit_veh_s=st['per_seed'][0]['mean_per_unit_veh_s'], n_units=st['per_seed'][0]['n_units'],
              advantage_veh_s=advantage, later_harm=('pass' if later else 'fail') if later is not None else 'not_measured',
              safety_ok=safe, safety_source=safety_source,
              tape=dict(branches_dir=str(branches_dir), verification_id=receipt['verification_id'],
                        branch=dict(branch), window=dict(window),
                        intent_entry=(f'{receipt["verification_id"]}/{branch["branch"]}' if seed_index == 1
                                      else f'{receipt["verification_id"]}/seed2/{branch["branch"]}')))
    ex.update(_seed_cooperation(bank, c, seed_index, plan, window, ok, group_size))
    return ex


def view_per_seed(c, first_receipt, bank):
    """One confirmation under per_seed_v1 -> (view, n_examples_attempted, n_examples_missing)."""
    out = copy.deepcopy(c)
    out.update(legacy_admitted=c.get('admitted', False), bank_rule=PER_SEED, statistical_rule=RULE_PS1)
    co = c.get('cooperation')
    members = ((first_receipt or {}).get('chosen_plan') or {}).get('members')
    group_size = (co or {}).get('group_size') if (co or {}).get('group_size') is not None else len(members or [])
    if 'confirmation_branches_complete' in (c.get('checks') or {}):
        examples, attempted = [], 0
        reason = 'confirmation branches incomplete; never imputed'
    else:
        examples, attempted = [], 2
        specs = []
        if first_receipt is None:
            specs.append((1, _Missing('first-seed receipt not found in the bank or its sources')))
        else:
            specs.append((1, (first_receipt, Path(first_receipt['_branches']), first_receipt['chosen']+1,
                              c.get('first_seed_advantage_veh_s', first_receipt.get('grade_veh_s')))))
        try:
            rep_root = bank/'confirm_replication'
            rep = _receipt(rep_root, c['replication'])
            if rep['status'] not in ('accepted', 'rejected'):
                raise _Missing('replication receipt ungraded')
            specs.append((2, (rep, rep_root/'branches', 1, c.get('replication_advantage_veh_s'))))
        except _Missing as e:
            specs.append((2, e))
        for k, spec in specs:
            try:
                if isinstance(spec, _Missing):
                    raise spec
                examples.append(_seed_example(bank, c, k, *spec, group_size=group_size))
            except _Missing as e:
                examples.append(dict(seed_index=k, example_id=f'{c["verification_id"]}#s{k}', admitted=False,
                                     status='data_missing', reason='per-unit data missing: '+str(e),
                                     cooperative=False, cooperative_status='data_missing'))
        reason = '; '.join(f's{e["seed_index"]}: {e["reason"]}' for e in examples)
    missing = sum(e['status'] == 'data_missing' for e in examples)
    admitted_seeds = [e['seed_index'] for e in examples if e['admitted']]
    coop_seeds = [e['seed_index'] for e in examples if e['cooperative'] is True]
    out['admitted'] = bool(admitted_seeds)
    out['per_seed_v1'] = dict(examples=examples, admitted_seeds=admitted_seeds, cooperative_seeds=coop_seeds,
                              replicated=len(coop_seeds) == 2, reason=reason)
    if co is not None or group_size >= 2:
        v = copy.deepcopy(co or {})
        v.update(legacy_cooperative=v.get('cooperative'), cooperative=bool(coop_seeds),
                 cooperative_seeds=coop_seeds, replicated=len(coop_seeds) == 2, needs_second_seed_solos=False,
                 cooperative_status=('replicated' if len(coop_seeds) == 2 else
                                     f'seed{coop_seeds[0]}_only' if coop_seeds else
                                     ('not_admitted' if not examples else examples[0]['cooperative_status'])),
                 statistical_reason='per_seed_v1 within-seed synergy labels: '+(
                     ', '.join(f's{e["seed_index"]}={e["cooperative_status"]}' for e in examples) or 'none'))
        out['cooperation'] = v
    return out, (attempted if attempted else 0), missing


def confirmations_per_seed(original, rb, root, arm, sources=()):
    """{vid: per_seed_v1 view}. A record with every example missing is data_missing; raises
    BankDataMissing only when NO complete confirmation has any evaluable example (as boot_v2)."""
    conf = original(root, arm)
    check_keys(conf.values(), bank_rule='boot_v2')
    bank = Path(rb.bank_root(root, arm))
    from .solo_seed2_links import merge
    conf = merge(bank, conf)

    def key(vid):
        return (PER_SEED, str(root), arm, vid, conf[vid].get('confirmation_sha256'))
    need = [vid for vid in conf if key(vid) not in _CACHE]
    if need:
        receipts = {r['verification_id']: r for r in rb.bank_decisions(root, arm, sources=sources)}
        fresh, attempted, all_missing, first = {}, 0, 0, {}
        for vid in need:
            v, n, miss = view_per_seed(conf[vid], receipts.get(vid), bank)
            fresh[vid] = v
            if n:
                attempted += 1
                if miss >= n:
                    all_missing += 1
                    if len(first) < 3:
                        first[vid] = v['per_seed_v1']['reason']
        if attempted and all_missing >= attempted:
            raise BankDataMissing(
                f'per_seed_v1: per-unit outcome data absent for every example of all {attempted} complete '
                f'confirmations of arm {arm}; required: first-seed receipt + policy/chosen tapes and/or '
                'replication receipt + policy/cand1 tapes with per-second team_delay_vs fingerprints. First '
                'missing: '+json.dumps(first))
        for vid, v in fresh.items():
            _CACHE[key(vid)] = v
    return {vid: _CACHE[key(vid)] for vid in conf}


def _install_per_seed(rb, sources=None):
    if getattr(rb.confirmations, 'bank_rule', None) == PER_SEED:
        return
    if getattr(rb.confirmations, 'statistical_1727', False):
        raise ValueError('t1727 adapter already installed in this process; refusing to stack rules')
    original = rb.confirmations
    srcs = {k: tuple(v) for k, v in (sources or {}).items()}

    def confirmations(root, arm):
        return confirmations_per_seed(original, rb, root, arm, srcs.get(arm, ()))
    confirmations.bank_rule = PER_SEED
    confirmations.original = original
    rb.confirmations = confirmations


def report_per_seed(rb, root, arm, sources=()):
    """Per-example table + counts (admitted per arm / seed index / policy version, cooperative)."""
    base = getattr(rb.confirmations, 'original', rb.confirmations)
    conf = confirmations_per_seed(base, rb, root, arm, tuple(sources))
    rows = []
    for vid, c in sorted(conf.items()):
        ps = c['per_seed_v1']
        for e in ps['examples']:
            syn = e.get('synergy') or {}
            rows.append(dict(verification_id=vid, example_id=e['example_id'], seed_index=e['seed_index'],
                             policy_version=c.get('policy_version'), admitted=e['admitted'], status=e['status'],
                             reason=e['reason'], unit_source=(e.get('admission') or {}).get('unit_source'),
                             lower95_veh_s=e.get('lower95_one_sided_veh_s'), total_veh_s=e.get('total_veh_s'),
                             mean_per_unit_veh_s=e.get('mean_per_unit_veh_s'), n_units=e.get('n_units'),
                             advantage_veh_s=e.get('advantage_veh_s'), later_harm=e.get('later_harm'),
                             safety_ok=e.get('safety_ok'), safety_source=e.get('safety_source'),
                             cooperative=e['cooperative'], cooperative_status=e['cooperative_status'],
                             synergy_lower95_veh_s=syn.get('lower95_one_sided_veh_s'),
                             synergy_total_veh_s=syn.get('pooled_total_veh_s'),
                             best_solo_member=syn.get('best_solo_member'), plan_replicated=ps['replicated']))
    adm = [r for r in rows if r['admitted']]
    versions = sorted({str(r['policy_version']) for r in rows})
    per_plan = [len(c['per_seed_v1']['admitted_seeds']) for c in conf.values()]
    return dict(arm=arm, bank_rule=PER_SEED, rule=RULE_PS1, B=BOOT_B, rng_seed=BOOT_RNG_SEED, block_s=BLOCK_S,
                confirmations=len(conf), examples=len(rows), examples_admitted=len(adm),
                legacy_admitted_plans=sum(bool(c.get('legacy_admitted')) for c in conf.values()),
                plans_contributing={str(n): per_plan.count(n) for n in (0, 1, 2)},
                admitted_by_seed={str(k): sum(r['seed_index'] == k for r in adm) for k in (1, 2)},
                cooperative_examples=sum(r['cooperative'] is True for r in rows),
                cooperative_by_seed={str(k): sum(r['cooperative'] is True and r['seed_index'] == k for r in rows)
                                     for k in (1, 2)},
                replicated_cooperative_plans=sum(c['per_seed_v1']['replicated'] for c in conf.values()),
                no_solos_admitted=sum(r['cooperative_status'] == 'no_solos' for r in adm),
                admitted_later_harm_not_measured=sum(r['later_harm'] == 'not_measured' for r in adm),
                data_missing_examples=sum(r['status'] == 'data_missing' for r in rows),
                by_policy_version={v: dict(admitted=sum(str(r['policy_version']) == v for r in adm),
                                           cooperative=sum(str(r['policy_version']) == v and r['cooperative'] is True
                                                           for r in rows)) for v in versions},
                entries=rows)


def install(rb, bank_rule='boot_v2', sources=None):
    """Process-local V2V reader adapter; never changes legacy training or files.

    bank_rule='boot_v2' (default, rule v2), 't1727' (MAIN17:27 two-seed Student-t, unchanged) or
    'per_seed_v1' (owner 2026-11-03 21:45: each (plan, seed) verification is its own example).
    sources: {arm: [LS-collect roots]} (the V2V cfg['replay_sources']) so first-seed receipts of
    live-labelled entries are found. Without install() the legacy readers are unchanged."""
    if bank_rule not in BANK_RULES:
        raise ValueError(f'unknown bank_rule {bank_rule!r}')
    have = getattr(rb.confirmations, 'bank_rule', None)
    if have is not None and have != bank_rule:
        raise ValueError(f'bank rule {have!r} already installed in this process; refusing to stack {bank_rule!r}')
    if bank_rule == 't1727':
        return _install_t1727(rb)
    if bank_rule == PER_SEED:
        return _install_per_seed(rb, sources)
    if getattr(rb.confirmations, 'bank_rule', None) == 'boot_v2':
        return
    if getattr(rb.confirmations, 'statistical_1727', False):
        raise ValueError('t1727 adapter already installed in this process; refusing to stack rules')
    original = rb.confirmations
    srcs = {k: tuple(v) for k, v in (sources or {}).items()}
    def confirmations(root, arm):
        return confirmations_v2(original, rb, root, arm, srcs.get(arm, ()))
    confirmations.bank_rule = 'boot_v2'
    confirmations.original = original
    rb.confirmations = confirmations


def report(rb, root, arm, sources=()):
    """Per-entry table (bounds, per-seed means, n units, decision reason) under rule v2."""
    base = getattr(rb.confirmations, 'original', rb.confirmations)
    conf = confirmations_v2(base, rb, root, arm, tuple(sources))
    rows = []
    for vid, c in sorted(conf.items()):
        v2 = c.get('bank_rule_v2') or {}
        a = v2.get('admission') or {}
        co = c.get('cooperation') or {}
        s = co.get('synergy_v2') or {}
        rows.append(dict(verification_id=vid, legacy_admitted=c.get('legacy_admitted'), admitted=c['admitted'],
                         reason=v2.get('reason'), unit_source=a.get('unit_source'),
                         lower95_veh_s=a.get('lower95_one_sided_veh_s'), pooled_total_veh_s=a.get('pooled_total_veh_s'),
                         per_seed_means_veh_s=v2.get('per_seed_means_veh_s'),
                         per_seed_totals_veh_s=v2.get('per_seed_totals_veh_s'), n_units=v2.get('n_units'),
                         group_size=co.get('group_size'), cooperative=co.get('cooperative'),
                         cooperative_status=co.get('cooperative_status'),
                         synergy_lower95_veh_s=s.get('lower95_one_sided_veh_s'),
                         synergy_total_veh_s=s.get('pooled_total_veh_s'),
                         synergy_per_seed_means_veh_s=[x['mean_per_unit_veh_s'] for x in s.get('per_seed', [])],
                         needs_second_seed_solos=co.get('needs_second_seed_solos', False)))
    return dict(arm=arm, rule=RULE_V2, B=BOOT_B, rng_seed=BOOT_RNG_SEED, block_s=BLOCK_S,
                confirmations=len(rows), admitted=sum(r['admitted'] for r in rows),
                legacy_admitted=sum(bool(r['legacy_admitted']) for r in rows),
                cooperative_replicated=sum(r['cooperative'] is True for r in rows),
                single_seed=sum(r['cooperative_status'] == 'single_seed' for r in rows),
                needs_second_seed_solos=[r['verification_id'] for r in rows if r['needs_second_seed_solos']],
                entries=rows)


def main(argv=None):
    """Read-only preview (no SUMO, no writes except --out):
    python -m tools.coopmwm.bank_statistics preview --root R --arm B --source S --out BANK_BOOT_PREVIEW.json
    [--rule boot_v2|per_seed_v1]  (default boot_v2, unchanged)"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('preview',))
    ap.add_argument('--root', required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--source', action='append', default=[])
    ap.add_argument('--out', required=True)
    ap.add_argument('--rule', default='boot_v2', choices=('boot_v2', PER_SEED))
    a = ap.parse_args(argv)
    from . import replay_bank as rb
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'{out} exists; never overwritten')
    value = (report_per_seed if a.rule == PER_SEED else report)(rb, a.root, a.arm, a.source)
    out.write_text(json.dumps(value, indent=1))
    print(json.dumps({k: v for k, v in value.items() if k != 'entries'}, indent=1))


if __name__ == '__main__':
    main()
