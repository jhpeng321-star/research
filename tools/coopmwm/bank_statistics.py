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

BANK_RULES = ('boot_v2', 't1727')
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


def install(rb, bank_rule='boot_v2', sources=None):
    """Process-local V2V reader adapter; never changes legacy training or files.

    bank_rule='boot_v2' (default, rule v2) or 't1727' (MAIN17:27 two-seed Student-t, unchanged).
    sources: {arm: [LS-collect roots]} (the V2V cfg['replay_sources']) so first-seed receipts of
    live-labelled entries are found. Without install() the legacy readers are unchanged."""
    if bank_rule not in BANK_RULES:
        raise ValueError(f'unknown bank_rule {bank_rule!r}')
    if bank_rule == 't1727':
        return _install_t1727(rb)
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
    python -m tools.coopmwm.bank_statistics preview --root R --arm B --source S --out BANK_BOOT_PREVIEW.json"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('preview',))
    ap.add_argument('--root', required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--source', action='append', default=[])
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    from . import replay_bank as rb
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'{out} exists; never overwritten')
    value = report(rb, a.root, a.arm, a.source)
    out.write_text(json.dumps(value, indent=1))
    print(json.dumps({k: v for k, v in value.items() if k != 'entries'}, indent=1))


if __name__ == '__main__':
    main()
