"""E2 stage 1: the two matched controls of the original four-plan bank (zero SUMO).

From the ORIGINAL bank F (A42_BIG_BANK_D2800_p60.json, sha 7e7dc76b, four plans):

  S  Onset_Synchronized_Copy          every plan copied with ONLY its non-zero onsets set to 0; hold durations, speeds,
                                      decel limits, lanes, ranks, trigger, zone, spread, cooldown, slots, priority
                                      (verified lb95) unchanged.                       -> E2_S_BANK.json
  Hv Resource_Matched_Uniform_Hold    for v in {18, 19, 20, 21, 22}: same members (lanes / ranks), trigger, zone,
                                      spread, cooldown, priority, per-vehicle hold durations and decel limits; all
                                      onsets 0; ONE common target speed v for every role of every plan.
                                                                                       -> E2_H<v>_BANK.json

A copy is a new descriptor, so its 16-hex strategy key and its family name change (S_<family>, H<v>_<family>);
nothing else may. `diff_report` is the checker: F vs S differ only in ids / names / onsets; F vs Hv have identical
per-plan vehicle counts and sums of hold durations (and differ only in ids / names / onsets / target speeds).

Member selection (oracle_plan.select_members) reads lane, rank, zone, spread and trigger only, so F, S and every Hv
choose the same group in the same traffic state; they are feasible at the same times until their histories diverge.

usage: e2_build.py --bank <A42_BIG_BANK_D2800_p60.json> --out OUT [--n-worlds N] [--code-root TREE] [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cvlib as L  # noqa: E402
import stats as S  # noqa: E402

V_COMMON = (18, 19, 20, 21, 22)
ROLE_FIELDS = ('lane', 'rank', 'v_w', 'dmax', 'hold_s', 'onset_s', 'lc_target')
PLAN_FIELDS = ('trigger', 'zone', 'spread_m', 'cooldown_s', 'slots', 'gate', 'start_gate', 'phases')


def sync_copy(plan):
    """S: only the non-zero onsets become 0."""
    s = copy.deepcopy(plan['strategy'])
    for r in s['roles']:
        if r['onset_s'] != 0:
            r['onset_s'] = 0
    s['name'] = f'S_{plan["family"]}'
    return dict(family=s['name'], strategy=s, lb95_veh_s=plan['lb95_veh_s'], source_family=plan['family'])


def uniform_hold(plan, v):
    """H: all onsets 0 and one common target speed; everything else (incl. per-vehicle durations, decel) kept."""
    s = copy.deepcopy(plan['strategy'])
    for r in s['roles']:
        r['onset_s'] = 0
        r['v_w'] = float(v)
    s['name'] = f'H{int(v)}_{plan["family"]}'
    return dict(family=s['name'], strategy=s, lb95_veh_s=plan['lb95_veh_s'], source_family=plan['family'])


def diff_fields(a, b):
    """Sorted list of the descriptor fields in which strategies ``a`` and ``b`` differ (names / keys excluded):
    plan-level field names and 'roles[i].<field>'; 'roles.count' when the numbers of roles differ."""
    out = [k for k in PLAN_FIELDS if a.get(k) != b.get(k)]
    if len(a['roles']) != len(b['roles']):
        return sorted(out+['roles.count'])
    for i, (x, y) in enumerate(zip(a['roles'], b['roles'])):
        out += [f'roles[{i}].{k}' for k in ROLE_FIELDS if x.get(k) != y.get(k)]
    return sorted(out)


def _kinds(fields):
    return sorted({f.split('.')[-1] for f in fields})


def check_pair(f, c, kind):
    """One original plan ``f`` against its copy ``c`` (kind 'S' or 'H'). Raises Refused on any violated invariant;
    -> the diff row."""
    fs, cs = f['strategy'], c['strategy']
    fields = diff_fields(fs, cs)
    kinds = _kinds(fields)
    rf, rc = S.planned_resource(fs), S.planned_resource(cs)
    allowed = {'onset_s'} if kind == 'S' else {'onset_s', 'v_w'}
    if not set(kinds) <= allowed:
        raise L.Refused(f'{c["family"]}: differs from {f["family"]} in {kinds} (allowed: {sorted(allowed)})')
    if rf != rc:
        raise L.Refused(f'{c["family"]}: vehicle count / hold-duration sum differs from {f["family"]}')
    if c['lb95_veh_s'] != f['lb95_veh_s']:
        raise L.Refused(f'{c["family"]}: priority differs')
    if any(r['onset_s'] != 0 for r in cs['roles']):
        raise L.Refused(f'{c["family"]}: a non-zero onset is left')
    if kind == 'S':
        changed = [i for i, (x, y) in enumerate(zip(fs['roles'], cs['roles'])) if x['onset_s'] != y['onset_s']]
        if changed != [i for i, r in enumerate(fs['roles']) if r['onset_s'] != 0]:
            raise L.Refused(f'{c["family"]}: S must change exactly the non-zero onsets')
    elif len({r['v_w'] for r in cs['roles']}) != 1:
        raise L.Refused(f'{c["family"]}: target speed is not common')
    return dict(original=f['family'], copy=c['family'], kind=kind, differing_fields=fields, differing_kinds=kinds,
                identical_to_original=not fields, key_original=L.strategy_key(fs), key_copy=L.strategy_key(cs),
                vehicles=rc['vehicles'], hold_s_sum=rc['hold_s_sum'])


def onset_report(plans):
    rows = [dict(family=p['family'], lb95_veh_s=p['lb95_veh_s'], n_roles=len(p['strategy']['roles']),
                 cooldown_s=p['strategy']['cooldown_s'], onsets=[r['onset_s'] for r in p['strategy']['roles']],
                 holds=[r['hold_s'] for r in p['strategy']['roles']], v_w=[r['v_w'] for r in p['strategy']['roles']],
                 dmax=[r['dmax'] for r in p['strategy']['roles']],
                 n_nonzero_onsets=sum(1 for r in p['strategy']['roles'] if r['onset_s'] != 0)) for p in plans]
    return dict(plans=rows, any_nonzero_onset=any(r['n_nonzero_onsets'] for r in rows),
                plans_with_nonzero_onset=sum(1 for r in rows if r['n_nonzero_onsets']),
                note='S differs from F exactly for the plans with a non-zero onset; if none had one, S == F')


def build(bank, cell=L.CELL):
    plans = L.bank_strategies(bank, cell)              # executor priority order; keys re-derived and checked
    banks, diffs = {}, []
    s_plans = [sync_copy(p) for p in plans]
    diffs += [check_pair(f, c, 'S') for f, c in zip(plans, s_plans)]
    banks['E2_S_BANK.json'] = dict(arm='S', label='Onset_Synchronized_Copy', plans=s_plans)
    for v in V_COMMON:
        h = [uniform_hold(p, v) for p in plans]
        diffs += [check_pair(f, c, 'H') for f, c in zip(plans, h)]
        banks[f'E2_H{v}_BANK.json'] = dict(arm=f'H{v}', label=f'Resource_Matched_Uniform_Hold v={v} m/s', plans=h)
    for name, b in banks.items():
        order = [p['source_family'] for p in sorted(
            b['plans'], key=lambda p: (-p['lb95_veh_s'], L.strategy_key(p['strategy']), p['family']))]
        if order != [p['family'] for p in plans]:
            raise L.Refused(f'{name}: executor priority order differs from the original bank')
        b['doc'] = L.bank_doc(b['plans'], cell)
    return plans, banks, diffs


TUNE_BLOCK = 'dev'      # the Development set: first worlds of the already-viewed a35_D2800_p60 panel


def tune_waves(banks, n_worlds):
    """Manual tuning of the uniform-hold speed on the Development worlds: U0 + H18..H22 (runner arm CELL_BANK,
    start-whenever-feasible executor, unchanged). F and S are NOT run here; e2_select picks one speed by the
    registered rule and writes the confirmation waves."""
    waves = [dict(wave='DEV_U0', kind='U0', block=TUNE_BLOCK, n_worlds=n_worlds)]
    for name, b in banks.items():
        if b['arm'].startswith('H'):
            waves.append(dict(wave=f'E2T_{b["arm"]}', kind='BANK', block=TUNE_BLOCK, n_worlds=n_worlds, bank=name,
                              bank_sha256=L.sha_bytes(L.text_of(b['doc']).encode()), arm=b['arm']))
    return waves


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--bank', required=True, help='the original four-plan bank (sha 7e7dc76b checked)')
    p.add_argument('--out', default=str(L.HERE/'out'))
    p.add_argument('--tune-worlds', type=int, help='Development worlds for the H speed tuning (registered: 24)')
    p.add_argument('--code-root')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    sha = L.sha_file(a.bank)
    if sha != L.ORIGINAL_BANK_SHA256:
        raise L.Refused(f'{a.bank}: sha {sha[:8]} is not the registered original bank {L.ORIGINAL_BANK_SHA256[:8]}')
    plans, banks, diffs = build(L.read_json(a.bank))
    cc = None
    if a.code_root:
        cc = L.crosscheck(a.code_root, [p['strategy'] for b in banks.values() for p in b['plans']])['status']
    files = {}
    for name, b in banks.items():
        text = L.text_of(b['doc'])
        state = 'dry-run' if a.dry_run else L.write_once(Path(a.out)/'e2'/'banks'/name, text)
        files[name] = dict(arm=b['arm'], label=b['label'], sha256=L.sha_bytes(text.encode()), file=state,
                           families=[p['family'] for p in b['plans']])
    doc = dict(schema='claim_validation.e2.manifest.v1', run_id=L.RUN_ID, cell=L.CELL,
               original_bank=dict(name=Path(a.bank).name, sha256=sha, families=[p['family'] for p in plans]),
               onsets=onset_report(plans), v_common=list(V_COMMON), banks=files, diffs=diffs,
               invariants=('F vs S: only ids / names / non-zero onsets; F vs H: same per-plan vehicle count and '
                           'hold-duration sum, only ids / names / onsets / target speed; same executor priority order'),
               reference='U0 = frozen speed-only base MAPPO d9042cc5 (not MAPPO60C)')
    if a.tune_worlds:
        w = tune_waves(banks, a.tune_worlds)
        doc.update(tune_waves=w, tune_worlds=a.tune_worlds, episodes=len(w)*a.tune_worlds,
                   sim_s=len(w)*a.tune_worlds*L.HORIZON_S)
    doc = L.seal(doc)
    state = 'dry-run' if a.dry_run else L.write_once(Path(a.out)/'e2'/'E2_MANIFEST.json', L.text_of(doc))
    print(L.text_of(dict(stage='e2_build', dry_run=a.dry_run, manifest=state, crosscheck=cc, onsets=doc['onsets'],
                         banks={k: v['sha256'][:12] for k, v in files.items()},
                         s_identical_to_f=[d['copy'] for d in diffs if d['kind'] == 'S' and d['identical_to_original']],
                         h_identical_to_f=[d['copy'] for d in diffs if d['kind'] == 'H' and d['identical_to_original']],
                         sim_s=doc.get('sim_s'), doc_sha256=doc['doc_sha256'])), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
