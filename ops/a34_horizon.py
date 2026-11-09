"""AMENDMENT 34 (evaluation-horizon robustness; MAIN 2026-11-10, registered before any amendment-33/34/35 outcome):
U0, LEARNED_TRIGGER (8e0ddd3e) and DUO_HOLD (duo12_v20) on the sealed p20c panel 993100-993291, each episode the
sealed amendment-28/32 episode up to t = 1,200 s, then no new arrivals until the network is empty or t = 2,400 s
(tools/coopmwm/a34_horizon.py). Phase horizon_a34, cap 1,400,000 sim-s, own ledger in the A34 final root.

Thin driver over ops/nm_finaleval.py (setting p20c) and ops/a32_ablation.py: the specs are the registered sealed
specs (U0 / LEARNED_TRIGGER: nm_finaleval.episode_specs on the bank-root bindings; DUO_HOLD: a32_specs) with only
job_id ('a34h_' prefix), sim_s (2,400 precharge) and the routing field horizon_a34 added; checkpoint / world /
window / panel are the sealed identity. The final root on the run host mirrors the binding files (BOUND_U0,
ORACLE_BINDING, PRERUN_PLAN, prerun_plant, oracle_banks, a32_banks) with a fresh ledger.

Commands (python -m ops.a34_horizon <cmd> --final-root <root>):
  dry               zero physics: build every spec, check bindings/digests, print counts
  run               --arms U0,LEARNED_TRIGGER,DUO_HOLD [--first N] --workers 24 --chunk 48 (STOP_A34 stops after the
                    chunk in flight). --first 2 = the plant-identity worlds (2 per arm, canary first per arm)
  identity-compare  (laptop) --bank-root <p20c bank copy> --a32-root <a32 root copy>: J and safety at 1,200 s of the
                    first 2 worlds per arm vs the sealed records (|dJ| <= 1e-6, identical safety + [300,1200) log content)
  analyse           (laptop) --bank-root --a32-root --out: the registered amendment-34 contrasts
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import nm_finaleval as N  # noqa: E402
import a32_ablation as A  # noqa: E402

A34_ARMS = ('U0', 'LEARNED_TRIGGER', 'DUO_HOLD')
A34_PHASE = 'horizon_a34'
A34_CAP_SIM_S = 1_400_000
STOP_A34 = 'STOP_A34'
PREFIX = 'a34h_'
AUTHORITY = ('MAIN 2026-11-10 ~04:30 AMENDMENT 34 (owner relayed reviewer request; owner authorized "start working '
             'on it"): evaluation-horizon robustness of U0 / LEARNED_TRIGGER / DUO_HOLD on the sealed p20c panel '
             '993100-993291; phase horizon_a34 cap 1,400,000 sim-s')
CONTRASTS = (('LEARNED_TRIGGER', 'U0'), ('DUO_HOLD', 'U0'), ('LEARNED_TRIGGER', 'DUO_HOLD'))


def wave(arm):
    return f'A34_{arm}'


setup = A.setup


class A34Runtime(N.PrerunRuntime):
    """The pre-run plant of the mirror root, own ledger phase horizon_a34, cap = allocation = 1,400,000."""
    phase = A34_PHASE

    def check_capacity(self, amount):
        rows = N.fe.ledger_rows(self.final_root)
        if sum(r['sim_s'] for r in rows)+amount > A34_CAP_SIM_S:
            raise N.Blocked(f'A34: {A34_PHASE} cap {A34_CAP_SIM_S} would be exceeded')

    def reserve(self, cfg, jobs):
        N.own_reserve(self.final_root, jobs, A34_PHASE, cap=A34_CAP_SIM_S, calls=self.calls, allocation=A34_CAP_SIM_S)


def sealed_specs(arm, panel, root, *, check_files=True):
    """The registered sealed specs of ``arm`` (amendment 28: U0 / LEARNED_TRIGGER; amendment 32: DUO_HOLD)."""
    root = Path(root)
    u0, oracle = A.bindings(root)
    plant = root/N.PRERUN_PLANT
    if arm == 'U0':
        return N.episode_specs('U0', panel, None, plant, 'mappo_frozen', None, u0)
    if arm == 'LEARNED_TRIGGER':
        return N.episode_specs('LEARNED_TRIGGER', panel, None, plant, 'mappo_frozen', oracle, u0)
    if arm == 'DUO_HOLD':
        return A.a32_specs('DUO_HOLD', panel, root, check_files=check_files)
    raise N.Blocked(f'no A34 arm {arm!r}')


def a34_specs(arm, panel, final_root, *, check_files=True):
    from tools.coopmwm import a34_horizon as H
    field = H.horizon_field()
    return [dict(s, job_id=PREFIX+s['job_id'], sim_s=H.END_S, horizon_a34=dict(field))
            for s in sealed_specs(arm, panel, final_root, check_files=check_files)]


def check_a34(record, spec):
    """The sealed arm checks (nm_finaleval.check_episode / a32 pins) + the A34 pins and receipt."""
    if spec.get('horizon_a34') is None:
        return A.check_a32(record, spec)
    from tools.coopmwm import a34_horizon as H
    what = f'A34 {spec["arm"]} {spec["world_id"]}'
    out = A.check_a32(record, spec)
    if record.get('a34_sha256_loaded') != spec['horizon_a34']['a34_sha256']:
        raise N.Blocked(f'{what}: a34_horizon actually loaded differs from its registration')
    a = record.get('a34')
    if not isinstance(a, dict) or set(H.RECEIPT_KEYS) - set(a):
        raise N.Blocked(f'{what}: a34 receipt missing or incomplete')
    if a['end_reason'] not in ('network_empty', 'end_s') or not H.ARRIVALS_END_S <= a['t_end_s'] <= H.END_S:
        raise N.Blocked(f'{what}: a34 end outside [1200, 2400]')
    if a['withdrawn'] > a['future_trips'] or a['actual_sim_s'] > H.END_S+1e-6:
        raise N.Blocked(f'{what}: a34 withdrawal / charge accounting inconsistent')
    if a['end_reason'] == 'network_empty' and (a['never_entered'] or a['on_road_at_end']):
        raise N.Blocked(f'{what}: network_empty with vehicles left')
    if not isinstance(a.get('ext_log'), dict) or a['ext_log'].get('file') != H.EXT_LOG:
        raise N.Blocked(f'{what}: extended vehicle log receipt missing')
    if abs(float(record['J_veh_s'])-float(a['totals_end']['team_delay_vs'])) > 1e-6:
        raise N.Blocked(f'{what}: J_veh_s is not the plant total at the end')
    return dict(out, a34=deepcopy(a), a34_sha256_loaded=record['a34_sha256_loaded'])


N.CHECKS['episode'] = check_a34


def execute(final_root, panel, waves, *, workers, chunk, log, first=None):
    """a32_ablation.execute with the A34 runtime/check; ``first`` = per-arm coverage target of this invocation."""
    final_root = Path(final_root)
    runtime = A34Runtime(N.Runtime(N.RUNNER), final_root)
    plant = final_root/N.PRERUN_PLANT
    cfg = runtime.config(plant)
    if not os.environ.get('COOPMWM_MAP_SPEC'):
        raise N.Blocked('set COOPMWM_MAP_SPEC before physics')
    status = runtime.runner_status()
    if not status['ok'] or not status.get('learned'):
        raise N.Blocked(f'runner cannot run the A34 arms: {status}')
    N.map_check(N.bind_map_api(None), panel)
    plan = N.prerun_plan(final_root)
    u0, _ = A.bindings(final_root)
    if (u0['residual_path'], u0['residual_sha256']) != (plan['u0']['path'], plan['u0']['sha256']):
        raise N.Blocked('the U0 binding residual is not the pre-run u0 copy')

    def target(specs):
        return len(specs) if first is None else min(first, len(specs))

    def coverage():
        return {name: (N.published(final_root, name, specs, check_a34)[2], len(specs)) for name, _, specs in waves}
    rounds = 0
    while True:
        cov = coverage()
        pending = [w for w in waves if cov[w[0]][0] < target(w[2])]
        if not pending:
            break
        if (final_root/STOP_A34).exists():
            log(dict(stopped=True, coverage=cov))
            return dict(status='STOPPED', coverage=cov)
        budget, limits, selection = chunk, {}, []
        for name, kind, specs in pending:
            take = min(target(specs)-cov[name][0], budget)
            selection.append((name, kind, specs))
            limits[name] = take
            budget -= take
            if budget == 0:
                break
        N.run_waves(selection, root=plant, final_root=final_root, cfg=cfg, runtime=runtime, workers=workers,
                    base='mappo_frozen', chunked=True, limit=limits)
        rounds += 1
        log(dict(round=rounds, published={k: v[0] for k, v in coverage().items()},
                 charged_sim_s=sum(r['sim_s'] for r in N.fe.ledger_rows(final_root))))
    cov = coverage()
    return dict(status='PASS', coverage=cov, charged_sim_s=sum(r['sim_s'] for r in N.fe.ledger_rows(final_root)))


# ------------------------------------------------------------------------------------------------ laptop

def sealed_records(panel, bank_root, a32_root):
    """{arm: {world_id: sealed record}}, {arm: root}: U0 / LEARNED_TRIGGER from the amendment-28 bank root copy,
    DUO_HOLD from the amendment-32 root copy (whole waves, receipts verified)."""
    have, roots = {}, {}
    for arm in A34_ARMS:
        root = Path(a32_root if arm == 'DUO_HOLD' else bank_root)
        specs = sealed_specs(arm, panel, root, check_files=False)
        check = A.check_a32 if arm == 'DUO_HOLD' else A._ORIG_CHECK
        recs, _, cov = N.published(root, arm, specs, check)
        if cov != len(specs):
            raise N.Blocked(f'{arm}: {cov}/{len(specs)} sealed records in {root}')
        have[arm], roots[arm] = {r['world_id']: r for r in recs}, root
    return have, roots


def a34_records(panel, root, *, partial=False):
    have = {}
    for arm in A34_ARMS:
        specs = a34_specs(arm, panel, root, check_files=False)
        recs, _, cov = N.published(root, wave(arm), specs, check_a34)
        if cov != len(specs) and not partial:
            raise N.Blocked(f'A34 {arm}: {cov}/{len(specs)} published in {root}')
        have[arm] = {r['world_id']: r for r in recs}
    return have


def _content_sha(path):
    with gzip.open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def identity(panel, a34_root, bank_root, a32_root, k=N.IDENTITY_WORLDS):
    sealed, roots = sealed_records(panel, bank_root, a32_root)
    got = a34_records(panel, a34_root, partial=True)
    worlds = [r['world_id'] for r in panel['worlds']][:k]
    rows = []
    for arm in A34_ARMS:
        for w in worlds:
            if w not in got[arm]:
                rows.append(dict(arm=arm, world_id=w, status='missing'))
                continue
            r, s = got[arm][w], sealed[arm][w]
            mine = Path(a34_root)/'evaluation'/r.get('work_dir', r['job_id'])/N.VEHICLE_LOG
            ref = roots[arm]/'evaluation'/s.get('work_dir', s['job_id'])/N.VEHICLE_LOG
            log_equal = (_content_sha(mine) == _content_sha(ref)) if mine.is_file() and ref.is_file() else None
            same_spec = all(r[x] == s[x] for x in ('arm', 'world', 'checkpoint', 'window', 'panel_sha256'))
            rows.append(dict(arm=arm, world_id=w, J_1200_a34=r['a34']['J_1200_veh_s'], J_sealed=s['J_veh_s'],
                             abs_dJ=abs(r['a34']['J_1200_veh_s']-s['J_veh_s']),
                             safety_equal=r['a34']['safety_1200'] == s['safety'], safety_1200=r['a34']['safety_1200'],
                             safety_sealed=s['safety'], vehicle_log_300_1200_content_equal=log_equal,
                             sealed_identity_fields_equal=same_spec, t_end_s=r['a34']['t_end_s'],
                             end_reason=r['a34']['end_reason'], J_ext=r['J_veh_s']))
    ok = (len(rows) == len(A34_ARMS)*k and all(x.get('abs_dJ', 1) <= 1e-6 and x['safety_equal']
                                              and x['sealed_identity_fields_equal']
                                              and x['vehicle_log_300_1200_content_equal'] is not False for x in rows))
    return dict(schema='coopmwm.a34_horizon.identity.v1', verdict='PASS' if ok else 'FAILED', worlds=rows,
                rule=('registered plant identity: the extended run reproduces the sealed per-world J over [300,1200) '
                      'exactly (|dJ| <= 1e-6 veh-s) on 2 worlds per arm; also identical safety counters at 1,200 s, '
                      'identical sealed identity fields and identical decompressed [300,1200) vehicle-log content'))


def analyse(panel, a34_root, bank_root, a32_root, *, cache=None):
    import numpy as np
    from tools.coopmwm import a34_horizon as H
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    sealed, roots = sealed_records(panel, bank_root, a32_root)
    ext = a34_records(panel, a34_root)
    for arm in A34_ARMS:                                    # pairing = the sealed identity of every episode
        for w in worlds:
            r, s = ext[arm][w], sealed[arm][w]
            if any(r[x] != s[x] for x in ('arm', 'world', 'checkpoint', 'window', 'panel_sha256')):
                raise N.Blocked(f'A34 {arm} {w}: identity differs from the sealed episode')
    # per-vehicle reports: original (sealed [300,1200) logs) and extended (all vehicles over [300, end))
    orig_rep = {a: {w: A.report_of(roots[a], sealed[a][w]) for w in worlds} for a in A34_ARMS}
    cache_v = json.loads(Path(cache).read_text()) if cache and Path(cache).is_file() else {}
    ext_rep = {}
    for a in A34_ARMS:
        ext_rep[a] = {}
        for w in worlds:
            r = ext[a][w]
            key = f'{a}|{w}|{r["vehicle_log"]["sha256"]}|{r["a34"]["ext_log"]["sha256"]}'
            if key not in cache_v:
                d = Path(a34_root)/'evaluation'/r.get('work_dir', r['job_id'])
                cache_v[key] = H.merged_report(d, r, N.vehicle_log_report)
            ext_rep[a][w] = cache_v[key]
    if cache:
        Path(cache).write_text(json.dumps(cache_v))
    bad = [(a, w) for a in A34_ARMS for w in worlds if not ext_rep[a][w]['ok'] or not orig_rep[a][w]['ok']]
    if bad:
        raise N.Blocked(f'invalid vehicle-log report(s): {bad[:5]}')

    def j(have, a, key='J_veh_s'):
        return np.array([have[a][w][key] for w in worlds], float)

    def cv(reps, a, *, drop_never=False):
        out = []
        for w in worlds:
            pv = reps[a][w]['per_vehicle']
            never = set(reps[a][w].get('never_entered_ids') or ()) if drop_never else set()
            out.append(N.cvar90([v[0] for k, v in pv.items() if k not in never]))
        return np.array(out, float)

    def sig(ci):
        return bool(ci[1] < 0 or ci[0] > 0)
    contrasts = []
    for t, ref in CONTRASTS:
        orig_total = N.fe.paired_stats(j(sealed, t), j(sealed, ref), bootstrap)
        ext_total = N.fe.paired_stats(j(ext, t), j(ext, ref), bootstrap)
        j1200 = N.fe.paired_stats(np.array([ext[t][w]['a34']['J_1200_veh_s'] for w in worlds]),
                                  np.array([ext[ref][w]['a34']['J_1200_veh_s'] for w in worlds]), bootstrap)
        got = N._paired_tail(orig_rep, worlds, t, ref, N.cvar90)
        orig_cvar = N.fe.paired_stats(got[0], got[1], bootstrap, pct=False)
        ext_cvar = N.fe.paired_stats(cv(ext_rep, t), cv(ext_rep, ref), bootstrap, pct=False)
        ext_cvar_logged = N.fe.paired_stats(cv(ext_rep, t, drop_never=True), cv(ext_rep, ref, drop_never=True),
                                            bootstrap, pct=False)
        excl = dict(treatment_not_in_reference=0, reference_not_in_treatment=0, paired=0)
        per_world_excl = []
        for w in worlds:
            a_, r_ = set(orig_rep[t][w]['per_vehicle']), set(orig_rep[ref][w]['per_vehicle'])
            excl['treatment_not_in_reference'] += len(a_-r_)
            excl['reference_not_in_treatment'] += len(r_-a_)
            excl['paired'] += len(a_ & r_)
            per_world_excl.append(len(a_-r_)+len(r_-a_))
        excl.update(excluded_total=excl['treatment_not_in_reference']+excl['reference_not_in_treatment'],
                    excluded_per_world_mean=float(np.mean(per_world_excl)), excluded_per_world_max=int(max(per_world_excl)))
        pair = dict(B={w: ext_rep[t][w] for w in worlds}, A={w: ext_rep[ref][w] for w in worlds})
        pair[ref] = pair['A']
        groups = N.co_primary_groups(pair, worlds, bootstrap, reference=ref)
        hb = N.fe.paired_stats(np.array([ext[t][w]['safety']['hard_brakes'] for w in worlds], float),
                               np.array([ext[ref][w]['safety']['hard_brakes'] for w in worlds], float), bootstrap)
        survives_total = (not sig(orig_total['ci95'])) or (
            np.sign(ext_total['diff']) == np.sign(orig_total['diff']) and sig(ext_total['ci95']))
        survives_cvar = (not sig(orig_cvar['ci95'])) or (
            np.sign(ext_cvar['diff']) == np.sign(orig_cvar['diff']) and sig(ext_cvar['ci95']))
        contrasts.append(dict(
            treatment=t, reference=ref, convention='diff = treatment - reference; negative = treatment has less delay',
            original_total=orig_total, original_total_significant=sig(orig_total['ci95']),
            extended_total=ext_total, extended_total_significant=sig(ext_total['ci95']),
            J_1200_from_extended_runs=j1200, total_survives_horizon=bool(survives_total),
            original_cvar90_common_vehicles=orig_cvar, original_cvar90_significant=sig(orig_cvar['ci95']),
            extended_cvar90_all_vehicles=ext_cvar, extended_cvar90_all_vehicles_excluding_never_entered=ext_cvar_logged,
            cvar90_survives_horizon=bool(survives_cvar),
            original_cvar90_common_vehicle_exclusions=excl, extended_groups_descriptive=groups,
            extended_hard_brakes_minus_ref=hb))
    arms = {}
    for a in A34_ARMS:
        rec = [ext[a][w]['a34'] for w in worlds]
        ends = np.array([x['t_end_s'] for x in rec], float)
        arms[a] = dict(
            mean_J_original=float(j(sealed, a).mean()), mean_J_extended=float(j(ext, a).mean()),
            mean_J_1200_extended_runs=float(np.mean([x['J_1200_veh_s'] for x in rec])),
            mean_J_after_1200=float(np.mean([e['J_veh_s']-x['J_1200_veh_s'] for e, x in
                                             zip((ext[a][w] for w in worlds), rec)])),
            end=dict(network_empty=sum(x['end_reason'] == 'network_empty' for x in rec),
                     reached_2400=sum(x['end_reason'] == 'end_s' for x in rec),
                     t_end_mean=float(ends.mean()), t_end_min=float(ends.min()), t_end_max=float(ends.max())),
            never_entered=dict(worlds=sum(1 for x in rec if x['never_entered']),
                               vehicles=sum(len(x['never_entered']) for x in rec),
                               charged_s=float(sum(ext_rep[a][w]['never_entered_charged_s'] for w in worlds))),
            on_road_at_2400=sum(x['on_road_at_end'] for x in rec),
            withdrawn_trips=dict(total=sum(x['withdrawn'] for x in rec), future=sum(x['future_trips'] for x in rec)),
            source_wait_after_1200_vs=float(np.mean([x['totals_end']['source_wait_vs']-x['totals_1200']['source_wait_vs']
                                                     for x in rec])),
            vehicles_extended=int(sum(len(ext_rep[a][w]['per_vehicle']) for w in worlds)),
            vehicles_original=int(sum(len(orig_rep[a][w]['per_vehicle']) for w in worlds)),
            mean_cvar90_extended_all=float(cv(ext_rep, a).mean()),
            safety_extended={k: int(sum(ext[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY},
            ttc_events_after_1200=int(sum(x['ttc_events_ext'] for x in rec)),
            safety_1200={k: int(sum(x['safety_1200'][k] for x in rec)) for k in N.SAFETY},
            actual_sim_s=float(sum(x['actual_sim_s'] for x in rec)))
    identity_all = {a: float(max(abs(ext[a][w]['a34']['J_1200_veh_s']-sealed[a][w]['J_veh_s']) for w in worlds))
                    for a in A34_ARMS}
    interp = dict(
        rule=('pre-registered (amendment 34): the original result survives the horizon iff the sign of each original '
              'significant contrast is unchanged and its CI still excludes 0; otherwise not robust'),
        total_delay={f'{c["treatment"]}-{c["reference"]}': dict(original_significant=c['original_total_significant'],
                                                                survives=c['total_survives_horizon'])
                     for c in contrasts},
        cvar90={f'{c["treatment"]}-{c["reference"]}': dict(original_significant=c['original_cvar90_significant'],
                                                           survives=c['cvar90_survives_horizon'])
                for c in contrasts})
    interp['robust_total_delay'] = all(v['survives'] for v in interp['total_delay'].values())
    interp['robust_cvar90'] = all(v['survives'] for v in interp['cvar90'].values())
    return N.seal(dict(schema='coopmwm.a34_horizon.analysis.v1', amendment='34', authority=AUTHORITY,
                       panel_sha256=panel['panel_sha256'], n_worlds=len(worlds), bootstrap=bootstrap,
                       horizon=dict(arrivals_end_s=H.ARRIVALS_END_S, end_s=H.END_S, J_window='[300, end)'),
                       contrasts=contrasts, arms=arms, interpretation=interp,
                       identity_all_worlds_max_abs_dJ_1200=identity_all,
                       pooling='never pooled; sealed U0 / LEARNED_TRIGGER (amendment 28) and DUO_HOLD (amendment 32) '
                               'records are the original-horizon reference'),
                  'analysis_sha256')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('dry', 'run', 'identity-compare', 'analyse'))
    p.add_argument('--final-root', required=True)
    p.add_argument('--bank-root')
    p.add_argument('--a32-root')
    p.add_argument('--arms', default=','.join(A34_ARMS))
    p.add_argument('--first', type=int)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--chunk', type=int, default=48)
    p.add_argument('--out')
    a = p.parse_args(argv)
    panel = setup()
    root = Path(a.final_root)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T')), sort_keys=True, default=str)
        print(line, flush=True)
        with (root/'A34_LOG.jsonl').open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    arms = tuple(x for x in a.arms.split(',') if x)
    if not set(arms) <= set(A34_ARMS):
        p.error(f'arms are {A34_ARMS}')
    if a.command == 'dry':
        from tools.coopmwm import a34_horizon as H
        specs = {arm: a34_specs(arm, panel, root) for arm in arms}
        out = {arm: len(s) for arm, s in specs.items()}
        out['sim_s_total'] = sum(x['sim_s'] for s in specs.values() for x in s)
        out['a34_sha256'] = H.a34_digest()
        out['first_jobs'] = {arm: [x['job_id'] for x in s[:2]] for arm, s in specs.items()}
        print(json.dumps(out, indent=1))
    elif a.command == 'run':
        import fcntl
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = [(wave(arm), 'episode', a34_specs(arm, panel, root)) for arm in arms]
            log(dict(start='run', waves=[w[0] for w in waves], workers=a.workers, chunk=a.chunk, first=a.first,
                     pid=os.getpid()))
            res = execute(root, panel, waves, workers=a.workers, chunk=a.chunk, log=log, first=a.first)
            log(dict(done='run', **res))
    elif a.command == 'identity-compare':
        print(json.dumps(identity(panel, root, a.bank_root, a.a32_root), indent=1))
    else:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        value = analyse(panel, root, Path(a.bank_root), Path(a.a32_root), cache=out/'ext_reports_cache.json')
        N.publish(out/'A34_RESULTS.json', value, replace=True)
        print(json.dumps(dict(analysis_sha256=value['analysis_sha256'], interpretation=value['interpretation'])))


if __name__ == '__main__':
    main()
