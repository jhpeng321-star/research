"""Outcome-blind final world-job plan with pinned baseline and paired v5 hosts."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from tools.rb2int.audit import ROOT, load, sha

MODELS = ('WM2', 'FULL-v4', 'FULL-v5', 'FULL-v5-LP')


def build_plan(registration):
    worlds = registration['final_comparison_worlds']
    primary = set(registration['replicated_primary_worlds'])
    cells = {s: c for c in registration['cells'] for s in c['seeds']}
    # Administrative priority only: complete replication-critical primary cells
    # before large strata and descriptive singletons. No forecast/target is read.
    ordered = sorted(worlds, key=lambda s: (s not in primary, cells[s]['n'], s))
    paired_host = {s: ('gpuA' if i % 7 < 4 else 'gpuR') for i, s in enumerate(ordered)}
    jobs = []
    for seed in ordered:
        for model in MODELS:
            baseline = model in ('WM2', 'FULL-v4')
            if baseline and seed in registration['reserved8']:
                continue  # reuse the already completed original K8 task set.
            host = 'gpuR' if model == 'WM2' else 'gpuA' if model == 'FULL-v4' else paired_host[seed]
            jobs.append(dict(id=model.replace('FULL-', '').replace('WM2', 'wm2').lower()+'_'+str(seed),
                model=model, seed=seed, host=host, futures=list(range(8)),
                primary=seed in primary, stratum_worlds=cells[seed]['n']))
    if len(jobs) != 216 or sum(len(j['futures']) for j in jobs) != 1728:
        raise ValueError('fixed58-world/old-reserved-reuse task census changed')
    return dict(schema='rb2int-final-world-plan/1', models=list(MODELS), jobs=jobs,
        final_worlds=worlds, primary_worlds=registration['replicated_primary_worlds'],
        K=8, arms=registration['arms'], horizons_s=registration['horizons_s'],
        new_world_jobs=216, new_future_tasks=1728, reused_baseline_future_tasks=128,
        slots=dict(gpuA=4, gpuR=3), memory_gib_per_worker=5,
        paired_v5_world_host=paired_host, ordering='primary replicated cells by n then seed; descriptive singletons last; fixed model order',
        scheduling_uses_outcomes=False, baseline_hosts=dict(WM2='gpuR', **{'FULL-v4':'gpuA'}),
        v5_lp_same_host_for_every_world=True,
        inference_only=True, new_sumo_runs=0, automatic_scientific_retry=False,
        dispatch_cutoff='2026-11-02T09:00:00+08:00',
        cutoff_policy='no new world job after cutoff; running own world jobs finish; all attrition retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    registration = ROOT/'docs/rb2int/EVALUATION_REGISTRATION.json'
    plan = build_plan(load(registration))
    plan.update(registered_at=datetime.now().astimezone().isoformat(),
                registration_sha256=sha(registration), source_sha256=sha(__file__))
    out = Path(args.out)
    if not out.parent.is_dir() or out.exists():
        raise ValueError('exclusive final plan receipt required')
    with out.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=1)
    print(json.dumps(dict(receipt=str(out), sha256=sha(out), new_future_tasks=1728,
        jobs_by_host={h:sum(j['host']==h for j in plan['jobs']) for h in ('gpuA','gpuR')},
        slots=plan['slots'], v5_lp_same_host_for_every_world=True), indent=1))


if __name__ == '__main__':
    main()
