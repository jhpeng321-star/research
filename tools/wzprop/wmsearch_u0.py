"""S-WMSEARCH (strategy-level): SUMO paired-seed verification of WM / structured / random strategies on the MAPPO
u0 base, with the S-REGIMEU0 harness (tools.coopmwm.regimeu0 + imitreplay on c4/imitreplay; same plant, ledger, CRN
and analysis conventions) and strategy_hook.make_hook (+ the eval-only per-vehicle logger, PERVEH.json.gz per job).
Owner 2026-11-07: the reference incumbent is the gentle hold wall_g_solo2; the firm 15 m/s solo2 is banned (optional
labelled comparator arm only); every output carries per-group fairness (origin SUMO lane 0/1/2, HDV / CAV: paired
mean per-vehicle delay change with 95% CIs, the 10 s/veh ALLOWED flag, CVaR90 / p90 / Gini changes).

    python -m tools.wzprop.wmsearch_u0 plan    --root R --regimeu0-root U --arms ARMS.json --cell D3200_p60 --seeds 989770-989785
    python -m tools.wzprop.wmsearch_u0 run     --root R [--workers 16]          (cpuV, nice 19, CUDA_VISIBLE_DEVICES=)
    python -m tools.wzprop.wmsearch_u0 worker  --root R --job ID
    python -m tools.wzprop.wmsearch_u0 analyze --root R [--out A.json]

Arms per seed (paired CRN: same scenario bytes, SUMO seed, noise seed): u0, inc (wall_g_solo2), [solo2 (banned
comparator, --with-banned)], [ref_* (S-MULTICRIT duo12 / gate0, --k-ref)], wm_* (archive a: delay LCB), bnd_*
(b: delay LCB within the 10 s/veh bound), frr_* (c: fairness-improving), str_*, rnd_* (ARMS.json from
`python -m tools.wzprop strategy`). Episode 1,200 s, native warmup 0-300 s, rules act on 300-1,200 s (S-REGIMEU0).
J = team delay over [300, 1200) (regimeu0.control_j).
Convention (as S-REGIMEU0): diff = J(strategy) - J(reference); negative = the strategy is better.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from . import geometry as G
from . import pervehicle as PV
from . import strategy_search as SS

SCHEMA = 'wmc.wmsearch_u0.v2'
STABLE_ID = 'S-WMSEARCH'
PHASE = 'wm_search'
CAP = 280_000
EPISODE_S = 1200
BOOT, BOOT_SEED = 10_000, 20261107
ARM_PREFIX = dict(reference='ref', wm='wm', wm_bounded='bnd', wm_fairer='frr', structured='str', random='rnd',
                  wm_fair='fair')
INC_ARM, BANNED_ARM = 'inc', 'solo2'
REFS = ('u0', INC_ARM, BANNED_ARM)


def arm_specs(arms_doc, cell, k=4, k_fair=0, k_ref=0, k_bounded=None, k_fairer=None, with_banned=False):
    """{arm id: strategy spec or None (u0)} in a fixed order: u0, inc, [solo2], [ref_*], wm_*, bnd_*, frr_*, str_*,
    rnd_*[, fair_*]. A strategy already present as an earlier arm is not repeated (the group's next one is used)."""
    d = arms_doc[cell]
    out = dict(u0=None)
    out[INC_ARM] = SS.validate(dict(d['incumbent']))
    if with_banned:
        out[BANNED_ARM] = SS.validate(dict(d.get('banned_comparator') or SS.SOLO2), allow_banned=True)
    kk = dict(reference=k_ref, wm=k, wm_bounded=k if k_bounded is None else k_bounded,
              wm_fairer=k if k_fairer is None else k_fairer, structured=k, random=k, wm_fair=k_fair)
    for group, prefix in ARM_PREFIX.items():
        have = {SS.strategy_key(x) for x in out.values() if x}
        i = 0
        for s in (d.get(group) or []):
            if i >= kk[group]:
                break
            if SS.strategy_key(s) in have:
                continue
            out[f'{prefix}_{i}'] = SS.validate(dict(s))
            have.add(SS.strategy_key(s))
            i += 1
    return out


def plan_jobs(cell, seeds, specs, world_of):
    """Paired groups (seed-major), one job per (seed, arm), each with its strategy spec."""
    d, p = cell.split('_')
    demand, pen = int(d[1:]), int(p[1:]) / 100.0
    groups = []
    for s in seeds:
        G.check_seed(s)
        w = world_of(int(s), demand, pen)
        groups.append(dict(stage='paired', cell=cell, jobs=[
            dict(id=f'{cell}_s{s}_{arm}', arm=arm, strategy=None if spec is None else spec['name'],
                 strategy_spec=spec, world=w, sim_s=EPISODE_S) for arm, spec in specs.items()]))
    return groups


def boot_ci(diffs, n=BOOT, seed=BOOT_SEED):
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [means[int(.025 * n)], means[int(.975 * n) - 1]]


def analyze_rows(rows, specs, n_boot=BOOT):
    """rows: [{seed, J: {arm: float}, fair: {component: {arm: pervehicle.fairness_of}}}] -> per arm paired J stats vs
    u0 / inc / solo2, per-group fairness vs u0 and vs inc for both per-vehicle delay components ('road' primary) with
    the 10 s/veh ALLOWED flag, and per search-arm summaries ('best' among allowed strategies only)."""
    out = {}
    fair = {comp: {r['seed']: (r.get('fair') or {}).get(comp) or {} for r in rows} for comp in PV.COMPONENTS}
    for arm in specs:
        if arm == 'u0':
            continue
        rec = dict(strategy=(specs[arm] or {}).get('name'))
        for ref in REFS:
            if arm == ref or ref not in specs:
                continue
            diffs = [r['J'][arm] - r['J'][ref] for r in rows if arm in r['J'] and ref in r['J']]
            if diffs:
                rec[f'vs_{ref}'] = dict(n=len(diffs), mean=sum(diffs) / len(diffs), ci95=boot_ci(diffs, n_boot),
                                        better_share=sum(x < 0 for x in diffs) / len(diffs))
        rec['fairness'] = {comp: {ref: PV.paired_fairness(fair[comp], arm, ref, n_boot=n_boot)
                                  for ref in ('u0', INC_ARM) if ref != arm and ref in specs}
                           for comp in PV.COMPONENTS}
        f0 = rec['fairness']['road'].get('u0')
        rec['allowed'] = None if f0 is None else f0['allowed_point']
        rec['allowed_ci_upper'] = None if f0 is None else f0['allowed_ci_upper']
        out[arm] = rec
    summary = {}
    for group, prefix in ARM_PREFIX.items():
        arms = [a for a in out if a.startswith(prefix + '_')]
        allowed = [a for a in arms if out[a].get('allowed')]
        lowers = [a for a in allowed if out[a].get('vs_u0') and out[a]['vs_u0']['ci95'][1] < 0]
        beats = [a for a in allowed if out[a].get('vs_inc') and out[a]['vs_inc']['ci95'][1] < 0]
        best = min(allowed, key=lambda a: out[a].get('vs_u0', {}).get('mean', float('inf')), default=None)
        fairest = min((a for a in arms if out[a].get('vs_u0') and out[a]['vs_u0']['mean'] <= 0
                       and (out[a]['fairness']['road'].get('u0') or {}).get('worst_group_diff') is not None),
                      key=lambda a: out[a]['fairness']['road']['u0']['worst_group_diff'], default=None)
        summary[group] = dict(n_strategies=len(arms), n_allowed=len(allowed), n_allowed_lower_u0_ci=len(lowers),
                              n_allowed_beat_inc_ci=len(beats), allowed=allowed, beat_inc=beats, best_allowed=best,
                              best_allowed_vs_u0=None if best is None else out[best].get('vs_u0'),
                              fairest_not_worse=fairest,
                              fairest_worst_group_diff=None if fairest is None else
                              out[fairest]['fairness']['road']['u0']['worst_group_diff'])
    return dict(arms=out, summary=summary,
                convention='diff = arm - reference (same seed): J over [300, 1200) s veh-s, group mean per-vehicle '
                           'delay change s/veh; negative = better. ALLOWED = every group change <= 10 s/veh vs u0 '
                           '(point; ci95 upper reported); fairness component road = window road delay (primary).')


# ------------------------------------------------------------------------------------------ harness wrappers
def _harness():
    from tools.coopmwm import imitreplay as ir
    from tools.coopmwm import regimeu0 as R
    return ir, R


def plan(root, *, regimeu0_root, arms_path, cell, seeds, k=4, cap=CAP, k_fair=0, **arm_kw):
    ir, R = _harness()
    root = ir.guard_path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'PLAN.json').exists():
        raise ValueError('PLAN.json already frozen')
    src = Path(regimeu0_root) / 'inputs'
    (root / 'inputs').mkdir(exist_ok=True)
    files = {}
    import shutil
    for name in ('CONFIG.json', 'u0.pt'):
        shutil.copyfile(ir.guard_path(src / name), root / 'inputs' / name)
        files[name] = dict(source=str(src / name), sha256=ir.sha_file(root / 'inputs' / name))
    ref = ir._read(Path(regimeu0_root) / 'PLAN.json')['inputs']['u0.pt']['sha256']
    if files['u0.pt']['sha256'] != ref:
        raise ValueError('MAPPO u0 differs from the S-REGIMEU0 base')
    specs = arm_specs(json.loads(G.check_path(arms_path).read_text()), cell, k, k_fair, **arm_kw)
    groups = plan_jobs(cell, seeds, specs, R.world_of)
    ir._write_new(root / 'REGISTRATION.json', dict(schema=SCHEMA, stable_id=STABLE_ID, phase=PHASE, cap_sim_s=int(cap),
                                                   module_sha256=ir.sha_file(__file__), seeds=list(seeds),
                                                   arms=list(specs), unix=time.time()))
    body = dict(schema=SCHEMA + '.plan', stable_id=STABLE_ID, phase=PHASE, cap_sim_s=int(cap), inputs=files,
                groups=groups, specs=specs, planned_sim_s=sum(j['sim_s'] for g in groups for j in g['jobs']),
                module_sha256=ir.sha_file(__file__), unix=time.time())
    if body['planned_sim_s'] > cap:
        raise ValueError(f"planned {body['planned_sim_s']} > cap {cap}")
    body['plan_sha256'] = ir.digest({k2: v for k2, v in body.items() if k2 != 'unix'})
    ir._write_new(root / 'PLAN.json', body)
    return body


def worker(root, ident):
    """regimeu0.worker with the strategy hook (identical plant, ledger checks and summary) + PERVEH.json.gz."""
    ir, R = _harness()
    import gzip
    from . import strategy_hook as SH
    root = Path(root)
    body = ir._read(root / 'PLAN.json')
    job = next(j for g in body['groups'] for j in g['jobs'] if j['id'] == ident)
    original = R.RuleHook
    holder = {}

    def factory(_name):
        holder['hook'] = SH.make_hook(job['strategy_spec'], warmup_s=R.WARMUP_S)
        return holder['hook']
    R.RuleHook = factory
    try:
        R._job = lambda _root, _ident: dict(job, strategy=None if job['strategy_spec'] is None else job['strategy'])
        return R.worker(root, ident)
    finally:
        R.RuleHook = original
        hook = holder.get('hook')
        work = root / 'jobs' / ident
        if hook is not None and hook.pv is not None and work.exists() and not (work / 'PERVEH.json.gz').exists():
            with gzip.open(work / 'PERVEH.json.gz', 'xt', encoding='utf-8') as f:
                json.dump(hook.pv.finish(), f)


def run(root, **kw):
    """regimeu0.run, but every worker is `python -m tools.wzprop.wmsearch_u0 worker` (own PIDs, ledger, cap)."""
    ir, R = _harness()
    import subprocess
    real = subprocess.Popen

    def popen(cmd, *a, **k2):
        if isinstance(cmd, list) and 'tools.coopmwm.regimeu0' in cmd:
            cmd = [c if c != 'tools.coopmwm.regimeu0' else 'tools.wzprop.wmsearch_u0' for c in cmd]
        return real(cmd, *a, **k2)
    R.subprocess.Popen = popen
    try:
        return R.run(root, **kw)
    finally:
        R.subprocess.Popen = real


def collect(root):
    """(plan body, [{seed, J, fair, perveh_missing}]) from the job summaries and per-vehicle records of one root."""
    ir, R = _harness()
    root = Path(root)
    body = ir._read(root / 'PLAN.json')
    rows = []
    for g in body['groups']:
        rec = dict(seed=g['jobs'][0]['world']['seed'], J={}, fair={c: {} for c in PV.COMPONENTS}, perveh_missing=[])
        for j in g['jobs']:
            p = root / 'jobs' / j['id'] / 'SUMMARY.json.gz'
            if not p.exists():
                continue
            s = ir._read(p)
            if s.get('error'):
                continue
            jv = R.control_j(s)
            if jv is not None:
                rec['J'][j['arm']] = jv
            q = root / 'jobs' / j['id'] / 'PERVEH.json.gz'
            pv = ir._read(q) if q.exists() else None
            for c in PV.COMPONENTS:
                fo = PV.fairness_of(pv, c)
                if fo is not None:
                    rec['fair'][c][j['arm']] = fo
            if pv is None or pv.get('error'):
                rec['perveh_missing'].append(j['arm'])
        rows.append(rec)
    return body, rows


def analyze(root):
    ir, _R = _harness()
    body, rows = collect(root)
    res = analyze_rows(rows, body['specs'])
    res.update(schema=SCHEMA + '.analysis', root=str(root), plan_sha256=body['plan_sha256'], spent=ir.spent(Path(root)),
               seeds=len(rows), perveh_missing={r['seed']: r['perveh_missing'] for r in rows if r['perveh_missing']})
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop.wmsearch_u0')
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('plan')
    a.add_argument('--root', required=True)
    a.add_argument('--regimeu0-root', required=True)
    a.add_argument('--arms', required=True)
    a.add_argument('--cell', default='D3200_p60')
    a.add_argument('--seeds', required=True, help='first-last')
    a.add_argument('--k', type=int, default=4)
    a.add_argument('--k-fair', type=int, default=0, help='extra strict-FAIR WM arms (budget: +19,200 sim-s each)')
    a.add_argument('--k-ref', type=int, default=0, help='S-MULTICRIT reference arms (duo12, gate0)')
    a.add_argument('--k-bounded', type=int, help='archive (b) arms (default --k)')
    a.add_argument('--k-fairer', type=int, help='archive (c) arms (default --k)')
    a.add_argument('--with-banned', action='store_true', help='add the banned solo2 comparator arm')
    a = sub.add_parser('run')
    a.add_argument('--root', required=True)
    a.add_argument('--workers', type=int, default=16)
    a.add_argument('--low-workers', type=int, default=8)
    a.add_argument('--load-high', type=float, default=44.0)
    a.add_argument('--normal-priority', action='store_true', help='only if MAIN allows (regimeu0.run rule)')
    a = sub.add_parser('worker')
    a.add_argument('--root', required=True)
    a.add_argument('--job', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('--root', required=True)
    a.add_argument('--out')
    x = ap.parse_args(argv)
    if x.cmd == 'plan':
        lo, hi = (int(v) for v in x.seeds.split('-'))
        body = plan(x.root, regimeu0_root=x.regimeu0_root, arms_path=x.arms, cell=x.cell, seeds=range(lo, hi + 1),
                    k=x.k, k_fair=x.k_fair, k_ref=x.k_ref, k_bounded=x.k_bounded, k_fairer=x.k_fairer,
                    with_banned=x.with_banned)
        print(json.dumps(dict(planned_sim_s=body['planned_sim_s'], arms=list(body['specs']),
                              plan_sha256=body['plan_sha256'])))
    elif x.cmd == 'run':
        print(run(x.root, workers=x.workers, low_workers=x.low_workers, load_high=x.load_high,
                  normal_priority=x.normal_priority))
    elif x.cmd == 'worker':
        worker(x.root, x.job)
    else:
        res = analyze(x.root)
        if x.out:
            Path(x.out).write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(res['summary'], indent=1, default=str))


if __name__ == '__main__':
    main()
