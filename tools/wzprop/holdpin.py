"""CANCELLED by the owner (see CANCELLED); not runnable.

S-HOLDPIN (phase hold_pin_check, <= 100,000 sim-s; MAIN 2026-11-07 ~00:5x): does the UNPINNED solo2 hold - what the
policy's N2 executor runs (wmverify_n2: identical speed writes, NO laneChangeMode 513 pin) - keep the gain that
S-REGIMEU0 measured with the pinned solo2 (regimeu0.RuleHook: G5 own-lane target = lcMode 513)?

Treatment: solo2 executed by wmverify_hook executor 'n2' (WM-C StrategyCore trigger/selection/timing = the S-REGIMEU0
rule; hold law through the plant single writer; no lane target) on top of MAPPO u0, on the EXACT S-REGIMEU0 worlds
(CRN): D3200_p60 989850-989873, D3200_p40 989850-989873, D2400_p60 989724-989747 (24 seeds each) + the identity
episode (u0 989599 D3200_p60 = 53,269.32889342696 veh-s). Comparators are REUSED, not re-run: the S-REGIMEU0 u0 and
pinned wall_h_solo2 episodes of the same worlds (REGIMEU0_RESULTS.jsonl f473caad; their SUMMARY series in the local
regimeu0 roots for the CRN check). 73 episodes = 87,600 sim-s.

    python -m tools.wzprop.holdpin registration --regimeu0-root U                     (text for REGISTRATION.md)
    python -m tools.wzprop.holdpin plan    --root R --regimeu0-root U --registration REGISTRATION.md
    python -m tools.wzprop.holdpin run     --root R --workers 12        (nice 19, CUDA_VISIBLE_DEVICES=,
                                           COOPMWM_MAP_SPEC=<wz3to2 MAP_SPEC>, COOPMWM_G5_RUNTIME=1)
    python -m tools.wzprop.wmverify worker --root R --job ID            (spawned by run)
    python -m tools.wzprop.holdpin analyze --root R --regimeu0 U1 [U2 ...] --out HOLDPIN_ANALYSIS.json

Analysis (fixed before results): per cell paired diffs over [300, 1200) s (regimeu0.control_j), diff = arm -
reference, negative = the arm is better: unpinned - u0 and unpinned - pinned, regimeu0 bootstrap (10,000, RNG
20261106, 2.5/97.5%), per-seed better/worse shares, relative to the u0 mean; CRN: the unpinned episode's per-second
delay series equals the S-REGIMEU0 u0 series before its first activation (cross-host determinism) and the pinned
series before its first activation; lane leaving: share of holds whose holder left its WM lane during the hold, off
seconds / held seconds; safety: collisions, teleports, hard braking, TTC < 3 s (sums + paired deltas).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from . import strategy_search as SS
from . import wmverify as V

SCHEMA = 'wmverify.holdpin.v1'
STABLE_ID = 'S-HOLDPIN'
PHASE = 'hold_pin_check'
CAP = 100_000
ARM = 'solo2_unpinned'
PINNED = 'wall_h_solo2'
CELLS = (('D3200_p60', 3200, 0.6, tuple(range(989850, 989874))),
         ('D3200_p40', 3200, 0.4, tuple(range(989850, 989874))),
         ('D2400_p60', 2400, 0.6, tuple(range(989724, 989748))))
RESULTS_SHA256_PREFIX = 'f473caad'


def spec():
    return dict(SS.SOLO2, name=ARM, gate=None)


def groups(world_of):
    out = [dict(stage='identity', jobs=[dict(id=V.IDENTITY_JOB, arm='u0', strategy=None, strategy_spec=None,
                                             executor='u0', world=world_of(989599), sim_s=V.EPISODE_S)])]
    depth = max(len(c[3]) for c in CELLS)
    for i in range(depth):
        for cell, demand, share, seeds in CELLS:
            if i >= len(seeds):
                continue
            w = world_of(seeds[i], demand, share)
            if w['cell_key'] != cell:
                raise ValueError(cell)
            out.append(dict(stage='paired', cell=cell, jobs=[dict(
                id=f'{cell}_s{seeds[i]}_{ARM}', arm=ARM, strategy=ARM, strategy_spec=spec(), executor='n2',
                world=w, sim_s=V.EPISODE_S)]))
    return out


def registration_text(regimeu0_u0_sha256=None):
    planned = V.EPISODE_S * (1 + sum(len(c[3]) for c in CELLS))
    lines = [
        f'### {STABLE_ID} (phase {PHASE}) - registration (append BEFORE any physics)',
        '- Question (MAIN 2026-11-07 ~00:5x): does the UNPINNED solo2 hold - the policy N2 executor (FULLPOL clean5a '
        'group_hold/SoloHold; wmverify_n2: identical speed writes, no laneChangeMode 513 pin) - keep the gain S-REGIMEU0 '
        'measured with the pinned solo2 (regimeu0.RuleHook, G5 own-lane target = lcMode 513)?',
        '- Closest prior: S-REGIMEU0 (ANALYSIS c0b578f8, REGIMEU0_RESULTS.jsonl f473caad): pinned solo2 -24% at '
        'D3200 vs u0, harmful at D2400. Exact difference: the lcMode pin only (same trigger, selection, timing, hold '
        'law, plant writer, worlds).',
        f'- Treatment: {ARM} = SOLO2 through wmverify_hook executor n2 (StrategyCore + hold law through the plant '
        'single writer, no lane target) on MAPPO u0 d9042cc5 (S-REGIMEU0 inputs u0.pt sha256 '
        f'{regimeu0_u0_sha256 or "<from regimeu0 PLAN>"}), coopmwm clean4 plant, wz3to2, 1,200 s, native warmup '
        '0-300 s, J = team delay over [300, 1200) s.',
        '- Worlds (exact S-REGIMEU0 CRN worlds, TRAIN-type): ' + '; '.join(
            f'{c} {s[0]}-{s[-1]} ({len(s)})' for c, _d, _p, s in CELLS) + '; + identity u0 989599 D3200_p60.',
        '- Comparators REUSED (not re-run): S-REGIMEU0 u0 and wall_h_solo2 (pinned) episodes of the same worlds '
        '(local regimeu0 roots regimeu0, regimeu0_z2, regimeu0_v2, regimeu0_v3, regimeu0_z5).',
        f'- Budget (phase {PHASE}): cap {CAP:,} sim-s; planned {planned:,} sim-s (73 episodes).',
        '- Host: gpuA CPU, <= 12 workers, nice 19, CUDA_VISIBLE_DEVICES= (WM-E GPU training has priority).',
        '- Analysis (fixed before results): per cell paired diffs unpinned - u0 and unpinned - pinned (negative = '
        'unpinned better), regimeu0 bootstrap 10,000 RNG 20261106 2.5/97.5%, per-seed shares, relative to the u0 '
        'mean; CRN: unpinned series == S-REGIMEU0 u0 and pinned series before the first activation; identity J '
        'exact; lane leaving share of holds (holder WM lane != lane at activation during the hold), off/held seconds; '
        'safety: collisions, teleports, hard braking, TTC<3 s sums and paired deltas.',
        '- Stop rules: STOP file; cap; identity or CRN failure -> stop and report; charged jobs never re-run.',
        '- Code: c4/wmverify (tools/wzprop/holdpin.py, wmverify.py, wmverify_hook.py executor n2).',
    ]
    return '\n'.join(lines) + '\n', planned


CANCELLED = ('S-HOLDPIN is CANCELLED by the owner (REGISTRATION: "S-HOLDPIN check skipped"; the firm 15 m/s '
             'hold is banned): plan and run refuse; a new owner-approved registration block is required')


def plan(root, *, regimeu0_root, registration_md, cap=CAP):
    raise ValueError(CANCELLED)


def _plan_cancelled_reference(root, *, regimeu0_root, registration_md, cap=CAP):
    """The former plan body (kept for the record; unreachable)."""
    ir, R = V._harness()
    root = V.guard_path(root)
    if (root / 'PLAN.json').exists():
        raise ValueError('PLAN.json already frozen')
    text = V.guard_path(registration_md).read_text(encoding='utf-8')
    missing = [x for x in (STABLE_ID, PHASE, f'{CAP:,}') if x not in text]
    if missing:
        raise ValueError(f'REGISTRATION.md lacks the S-HOLDPIN registration items {missing}')
    gr = groups(R.world_of)
    planned = sum(j['sim_s'] for g in gr for j in g['jobs'])
    if planned > int(cap) or int(cap) > CAP:
        raise ValueError(f'planned {planned} / cap {cap} outside the registered {CAP}')
    root.mkdir(parents=True, exist_ok=True)
    src = Path(regimeu0_root) / 'inputs'
    (root / 'inputs').mkdir(exist_ok=True)
    files = {}
    for name in ('CONFIG.json', 'u0.pt'):
        shutil.copyfile(V.guard_path(src / name), root / 'inputs' / name)
        files[name] = dict(source=str(src / name), sha256=ir.sha_file(root / 'inputs' / name))
    ref = V._read(Path(regimeu0_root) / 'PLAN.json')['inputs']['u0.pt']['sha256']
    if files['u0.pt']['sha256'] != ref:
        raise ValueError('MAPPO u0 differs from the S-REGIMEU0 base')
    files['u0.pt']['regimeu0_sha256'] = ref
    reg_sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    shas = V.module_shas()
    V._write_new(root / 'REGISTRATION.json', dict(schema=SCHEMA, stable_id=STABLE_ID, phase=PHASE, cap_sim_s=int(cap),
                                                  cells=[c[0] for c in CELLS], registration_md_sha256=reg_sha,
                                                  module_sha256=shas, unix=time.time()))
    body = dict(schema=SCHEMA + '.plan', stable_id=STABLE_ID, phase=PHASE, cap_sim_s=int(cap), dry_run=False,
                inputs=files, groups=gr, planned_sim_s=planned, module_sha256=shas, registration_md_sha256=reg_sha,
                unix=time.time())
    body['plan_sha256'] = ir.digest({k: v for k, v in body.items() if k != 'unix'})
    V._write_new(root / 'PLAN.json', body)
    return body


def run(root, **kw):
    raise ValueError(CANCELLED)


def _summaries(root, ir):
    """{(cell, seed, arm): summary} of one regimeu0-format root (ok and failed)."""
    root = Path(root)
    body = V._read(root / 'PLAN.json')
    out = {}
    for g in body['groups']:
        if g['stage'] != 'paired':
            continue
        for j in g['jobs']:
            p = root / 'jobs' / j['id'] / 'SUMMARY.json.gz'
            if p.exists():
                out[(j['world']['cell_key'], j['world']['seed'], j['arm'])] = (ir._read(p), root / 'jobs' / j['id'])
    return out


def analyze(roots, regimeu0_roots):
    ir, R = V._harness()
    treat, ref = {}, {}
    identity = None
    for r in roots:
        treat.update(_summaries(r, ir))
        body = V._read(Path(r) / 'PLAN.json')
        for g in body['groups']:
            if g['stage'] == 'identity':
                p = Path(r) / 'jobs' / g['jobs'][0]['id'] / 'SUMMARY.json.gz'
                if p.exists():
                    tot = (ir._read(p).get('totals') or {}).get('team_delay_vs')
                    identity = dict(team_delay_vs=tot, expected=R.IDENTITY['expected_team_delay_vs'],
                                    equal=tot is not None and abs(tot - R.IDENTITY['expected_team_delay_vs']) < 1e-6)
    for r in regimeu0_roots:
        ref.update(_summaries(r, ir))
    out = dict(schema=SCHEMA + '.analysis', stable_id=STABLE_ID, phase=PHASE, roots=[str(r) for r in roots],
               regimeu0_roots=[str(r) for r in regimeu0_roots], identity=identity,
               spent=sum(ir.spent(r) for r in roots),
               convention='diff = J(arm) - J(reference) over [300, 1200) s, veh-s; negative = the arm is better',
               cells={})
    for cell, _d, _p, seeds in CELLS:
        rows, crn_bad, failed, tracks = [], [], [], []
        for s in seeds:
            t = treat.get((cell, s, ARM))
            u, pin = ref.get((cell, s, 'u0')), ref.get((cell, s, PINNED))
            if t is None:
                continue
            ts, tdir = t
            if ts.get('error'):
                failed.append(dict(seed=s, error=ts['error']['error'][:300]))
                continue
            side = tdir / 'WMVERIFY.json'
            if side.exists():
                tracks += ((V._read(side).get('receipts') or {}).get('lane_track') or {}).get('records') or []
            first_act = min((a['t'] for a in ts.get('activations') or ()), default=None)
            rec = dict(seed=s, J=R.control_j(ts), safety=ir.safety(ts), activations=len(ts.get('activations') or ()),
                       first_activation_s=first_act)
            for name, x in (('u0', u), ('pinned', pin)):
                if x is None or x[0].get('error'):
                    continue
                xs = x[0]
                rec[f'J_{name}'] = R.control_j(xs)
                rec[f'safety_{name}'] = ir.safety(xs)
                fd = ir.first_divergence(xs, ts)
                rec[f'first_divergence_vs_{name}_s'] = fd
                ok = fd is None or (first_act is not None and fd >= first_act)
                if name == 'pinned':
                    pin_first = min((a['t'] for a in xs.get('activations') or ()), default=None)
                    ok = fd is None or (first_act is not None and pin_first is not None
                                        and fd >= min(first_act, pin_first))
                    rec['same_first_activation_as_pinned'] = pin_first == first_act
                rec[f'crn_ok_vs_{name}'] = ok
                if not ok:
                    crn_bad.append(dict(seed=s, reference=name, first_divergence_s=fd, first_activation_s=first_act))
            rows.append(rec)
        cell_out = dict(n=len(rows), failed=failed, crn_violations=crn_bad, seeds=rows)
        u_mean = None
        for name in ('u0', 'pinned'):
            d = [r['J'] - r[f'J_{name}'] for r in rows if f'J_{name}' in r]
            if not d:
                continue
            lo, hi = R.boot_ci(d)
            base = [r[f'J_{name}'] for r in rows if f'J_{name}' in r]
            if name == 'u0':
                u_mean = sum(base) / len(base)
            cell_out[f'unpinned_minus_{name}'] = dict(
                n=len(d), mean_diff=sum(d) / len(d), ci95=[lo, hi],
                rel_to_u0_mean=None if u_mean is None else (sum(d) / len(d)) / u_mean,
                rel_ci95_to_u0_mean=None if u_mean is None else [lo / u_mean, hi / u_mean],
                better_share=sum(x < 0 for x in d) / len(d), worse_share=sum(x > 0 for x in d) / len(d))
        pd = [r['J_pinned'] - r['J_u0'] for r in rows if 'J_pinned' in r and 'J_u0' in r]
        if pd:
            lo, hi = R.boot_ci(pd)
            cell_out['pinned_minus_u0_same_seeds'] = dict(n=len(pd), mean_diff=sum(pd) / len(pd), ci95=[lo, hi])
        saf = {}
        for k in ('collisions', 'teleports', 'hard_brake', 'ttc_below_3_vs'):
            vals = [r['safety'].get(k) or 0. for r in rows]
            saf[k] = sum(vals)
            for name in ('u0', 'pinned'):
                q = [(r['safety'].get(k) or 0.) - (r[f'safety_{name}'].get(k) or 0.) for r in rows
                     if f'safety_{name}' in r]
                if q:
                    lo, hi = R.boot_ci(q)
                    saf[f'{k}_minus_{name}'] = dict(mean=sum(q) / len(q), ci95=[lo, hi])
        cell_out['safety'] = saf
        cell_out['lane_leaving'] = dict(
            holds=len(tracks), left=sum(1 for x in tracks if x['off_ticks']),
            left_share=(sum(1 for x in tracks if x['off_ticks']) / len(tracks)) if tracks else None,
            off_seconds=sum(x['off_ticks'] for x in tracks), held_seconds=sum(x['ticks'] for x in tracks),
            note='records capped at 60 per episode in WMVERIFY.json receipts')
        cell_out['same_first_activation_as_pinned'] = sum(1 for r in rows if r.get('same_first_activation_as_pinned'))
        out['cells'][cell] = cell_out
    out['integrity'] = dict(identity_ok=bool(identity and identity['equal']),
                            crn_ok=all(not c['crn_violations'] for c in out['cells'].values()),
                            failed=sum(len(c['failed']) for c in out['cells'].values()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop.holdpin', description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('registration')
    a.add_argument('--regimeu0-root')
    a = sub.add_parser('plan')
    a.add_argument('--root', required=True)
    a.add_argument('--regimeu0-root', required=True)
    a.add_argument('--registration', required=True)
    a = sub.add_parser('run')
    a.add_argument('--root', required=True)
    a.add_argument('--workers', type=int, default=12)
    a.add_argument('--low-workers', type=int, default=8)
    a.add_argument('--load-high', type=float, default=44.0)
    a = sub.add_parser('analyze')
    a.add_argument('--root', required=True, nargs='+')
    a.add_argument('--regimeu0', required=True, nargs='+')
    a.add_argument('--out', required=True)
    x = ap.parse_args(argv)
    if x.cmd == 'registration':
        u0 = V._read(Path(x.regimeu0_root) / 'PLAN.json')['inputs']['u0.pt']['sha256'] if x.regimeu0_root else None
        print(registration_text(u0)[0], end='')
    elif x.cmd == 'plan':
        body = plan(x.root, regimeu0_root=x.regimeu0_root, registration_md=x.registration)
        print(json.dumps(dict(plan_sha256=body['plan_sha256'], planned_sim_s=body['planned_sim_s'])))
    elif x.cmd == 'run':
        print(run(x.root, workers=x.workers, low_workers=x.low_workers, load_high=x.load_high))
    else:
        out = Path(x.out)
        res = analyze(x.root, x.regimeu0)
        with out.open('x', encoding='utf-8') as f:
            f.write(json.dumps(res, indent=1, sort_keys=True, default=str) + '\n')
        print(json.dumps({c: {k: v.get(k) for k in ('n', 'unpinned_minus_u0', 'unpinned_minus_pinned', 'lane_leaving')}
                          for c, v in res['cells'].items()}, indent=1, default=str))


if __name__ == '__main__':
    main()
