"""Claim-4 check: does the WM estimate the expected benefit of S-REGIME strategies? (zero SUMO)

States: TRAIN world-cell tapes of WM-D (wzqual_native_tape/1, identity-gated vs the labelled natives) at the S-REGIME
regimes (cells), sampled at fixed decision times t0 (outcome-blind grid). At each state the WM sees only the
observable content (tapes.enrich rules: CAV self-states, CAV-detected neighbours, RSU entrance counters).

Strategies (S-REGIME rg_core.STRATEGIES, the probes' registered pickers and parameters = vendored pg_plan):
  wall_g, wall_h          pick_wall (one zone CAV per lane, spread <= 200 m) + VARIANTS params
  wall_{g,h}_solo{L}      the same wall geometry, only the member in SUMO lane L holds
  pair_t                  pick_pair + pair_t params (anchor release X_B - 150)
  solo2_multi             one activation = wall_h_solo2 at a state (its 2-slot concurrency is an episode property)
Not evaluable by the WM planning API (roles other than 'hold'): lane_balance, zipper, early_merge (COOPKINDS roles),
wall_fill, lane_redistribute (LCMOVES 'prog'): reported as such, never imputed.

WM expected benefit of a strategy at a regime = mean over states of the WM T = J(base) - J(strategy) on [t0, t0+300)
(CRN-paired WM draws), with T = 0 where the strategy's trigger geometry is absent (the episode strategy does not act).
Measured (S-REGIME, native base): per paired seed J_total(base) - J_total(strategy) over the 1,200 s episode.
Agreement: per regime and pooled Spearman across strategies, sign accuracy (all pairs, and pairs whose measured
95% CI excludes 0), world/seed bootstrap CIs (2,000 draws, RNG 20261106). Units differ (300 s activation vs
1,200 s episode); only ranks and signs are compared.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import compare as C
from . import geometry as G
from . import n3
from . import tapes as TP

HOLD_STRATEGIES = ['wall_g', 'wall_h', 'pair_t'] + [f'{w}_solo{ln}' for w in ('wall_g', 'wall_h') for ln in (0, 1, 2)]
ALIASES = {'solo2_multi': 'wall_h_solo2'}
NOT_EVALUABLE = {'lane_balance': 'COOPKINDS lane_to role', 'zipper': 'COOPKINDS zip roles',
                 'early_merge': 'COOPKINDS merge_early/tau roles', 'wall_fill': "LCMOVES 'prog' role",
                 'lane_redistribute': "LCMOVES 'prog' role"}
DEFAULT_T0 = (450, 650, 850)
B_BOOT, RNG = 2000, 20261106


def strategy_members(name, cavs):
    """Member specs of a hold strategy at a snapshot (rg_core.pick_group semantics), else None."""
    P, _ = n3.modules()
    snap = dict(cavs=cavs)
    if name.startswith('wall'):
        wall = P.pick_wall(snap)
        if not wall:
            return None
        var = name.split('_solo')[0]
        prm = P.VARIANTS[var][1]
        mem = [P.member_spec(c, prm) for c in wall]
        if '_solo' in name:
            lane = int(name[-1])
            mem = [m for m in mem if m['lane'] == lane]
        return mem or None
    if name == 'pair_t':
        pair = P.pick_pair(snap)
        if not pair:
            return None
        prm = P.VARIANTS['pair_t'][1]
        return [P.member_spec(c, prm, anchor=pair[0]['id']) for c in pair]
    raise KeyError(name)


def build_states(tapes_root, cells, out_path, t0s=DEFAULT_T0, strategies=HOLD_STRATEGIES):
    """MOMENTS-format rows (one per tape x t0) whose variants are the feasible strategies + observed content."""
    root = Path(tapes_root) / 'runs'
    rows, report = [], dict(states=0, tapes=0, by_cell={})
    for cell in cells:
        d = root / cell
        if not d.exists():
            continue
        for sdir in sorted(d.glob('s*')):
            tp = sdir / 'prefix' / 'TAPE.npz'
            if not tp.exists():
                continue
            tape = TP.load_tape(tp)
            entries = TP.first_appearance(tape)
            seed = int(tape['meta']['seed'])
            report['tapes'] += 1
            for t0 in t0s:
                frame = TP.frame_at(tape, float(t0))
                cavs, det, hidden = TP.observable(frame)
                cav_rows = [dict(id=c['id'], edge=c['edge'], lane=c['lane'], x=round(c['x'], 3), v=round(c['v'], 4))
                            for c in cavs]
                variants = {}
                for s in strategies:
                    mem = strategy_members(s, cav_rows)
                    if mem:
                        fam = 'abreast_wall' if s.startswith('wall') else 'closing_pair'
                        variants[s] = dict(family=fam, params=dict(mem[0]['params']), members=mem)
                dem, pen = cell.split('_')
                rows.append(dict(cell=cell, demand=int(dem[1:]), penetration=int(pen[1:]) / 100.0, seed=seed,
                                 window=None, t0=int(t0), valid=True, sid=f'rgwm_{cell}_s{seed}_t{t0}',
                                 all_cavs_t0=cav_rows, variants=variants,
                                 feasible={s: s in variants for s in strategies},
                                 observed=dict(vehicles=det, counter=TP.counter_at(tape, float(t0), entries),
                                               coverage=dict(cavs=len(cavs), detected=len(det), undetected=hidden),
                                               tape=dict(path=str(tp), seed=seed, cell=cell))))
                report['states'] += 1
                report['by_cell'][cell] = report['by_cell'].get(cell, 0) + 1
    out_path = G.check_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('x') as fh:
        fh.write(json.dumps(rows))
    return report


# ------------------------------------------------------------------------------------------ measured
def measured(results_paths, cells):
    """{cell: {strategy: {seed: base - strategy J_total}}} from S-REGIME episode rows (native base)."""
    eps = defaultdict(dict)
    for p in results_paths:
        for line in G.check_path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('status') != 'ok' or r['cell'] not in cells:
                continue
            eps[(r['cell'], int(r['seed']))][r['arm']] = float(r['J_total'])
    out = defaultdict(lambda: defaultdict(dict))
    for (cell, seed), arms in eps.items():
        if 'base' not in arms:
            continue
        for arm, j in arms.items():
            if arm != 'base':
                out[cell][arm][seed] = arms['base'] - j
    return out


# ------------------------------------------------------------------------------------------ WM side
def wm_values(states_path, scores_path):
    """{cell: {strategy: {seed: [T per state]}}}, infeasible strategy at a state -> 0 (does not act)."""
    states = json.loads(G.check_path(states_path).read_text())
    sc = C.read_scores(scores_path)
    errs = set()
    for line in G.check_path(scores_path).read_text().splitlines():
        if line.strip() and 'error' in json.loads(line):
            errs.add(json.loads(line)['sid'])
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for m in states:
        if m['sid'] in errs:
            continue
        scored = any((m['sid'], s) in sc for s in m['variants'])
        if m['variants'] and not scored:
            continue                                   # state not scored (missing): skipped, never imputed
        for s, ok in m['feasible'].items():
            t = sc[(m['sid'], s)]['T']['mean'] if ok else 0.0
            out[m['cell']][s][int(m['seed'])].append(float(t))
    return out


def _mean_by_unit(d):
    vals = [float(np.mean(v)) if isinstance(v, list) else float(v) for v in d.values()]
    return float(np.mean(vals)) if vals else None


def _boot(d, rng):
    keys = list(d)
    if not keys:
        return None
    pick = rng.integers(0, len(keys), len(keys))
    return _mean_by_unit({i: d[keys[j]] for i, j in enumerate(pick)})


def agreement(wm, meas, cells, strategies):
    pairs = []
    for cell in cells:
        for s in strategies:
            src = ALIASES.get(s, s)
            if s in NOT_EVALUABLE or not wm.get(cell, {}).get(src) or not meas.get(cell, {}).get(s):
                continue
            pairs.append((cell, s, src))
    return pairs


def metrics_from(wm_means, meas_means, pairs, meas_conf=None):
    out = {}
    by_cell = defaultdict(list)
    for cell, s, _src in pairs:
        by_cell[cell].append((wm_means[(cell, s)], meas_means[(cell, s)]))
    out['spearman_by_cell'] = {c: C.spearman([a for a, _ in v], [b for _, b in v]) for c, v in by_cell.items()}
    vals = [x for x in out['spearman_by_cell'].values() if x is not None]
    out['spearman_within_mean'] = float(np.mean(vals)) if vals else None
    allw = [wm_means[(c, s)] for c, s, _ in pairs]
    allm = [meas_means[(c, s)] for c, s, _ in pairs]
    out['spearman_pooled'] = C.spearman(allw, allm)
    sign = [(np.sign(a) == np.sign(b)) for a, b in zip(allw, allm)]
    out['sign_accuracy'] = float(np.mean(sign)) if sign else None
    if meas_conf is not None:
        conf = [(np.sign(wm_means[(c, s)]) == np.sign(meas_means[(c, s)])) for c, s, _ in pairs if meas_conf[(c, s)]]
        out['sign_accuracy_confident'] = float(np.mean(conf)) if conf else None
        out['n_confident'] = len(conf)
    out['n_pairs'] = len(pairs)
    return out


def run(states_path, scores_path, results_paths, *, label='WM', checkpoint=None, cells=None, strategies=None):
    states = json.loads(G.check_path(states_path).read_text())
    cells = sorted(cells or {m['cell'] for m in states})
    strategies = list(strategies or (HOLD_STRATEGIES + ['solo2_multi'] + list(NOT_EVALUABLE)))
    wm = wm_values(states_path, scores_path)
    meas = measured(results_paths, set(cells))
    pairs = agreement(wm, meas, cells, strategies)
    wm_means = {(c, s): _mean_by_unit(wm[c][src]) for c, s, src in pairs}
    meas_means = {(c, s): _mean_by_unit(meas[c][s]) for c, s, _ in pairs}
    rng = np.random.default_rng(RNG)
    meas_ci = {}
    for c, s, _ in pairs:
        bs = [_boot(meas[c][s], rng) for _ in range(B_BOOT)]
        meas_ci[(c, s)] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
    conf = {k: (ci[0] > 0 or ci[1] < 0) for k, ci in meas_ci.items()}
    point = metrics_from(wm_means, meas_means, pairs, conf)
    boots = defaultdict(list)
    for _ in range(B_BOOT):                            # resample WM worlds and REGIME seeds per regime
        wb = {(c, s): _boot(wm[c][src], rng) for c, s, src in pairs}
        mb = {(c, s): _boot(meas[c][s], rng) for c, s, _ in pairs}
        mm = metrics_from(wb, mb, pairs)
        for k in ('spearman_pooled', 'spearman_within_mean', 'sign_accuracy'):
            if mm.get(k) is not None:
                boots[k].append(mm[k])
    ci = {k: [round(float(np.quantile(v, .025)), 4), round(float(np.quantile(v, .975)), 4)] for k, v in boots.items()}
    table = []
    for c, s, src in pairs:
        n_states = sum(len(v) for v in wm[c][src].values())
        feas = float(np.mean([t != 0.0 for v in wm[c][src].values() for t in v])) if n_states else None
        table.append(dict(cell=c, strategy=s, wm_strategy=src, wm_benefit_per_300s=round(wm_means[(c, s)], 1),
                          wm_worlds=len(wm[c][src]), wm_states=n_states, wm_feasible_share=feas,
                          measured_benefit_per_1200s=round(meas_means[(c, s)], 1),
                          measured_ci95=[round(x, 1) for x in meas_ci[(c, s)]], measured_seeds=len(meas[c][s]),
                          measured_confident=conf[(c, s)]))
    return dict(schema='wmc-regime-benefit-v1', label=label, checkpoint=checkpoint, states=str(states_path),
                scores=str(scores_path), results=[str(p) for p in results_paths], cells=cells, point=point,
                ci95=ci, table=table, not_evaluable=NOT_EVALUABLE, aliases=ALIASES,
                sign_convention='benefit = J(base) - J(strategy), veh-s; positive = strategy better',
                interpretation='WM per-activation expected benefit (300 s, observable TRAIN states) vs S-REGIME '
                               'per-episode measured benefit (1,200 s, paired seeds): ranks and signs only')


def markdown(res):
    p, ci = res['point'], res['ci95']

    def f(x, k=None):
        if x is None:
            return 'n/a'
        s = f'{x:.3f}'
        if k and k in ci:
            s += f' [{ci[k][0]:.3f}, {ci[k][1]:.3f}]'
        return s
    L = [f"**{res['label']}** - pooled Spearman {f(p['spearman_pooled'], 'spearman_pooled')}, within-regime mean "
         f"{f(p['spearman_within_mean'], 'spearman_within_mean')}, sign accuracy {f(p['sign_accuracy'], 'sign_accuracy')}"
         f" (n={p['n_pairs']}); sign accuracy on measured-confident pairs {f(p.get('sign_accuracy_confident'))} "
         f"(n={p.get('n_confident')})", '',
         '| regime | strategy | WM benefit / 300 s | feasible share | measured benefit / 1200 s [95% CI] |',
         '|---|---|---|---|---|']
    for r in res['table']:
        L.append(f"| {r['cell']} | {r['strategy']} | {r['wm_benefit_per_300s']:+.0f} | "
                 f"{(r['wm_feasible_share'] or 0):.2f} | {r['measured_benefit_per_1200s']:+.0f} "
                 f"[{r['measured_ci95'][0]:+.0f}, {r['measured_ci95'][1]:+.0f}] |")
    L.append('')
    L.append('Per-regime Spearman: ' + ', '.join(f'{c} {f(v)}' for c, v in p['spearman_by_cell'].items()))
    return '\n'.join(L)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop regime')
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('states')
    b.add_argument('--tapes', required=True)
    b.add_argument('--cells', required=True, help='comma list, e.g. D3200_p60,D3200_p40')
    b.add_argument('--t0', default=','.join(map(str, DEFAULT_T0)))
    b.add_argument('--out', required=True)
    r = sub.add_parser('analyze')
    r.add_argument('--states', required=True)
    r.add_argument('--scores', required=True)
    r.add_argument('--results', action='append', required=True)
    r.add_argument('--label', default='WM')
    r.add_argument('--checkpoint')
    r.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'states':
        print(json.dumps(build_states(a.tapes, a.cells.split(','), a.out, tuple(int(x) for x in a.t0.split(',')))))
        return
    res = run(a.states, a.scores, a.results, label=a.label, checkpoint=a.checkpoint)
    out = G.check_path(a.out)
    with Path(out).open('x') as fh:
        fh.write(json.dumps(res, indent=1, default=float))
    with Path(out).with_suffix('.md').open('x') as fh:
        fh.write(markdown(res) + '\n')
    print(markdown(res))


if __name__ == '__main__':
    main()

