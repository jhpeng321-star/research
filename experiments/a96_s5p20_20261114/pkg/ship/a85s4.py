"""A85 S4 'look elsewhere' WM plan search (prep 2026-11-13). Zero SUMO, never trains.

S4 = the A75 S2b search (a75ss cae54dc1 over strategy_search 02e119c1, both imported UNCHANGED from this directory /
the code tree) with ONE extra rule, read from the cfg key ``exclude_lane_sets`` (S4: [[1, 2]]):

  a candidate whose set of role lanes ({r['lane'] for r in roles}, SUMO lane index; lane 0 = closing lane, lanes 1/2 =
  through lanes) equals an excluded set is rejected AT GENERATION TIME and replaced by a fresh draw from the same
  random.Random(seed) stream:
    - initial random plans / immigrants: SS.random_strategy is drawn again until the plan is not excluded;
    - mutation children: the parent choice AND the mutation are drawn again (rng.choice(parents) + SS.mutate);
      a mutation that returns None (unchanged S2b behaviour: invalid child, later dropped by run()) is NOT replaced;
    - hand-library seeds (SS.seeds() + SS.dense_seeds()) with an excluded lane set are left out of the initial
      population. Comparators (SS.COMPARATORS, added to the finals) and the novelty reference set (all seeds) are
      unchanged.
So every configured population size (n_init, n_children, n_immigrants) is kept. With ``exclude_lane_sets`` absent or
empty every line below runs exactly like a75ss.search_cell (same RNG calls; tests/test_a85s4.py).

Lines that differ from a75ss.search_cell are marked ``# A85-S4``. res['a85'] records the filter, rejection counts and
the per-generation generated counts.

Driver (same arguments as a75ss; GPU box, cwd = code tree code_split/code_v12r7b):
  python -m a85s4 --states data/regime_STATES.json --cell D2800_p60 --config CFG.json --out OUT --ckpt wm_s.pt ...
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import random

from tools.wzprop import strategy_search as SS

import a75ss as A

SCHEMA = 'a85.s4.lanefilter.v1'
MAX_REDRAW = 10000          # a generator that cannot leave the excluded sets fails loudly instead of hanging


def lane_set(s):
    """Set of role lanes in the strategy's own lane field (SUMO lane index, 0 = closing lane)."""
    return frozenset(int(r['lane']) for r in s['roles'])


def excluded_sets(cfg):
    return frozenset(frozenset(int(x) for x in ls) for ls in ((cfg or {}).get('exclude_lane_sets') or []))


class LaneFilter:
    """Generation-time filter with counters (excl empty = pass-through, no extra RNG use)."""

    def __init__(self, excl):
        self.excl = frozenset(excl)
        self.rejected = dict(seed=[], random=0, mutation=0)
        self.generated = []           # per generation: number of candidates handed to run()

    def bad(self, s):
        return s is not None and bool(self.excl) and lane_set(s) in self.excl

    def random(self, rng, cfg, dense, hold_only, rc):
        for _ in range(MAX_REDRAW):
            s = SS.random_strategy(rng, cfg['max_roles'], dense, hold_only, rc)
            if not self.bad(s):
                return s
            self.rejected['random'] += 1
        raise RuntimeError('A85-S4: random generator stuck in excluded lane sets')

    def child(self, rng, parents, cfg, dense, hold_only):
        for _ in range(MAX_REDRAW):
            s = SS.mutate(rng, rng.choice(parents), cfg['p_mutate'], dense, hold_only)
            if not self.bad(s):
                return s
            self.rejected['mutation'] += 1
        raise RuntimeError('A85-S4: mutation stuck in excluded lane sets')

    def seeds(self, pop):
        out = []
        for s in pop:
            if self.bad(s):
                self.rejected['seed'].append(s['name'])
            else:
                out.append(s)
        return out

    def info(self):
        return dict(schema=SCHEMA, exclude_lane_sets=sorted(sorted(x) for x in self.excl),
                    rejected_seeds=list(self.rejected['seed']), rejected_random=self.rejected['random'],
                    rejected_mutation=self.rejected['mutation'], generated_per_generation=list(self.generated))


def initial_population(rng, states, cfg, flt, extra=()):
    """Screen-state draw + generation 0, exactly as a75ss.search_cell (cfg already merged with SS.DEFAULTS and the
    SS globals set). Returns (screen, pop, rc)."""
    dense, hold_only = bool(cfg['dense']), bool(cfg['hold_only'])
    screen = rng.sample(states, min(len(states), int(cfg['n_screen_states'])))
    pop = [s for s in SS.seeds() if not (hold_only and any(r['lc_target'] is not None for r in s['roles']))]
    if dense:
        pop += SS.dense_seeds()
    if SS.ONSET_MAX is not None:
        pop = [s for s in pop if all(int(r['onset_s']) <= int(SS.ONSET_MAX) for r in s['roles'])]
    if SS.SEARCH_GRID:
        pop = [s for s in pop if all(r[k] in SS.SEARCH_GRID[k] for r in s['roles'] for k in SS.SEARCH_GRID)]
    pop = flt.seeds(pop)                                                               # A85-S4
    rc = tuple(cfg['role_counts']) if cfg.get('role_counts') else None
    pop += list(extra)
    pop += [flt.random(rng, cfg, dense, hold_only, rc) for _ in range(int(cfg['n_init']))]   # A85-S4
    return screen, pop, rc


def set_globals(cfg):
    """The SS module globals a75ss.search_cell sets before generating (cfg merged with SS.DEFAULTS)."""
    SS.UNCOND, SS.F_VS_BASE = bool(cfg.get('unconditional')), bool(cfg.get('f_vs_base'))
    SS.NOV_W = float(cfg.get('novelty_weight', 0.0))
    SS.MIN_ROLES = int(cfg.get('min_roles', 0))
    SS.LANE2_EXTRA_S = float(cfg.get('lane2_extra_margin_s', 0.0))
    SS.REC_W = float(cfg.get('rec_weight', 0.0))


def search_cell(wm, states, cfg=None, log=None):
    """a75ss.search_cell + the A85-S4 generation-time lane-set filter (see module docstring)."""
    cfg = dict(SS.DEFAULTS, **(cfg or {}))
    set_globals(cfg)
    extra = [dict(s) for s in (cfg.get('extra_seeds') or [])]
    for s in extra:
        SS.validate(s)
    SS.ONSET_MAX = cfg.get('onset_max')
    SS.SEARCH_GRID = cfg.get('search_grid')
    act_first = bool(A._A.get('rank_activated_first'))
    flt = LaneFilter(excluded_sets(cfg))                                               # A85-S4

    def act(r):
        return int(A.activated(r)) if act_first else 0
    rng = random.Random(cfg['seed'])
    lam, groups = float(cfg['lam']), bool(cfg['fairness'])
    sk = dict(fair_tol=float(cfg['fair_tol']), bound_s=float(cfg['bound_s']), bound_stat=cfg['bound_stat'],
              fairer_total=cfg['fairer_total'], harm_cal=cfg['harm_cal'])
    dense, hold_only = bool(cfg['dense']), bool(cfg['hold_only'])
    screen, pop, rc = initial_population(rng, states, cfg, flt, extra)                 # A85-S4 (same draws)
    flt.generated.append(len(pop))                                                     # A85-S4
    evaluated, total = {}, 0
    archives = {a: {} for a in SS.ARCHIVES}
    comp_keys = {SS.strategy_key(c) for c in SS.COMPARATORS}

    def run(cands, sts, K):
        nonlocal total
        cands = [SS.uncond(c) for c in cands]
        cands = [c for c in cands if c is not None and len(c['roles']) >= SS.MIN_ROLES]
        cands = [c for c in cands if c is not None and SS.strategy_key(c) not in evaluated and not SS.banned(c)]
        if not cands:
            return
        per, n = SS.evaluate(wm, sts, cands, K, cfg['H'], groups=groups)
        total += n
        for c in cands:
            k = SS.strategy_key(c)
            res = dict(strategy=c, key=k, descriptor=list(SS.descriptor(c)), novelty=SS.novelty(c),
                       **SS.summarize(per[k], lam, **sk))
            evaluated[k] = res
            d = SS.descriptor(c)
            for a, arch in archives.items():
                if SS.admits(a, res) and (d not in arch or SS.rank_key(a)(res) < SS.rank_key(a)(arch[d])):
                    arch[d] = res
        if log:
            log(dict(evaluated=len(evaluated), rollouts=total, **{f'archive_{a}': len(v) for a, v in archives.items()}))

    def top(arch, a, n):
        return sorted(arch.values(), key=SS.rank_key(a))[:n]

    run(pop, screen, cfg['K_screen'])
    for _ in range(int(cfg['generations'])):
        ranked = sorted(evaluated.values(),
                        key=lambda r: (act(r), r['score'] + SS.NOV_W * min(SS.NOV_CAP, r.get('novelty', 0.0))),
                        reverse=True)
        parents = [r['strategy'] for r in ranked[:int(cfg['n_elite'])]]
        for a in SS.ARCHIVES:
            parents += [r['strategy'] for r in top(archives[a], a, int(cfg['n_elite']) if a != 'fair'
                                                   else int(cfg['n_elite']) // 2)]
        parents += extra
        kids = [flt.child(rng, parents, cfg, dense, hold_only)                         # A85-S4
                for _j in range(int(cfg['n_children']))]
        kids += [flt.random(rng, cfg, dense, hold_only, rc)                            # A85-S4
                 for _j in range(int(cfg['n_immigrants']))]
        flt.generated.append(len(kids))                                                # A85-S4
        run(kids, screen, cfg['K_screen'])
    nf = int(cfg['finalists'])
    fin = {}
    for r in sorted(evaluated.values(), key=lambda r: (act(r), r['score']), reverse=True)[:nf]:
        fin[r['key']] = r
    for a in ('F', 'D', 'bounded', 'fairer', 'fair', 'delay'):
        for r in top(archives[a], a, nf):
            if len(fin) < 4 * nf:
                fin.setdefault(r['key'], r)
    finals = [dict(f['strategy']) for f in fin.values()]
    for c in SS.COMPARATORS:
        if SS.strategy_key(c) not in {SS.strategy_key(f) for f in finals}:
            finals.append(dict(c))
    fstates = states
    if int(cfg['n_final_states']) and len(states) > int(cfg['n_final_states']):
        fstates = random.Random(cfg['seed'] + 1).sample(states, int(cfg['n_final_states']))
    per, n = SS.evaluate(wm, fstates, finals, cfg['K_final'], cfg['H'], groups=groups)
    total += n
    final = []
    for f in finals:
        k = SS.strategy_key(f)
        final.append(dict(strategy=f, key=k, descriptor=list(SS.descriptor(f)), comparator=k in comp_keys,
                          novelty=SS.novelty(f), banned=SS.banned(f), **SS.summarize(per[k], lam, **sk)))
    final.sort(key=lambda r: (act(r), r['score']), reverse=True)
    cands = [r for r in final if not r['comparator'] and not r['banned']]

    def best(a):
        xs = sorted((r for r in cands if SS.admits(a, r)), key=SS.rank_key(a))
        return xs[0] if xs else None
    beats = [r for r in cands if r['gain_vs_inc_lcb'] > 0 and r.get('bounded') is not False]
    fairness = ('per-vehicle' if final and final[0].get('bounded') is not None else
                'groups only' if final and final[0].get('fair') is not None else
                'unavailable (WM has no per-group delay)')

    def listing(a):
        return [dict(descriptor=list(d), key=r['key'], name=r['strategy']['name'], score=r['score'],
                     worst_group_inc=r.get('worst_group_inc'), cvar90_inc=r.get('cvar90_inc'))
                for d, r in sorted(archives[a].items(), key=lambda kv: SS.rank_key(a)(kv[1]))]
    out = dict(schema='wmc-strategy-search-v3', config=cfg, incumbent=SS.INCUMBENT['name'],
               banned_v_w=list(SS.BANNED_V_W),
               n_states=len(states), n_screen_states=len(screen), n_final_states=len(fstates),
               evaluated=len(evaluated), wm_rollouts=total, fairness=fairness,
               archive_cells={a: len(v) for a, v in archives.items()}, final=final,
               best={a: best(a) for a in SS.ARCHIVES}, best_beating_incumbent=beats[0] if beats else None,
               comparators={r['strategy']['name']: {k: r.get(k) for k in (
                   'gain_vs_base', 'gain_vs_base_lcb', 'group_inc', 'worst_group_inc', 'cvar90_inc', 'bounded')}
                   for r in final if r['comparator']},
               archives={a: listing(a) for a in SS.ARCHIVES},
               screened=[dict(key=r['key'], name=r['strategy']['name'], n_roles=len(r['strategy']['roles']),
                              trigger=r['strategy']['trigger'], feasible=r['feasible'],
                              gain_vs_base=r['gain_vs_base'], gain_vs_inc=r['gain_vs_inc'],
                              fair_cal=r.get('fair_cal')) for r in evaluated.values()])
    if flt.excl:                                                                       # A85-S4
        out['a85'] = dict(flt.info(), a85s4_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return out


def main(argv=None):
    """a75ss.main unchanged (checkpoint pin, outputs, TOP writer), routed through the S4 search_cell."""
    A.search_cell = search_cell                                                        # A85-S4
    return A.main(argv)


if __name__ == '__main__':
    main()
