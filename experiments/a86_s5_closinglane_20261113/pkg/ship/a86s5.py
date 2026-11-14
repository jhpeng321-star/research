"""A86 S5 closing-lane WM plan search with structural niching (prep 2026-11-13). Zero SUMO, never trains.

A86-S5 = a85s4.py (A85 S4, sha 7ca82976) + structural NICHING, switched by the cfg key ``niching`` (absent = exactly
a85s4). Code lines that differ from a85s4.py are marked ``# A86-S5``. Per-plan scoring is untouched (same SS.evaluate /
SS.summarize / score formula, states, horizon, K, WM). Niche of a plan = (form family, sorted role-lane tuple), both
from the plan's structure only (no outcome):
  form family = A80 FORM_RULE (a80_s2b.form_family, copied verbatim below): gap > pulse > taper > zone > rolling >
  short > other (first match).
``niching`` = {"definition": "a80_form_family x role_lane_set", "parents": bool, "finalists": bool, "top_cap": int}:
  parents    each generation's parent lists (the global elite list and each archive's top list) are taken from the
             NICHE ROUND-ROBIN order of the same ranked list: best plan of each niche first (niches ordered by their
             best plan), then each niche's second best, ...; activated plans stay ahead of non-activated ones when
             rank_activated_first is set (round-robin inside each activation tier). The counts are unchanged.
  finalists  the same round-robin order for the score finalists and the per-archive finalists (counts unchanged).
  top_cap    after the unchanged a75ss.main writer: its global top file is renamed
             TOP_OFFER_CONDITIONS_GLOBAL_UNCAPPED_<cell>.json and a second file
             TOP_OFFER_CONDITIONS_NICHECAP<k>_FOR_SUMO_<cell>.json is written with the same row format: the same
             eligible final plans (not comparator, not banned, activated; final order), taken niche-best round-robin
             (then second best, ...), at most top_cap per niche, up to --top-bank rows. NICHE_TOP_<cell>.json lists
             both selections with niche labels.
Driver: python -m a86s5 (same arguments as a75ss).

---- a85s4 docstring (unchanged) ----
A85 S4 'look elsewhere' WM plan search (prep 2026-11-13). Zero SUMO, never trains.

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

import argparse                                                                    # A86-S5
import hashlib
import json                                                                        # A86-S5
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


# ------------------------------------------------------------------------------------------ A86-S5 niching
NICHE_SCHEMA = 'a86.s5.niche.v1'                                                   # A86-S5
NICHE_DEFINITION = 'a80_form_family x role_lane_set'                               # A86-S5
FORMS = ('gap', 'pulse', 'taper', 'zone', 'rolling', 'short', 'other')             # A86-S5 (a80_s2b.FORMS)


def form_family(spec):                                                             # A86-S5 verbatim a80_s2b 48c5c448
    """FORM_RULE."""
    roles = spec['roles']
    if any(r.get('start') == 'gap' for r in roles):
        return 'gap'
    if any('pulse_on_s' in r for r in roles):
        return 'pulse'
    if any('taper_in_s' in r or 'taper_out_s' in r for r in roles):
        return 'taper'
    if spec['zone'] in ('mid', 'merge', 'custom'):
        return 'zone'
    if len(roles) >= 2 and len({int(r['onset_s']) for r in roles}) >= 2:
        return 'rolling'
    if max(int(r['hold_s']) for r in roles) <= 45:
        return 'short'
    return 'other'


def niche(s):                                                                      # A86-S5
    """(form family, sorted role lanes) of a strategy dict: structure only."""
    return (form_family(s), tuple(sorted(lane_set(s))))


def niche_label(s):                                                                # A86-S5
    f, ls = niche(s)
    return f + ':L' + ''.join(str(x) for x in ls)


def niching_cfg(cfg):                                                              # A86-S5
    """cfg['niching'] -> dict(parents, finalists, top_cap) or None (absent/empty = a85s4 behaviour)."""
    n = (cfg or {}).get('niching') or None
    if not n:
        return None
    if n.get('definition') != NICHE_DEFINITION:
        raise ValueError(f'niching.definition must be {NICHE_DEFINITION!r}')
    bad = set(n) - {'definition', 'parents', 'finalists', 'top_cap'}
    if bad:
        raise ValueError(f'unknown niching keys {sorted(bad)}')
    return dict(parents=bool(n.get('parents')), finalists=bool(n.get('finalists')), top_cap=int(n.get('top_cap') or 0))


def niche_order(rows, tier=lambda r: 0, cap=None):                                 # A86-S5
    """Round-robin order of an already ranked list of result rows (r['strategy']): inside each consecutive block of
    equal tier (activation tier), the best row of each niche (niches in order of their best row), then each niche's
    2nd best, ...; cap = max rows per niche (None = keep all). Pure and deterministic (no RNG); the input order is the
    only tie-break."""
    out, i = [], 0
    while i < len(rows):
        j = i
        while j < len(rows) and tier(rows[j]) == tier(rows[i]):
            j += 1
        buckets = {}
        for r in rows[i:j]:
            buckets.setdefault(niche(r['strategy']), []).append(r)
        lim = max(len(b) for b in buckets.values()) if cap is None else int(cap)
        for depth in range(lim):
            for b in buckets.values():
                if depth < len(b):
                    out.append(b[depth])
        i = j
    return out


def capped_top(final, k, cap):                                                     # A86-S5
    """(niche-capped top k, unchanged global top k) from the final list (final order; eligibility = the filter of
    the a75ss.main TOP writer: not comparator, not banned, activated)."""
    elig = [r for r in final if not r['comparator'] and not r['banned'] and A.activated(r)]
    return niche_order(elig, cap=cap)[:int(k)], elig[:int(k)]


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
    nch = niching_cfg(cfg)                                                             # A86-S5
    nlog = dict(gen0_niches=None, parent_niches=[], finalist_niches=None)              # A86-S5

    def act(r):
        return int(A.activated(r)) if act_first else 0
    rng = random.Random(cfg['seed'])
    lam, groups = float(cfg['lam']), bool(cfg['fairness'])
    sk = dict(fair_tol=float(cfg['fair_tol']), bound_s=float(cfg['bound_s']), bound_stat=cfg['bound_stat'],
              fairer_total=cfg['fairer_total'], harm_cal=cfg['harm_cal'])
    dense, hold_only = bool(cfg['dense']), bool(cfg['hold_only'])
    screen, pop, rc = initial_population(rng, states, cfg, flt, extra)                 # A85-S4 (same draws)
    flt.generated.append(len(pop))                                                     # A85-S4
    if nch:                                                                            # A86-S5
        labels = [niche_label(s) for s in pop]                                         # A86-S5
        nlog['gen0_niches'] = {x: labels.count(x) for x in sorted(set(labels))}        # A86-S5
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

    def top(arch, a, n, rr=False):                                                     # A86-S5 (rr arg)
        xs = sorted(arch.values(), key=SS.rank_key(a))                                 # A86-S5
        if rr:                                                                         # A86-S5
            xs = niche_order(xs, tier=lambda r: -act(r))                               # A86-S5
        return xs[:n]                                                                  # A86-S5

    rr_par = bool(nch and nch['parents'])                                              # A86-S5
    rr_fin = bool(nch and nch['finalists'])                                            # A86-S5
    run(pop, screen, cfg['K_screen'])
    for _ in range(int(cfg['generations'])):
        ranked = sorted(evaluated.values(),
                        key=lambda r: (act(r), r['score'] + SS.NOV_W * min(SS.NOV_CAP, r.get('novelty', 0.0))),
                        reverse=True)
        if rr_par:                                                                     # A86-S5
            ranked = niche_order(ranked, tier=act)                                     # A86-S5
        parents = [r['strategy'] for r in ranked[:int(cfg['n_elite'])]]
        for a in SS.ARCHIVES:
            parents += [r['strategy'] for r in top(archives[a], a, int(cfg['n_elite']) if a != 'fair'
                                                   else int(cfg['n_elite']) // 2, rr_par)]  # A86-S5 (rr_par)
        if nch:                                                                        # A86-S5
            nlog['parent_niches'].append(sorted({niche_label(p) for p in parents}))    # A86-S5
        parents += extra
        kids = [flt.child(rng, parents, cfg, dense, hold_only)                         # A85-S4
                for _j in range(int(cfg['n_children']))]
        kids += [flt.random(rng, cfg, dense, hold_only, rc)                            # A85-S4
                 for _j in range(int(cfg['n_immigrants']))]
        flt.generated.append(len(kids))                                                # A85-S4
        run(kids, screen, cfg['K_screen'])
    nf = int(cfg['finalists'])
    fin = {}
    fin_rank = sorted(evaluated.values(), key=lambda r: (act(r), r['score']), reverse=True)  # A86-S5
    if rr_fin:                                                                         # A86-S5
        fin_rank = niche_order(fin_rank, tier=act)                                     # A86-S5
    for r in fin_rank[:nf]:                                                            # A86-S5
        fin[r['key']] = r
    for a in ('F', 'D', 'bounded', 'fairer', 'fair', 'delay'):
        for r in top(archives[a], a, nf, rr_fin):                                      # A86-S5 (rr_fin)
            if len(fin) < 4 * nf:
                fin.setdefault(r['key'], r)
    finals = [dict(f['strategy']) for f in fin.values()]
    if nch:                                                                            # A86-S5
        nlog['finalist_niches'] = sorted({niche_label(f) for f in finals})             # A86-S5
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
        out['a85'] = dict(flt.info(), a86s5_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())  # A86-S5
    if nch:                                                                            # A86-S5
        out['a86'] = dict(schema=NICHE_SCHEMA, definition=NICHE_DEFINITION, niching=nch, **nlog,  # A86-S5
                          final_niches={r['key']: niche_label(r['strategy']) for r in final},  # A86-S5
                          a86s5_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())  # A86-S5
    return out


def write_niche_top(out, cell, k, cap):                                            # A86-S5
    """Post-process after the unchanged a75ss.main TOP writer (module docstring, top_cap)."""
    out = Path(out)
    res = json.loads((out / f'STRATEGY_SEARCH_{cell}.json').read_text())
    capped, glob = capped_top(res['final'], k, cap)
    g_file = out / f'TOP_OFFER_CONDITIONS_{cell}.json'
    if [r['strategy_key'] for r in json.loads(g_file.read_text())] != [r['key'] for r in glob]:
        raise SystemExit('A86-S5: the a75ss TOP file differs from the recomputed global top (no niche file written)')
    g_new = out / f'TOP_OFFER_CONDITIONS_GLOBAL_UNCAPPED_{cell}.json'
    g_file.rename(g_new)
    rows = [dict(family=r['strategy']['name'], strategy_key=r['key'], score=r['score'], gain_vs_base=r['gain_vs_base'],
                 gain_vs_base_lcb=r['gain_vs_base_lcb'], feasible=r['feasible'], extended=A.is_extended(r['strategy']),
                 offer_condition=A.offer_condition(r['strategy'])) for r in capped]   # row keys as in a75ss.main
    c_file = out / f'TOP_OFFER_CONDITIONS_NICHECAP{int(cap)}_FOR_SUMO_{cell}.json'
    c_file.write_text(json.dumps(rows, indent=1))

    def lst(xs):
        return [dict(rank=i + 1, strategy_key=r['key'], family=r['strategy']['name'],
                     niche=niche_label(r['strategy']), score=r['score']) for i, r in enumerate(xs)]
    (out / f'NICHE_TOP_{cell}.json').write_text(json.dumps(dict(
        schema=NICHE_SCHEMA, definition=NICHE_DEFINITION, top_cap=int(cap), top_bank=int(k), for_sumo=c_file.name,
        global_uncapped=g_new.name, niche_capped=lst(capped), global_top=lst(glob),
        n_niches_capped=len({niche(r['strategy']) for r in capped}),
        n_niches_global=len({niche(r['strategy']) for r in glob})), indent=1))
    return capped, glob


def main(argv=None):
    """a75ss.main unchanged (checkpoint pin, outputs, TOP writer), routed through the S5 search_cell; then, when the
    cfg sets niching.top_cap > 0, the niche-capped TOP file (write_niche_top)."""
    A.search_cell = search_cell                                                        # A85-S4
    ap = argparse.ArgumentParser(add_help=False)                                       # A86-S5
    for o in ('--config', '--out', '--cell'):                                          # A86-S5
        ap.add_argument(o)                                                             # A86-S5
    ap.add_argument('--top-bank', type=int, default=12)                                # A86-S5 (a75ss default)
    a, _ = ap.parse_known_args(argv)                                                   # A86-S5 (None = sys.argv)
    ret = A.main(argv)
    raw = json.loads(Path(a.config).read_text(encoding='utf-8')) if a.config else {}  # A86-S5
    nch = niching_cfg(raw.get('config', raw))                                          # A86-S5
    if nch and nch['top_cap'] > 0:                                                     # A86-S5
        write_niche_top(a.out, a.cell, a.top_bank, nch['top_cap'])                     # A86-S5
    return ret                                                                         # A86-S5


if __name__ == '__main__':
    main()
