"""A80 test fixture, zero SUMO, no WM: a SYNTHETIC TOP_OFFER_CONDITIONS_D2800_p60.json written with the a75ss writer.

Plans are drawn by the a75ss generator itself (install(A75B cfg), random_strategy / mutate, the search_cell settings
SEARCH_GRID / UNCOND / MIN_ROLES, then strategy_search.uncond) and written with a75ss.offer_condition / is_extended in
exactly the row layout of a75ss.main (family, strategy_key, score, gain_vs_base, gain_vs_base_lcb, feasible, extended,
offer_condition). Scores are FAKE (descending placeholders): the file only fixes the schema for the A80 driver tests.
The 12 rows are chosen to cover the generator's form space (wall + lanes triggers, gap / pulse / taper / dv_rel / zone
custom, a plain plan); selection is by structure only.

usage: python -I make_synth_top.py <code_v12r7b dir> <a75ship dir> <A75B_SEARCH_CFG.json> <out.json>
"""
import json
import os
import random
import sys


def main(code, ship, cfg_path, out):
    sys.path[:0] = [ship, code]
    os.chdir(code)
    import a75ss
    SS = a75ss.SS
    raw = json.load(open(cfg_path, encoding='utf-8'))
    cfg = dict(SS.DEFAULTS, **raw.get('config', raw))
    a75ss.install(cfg)
    SS.UNCOND, SS.MIN_ROLES = bool(cfg['unconditional']), int(cfg['min_roles'])
    SS.SEARCH_GRID, SS.ONSET_MAX = cfg.get('search_grid'), cfg.get('onset_max')
    rng = random.Random(20261180)
    rc = tuple(cfg['role_counts'])
    pool, seen = [], set()

    def take(s):
        s = SS.uncond(s)
        if s is None or len(s['roles']) < SS.MIN_ROLES or SS.banned(s):
            return
        k = SS.strategy_key(s)
        if k not in seen:
            seen.add(k)
            pool.append(s)
    for _ in range(400):
        take(SS.random_strategy(rng, cfg['max_roles'], True, True, rc))
    for _ in range(400):
        take(SS.mutate(rng, rng.choice(pool), cfg['p_mutate'], True, True))

    def has(s, k, v=None):
        return any(k in r and (v is None or r[k] == v) for r in s['roles'])
    wants = [lambda s: s['trigger'] == 'wall' and a75ss.is_extended(s) and has(s, 'start', 'gap'),
             lambda s: s['trigger'] == 'wall' and a75ss.is_extended(s) and has(s, 'pulse_on_s'),
             lambda s: s['trigger'] == 'lanes' and has(s, 'start', 'gap') and has(s, 'gap_release', 'clear'),
             lambda s: has(s, 'start', 'gap') and not has(s, 'gap_release'),
             lambda s: has(s, 'pulse_on_s') and has(s, 'taper_in_s'),
             lambda s: has(s, 'dv_rel') and not has(s, 'start', 'gap'),
             lambda s: s.get('zone_d') is not None,
             lambda s: s['zone'] in ('mid', 'merge') and not has(s, 'start', 'gap'),
             lambda s: has(s, 'taper_out_s'),
             lambda s: not a75ss.is_extended(s) and s['trigger'] == 'wall',
             lambda s: not a75ss.is_extended(s) and s['trigger'] == 'lanes',
             lambda s: a75ss.is_extended(s) and len({r['onset_s'] for r in s['roles']}) > 1]
    top, used = [], set()
    for w in wants:
        s = next((x for x in pool if w(x) and SS.strategy_key(x) not in used), None)
        if s is None:
            raise SystemExit('synthetic fixture: no pool plan for a wanted structure')
        used.add(SS.strategy_key(s))
        top.append(s)
    rows = [dict(strategy=s, key=SS.strategy_key(s), score=round(40.0 - 2.5*i, 2), gain_vs_base=round(60.0 - 3*i, 2),
                 gain_vs_base_lcb=round(40.0 - 2.5*i, 2), feasible=1.0) for i, s in enumerate(top)]
    # the a75ss.main writer, verbatim row layout
    text = json.dumps(
        [dict(family=r['strategy']['name'], strategy_key=r['key'], score=r['score'], gain_vs_base=r['gain_vs_base'],
              gain_vs_base_lcb=r['gain_vs_base_lcb'], feasible=r['feasible'], extended=a75ss.is_extended(r['strategy']),
              offer_condition=a75ss.offer_condition(r['strategy'])) for r in rows], indent=1)
    with open(out, 'x', encoding='utf-8', newline='\n') as f:
        f.write(text)
    print(out, len(rows), 'rows')


if __name__ == '__main__':
    main(*sys.argv[1:5])
