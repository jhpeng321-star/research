"""E1 stage 1: the fixed candidate pool of each search seed (zero SUMO, zero WM).

For every search seed: 120 distinct legal hold-only plan descriptors, 30 per group size 3/4/5/6, drawn uniformly
from the descriptor space the registered search and executor support (strategy_search.random_strategy with
hold_only + unconditional, on cfg['search_grid']): role slots, target speed, decel limit, hold, onset, zone, spread;
trigger 'wall' with the search's own probability 0.4 for 3-role plans (a wall has at most three roles), else
'lanes'; cooldown = max(onset + hold) (the search's rule for a random plan). Stratification = round-robin over the
sizes that still have quota, so the generation order interleaves sizes.

Reads ONLY the search config (the grid). Never reads a WM score, a search output or a SUMO result.

usage:  e1_pool.py --cfg SEARCH_CFG.json --out OUT_DIR [--seeds 11 23 ...] [--code-root TREE] [--dry-run]
resume: an existing identical POOL file is kept; a different one is refused (never overwritten).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cvlib as L  # noqa: E402

STREAM = 'claim_validation.e1_pool.v1'
MAX_ATTEMPTS = 100000


def draw(rng, n, grid):
    """One uniform draw of an n-role hold-only plan (not yet checked)."""
    trigger = 'wall' if n <= 3 and rng.random() < L.WALL_P_SMALL else 'lanes'
    picks = rng.sample(L.WALL_SLOTS if trigger == 'wall' else L.SLOTS, n)
    roles = [L.make_role(ln, rk, rng.choice(grid['v_w']), rng.choice(grid['dmax']), rng.choice(grid['hold_s']),
                         rng.choice(grid['onset_s'])) for ln, rk in picks]
    return L.make_strategy('', trigger, rng.choice(L.ZONES), rng.choice(L.SPREADS), roles)


def generate_pool(seed, grid, sizes=L.SIZES, per_size=L.PER_SIZE):
    """-> dict(pool=[entry], attempts=[row]). Deterministic in (seed, grid, sizes, per_size)."""
    rng = L.stable_rng(STREAM, seed)
    quota = {n: per_size for n in sizes}
    seen, pool, attempts = set(), [], []
    while any(quota.values()):
        for n in sizes:
            if not quota[n]:
                continue
            if len(attempts) >= MAX_ATTEMPTS:
                raise L.Refused(f'seed {seed}: pool not filled after {MAX_ATTEMPTS} draws')
            s = draw(rng, n, grid)
            key, h = L.strategy_key(s), L.desc_hash(s)
            s['name'] = f'e1s{seed}_{key}'
            why = L.static_legality(s, grid, min(sizes), max(sizes))
            if why is None and h in seen:
                why = 'duplicate'
            attempts.append(dict(attempt=len(attempts), size=n, desc_hash=h, key=key,
                                 status='accepted' if why is None else 'rejected', reason=why))
            if why is not None:
                continue
            seen.add(h)
            quota[n] -= 1
            pool.append(dict(index=len(pool), attempt=len(attempts)-1, desc_hash=h, key=key, size=n,
                             category=L.category(s), strategy=s))
    return dict(pool=pool, attempts=attempts)


def pool_doc(seed, cfg_path, cfg):
    grid = L.load_grid(cfg)
    got = generate_pool(seed, grid)
    rej = {}
    for a in got['attempts']:
        if a['reason']:
            rej[a['reason']] = rej.get(a['reason'], 0)+1
    cat = {}
    for e in got['pool']:
        cat[e['category']] = cat.get(e['category'], 0)+1
    return L.seal(dict(
        schema='claim_validation.e1.pool.v1', run_id=L.RUN_ID, cell=L.CELL, search_seed=int(seed), stream=STREAM,
        grid=grid, cfg_sha256=L.sha_file(cfg_path), sizes=list(L.SIZES), per_size=L.PER_SIZE,
        rule=('uniform over the registered hold-only unconditional descriptor space, 30 per size, round-robin over '
              'sizes; static legality shared by both arms; de-duplicated by sha256 of the normalised descriptor'),
        inputs_read=['search config (grid only)'], n=len(got['pool']), n_attempts=len(got['attempts']),
        rejections=rej, category_counts=dict(sorted(cat.items())), pool=got['pool'], attempts=got['attempts']))


def pool_path(out, seed):
    return Path(out)/'e1'/'pool'/f'POOL_seed{int(seed)}.json'


def load_pool(out, seed):
    doc = L.verify_seal(L.read_json(pool_path(out, seed)), what=f'POOL seed {seed}')
    if doc['search_seed'] != int(seed) or doc['n'] != len(L.SIZES)*L.PER_SIZE:
        raise L.Refused(f'POOL seed {seed}: not the registered pool')
    return doc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cfg', required=True, help='the registered search config (only search_grid / onset_max read)')
    p.add_argument('--out', default=str(L.HERE/'out'))
    p.add_argument('--seeds', type=int, nargs='*', default=list(L.SEARCH_SEEDS))
    p.add_argument('--code-root', help='registered code tree: cross-check keys / bank docs against it')
    p.add_argument('--dry-run', action='store_true', help='generate and report; write nothing')
    a = p.parse_args(argv)
    cfg = L.read_json(a.cfg)
    cfg = cfg.get('config', cfg)                      # a STRATEGY_SEARCH output embeds its config under 'config'
    report = []
    for seed in a.seeds:
        doc = pool_doc(seed, a.cfg, cfg)
        sizes = {n: sum(1 for e in doc['pool'] if e['size'] == n) for n in L.SIZES}
        cc = L.crosscheck(a.code_root, [e['strategy'] for e in doc['pool']]) if a.code_root else None
        state = 'dry-run' if a.dry_run else L.write_once(pool_path(a.out, seed), L.text_of(doc))
        report.append(dict(seed=seed, n=doc['n'], sizes=sizes, attempts=doc['n_attempts'],
                           rejections=doc['rejections'], doc_sha256=doc['doc_sha256'], file=state,
                           crosscheck=None if cc is None else cc['status']))
    print(L.text_of(dict(stage='e1_pool', dry_run=a.dry_run, grid=L.load_grid(cfg), pools=report)), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
