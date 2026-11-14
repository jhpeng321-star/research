"""E1 stage 3: verification job specs (zero SUMO; this file launches nothing).

Per search seed: the union of the two arms' top-six candidates, each to be run ALONE (runner arm A37_PLAN: the
one-row bank of that plan, start-whenever-feasible executor) on that seed's 24 verification worlds, paired with the
base policy U0 on the same worlds. A candidate shared by both arms is one job set: the cache key is
(descriptor hash, world block, world index, versions), so it runs once and neither arm's order is touched.

  plan   write out/e1/banks/E1_CANDS_seed<N>.json (candidate bank per seed), out/e1/verify/VERIFY_PLAN.json
         (waves + cache keys + budget) ; --seed-blocks binds registered world seeds (optional until MAIN registers)

The world seeds are NOT chosen here: --seed-blocks is a MAIN-registered JSON {"<search seed>": {"start": S, "n": 24}}.
Without it the plan is symbolic (world_index 0..23 of block 'e1v_seed<N>') and box_driver.py refuses to run it.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cvlib as L  # noqa: E402
import e1_order as E  # noqa: E402

#: what a cached verification result depends on besides the plan and the world (bound to digests on the box)
VERSIONS = dict(cell=L.CELL, u0_sha256=L.U0_MAPPO_SHA256, executor='a37_cellbank.CellBankCore/A37_PLAN',
                executor_digest='bound on the box (a37_digest; cvlib.crosscheck)', horizon_s=L.HORIZON_S,
                bank_schema='coopmwm.a37_cellbank.v1.bank')


def cache_key(desc_hash, block, world_index, versions=None):
    return L.sha_json(dict(desc_hash=desc_hash, block=block, world_index=int(world_index),
                           versions=versions or VERSIONS))


def union_candidates(orders, pool_by_hash):
    """orders = {arm: [top hashes]} -> candidates in first-appearance order (WM top six, then Random's new ones),
    each with the arms / ranks that selected it. The order here is only a job order."""
    out, at = [], {}
    for arm in E.ARMS:
        for rank, h in enumerate(orders[arm], 1):
            if h not in at:
                e = pool_by_hash[h]
                at[h] = len(out)
                out.append(dict(desc_hash=h, key=e['key'], family=e['strategy']['name'], size=e['size'],
                                category=e['category'], strategy=e['strategy'], selected_by={}))
            out[at[h]]['selected_by'][arm] = rank
    return out


def seed_plan(out, seed, worlds=None):
    pool = L.verify_seal(L.read_json(E.pool_file(out, seed)), what=f'POOL seed {seed}')
    by_hash = {e['desc_hash']: e for e in pool['pool']}
    orders, shas = {}, {}
    for arm in E.ARMS:
        d = L.verify_seal(L.read_json(E.order_path(out, arm, seed)), what=f'ORDER {arm} seed {seed}')
        if d['pool_doc_sha256'] != pool['doc_sha256']:
            raise L.Refused(f'ORDER {arm} seed {seed}: ordered another pool')
        orders[arm], shas[arm] = d['top'], d['doc_sha256']
    cands = union_candidates(orders, by_hash)
    bank = L.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0) for c in cands])
    block = f'e1v_seed{int(seed)}'
    waves = [dict(wave=f'E1V{seed}_U0', kind='U0', block=block, n_worlds=L.N_VERIFY_WORLDS,
                  cache_keys=[cache_key('U0', block, i) for i in range(L.N_VERIFY_WORLDS)])]
    for c in cands:
        waves.append(dict(wave=f'E1V{seed}_{c["key"]}', kind='PLAN', block=block, n_worlds=L.N_VERIFY_WORLDS,
                          bank=f'E1_CANDS_seed{int(seed)}.json', family=c['family'], desc_hash=c['desc_hash'],
                          cache_keys=[cache_key(c['desc_hash'], block, i) for i in range(L.N_VERIFY_WORLDS)]))
    return dict(search_seed=int(seed), block=block, world_seeds=worlds, pool_doc_sha256=pool['doc_sha256'],
                order_doc_sha256=shas, top={a: orders[a] for a in E.ARMS},
                candidates=[{k: v for k, v in c.items() if k != 'strategy'} for c in cands],
                n_unique=len(cands), n_shared=sum(1 for c in cands if len(c['selected_by']) == 2),
                bank=dict(name=f'E1_CANDS_seed{int(seed)}.json', sha256=L.sha_bytes(L.text_of(bank).encode())),
                waves=waves), bank, cands


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('plan',))
    p.add_argument('--out', default=str(L.HERE/'out'))
    p.add_argument('--seeds', type=int, nargs='*', default=list(L.SEARCH_SEEDS))
    p.add_argument('--seed-blocks', help='MAIN-registered world blocks (never chosen by this code)')
    p.add_argument('--code-root', help='cross-check the candidate banks against the registered executor')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    blocks = L.load_seed_blocks(a.seed_blocks, a.seeds, L.N_VERIFY_WORLDS) if a.seed_blocks else {}
    seeds, rep, episodes = [], [], 0
    for seed in a.seeds:
        sp, bank, cands = seed_plan(a.out, seed, blocks.get(str(seed)))
        cc = L.crosscheck(a.code_root, [c['strategy'] for c in cands]) if a.code_root else None
        state = 'dry-run' if a.dry_run else L.write_once(Path(a.out)/'e1'/'banks'/sp['bank']['name'], L.text_of(bank))
        n = sum(w['n_worlds'] for w in sp['waves'])
        episodes += n
        seeds.append(sp)
        rep.append(dict(seed=seed, unique_candidates=sp['n_unique'], shared_by_both_arms=sp['n_shared'], episodes=n,
                        bank=state, crosscheck=None if cc is None else cc['status']))
    doc = L.seal(dict(schema='claim_validation.e1.verify_plan.v1', run_id=L.RUN_ID, cell=L.CELL, versions=VERSIONS,
                      reference='U0 = frozen speed-only base MAPPO d9042cc5 (not MAPPO60C)',
                      symbolic=not blocks, seeds=seeds, episodes=episodes, sim_s=episodes*L.HORIZON_S,
                      budget_note='upper bound 6 x 24 x (12 + 1) x 1200 = 2,246,400 sim-s when no candidate is shared'))
    state = 'dry-run' if a.dry_run else L.write_once(Path(a.out)/'e1'/'verify'/'VERIFY_PLAN.json', L.text_of(doc))
    print(L.text_of(dict(stage='e1_verify plan', dry_run=a.dry_run, symbolic=not blocks, seeds=rep,
                         episodes=episodes, sim_s=episodes*L.HORIZON_S, plan=state,
                         doc_sha256=doc['doc_sha256'])), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
