"""HICAV probe summary (zero simulation): MAPPO-vs-native sanity, per-share wall frequency/synergy,
4/6-slot observability, safety counters. Reads only this probe's own receipts/results (no truth files)."""
import json
import os
from pathlib import Path
import statistics as st
import sys

ROOT = Path(os.environ.get('HICAV_ROOT', '/home/ubuntu/hicavprobe20261106'))
KEYS = ('team_delay_vs', 'road_delay_vs', 'source_wait_vs', 'collisions', 'teleports', 'ttc_below_3_vs', 'hard_brake')


def jl(name):
    p = ROOT/'results'/name
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def share_of(w):
    return int(round(float(w['penetration'])*100))


def slots_needed(rows, members, range_m=100.0):
    cav = {r['vid']: r for r in rows if r['is_cav']}
    best = None
    for m in members:
        if m not in cav:
            continue
        me = cav[m]
        order = [r['vid'] for r in sorted([r for r in cav.values() if r['vid'] != m and abs(r['s_m']-me['s_m']) <= range_m],
                                          key=lambda r: (abs(r['s_m']-me['s_m']), r['vid']))]
        if not all(x in order for x in members if x != m):
            continue
        need = 1+max([order.index(x)+1 for x in members if x != m] or [0])
        best = need if best is None else min(best, need)
    return best


def span(rows, members):
    s = [r['s_m'] for r in rows if r['vid'] in members]
    return max(s)-min(s) if s else None


def receipt_window(work, vid):
    p = Path(work)/'verifications'/f'{vid}.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    out = dict(mappo_vs_native={}, mappo_episodes={}, shares={})
    # ---- MAPPO arm (collection episodes) and native
    coll = {}
    for d in sorted((ROOT/'collect').glob('*_a2')):
        p = d/'COLLECTION_RECEIPT.json'
        if p.exists():
            r = json.loads(p.read_text())
            coll[r['world']['seed']] = r
    nat = {}
    for d in sorted((ROOT/'native').glob('*')):
        p = d/'NATIVE_RECEIPT.json'
        if p.exists():
            r = json.loads(p.read_text())
            nat[r['world']['seed']] = r
    for seed, r in sorted(coll.items()):
        sh = share_of(r['world'])
        m = r['metrics'].get('900') or {}
        out['mappo_episodes'].setdefault(sh, []).append(dict(
            seed=seed, actual_sim_s=r['actual_sim_s'], **{k: m.get(k) for k in KEYS},
            override_fraction=(r['overrides']/r['requests']) if r.get('requests') else None,
            max_cavs=max([c['n_cav_rows'] for c in r.get('cav_counts', [])] or [None])))
    for seed, n in sorted(nat.items()):
        sh = share_of(n['world'])
        row = dict(seed=seed)
        for h in ('600', '900'):
            mm, nn = (coll.get(seed) or {}).get('metrics', {}).get(h), n['metrics'].get(h)
            if mm and nn:
                row[h] = dict(mappo={k: mm[k] for k in KEYS}, native={k: nn[k] for k in KEYS},
                              diff_mappo_minus_native={k: mm[k]-nn[k] for k in KEYS})
        out['mappo_vs_native'].setdefault(sh, []).append(row)
    # ---- walls
    A, B, C = jl('A.jsonl'), jl('B.jsonl'), jl('C.jsonl')
    ctx_cache = {}

    def ctx(row):
        d = row.get('ctx_dir') or str(ROOT/'collect'/row['state_id'].rsplit('_t', 1)[0]/f"ctx_t{row['t0']}")
        if d not in ctx_cache:
            ctx_cache[d] = json.loads((Path(d)/'CONTEXT.json').read_text())
        return ctx_cache[d]
    for sh in sorted({share_of(r['world']) for r in A}):
        a = [r for r in A if share_of(r['world']) == sh]
        moments = sorted({r['state_id'] for r in a})
        walls = [r for r in a if 'group_id' in r]
        errors = [r for r in a if 'group_id' not in r]
        b = {(r['state_id'], r['group_id']): r for r in B if share_of(r['world']) == sh}
        c = [r for r in C if share_of(r['world']) == sh]
        tried = []
        for r in walls:
            cx = ctx(r)
            sl = slots_needed(cx['rows'], r['members'])
            br = b.get((r['state_id'], r['group_id']))
            rec = receipt_window(r['work'], r['verification_id']) if r.get('work') else None
            ttc = None
            if rec and rec.get('branches'):
                g = rec['branches'][r['branch_index']]
                p0 = rec['branches'][0]
                if g.get('window') and p0.get('window') and 'ttc_below_3_vs' in g['window']:
                    ttc = g['window']['ttc_below_3_vs']-p0['window']['ttc_below_3_vs']
            tried.append(dict(state_id=r['state_id'], t0=r['t0'], k=r['n_members'], members=r['members'],
                              J_policy=r['J_policy'], group_adv=r.get('group_adv'), adv_lb95=r.get('adv_lb95'),
                              beat_policy=(r.get('adv_lb95') or -1) > 0,
                              synergy=(br or {}).get('synergy'), synergy_lb95=(br or {}).get('synergy_lb95'),
                              best_solo_adv=(br or {}).get('best_solo_adv'),
                              synergy_status=(br or {}).get('synergy_status'),
                              policy_reproduced=(br or {}).get('policy_reproduced'),
                              admitted_seed1=(br or {}).get('synergy_status') == 'admitted',
                              slots_needed=sl, fits4=sl is not None and sl <= 4, fits6=sl is not None and sl <= 6,
                              span_m=span(cx['rows'], r['members']),
                              safety_delta=dict(r.get('safety') or {}, ttc_below_3_vs=ttc)))
        adm_m = sorted({t['state_id'] for t in tried if t['admitted_seed1']})
        beat_m = sorted({t['state_id'] for t in tried if t['beat_policy']})
        per_seed = {}
        for t in tried:
            if t['admitted_seed1']:
                per_seed.setdefault((t['state_id'], t['k']), {})[1] = True
        for r in c:
            per_seed.setdefault((r['state_id'], r['n_members']), {})[r['seed_index']] = r.get('synergy_status') == 'admitted'
        out['shares'][sh] = dict(
            moments_screened=len(moments), walls_tried=len(walls), screen_errors=[e.get('error') for e in errors],
            moments_with_wall_beating_policy=len(beat_m), walls_beating_policy=sum(t['beat_policy'] for t in tried),
            walls_synergy_measured=sum(t['synergy_lb95'] is not None for t in tried),
            moments_with_admissible_wall_seed1=len(adm_m), admissible_moments=adm_m,
            walls_admitted_seed1=sum(t['admitted_seed1'] for t in tried),
            fits4_all=sum(t['fits4'] for t in tried), fits6_all=sum(t['fits6'] for t in tried),
            fits_by_k={k: dict(n=sum(t['k'] == k for t in tried), fits4=sum(t['k'] == k and t['fits4'] for t in tried),
                               fits6=sum(t['k'] == k and t['fits6'] for t in tried)) for k in (3, 4, 5, 6)},
            replicate_rows=[dict(state_id=r['state_id'], k=r['n_members'], seed=r['seed_index'],
                                 group_adv=r.get('group_adv'), adv_lb95=r.get('adv_lb95'), synergy=r.get('synergy'),
                                 synergy_lb95=r.get('synergy_lb95'), status=r.get('synergy_status'),
                                 safety=r.get('safety')) for r in c],
            per_seed_admission={f'{k[0]}|k{k[1]}': v for k, v in per_seed.items()},
            walls=tried)
    out['ledger'] = dict(sim_s=sum(json.loads(x)['sim_s'] for x in (ROOT/'CHARGES.jsonl').read_text().splitlines() if x.strip()),
                         rows=sum(1 for x in (ROOT/'CHARGES.jsonl').read_text().splitlines() if x.strip()))
    json.dump(out, sys.stdout, indent=1, default=str)


if __name__ == '__main__':
    main()
