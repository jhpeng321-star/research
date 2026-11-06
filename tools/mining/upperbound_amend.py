"""Upper-bound amendment A1/A2 (registered in AMENDMENT_A1A2.json before execution; same ledger/cap).

A1 (synergy scan, search seed): for each ACTING situation, admitted searched walls other than the chosen move:
all whose member solos already exist (free) + the best of each further distinct member set (solos run).
A2 (transfer): held-out CRN seed 4, policy + chosen move only (no solos), acting situations.
Both in fixed world-seed order and only while the whole unit fits above the 3,000 sim-s reserve.
"""
import json
import time

from tools.mining import upperbound as ub


def load(sit):
    for row in ub._rows('branches.jsonl'):
        if row['sid'] == sit.sid:
            sit.res.setdefault(row['seed'], {})[row['cand']] = row
    for x in ub._rows('plans.jsonl'):
        if x['sid'] == sit.sid and x['id'].startswith('ub_wall_'):
            q = x['ub']
            p = ub.wall_plan(sit.ctx, tuple(x['members']), q['onset'], q['hold'], q['depth'], q['release'])
            assert p['id'] == x['id']
            sit.plans[p['id']] = p


def a1(eng, sit, best_id):
    """Synergy scan on the search seed: among searched multi-CAV walls admitted vs base (adv LB95 > 0),
    other than the chosen move: (i) every one whose member solos already exist (free), and (ii) the
    highest-adv candidate of each further DISTINCT member set (its member solos run, shared by
    member+timing). Situation skipped as a unit if its new solos do not fit above the reserve."""
    res = sit.res[1]
    pol = res['policy']
    sets = {frozenset(sit.plans[best_id]['members'])}
    pos = sorted([c for c in res if c in sit.plans and c != best_id and res[c]['J'] is not None
                  and pol['J']-res[c]['J'] > 0], key=lambda c: (res[c]['J'], c))
    chosen, need = [], {}
    for c in pos:
        a = ub.stats(pol, res[c], sit.window)
        if a['lb95'] <= 0:
            continue
        new = [x for x in ub.solo_plans(sit.plans[c]) if x['id'] not in res and x['id'] not in need]
        key = frozenset(sit.plans[c]['members'])
        if new and key in sets:
            continue
        sets.add(key)
        chosen.append((c, a))
        for x in new:
            need[x['id']] = x
    cost = (sit.t0+ub.SPAN)*len(need)
    if ub.CAP-ub.spent()-ub.RESERVE < cost:
        return [dict(sid=sit.sid, amendment='A1', status='skipped_budget', need=cost)]
    if need:
        eng.batch(sit, 1, 'a1_solo', list(need.values()), need_policy=False)
    res = sit.res[1]
    out = []
    for c, a in chosen:
        g = sit.plans[c]
        ids = [x['id'] for x in ub.solo_plans(g)]
        sj = {i: res[i]['J'] for i in ids}
        row = dict(sid=sit.sid, amendment='A1', seed=1, group=c, n_members=len(g['members']), ub=g['ub'],
                   adv=pol['J']-res[c]['J'], adv_lb95=a['lb95'], solos={i: pol['J']-v for i, v in sj.items()},
                   observability=ub.observability(sit.ctx, g['members']), safety=ub.safety(pol, res[c]))
        if any(v is None for v in sj.values()):
            row['synergy_status'] = 'incomplete'
            out.append(row)
            continue
        b = min(sj, key=lambda k: (sj[k], k))
        st = ub.stats(res[b], res[c], sit.window)
        row.update(best_solo=b, best_solo_adv=pol['J']-sj[b], synergy=sj[b]-res[c]['J'], synergy_lb95=st['lb95'],
                   synergy_status='admitted' if st['lb95'] > 0 else 'not_admitted')
        out.append(row)
    return out


def a2(eng, sit, move):
    cost = 2*(sit.t0+ub.SPAN)
    if ub.CAP-ub.spent()-ub.RESERVE < cost:
        return dict(sid=sit.sid, amendment='A2', seed=4, status='skipped_budget', need=cost)
    row = ub.evaluate_seed(eng, sit, 4, move, 'a2_heldout', solos_if_admitted=False)
    row['amendment'] = 'A2'
    return row


def main():
    assert (ub.ROOT/'AMENDMENT_A1A2.json').exists() and (ub.ROOT/'DONE.json').exists()
    assert not (ub.ROOT/'AMEND_DONE.json').exists()
    sits_ = json.loads((ub.ROOT/'results'/'SITUATIONS.json').read_text())
    acting = json.loads((ub.ROOT/'results'/'ACTING.json').read_text())
    best = json.loads((ub.ROOT/'results'/'BEST.json').read_text())
    sits = [ub.Situation(d) for s, d in sorted(sits_['situations'].items(), key=lambda kv: int(kv[0]))]
    sits = [s for s in sits if s.sid in acting]
    for s in sits:
        load(s)
    eng = ub.Engine(ub.WORKERS)
    ub.log('amend_start', spent=ub.spent())
    for s in sits:                      # fixed order; each situation's solos run in parallel inside
        for row in a1(eng, s, best[s.sid]):
            ub._append(ub.ROOT/'results'/'amend_a1.jsonl', row)
    ub.log('a1_done', spent=ub.spent())
    from concurrent.futures import ThreadPoolExecutor
    plan_ok = [s for s in sits if best[s.sid] in s.plans]
    order, room = [], ub.CAP-ub.spent()-ub.RESERVE
    for s in plan_ok:
        if 2*(s.t0+ub.SPAN) <= room:
            order.append(s)
            room -= 2*(s.t0+ub.SPAN)
    with ThreadPoolExecutor(max_workers=max(1, len(order))) as tp:
        for row in tp.map(lambda s: a2(eng, s, best[s.sid]), order):
            ub._append(ub.ROOT/'results'/'amend_a2.jsonl', row)
    skipped = [s.sid for s in plan_ok if s not in order]
    ub.log('a2_done', spent=ub.spent(), skipped=skipped)
    ub._write(ub.ROOT/'AMEND_DONE.json', dict(unix=time.time(), spent=ub.spent(), a2_skipped=skipped))


if __name__ == '__main__':
    main()
