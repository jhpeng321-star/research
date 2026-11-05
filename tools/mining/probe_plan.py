"""Cooperation probe planner: slates + WM queries per context, and screening jobs from WM ranks.

plan <setting> <ctx_dir>...   -> jobs/<setting>/<sid>.slate.json + wmq/<setting>__<sid>__<k>.query.json
screen <setting> <top_ctx> <out_jobs.json> [rank_dir]
    -> one screening job per ranked context (seed 1 only): its WM-top group by predicted synergy
       (ties: saving), contexts ordered by that predicted synergy; at most <top_ctx> contexts.
"""
import json
import os
from pathlib import Path
import sys

from tools.mining import probe
from tools.mining.probe_wmq import write_wmq

ROOT = probe.ROOT


def plan(setting, ctx_dirs):
    from tools.mining.mine_verify import policy_rows
    out = []
    for d in ctx_dirs:
        d = Path(d)
        ctx = json.loads((d/'CONTEXT.json').read_text())
        sid = ctx['state_id']
        slate = probe.slate(ctx)
        g5 = []
        if setting in ('hp', 'lowcoop'):
            prow, ctx['_cmds'] = policy_rows(d, int(ctx['t0']), 130)
            ctx['_rows'] = prow
            try:
                g5, notes = probe.g5_slate(ctx, prow)
            except Exception as e:
                g5, notes = [], [repr(e)]
            g5 = [p for p in g5 if p.get('g5_move') and p['family'] in ('open_gap_merge', 'yield_lane')]
        jd = ROOT/'jobs'/setting
        jd.mkdir(parents=True, exist_ok=True)
        (jd/f'{sid}.slate.json').write_text(json.dumps(dict(ctx_dir=str(d), plans=slate+g5)))
        qs = []
        if slate:
            qs.append(write_wmq(d, slate, ROOT/'wmq'/f'{setting}__{sid}__E.query.json'))
        pairs = {}
        for p in g5:
            pairs.setdefault(tuple(p['members']), []).append(p)
        for i, (pair, ps) in enumerate(sorted(pairs.items())):
            try:
                qs.append(write_wmq(d, ps, ROOT/'wmq'/f'{setting}__{sid}__G{i}.query.json'))
            except Exception as e:
                print(json.dumps(dict(sid=sid, pair=pair, error=repr(e))), flush=True)
        out.append(dict(sid=sid, n_slate=len(slate), n_g5=len(g5), queries=[str(q) for q in qs]))
        print(json.dumps(out[-1]), flush=True)
    return out


def screen(setting, top_ctx, out_path, rank_dir=None, seeds=(1,), skip_tags=()):
    rank_dir = Path(rank_dir or ROOT/'wmrank')
    best = []
    for slate_path in sorted((ROOT/'jobs'/setting).glob('*.slate.json')):
        sid = slate_path.name[:-len('.slate.json')]
        s = json.loads(slate_path.read_text())
        plans = {p['id']: p for p in s['plans']}
        ranked = []
        for r in rank_dir.glob(f'{setting}__{sid}__*.rank.json'):
            ranked += json.loads(r.read_text())['ranked']
        if not ranked:
            continue
        ranked.sort(key=lambda r: (-r['wm_mean_synergy'], -r['wm_saving_vs_policy']))
        top = ranked[0]
        best.append((top['wm_mean_synergy'], top['wm_saving_vs_policy'], sid, s['ctx_dir'], plans[top['plan_id']],
                     top))
    best.sort(key=lambda x: (-x[0], -x[1]))
    jobs = []
    for syn, sav, sid, ctx_dir, p, top in best[:int(top_ctx)]:
        tag = f'{setting}_scr_{sid}_{p["id"][:60]}'.replace('/', '_')
        if tag in skip_tags:
            continue
        jobs.append(dict(ctx_dir=ctx_dir, plans=[p], seeds=list(seeds), tag=tag,
                         wm=dict(mean_synergy=syn, saving=sav, draws=top.get('wm_synergy_draws'))))
    Path(out_path).write_text(json.dumps(jobs))
    return [(j['tag'], j['wm']['mean_synergy'], j['wm']['saving']) for j in jobs]


if __name__ == '__main__':
    if sys.argv[1] == 'plan':
        plan(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == 'screen':
        print(json.dumps(screen(sys.argv[2], int(sys.argv[3]), sys.argv[4],
                                sys.argv[5] if len(sys.argv) > 5 else None), indent=1))


def auto(setting, cap, max_launch=6, min_syn=None):
    """Launch seed-1 screening per newly WM-ranked query (its top predicted-synergy group)."""
    import subprocess
    spent = probe.spent()
    used = sum(v for k, v in spent.items() if k.split('_')[0] == setting)
    launched = []
    rank_dir = ROOT/'wmrank'
    for r in sorted(rank_dir.glob(f'{setting}__*.rank.json'), key=lambda p: p.stat().st_mtime):
        if len(launched) >= max_launch or used >= cap:
            break
        b = r.name.replace('.rank.json', '')
        _, sid, k = b.split('__')
        tag = f'{setting}_scr_{sid}__{k}' + ('v' if k.startswith('G') else '')
        old_tag = f'{setting}_scr_{sid}'
        jp = ROOT/'jobs'/f'auto_{tag}.json'
        if jp.exists() or (ROOT/'claims'/(tag+'.claim')).exists():
            continue
        if k == 'E' and ((ROOT/'claims'/(old_tag+'.claim')).exists() or (ROOT/'jobs'/f'auto_{old_tag}.json').exists()):
            continue
        x = json.loads(r.read_text())
        q = json.loads((ROOT/'wmq'/(b+'.query.json')).read_text())
        if x.get('n_actions') != len(q['actions']) or not x['ranked']:
            continue
        s = json.loads((ROOT/'jobs'/setting/f'{sid}.slate.json').read_text())
        plans = {p['id']: p for p in s['plans']}
        ranked = sorted(x['ranked'], key=lambda r: (-r['wm_mean_synergy'], -r['wm_saving_vs_policy']))
        top = ranked[0]
        p = plans[top['plan_id']]
        job = [dict(ctx_dir=s['ctx_dir'], plans=[p], seeds=[1], tag=tag,
                    wm=dict(mean_synergy=top['wm_mean_synergy'], saving=top['wm_saving_vs_policy'],
                            best_solo_saving=top.get('wm_best_solo_saving'), plan_id=p['id']))]
        jp.write_text(json.dumps(job))
        env = dict(os.environ, PROBE_WORKERS=str(len(p['members'])+2))
        if setting == 'lowcoop':
            env['PROBE_HDV_VARIANT'] = 'lowcoop_v1'
        else:
            env.pop('PROBE_HDV_VARIANT', None)
        if k.startswith('G'):
            env['C4NM_V2V'] = '1'     # G5 lane moves need the single-writer V2V lane runtime (as mining)
        log = open(ROOT/'logs'/f'auto_{tag}.log', 'w')
        pr = subprocess.Popen(['nice', '-n', '19', sys.executable, '-B', '-u', '-m', 'tools.mining.probe', 'verify',
                               setting, str(jp)], env=env, stdout=log, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(ROOT))
        with (ROOT/'PIDS.txt').open('a') as f:
            f.write(f'auto {tag} {pr.pid}' + chr(10))
        used += (len(p['members'])+2)*(int(json.loads((Path(s['ctx_dir'])/'CONTEXT.json').read_text())['t0'])
                                       + p['duration_s']+120)
        launched.append((tag, round(top['wm_mean_synergy'], 2), round(top['wm_saving_vs_policy'], 2), pr.pid))
    print(json.dumps(dict(setting=setting, used=used, launched=launched)))


if __name__ == '__main__' and sys.argv[1] == 'auto':
    auto(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 6)
