"""ON-BOX adapter (Box 3, Linux, inside the registered code tree): turns the neutral wave lists written by
e1_verify / e1_bank / e2_build / e2_replay into runner jobs of the REGISTERED executor and collects the results
back into the neutral JSON the local stages read. It adds no logic of its own: it is the A68 pattern
(ops/a68_research60c.core: a private clone of ops/a42_bigp60.py with amendment names replaced) minus the
MAPPO60C executor rebinding, because the reference of these experiments is U0 = the frozen base MAPPO.

STATUS: WRITTEN, NOT EXECUTED. It cannot be imported on the Windows build host (the tree is a sparse checkout there
and the driver needs fcntl / the runner). Before any charged episode run, in this order, on the box:
    box_driver.py crosscheck ...   (mirror == registered code)      box_driver.py dry ...   (specs build, zero SUMO)
A failure of either is a build defect to report, not something to work around.

It refuses to run until MAIN has registered the seed blocks: REGISTRATION.json (template: patches/) names, per
block, a setting that must exist in ops/nm_finaleval.SETTINGS with a pinned panel sha, and the runner ranges
attribute that must exist in tools/coopmwm/nm_eval_runner. One process = one block (nm_finaleval.apply_setting).

  crosscheck --registration R --banks-dir DIR...               mirror checks for every bank file found
  prepare    --registration R --block B --waves F --final-root ROOT --banks-dir DIR...   copy banks (write-once)
  dry        --registration R --block B --waves F --final-root ROOT                      specs + budget, zero SUMO
  run        ... --workers 46 --chunk 96                        charged; resumable (published episodes are kept)
  collect    ... --kind e1verify|e1confirm|e2main|e2discover|e2branch|raw --out FILE [--selection REPLAY_SELECTION]
--waves takes several documents: the shared confirmation block `conf` is run ONCE with FROZEN_BANKS.json and
E2_SELECTION.json together (one CONF_U0 wave); the Development block `dev` = existing setting a35_D2800_p60.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cvlib as L  # noqa: E402
import stats as S  # noqa: E402

WAVE_KEYS = ('waves', 'confirm_waves', 'discover_waves', 'tune_waves')


def find_waves(doc, block):
    """Every wave dict of ``block`` in a stage document (VERIFY_PLAN seeds[*].waves, FROZEN_BANKS confirm_waves,
    E2_MANIFEST waves, REPLAY_PLAN discover_waves, REPLAY_SELECTION waves), in document order."""
    out = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in WAVE_KEYS and isinstance(v, list):
                    out.extend(w for w in v if isinstance(w, dict) and w.get('block') == block)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(doc)
    if len({w['wave'] for w in out}) != len(out):
        raise L.Refused('duplicate wave names')
    return out


def merge_waves(docs, block):
    """Waves of one block from several stage documents (the shared confirmation block: FROZEN_BANKS + E2_SELECTION).
    A wave named in two documents must be identical (the shared CONF_U0) and is kept once."""
    out = {}
    for doc in docs:
        for w in find_waves(doc, block):
            if out.setdefault(w['wave'], w) != w:
                raise L.Refused(f'wave {w["wave"]} differs between documents')
    return list(out.values())


def world_indices(wave, n_panel):
    idx = list(wave['world_indices']) if 'world_indices' in wave else list(range(int(wave['n_worlds'])))
    if not idx or max(idx) >= n_panel or len(set(idx)) != len(idx):
        raise L.Refused(f'{wave["wave"]}: worlds {idx[:3]}.. do not fit the registered panel of {n_panel}')
    return idx


def core(reg, block, code_root):
    """The registered A42 driver code under the claim-validation names (A68 pattern)."""
    ops = Path(code_root)/'ops'
    for p in (str(ops), str(code_root), str(Path(code_root)/'pytools')):   # box layout: <src>/pytools/tools
        if p not in sys.path:
            sys.path.insert(0, p)
    import nm_finaleval as N
    import a50_fullp60 as A50
    b = reg['blocks'].get(block)
    if not b or not b.get('setting') or not b.get('panel_sha256'):
        raise L.Refused(f'block {block}: not registered (setting + pinned panel sha required)')
    if b['setting'] not in N.SETTINGS:
        raise L.Refused(f'setting {b["setting"]} is not in ops/nm_finaleval.SETTINGS (apply the registered table edit)')
    tag = f'{reg["job_prefix"]}{block.replace("_", "")}'
    over = dict(
        A42_PHASE=reg['phase'], A42_CAP_SIM_S=int(reg['cap_sim_s']), STOP_A42=f'STOP_{reg["label"]}',
        HOST_BASE=os.environ.get('CV_HOST_BASE', reg['host_base']),
        VERIFY_SETTING=b['setting'], CONFIRM_SETTING=b['setting'], BANK_DIR=reg.get('bank_dir', 'cv_banks'),
        IDENT_WAVE=f'{reg["label"]}_IDENT_UNUSED', AMENDMENT=str(reg['amendment']), LABEL=reg['label'],
        JOB_PREFIX=reg['job_prefix'], VERIFY_TAG=tag, CONFIRM_TAG=tag, IDENT_TAG=f'{tag}i',
        VERIFY_WAVE_PREFIX=f'{reg["label"]}V_', CONFIRM_WAVE_PREFIX=f'{reg["label"]}C_',
        RESULTS_FILE=f'{reg["label"]}_RESULTS.json', LOG_FILE=f'{reg["label"]}_LOG.jsonl',
        SCHEMA_PREFIX=f'coopmwm.{reg["job_prefix"]}', RUNNER_RANGES_ATTR=b['ranges_attr'],
        RESULT_LABEL=reg['result_label'], AUTHORITY=reg['authority'])
    D = A50._clone(f'cv_core_{block}', ops/'a42_bigp60.py', over)
    D.HOST_ROOT = f'{D.HOST_BASE}/root'
    D.SETTINGS = (b['setting'],)
    D.A42Runtime.phase = reg['phase']
    panel = D.setup(b['setting'])
    if panel['panel_sha256'] != b['panel_sha256'] or panel['cell']['cell_key'] != L.CELL:
        raise L.Refused(f'{b["setting"]}: not the registered panel')
    seeds = [r['world']['seed'] for r in panel['worlds']]
    L.check_world_seeds(seeds)
    if b.get('world_seeds') and list(b['world_seeds']) != seeds:
        raise L.Refused(f'{b["setting"]}: panel seeds differ from the registered block')
    return N, D, panel, tag


def specs_of(D, panel, root, wave, tag):
    """The runner specs of one neutral wave on its worlds (U0 binding untouched: reference = base MAPPO)."""
    idx = world_indices(wave, len(panel['worlds']))
    if wave['kind'] == 'U0':
        full = D.confirm_specs('U0', panel, root)
    else:
        entry = D._bank_entry(root, wave['bank'])
        if wave.get('bank_sha256') and entry['file_sha256'] != wave['bank_sha256']:
            raise L.Refused(f'{wave["wave"]}: bank file differs from the frozen sha')
        t = f'{tag}_{wave["wave"].lower()}'
        if wave['kind'] == 'PLAN':
            full = D._a37_specs(D.RUNNER_ARM['PLAN'], panel, root, entry, family=wave['family'], tag=t)
        elif wave['kind'] == 'BANK':
            full = D._a37_specs(D.RUNNER_ARM['BIG_BANK'], panel, root, entry, tag=t)
        else:
            raise L.Refused(f'{wave["wave"]}: unknown kind {wave["kind"]}')
    return [full[i] for i in idx]


def build(D, panel, root, waves, tag):
    return [(w['wave'], 'episode', specs_of(D, panel, root, w, tag)) for w in waves]


def stage_banks(D, root, waves, dirs):
    done = {}
    for w in waves:
        name = w.get('bank')
        if not name or name in done:
            continue
        src = next((Path(d)/name for d in dirs if (Path(d)/name).is_file()), None)
        if src is None:
            raise L.Refused(f'{name}: not found in {dirs}')
        if w.get('bank_sha256') and L.sha_file(src) != w['bank_sha256']:
            raise L.Refused(f'{name}: differs from the frozen sha')
        doc = L.read_json(src)
        L.crosscheck(str(Path(D.__file__).resolve().parents[1]),
                     [p['strategy'] for p in L.bank_strategies(doc)])
        D.C37.write_once(D.bank_path(root, name), src.read_text(encoding='utf-8'))
        done[name] = L.sha_file(src)
    return done


def collect(N, D, panel, root, waves, tag, selection=None):
    """Generic collection: per wave and world the total delay, safety counts and executor receipts; per non-U0 wave
    the registered comparison against the block's U0 wave on the common worlds (groups labelled by U0, CVaR90)."""
    index = {r['world_id']: i for i, r in enumerate(panel['worlds'])}
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    have, out = {}, dict(waves={}, compare={})
    for name, _, specs in build(D, panel, root, waves, tag):
        have[name], _src = D.C37._load(root, name, specs)
        out['waves'][name] = {
            w: dict(world_index=index[w], seed=seed[w], J=r['J_veh_s'], safety=r.get('safety'),
                    activations=int((r.get('a37') or {}).get('activations') or 0),
                    acts=[dict(t=a['t'], family=a['family'], members=a['members'], release=a.get('release'))
                          for a in (r.get('a37') or {}).get('acts') or []],
                    work_dir=r.get('work_dir', r['job_id']))
            for w, r in have[name].items()}
    u0 = next((w['wave'] for w in waves if w['kind'] == 'U0'), None)
    bootstrap = dict(panel['contract']['bootstrap'])
    if u0 is not None:
        for name in have:
            if name == u0:
                continue
            worlds = sorted(set(have[name]) & set(have[u0]), key=index.get)
            reports = D.C37._reports(root, {name: have[name], u0: have[u0]}, worlds)
            c = N._bank_compare(have, reports, worlds, name, u0, bootstrap)
            g = (c.get('groups') or {}).get('groups') or {}
            out['compare'][name] = dict(
                n_worlds=len(worlds), total_minus_ref=c.get('total_minus_ref'), tail_cvar90=c.get('tail_cvar90'),
                groups_all_pass=(c.get('groups') or {}).get('all_pass'),
                group_increase={k: v.get('increase_mean_s') for k, v in g.items()},
                hard_brake_vs={a: {w: reports[a][w]['accel']['hard_brake_vs'] for w in worlds} for a in (name, u0)})
    if selection is not None:
        for s in selection['states']:
            for name, rows in out['waves'].items():
                if s['world_id'] in rows:
                    log = Path(root)/'evaluation'/rows[s['world_id']]['work_dir']/N.VEHICLE_LOG
                    rows[s['world_id']]['prefix_digest'] = S.log_prefix_digest(log, s['t_star'])
    return out, u0


def reduce(kind, raw, u0, waves, selection=None):
    """Generic collection -> the neutral file of one local stage."""
    W, by = raw['waves'], {w['wave']: w for w in waves}
    Jw = {n: {w: r['J'] for w, r in rows.items()} for n, rows in W.items()}
    if kind == 'raw':
        return raw
    if kind == 'e1verify':
        return dict(u0=Jw[u0], plans={by[n]['family']: dict(
            J=Jw[n], group_increase=raw['compare'][n]['group_increase'], groups_all_pass=raw['compare'][n]['groups_all_pass'],
            activations_per_ep=sum(r['activations'] for r in W[n].values())/len(W[n]),
            tail_cvar90=raw['compare'][n]['tail_cvar90']) for n in W if n != u0})
    if kind in ('e1confirm', 'e2main'):
        return dict(u0=Jw[u0], banks={by[n]['bank_sha256']: Jw[n] for n in W if n != u0},
                    arms={by[n].get('arm', n): by[n]['bank_sha256'] for n in W if n != u0},
                    compare={by[n]['bank_sha256']: raw['compare'][n] for n in W if n != u0},
                    acts={by[n]['bank_sha256']: {w: r['acts'] for w, r in W[n].items()} for n in W if n != u0})
    if kind == 'e2discover':
        (n,) = [n for n in W if by[n]['kind'] == 'BANK']
        return {w: dict(world_index=r['world_index'], acts=[dict(t=a['t'], family=a['family'], members=a['members'])
                                                           for a in r['acts']]) for w, r in W[n].items()}
    if kind == 'e2branch':
        out = {}
        for s in selection['states']:
            w = s['world_id']
            f1, s1 = (next(n for n in W if by[n].get('family') == s[b] and w in W[n]) for b in ('F1', 'S1'))
            out[w] = {b: dict(J=W[n][w]['J'], activations=W[n][w]['activations'], acts=W[n][w]['acts'],
                              prefix_digest=W[n][w].get('prefix_digest'), safety=W[n][w]['safety'])
                      for b, n in (('U0', u0), ('F1', f1), ('S1', s1))}
        return out
    raise L.Refused(f'unknown kind {kind}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('crosscheck', 'prepare', 'dry', 'run', 'collect'))
    p.add_argument('--registration', required=True)
    p.add_argument('--code-root', default=os.environ.get('CV_CODE_ROOT', '.'))
    p.add_argument('--block')
    p.add_argument('--waves', nargs='+', help='one or more stage documents')
    p.add_argument('--final-root')
    p.add_argument('--banks-dir', nargs='*', default=[])
    p.add_argument('--kind', default='raw')
    p.add_argument('--selection')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    a = p.parse_args(argv)
    reg = L.read_json(a.registration)
    if a.command == 'crosscheck':
        rep = {}
        for d in a.banks_dir:
            for f in sorted(Path(d).glob('*.json')):
                rep[f.name] = L.crosscheck(a.code_root, [x['strategy'] for x in L.bank_strategies(L.read_json(f))])
        if any(v['status'] != 'PASS' for v in rep.values()) or not rep:
            raise L.Refused(f'crosscheck did not pass: {rep}')
        print(json.dumps(rep, indent=1))
        return 0
    N, D, panel, tag = core(reg, a.block, a.code_root)
    root = Path(a.final_root)
    waves = merge_waves([L.read_json(f) for f in a.waves], a.block)
    if not waves:
        raise L.Refused(f'no wave of block {a.block} in {a.waves}')
    if a.command == 'prepare':
        print(json.dumps(stage_banks(D, root, waves, a.banks_dir), indent=1))
        return 0
    ws = build(D, panel, root, waves, tag)
    if a.command == 'dry':
        print(json.dumps(dict(block=a.block, panel_sha256=panel['panel_sha256'], charged=D.charged(root),
                              cap=int(reg['cap_sim_s']), sim_s=sum(len(s)*N.HORIZON_S for _, _, s in ws),
                              waves={n: dict(n=len(s), first=s[0]['job_id'], last=s[-1]['job_id'], arm=s[0]['arm'],
                                             seeds=[s[0]['world']['seed'], s[-1]['world']['seed']])
                                     for n, _, s in ws}), indent=1))
    elif a.command == 'run':
        import fcntl

        def log(x):
            line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), block=a.block), sort_keys=True,
                              default=str)
            print(line, flush=True)
            with (root/D.LOG_FILE).open('a', encoding='utf-8') as f:
                f.write(line+'\n')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pending = sum(len(s)-N.published(root, w, s, D.C37.check_a37)[2] for w, _, s in ws)
            if D.charged(root)+pending*N.HORIZON_S > int(reg['cap_sim_s']):
                raise L.Refused('the registered cap would be exceeded: nothing is run')
            log(dict(start=a.block, waves=[(w[0], len(w[2])) for w in ws], pending=pending, pid=os.getpid()))
            # no automatic replay: a failed dispatch raises and is reported (as A68)
            log(dict(done=a.block, **D.execute(root, ws, workers=a.workers, chunk=a.chunk, log=log, panel=panel)))
    else:
        sel = L.verify_seal(L.read_json(a.selection)) if a.selection else None
        raw, u0 = collect(N, D, panel, root, waves, tag, sel)
        doc = reduce(a.kind, raw, u0, waves, sel)
        state = L.write_once(a.out, L.text_of(doc))
        print(json.dumps(dict(collected=a.kind, out=a.out, file=state, sha256=L.sha_file(a.out))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
