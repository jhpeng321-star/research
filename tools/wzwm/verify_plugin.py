"""Plug-in: WM proposals -> the unchanged N3/PLANGEN SUMO verification funnel.

Stages (python -m tools.wzwm verify STAGE --root ROOT ...), each writing a NEW jobs file (never appends):
  init              ROOT layout + WMC_ROOT.json (source/natives root, map spec, cap, phase)
  natives           fresh worlds only: ROOT/WORLDS.json -> pg_plan.plan_natives (900 s, saved t0 states)
  geometry-moments  fresh worlds only: pg_plan.moments() (geometry-only t0 + structured B / random R
                    variants) -> ROOT/MOMENTS_GEOMETRY.json (sid prefix wmc_), the proposer's input
  moments           ROOT/MOMENTS.json from the source moments + PROPOSALS.jsonl (WM variants; validated
                    against pg_plan.member_spec; optional --with-random keeps the source R variants as a
                    count-matched arm in the same funnel), or --reference (source structured+random as-is)
  screen            r0 base + every variant group (pg_plan.plan_screen body), H = 300 s
  solos|rep1|rep2   pg_plan.plan_solos / plan_rep1 / plan_rep2 verbatim (best-T per family solos, one
                    replicated candidate per moment and arm, dithered replicates, 120 s extensions)
  analyze           pg_analyze.verdict per variant (admission: every replicate LB95(T)>0, LB95(S)>0,
                    S>=1 veh-s; later harm T>0 over [t0,t0+420); 0 collisions/teleports) + example rows
                    -> WMC_SUMMARY.json, WMC_MOVES_B.jsonl (WM), WMC_MOVES_R.jsonl, WMC_NEGATIVES.jsonl
Between stages: python -m tools.wzwm run JOBS --root ROOT --cap C (precharge ledger, own PIDs).

Same moments as the reference arms: natives (saved t0 states, scenario) are read from the source root
and never written; the dither targets of the source moment are reused unless a WM member is one of
them (then pg_plan.dither_targets re-picks on the extended member union with the source sid salt).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import geometry as G
from . import n3

ROOT_FILE = 'WMC_ROOT.json'
PHASE_ID = 'WMC-wz3to2'
KEEP = ('cell', 'demand', 'penetration', 'seed', 'window', 't0', 'valid', 'sid', 'inserted_t0', 'approach_speed',
        'closing_lane_slow', 'n_zone_cavs', 'all_cavs_t0')


def _write_new(path, text):
    path = G.check_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as fh:
        fh.write(text)
    return path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def root_config(root):
    p = G.check_path(Path(root) / ROOT_FILE)
    return json.loads(p.read_text())


def bind_root(root):
    cfg = root_config(root)
    return n3.bind(root, natives_root=cfg.get('natives_root'), map_spec=cfg.get('map_spec'), cap=cfg['cap'])


def init(root, *, cap, natives_root=None, map_dir=None, worlds=None, phase='wmc', note=''):
    """natives_root: an existing N3/PLANGEN root whose natives/moments are reused (comparison mode);
    None: fresh natives in ROOT (switch-over collection; needs map_dir and worlds)."""
    root = G.check_path(Path(root).resolve())
    if (root / ROOT_FILE).exists():
        raise FileExistsError(f'{root / ROOT_FILE} exists; use a new root (never overwrite a run)')
    for d in ('runs', 'jobs', 'ledger', 'results'):
        (root / d).mkdir(parents=True, exist_ok=True)
    map_spec = None
    if natives_root:
        natives_root = str(G.check_path(Path(natives_root).resolve()))
        map_spec = str(Path(natives_root) / 'maps' / 'wz3to2' / 'MAP_SPEC.json')
    else:
        if not map_dir or not worlds:
            raise ValueError('fresh natives need --map-dir and --worlds')
        dst = root / 'maps' / 'wz3to2'
        shutil.copytree(G.check_path(map_dir), dst)
        map_spec = str(dst / 'MAP_SPEC.json')
        rows = json.loads(G.check_path(worlds).read_text())
        for w in rows:
            G.check_seed(w['seed'], where=f'({worlds})')
        _write_new(root / 'WORLDS.json', json.dumps(rows))
    cfg = dict(schema='wmc-root-v1', root=str(root), natives_root=natives_root, map_spec=map_spec, cap=float(cap),
               phase=phase, created=time.time(), note=note, n3_vendor=n3.verify())
    _write_new(root / ROOT_FILE, json.dumps(cfg, indent=1, sort_keys=True))
    return cfg


def _filter(P, rows, reserve):
    if reserve is None:
        return [r for r in rows if not (Path(r['out_dir']) / 'RESULT.json').exists()]
    return P.fit(rows, reserve=float(reserve))


def write_jobs(rows, out):
    text = ''.join(json.dumps(r) + '\n' for r in rows)
    _write_new(out, text)
    return dict(jobs=len(rows), sim_s=sum(float(r['t_end'] if r['kind'] == 'native' else r['H']) for r in rows),
                out=str(out))


# ------------------------------------------------------------------------------------------ fresh worlds
def plan_natives(root, out, reserve=None):
    P, _ = bind_root(root)
    if root_config(root).get('natives_root'):
        raise ValueError('this root reuses another root\'s natives; natives are never re-run here')
    return write_jobs(_filter(P, P.plan_natives(), reserve), out)


def geometry_moments(root):
    P, _ = bind_root(root)
    if root_config(root).get('natives_root'):
        raise ValueError('geometry moments come from the source root in reuse mode')
    rows = P.moments()
    for m in rows:
        if m.get('sid', '').startswith('pg_'):
            m['sid'] = 'wmc_' + m['sid'][3:]
    out = _write_new(Path(root) / 'MOMENTS_GEOMETRY.json', json.dumps(rows))
    return dict(moments=len(rows), valid=sum(bool(m['valid']) for m in rows), out=str(out))


# ------------------------------------------------------------------------------------------ moments
def proposer_of(v):
    if v.get('family') == 'R_random':
        return 'random'
    return 'wm' if v.get('wm') is not None else 'structured'


def validate_variant(P, m, vid, v):
    """Every member is a t0 CAV of this moment and its spec equals pg_plan.member_spec (same executor)."""
    by = {c['id']: c for c in m['all_cavs_t0']}
    if not v.get('members'):
        raise ValueError(f"{m['sid']}/{vid}: no members")
    ids = [mm['id'] for mm in v['members']]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{m['sid']}/{vid}: duplicate members")
    for mm in v['members']:
        c = by.get(mm['id'])
        if c is None or c['edge'] != 'up':
            raise ValueError(f"{m['sid']}/{vid}: member {mm['id']} is not a CAV on 'up' at t0")
        if mm['role'] != 'hold':
            raise ValueError(f"{m['sid']}/{vid}: role {mm['role']!r} has no N3 executor")
        ref = P.member_spec(c, v['params'], anchor=mm['params'].get('anchor'))
        if ref != mm:
            raise ValueError(f"{m['sid']}/{vid}: member spec differs from pg_plan.member_spec: {mm} != {ref}")
    return ids


def build_moments(root, source_moments, proposals=None, *, with_random=False, reference=False, allow_stub=False):
    """ROOT/MOMENTS.json (pg_plan.moments_loaded() format) for the WM arm (or the reference arms)."""
    if (proposals is None) == (not reference):
        raise ValueError('give exactly one of --proposals or --reference')
    P, _ = bind_root(root)
    src_path = G.check_path(source_moments)
    src = [m for m in json.loads(src_path.read_text()) if m.get('valid')]
    props = {}
    if proposals:
        for line in G.check_path(proposals).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get('backend', {}).get('backend') == 'stub' and not allow_stub:
                    raise ValueError('proposals come from the StubWM; refuse to spend SUMO on them (--allow-stub '
                                     'for dry runs only)')
                props[r['sid']] = r
    out, counts = [], Counter()
    counts['proposals_without_moment'] = len(set(props) - {m['sid'] for m in src})
    for m in src:
        G.check_seed(m['seed'], where=f"(moment {m['sid']})")
        row = {k: m[k] for k in KEEP if k in m}
        if reference:
            variants = {vid: v for vid, v in m['variants'].items()}
        else:
            pr = props.get(m['sid'])
            if pr is None:
                counts['no_proposal'] += 1
                continue
            if pr['t0'] != m['t0']:
                raise ValueError(f"{m['sid']}: proposal t0 {pr['t0']} != moment t0 {m['t0']}")
            if pr.get('abstain'):
                counts['abstained'] += 1
                continue
            variants = {vid: dict(family=v['family'], params=v['params'], members=v['members'], wm=v.get('wm'))
                        for vid, v in pr['variants'].items()}
            if with_random:
                for vid, v in m['variants'].items():
                    if v['family'] == 'R_random':
                        variants[vid] = v
        if not variants:
            counts['no_variants'] += 1
            continue
        for vid, v in variants.items():
            validate_variant(P, m, vid, v)
        union = sorted({mm['id'] for v in variants.values() for mm in v['members']})
        src_dith = {str(k): v for k, v in (m.get('dither_ids') or {}).items()}
        if src_dith and not set(src_dith.values()) & set(union):
            dith, how = src_dith, 'source'
        else:
            dith = {str(k): v for k, v in P.dither_targets({'cavs': m['all_cavs_t0']},
                                                              set(union) | set(m.get('union') or []),
                                                              m['sid']).items()}
            how = 'recomputed'
            counts['dither_recomputed'] += 1
        row.update(variants=variants, union=union, dither_ids=dith, dither_source=how,
                   source_variants=sorted(m['variants']), source_moments=str(src_path))
        out.append(row)
        counts['moments'] += 1
    mpath = _write_new(Path(root) / 'MOMENTS.json', json.dumps(out))
    receipt = dict(schema='wmc-moments-receipt-v1', moments=str(mpath), moments_sha256=_sha(mpath),
                   source_moments=str(src_path), source_sha256=_sha(src_path),
                   proposals=None if proposals is None else str(proposals),
                   proposals_sha256=None if proposals is None else _sha(proposals), with_random=bool(with_random),
                   reference=bool(reference), counts=dict(counts),
                   variants=dict(Counter(proposer_of(v) for m in out for v in m['variants'].values())))
    _write_new(Path(root) / 'MOMENTS_RECEIPT.json', json.dumps(receipt, indent=1, sort_keys=True))
    return receipt


# ------------------------------------------------------------------------------------------ funnel stages
def plan_stage(root, stage, out, reserve=None):
    P, _ = bind_root(root)
    if stage == 'screen':
        rows = []
        for m in P.moments_loaded():
            rows.append(P.bjob(m, 0, 'base', [], P.H_MAIN, save_at=[P.H_MAIN]))
            for vid, v in m['variants'].items():
                rows.append(P.bjob(m, 0, f'g__{vid}', v['members'], P.H_MAIN, save_at=[P.H_MAIN]))
    elif stage == 'solos':
        rows = P.plan_solos()
    elif stage == 'rep1':
        rows = P.plan_rep1()
    elif stage == 'rep2':
        rows = P.plan_rep2()
    else:
        raise ValueError(f'unknown stage {stage!r}')
    return write_jobs(_filter(P, rows, reserve), out)


# ------------------------------------------------------------------------------------------ analysis
def _tag(row, proposer, v, phase_id):
    row['verification_id'] = 'wmcv_' + row['verification_id'].split('_', 1)[1]
    row['source'] = (f'{phase_id}: N3/PLANGEN funnel (vendored pg_plan/pg_analyze, sha-pinned) on {proposer} '
                     'proposals; SUMO-only, matched CRN branches from the saved native state')
    row['world']['split'] = 'TRAIN 989200-989599 (wz3to2)'
    row['newmap2_id'] = phase_id
    row['phase'] = 'wmc'
    row['proposer'] = proposer
    row['wm_prediction'] = v.get('wm')
    row['binding_note'] = ('group_lane_bank.binding() world-seed range check (986000-986071) must admit the wz3to2 '
                           'TRAIN range 989200-989599 before ingestion (N4; N11 F1)')
    return row


def analyze(root, out_dir=None, phase_id=PHASE_ID):
    P, A = bind_root(root)
    out = G.check_path(Path(out_dir) if out_dir else Path(root) / 'results')
    out.mkdir(parents=True, exist_ok=True)
    allm = [m for m in json.loads((Path(root) / 'MOMENTS.json').read_text()) if m.get('valid')]
    verdicts, B, R, NEG = [], [], [], []
    for m in allm:
        for vid, v in m['variants'].items():
            vd = A.verdict(m, vid)
            vd['v2v'] = A.v2v(m, vid)
            vd['proposer'] = proposer_of(v)
            vd['wm_prediction'] = v.get('wm')
            vd['n_members'] = len(v['members'])
            verdicts.append(vd)
            if vd['admitted']:
                rows = [_tag(r, vd['proposer'], v, phase_id) for r in A.example_rows(m, vd)]
                (R if vd['proposer'] == 'random' else B).extend(rows)
            else:
                NEG.append(dict(schema='newmap2-negative-v1', arm=vd['arm'], proposer=vd['proposer'],
                                situation_id=m['sid'],
                                world=dict(map='wz3to2', cell=m['cell'], demand=m['demand'],
                                           penetration=m['penetration'], seed=m['seed']),
                                t0=m['t0'], family=vd['family'], variant=vid, members=vd['members'],
                                member_lanes=vd['member_lanes'], params=vd['params'], admitted=False,
                                status='rejected', stage=vd['stage'], reason=vd['reason'],
                                advantage_veh_s=vd.get('T_r0'), adv_lower95=vd.get('T_lb_r0'),
                                synergy_veh_s=vd.get('S_r0'), synergy_lower95=vd.get('S_lb_r0'), reps=vd.get('reps'),
                                safety=vd.get('safety_r0'), observable=vd['v2v']['observable_4slot_t0'],
                                v2v_range=vd['v2v'], wm_prediction=v.get('wm'), newmap2_id=phase_id, phase='wmc'))
    funnel = defaultdict(lambda: dict(variants=0, stages=Counter(), admitted=0, moments=set(), admitted_moments=set()))
    for vd in verdicts:
        f = funnel[vd['proposer']]
        f['variants'] += 1
        f['stages'][vd['stage']] += 1
        f['moments'].add(vd['sid'])
        if vd['admitted']:
            f['admitted'] += 1
            f['admitted_moments'].add(vd['sid'])
    fun = {k: dict(variants=f['variants'], stages=dict(f['stages']), admitted_variants=f['admitted'],
                   moments=len(f['moments']), admitted_moments=len(f['admitted_moments']))
           for k, f in funnel.items()}
    led = Path(root) / 'ledger' / 'CHARGES.jsonl'
    charged = sum(float(json.loads(x)['planned_sim_s']) for x in led.read_text().splitlines() if x.strip()) \
        if led.exists() else 0.0
    summary = dict(schema='wmc-summary-v1', phase_id=phase_id, root=str(root), n_moments=len(allm), funnel=fun,
                   ledger_sim_s=charged, n3_vendor=n3.verify(), verdicts=verdicts,
                   statistic='pg_analyze.verdict (NEWMAP2 rule incl. amendments 3-4), vendored verbatim')
    files = {}
    for name, rows in (('WMC_SUMMARY.json', None), ('WMC_MOVES_B.jsonl', B), ('WMC_MOVES_R.jsonl', R),
                       ('WMC_NEGATIVES.jsonl', NEG)):
        text = json.dumps(summary, indent=1, default=str) if rows is None else \
            ''.join(json.dumps(r, default=str) + '\n' for r in rows)
        files[name] = str(_write_new(out / name, text))
    return dict(funnel=fun, rows=dict(B=len(B), R=len(R), negatives=len(NEG)), files=files, ledger_sim_s=charged)


# ------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m tools.wzwm verify')
    ap.add_argument('stage', choices=('init', 'natives', 'geometry-moments', 'moments', 'screen', 'solos', 'rep1',
                                      'rep2', 'analyze'))
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', help='jobs file (planning stages) or results dir (analyze)')
    ap.add_argument('--fit', type=float, default=None, help='pg_plan.fit reserve: whole moment units under the cap')
    ap.add_argument('--cap', type=float, help='init: hard sim-s cap of this root')
    ap.add_argument('--natives-root', help='init: reuse this N3/PLANGEN root\'s natives and moments')
    ap.add_argument('--map-dir', help='init (fresh): wz3to2 map directory to copy')
    ap.add_argument('--worlds', help='init (fresh): WORLDS.json [{demand, penetration, seed}] (TRAIN only)')
    ap.add_argument('--note', default='')
    ap.add_argument('--source-moments', help='moments: source MOMENTS.json (default natives-root/MOMENTS.json '
                                             'or ROOT/MOMENTS_GEOMETRY.json)')
    ap.add_argument('--proposals')
    ap.add_argument('--with-random', action='store_true')
    ap.add_argument('--reference', action='store_true')
    ap.add_argument('--allow-stub', action='store_true')
    a = ap.parse_args(argv)
    if a.stage == 'init':
        res = init(a.root, cap=a.cap, natives_root=a.natives_root, map_dir=a.map_dir, worlds=a.worlds, note=a.note)
    elif a.stage == 'natives':
        res = plan_natives(a.root, a.out, a.fit)
    elif a.stage == 'geometry-moments':
        res = geometry_moments(a.root)
    elif a.stage == 'moments':
        cfg = root_config(a.root)
        src = a.source_moments or (Path(cfg['natives_root']) / 'MOMENTS.json' if cfg.get('natives_root')
                                   else Path(a.root) / 'MOMENTS_GEOMETRY.json')
        res = build_moments(a.root, src, a.proposals, with_random=a.with_random, reference=a.reference,
                            allow_stub=a.allow_stub)
    elif a.stage == 'analyze':
        res = analyze(a.root, a.out)
    else:
        res = plan_stage(a.root, a.stage, a.out, a.fit)
    print(json.dumps(res, default=str, indent=1))
    return res


if __name__ == '__main__':
    main()
