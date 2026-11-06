"""Stage chaining (pg_wave.sh equivalent) and the CPU dry run of the whole WM-C pipeline on fakes.

    python -m tools.wzwm wave --root ROOT [--workers 20] [--core nm2] [--fit RESERVE] [--stages screen,solos,rep1,rep2]
        plan + run each funnel stage in order (each stage reads the previous stage's RESULTs), then analyze
    python -m tools.wzwm dryrun --out DIR
        fake natives -> geometry moments -> reference root (structured + random) -> StubWM proposals ->
        WM root -> both funnels on the fake core -> analyze -> compare.  CPU only, no SUMO.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import compare as C
from . import geometry as G
from . import proposer as PR
from . import runner as RN
from . import verify_plugin as V
from . import wm_backend as W

FUNNEL = ('screen', 'solos', 'rep1', 'rep2')


def wave(root, *, workers=20, core='nm2', fit=None, stages=FUNNEL, analyze_out=None, inprocess=False, world=None,
         tag=None):
    root = Path(root)
    cfg = V.root_config(root)
    tag = tag or time.strftime('%Y%m%d_%H%M%S')
    log = []
    for st in stages:
        jf = root / 'jobs' / f'{st}_{tag}.jsonl'
        plan = V.plan_stage(root, st, jf, fit)
        res = RN.run(RN.read_jobs(jf), root, cfg['cap'], workers=workers, core=core, phase=cfg.get('phase', 'wmc'),
                     inprocess=inprocess, world=world)
        log.append(dict(stage=st, plan=plan, run=res))
    out = None
    if analyze_out is not False:
        out = V.analyze(root, analyze_out or root / 'results' / f'analyze_{tag}')
    return dict(stages=log, analyze=out)


def _fake_map(dirpath):
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    (d / 'MAP_SPEC.json').write_text(json.dumps(dict(name='wz3to2', note='fake map for the CPU dry run')))
    return d


def dry_run(out, *, worlds=None, k_matched=True, wm=None, cfg=None, cap=10_000_000.0):
    """Whole pipeline on fakes (no SUMO). Returns paths, funnel counts and the comparison."""
    out = G.check_path(Path(out).resolve())
    out.mkdir(parents=True, exist_ok=True)
    worlds = worlds or [dict(demand=3200, penetration=0.6, seed=989400), dict(demand=3200, penetration=0.4, seed=989401)]
    wf = out / 'WORLDS_in.json'
    wf.write_text(json.dumps(worlds))
    ref, wmr = out / 'ref', out / 'wm'
    V.init(ref, cap=cap, map_dir=_fake_map(out / 'fake_map'), worlds=wf, note='dry run reference (fake)')
    jf = ref / 'jobs' / 'natives.jsonl'
    V.plan_natives(ref, jf)
    RN.run(RN.read_jobs(jf), ref, cap, core='fake', inprocess=True)
    V.geometry_moments(ref)
    geo = ref / 'MOMENTS_GEOMETRY.json'
    V.build_moments(ref, geo, reference=True)
    ref_wave = wave(ref, core='fake', inprocess=True, analyze_out=ref / 'results', tag='dry')
    wm = wm or W.StubWM()
    props = out / 'PROPOSALS.jsonl'
    receipt = PR.propose_file(geo, props, wm, PR.config(cfg), k_matched=k_matched)
    V.init(wmr, cap=cap, natives_root=ref, note='dry run WM root (fake)')
    V.build_moments(wmr, geo, props, allow_stub=True)
    wm_wave = wave(wmr, core='fake', inprocess=True, analyze_out=wmr / 'results', tag='dry')
    cmp = C.compare([ref / 'results' / 'WMC_SUMMARY.json', wmr / 'results' / 'WMC_SUMMARY.json'], props)
    (out / 'COMPARE.json').write_text(json.dumps(cmp, indent=1, default=float))
    return dict(ref=str(ref), wm=str(wmr), proposals=str(props), proposals_receipt=receipt,
                ref_funnel=ref_wave['analyze']['funnel'], wm_funnel=wm_wave['analyze']['funnel'], compare=cmp)


def main_wave(argv=None):
    ap = argparse.ArgumentParser(prog='python -m tools.wzwm wave')
    ap.add_argument('--root', required=True)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--core', default='nm2')
    ap.add_argument('--fit', type=float, default=None)
    ap.add_argument('--stages', default=','.join(FUNNEL))
    ap.add_argument('--analyze-out')
    a = ap.parse_args(argv)
    res = wave(a.root, workers=a.workers, core=a.core, fit=a.fit, stages=tuple(a.stages.split(',')),
               analyze_out=a.analyze_out)
    print(json.dumps(res, indent=1, default=str))


def main_dryrun(argv=None):
    ap = argparse.ArgumentParser(prog='python -m tools.wzwm dryrun')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    res = dry_run(a.out)
    for arm, m in res['compare']['metrics'].items():
        print(arm, dict(verified=m['admitted_variants'], sim_s=m['sim_s'], per_ksim=m['verified_per_ksim']))
    print(json.dumps(dict(ref_funnel=res['ref_funnel'], wm_funnel=res['wm_funnel']), indent=1))
