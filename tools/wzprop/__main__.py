"""python -m tools.wzprop {propose|score|rankcheck|verify|run|wave|compare|labels|dryrun} ...

propose  (WM host)  MOMENTS.json -> PROPOSALS.jsonl (+ _RECEIPT.json); zero SUMO
score    (WM host)  WM predictions of the variants already in a MOMENTS.json -> SCORES.jsonl; zero SUMO
rankcheck (any)     WM ranking vs existing SUMO verdicts (Spearman T/S, AUC admitted, top-1) -> JSON
enrich   (any)      MOMENTS.json + WM-D tapes -> observable moments (CAV-detected neighbours, RSU counters)
strategy (WM host)  strategy-level search per demand cell vs the solo2 incumbent -> STRATEGY_SEARCH_*.json, ARMS.json
opensearch (WM host) open roaming search (free programs, CEM/evolution, MAP-Elites, LCB - disagreement)
regime   (any)      claim-4: S-REGIME strategy states from tapes / WM-vs-measured benefit agreement
wmrank   (WM host)  score + bootstrap rank metrics with one WM checkpoint (baseline WM2 / WM-E) -> DIR
rankmetrics (any)   bootstrap rank metrics from SCORES.jsonl + summaries -> JSON + .md
verify   (cpuV)     plug-in stages: init | natives | geometry-moments | moments | screen | solos | rep1 | rep2 | analyze
run      (cpuV)     run one jobs file under the root's precharge ledger and cap
wave     (cpuV)     screen -> solos -> rep1 -> rep2 (plan + run each) -> analyze
compare  (any)      matched-effort WM vs structured vs random report -> COMPARE.json
labels   (any)      wz3to2 W3 wall-fit labels from funnel summaries -> LABELS.jsonl
dryrun   (any)      the whole pipeline on fakes (StubWM + fake core), CPU only
"""
from __future__ import annotations

import argparse
import json
import sys


def _kv(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition('=')
        if not v:
            raise SystemExit(f'expected key=value, got {it!r}')
        out[k] = v
    return out


def propose(argv):
    from . import proposer as PR
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop propose')
    ap.add_argument('--moments', required=True, help='source MOMENTS.json (N3/PLANGEN root or MOMENTS_GEOMETRY.json)')
    ap.add_argument('--out', required=True, help='new PROPOSALS.jsonl')
    ap.add_argument('--wm', default='auto', help="auto | wma | wma:PATH | stub | pkg.mod:factory")
    ap.add_argument('--wm-api', action='append', help='override WMA_READY api entries, e.g. evaluate=pkg.mod:fn')
    ap.add_argument('--wm-arg', action='append', help='backend load kwargs key=value (JSON values allowed)')
    ap.add_argument('--k', default='matched', help="'matched' (= structured variant count per moment) or an int")
    ap.add_argument('--config', help='JSON dict of proposer.DEFAULTS overrides')
    ap.add_argument('--source-root', help='root whose runs/<cell>/s<seed>/native/states hold the t0 states')
    ap.add_argument('--sids', help='comma list of moment sids (default all valid)')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--wallfit-head', help='W3 HEAD.pt: optional re-ranker of the WM finalists (default off)')
    a = ap.parse_args(argv)
    wm = _backend(a)
    over = json.loads(a.config) if a.config else {}
    matched = a.k == 'matched'
    if not matched:
        over['k'] = int(a.k)
    cfg = PR.config(over)
    rec = PR.propose_file(a.moments, a.out, wm, cfg, source_root=a.source_root,
                          sids=a.sids.split(',') if a.sids else None, head=a.wallfit_head, k_matched=matched,
                          limit=a.limit)
    print(json.dumps(rec, indent=1, default=str))


def _backend(a):
    from . import wm_backend as W
    kw = {}
    for k, v in _kv(a.wm_arg).items():
        try:
            kw[k] = json.loads(v)
        except ValueError:
            kw[k] = v
    return W.load_backend(a.wm, api=_kv(a.wm_api), **kw)


def score(argv):
    from . import proposer as PR
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop score')
    ap.add_argument('--moments', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--wm', default='auto')
    ap.add_argument('--wm-api', action='append')
    ap.add_argument('--wm-arg', action='append')
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--H', type=int, default=300)
    ap.add_argument('--source-root')
    ap.add_argument('--sids')
    ap.add_argument('--solos-labelled-by', action='append', help='summary JSON: solos only for its S-labelled variants')
    ap.add_argument('--no-solos', action='store_true', help='T only (group vs base), e.g. the regime strategy check')
    ap.add_argument('--mem-fraction', type=float, help='torch.cuda per-process memory cap (shared GPU)')
    a = ap.parse_args(argv)
    if a.mem_fraction:
        import torch
        torch.cuda.set_per_process_memory_fraction(float(a.mem_fraction))
    sv = PR.labelled_solo_variants(a.solos_labelled_by) if a.solos_labelled_by else None
    if a.no_solos:
        sv = set()
    print(json.dumps(PR.score_moments(_backend(a), a.moments, a.out, K=a.K, H=a.H, source_root=a.source_root,
                                      sids=a.sids.split(',') if a.sids else None, solo_variants=sv),
                     indent=1, default=str))


def wmrank(argv):
    """One-shot zero-SUMO WM ranking check: score the verified variants with a WM checkpoint, then the
    bootstrap rank metrics. Same command for the un-fine-tuned WM2 and WM-E's fine-tuned checkpoint."""
    import time
    from . import geometry as G
    from . import proposer as PR
    from . import rankcheck as RC
    from . import wm_backend as W
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop wmrank')
    ap.add_argument('--ready', required=True, help='WMA_READY.json (api entries)')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--map-spec', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-rows', type=int, default=96)
    ap.add_argument('--mem-fraction', type=float, default=None, help='torch.cuda per-process memory fraction')
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--moments', required=True, help='tapes.enrich output (observable moments)')
    ap.add_argument('--summary', action='append', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--H', type=int, default=300)
    ap.add_argument('--sids')
    ap.add_argument('--solos', choices=('all', 'labelled'), default='all',
                    help="'labelled': solos only for the summaries' S-labelled variants (cheaper)")
    ap.add_argument('--out', required=True, help='NEW directory')
    a = ap.parse_args(argv)
    out = G.check_path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    import torch
    if a.threads:
        torch.set_num_threads(int(a.threads))
    if a.mem_fraction and a.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(float(a.mem_fraction))
    t0 = time.time()
    wm = W.load_backend(f'wma:{a.ready}', checkpoint=a.checkpoint, map_spec=a.map_spec, device=a.device,
                        max_rows=a.max_rows)
    if a.checkpoint and not getattr(wm.handle, 'meta', {}).get('trained', True):
        raise SystemExit('loaded an untrained model')
    sv = PR.labelled_solo_variants(a.summary) if a.solos == 'labelled' else None
    rec = PR.score_moments(wm, a.moments, out / 'SCORES.jsonl', K=a.K, H=a.H,
                           sids=a.sids.split(',') if a.sids else None, solo_variants=sv)
    rec.update(handle_meta=getattr(wm.handle, 'meta', None), label=a.label, wall_s_total=round(time.time() - t0, 1),
               device=a.device, max_rows=a.max_rows,
               cuda_peak_gb=(round(torch.cuda.max_memory_allocated() / 2**30, 2) if a.device.startswith('cuda')
                             else None))
    (out / 'SCORE_RECEIPT.json').write_text(json.dumps(rec, indent=1, default=str))
    RC.main(['--scores', str(out / 'SCORES.jsonl'), '--label', a.label, '--checkpoint', a.checkpoint,
             '--out', str(out / 'RANKCHECK.json')] + [x for s in a.summary for x in ('--summary', s)])


def rankcheck(argv):
    from . import compare as C
    from . import geometry as G
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop rankcheck')
    ap.add_argument('--summary', action='append', required=True)
    ap.add_argument('--scores', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    res = C.rank_check(a.summary, a.scores)
    with G.check_path(a.out).open('x') as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps(dict(all=res['all'], top1=res['top1_admitted_given_any']), indent=1))


def opensearch(argv):
    """Open (roaming) WM search over free programs + family seeds; output rows plug into verify moments."""
    from . import geometry as G
    from . import open_search as OS
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop opensearch')
    ap.add_argument('--moments', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--wm', default='auto')
    ap.add_argument('--wm-api', action='append')
    ap.add_argument('--wm-arg', action='append')
    ap.add_argument('--config', help='JSON dict of open_search.DEFAULTS overrides')
    ap.add_argument('--sids')
    a = ap.parse_args(argv)
    wm = _backend(a)
    cfg = OS.config(json.loads(a.config) if a.config else None)
    ms = [m for m in json.loads(G.check_path(a.moments).read_text()) if m.get('valid')]
    if a.sids:
        ms = [m for m in ms if m['sid'] in set(a.sids.split(','))]
    with G.check_path(a.out).open('x') as fh:
        for m in ms:
            row, _arch, _ev = OS.search(wm, m, cfg)
            fh.write(json.dumps(row, sort_keys=True, default=str) + '\n')
            fh.flush()
    print(json.dumps(dict(out=a.out, moments=len(ms), backend=wm.meta), default=str))


def strategy(argv):
    """Strategy-level WM search per demand cell (incumbent solo2) -> STRATEGY_SEARCH_<cell>.json + ARMS.json."""
    import time
    from . import geometry as G
    from . import strategy_search as SS
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop strategy')
    ap.add_argument('--states', required=True, help='observable start states (regime states / enriched moments)')
    ap.add_argument('--cells', required=True)
    ap.add_argument('--out', required=True, help='NEW directory')
    ap.add_argument('--wm', default='auto')
    ap.add_argument('--wm-api', action='append')
    ap.add_argument('--wm-arg', action='append')
    ap.add_argument('--config', help='JSON dict of strategy_search.DEFAULTS overrides')
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--mem-fraction', type=float, help='per worker process (with --workers) or this process')
    ap.add_argument('--workers', type=int, default=0, help='parallel WM worker processes (CPU-bound rollouts)')
    a = ap.parse_args(argv)
    out = G.check_path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if a.workers:
        from . import wm_backend as W
        kw = {}
        for k2, v in _kv(a.wm_arg).items():
            try:
                kw[k2] = json.loads(v)
            except ValueError:
                kw[k2] = v
        local = dict(kw, device='cpu') if 'device' in kw else kw
        wm = W.ParallelWM(a.wm, a.workers, api=_kv(a.wm_api), local_kw=local, mem_fraction=a.mem_fraction, **kw)
    else:
        if a.mem_fraction:
            import torch
            torch.cuda.set_per_process_memory_fraction(float(a.mem_fraction))
        wm = _backend(a)
    cfg = json.loads(a.config) if a.config else None
    states = [m for m in json.loads(G.check_path(a.states).read_text()) if m.get('valid')]
    arms = {}
    for cell in a.cells.split(','):
        st = [m for m in states if m['cell'] == cell]
        t0 = time.time()
        log = (out / f'LOG_{cell}.jsonl').open('x')
        res = SS.search_cell(wm, st, cfg, log=lambda d: (log.write(json.dumps(d) + '\n'), log.flush()))
        log.close()
        res.update(cell=cell, backend=wm.meta, wall_s=round(time.time() - t0, 1), new_sumo_sim_s=0)
        (out / f'STRATEGY_SEARCH_{cell}.json').write_text(json.dumps(res, indent=1, default=str))
        arms[cell] = dict(wm=SS.select_for_sumo(res, a.k), wm_fair=SS.select_for_sumo(res, a.k, fair=True),
                          structured=SS.structured_arm(a.k), random=SS.random_arm(a.k), incumbent=SS.SOLO2,
                          best_beating_solo2=(res['best_beating_solo2'] or {}).get('strategy'),
                          best_fair=(res['best_fair'] or {}).get('strategy'), fairness=res['fairness'])
        print(cell, 'evaluated', res['evaluated'], 'archive', res['archive_cells'], 'best',
              (res['final'][0]['strategy']['name'], res['final'][0]['gain_vs_solo2'], res['final'][0]['gain_vs_solo2_lcb']))
    (out / 'ARMS.json').write_text(json.dumps(arms, indent=1))


def labels(argv):
    from . import wallfit_bridge as WB
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop labels')
    ap.add_argument('--summary', action='append', required=True, help='NM2/PG/LC/WMC summary JSON (repeatable)')
    ap.add_argument('--moments', action='append', required=True, help='matching MOMENTS.json (repeatable)')
    ap.add_argument('--wm-scores', action='append', help='score output for non-WM variants (optional)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    print(json.dumps(WB.build_labels(a.summary, a.moments, a.out, a.wm_scores), indent=1))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == 'propose':
        return propose(rest)
    if cmd == 'labels':
        return labels(rest)
    if cmd == 'score':
        return score(rest)
    if cmd == 'rankcheck':
        return rankcheck(rest)
    if cmd == 'wmrank':
        return wmrank(rest)
    if cmd == 'strategy':
        return strategy(rest)
    if cmd == 'opensearch':
        return opensearch(rest)
    if cmd == 'regime':
        from .regime_wm import main as rg
        rg(rest)
        return 0
    if cmd == 'rankmetrics':
        from .rankcheck import main as rm
        rm(rest)
        return 0
    if cmd == 'enrich':
        from .tapes import enrich
        if len(rest) != 3:
            raise SystemExit('python -m tools.wzprop enrich MOMENTS.json TAPES_ROOT OUT.json')
        print(json.dumps(enrich(*rest)))
        return 0
    if cmd == 'verify':
        from .verify_plugin import main as m
    elif cmd == 'run':
        from .runner import main as m
    elif cmd == 'wave':
        from .pipeline import main_wave as m
    elif cmd == 'dryrun':
        from .pipeline import main_dryrun as m
    elif cmd == 'compare':
        from .compare import main as m
    else:
        print(__doc__)
        return 2
    m(rest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
