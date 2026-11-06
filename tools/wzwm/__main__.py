"""python -m tools.wzwm {propose|score|rankcheck|verify|run|wave|compare|labels|dryrun} ...

propose  (WM host)  MOMENTS.json -> PROPOSALS.jsonl (+ _RECEIPT.json); zero SUMO
score    (WM host)  WM predictions of the variants already in a MOMENTS.json -> SCORES.jsonl; zero SUMO
rankcheck (any)     WM ranking vs existing SUMO verdicts (Spearman T/S, AUC admitted, top-1) -> JSON
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
    ap = argparse.ArgumentParser(prog='python -m tools.wzwm propose')
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
    ap = argparse.ArgumentParser(prog='python -m tools.wzwm score')
    ap.add_argument('--moments', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--wm', default='auto')
    ap.add_argument('--wm-api', action='append')
    ap.add_argument('--wm-arg', action='append')
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--H', type=int, default=300)
    ap.add_argument('--source-root')
    ap.add_argument('--sids')
    a = ap.parse_args(argv)
    print(json.dumps(PR.score_moments(_backend(a), a.moments, a.out, K=a.K, H=a.H, source_root=a.source_root,
                                      sids=a.sids.split(',') if a.sids else None), indent=1, default=str))


def rankcheck(argv):
    from . import compare as C
    from . import geometry as G
    ap = argparse.ArgumentParser(prog='python -m tools.wzwm rankcheck')
    ap.add_argument('--summary', action='append', required=True)
    ap.add_argument('--scores', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    res = C.rank_check(a.summary, a.scores)
    with G.check_path(a.out).open('x') as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps(dict(all=res['all'], top1=res['top1_admitted_given_any']), indent=1))


def labels(argv):
    from . import wallfit_bridge as WB
    ap = argparse.ArgumentParser(prog='python -m tools.wzwm labels')
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
