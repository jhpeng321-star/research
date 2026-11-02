"""Refresh the learned search after every new B update; write ACTIVATION_READY on a PASS.

    python -m tools.coopmwm.learned_search_watch [--once] [--poll-s 600] [--max-hours 30]

Owner/MAIN 2026-11-02 approved switching B to the learned search, but only on
condition that it PASSES the registered offline check (prereg 9b is satisfied by
that approval; 9a stays the gate). This watcher is CPU only, and on the remote side
it runs read-only ``ls``, ``find`` and ``tar -c``. It never touches c4-coopnative
or the live services; MAIN does the switch.

Each new ``B_UPDATE_###.json`` on cpuQ triggers one refresh, as does the first poll.
A refresh:
- re-collects receipts (``learned_search_report.run``);
- retrains on the frozen configuration;
- re-runs the WM-pool check AND the verification-labelled check;
- runs the evaluations curve (once >= 48 full pools) and the rank 5-8 estimate;
- estimates the wall-clock saving per B update;
- commits the refreshed report.

READY condition (registered minimums, prereg 5/7). READY needs one of:
- 5W: the WM-pool check PASS, with >= 160 full-pool searches from >= 32 TRAIN
  worlds and >= 8 held-out worlds / 40 held-out searches;
- 5V: the verification-labelled check PASS on its own minimum (>= 8 worlds and
  >= 40 lists with an exact-best candidate).

In either case 5V must not show harm. When READY, the watcher writes
``docs/paper_results/LEARNED_SEARCH_ACTIVATION_READY.json``, commits it and exits
with code 0, so the agent can message MAIN at once.
"""
import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

from . import learned_search_report as lr
from . import mppi_search as ms

QROOT = '/home/ubuntu/coopmwm20261102_warm1'
REPO = Path(__file__).resolve().parents[2]
READY = REPO/'docs/paper_results/LEARNED_SEARCH_ACTIVATION_READY.json'
REPORT = REPO/'docs/paper_results/LEARNED_SEARCH_REPORT.md'
NONINFERIOR = -.05     # quality - plain_alt CI lower bound for a mode to count as non-inferior (descriptive)
# MAIN 2026-11-02 evening: B/R paused for LS-COLLECT (frozen B u3 policy on TRAIN worlds, plain
# search logging full pools on 4 GPU slots, exact SUMO branches for top-8 + policy = k=8 gold).
REGISTRATION_NAME = 'LS_COLLECT_REGISTRATION.json'
BANK_NAME = 'BANK_STATUS.json'          # B-REPLAY verified-plan bank (MAIN 2026-11-02 evening)
LOCAL_REGISTRATION_ROOTS = ('E:/agent-work/c4-coopnative/docs/paper_results', 'E:/agent-work/_sol')
DEADLINE = dt.datetime(2026, 11, 3, 10, 0, tzinfo=dt.timezone(dt.timedelta(hours=8)))
REFRESH_S = 3600.
# Measured warm fast-path full-medium search wall per query (docs/paper_results/COOPMWM.md, MAIN
# throughput table: 2 CAVs gpuA 559.567 s, 4 CAVs gpuR 1029.964 s, 6 CAVs gpuA 898.275 s);
# 3 and 5 CAVs linearly interpolated.
SEARCH_WALL_S = {2: 559.567, 3: 794.766, 4: 1029.964, 5: 964.120, 6: 898.275}
DIRS = ['requests', 'responses', 'proposer_receipts', 'verifications']


def b_updates():
    out = lr.ssh('cpuQ', f"cd {QROOT} && ls B_UPDATE_[0-9][0-9][0-9].json 2>/dev/null; true").decode().split()
    return sorted(int(x[9:12]) for x in out if x.startswith('B_UPDATE_'))


def fetch_updates(cache):
    """Read-only copy of the B update receipts (timing/GPU-busy only are used)."""
    local = Path(cache)/'updates'; local.mkdir(parents=True, exist_ok=True)
    return _fetch_glob(local)


def _fetch_glob(local):
    import io
    import tarfile
    raw = lr.ssh('cpuQ', f"cd {QROOT} && tar -cf - B_UPDATE_*.json 2>/dev/null; true")
    n = 0
    if raw:
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.startswith('B_UPDATE_') and m.name.endswith('.json'):
                    (local/m.name).write_bytes(tf.extractfile(m).read()); n += 1
    return n


def screen_share(sizes, top_k=4):
    """Share of a search's WM futures spent in the screen (vs the K8 final re-score)."""
    out = []
    for m in sizes:
        sc = ms.scaling(m, ms.canonical_knots(), top_k)
        screen = (sc['population']+2)*ms.REGISTERED['screen_K']*ms.REGISTERED['iterations']
        final = (2*top_k+2)*ms.REGISTERED['final_K']
        out.append(screen/(screen+final))
    return sum(out)/len(out) if out else None


def wallclock(cache, eval_saving, sizes):
    """Expected wall-clock saving per B update if the GPU WM queue is the bottleneck.

    For each update receipt: busy share = sum of completed inference seconds over the
    GPU slots / (slots x update wall). Saving ~ busy share x screen share x
    evaluation saving. This is an upper bound when episodes block on queries, and
    zero when the GPUs are idle.
    """
    rows = []
    for f in sorted((Path(cache)/'updates').glob('B_UPDATE_[0-9][0-9][0-9].json')):
        d = json.loads(f.read_text()); g = d.get('gpu_busy') or {}
        slots = g.get('slots') or {}
        if not slots or not d.get('wall_s'):
            continue
        busy = sum(v.get('completed_inference_s', 0.) for v in slots.values())/(len(slots)*float(d['wall_s']))
        rows.append(dict(update=d.get('update'), wall_s=float(d['wall_s']), gpu_busy_share=busy))
    sh4, sh8 = screen_share(sizes, 4), screen_share(sizes, 8)
    out = dict(updates=rows, screen_share_top4=sh4, screen_share_top8=sh8,
               basis='saving_fraction ~ gpu_busy_share x screen_share x evaluation_saving (GPU-bound upper bound)')
    if rows and eval_saving is not None and sh4 is not None:
        frac = [r['gpu_busy_share']*sh4*eval_saving for r in rows]
        out.update(expected_wall_saving_fraction=sum(frac)/len(frac),
                   expected_wall_saving_s=sum(f*r['wall_s'] for f, r in zip(frac, rows))/len(rows),
                   top8_final_overhead_note='top-8 doubles the K8 finalists (16 vs 8): screen share '
                                            f'{sh4:.3f} -> {sh8:.3f} of WM futures')
    return out


def expected_saving(curve):
    """Per-mode evaluation saving vs an independent plain search; recommended mode =
    largest saving among modes whose final elite quality is non-inferior to plain_alt."""
    if not isinstance(curve, dict) or not curve.get('summary'):
        return dict(available=False, reason=str(curve)), None
    budget = curve['mean_budget']; out = {}
    for mode in ('seeds', 'warm', 'both', 'warm_it1', 'warm_it2'):
        e = curve['summary'].get(mode, {}).get('elite')
        if not e:
            continue
        q = e.get('quality_minus_plain_alt') or {}
        used = curve['summary'][mode]['evaluations_fraction']
        save = (1-used) if used < .999 else -(e.get('evals_minus_plain_alt') or {}).get('mean', 0.)/budget
        out[mode] = dict(evaluation_saving_fraction=save, evaluations_used_fraction=used,
                         quality_minus_plain_alt=q, noninferior=bool(q and q.get('ci_lo', -1) > NONINFERIOR))
    ok = {m: v for m, v in out.items() if v['noninferior'] and v['evaluation_saving_fraction'] > 0}
    best = max(ok, key=lambda m: ok[m]['evaluation_saving_fraction']) if ok else None
    return dict(available=True, modes=out, recommended_mode=best,
                note='surrogate replay of held-out full WM pools; plain_alt = independent plain search'), best


def _paths(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            _paths(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _paths(v, out)
    elif isinstance(obj, str) and obj.startswith('/') and not lr._forbidden(obj):
        out.add(obj.rstrip('/'))
    return out


def find_registration(cache):
    """LS_COLLECT_REGISTRATION.json: local (c4-coopnative docs, _sol; read-only) or cpuQ
    (read-only find). Returns (registration dict or None, where)."""
    for root in LOCAL_REGISTRATION_ROOTS:
        hits = sorted(Path(root).rglob(REGISTRATION_NAME)) if Path(root).exists() else []
        hits = [h for h in hits if not lr._forbidden(h)]
        if hits:
            return json.loads(hits[-1].read_text(encoding='utf-8')), str(hits[-1])
    prune = ' -o '.join(f"-iname '*{f}*'" for f in lr.FORBIDDEN)
    try:
        out = lr.ssh('cpuQ', f"find /home/ubuntu -maxdepth 3 \\( {prune} \\) -prune -o -name {REGISTRATION_NAME} "
                             f"-print 2>/dev/null | sort | tail -1; true").decode().strip()
        if out:
            local = Path(cache)/'registration'; local.mkdir(parents=True, exist_ok=True)
            raw = lr.ssh('cpuQ', f"cat '{out}'"); (local/REGISTRATION_NAME).write_bytes(raw)
            return json.loads(raw.decode('utf-8')), 'cpuQ:'+out
    except (RuntimeError, ValueError):
        pass
    return None, None


def ls_sources(registration):
    """Collection sources = registered absolute paths: /home/ubuntu/... on cpuQ, /root/... on gpuA
    and gpuR (each probed read-only); only requests/responses/proposer_receipts/verifications."""
    paths = sorted(_paths(registration, set()))
    src = {}
    for path in paths:
        hosts = ['cpuQ'] if path.startswith('/home/ubuntu') else ['gpuA', 'gpuR'] if path.startswith('/root') else []
        root = path if not path.endswith('.json') else path.rsplit('/', 1)[0]
        for h in hosts:
            src.setdefault(h, dict(roots=[], dirs=DIRS, name_glob='*'))
            if root not in src[h]['roots']:
                src[h]['roots'].append(root)
    return src


def merged_sources(cache):
    reg, where = find_registration(cache)
    src = {h: dict(v, roots=list(v['roots']), dirs=list(v['dirs'])) for h, v in lr.SOURCES.items()}
    try:   # B-REPLAY bank: its directory (and anything it lists) is a collection root
        prune = ' -o '.join(f"-iname '*{f}*'" for f in lr.FORBIDDEN)
        banks = lr.ssh('cpuQ', f"find /home/ubuntu -maxdepth 3 \( {prune} \) -prune -o -name {BANK_NAME} -print "
                               f"2>/dev/null; true").decode().split()
        for b in banks:
            root = b.rsplit('/', 1)[0]
            extra = {}
            try:
                extra = json.loads(lr.ssh('cpuQ', f"cat '{b}'").decode('utf-8'))
            except (RuntimeError, ValueError):
                pass
            src.setdefault('cpuQ#bank', dict(host='cpuQ', roots=[], dirs=DIRS, name_glob='*'))
            for path in [root]+[p for p in sorted(_paths(extra, set())) if p.startswith('/home/ubuntu')]:
                path = path.rsplit('/', 1)[0] if path.endswith('.json') else path
                if path not in src['cpuQ#bank']['roots']:
                    src['cpuQ#bank']['roots'].append(path)
            for h in ('gpuA', 'gpuR'):
                roots = [p for p in sorted(_paths(extra, set())) if p.startswith('/root')]
                if roots:
                    src[h+'#bank'] = dict(host=h, roots=roots, dirs=DIRS, name_glob='*')
            where = (where or '')+f' bank:{b}'
    except RuntimeError:
        pass
    if reg:
        for h, v in ls_sources(reg).items():
            # registered LS-COLLECT roots are searched under their own path (any name)
            src[h+'#ls'] = dict(v, host=h)
    return src, where


def search_time_saving(saving_fraction, sizes):
    """Expected GPU search-time saving per search at equal elite quality (GPU-bound):
    measured wall per query x screen share of WM futures x screen-evaluation saving."""
    if not sizes or saving_fraction is None:
        return dict(available=False)
    per = []
    for m in sizes:
        t = SEARCH_WALL_S[min(6, max(2, m))]
        per.append(dict(m=m, wall_s=t, saved_s=t*screen_share([m], 4)*saving_fraction))
    by = {}
    for r in per:
        by.setdefault(r['m'], []).append(r['saved_s'])
    return dict(available=True, evaluation_saving_fraction=saving_fraction,
                mean_saved_s_per_search=sum(r['saved_s'] for r in per)/len(per),
                mean_wall_s_per_search=sum(r['wall_s'] for r in per)/len(per),
                saved_s_by_group_size={str(k): sum(v)/len(v) for k, v in sorted(by.items())},
                basis='SEARCH_WALL_S (COOPMWM.md measured) x screen share (top-4) x saving; GPU-bound, '
                      'final K8 re-score unchanged; top-8 adds 8 finalists x K8 futures')


def ready(res):
    c, v = res.get('offline_check') or {}, res.get('verification_check') or {}
    wm = bool(c.get('passed') and c.get('enough_data') and res.get('registered_minimum_met'))
    ver = bool(v.get('passed') and v.get('enough_data'))
    return (wm or ver) and not v.get('harm'), wm, ver


def git_commit(paths, msg):
    subprocess.run(['git', '-C', str(REPO), 'add', '--']+[str(p) for p in paths if Path(p).exists()], check=False)
    subprocess.run(['git', '-C', str(REPO), 'commit', '-q', '-m',
                    msg+'\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'], check=False)


def refresh(cache, tag, sources=None):
    res = lr.run(cache, REPORT, fetch_remote=True, sources=sources)
    try:
        fetch_updates(cache)
    except Exception as e:   # noqa: BLE001 - timing is descriptive; never blocks the check
        res['update_fetch_error'] = str(e)[:200]
    saving, mode = expected_saving(res.get('evaluations_curve'))
    sizes = [len(r['members']) for r in lr.load_cache(cache)[0] if lr.is_full_pool(r)]
    sv = saving['modes'][mode]['evaluation_saving_fraction'] if mode else 0.
    res['expected_saving'] = saving
    res['wallclock'] = wallclock(cache, sv, sizes)
    res['search_time_saving'] = search_time_saving(sv, sizes)
    ok, wm, ver = ready(res)
    res['ready'] = dict(ready=ok, wm_pool_check=wm, verification_check=ver)
    lr.write_report(res, REPORT)
    with REPORT.open('a', encoding='utf-8') as f:
        f.write(f"\n## Watch refresh {tag}\n- READY: **{ok}** (WM-pool {wm}, verification {ver}); expected saving "
                f"{json.dumps(saving.get('modes', {}).get(mode) if mode else 'none non-inferior')}; wall-clock "
                f"{json.dumps({k: v for k, v in res['wallclock'].items() if k != 'updates'})}; search-time saving "
                f"{json.dumps(res['search_time_saving'])}; nested best-of-4 vs 8 "
                f"{json.dumps(res.get('nested_best_of_4_vs_8'), default=str)}.\n")
    files = [REPORT, REPORT.with_suffix('.json')]
    if ok:
        body = dict(
            generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), refresh=tag,
            owner_approval='MAIN 2026-11-02: owner approves switching B to the learned search on condition it PASSES '
                           'the registered offline check (prereg 9b)',
            checkpoint=res.get('checkpoint'), passed=dict(wm_pool_check=wm, verification_check=ver),
            wm_pool_check=res.get('offline_check'), verification_check=res.get('verification_check'),
            data=dict(full_pool_searches=res.get('full_pool_searches'), worlds=res.get('worlds'),
                      verified_searches=res.get('verified_searches'), post_fix_searches=res.get('post_fix_searches'),
                      command_mirror=res.get('command_mirror')),
            expected_evaluation_saving=saving, recommended_mode=mode,
            learned_search_config=dict(mode=mode, top_k=8, generic_share=.2, novelty_share=.1, novelty_oversample=4),
            rank5to8_best_share=res.get('rank5to8'), nested_best_of_4_vs_8=res.get('nested_best_of_4_vs_8'),
            wallclock=res['wallclock'], search_time_saving=res['search_time_saving'],
            integration='mppi_runtime.plan: model, meta = plan_proposer.load_checkpoint(<cache ckpt>); verify '
                        'meta["pt_sha256"]; search(..., **plan_proposer.learned_search_kwargs(record, group, model, '
                        'regime, mode, policy, bounds))',
            checkpoint_path=str(sorted((Path(cache)/'checkpoints').glob('*.pt'))[-1]))
        READY.write_text(json.dumps(body, indent=1, default=str)); files.append(READY)
    git_commit(files, f'learned search watch refresh {tag}: READY={ok}')
    return ok, res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cache', default='E:/agent-work/_plan_proposer_cache')
    ap.add_argument('--poll-s', type=float, default=600.)
    ap.add_argument('--max-hours', type=float, default=30.)
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args(argv)
    state = Path(a.cache)/'watch_state.json'
    seen = json.loads(state.read_text()).get('updates', []) if state.exists() else []
    end = min(time.time()+3600*a.max_hours, DEADLINE.timestamp())
    first, last, reg_seen = True, 0., None
    while True:
        deadline = time.time() >= end
        try:
            ups = b_updates()
        except RuntimeError as e:
            print('poll error', e, flush=True)
            if not deadline:
                time.sleep(a.poll_s); continue
            ups = seen
        sources, where = merged_sources(a.cache)
        due = first or ups != seen or (where and where != reg_seen) or time.time()-last >= REFRESH_S or deadline
        if due:
            tag = (f"LS@{dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%m%d_%H%M')}+08"
                   if where else f"B_u{max(ups) if ups else 0:03d}")+('_DEADLINE' if deadline else '')
            ok, res = refresh(a.cache, tag, sources)
            seen, last, reg_seen = ups, time.time(), where
            state.write_text(json.dumps(dict(updates=seen, last=tag, ready=ok, registration=where)))
            print(json.dumps(dict(tag=tag, ready=res['ready'], full_pools=res.get('full_pool_searches'),
                                  verified=res.get('verified_searches'), registration=where)), flush=True)
            if ok:
                return 0
            if deadline:
                return 4
        first = False
        if a.once:
            return 1
        time.sleep(a.poll_s)


if __name__ == '__main__':
    raise SystemExit(main())
