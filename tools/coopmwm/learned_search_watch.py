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


def ready(res):
    c, v = res.get('offline_check') or {}, res.get('verification_check') or {}
    wm = bool(c.get('passed') and c.get('enough_data') and res.get('registered_minimum_met'))
    ver = bool(v.get('passed') and v.get('enough_data'))
    return (wm or ver) and not v.get('harm'), wm, ver


def git_commit(paths, msg):
    subprocess.run(['git', '-C', str(REPO), 'add', '--']+[str(p) for p in paths if Path(p).exists()], check=False)
    subprocess.run(['git', '-C', str(REPO), 'commit', '-q', '-m',
                    msg+'\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'], check=False)


def refresh(cache, tag):
    res = lr.run(cache, REPORT, fetch_remote=True)
    try:
        fetch_updates(cache)
    except Exception as e:   # noqa: BLE001 - timing is descriptive; never blocks the check
        res['update_fetch_error'] = str(e)[:200]
    saving, mode = expected_saving(res.get('evaluations_curve'))
    sizes = [len(r['members']) for r in lr.load_cache(cache)[0] if lr.is_full_pool(r)]
    sv = saving['modes'][mode]['evaluation_saving_fraction'] if mode else 0.
    res['expected_saving'] = saving
    res['wallclock'] = wallclock(cache, sv, sizes)
    ok, wm, ver = ready(res)
    res['ready'] = dict(ready=ok, wm_pool_check=wm, verification_check=ver)
    lr.write_report(res, REPORT)
    with REPORT.open('a', encoding='utf-8') as f:
        f.write(f"\n## Watch refresh {tag}\n- READY: **{ok}** (WM-pool {wm}, verification {ver}); expected saving "
                f"{json.dumps(saving.get('modes', {}).get(mode) if mode else 'none non-inferior')}; wall-clock "
                f"{json.dumps({k: v for k, v in res['wallclock'].items() if k != 'updates'})}.\n")
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
            rank5to8_best_share=res.get('rank5to8'), wallclock=res['wallclock'],
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
    end = time.time()+3600*a.max_hours
    first = True
    while time.time() < end:
        try:
            ups = b_updates()
        except RuntimeError as e:
            print('poll error', e, flush=True); time.sleep(a.poll_s); continue
        if first or ups != seen:
            tag = f"B_u{max(ups) if ups else 0:03d}"
            ok, res = refresh(a.cache, tag)
            seen = ups; state.write_text(json.dumps(dict(updates=seen, last=tag, ready=ok)))
            print(json.dumps(dict(tag=tag, ready=res['ready'], full_pools=res.get('full_pool_searches'),
                                  verified=res.get('verified_searches'))), flush=True)
            if ok:
                return 0
        first = False
        if a.once:
            return 1
        time.sleep(a.poll_s)
    return 3


if __name__ == '__main__':
    raise SystemExit(main())
