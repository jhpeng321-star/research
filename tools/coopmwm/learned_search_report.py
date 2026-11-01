"""Real-data path for the learned search: read-only collection, training, checks, report.

    python -m tools.coopmwm.learned_search_report [--no-fetch] [--cache DIR] [--report PATH]

1. COLLECT (read-only). Over SSH (BatchMode, bounded connection-only retries)
   it runs only ``find`` and ``tar -c`` on the remote hosts, never writes there,
   and copies missing ``*.json`` files into a local cache under
   ``<cache>/<host>/<remote path>``. Sources:
   - cpuQ plan queues ``<root>/requests`` + ``<root>/responses``. These are arm-B
     MPPI searches; arm R uses ``centre_*`` and is never collected.
   - any ``proposer_receipts/`` directory on cpuQ/gpuA/gpuR. These are full-pool
     ``RecordingEvaluator`` receipts, once the live service logs them.
   Directories whose path contains sealed/holdout/heldout/cal_split/chunk5/j4b/inc1
   are pruned before descent and never opened.
2. LOAD. Requests are paired with responses (``plan_proposer.receipt_from_service``).
   Only TRAIN worlds 986000..986071 are kept; dev/check/final panels are dropped.
   SUMO-realized advantages come from the ``search_learning.entries`` embedded
   in later requests.
3. TRAIN + OFFLINE CHECK (frozen ``plan_proposer.REGISTERED``, world-grouped split).
   This is informational only: activation needs the registered minimum, a PASS
   AND explicit owner approval recorded by MAIN (PLAN_PROPOSER_PREREG.md). The
   script never activates anything.
4. EVALUATIONS CURVE (``learned_search_eval``) on held-out full-pool receipts.
   It runs only when >= 48 full-pool searches exist.
5. REPORT: ``docs/paper_results/LEARNED_SEARCH_REPORT.md`` (+ ``.json``) and a
   checkpoint in the cache, with SHA256s.
"""
import argparse
import datetime as dt
import io
import json
import shlex
import subprocess
import tarfile
import time
from pathlib import Path

from . import learned_search_eval as le
from . import plan_proposer as pp

FORBIDDEN = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1')
TRAIN_WORLDS = range(986000, 986072)
SOURCES = dict(
    cpuQ=dict(roots=['/home/ubuntu'], dirs=['requests', 'responses', 'proposer_receipts'], name_glob='coopmwm2026*'),
    gpuA=dict(roots=['/root'], dirs=['proposer_receipts'], name_glob='coopmwm2026*'),
    gpuR=dict(roots=['/root'], dirs=['proposer_receipts'], name_glob='coopmwm2026*'))
MIN_FULL_POOL = 48
MIN_TRAIN_INFO = dict(searches=24, worlds=4)          # below this: counts only, no training
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15']
CONN_ERRORS = ('Connection closed', 'Connection reset', 'Connection timed out', 'timed out',
               'Could not resolve', 'kex_exchange', 'Broken pipe')


def _forbidden(path):
    p = str(path).lower()
    return any(f in p for f in FORBIDDEN)


def ssh(host, cmd, stdin=None, tries=3):
    """Read-only remote command; retries ONLY connection-level failures (bounded backoff)."""
    for i in range(tries):
        r = subprocess.run(SSH+[host, cmd], input=stdin, capture_output=True, timeout=600)
        err = r.stderr.decode('utf-8', 'replace')
        if r.returncode == 255 and any(e in err for e in CONN_ERRORS) and i < tries-1:
            time.sleep(10*(i+1)); continue
        if r.returncode != 0:
            raise RuntimeError(f'{host}: exit {r.returncode}: {err.strip()[:300]}')
        return r.stdout
    raise RuntimeError(f'{host}: connection retries exhausted')


def remote_dirs(host, src):
    prune = ' -o '.join(f"-iname '*{f}*'" for f in FORBIDDEN)
    names = ' -o '.join(f"-name {shlex.quote(d)}" for d in src['dirs'])
    roots = ' '.join(shlex.quote(r) for r in src['roots'])
    cmd = (f"find {roots} -mindepth 1 -maxdepth 4 \\( {prune} \\) -prune -o -type d \\( {names} \\) "
           f"-path {shlex.quote('*/'+src['name_glob']+'/*')} -print 2>/dev/null; true")
    return [d for d in ssh(host, cmd).decode().split() if d and not _forbidden(d)]


def fetch(host, remote, cache):
    local = Path(cache)/host/remote.strip('/')
    local.mkdir(parents=True, exist_ok=True)
    listing = ssh(host, f"find {shlex.quote(remote)} -maxdepth 1 -type f -name '*.json' -printf '%f\\n'").decode().split()
    have = {p.name for p in local.glob('*.json')}
    missing = [f for f in listing if f not in have and not _forbidden(f)]
    for i in range(0, len(missing), 500):
        chunk = missing[i:i+500]
        raw = ssh(host, f"cd {shlex.quote(remote)} && tar -cf - -T -", stdin=('\n'.join(chunk)+'\n').encode())
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            for m in tf.getmembers():
                name = Path(m.name).name
                if m.isfile() and name in chunk:
                    (local/name).write_bytes(tf.extractfile(m).read())
    return dict(host=host, remote=remote, listed=len(listing), fetched=len(missing))


def collect(cache, sources=SOURCES):
    log = []
    for host, src in sources.items():
        try:
            for d in remote_dirs(host, src):
                log.append(fetch(host, d, cache))
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            log.append(dict(host=host, error=str(e)[:300]))
    return log


def load_cache(cache, train_worlds=TRAIN_WORLDS):
    """Pair cached requests/responses per queue root; load full-pool receipts."""
    cache = Path(cache); receipts, stats = [], dict(pairs=0, skipped={}, non_train=0, full_pool_files=0)
    entries = {}
    for req_dir in sorted(cache.rglob('requests')):
        if _forbidden(req_dir) or not (req_dir.parent/'responses').is_dir():
            continue
        for req in sorted(req_dir.glob('*.json')):
            resp = req_dir.parent/'responses'/req.name
            if not resp.exists():
                continue
            request = json.loads(req.read_text())
            for e in (request.get('search_learning') or {}).get('entries') or []:
                entries[e.get('plan')] = e
            stats['pairs'] += 1
            rec, why = pp.receipt_from_service(request, json.loads(resp.read_text()))
            if rec is None:
                stats['skipped'][why] = stats['skipped'].get(why, 0)+1; continue
            receipts.append(rec)
    by_id = {r['search_id']: r for r in receipts}
    for d in sorted(cache.rglob('proposer_receipts')):
        if _forbidden(d):
            continue
        got, _ = pp.load_receipts([d]); stats['full_pool_files'] += len(got)
        for r in got:
            by_id[r['search_id']] = r                   # full pool supersedes the top-k-only view
    out = []
    for r in by_id.values():
        try:
            ok = int(r.get('world_id')) in train_worlds
        except (TypeError, ValueError):
            ok = False
        (out.append(r) if ok else stats.__setitem__('non_train', stats['non_train']+1))
    pp.attach_sumo(out, list(entries.values()))
    stats['sumo_entries'] = len(entries)
    return out, stats


def is_full_pool(r):
    return any(c.get('stage', 'screen') == 'screen' for c in r.get('candidates') or [])


def run(cache, report, fetch_remote=True):
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    when = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    fetch_log = collect(cache) if fetch_remote else [dict(note='no fetch; cache only')]
    receipts, stats = load_cache(cache)
    searches, why = pp.build_dataset(receipts)
    with_elites = [s for s in searches if s['elites']]
    full = [r for r in receipts if is_full_pool(r)]
    res = dict(generated_utc=when, cache=str(cache), fetch=fetch_log, load=stats, receipts=len(receipts),
               full_pool_receipts=len(full), searches=len(searches), searches_with_elites=len(with_elites),
               worlds=len({s['world'] for s in searches}), dataset_skips=len(why),
               activation='NOT ACTIVATED: requires registered minimum + offline PASS + explicit owner approval via MAIN')
    if len(searches) >= MIN_TRAIN_INFO['searches'] and res['worlds'] >= MIN_TRAIN_INFO['worlds']:
        tr, va, vw = pp.world_split(searches)
        model, info = pp.train(tr)
        check = pp.offline_check(model, va)
        ck = pp.save_checkpoint(model, dict(validation_worlds=vw, dataset_sha256=pp._digest(
            sorted(r['search_id'] for r in receipts)), offline_check={k: v for k, v in check.items() if k != 'per_search'}),
            cache/'checkpoints'/f'proposer_{when[:19].replace(":", "")}')
        res.update(train=dict(train_searches=info['train_searches'], selected_step=info['selected_step']),
                   offline_check={k: v for k, v in check.items() if k != 'per_search'},
                   checkpoint=dict(pt_sha256=ck['pt_sha256'], json_sha256=ck['json_sha256']),
                   registered_minimum_met=len(with_elites) >= 200 and res['worlds'] >= 32)
        if len(full) >= MIN_FULL_POOL:
            held = [r for r in full if str(r.get('world_id')) in set(vw)]
            curve = le.evaluations_curve(model, held)
            res['evaluations_curve'] = {k: v for k, v in curve.items() if k not in ('per_search', 'cfg')}
        else:
            res['evaluations_curve'] = f'not run: {len(full)} full-pool receipts < {MIN_FULL_POOL}'
    else:
        res['train'] = f"not run: need >= {MIN_TRAIN_INFO['searches']} B searches from >= {MIN_TRAIN_INFO['worlds']} TRAIN worlds"
    write_report(res, Path(report))
    return res


def _fmt(x, nd=3):
    return f'{x:.{nd}f}' if isinstance(x, float) else str(x)


def write_report(res, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix('.json').write_text(json.dumps(res, indent=1, default=str))
    L = ['# LEARNED SEARCH REPORT (plan proposer; informational, never auto-activated)', '',
         f"Generated {res['generated_utc']} by `python -m tools.coopmwm.learned_search_report` (CPU only; no SUMO/GPU).",
         f"**{res['activation']}.**", '',
         '## Data (read-only copies; TRAIN worlds 986000-986071 only)',
         f"- Collection: {sum(1 for f in res['fetch'] if 'remote' in f)} remote dirs, "
         f"{sum(f.get('fetched', 0) for f in res['fetch'])} new files; errors: "
         f"{[f['host']+': '+f['error'][:80] for f in res['fetch'] if 'error' in f] or 'none'}.",
         f"- Request/response pairs {res['load']['pairs']}; usable B search receipts {res['receipts']} "
         f"(skipped {res['load']['skipped'] or 'none'}; non-TRAIN dropped {res['load']['non_train']}); "
         f"SUMO entries {res['load']['sumo_entries']}.",
         f"- Searches {res['searches']} ({res['searches_with_elites']} with elites) from {res['worlds']} worlds; "
         f"full-pool receipts {res['full_pool_receipts']} (curve needs >= {MIN_FULL_POOL}).", '']
    if isinstance(res.get('train'), str):
        L += ['## Training / offline check', f"- {res['train']}.", '']
    else:
        c = res['offline_check']
        L += ['## Offline check (held-out worlds; vs the search\'s own initial distribution)',
              f"- Train searches {res['train']['train_searches']} (selected step {res['train']['selected_step']}); "
              f"held-out {c.get('searches')} searches / {c.get('worlds')} worlds; enough data: {c.get('enough_data')}; "
              f"registered minimum (200 searches, 32 worlds) met: {res['registered_minimum_met']}.",
              f"- PASS: **{c.get('passed')}**. LL gain {_fmt(c.get('loglik_gain', {}).get('mean'))} nats/elite "
              f"CI {c.get('loglik_gain', {}).get('ci95')}; nearest-elite distance change vs initial "
              f"{_fmt(c.get('nearest_elite_distance_change', {}).get('mean'))} m/s^2 CI "
              f"{c.get('nearest_elite_distance_change', {}).get('ci95')}; vs centre "
              f"{_fmt(c.get('distance_change_vs_centre', {}).get('mean'))} CI {c.get('distance_change_vs_centre', {}).get('ci95')}.",
              f"- Checkpoint (local cache) .pt SHA256 `{res['checkpoint']['pt_sha256']}`.", '']
        ev = res.get('evaluations_curve')
        L += ['## Evaluations needed to reach the plain search\'s final elite quality (surrogate replay)']
        if isinstance(ev, str) or not ev or not ev.get('summary'):
            L += [f"- {ev if isinstance(ev, str) else 'no held-out full-pool search usable'}."]
        else:
            L += [f"- {ev['searches']} held-out searches / {ev['worlds']} worlds; mean screen budget {_fmt(ev['mean_budget'], 1)}; "
                  f"median surrogate LOO R^2 {_fmt(ev['median_surrogate_loo_r2'], 2)}.",
                  '', '| mode | evaluations used (x plain) | evals to plain elite quality (x plain budget) | reached '
                  '| mode - plain evals (95% CI) | final elite quality (95% CI) | elite coverage | strict max reached |',
                  '|---|---|---|---|---|---|---|---|']
            for m, s in ev['summary'].items():
                e = s['elite']; d, q = e['minus_plain'], e['final_quality']
                L.append(f"| {m} | {_fmt(s['evaluations_fraction'], 2)} | {_fmt(e['mean_fraction_of_plain_budget'], 2)} | "
                         f"{_fmt(e['reached_rate'], 2)} | {_fmt(d['mean'], 1)} [{_fmt(d['ci_lo'], 1)}, {_fmt(d['ci_hi'], 1)}] | "
                         f"{_fmt(q['mean'], 2)} [{_fmt(q['ci_lo'], 2)}, {_fmt(q['ci_hi'], 2)}] | {_fmt(s['elite_coverage'], 2)} | "
                         f"{_fmt(s['reached_rate'], 2)} |")
            L += ['', 'Quality: 0 = centre seed (unchanged policy), 1 = plain 3-iteration final best. Horizons are '
                  'identical in every mode (screen K4/H300, final K8/H_final); warm_it1/2 use fewer CEM iterations only. '
                  'Warm modes keep 20% generic + 10% novelty slices.']
        L += ['']
    L += ['## Caveats', '- Surrogate replays are noise-free fits of each search\'s own WM screen pool; they estimate '
          'search efficiency, not SUMO benefit.',
          '- Top-k-only service receipts carry no screened pool; the evaluations curve needs `RecordingEvaluator` logging '
          '(`<queue_root>/proposer_receipts/<state_id>.json`).',
          '- The script never activates the proposer and never writes to remote hosts.']
    path.write_text('\n'.join(L)+'\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cache', default='E:/agent-work/_plan_proposer_cache')
    ap.add_argument('--report', default=str(Path(__file__).resolve().parents[2]/'docs/paper_results/LEARNED_SEARCH_REPORT.md'))
    ap.add_argument('--no-fetch', action='store_true')
    a = ap.parse_args(argv)
    res = run(a.cache, a.report, not a.no_fetch)
    print(json.dumps({k: v for k, v in res.items() if k in ('receipts', 'full_pool_receipts', 'searches', 'worlds',
                                                         'offline_check', 'evaluations_curve', 'train')},
                     indent=1, default=str)[:4000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
