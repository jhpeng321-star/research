"""Exclusive bounded saved-evidence collector; NEVER dispatches inference.

Preserves existing remote workers/monitors. Collects results, scores complete K8
ensembles, publishes the08:00 interim/fixed09:30 engineering freeze/final report,
and commits only its own publication files. Foreign work or errors stop it.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.rbint.noon_report import render
from tools.rbint.local_resources import child_incarnation, local_resources
from tools.rbeval.core import digest, load, save_new

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'docs/paper_results/WMPHYS_MORNING.md'
HANDOFF = ROOT/'HANDOFF.md'
PUBLISH = ROOT/'docs/rbint/noon'
EVIDENCE = ROOT/'runs/rbint_evidence'
STATE = ROOT/'runs/rbint_noon'
FREEZE = datetime.fromisoformat('2026-11-01T09:30:00+08:00').timestamp()
INTERIM = datetime.fromisoformat('2026-11-01T08:00:00+08:00').timestamp()
FINAL = datetime.fromisoformat('2026-11-01T11:35:00+08:00').timestamp()
DEADLINE = datetime.fromisoformat('2026-11-01T11:45:00+08:00').timestamp()
MODEL_SOURCE = '22cc32f65a80af482a1a277c5f22377fa727d0a3'
VARIANTS = {
    'A2': ('gpuA', '/root/rbeval20261031/runa/WM-PHYS-A2'),
    'B2': ('gpuR', '/root/rbeval20261031/runb/WM-PHYS-B2'),
    'C2': ('gpuA', '/root/rbeval20261031/runa/WM-PHYS-C2'),
    'FULL-v3': ('gpuA', '/root/rbeval20261031/int/WM-PHYS-FULL-v3-gpu'),
    'FULL-v4': ('gpuA', '/root/rbeval20261031/int/WM-PHYS-FULL-v4-gpu')}


def git(*args):
    p = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, timeout=60)
    if p.returncode:
        raise RuntimeError(p.stdout+p.stderr)
    return p.stdout.rstrip()


def guard(head=None, allowed=()):
    if git('branch', '--show-current') != 'c4/rb-int':
        raise RuntimeError('foreign branch: collector stopped')
    if head is not None and git('rev-parse', 'HEAD') != head:
        raise RuntimeError('HEAD changed during analysis: preserve foreign work')
    if git('diff', '--cached', '--name-only'):
        raise RuntimeError('foreign staged files: collector stopped')
    names = git('status', '--porcelain', '--untracked-files=all')
    for line in names.splitlines():
        name = line[3:].replace('\\', '/').strip('"')
        if not any(name == p or name.startswith(p.rstrip('/')+'/') for p in allowed):
            raise RuntimeError('foreign/uncommitted work: '+name)


def command(argv, directory, name, *, cwd=ROOT, optional_missing=False):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    args = [sys.executable, '-B', *map(str, argv)]
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 0
    with (directory/(name+'.log')).open('x', encoding='utf-8') as log:
        child = subprocess.Popen(args, cwd=cwd, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, creationflags=flags)
        save_new(directory/(name+'_PROCESS.json'), dict(pid=child.pid,
            **child_incarnation(child), argv=args, analysis_only=True))
        limit = max(1, min(240, DEADLINE-time.time()))
        try:
            rc = child.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            # This handle belongs exclusively to the CPU analysis child just
            # created above, never an inferred/peer PID or process-name match.
            child.terminate()
            child.wait(timeout=15)
            raise RuntimeError('own CPU analysis exceeded bounded publication deadline')
    save_new(directory/(name+'_EXIT.json'), dict(pid=child.pid, returncode=rc))
    if rc:
        text = (directory/(name+'.log')).read_text(encoding='utf-8')
        if optional_missing and ('No such file' in text or 'AssertionError' in text):
            return False
        raise RuntimeError(name+' failed: '+text[-3000:])
    return True


def fingerprint(snapshot):
    receipt = load(Path(snapshot)/'SNAPSHOT_RECEIPT.json')
    names = receipt['hashes']
    worlds = {}
    for name, sha in names.items():
        part = Path(name).name
        if part.startswith('w') and '_k' in part and part.endswith('.json'):
            seed, future = part[1:-5].split('_k')
            worlds.setdefault(int(seed), set()).add(int(future))
    complete = sorted(s for s, futures in worlds.items() if futures == set(range(8)))
    selected = {name: sha for name, sha in names.items() if name.endswith('IDENTITY.json') or
                any(Path(name).name.startswith(f'w{s}_k') for s in complete)}
    return dict(tasks=sum(len(k) for k in worlds.values()), complete=complete,
        sha256=hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest())


def snapshot(host, roots, target, round_dir, name, optional=False):
    ok = command(['-m', 'tools.rbint.snapshot', '--host', host, '--roots', *roots,
                  '--out', target], round_dir, name, optional_missing=optional)
    return fingerprint(target) if ok else None


def next_wake(now):
    future = [x for x in (INTERIM, FREEZE, FINAL) if x > now]
    return min([now+1200, *future])


def update_handoff(text, block):
    first, last = '<!-- NOON-COLLECTOR:BEGIN -->', '<!-- NOON-COLLECTOR:END -->'
    if first in text:
        before, rest = text.split(first, 1)
        _, after = rest.split(last, 1)
        return before+first+'\n'+block+'\n'+last+after
    return text.rstrip()+'\n\n'+first+'\n'+block+'\n'+last+'\n'


def collect(state, *, now=None):
    now = time.time() if now is None else now
    if now >= DEADLINE:
        raise RuntimeError('publication deadline expired')
    guard()
    resources = local_resources()
    if resources['cpu_percent'] > 75 or resources['available_bytes'] < 2**30:
        return dict(state, resource_wait=True)
    head = git('rev-parse', 'HEAD')
    stamp = datetime.fromtimestamp(now).astimezone().strftime('%Y%m%dT%H%M%S%z')
    target = PUBLISH/stamp
    target.mkdir()
    allowed = [str(target.relative_to(ROOT)).replace('\\', '/')]
    local_time = datetime.fromtimestamp(now).astimezone().isoformat(timespec='seconds')
    sources = [Path(__file__), Path(__file__).with_name('noon_report.py'),
               Path(__file__).with_name('compare.py'), Path(__file__).with_name('score_endpoints.py'),
               ROOT/'tools/rbeval/score.py']
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    publication = dict(local_time=local_time, label='final' if now >= FINAL else 'interim',
        source_head=head, code_hashes=source_hashes, variants={}, read_only_remote=True,
        receipt_directory=str(target.relative_to(ROOT)).replace('\\', '/'))
    old_cache = state.get('cache', {})
    cache = dict(old_cache)
    baseline = state.get('baseline')
    if not baseline:
        path = EVIDENCE/('noon_wm2_'+stamp)
        info = snapshot('gpuR', [f'/root/rbeval20261031/wm2_shard{i}' for i in range(4)],
                        path, target, 'snapshot_wm2')
        if len(info['complete']) != 20:
            raise RuntimeError('original baseline expected complete; never invent missing outputs')
        baseline = str(path)
    endpoint_baseline = state.get('endpoint_baseline')
    endpoint_info = fingerprint(endpoint_baseline) if endpoint_baseline else None
    if endpoint_info is None or len(endpoint_info['complete']) < 20:
        path = EVIDENCE/('noon_wm2_hdv60_'+stamp)
        endpoint_info = snapshot('gpuR', [f'/root/rbeval20261031/int/WM2-HDV60/shard{i}' for i in range(3)],
                                 path, target, 'snapshot_wm2_hdv60')
        endpoint_baseline = str(path)
    for key, (host, remote) in VARIANTS.items():
        path = EVIDENCE/('noon_'+key+'_'+stamp)
        info = snapshot(host, [remote+f'/shard{i}' for i in range(3)], path, target,
                        'snapshot_'+key, optional=key == 'C2')
        if info is None:
            publication['variants'][key] = dict(status='NO_SAVED_SHARDS', tasks=0)
            continue
        item = dict(info, snapshot=str(path))
        identity = hashlib.sha256((info['sha256']+endpoint_info['sha256']).encode()).hexdigest()
        previous = old_cache.get(key)
        if previous and previous['identity'] == identity:
            item.update(previous['reports'])
            item['reused_identical_complete_ensemble_scores'] = True
        else:
            report = target/key
            report.mkdir()
            command(['-m', 'tools.rbint.compare', '--candidate', path, '--baseline', baseline,
                     '--out', report/'paired'], target, 'compare_'+key)
            item['comparison'] = str(report/'paired/COMPARISON.json')
            if info['complete']:
                command(['-m', 'tools.rbeval.score', '--inputs', path, '--allow-partial',
                         '--out', report/'distribution'], target, 'distribution_'+key)
                item['distribution'] = str(report/'distribution')
                command(['-m', 'tools.wmdecide', 'credibility', '--eval-root', ROOT,
                         '--inputs', path, '--allow-partial', '--out', report/'credibility'],
                         target, 'credibility_'+key)
                item['credibility'] = str(report/'credibility')
                if key.startswith('FULL-') and set(info['complete']) & set(endpoint_info['complete']):
                    command(['-m', 'tools.rbint.score_endpoints', '--candidate', path,
                        '--baseline', endpoint_baseline, '--factual', ROOT/'docs/rbint/hdv60/FACTUAL_065e9853.json',
                        '--out', report/'endpoints'], target, 'endpoints_'+key)
                    item['endpoints'] = str(report/'endpoints/HDV60_COMPARISON.json')
            cache[key] = dict(identity=identity, reports={n: item[n] for n in
                ('comparison', 'distribution', 'credibility', 'endpoints') if n in item})
        publication['variants'][key] = item
    event_now = time.time()
    publication['label'] = 'final' if event_now >= FINAL else 'interim'
    if event_now >= FREEZE and not state.get('freeze'):
        spec = load(ROOT/'docs/rbint/full_v4_gpu/SPEC.json')
        assert spec['stage']['commit'] == MODEL_SOURCE and spec['owner'] == 'S-RB-INT'
        freeze = target/'FREEZE.json'
        save_new(freeze, dict(unix=event_now, candidate='WM-PHYS-FULL-v4-gpu', source_commit=MODEL_SOURCE,
            recipe=spec['runtime']+'/configs/wm_phys.toml', immutable_stage=spec['stage'],
            evidence_reuse='same source; successful tasks retained, no reruns',
            basis='latest integrated engineering/causal settings/arrivals; no reserved-outcome parameter tuning',
            fitted_lc=False, qualification='not implied by engineering freeze'))
        state['freeze'] = str(freeze.relative_to(ROOT)).replace('\\', '/')
    publication['freeze'] = state.get('freeze') or 'PENDING'
    if event_now >= INTERIM and not state.get('interim08'):
        state['interim08'] = publication['receipt_directory']
    publication['interim08'] = state.get('interim08') or 'PENDING'
    save_new(target/'PUBLICATION.json', publication)
    # No source, HEAD, staged file, handoff or report may change while scoring.
    guard(head, allowed)
    assert all(digest(ROOT/p) == sha for p, sha in source_hashes.items()), 'collector source changed'
    text = render(publication)
    REPORT.write_text(text, encoding='utf-8')
    block = ('## Bounded noon collector\n'+local_time+'; read-only remote collection, no inference dispatch.\n'
        f"Latest publication `{publication['receipt_directory']}/PUBLICATION.json`; "
        f"08interim `{publication['interim08']}`,09:30freeze `{publication['freeze']}`.\n"
        'Source22cc32f6 FULL-v4 and prior A/B/C/FULL sources remain immutable.\n'
        'Scored-world-only fixed-stratum CIs; singleton uncertainty NA; prior CIs retracted.\n'
        'Existing remote supervisors retain own-PID11:30 deadline. Final collection11:35, report by11:45.\n'
        'Collector owns ONLY its local CPU analysis children and publication files; foreign work stops it.')
    HANDOFF.write_text(update_handoff(HANDOFF.read_text(encoding='utf-8'), block), encoding='utf-8')
    names = [str(target.relative_to(ROOT)), str(REPORT.relative_to(ROOT)), str(HANDOFF.relative_to(ROOT))]
    guard(head, [*allowed, *names[1:]])
    # Inspect exact status/diff/recent style before every owner-requested commit.
    save_new(STATE/('GIT_REVIEW_'+stamp+'.json'), dict(status=git('status', '--short'),
        diff=git('diff', '--stat'), recent=git('log', '--oneline', '-10'), intended=names))
    git('diff', '--check', '--', *names[1:])
    git('add', '--', *names)
    staged = git('diff', '--cached', '--name-only').splitlines()
    assert all(any(name == p.replace('\\', '/') or name.startswith(p.replace('\\', '/')+'/') for p in names)
               for name in staged), 'unintended staged file'
    git('commit', '--quiet', '-m', 'S-RB-INT: publish '+publication['label']+' saved-evidence refresh '+stamp)
    result = dict(cache=cache, baseline=baseline, endpoint_baseline=endpoint_baseline,
                  interim08=state.get('interim08'), freeze=state.get('freeze'),
                  last_publication=publication['receipt_directory'], commit=git('rev-parse', 'HEAD'),
                  final=event_now >= FINAL)
    save_new(STATE/('STATE_'+stamp+'.json'), result)
    print(local_time, publication['label'], result['commit'], flush=True)
    return result


def main():
    global STATE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('start', 'watch'))
    p.add_argument('--state-root', default=str(STATE))
    p.add_argument('--recovery-receipt')
    args = p.parse_args()
    STATE = Path(args.state_root).resolve()
    assert STATE.parent == ROOT/'runs'
    if args.action == 'start':
        guard()
        assert time.time() < FINAL
        assert STATE.parent.is_dir() and EVIDENCE.is_dir() and PUBLISH.parent.is_dir()
        if STATE.name != 'rbint_noon':
            assert args.recovery_receipt, 'diagnosed recovery receipt required'
            prior = load(Path(args.recovery_receipt))
            assert prior['automatic_retry'] is False and prior['remote_inference_dispatched'] is False
        STATE.mkdir(exist_ok=True)
        PUBLISH.mkdir(exist_ok=True)
        save_new(STATE/'DISPATCH_ONCE.json', dict(unix=time.time(), deadline=DEADLINE,
            interval_s=1200, analysis_only=True, automatic_retry=False, head=git('rev-parse', 'HEAD'),
            source_sha256=digest(Path(__file__))))
        argv = [sys.executable, '-B', '-u', '-m', 'tools.rbint.continue_noon', 'watch', '--state-root', str(STATE)]
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP |
                 subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == 'nt' else 0
        with (STATE/'watch.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=flags, start_new_session=os.name != 'nt')
        receipt = dict(pid=child.pid, **child_incarnation(child), argv=argv,
                       source_sha256=digest(Path(__file__)), deadline=DEADLINE)
        save_new(STATE/'PROCESS.json', receipt)
        print(json.dumps(receipt))
        return
    assert (STATE/'DISPATCH_ONCE.json').is_file()
    save_new(STATE/'WATCH_ONCE.json', dict(pid=os.getpid(), unix=time.time()))
    state = {}
    try:
        while time.time() < DEADLINE:
            state = collect(state)
            if state.get('final'):
                save_new(STATE/'COMPLETE.json', dict(unix=time.time(), state=state))
                return
            time.sleep(max(1, next_wake(time.time())-time.time()))
        raise RuntimeError('hard publication deadline reached')
    except Exception as error:
        save_new(STATE/'STOP.json', dict(unix=time.time(), error=repr(error), state=state,
            automatic_retry=False, remote_inference_dispatched=False))
        raise


if __name__ == '__main__':
    main()
