"""Standby for natural old deadline exits, then collect/publish the delayed full panel.

One successor owns collection AND interim publication, after both prior owners
have exited. No inference dispatch or process signals; every completed cache and
score is reused. This is the owner17:35 later-delivery operational amendment.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from tools.rb2int.audit import ROOT
from tools.rb2int.collect import collect_host, registry
from tools.rb2int.finish import PYTHON, git_commit
from tools.rb2int.interim import scopes, analyze, render
from tools.rb2int import interim_watch as interim
from tools.rb2int.resume_ci import alive
from tools.rbint.local_resources import child_incarnation
from tools.rbeval.core import load, digest, save_new

DEADLINE = datetime.fromisoformat('2026-11-04T09:55:00+08:00').timestamp()
OLD_COLLECTOR = ROOT/'runs/rb2int_finish_v3'
OLD_PUBLISHER = ROOT/'runs/rb2int_interim_v3'
OPERATIONAL_NOTE = '''
## Owner17:35 GPU-priority / later-delivery amendment

New admission is limited to2evaluate processes on gpuA and1on gpuR. Original
children finished without signals; only the drained own CPU dispatchers were
retired before exclusive claim-preserving handover. Remaining reserved FULL-v5
jobs moved first, otherwise original queue order was preserved. No candidate,
recipe, panel, fit, task outcome or statistical gate was changed. Completed and
running claims were never retried or redispatched. A prospective late pass starts
only after the original cutoff pass naturally exits; its finite admission bound
is Nov4 09:00+08 and collection/publication bound09:55. Delivery occurs immediately
when the full frozen cache completes. Original deadline incompleteness is retained.
'''


def old_owners_ready(collector, publisher, is_alive):
    """No takeover from a live owner, a source STOP, or partial completed final."""
    if (publisher/'COMPLETE.json').exists() or (collector/'COMPLETE.json').exists():
        return 'complete'
    for folder in (collector, publisher):
        stop = folder/'STOP.json'
        if stop.exists():
            allowed_final_guard = (folder == collector and load(stop)['error'] ==
                "ValueError('progress report changed during collection; preserve edits and assemble explicitly')")
            if not allowed_final_guard:
                raise ValueError('unexpected prior STOP: preserve explicit recovery')
    for folder in (collector, publisher):
        if not (folder/'INCOMPLETE_AT_DEADLINE.json').exists() or is_alive(load(folder/'PROCESS.json')):
            return 'wait'
    return 'ready'


def previous_publications():
    latest = {}
    for path in sorted(interim.EVIDENCE.glob('*/SUMMARY.json')):
        record = load(path)
        text = path.with_name('INTERIM.md')
        if not text.is_file():
            raise ValueError('partial prior interim: preserve explicit recovery')
        stamp = path.parent.name.split('_', 1)[0]
        latest[record['scope']] = dict(scope=record['scope'], receipt=str(path),
            sha256=digest(path), markdown_sha256=digest(text),
            unix=datetime.strptime(stamp, '%Y%m%dT%H%M%S%z').timestamp())
    return latest


def run(out):
    # Process-local display amendment: existing publisher/source bytes are intact.
    interim.LIMITS += '\n'+OPERATIONAL_NOTE
    watched = [Path(__file__), ROOT/'tools/rb2int/finish.py',
        OLD_COLLECTOR/'PROCESS.json', OLD_PUBLISHER/'PROCESS.json',
        ROOT/'tools/rb2int/interim_watch.py', ROOT/'tools/rb2int/interim.py',
        *[ROOT/'tools/rb2int'/name for name in ('collect.py', 'prepare_scoring.py',
          'score_gate.py', 'compare.py', 'credibility.py', 'report.py',
          'predictions.py', 'uncertainty.py', 'gate_bootstrap.py')]]
    hashes = {str(path): digest(path) for path in watched}
    save_new(out/'START_ONCE.json', dict(source_hashes=hashes, deadline=DEADLINE,
        old_collector=load(OLD_COLLECTOR/'PROCESS.json'),
        old_publisher=load(OLD_PUBLISHER/'PROCESS.json'),
        inference_dispatch=False, process_signals=False, standby_only=True))
    while True:
        if any(digest(path) != value for path, value in hashes.items()):
            raise ValueError('late source changed; explicit recovery required')
        state = old_owners_ready(OLD_COLLECTOR, OLD_PUBLISHER, alive)
        if state == 'complete':
            save_new(out/'PREVIOUS_FINAL_COMPLETE.json', dict(no_takeover=True))
            return
        if state == 'ready':
            break
        if time.time() >= DEADLINE:
            raise TimeoutError('prior owners never yielded by late deadline')
        print('STANDBY prior collector/publisher retain ownership', flush=True)
        time.sleep(60)
    save_new(out/'TAKEOVER_ONCE.json', dict(old_owners_exited=True,
        time=datetime.now().astimezone().isoformat(), cache_reused=True,
        inference_dispatch=False, process_signals=False))
    published = previous_publications()
    registration = load(ROOT/'docs/rb2int/EVALUATION_REGISTRATION.json')
    metadata = {row['seed']: row for row in load(interim.DATA/'REAL_OBSERVABLES.json')['episodes']}
    last_collect = 0
    report_hash = digest(interim.REPORT)
    while True:
        if any(digest(path) != value for path, value in hashes.items()):
            raise ValueError('late source changed; preserve completed work')
        if digest(interim.REPORT) != report_hash:
            raise ValueError('report changed outside late sole publisher; preserve external edits')
        now = time.time()
        if now-last_collect >= 1200:
            stamp = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')
            receipts, results = [], []
            for host in ('gpuA', 'gpuR'):
                receipt = ROOT/'docs/rb2int'/('COLLECT_LATE_'+stamp+'_'+host+'.json')
                results.append(collect_host(host, interim.CACHE/host, receipt))
                receipts.append(receipt)
            git_commit(receipts, 'S-RB2-INT: collect delayed frozen final worlds '+stamp)
            last_collect = now
            print('COLLECT_LATE', stamp, sum(row['complete_world_jobs'] for row in results), '/216', flush=True)
        worlds, count = interim.cache_state()
        for scope, selected in scopes(worlds, registration, metadata).items():
            if scope in published and now-published[scope]['unix'] < 10800:
                continue
            stamp = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')
            folder = interim.EVIDENCE/(stamp+'_'+scope)
            folder.mkdir()
            runs = folder/'RUNS.json'
            registry(interim.CACHE, runs)
            summary = folder/'SUMMARY.json'
            result = analyze(runs, interim.DATA/'REAL_OBSERVABLES.json',
                interim.DATA/'FACTUAL_ENDPOINTS.json', interim.SHORT, selected, scope, summary)
            text = folder/'INTERIM.md'
            text.write_text(render(result, stamp), encoding='utf-8')
            published[scope] = dict(scope=scope, receipt=str(summary), sha256=digest(summary),
                markdown_sha256=digest(text), unix=now)
            blocks = []
            for entry in published.values():
                path = Path(entry['receipt']).with_name('INTERIM.md')
                if digest(entry['receipt']) != entry['sha256'] or digest(path) != entry['markdown_sha256']:
                    raise ValueError('prior interim bytes changed')
                blocks.append(path.read_text(encoding='utf-8'))
            block = '\n'.join(blocks)+'\n'+interim.LIMITS
            interim.REPORT.write_text(interim.replace_block(interim.REPORT.read_text(encoding='utf-8'), block), encoding='utf-8')
            report_hash = digest(interim.REPORT)
            git_commit([runs, summary, text, interim.REPORT],
                'S-RB2-INT: publish delayed INTERIM v5-v4 '+scope+' '+stamp)
        if count == 216:
            result = interim.append_final(list(published.values()))
            save_new(out/'COMPLETE.json', result)
            print('FINAL DELIVERED', flush=True)
            return
        if now >= DEADLINE:
            receipt = ROOT/'docs/rb2int/LATE_INCOMPLETE_AT_DEADLINE.json'
            save_new(receipt, dict(complete_world_jobs=count, expected=216,
                qualification='NOT_ESTABLISHED; incomplete at late deadline'))
            git_commit([receipt], 'S-RB2-INT: retain explicit delayed-panel incompleteness')
            return
        time.sleep(120)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'watch'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if not out.parent.is_dir():
        raise ValueError('existing continuation parent required')
    out.mkdir(exist_ok=True)
    if args.action == 'start':
        save_new(out/'DISPATCH_ONCE.json', dict(source_sha256=digest(__file__),
            inference_dispatch=False, collection_active=False, standby_only=True))
        argv = [str(PYTHON), '-B', '-u', '-m', 'tools.rb2int.late_watch', 'watch', '--out', str(out)]
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        with (out/'watch.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=flags, start_new_session=os.name != 'nt',
                env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONIOENCODING='utf-8'))
        receipt = dict(pid=child.pid, argv=argv, **child_incarnation(child),
            inference_dispatch=False, standby_only=True)
        save_new(out/'PROCESS.json', receipt)
        print(json.dumps(receipt))
        return
    try:
        run(out)
    except Exception as error:
        save_new(out/'STOP.json', dict(error=repr(error), automatic_retry=False,
            inference_repeated=False, process_signals=False))
        raise


if __name__ == '__main__':
    main()
