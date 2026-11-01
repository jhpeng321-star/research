"""Cache-only interim publisher and explicit final report-guard continuation.

The existing finish_v2 process alone collects forecasts. This process neither
connects to evaluation hosts nor dispatches inference. It assumes final delivery
only after finish_v2's expected progress-report guard STOP and verified full cache.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

from tools.rb2int.collect import registry
from tools.rb2int.finish import PYTHON, git_commit, score_once, publish
from tools.rb2int.interim import scopes, analyze, render
from tools.rbint.local_resources import child_incarnation
from tools.rbeval.core import load, digest, save_new

COLLECTOR = ROOT/'runs/rb2int_finish_v3'
CACHE = ROOT/'runs/rb2int_final_cache'
REPORT = ROOT/'docs/paper_results/WMPHYS_V5.md'
EVIDENCE = ROOT/'docs/rb2int/interim'
DATA = ROOT/'docs/rb2int/scoring_data_v1'
SHORT = ROOT/'docs/rb2int/SHORT_TARGETS_BOUNDED_V4.json'
DEADLINE = datetime.fromisoformat('2026-11-02T09:55:00+08:00').timestamp()
BEGIN, END = '<!-- RB2-INTERIM:BEGIN -->', '<!-- RB2-INTERIM:END -->'
LIMITS = '''
### Interim CI numerical correction

The16:46snapshot is preserved. Its intervals printed as equal endpoints for
constant direction-accuracy columns are superseded by NA (zero estimable variance).
Centered variance reduction removes floating-point accumulation artifacts; all
point estimates and forecasts are unchanged. No model selection or forecast replay.

## Frozen-candidate engineering limits (later worker development evidence)

LANE's later free-running development diagnostic rejects fitted-LC adoption on
lane-frequency/recall evidence. BREAK reports that frozen116b2147 targets a retired
mode2 learned mirror, not the actual shared CAV actuator. Therefore the frozen v5
switch must not be described as verified mode2 actuator consistency. These later
findings do not change sourceb2df7458 or the running candidates; all outcomes remain
reported. LP remains a narrow learned departure-parameter variant. No integrated
performance or cooperation/planner authorization follows from engineering readiness.
'''


def commit_when_quiet(paths, message):
    """Avoid the sole collector's20min commit window; never redo analysis."""
    from tools.rb2int.finish import command
    deadline = time.time()+1500
    while True:
        staged = command(['git','diff','--cached','--name-only']).splitlines()
        if any(not p.startswith('docs/rb2int/COLLECT_') for p in staged):
            raise ValueError('foreign non-collector staging present; preserve it')
        log = COLLECTOR/'watch.log'
        polls = [line.split()[1] for line in log.read_text(encoding='utf-8').splitlines()
                 if line.startswith('COLLECT ')] if log.exists() else []
        terminal = any((COLLECTOR/name).is_file() for name in ('STOP.json','INCOMPLETE_AT_DEADLINE.json'))
        age = time.time()-datetime.strptime(polls[-1],'%Y%m%dT%H%M%S%z').timestamp() if polls else None
        if not staged and (terminal or age is not None and 30 <= age <= 1000):
            return git_commit(paths,message)
        if time.time() >= deadline:
            raise TimeoutError('no quiet collector commit window; preserve completed analysis')
        time.sleep(2)


def recovered_publication():
    """Reuse the completed corrected16:57analysis after commit-only contention."""
    previous = ROOT/'runs/rb2int_interim_v2'
    expected = "ValueError('foreign staging present; preserve it')"
    if not (previous/'STOP.json').is_file() or load(previous/'STOP.json')['error'] != expected:
        raise ValueError('expected prior commit-only STOP required')
    from tools.rb2int.resume_ci import alive
    if alive(load(previous/'PROCESS.json')):
        raise ValueError('prior cache publisher is still alive')
    repair = load(ROOT/'runs/rb2int_ci_resume_v1/COMPLETE.json')
    for name in ('uncertainty.py','credibility.py','gate_bootstrap.py'):
        path = ROOT/'tools/rb2int'/name
        if repair['source_hashes'][str(path)] != digest(path):
            raise ValueError('corrected snapshot CI lineage changed')
    receipt = EVIDENCE/'20261101T165709+0800_available_replicated_cells/SUMMARY.json'
    data = load(receipt)
    if data['source_sha256'] != digest(ROOT/'tools/rb2int/interim.py'):
        raise ValueError('completed corrected analysis source changed')
    return dict(scope=data['scope'],receipt=str(receipt),sha256=digest(receipt),
        markdown_sha256=digest(receipt.with_name('INTERIM.md')))


def replace_block(text, block):
    if BEGIN in text:
        if text.count(BEGIN) != 1 or text.count(END) != 1:
            raise ValueError('ambiguous interim report markers')
        start, stop = text.index(BEGIN), text.index(END)+len(END)
        return text[:start]+BEGIN+'\n'+block+END+text[stop:]
    return text.rstrip()+'\n\n'+BEGIN+'\n'+block+END+'\n'


def cache_state():
    complete = {'FULL-v4':set(range(969432,969440)), 'FULL-v5':set()}
    jobs = []
    for host in ('gpuA','gpuR'):
        for path in sorted((CACHE/host).glob('*/COLLECTED.json')):
            record = load(path)
            job = record['job'];jobs.append(job['id'])
            if job['model'] in complete:
                complete[job['model']].add(job['seed'])
    if len(jobs) != len(set(jobs)):
        raise ValueError('duplicate cache world jobs')
    return sorted(complete['FULL-v4'] & complete['FULL-v5']), len(jobs)


def final_guard_ready():
    stop = COLLECTOR/'STOP.json'
    if not stop.is_file():
        return False
    if load(stop)['error'] != "ValueError('progress report changed during collection; preserve edits and assemble explicitly')":
        raise ValueError('collector stopped for an unexpected reason; preserve explicit recovery')
    start = load(COLLECTOR/'START_ONCE.json')
    if any(digest(p) != h for p,h in start['source_hashes'].items()):
        raise ValueError('analysis sources changed since original collector launch')
    return True


def append_final(short_reports):
    delivery = ROOT/'runs/rb2int_delivery_v1'
    delivery.mkdir(exist_ok=True)
    report, scores = score_once(delivery)
    receipt = report.with_suffix('.delivery.json')
    original = load(receipt)
    save_new(delivery/'ORIGINAL_REPORT_DELIVERY.json', original)
    appendix = LIMITS+'\n## Owner16:28 short-horizon/interim appendix\n'
    appendix += 'Original interim snapshots and detailed short-horizon bounds: docs/rb2int/interim/. Final primary54 short-horizon controller comparison follows.\n'
    # On the completed cache, compute the registered54 and reserved8, not merely
    # the last available-cell look. Same stored forecasts, no new inference.
    registration = load(ROOT/'docs/rb2int/EVALUATION_REGISTRATION.json')
    for scope, worlds in (('primary54',registration['replicated_primary_worlds']),
                          ('reserved8',registration['reserved8'])):
        out = delivery/('OWNER_ADDENDUM_'+scope+'.json')
        result = analyze(delivery/'RUNS.json',DATA/'REAL_OBSERVABLES.json',DATA/'FACTUAL_ENDPOINTS.json',
            SHORT,worlds,scope,out)
        appendix += '\n'+render(result,datetime.now().astimezone().isoformat())
    report.write_text(report.read_text(encoding='utf-8')+'\n'+appendix,encoding='utf-8')
    receipt.write_text(json.dumps(dict(original,report_sha256=digest(report),
        original_report_delivery_sha256=digest(delivery/'ORIGINAL_REPORT_DELIVERY.json'),
        owner_addendum_source_sha256=digest(__file__),interim_snapshots=short_reports),indent=1),encoding='utf-8')
    return publish(delivery,report,scores)


def run(out):
    if (COLLECTOR/'COMPLETE.json').exists():
        raise ValueError('original final publication already complete; preserve it')
    expected_collector = load(COLLECTOR/'PROCESS.json')
    argv = expected_collector['argv']
    if (Path(argv[-1]) != COLLECTOR or Path(argv[3]) != ROOT/'tools/rb2int/finish.py'
            or argv[-3:-1] != ['watch','--out'] or expected_collector.get('inference_dispatch') is not False):
        raise ValueError('collector ownership/command changed')
    files = [Path(__file__), ROOT/'tools/rb2int/interim.py', SHORT]
    hashes = {str(p):digest(p) for p in files}
    save_new(out/'START_ONCE.json',dict(source_hashes=hashes,collector=expected_collector,
        cache_only=True,inference_dispatch=False,collection_dispatch=False,deadline=DEADLINE))
    report_hash = digest(REPORT)
    EVIDENCE.mkdir(exist_ok=True)
    registration = load(ROOT/'docs/rb2int/EVALUATION_REGISTRATION.json')
    real = load(DATA/'REAL_OBSERVABLES.json')['episodes']
    metadata = {r['seed']:r for r in real}
    seed = recovered_publication()
    save_new(out/'REUSED_PUBLICATION.json',seed)
    published = [seed]
    stamp = Path(seed['receipt']).parent.name.split('_',1)[0]
    last = {seed['scope']:datetime.strptime(stamp,'%Y%m%dT%H%M%S%z').timestamp()}
    while True:
        if any(digest(p) != h for p,h in hashes.items()):
            raise ValueError('interim source/target changed; explicit recovery required')
        if digest(REPORT) != report_hash:
            raise ValueError('report changed outside interim publisher; preserve external edits')
        worlds, count = cache_state()
        available = scopes(worlds,registration,metadata)
        now = time.time()
        for scope, selected in available.items():
            # First complete replicated set and reserved8 each publish immediately;
            # further snapshots in either scope at approximately3h intervals.
            if scope in last and now-last[scope] < 10800:
                continue
            stamp = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')
            folder = EVIDENCE/(stamp+'_'+scope)
            folder.mkdir()
            runs = folder/'RUNS.json';registry(CACHE,runs)
            summary = folder/'SUMMARY.json'
            result = analyze(runs,DATA/'REAL_OBSERVABLES.json',DATA/'FACTUAL_ENDPOINTS.json',
                SHORT,selected,scope,summary)
            section = render(result,stamp)
            text_path = folder/'INTERIM.md';text_path.write_text(section,encoding='utf-8')
            published.append(dict(scope=scope,receipt=str(summary),sha256=digest(summary),markdown_sha256=digest(text_path)))
            last[scope]=now
            # Keep newest reserved and newest replicated view together.
            latest = {}
            for entry in published:
                path = Path(entry['receipt']).with_name('INTERIM.md')
                if digest(entry['receipt']) != entry['sha256'] or digest(path) != entry['markdown_sha256']:
                    raise ValueError('published interim evidence changed; preserve explicit recovery')
                latest[entry['scope']] = path
            block = '\n'.join(p.read_text(encoding='utf-8') for p in latest.values())+'\n'+LIMITS
            REPORT.write_text(replace_block(REPORT.read_text(encoding='utf-8'),block),encoding='utf-8')
            report_hash = digest(REPORT)
            commit_when_quiet([runs,summary,text_path,REPORT], 'S-RB2-INT: publish INTERIM v5-v4 '+scope+' '+stamp)
            print('PUBLISHED',stamp,scope,len(selected),flush=True)
        if count == 216 and final_guard_ready():
            save_new(out/'FINAL_TAKEOVER_ONCE.json',dict(collector_stop_sha256=digest(COLLECTOR/'STOP.json'),
                complete_world_jobs=count,source_hashes=hashes,no_inference=True))
            result = append_final(published)
            save_new(out/'COMPLETE.json',result);print('FINAL DELIVERED',flush=True);return
        if (COLLECTOR/'STOP.json').exists() and not final_guard_ready():
            raise ValueError('unexpected collector stop')
        if (COLLECTOR/'INCOMPLETE_AT_DEADLINE.json').exists() or now >= DEADLINE:
            stamp = datetime.now().astimezone().isoformat()
            status = dict(label='INTERIM; INCOMPLETE AT DEADLINE',time=stamp,complete_world_jobs=count,
                expected=216,matched_v5_v4_worlds=worlds,qualification='NOT_ESTABLISHED',inference_repeated=False)
            save_new(out/'INCOMPLETE_AT_DEADLINE.json',status)
            receipt = EVIDENCE/'INCOMPLETE_AT_DEADLINE.json';save_new(receipt,status)
            block = f'## INTERIM — incomplete evaluation at {stamp}\n{count}/216 world jobs complete. Full gate/switching NOT ESTABLISHED. Partial snapshots remain below.\n'
            REPORT.write_text(REPORT.read_text(encoding='utf-8')+'\n'+block,encoding='utf-8')
            commit_when_quiet([receipt,REPORT],'S-RB2-INT: retain explicit incomplete final-panel deadline')
            return
        if (COLLECTOR/'COMPLETE.json').exists():
            raise ValueError('uncoordinated final publication; preserve it')
        print('CACHE',count,'/216; matched',len(worlds),'ready',list(available),flush=True)
        time.sleep(120)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('start','watch'))
    parser.add_argument('--out',default=str(ROOT/'runs/rb2int_interim_v1'))
    args = parser.parse_args();out = Path(args.out)
    if not out.parent.is_dir():
        raise ValueError('existing continuation parent required')
    out.mkdir(exist_ok=True)
    if args.action == 'start':
        save_new(out/'DISPATCH_ONCE.json',dict(source_sha256=digest(__file__),inference_dispatch=False,collection_dispatch=False))
        argv = [str(PYTHON),'-B','-u',str(Path(__file__).resolve()),'watch','--out',str(out)]
        flags = subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0
        with (out/'watch.log').open('x',encoding='utf-8') as log:
            child = subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=flags,
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONIOENCODING='utf-8'))
        receipt = dict(pid=child.pid,argv=argv,**child_incarnation(child),cache_only=True)
        save_new(out/'PROCESS.json',receipt);print(json.dumps(receipt));return
    try:
        run(out)
    except Exception as error:
        save_new(out/'STOP.json',dict(error=repr(error),automatic_retry=False,inference_repeated=False))
        raise


if __name__ == '__main__':
    main()
