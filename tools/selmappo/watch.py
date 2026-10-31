"""Read-only five-minute observer and final169 handoff; never launches physics."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .launch import EVIDENCE
from .progress import snapshot
from .protocol import digest, safe_path
from .remote import run

BUILD = 'build_9bb7869d8f590818'
BINDING = '9bb7869d8f590818c46980a16a63be00f72e00d3491dec95f903649dfa7bf790'
PEER = Path('E:/agent-work/c4-c1switch2/SELMAPPO_FINAL.json')
LOCK = EVIDENCE/'WATCH_MAIN0208_ONCE.json'
REPORT = Path('docs/paper_results/SELMAPPO.md')


def collect():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        snapshot(BUILD)
    return json.loads(next(line.removeprefix('PROGRESS_RECEIPT ') for line in
                           output.getvalue().splitlines() if line.startswith('PROGRESS_RECEIPT ')))


def report(record):
    assert record['source_binding'] == BINDING
    text = '\n'.join([
        '# Fleet PPO controller-selector training status', '',
        f"Observed UTC: {record['timestamp_utc']}",
        f"- cpuQ supervisor: {record['own_pid']}; alive={record['own_pid_exists']}; nice={record['own_nice']}.",
        f"- Genuine completed PPO updates: {record['completed_updates']}/169; final169 eligible={record['final169_eligible']}.",
        f"- New production: {record['production_completed_episodes']} complete, {record['outstanding_episodes']} outstanding; failures={record['failure_count']}.",
        f"- Charged: {record['charged_sim_s']:,.0f} sim-s, including43,200historical; exact total cap7,012,800.",
        f"- Actual sim-s bounds: {record['actual_sim_s_lower_bound']:,.0f}..{record['actual_sim_s_upper_bound']:,.0f} (old unknown actuals retained).",
        '- MAIN02:08 schedule23+23+167x46; 46fewer gradient episodes than original; final169only.',
        '- Every reservation yields above0.8cores or below8GiB RAM/2GiB disk; nice19, one thread, CUDA hidden.',
        '- Training observations are not efficacy results. Evaluation remains exclusively C1SWITCH2-owned.',
        '- Local14,400evaluation reservation stays UNSTARTED; peer receives frozen final169 only.',
        f'- Immutable build: `{BUILD}`; binding `{BINDING}`.',
        '- Receipts: `docs/paper_results/selmappo/`; protocol/design and previous stops: `HANDOFF.md`.',
        '', 'This status is maintained by one read-only five-minute observer. It cannot launch, retry,',
        'resume or stop training/evaluation and does not signal any process.', ''
    ])
    safe_path(REPORT).write_text(text, encoding='utf-8')


def freeze():
    result = run('cpuQ', f'''
import dataclasses,hashlib,json,os,sys
from pathlib import Path
root=Path.home()/'selmappo20261031'/{BUILD!r};out=root/'training'
stage=json.loads((root/'STAGE.json').read_text());assert stage['binding']=={BINDING!r}
for name,expected in stage['source_sha256'].items():
 assert hashlib.sha256((root/'pytools'/name).read_bytes()).hexdigest()==expected
def rows(name):return [json.loads(s) for s in (out/name).read_text().splitlines()]
updates=rows('training.jsonl');assert len(updates)==169
assert [r['iteration'] for r in updates]==list(range(1,170))
assert [r['episodes'] for r in updates]==[23,23]+[46]*167
assert any(r['status']=='TRAIN_COMPLETE_AWAIT_REGISTERED_PANEL' for r in rows('status.jsonl'))
assert not (out/'failure.jsonl').exists()
ledger=rows('ledger.jsonl');assert ledger[0]['event']=='HISTORY' and ledger[0]['charged_sim_s']==43200
starts=[r for r in ledger if r['event']=='START'];ends=[r for r in ledger if r['event']=='END']
assert len(starts)==len(ends)==7728 and {{r['id'] for r in starts}}=={{r['id'] for r in ends}}
assert {{r['id'] for r in starts}}=={{f'fleet:{{s}}' for s in range(18000046,18007774)}}
assert all(r['actual_sim_s']==900 for r in ends)
assert 43200+sum(r['reserved_sim_s'] for r in starts)==6998400
env=json.loads((root/'ENV.json').read_text());sys.path[:0]=env['PYTHONPATH'].split(':')
os.environ['CUDA_VISIBLE_DEVICES']='';os.nice(19)
import torch
torch.set_num_threads(1)
from tools.selmappo.selector import FrozenFleetSelector
from tools.selmappo.learning import Selector,Updater
from tools.selmappo.protocol import LIBRARY,PROTOCOL
checkpoint=out/'checkpoint_169.pt';sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
frozen=FrozenFleetSelector(checkpoint,sha)
record=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert record['iteration']==169 and record['protocol']==PROTOCOL and tuple(record['library'])==LIBRARY
assert all(torch.isfinite(p).all() for p in frozen.model.parameters())
sources={{name.removeprefix('tools/selmappo/'):h for name,h in stage['source_sha256'].items() if name.startswith('tools/selmappo/')}}
receipt=dict(status='FROZEN_FINAL',iteration=169,checkpoint_sha256=sha,
 source_binding=stage['binding'],sources=sources,design_sha256=stage['design_sha256'],
 config=dataclasses.asdict(Updater(Selector()).cfg),protocol=PROTOCOL,library=LIBRARY,
 schedule=[23,23]+[46]*167,gradient_episodes=7728,omitted_gradient_episodes=46,
 charged_sim_s=6998400,evaluation_reserved_unstarted_sim_s=14400,total_cap_sim_s=7012800,
 historical_actual_bounds=[30600,37800],new_actual_sim_s=sum(r['actual_sim_s'] for r in ends),
 ledger_sha256=hashlib.sha256((out/'ledger.jsonl').read_bytes()).hexdigest(),
 training_log_sha256=hashlib.sha256((out/'training.jsonl').read_bytes()).hexdigest())
path=out/'FROZEN_FINAL.json';data=json.dumps(receipt,sort_keys=True,indent=2)
if path.exists():assert path.read_text()==data
else:
 with path.open('x') as f:f.write(data)
spec=dict(status='FROZEN_FINAL',root=str(root/'pytools/tools/selmappo'),sources=sources,
 checkpoint=str(checkpoint),checkpoint_sha256=sha,frozen_receipt=str(path),
 frozen_receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
print('FINAL_HANDOFF',json.dumps(dict(host='cpuQ',spec=spec,receipt=receipt)))
''', timeout=60)
    handoff = json.loads(next(line.removeprefix('FINAL_HANDOFF ') for line in result.stdout.splitlines()
                              if line.startswith('FINAL_HANDOFF ')))
    for path in (EVIDENCE/'FROZEN_FINAL_MAIN0208.json', PEER):
        path = safe_path(path)
        text = json.dumps(handoff, sort_keys=True, indent=2)
        if path.exists():
            assert path.read_text(encoding='utf-8') == text
        else:
            with path.open('x', encoding='utf-8') as stream:
                stream.write(text)
    return handoff


def watch():
    assert EVIDENCE.is_dir() and PEER.parent.is_dir() and REPORT.parent.is_dir()
    with LOCK.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), build=BUILD, binding=BINDING,
                       source_sha256=digest(__file__), started_unix=time.time()), stream)
    deadline = time.monotonic()+87000
    while time.monotonic() < deadline:
        try:
            record = collect()
            report(record)
            print(record['timestamp_utc'], 'updates', record['completed_updates'], 'failures',
                  record['failure_count'], 'alive', record['own_pid_exists'], flush=True)
            if record['final169_eligible']:
                freeze()
                return
            if not record['own_pid_exists']:
                return
        except Exception as exc:
            print('READ_ONLY_OBSERVER_ERROR', repr(exc), flush=True)
        time.sleep(300)


def start():
    assert EVIDENCE.is_dir() and not LOCK.exists()
    log = EVIDENCE/'WATCH_MAIN0208.log'
    with log.open('x', encoding='utf-8') as stream:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == 'nt' else 0
        proc = subprocess.Popen([sys.executable, '-m', 'tools.selmappo.watch', 'run'],
                                cwd=Path.cwd(), stdout=stream, stderr=subprocess.STDOUT,
                                creationflags=flags)
    print(json.dumps(dict(observer_pid=proc.pid, log=str(log), build=BUILD)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('start', 'run', 'once'))
    action = parser.parse_args().action
    if action == 'start':
        start()
    elif action == 'run':
        watch()
    else:
        report(collect())
