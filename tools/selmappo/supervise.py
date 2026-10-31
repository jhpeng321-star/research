"""Bounded operational heartbeat/commits; consumes the existing observer's receipts."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .protocol import digest, safe_path
from .remote import run
from .watch import BINDING, BUILD, EVIDENCE, LOCK as OBSERVER_LOCK

LOCK = EVIDENCE/'SUPERVISE_NOON_ONCE.json'
REPORT = Path('docs/paper_results/SELMAPPO_SUPERVISION.md')
DEADLINE = datetime.fromisoformat('2026-11-01T12:00:00+08:00').timestamp()
INTERVAL = 1200


def latest():
    paths = sorted(EVIDENCE.glob(f'PROGRESS_{BUILD}_*.json'))
    if not paths:
        raise RuntimeError('existing observer has no progress receipt')
    path = safe_path(paths[-1])
    for attempt in range(3):
        try:
            return path, json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(1)


def validate(record, previous=None, *, now=None):
    now = time.time() if now is None else now
    assert record['source_binding'] == BINDING
    assert record['own_pid'] == 645235
    age = now-datetime.fromisoformat(record['timestamp_utc']).timestamp()
    assert -60 <= age <= 660, f'observer receipt age {age:.0f}s'
    assert record['historical_charged_sim_s'] == 43200
    assert record['inherited_unresolved_reservations'] == 14
    assert record['training_budget_upper_sim_s'] == 7012800
    assert record['prospective_local_evaluation_sim_s'] == 14400
    assert not record['local_evaluation_started'] and not record['mechanics_replayed']
    assert 43200 <= record['charged_sim_s'] <= 6998400
    assert 0 <= record['completed_updates'] <= 169
    assert record['charged_sim_s'] == 43200+900*record['production_reserved_episodes']
    assert record['production_reserved_episodes'] <= 7728
    assert record['failure_count'] == 0, 'new training failure: agent repair needed'
    assert record['completed_result_totals']['collisions'] == 0
    assert record['completed_result_totals']['teleports'] == 0
    if previous is not None:
        for key in ('completed_updates', 'production_reserved_episodes',
                    'production_completed_episodes', 'charged_sim_s'):
            assert record[key] >= previous[key], f'counter regressed: {key}'


def identity():
    result = run('cpuQ', f'''
import hashlib,json,os,shutil,time
from pathlib import Path
root=Path.home()/'selmappo20261031'/{BUILD!r}
dispatch=json.loads((root/'DISPATCH.json').read_text())
assert dispatch['pid']==645235 and dispatch['binding']=={BINDING!r}
stage=json.loads((root/'STAGE.json').read_text());assert stage['binding']=={BINDING!r}
for name,sha in stage['source_sha256'].items():
 assert not any(x in name.lower() for x in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vehicle_params','driver_params','vtype','.rou.xml'))
 path=root/'pytools'/name
 assert path.is_relative_to(root/'pytools') and '..' not in Path(name).parts
 assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
p=Path('/proc/645235');alive=p.exists();record=dict(alive=alive,pid=645235,source_verified=True)
if alive:
 args=[a.decode() for a in (p/'cmdline').read_bytes().split(bytes([0])) if a]
 assert args==dispatch['command'], 'owned supervisor command changed'
 stat=(p/'stat').read_text().rsplit(')',1)[1].split()
 record.update(uid=p.stat().st_uid,start_ticks=int(stat[19]),state=stat[0],
               nice=os.getpriority(os.PRIO_PROCESS,645235),command=args)
 assert record['uid']==os.getuid()==1000 and record['nice']==19
 assert record['state']!='Z', 'owned supervisor is a zombie'
record.update(timestamp_unix=time.time(),load_1m=os.getloadavg()[0],cores=os.cpu_count(),
 disk_free_bytes=shutil.disk_usage(root).free)
print('SUPERVISOR_IDENTITY',json.dumps(record))
''', timeout=60)
    return json.loads(next(line.removeprefix('SUPERVISOR_IDENTITY ') for line in
                           result.stdout.splitlines() if line.startswith('SUPERVISOR_IDENTITY ')))


def git(*args):
    result = subprocess.run(['git', *args], capture_output=True, text=True, check=True)
    if result.stdout:
        print(result.stdout, end='', flush=True)
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr, flush=True)
    return result.stdout


def commit(paths, message):
    git('status', '--short', '--branch')
    git('diff', '--stat')
    git('log', '--oneline', '-10')
    assert not git('diff', '--cached', '--name-only'), 'unrelated staged work: do not auto-commit'
    permitted = {str(LOCK).replace('\\', '/'), str(REPORT).replace('\\', '/'),
                 'docs/paper_results/SELMAPPO.md',
                 'docs/paper_results/selmappo/FROZEN_FINAL_MAIN0208.json'}
    clean = []
    for path in paths:
        name = safe_path(path).relative_to(Path.cwd()).as_posix()
        assert name in permitted or (name.startswith(
            f'docs/paper_results/selmappo/PROGRESS_{BUILD}_') and name.endswith('.json'))
        clean.append(name)
    git('add', '--', *clean)
    git('diff', '--cached', '--stat')
    git('commit', '-m', message)


def tick(state):
    path, record = latest()
    problems = []
    owner = None
    try:
        validate(record, state.get('previous'))
    except Exception as exc:
        problems.append(f'{type(exc).__name__}: {exc}')
    try:
        owner = identity()
        assert owner.get('start_ticks', 48472972) == 48472972, 'owned PID starttime changed'
        if state.get('owner') and owner['alive']:
            assert owner['start_ticks'] == state['owner']['start_ticks'], 'PID reuse'
        if not owner['alive'] and not record['final169_eligible']:
            problems.append('Owned trainer exited before final169; agent repair needed.')
    except Exception as exc:
        problems.append(f'{type(exc).__name__}: {exc}')
    if owner and owner['alive'] and 'owner' not in state:
        state['owner'] = owner
    stamp = datetime.now(timezone.utc).isoformat()
    previous = state.get('previous')
    delta = '' if previous is None else (
        f"; delta +{record['completed_updates']-previous['completed_updates']} updates, "
        f"+{record['production_completed_episodes']-previous['production_completed_episodes']} episodes")
    line = (f"- {stamp}: {record['completed_updates']}/169 updates, "
            f"{record['production_completed_episodes']} production complete/"
            f"{record['outstanding_episodes']} outstanding, {record['failure_count']} failures, "
            f"{record['charged_sim_s']:,} sim-s charged{delta}. ")
    line += ('ALERT: '+'; '.join(problems)) if problems else (
        f"Owned PID645235 identity/source verified, nice19; load "
        f"{owner['load_1m']:.2f}/{owner['cores']} cores, free disk "
        f"{owner['disk_free_bytes']/2**30:.2f} GiB.")
    line += f' Receipt `{path.name}`.'
    if previous and record['production_completed_episodes'] == previous['production_completed_episodes']:
        line += ' No completed-episode advance this interval; inspect resource admission before any repair.'
    state.setdefault('lines', []).append(line)
    header = ('# Fleet PPO operational supervision through noon\n\n'
              'MAIN04:27: approximately20-minute health/progress publication and commits.\n'
              'Consumes existing observer3668 receipts; verifies exact cpuQ645235 command,\n'
              'UID/starttime/nice and immutable staged source hashes. Exclusive local lock.\n'
              'No physics, process signals, retries, dispatches or intermediate evaluation.\n'
              'Deadline2026-11-01T12:00:00+08:00; original observer/final handoff continues.\n'
              'Exact cap7,012,800; historical43,200 retained; evaluation14,400 unstarted.\n'
              'Alerts require an active agent for a registered same-cap repair.\n\n')
    safe_path(REPORT).write_text(header+'\n'.join(state['lines'])+'\n', encoding='utf-8')
    paths = [LOCK, REPORT, Path('docs/paper_results/SELMAPPO.md')]
    paths += sorted(EVIDENCE.glob(f'PROGRESS_{BUILD}_*.json'))
    final = EVIDENCE/'FROZEN_FINAL_MAIN0208.json'
    if final.exists():
        paths.append(final)
    commit(paths, f"selmappo: noon heartbeat update {record['completed_updates']} at {stamp[11:16]} UTC")
    state['previous'] = record
    print('AGENT_LOOP_TICK_SELMAPPO', json.dumps(dict(prompt='Check owned fleet trainer progress and repair any recorded alert within registered cap.',
          problems=problems, updates=record['completed_updates'], timestamp_utc=stamp)), flush=True)


def watch():
    assert EVIDENCE.is_dir() and OBSERVER_LOCK.exists() and time.time() < DEADLINE
    with LOCK.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), build=BUILD, source_binding=BINDING,
                       source_sha256=digest(__file__), started_unix=time.time(),
                       deadline_unix=DEADLINE, interval_s=INTERVAL), stream)
    state = {}
    while True:
        try:
            tick(state)
        except Exception as exc:
            print('SUPERVISION_ERROR_AGENT_ATTENTION', repr(exc), flush=True)
        remaining = DEADLINE-time.time()
        if remaining <= 0:
            return
        time.sleep(min(INTERVAL, remaining))


def start():
    assert EVIDENCE.is_dir() and not LOCK.exists() and time.time() < DEADLINE
    log = EVIDENCE/'SUPERVISE_NOON.log'
    with log.open('x', encoding='utf-8') as stream:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == 'nt' else 0
        proc = subprocess.Popen([sys.executable, '-m', 'tools.selmappo.supervise', 'run'],
                                stdout=stream, stderr=subprocess.STDOUT, cwd=Path.cwd(), creationflags=flags)
    print(json.dumps(dict(supervisor_pid=proc.pid, deadline_unix=DEADLINE, log=str(log))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('start', 'run', 'identity'))
    action = parser.parse_args().action
    if action == 'start':
        start()
    elif action == 'run':
        watch()
    else:
        print(json.dumps(identity(), indent=2))
