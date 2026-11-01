"""Read-only twenty-minute owned-job heartbeat; never dispatches or signals science."""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.c1switch.storage import write
from .operations import EVIDENCE, ssh
from .design import REGISTRATION

ROOT = Path(__file__).resolve().parents[2]
WATCH = EVIDENCE / 'WATCH_MAIN0427'
DEADLINE = datetime.fromisoformat('2026-11-01T12:00:00+08:00').timestamp()
CPU_ROOTS = {
    'cpuQ': '/home/ubuntu/c1switch2_run_eb0eda956a815f14',
    'gpuR': '/root/c1switch2_run_716d39f08986c84f',
}
SCAN = '''import hashlib,json,os,time
from pathlib import Path
def load(path):
 return json.loads(path.read_text()) if path.is_file() else None
def process(pid,root):
 p=Path('/proc',str(pid))
 if not p.exists():return dict(pid=pid,state='EXITED')
 fields=(p/'stat').read_text().split()
 command=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
 return dict(pid=pid,state=fields[2],start_ticks=int(fields[21]),uid=p.stat().st_uid,
             own_root_in_command=str(root) in command)
out=dict(unix=time.time(),host=HOST,load=os.getloadavg(),cpu={},gpu={})
for arm,name in CPU.items():
 root=Path(name)
 dispatch_name=('DISPATCH_CPU.json' if arm=='cpu' else 'DISPATCH_ORACLES.json' if arm=='wm_oracles' else 'DISPATCH_SELECTOR.json' if arm in ('fleet_ppo','cav_mappo') else 'DISPATCH_MODEL.json')
 dispatch=load(root/dispatch_name)
 if not dispatch:
  out['cpu'][arm]=dict(root=name,state='UNCLAIMED');continue
 row=dict(root=name,process=process(dispatch['pid'],root),complete=(root/('COMPLETE.json' if arm=='wm_oracles' else 'supervisor/COMPLETE.json')).is_file(),
          stop=load(root/'STOP'),episodes=0,decisions=0,closed_jobs=0,closed_actual_sim_s=0)
 raw=(root/'accounting.jsonl').read_bytes() if (root/'accounting.jsonl').is_file() else b''
 row.update(charged_sim_s=sum(json.loads(x)['sim_s'] for x in raw.splitlines()),
            ledger_sha256=hashlib.sha256(raw).hexdigest())
 for job in (root/'jobs').iterdir():
  row['episodes']+=(job/'PHYSICS.json').is_file()
  row['decisions']+=len(list(job.glob('pick_j*.json')))
  close=load(job/'CLOSE.json')
  if close:row['closed_jobs']+=1;row['closed_actual_sim_s']+=close['actual_sim_s']
 out['cpu'][arm]=row
for arm,name in GPU.items():
 root=Path(name);dispatch=load(root/'DISPATCH.json')
 if not dispatch:
  out['gpu'][arm]=dict(root=name,state='UNCLAIMED');continue
 started=load(root/'rpc/GPU_STARTED.json');finished=load(root/'rpc/GPU_FINISHED.json')
 row=dict(root=name,process=process(dispatch['pid'],root),started=started,finished=finished,
          stop=load(root/'rpc/STOP'),ready=(root/'rpc/READY.json').is_file(),
          requests=len(list((root/'rpc').glob('*.request.json'))),
          results=len(list((root/'rpc').glob('*.result.json'))))
 row['elapsed_wall_s']=None if not started else max(0,time.time()-started['unix'])
 out['gpu'][arm]=row
print(json.dumps(out))
'''


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def local_process(pid):
    """Query only the exact owned Windows PID, without signal-zero semantics."""
    if os.name != 'nt':
        return dict(pid=pid, state='NOT_CHECKED_NON_WINDOWS')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000 | 0x00100000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        return dict(pid=pid, state='EXITED' if error == 87 else 'QUERY_FAILED', error=error)
    try:
        code = ctypes.c_ulong()
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            return dict(pid=pid, state='QUERY_FAILED', error=ctypes.get_last_error())
        return dict(pid=pid, state='LIVE' if code.value == 259 else 'EXITED', exit_code=code.value)
    finally:
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle(handle)


def collect():
    configs = {}
    for arm in ('wm', 'bb'):
        configs[arm] = read_json(EVIDENCE / (arm.upper() + '_EXECUTION_BOUND.json'))
        if configs[arm] is None:
            configs[arm] = read_json(EVIDENCE / (arm.upper() + '_EXECUTION.json'))
    out = dict(unix=time.time(), protocol='S-C1SWITCH2/v2', hosts={}, alerts=[])
    for host in ('cpuQ', 'gpuR', 'gpuA'):
        cpus = {'cpu': CPU_ROOTS[host]} if host in CPU_ROOTS else {}
        gpus = {}
        for arm, config in configs.items():
            if config and host in config.get('cpus', {}):
                cpus[arm] = config['cpus'][host]
            if config and host in config['gpus']:
                gpus[arm] = config['gpus'][host]
        for arm in ('fleet_ppo', 'cav_mappo'):
            prepared = read_json(EVIDENCE / ('SELECTOR_PREPARED_'+arm+'_'+host+'.json'))
            if prepared:
                cpus[arm] = prepared['root']
        if host == 'gpuR':
            oracle = read_json(EVIDENCE / 'WM_ORACLE_CONTINUATION_DISPATCH.json')
            if oracle:
                cpus['wm_oracles'] = oracle['root']
        try:
            code = f'HOST={host!r};CPU={cpus!r};GPU={gpus!r}\n' + SCAN
            out['hosts'][host] = ssh(host, code, timeout=45)
        except Exception as exc:
            out['alerts'].append(f'{host}: collection transport error {exc!r}')
    for label, path in [('relay', EVIDENCE / 'RELAY_wm/STATUS.json'),
                        ('bb_wait', EVIDENCE / 'BB_FOLLOW/WAIT.json'),
                        ('bb_stop', EVIDENCE / 'BB_FOLLOW/STOP.json'),
                        ('bb_complete', EVIDENCE / 'BB_FOLLOW/COMPLETE.json')]:
        out[label] = read_json(path)
    correction = read_json(EVIDENCE / 'BB_FOLLOW_WAIT_RESUME_DISPATCH.json')
    follower = correction['pid'] if correction else 43404
    parked = read_json(EVIDENCE / 'BB_PARKED_0600.json')
    out['bb_relocation_stop'] = read_json(EVIDENCE / 'BB_RELOCATION_STOP.json')
    out['local_processes'] = {str(pid): local_process(pid) for pid in (31696, follower)}
    relay_complete = read_json(EVIDENCE / 'RELAY_wm/COMPLETE.json')
    if out['local_processes']['31696']['state'] != 'LIVE' and not relay_complete:
        out['alerts'].append('WM relay process missing before completion')
    if out['local_processes'][str(follower)]['state'] != 'LIVE' and not (
            out['bb_complete'] or out['bb_stop'] or parked or out['bb_relocation_stop']):
        out['alerts'].append('current BB follower process missing before completion')
    for host, data in out['hosts'].items():
        for kind in ('cpu', 'gpu'):
            for arm, row in data[kind].items():
                if row.get('stop'):
                    out['alerts'].append(f'{host}/{kind}/{arm}: STOP {row["stop"]!r}')
                proc = row.get('process')
                if proc and proc['state'] not in ('EXITED', 'Z') and not proc['own_root_in_command']:
                    out['alerts'].append(f'{host}/{kind}/{arm}: PID identity mismatch')
                if proc and proc['state'] in ('EXITED', 'Z') and not (row.get('complete') or row.get('finished') or row.get('stop')):
                    out['alerts'].append(f'{host}/{kind}/{arm}: unexpected process exit')
    if out['bb_stop']:
        out['alerts'].append(f'BB follower terminal STOP: {out["bb_stop"]!r}')
    if out['bb_relocation_stop']:
        out['alerts'].append(f'BB relocated startup terminal STOP: {out["bb_relocation_stop"]["error"]}')
    previous = read_json(WATCH / 'STATUS.json')
    if previous:
        for host, data in out['hosts'].items():
            for arm, row in data['cpu'].items():
                old = previous.get('hosts', {}).get(host, {}).get('cpu', {}).get(arm)
                if old and row.get('process') and not row['complete'] and not row['stop']:
                    keys = ('episodes', 'decisions', 'closed_jobs', 'charged_sim_s')
                    row['no_progress_ticks'] = old.get('no_progress_ticks', 0) + 1 if all(row[k] == old.get(k) for k in keys) else 0
                    if row['no_progress_ticks'] >= 2:
                        out['alerts'].append(f'{host}/cpu/{arm}: no progress across two heartbeats')
    registration = Path(REGISTRATION).read_text(encoding='utf-8')
    out['latest_task_registration'] = [line for line in registration.splitlines()
        if 'S-C1SWITCH2' in line][-5:]
    out['final_checkpoint_receipts'] = {name: (ROOT / name).is_file()
        for name in ('SELMAPPO_FINAL.json', 'SELMAPPO2_FINAL.json')}
    return out


def summary(snapshot):
    rows = [v['cpu'] for v in snapshot['hosts'].values()]
    parts = []
    for arm in ('cpu', 'wm', 'bb', 'fleet_ppo', 'cav_mappo', 'wm_oracles'):
        arms = [r[arm] for r in rows if arm in r and r[arm].get('process')]
        if arms:
            parts.append(f'{arm}:episodes={sum(r["episodes"] for r in arms)},'
                         f'decisions={sum(r["decisions"] for r in arms)},'
                         f'charged={sum(r["charged_sim_s"] for r in arms):g},'
                         f'closedactual={sum(r["closed_actual_sim_s"] for r in arms):g}')
    parts.append(f'alerts={len(snapshot["alerts"])}')
    return '; '.join(parts)


def commit_snapshot(snapshot):
    """Stage only this heartbeat's explicit files; refuse an already-staged index."""
    def git(*args):
        return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    staged = git('diff', '--cached', '--name-only')
    if staged.strip():
        raise RuntimeError('existing staged changes; heartbeat preserves index and refuses commit')
    stamp = str(int(snapshot['unix']))
    receipt = WATCH / ('PROGRESS_' + stamp + '.json')
    if receipt.exists():
        raise RuntimeError('immutable heartbeat already exists')
    write(receipt, snapshot)
    write(WATCH / 'STATUS.json', snapshot)
    timestamp = datetime.fromtimestamp(snapshot['unix'], timezone.utc).astimezone().isoformat(timespec='seconds')
    line = f'- {timestamp} {summary(snapshot)}; receipt `{receipt.as_posix()}`.\n'
    handoff = ROOT / 'HANDOFF.md'
    with handoff.open('a', encoding='utf-8') as stream:
        stream.write('\n' + line)
    paths = ['HANDOFF.md', receipt.as_posix(), (WATCH / 'STATUS.json').as_posix()]
    print(git('status', '--short'), end='')
    print(git('diff', '--stat', '--', *paths), end='')
    print(git('log', '--oneline', '-10'), end='')
    git('diff', '--check', '--', *paths)
    git('add', '--', *paths)
    print(git('commit', '-m', f'C1SWITCH2: owned-job heartbeat {timestamp[11:16].replace(":", "")}'), end='')
    print(line, end='', flush=True)


def run(*, resume=False):
    if not WATCH.is_dir():
        raise RuntimeError('watcher must be launched with its existing explicit receipt directory')
    if resume and not (WATCH / 'RESUME_ONCE.json').is_file():
        raise RuntimeError('explicit observer-only continuation claim required')
    with (WATCH / ('RUN_RESUME_ONCE.json' if resume else 'RUN_ONCE.json')).open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), unix=time.time(), deadline_unix=DEADLINE, interval_s=1200), stream)
    while True:
        started = time.time()
        try:
            commit_snapshot(collect())
        except Exception as exc:
            write(WATCH / 'ERROR.json', dict(unix=time.time(), error=repr(exc)))
            print(repr(exc), flush=True)
        if time.time() >= DEADLINE:
            write(WATCH / 'COMPLETE.json', dict(unix=time.time(), reason='noon monitoring deadline reached'))
            return
        time.sleep(max(0, min(started + 1200, DEADLINE) - time.time()))


def start(*, resume=False):
    if not resume and ((WATCH / 'DISPATCH.json').exists() or (WATCH / 'RUN_ONCE.json').exists()):
        raise RuntimeError('owned watcher already dispatched; never duplicate')
    if resume:
        original = read_json(WATCH / 'DISPATCH.json')
        if not original or local_process(original['pid'])['state'] != 'EXITED':
            raise RuntimeError('previous owned watcher must be exited before replacement')
        with (WATCH / 'RESUME_ONCE.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(predecessor=original, unix=time.time(), reason='track corrected follower PID'), stream)
    if not EVIDENCE.is_dir():
        raise RuntimeError('evidence parent absent')
    WATCH.mkdir(exist_ok=True)
    with (WATCH / ('WATCH_RESUME.log' if resume else 'WATCH.log')).open('x', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'tools.c1switch2.watch',
            'run-resume' if resume else 'run'],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS)
    result = dict(pid=process.pid, unix=time.time(), deadline_unix=DEADLINE, interval_s=1200)
    write(WATCH / ('RESUME_DISPATCH.json' if resume else 'DISPATCH.json'), result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('start', 'run', 'once', 'resume-start', 'run-resume'))
    command = parser.parse_args().command
    if command == 'start':
        start()
    elif command == 'resume-start':
        start(resume=True)
    elif command == 'run-resume':
        run(resume=True)
    elif command == 'run':
        run()
    else:
        commit_snapshot(collect())
