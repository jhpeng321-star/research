"""One locked laptop follower; remote clean3 chain retains all execution ownership."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

BASE = Path('E:/agent-work/_sol')
REPO = Path(__file__).resolve().parents[1]
STATUS = Path('E:/research-dense/runs/active/C4-V7FT-20261017/CLEAN1_STATUS.md')


def fetch(root, control):
    code = (REPO/'ops/clean3_follow_remote.py').read_text()+f'\nprint(json.dumps(probe({root!r}, {control!r}), sort_keys=True))\n'
    helper = ("import sys;sys.dont_write_bytecode=True;sys.path.insert(0,'/mnt/e/agent-work/c4-rerun-launch');"
        "from tools.coopmwm.cpuv_ssh import Transport;"
        "r=Transport().run('python3 -B -',timeout=40,input=sys.stdin.buffer.read(),capture_output=True);"
        "sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)")
    result = subprocess.run(['wsl', '-d', 'Ubuntu-22.04', '--', 'python3', '-c', helper],
                            input=code.encode(), capture_output=True, timeout=55)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace').replace('\x00', ''))
    return json.loads(result.stdout.decode().replace('\x00', ''))


def number(value):
    return 'N/A' if value is None else f'{value:.5g}'


def summary(snapshot, u):
    monitor = snapshot['monitors'][str(u)]
    arms = {row['arm']: row for row in monitor['arms']}
    learning = []
    for arm in ('B', 'R'):
        metrics = arms[arm]['metrics']
        mined = snapshot['mined'].get(str(u), {}).get(arm, {})
        learning.append(arm+' bankCE='+number(metrics.get('bank_move_ce'))+
            ' KL='+number(metrics.get('layer_kl'))+' mined(imitation/practice/executed)='+
            '/'.join(str(sum(mined.get(key, {}).values())) for key in
                     ('imitation', 'practice_starts', 'executed_joint_moves')))
    safety = {key: sum(row['metrics']['safety'][key] for row in arms.values())
              for key in ('collisions', 'teleports')}
    diagnostic = snapshot['diagnostics'].get(str(u))
    checks = []
    if diagnostic:
        for key, label in (('delta_all_new_veh_s', 'total vs common base'),
                           ('policy_cooperation_gain_veh_s', 'group-over-best-single synergy')):
            report = diagnostic.get('metric_reports', {}).get(key, {})
            checks.append(label+' '+json.dumps(report, sort_keys=True))
    stamp = datetime.now().astimezone().isoformat()
    return [stamp+f' CLEAN3 update {u}; A/B/R sealed through u{u:03}.',
        'CLEAN3 Learning '+ '; '.join(learning)+'.',
        'CLEAN3 Safety '+json.dumps(safety, sort_keys=True)+'; '+
            '; '.join(a+':'+arms[a]['action']+' '+','.join(arms[a]['diagnose']) for a in ('A', 'B', 'R'))+
            '; A full rule; B/R composite report-only; any collision/teleport pauses.',
        'CLEAN3 Check '+('; '.join(checks) if diagnostic else 'not scheduled/pending')+'.',
        'CLEAN3 Budget clean3='+str(snapshot['charged_sim_s'])+' sim-s; prior+mining reservation+clean3='+
            str(snapshot['protected_total_sim_s'])+'/'+str(snapshot['cap_sim_s'])+'; u24 synergy criterion unchanged.']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'/home/ubuntu/coopmwm20261106_clean3(?:_r\d+)?', args.root):
        raise ValueError('registered clean3 root required')
    if not re.fullmatch(r'/home/ubuntu/cpuv_setup_20261105/clean3_chain_[a-f0-9]{8}', args.control):
        raise ValueError('exact clean3 control required')
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    with (BASE/'CLEAN3_FOLLOW.lock').open('a+b') as lock:
        import msvcrt
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        state_path = BASE/'CLEAN3_FOLLOW_STATE.json'
        state = json.loads(state_path.read_text()) if state_path.is_file() else dict(
            root=args.root, control=args.control, summarized=[], faults=[])
        if state['root'] != args.root or state['control'] != args.control:
            raise ValueError('preserved follower state binds another experiment; inspect first')
        state['pid'] = os.getpid()
        while not (BASE/'CLEAN3_FOLLOW_STOP.json').exists():
            try:
                snapshot = fetch(args.root, args.control)
                launch = snapshot['launch']
                if launch:
                    if state.get('launch') and state['launch'] != launch:
                        raise ValueError('immutable launch identity changed')
                    state['launch'] = launch
                for key in sorted(snapshot['monitors'], key=int):
                    u = int(key)
                    if u in state['summarized'] or (u in (2, 4, 6, 12, 24) and key not in snapshot['diagnostics']):
                        continue
                    lines = summary(snapshot, u)
                    with STATUS.open('a', encoding='utf-8') as stream:
                        stream.write('\n'.join(lines)+'\n\n'); stream.flush(); os.fsync(stream.fileno())
                    print('\n'.join(lines), flush=True)
                    state['summarized'].append(u)
                for name, value in snapshot['faults'].items():
                    token = name+':'+hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                    if token in state['faults']:
                        continue
                    lines = [datetime.now().astimezone().isoformat()+' CLEAN3 fault '+name+'.',
                        'CLEAN3 Reason '+str(value.get('reason', value.get('error', 'see public receipt')))+'.',
                        'CLEAN3 Root '+args.root+'; exact evidence retained.',
                        'CLEAN3 Remote chain retains pause/control ownership; no follower signals or replay.',
                        'CLEAN3 Any safety failure remains paused; diagnose with tests before any authorized recovery.']
                    with STATUS.open('a', encoding='utf-8') as stream:
                        stream.write('\n'.join(lines)+'\n\n')
                    print('\n'.join(lines), flush=True); state['faults'].append(token)
                state.update(last_probe=snapshot['timestamp'], chain=snapshot['chain'], owners=snapshot['owners'])
                temporary = state_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(state, sort_keys=True, indent=2))
                os.replace(temporary, state_path)
                if args.once or snapshot['faults'] or (snapshot['chain'] or {}).get('status') == 'FAILED':
                    return
                if snapshot['owners'] and all(owner['complete'] for owner in snapshot['owners'].values()):
                    return
            except Exception as error:
                print(datetime.now().astimezone().isoformat()+' CLEAN3 FOLLOW '+repr(error), flush=True)
                if args.once:
                    raise
            time.sleep(60)


if __name__ == '__main__':
    main()
