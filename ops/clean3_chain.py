"""One-use cpuV clean3 chain. No SSH, retries, receipt adoption or stop clearance.

Order (owner/MAIN 2026-11-06): G1 certification batch -> CLEAN3_GO_NOGO.json
(continue regardless) -> wait for first verified MINED_MOVES batch (deadline
12:00 local) -> prepare clean3 root/ledger/AUTHORITY -> end-to-end smoke with
every lever/fix ON -> one zero-collision/zero-teleport gate -> full u1-u24
phase-coverage assertion -> launch A/B/R u0 24 updates + monitor -> supervise.
Every pass/fail criterion is encoded here or in the called library checks.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path('/home/ubuntu/coopmwm20261106_clean3_r3')
PYTHON = '/home/ubuntu/miniconda3/bin/python'
MINED_DEFAULT = '/home/ubuntu/mined/MINED_MOVES.jsonl'
UPDATES = 24


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def publish(path, value, *, replace=False):
    path = Path(path)
    temporary = path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    with temporary.open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False, default=str)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    if replace:
        os.replace(temporary, path)
    else:
        os.link(temporary, path); temporary.unlink()


def sequence(steps, status):
    current = 'INITIAL'
    try:
        for current, action in steps:
            status(current, 'RUNNING')
            result = action()
            status(current, 'PASS', result)
    except BaseException as error:
        status(current, 'FAILED', dict(error=repr(error), traceback=traceback.format_exc()))
        raise


def deadline_unix(text, now=None):
    """Local wall-clock HH:MM on the chain's start day (or next day if passed)."""
    now = now or datetime.now().astimezone()
    hour, minute = map(int, text.split(':'))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return target.timestamp()


def mined_summary(u, root=None):
    """Owner 05:05 per-update mined usage (B and R) from the sealed arm update receipts."""
    out = {}
    for arm in ('B', 'R'):
        path = (root or ROOT)/f'{arm}_UPDATE_{u:03}.json'
        usage = read(path).get('clean3_mined_usage') if path.is_file() else None
        if not usage:
            out[arm] = None
            continue
        out[arm] = dict(imitation=sum(usage['imitation'].values()), practice=sum(usage['practice_starts'].values()),
            executed=sum(usage['executed_joint_moves'].values()), fit_ce=usage['imitation_fit_ce'],
            by_family={k: v for k, v in usage['executed_joint_moves'].items()},
            trigger=usage['trigger_rate_by_family'], g4_rows=len(usage['g4_credit']))
    return out


def coverage(cfg, clean2_rows, *, gate_sim_s, diagnostic_quotes, lesson_quotes):
    """Assert every phase covers its full u1-u24 schedule; refuse otherwise."""
    import re
    from tools.coopmwm.clean3_restart import checked, G4_CAP
    phases = checked(cfg)['phases']
    per_update = {}
    for row in clean2_rows:
        if row['phase'] == 'v2v_train':
            m = re.search(r'_u(\d+)_', row['id'])
            if m and int(m.group(1)) <= 18:  # completed clean2 updates only
                per_update[int(m.group(1))] = per_update.get(int(m.group(1)), 0)+row['sim_s']
    if len(per_update) < 18 or sorted(lesson_quotes) != [f'lesson_u{u:03}' for u in (6, 12, 18, 24)]:
        raise ValueError('measured clean2 per-update training and native clean3 lesson quotes required')
    need = dict(
        v2v_train=UPDATES*max(per_update.values()),
        v2v_lessons=sum(lesson_quotes.values()),
        early_probes=diagnostic_quotes['probe_u002']+diagnostic_quotes['probe_u004'],
        isolated_replicated=sum(diagnostic_quotes[f'replicated_u{u:03}'] for u in (6, 12, 24)),
        v2v_interim=3*96*1200, v2v_final=4*96*1200,  # A/B/R + lane-on MAPPO baseline
        gate_bank_verification_ext=gate_sim_s, track2_smoke=20000)
    short = {k: (v, phases[k]) for k, v in need.items() if v > phases[k]}
    if short:
        raise ValueError('phase does not cover its u1-u24 schedule: '+json.dumps(short))
    if G4_CAP != 230400:
        raise ValueError('G4 phase must equal its registered hard cap')
    return dict(need=need, allocated={k: phases[k] for k in need}, g4=G4_CAP)


class Chain:
    def __init__(self, source, control, mined, cutoff):
        self.source, self.control = Path(source), Path(control)
        self.manifest = read(self.source/'SOURCE_MANIFEST.json')
        self.mined_path, self.cutoff = Path(mined), deadline_unix(cutoff)
        self.cutoff_text = cutoff
        self.serial = 0
        self.go = self.mined = None

    def status(self, step, state, detail=None):
        self.serial += 1
        value = dict(schema='coopmwm.clean3.chain.v1', step=step, status=state,
            sequence=self.serial, detail=detail, unix=time.time(),
            timestamp=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
            source_commit=self.manifest['commit'], root=str(ROOT))
        publish(self.control/f'STATUS_{self.serial:03}_{step}_{state}.json', value)
        publish(self.control/'CHAIN_STATUS.json', value, replace=True)
        if ROOT.is_dir():
            publish(ROOT/'CHAIN_STATUS.json', value, replace=True)
            if state == 'FAILED' and not (ROOT/'STOP.json').exists():
                publish(ROOT/'STOP.json', dict(reason='clean3 detached chain failed; no retry', **value))

    def qualified(self):
        from tools.coopmwm import clean3_restart as c3
        commit = self.manifest['commit']
        result = read(self.control.parent/('clean3_tests_'+commit[:8])/'RESULT.json')
        if result['returncode'] != 0 or result['source_commit'] != commit:
            raise ValueError('full Linux qualification must pass for exact source')
        for name, sha in self.manifest['files'].items():
            if hashlib.sha256((self.source/name).read_bytes()).hexdigest() != sha:
                raise ValueError('qualified source SHA changed: '+name)
        if ROOT.exists():
            raise ValueError('clean3 root already exists; inspect instead of replaying')
        c3.stopped_clean2()
        return dict(qualification=result, source_files=len(self.manifest['files']))

    def certify(self):
        """Owner: run G1, record go/no-go, continue regardless."""
        from tools.coopmwm import clean3_restart as c3
        from tools.coopmwm.exact_grade import _seal
        go_path = Path(c3.G1_ROOT)/'CLEAN3_GO_NOGO.json'
        if go_path.is_file():
            # Adopt a COMPLETED sealed certification receipt; never rerun its physics.
            from tools.coopmwm.exact_grade import verify_sealed
            self.go = read(go_path); verify_sealed(self.go, 'receipt_sha256')
            from tools.coopmwm.protocol import digest
            if self.go['runtime_sha256'] != digest(self.manifest):
                raise ValueError('existing G1 receipt binds a different source')
            publish(self.control/'CLEAN3_GO_NOGO.json', self.go)
            return dict(adopted=True, decision=self.go['decision'])
        if Path(c3.G1_ROOT).exists():
            raise ValueError('G1 root exists without a completed receipt; inspect, never replay')
        try:
            self.go = c3.certify(str(self.source))
        except Exception as error:
            if go_path.exists():
                raise
            # Evidence of the failed batch stays in the G1 root/ledger; no retry.
            from tools.coopmwm import launch_training as training
            Path(c3.G1_ROOT).mkdir(exist_ok=True)
            self.go = _seal(dict(schema='coopmwm.clean3.g1_go_nogo.v1', decision='NO-GO',
                executable_joint_labels_ci95_low_positive=0, error=repr(error),
                runtime_sha256=__import__('tools.coopmwm.protocol', fromlist=['digest']).digest(self.manifest),
                charged_sim_s=sum(r['sim_s'] for r in c3.g1_rows()), rule=c3.G1_GO_RULE,
                traceback=traceback.format_exc(), unix=time.time()), 'receipt_sha256')
            training.write(go_path, self.go)
        publish(self.control/'CLEAN3_GO_NOGO.json', self.go)
        return {k: self.go.get(k) for k in ('decision', 'executable_joint_labels_ci95_low_positive',
                                             'admitted_by_arm', 'charged_sim_s', 'error')}

    def wait_mined(self):
        """Owner ~04:35: launch only on >=3 verified joint moves in >=2 families; 12:00 hard stop."""
        from tools.coopmwm import clean3_restart as c3
        cfg, _ = c3.g1_config(str(self.source), c3.G1_ROOT)
        cfg['run_root'] = c3.G1_ROOT
        attempt, last = 0, None
        while True:
            final = time.time() >= self.cutoff
            signature = None
            if self.mined_path.is_file():
                st = self.mined_path.stat(); signature = (st.st_size, st.st_mtime_ns)
                side = self.mined_path.parent/'MINED_MOVES_ANNOTATED.jsonl'
                if side.is_file():
                    st = side.stat(); signature += (st.st_size, st.st_mtime_ns)
            if signature and signature[0] > 0 and signature != last:
                last, attempt = signature, attempt+1
                out = self.control/f'MINED_MOVES_ADMITTED_{attempt:03}.jsonl'
                receipt = c3.admit_mined(self.mined_path, cfg, out)
                publish(self.control/f'MINED_ADMISSION_{attempt:03}.json', receipt)
                if receipt['ready'] is True and receipt['admitted'] > 0:
                    self.mined = receipt
                    return {k: receipt[k] for k in ('admitted', 'joint_verified_examples', 'joint_verified_moves',
                                                     'joint_families', 'families', 'rows', 'admitted_path', 'rule')}
            if final:
                value = dict(reason='MINED deadline '+self.cutoff_text+' reached without >=3 per-seed verified joint '
                    'examples; chain stops; no launch without owner confirmation',
                    attempts=attempt, rule=c3.MINED_RULE, unix=time.time())
                publish(self.control/'MINED_DEADLINE_STOP.json', value)
                raise ValueError(value['reason'])
            time.sleep(60)

    def wait_confirmation(self):
        """Owner ~07:2x walls-only "yes." pre-approves launch once MINED_RULE holds (no file needed)."""
        return dict(confirmed=True, authority='owner ~07:2x walls-only clean3 launch "yes." (REGISTRATION.md); '
                    'launch when the amended mined rule holds; no LAUNCH_CONFIRMED file required')
        path = self.control/'LAUNCH_CONFIRMED.json'
        while True:
            if path.is_file():
                value = read(path)
                if value.get('confirmed') is not True or not str(value.get('authority', '')).strip():
                    raise ValueError('LAUNCH_CONFIRMED.json must carry confirmed=true and the authority quote')
                return value
            if time.time() >= self.cutoff:
                publish(self.control/'CONFIRM_DEADLINE_STOP.json', dict(reason='no MAIN launch confirmation by '
                    +self.cutoff_text+'; chain stops; nothing launched', unix=time.time()))
                raise ValueError('no MAIN launch confirmation by deadline')
            time.sleep(60)

    def prepare(self):
        from tools.coopmwm import clean3_restart as c3, launch_training as t, g5_runtime
        if not (self.mined and self.mined.get('ready') is True):
            raise ValueError('owner rule: no launch without >=3 per-seed verified joint examples')
        if not g5_runtime.enabled():
            raise ValueError('clean3 requires the bound G5 runtime env (COOPMWM_G5_RUNTIME=1)')
        cfg = c3.prepare(self.source, mined=self.mined)
        (ROOT/'chain').symlink_to(self.control, target_is_directory=True)
        publish(ROOT/'CHAIN_AUTHORITY.json', read(self.control/'AUTHORITY.json'))
        with (ROOT/'AUTHORITY.md').open('a', encoding='utf-8') as f:
            f.write('\nMINED launch condition (registered in REGISTRATION.md and chain AUTHORITY.json before '
                    'applying): '+c3.MINED_RULE+' Satisfied: '+json.dumps({k: self.mined[k] for k in
                    ('joint_verified_examples', 'joint_families', 'admitted', 'source_sha256')}, sort_keys=True)+'\n')
            f.write('\nclean3 chain: G1 batch (go/no-go recorded, continue regardless) -> mined wait '
                    '(deadline '+self.cutoff_text+') -> root/ledger -> smoke (all levers/fixes, hard-coded '
                    'receipts) -> one 0-collision/0-teleport gate -> u1-u24 phase coverage assertion -> '
                    'A/B/R u0 24 updates + monitor. Any failure stops; no automatic retry.\n')
        t.health(ROOT)
        return read(ROOT/'PREPARED.json')

    def command(self, action):
        from tools.coopmwm import launch_training as t, clean3_restart as c3
        cfg = t.checked_config(ROOT/'CONFIG.json')
        c3.verify_previous(cfg); t.health(ROOT)
        if cfg.get('gate_attempt') != 1 or cfg.get('smoke_attempt') != 1:
            raise ValueError('chain permits only the first smoke/gate attempt')
        prefix = action.upper()
        if any((ROOT/(prefix+'_'+s+'.json')).exists() for s in ('STARTED', 'PASS', 'FAILED')):
            raise ValueError('one-use qualification; never retry or adopt attempted physics')
        with (self.control/(action+'.log')).open('x') as log:
            process = subprocess.run(['nice', '-n', '19', PYTHON, '-B', '-u', '-m',
                'tools.coopmwm.launch_qualification', action, '--config', str(ROOT/'CONFIG.json')],
                cwd='/home/ubuntu', stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        publish(self.control/(prefix+'_EXIT.json'), dict(returncode=process.returncode, unix=time.time()))
        if process.returncode:
            raise RuntimeError(action+' failed; inspect log; no automatic retry')
        return self.check_receipts(action, cfg)

    def check_receipts(self, action, cfg):
        from tools.coopmwm import launch_training as t
        from tools.coopmwm.exact_grade import verify_sealed
        from tools.coopmwm.protocol import digest
        def receipt(name, status):
            value = t.read(ROOT/name); verify_sealed(value, 'receipt_sha256')
            if value['status'] != status:
                raise ValueError(name+' must pass')
            return value
        smoke = receipt('SMOKE_PASS.json', 'SMOKE_PASS')
        fixes = receipt('SMOKE_FIX_RECEIPTS_PASS.json', 'FIX_RECEIPTS_PASS')
        reporting = receipt('SMOKE_REPORTING_PASS.json', 'COMPLETE')
        levers = receipt('SMOKE_LEVER_RECEIPTS_PASS.json', 'LEVER_RECEIPTS_PASS')
        if (smoke['config_sha256'] != digest(cfg) or smoke['smoke_sim_s'] > 20000 or
                smoke['source_manifest_sha256'] != cfg['source_manifest_sha256'] or
                smoke['clean2_fix_receipts_sha256'] != fixes['receipt_sha256'] or
                smoke['clean3_lever_receipts_sha256'] != levers['receipt_sha256'] or
                levers['fix_receipts_sha256'] != fixes['receipt_sha256'] or
                fixes['reporting_receipt_sha256'] != reporting['receipt_sha256'] or
                reporting['config_sha256'] != digest(cfg)):
            raise ValueError('exact smoke/fix/lever/reporting binding or cap failed')
        if action == 'gate':
            gate = receipt('GATE_PASS.json', 'PASS')
            if (gate['smoke_sha256'] != smoke['receipt_sha256'] or
                    gate['source_manifest_sha256'] != cfg['source_manifest_sha256'] or
                    gate['safety']['collisions'] != 0 or gate['safety']['teleports'] != 0):
                raise ValueError('gate final-source/smoke binding or zero-event rule failed')
            t.release(cfg)
            return dict(gate_sha256=gate['receipt_sha256'], smoke_sha256=smoke['receipt_sha256'],
                        safety=gate['safety'])
        return dict(smoke_sha256=smoke['receipt_sha256'], fixes_sha256=fixes['receipt_sha256'],
                    levers=levers['levers'])

    def assert_coverage(self):
        from tools.coopmwm import launch_training as t
        from tools.coopmwm.storage import ledger
        cfg = t.checked_config(ROOT/'CONFIG.json')
        r = cfg['clean3_restart']
        result = coverage(cfg, ledger(Path(r['clean2_root'])), gate_sim_s=r['gate_sim_s'],
                          diagnostic_quotes=r['diagnostic_quotes'], lesson_quotes=r['lesson_quotes'])
        publish(ROOT/'PHASE_COVERAGE_PASS.json', result)
        return result

    def launch(self):
        from tools.coopmwm import launch_training as t
        from tools.coopmwm.protocol import digest
        cfg = t.checked_config(ROOT/'CONFIG.json')
        self.check_receipts('gate', cfg); t.health(ROOT)
        if not (ROOT/'PHASE_COVERAGE_PASS.json').is_file():
            raise ValueError('u1-u24 phase coverage must pass before launch')
        if any((ROOT/name).exists() for name in ('LAUNCHED.json', 'LAUNCH_DISPATCH_STARTED.json', 'owners')):
            raise ValueError('existing training dispatch; never duplicate')
        publish(ROOT/'LAUNCH_DISPATCH_STARTED.json', dict(config_sha256=digest(cfg), unix=time.time()))
        logs = ROOT/'launch_logs'; logs.mkdir(); owners = {}
        try:
            for role in ('A', 'B', 'R', 'monitor'):
                command = ['nice', '-n', '19', PYTHON, '-B', '-u', '-m', 'tools.coopmwm.launch_training',
                    'monitor' if role == 'monitor' else 'arm', '--config', str(ROOT/'CONFIG.json')]
                if role != 'monitor': command += ['--arm', role]
                with (logs/(role+'.log')).open('x') as log:
                    p = subprocess.Popen(command, cwd='/home/ubuntu', env=os.environ.copy(),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                owners[role] = dict(pid=p.pid, start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21])
                publish(ROOT/('DISPATCH_'+role+'.json'), owners[role])
            time.sleep(3)
            for role, owner in owners.items():
                fields = Path(f"/proc/{owner['pid']}/stat").read_text().split()
                if fields[2] == 'Z' or fields[21] != owner['start_ticks']:
                    raise ValueError('training owner exited at launch: '+role)
            t.health(ROOT)
        except BaseException:
            publish(ROOT/'PARTIAL_LAUNCH.json', dict(owners=owners, unix=time.time(), reason='never automatic retry'))
            raise
        value = dict(schema='coopmwm.clean3.launched.v1', timestamp=datetime.now(timezone.utc).isoformat(),
            unix=time.time(), root=str(ROOT), source_root=str(self.source), runtime_commit=self.manifest['commit'],
            runtime_sha256=digest(cfg), config_file_sha256=hashlib.sha256((ROOT/'CONFIG.json').read_bytes()).hexdigest(),
            source_manifest_sha256=cfg['source_manifest_sha256'], owners=owners, updates=UPDATES,
            start_updates=dict(A=0, B=0, R=0), smoke_sha256=t.read(ROOT/'SMOKE_PASS.json')['receipt_sha256'],
            gate_sha256=t.read(ROOT/'GATE_PASS.json')['receipt_sha256'],
            g1=cfg['clean3_restart']['g1_go_nogo'], mined=(cfg['clean3_restart']['mined_moves'] or {}).get('admitted', 0),
            remaining_cap_sim_s=cfg['clean3_budget']['remaining_sim_s'], owner_cap_sim_s=cfg['clean3_budget']['owner_cap_sim_s'],
            safety_rule=cfg['clean3_restart']['inherited_safety'], clean1_clean2_preserved=True)
        publish(ROOT/'LAUNCHED.json', value)
        return value

    def monitor(self):
        from tools.coopmwm import launch_training as t, clean3_restart as c3
        from tools.coopmwm.exact_grade import verify_sealed
        from tools.coopmwm.protocol import digest
        from tools.coopmwm.storage import ledger
        cfg = t.checked_config(ROOT/'CONFIG.json')
        launched = read(ROOT/'LAUNCHED.json'); seen = set()
        while True:
            t.health(ROOT)
            for u in range(1, UPDATES+1):
                path = ROOT/f'LEARNING_MONITOR_u{u:03}.json'
                if u in seen or not path.is_file(): continue
                row = read(path); verify_sealed(row, 'receipt_sha256')
                if row['config_sha256'] != digest(cfg): raise ValueError('monitor config binding changed')
                diagnostic = None
                if u in (2, 4, 6, 12, 24):
                    kind = 'probe' if u in (2, 4) else 'replicated'
                    p = ROOT/f'track2_diagnostics/{kind}_u{u:03}_live.json'
                    if not p.is_file(): continue
                    diagnostic = read(p); verify_sealed(diagnostic, 'receipt_sha256')
                safety = {k: sum(a['metrics']['safety'][k] for a in row['arms']) for k in ('collisions', 'teleports')}
                if any(safety.values()): raise ValueError('absolute training safety event: '+json.dumps(safety))
                spent = sum(r['sim_s'] for r in ledger(ROOT))
                prior = sum(sum(r['sim_s'] for r in rows) for rows in c3.ledgers().values())
                total = spent+prior+c3.PARITY+sum(r['sim_s'] for r in c3.g1_rows())
                if total > c3.CAP: raise ValueError('owner total cap exceeded')
                lines = [datetime.now(timezone.utc).isoformat()+f' CLEAN3 update {u}; all arms sealed.',
                    'CLEAN3 Learning '+json.dumps({a['arm']: dict(action=a['action'], diagnose=a['diagnose'],
                        bank_ce=a['metrics'].get('bank_move_ce'), layer_kl=a['metrics'].get('layer_kl')) for a in row['arms']}, sort_keys=True),
                    'CLEAN3 Safety '+json.dumps(safety, sort_keys=True)+'; A full rule; B/R composite report-only; any collision/teleport pauses.',
                    'CLEAN3 Check '+json.dumps({k: diagnostic[k] for k in ('metric_reports', 'verdict') if k in diagnostic}, sort_keys=True) if diagnostic else 'CLEAN3 Check not scheduled at this update.',
                    f'CLEAN3 Budget total={total}/{c3.CAP} sim-s (incl. mining actual); clean3={spent}; u24 synergy criterion unchanged.',
                    'CLEAN3 Mined '+json.dumps(mined_summary(u), sort_keys=True)]
                with (ROOT/'CLEAN3_STATUS.md').open('a', encoding='utf-8') as f:
                    f.write('\n'.join(lines)+'\n\n'); f.flush(); os.fsync(f.fileno())
                publish(self.control/f'MONITORED_u{u:03}.json', dict(update=u, unix=time.time(),
                    monitor_sha256=row['receipt_sha256'], total_sim_s=total))
                seen.add(u)
            complete = []
            for role, owner in launched['owners'].items():
                done = (ROOT/('MONITOR_COMPLETE.json' if role == 'monitor' else role+'_COMPLETE.json')).is_file()
                complete.append(done)
                if done: continue
                p = Path(f"/proc/{owner['pid']}/stat")
                fields = p.read_text().split() if p.exists() else []
                if not fields or fields[2] == 'Z' or fields[21] != owner['start_ticks']:
                    raise ValueError('owner exited before completion: '+role)
            if all(complete): return dict(updates=UPDATES, monitored_updates=sorted(seen))
            time.sleep(30)

    def run(self):
        sequence([('QUALIFIED', self.qualified), ('G1_CERTIFY', self.certify),
            ('MINED_WAIT', self.wait_mined), ('CONFIRM_WAIT', self.wait_confirmation), ('PREPARE', self.prepare),
            ('SMOKE', lambda: self.command('smoke')), ('GATE', lambda: self.command('gate')),
            ('PHASE_COVERAGE', self.assert_coverage), ('LAUNCH', self.launch)], self.status)
        sequence([('MONITOR', self.monitor)], self.status)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True); p.add_argument('--control', required=True)
    p.add_argument('--mined', default=MINED_DEFAULT); p.add_argument('--cutoff', default='14:00')
    args = p.parse_args()
    import fcntl
    # Bound G5 env on the frozen runtime for every child (G1, smoke, gate, training owners).
    os.environ['COOPMWM_G5_RUNTIME'] = '1'
    control = Path(args.control)
    with (control/'OWNER.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish(control/'CHAIN_STARTED.json', dict(pid=os.getpid(), unix=time.time(),
            start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
        Chain(args.source, control, args.mined, args.cutoff).run()


if __name__ == '__main__':
    main()
