"""One-use cpuV clean2 chain. No SSH, retries, receipt adoption or stop clearance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


ROOT = Path('/home/ubuntu/coopmwm20261105_clean2')
PYTHON = '/home/ubuntu/miniconda3/bin/python'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def publish(path, value, *, replace=False):
    path = Path(path)
    temporary = path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    with temporary.open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    if replace:
        os.replace(temporary, path)
    else:
        os.link(temporary, path); temporary.unlink()


def sequence(steps, status):
    """Fail closed at the first exception, including failed receipt checks."""
    current = 'INITIAL'
    try:
        for current, action in steps:
            status(current, 'RUNNING')
            result = action()
            status(current, 'PASS', result)
    except BaseException as error:
        status(current, 'FAILED', dict(error=repr(error), traceback=traceback.format_exc()))
        raise


class Chain:
    def __init__(self, source, control):
        self.source, self.control = Path(source), Path(control)
        self.manifest = read(self.source/'SOURCE_MANIFEST.json')
        self.cfg = None
        self.serial = 0

    def status(self, step, state, detail=None):
        self.serial += 1
        value = dict(schema='coopmwm.clean2.chain.v1', step=step, status=state,
            sequence=self.serial, detail=detail, unix=time.time(),
            timestamp=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
            source_commit=self.manifest['commit'], root=str(ROOT))
        publish(self.control/f'STATUS_{self.serial:03}_{step}_{state}.json', value)
        publish(self.control/'CHAIN_STATUS.json', value, replace=True)
        if ROOT.is_dir():
            publish(ROOT/'CHAIN_STATUS.json', value, replace=True)
            if state == 'FAILED' and not (ROOT/'STOP.json').exists():
                publish(ROOT/'STOP.json', dict(reason='clean2 detached chain failed; no retry', **value))

    def qualified(self):
        commit = self.manifest['commit']
        q = self.control.parent/('clean2_tests_'+commit[:8])
        result = read(q/'RESULT.json')
        if result['returncode'] != 0 or result['source_commit'] != commit:
            raise ValueError('full Linux qualification must pass for exact source')
        for name, sha in self.manifest['files'].items():
            if hashlib.sha256((self.source/name).read_bytes()).hexdigest() != sha:
                raise ValueError('qualified source SHA changed: '+name)
        if ROOT.exists():
            raise ValueError('clean2 already exists; inspect instead of replaying')
        return dict(qualification=result, source_files=len(self.manifest['files']))

    def prepare(self):
        # The existing receipt/accounting contract requires its final root and
        # sole ledger before physics. Preparation creates no training dispatch.
        # Smoke/gate remain prerequisites for all A/B/R training owners.
        from tools.coopmwm import launch_clean2 as c, launch_training as t
        self.cfg = c.prepare(self.source)
        publish(ROOT/'CHAIN_AUTHORITY.json', read(self.control/'AUTHORITY.json'))
        (ROOT/'chain').symlink_to(self.control, target_is_directory=True)
        with (ROOT/'AUTHORITY.md').open('a', encoding='utf-8') as f:
            f.write('\nMAIN18:19: one cpuV detached chain; exact existing smoke receipt checks -> '
                'one zero-collision/zero-teleport gate -> A/B/R u0 24 updates plus P3 monitor. '
                'Any environment or receipt failure stops; no automatic retry. '
                'Final root/sole ledger prepared before physics to retain exact config and '
                'append-only accounting identities. No training before both passes.\n')
        c.verify_previous(self.cfg)
        t.health(ROOT)
        return read(ROOT/'PREPARED.json')

    def command(self, action):
        from tools.coopmwm import launch_clean2 as c, launch_training as t
        cfg = t.checked_config(ROOT/'CONFIG.json')
        c.verify_previous(cfg); t.health(ROOT)
        if cfg.get('gate_attempt') != 1 or cfg.get('smoke_attempt') != 1:
            raise ValueError('chain permits only the first smoke/gate attempt')
        prefix = action.upper()
        if any((ROOT/(prefix+'_'+suffix+'.json')).exists() for suffix in ('STARTED','PASS','FAILED')):
            raise ValueError('one-use qualification; never retry or adopt attempted physics')
        with (self.control/(action+'.log')).open('x') as log:
            process = subprocess.run(['nice','-n','19',PYTHON,'-B','-u','-m',
                'tools.coopmwm.launch_qualification',action,'--config',str(ROOT/'CONFIG.json')],
                cwd='/home/ubuntu',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        publish(self.control/(prefix+'_EXIT.json'), dict(returncode=process.returncode, unix=time.time()))
        if process.returncode:
            raise RuntimeError(action+' failed; inspect log; no automatic retry')
        return self.check_receipts(action, cfg)

    def check_receipts(self, action, cfg):
        from tools.coopmwm import launch_training as t
        from tools.coopmwm.exact_grade import verify_sealed
        from tools.coopmwm.protocol import digest
        def receipt(name, status):
            value = t.read(ROOT/name); verify_sealed(value,'receipt_sha256')
            if value['status'] != status:
                raise ValueError(name+' must pass')
            return value
        smoke = receipt('SMOKE_PASS.json','SMOKE_PASS')
        fixes = receipt('SMOKE_FIX_RECEIPTS_PASS.json','FIX_RECEIPTS_PASS')
        reporting = receipt('SMOKE_REPORTING_PASS.json','COMPLETE')
        if (smoke['config_sha256'] != digest(cfg) or smoke['smoke_sim_s'] > 20000 or
                smoke['source_manifest_sha256'] != cfg['source_manifest_sha256'] or
                smoke['clean2_fix_receipts_sha256'] != fixes['receipt_sha256'] or
                fixes['reporting_receipt_sha256'] != reporting['receipt_sha256'] or
                reporting['config_sha256'] != digest(cfg)):
            raise ValueError('exact smoke/fix/reporting binding or cap failed')
        if action == 'gate':
            gate = receipt('GATE_PASS.json','PASS')
            if (gate['smoke_sha256'] != smoke['receipt_sha256'] or
                    gate['source_manifest_sha256'] != cfg['source_manifest_sha256']):
                raise ValueError('gate final-source/smoke binding failed')
            t.release(cfg)  # exact config, zero collisions/teleports, remaining-cap prefix
            return dict(gate_sha256=gate['receipt_sha256'], smoke_sha256=smoke['receipt_sha256'])
        return dict(smoke_sha256=smoke['receipt_sha256'], fixes_sha256=fixes['receipt_sha256'])

    def launch(self):
        from tools.coopmwm import launch_training as t
        from tools.coopmwm.protocol import digest
        cfg = t.checked_config(ROOT/'CONFIG.json')
        self.check_receipts('gate',cfg); t.health(ROOT)
        if any((ROOT/name).exists() for name in ('LAUNCHED.json','LAUNCH_DISPATCH_STARTED.json','owners','updates')):
            raise ValueError('existing training dispatch; never duplicate')
        publish(ROOT/'LAUNCH_DISPATCH_STARTED.json',dict(config_sha256=digest(cfg),unix=time.time()))
        logs = ROOT/'launch_logs'; logs.mkdir(); owners = {}
        try:
            for role in ('A','B','R','monitor'):
                command = ['nice','-n','19',PYTHON,'-B','-u','-m','tools.coopmwm.launch_training',
                    'monitor' if role == 'monitor' else 'arm','--config',str(ROOT/'CONFIG.json')]
                if role != 'monitor': command += ['--arm',role]
                with (logs/(role+'.log')).open('x') as log:
                    p = subprocess.Popen(command,cwd='/home/ubuntu',env=os.environ.copy(),
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                owners[role] = dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21])
                publish(ROOT/('DISPATCH_'+role+'.json'),owners[role])
            time.sleep(3)
            for role, owner in owners.items():
                fields = Path(f"/proc/{owner['pid']}/stat").read_text().split()
                if fields[2] == 'Z' or fields[21] != owner['start_ticks']:
                    raise ValueError('training owner exited at launch: '+role)
            t.health(ROOT)
        except BaseException:
            publish(ROOT/'PARTIAL_LAUNCH.json',dict(owners=owners,unix=time.time(),reason='never automatic retry'))
            raise
        value = dict(schema='coopmwm.clean2.launched.v1',timestamp=datetime.now(timezone.utc).isoformat(),
            unix=time.time(),root=str(ROOT),source_root=str(self.source),runtime_commit=self.manifest['commit'],
            runtime_sha256=digest(cfg),config_file_sha256=hashlib.sha256((ROOT/'CONFIG.json').read_bytes()).hexdigest(),
            source_manifest_sha256=cfg['source_manifest_sha256'],owners=owners,updates=24,
            start_updates=dict(A=0,B=0,R=0),smoke_sha256=t.read(ROOT/'SMOKE_PASS.json')['receipt_sha256'],
            gate_sha256=t.read(ROOT/'GATE_PASS.json')['receipt_sha256'],
            remaining_cap_sim_s=cfg['clean2_budget']['budget']['total'],owner_quote='restart',clean1_preserved_stopped=True)
        publish(ROOT/'LAUNCHED.json',value)
        return value

    def run(self):
        sequence([('QUALIFIED',self.qualified),('PREPARE',self.prepare),
            ('SMOKE',lambda:self.command('smoke')),('GATE',lambda:self.command('gate')),
            ('LAUNCH',self.launch)],self.status)
        sequence([('MONITOR',self.monitor)],self.status)

    def monitor(self):
        """Local public-receipt supervision survives loss of the laptop."""
        from tools.coopmwm import launch_training as t
        from tools.coopmwm.exact_grade import verify_sealed
        from tools.coopmwm.protocol import digest
        from tools.coopmwm.storage import ledger
        launched = read(ROOT/'LAUNCHED.json'); seen = set()
        while True:
            t.health(ROOT)
            cfg = self.cfg
            for u in range(1,25):
                path = ROOT/f'LEARNING_MONITOR_u{u:03}.json'
                if u in seen or not path.is_file(): continue
                row = read(path); verify_sealed(row,'receipt_sha256')
                if row['config_sha256'] != digest(cfg): raise ValueError('monitor config binding changed')
                diagnostic = None
                if u in (2,4,6,12,24):
                    kind = 'probe' if u in (2,4) else 'replicated'
                    p = ROOT/f'track2_diagnostics/{kind}_u{u:03}_live.json'
                    if not p.is_file(): continue
                    diagnostic = read(p); verify_sealed(diagnostic,'receipt_sha256')
                safety = {k:sum(a['metrics']['safety'][k] for a in row['arms']) for k in ('collisions','teleports')}
                if any(safety.values()): raise ValueError('absolute training safety event: '+json.dumps(safety))
                spent = sum(r['sim_s'] for r in ledger(ROOT))
                global_spent = spent+sum(r['sim_s'] for r in ledger(Path(cfg['clean2_restart']['clean1_root'])))+960
                if global_spent > 5000000: raise ValueError('global budget exceeded')
                lines = [datetime.now(timezone.utc).isoformat()+f' CLEAN2 update {u}; all arms sealed.',
                    'CLEAN2 Learning '+json.dumps({a['arm']:dict(action=a['action'],diagnose=a['diagnose'],
                        bank_ce=a['metrics'].get('bank_move_ce'),layer_kl=a['metrics'].get('layer_kl')) for a in row['arms']},sort_keys=True),
                    'CLEAN2 Safety '+json.dumps(safety,sort_keys=True)+'; B/R composite report-only; hard events pause.',
                    'CLEAN2 Check '+json.dumps({k:diagnostic[k] for k in ('metric_reports','verdict') if k in diagnostic},sort_keys=True) if diagnostic else 'CLEAN2 Check not scheduled at this update.',
                    f'CLEAN2 Budget global={global_spent}/5000000 sim-s; clean2={spent}; u24 synergy criterion unchanged.']
                with (ROOT/'CLEAN2_STATUS.md').open('a',encoding='utf-8') as f:
                    f.write('\n'.join(lines)+'\n\n'); f.flush(); os.fsync(f.fileno())
                publish(self.control/f'MONITORED_u{u:03}.json',dict(update=u,unix=time.time(),monitor_sha256=row['receipt_sha256'],global_sim_s=global_spent))
                seen.add(u)
            complete = []
            for role, owner in launched['owners'].items():
                done = (ROOT/('MONITOR_COMPLETE.json' if role=='monitor' else role+'_COMPLETE.json')).is_file()
                complete.append(done)
                if done: continue
                p = Path(f"/proc/{owner['pid']}/stat")
                fields = p.read_text().split() if p.exists() else []
                if not fields or fields[2]=='Z' or fields[21]!=owner['start_ticks']:
                    raise ValueError('owner exited before completion: '+role)
            if all(complete): return dict(updates=24,monitored_updates=sorted(seen))
            time.sleep(30)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--control',required=True)
    args = p.parse_args()
    import fcntl
    control = Path(args.control)
    with (control/'OWNER.lock').open('x') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        publish(control/'CHAIN_STARTED.json',dict(pid=os.getpid(),unix=time.time(),
            start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
        Chain(args.source,control).run()


if __name__ == '__main__':
    main()
