"""One-use clean5a (wz3to2, cpuZ or cpuV) chain. No SSH, retries, receipt adoption or stop clearance.

Order (MAIN clean5a GO 2026-11-06 19:54): exact-source Linux qualification on this host -> owner budget
binding (4M clean-training envelope, covered re-quote CLEAN5A_QUOTE.json + BUDGET_CLEAN4.json) -> zero-physics prepare of the
clean5a root -> end-to-end smoke with every component ON (hard-coded receipt checks) -> one
zero-collision/zero-teleport safety gate -> full u1-u24 phase-coverage assertion -> launch A/B/R
u0 x24 + monitor -> supervise. Every pass/fail criterion is encoded here or in the called checks.
Environment: COOPMWM_MAP_SPEC (staged wz3to2) and COOPMWM_G5_RUNTIME=1 for every child.
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

PYTHON = '/home/ubuntu/miniconda3/bin/python'
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


def coverage(cfg, *, episodes_per_update=16, episode_sim_s=1200):
    """Assert every clean4 phase covers its full u1-u24 schedule from the native manifests; refuse otherwise."""
    from tools.coopmwm.clean4_restart import checked
    phases = checked(cfg)['phases']
    r = cfg['clean4_restart']
    q, lessons = r['diagnostic_quotes'], r['lesson_quotes']
    practice = int(round(episodes_per_update*cfg['practice_weighting']['fraction']))
    per_update = 3*episodes_per_update*episode_sim_s+practice*episode_sim_s   # B practice A-references (R zero)
    need = dict(v2v_train=UPDATES*per_update, v2v_lessons=sum(lessons.values()),
        early_probes=q['probe_u002']+q['probe_u004'],
        isolated_replicated=sum(q[f'replicated_u{u:03}'] for u in (6, 12, 24)),
        v2v_final=1_020_960, gate_bank_verification_ext=r['gate_sim_s'],
        track2_smoke=cfg['clean2_smoke_reporting_quote']['complete_smoke_sim_s'],
)
    if checked(cfg).get('g4_cap_sim_s') != 230400:
        raise ValueError('G4 phase must equal its registered 230,400 hard cap')
    short = {k: (v, phases.get(k, 0)) for k, v in need.items() if v > phases.get(k, 0)}
    if short:
        raise ValueError('phase does not cover its u1-u24 schedule: '+json.dumps(short, sort_keys=True))
    if r['gate_sim_s']+need['track2_smoke'] > 40000:
        raise ValueError('smoke + gate exceed newmap2_smoke 40,000')
    return dict(need=need, allocated={k: phases[k] for k in need})


class Chain:
    def __init__(self, source, control):
        self.source, self.control = Path(source), Path(control)
        self.manifest = read(self.source/'SOURCE_MANIFEST.json')
        self.inputs = read(self.control/'PREPARE_INPUTS.json')
        from tools.coopmwm.clean5a_restart import ROOT
        self.root = Path(self.inputs.get('root', ROOT))
        self.serial = 0

    def status(self, step, state, detail=None):
        self.serial += 1
        value = dict(schema='coopmwm.clean5a.chain.v1', step=step, status=state, sequence=self.serial, detail=detail,
                     unix=time.time(), timestamp=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                     source_commit=self.manifest['commit'], root=str(self.root))
        publish(self.control/f'STATUS_{self.serial:03}_{step}_{state}.json', value)
        publish(self.control/'CHAIN_STATUS.json', value, replace=True)
        if self.root.is_dir():
            publish(self.root/'CHAIN_STATUS.json', value, replace=True)
            if state == 'FAILED' and not (self.root/'STOP.json').exists():
                publish(self.root/'STOP.json', dict(reason='clean5a detached chain failed; no retry', **value))

    def qualified(self):
        commit = self.manifest['commit']
        result = read(self.control.parent/('clean5a_tests_'+commit[:8])/'RESULT.json')
        if result['returncode'] != 0 or result['source_commit'] != commit:
            raise ValueError('full Linux qualification must pass for exact source')
        for name, sha in self.manifest['files'].items():
            if hashlib.sha256((self.source/name).read_bytes()).hexdigest() != sha:
                raise ValueError('qualified source SHA changed: '+name)
        if self.root.exists():
            raise ValueError('clean5a root already exists; inspect instead of replaying')
        if os.environ.get('COOPMWM_G5_RUNTIME') != '1' or not os.environ.get('COOPMWM_MAP_SPEC'):
            raise ValueError('clean5a chain requires COOPMWM_G5_RUNTIME=1 and COOPMWM_MAP_SPEC')
        if os.environ.get('COOPMWM_FROZEN_TRUNK'):
            raise ValueError('clean5a base is u0 d9042cc5: no trunk shim binding allowed')
        return dict(qualification=result, source_files=len(self.manifest['files']))

    def potential(self):
        from tools.coopmwm import clean4_potential as pg
        receipt = read(self.control/'POTENTIAL_GATE.json')
        pg.require_pass(receipt)
        return dict(decision=receipt['decision'], checks=receipt['checks'], receipt_sha256=receipt['receipt_sha256'])

    def budget(self):
        from tools.coopmwm import clean4_budget as b
        quote, approval = read(self.control/'CLEAN5A_QUOTE.json'), read(self.control/'BUDGET_CLEAN4.json')
        return b.approved(approval, quote)

    def prepare(self):
        from tools.coopmwm import clean5a_restart as c5
        i = self.inputs
        c5.prepare(self.source, contexts=read(i['contexts_path']), guidance_doc=read(i['guidance_path']),
                   quote_doc=read(self.control/'CLEAN5A_QUOTE.json'), approval=read(self.control/'BUDGET_CLEAN4.json'),
                   prior_runtime_sim_s=i['prior_runtime_sim_s'], host=i['host'], provenance=i['provenance'],
                   root=self.root)
        (self.root/'chain').symlink_to(self.control, target_is_directory=True)
        with (self.root/'AUTHORITY.md').open('a', encoding='utf-8') as f:
            f.write(chr(10)+'clean5a chain: qualification -> budget binding (CLEAN5A_QUOTE '
                    +read(self.control/'CLEAN5A_QUOTE.json')['receipt_sha256']
                    +') -> prepare -> smoke (all components, hard-coded receipts) -> one 0-collision/0-teleport gate -> '
                    'u1-u24 coverage -> A(frozen)/B/R u0 x24 + monitor. Any failure stops; no automatic retry.'+chr(10))
        return read(self.root/'PREPARED.json')

    def command(self, action):
        from tools.coopmwm import launch_training as t
        cfg = t.checked_config(self.root/'CONFIG.json')
        t.health(self.root)
        if cfg.get('gate_attempt') != 1 or cfg.get('smoke_attempt') != 1:
            raise ValueError('chain permits only the first smoke/gate attempt')
        prefix = action.upper()
        if any((self.root/(prefix+'_'+s+'.json')).exists() for s in ('STARTED', 'PASS', 'FAILED')):
            raise ValueError('one-use qualification; never retry or adopt attempted physics')
        with (self.control/(action+'.log')).open('x') as log:
            process = subprocess.run(['nice', '-n', '19', PYTHON, '-B', '-u', '-m', 'tools.coopmwm.launch_qualification',
                action, '--config', str(self.root/'CONFIG.json')], cwd='/home/ubuntu', stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
        publish(self.control/(prefix+'_EXIT.json'), dict(returncode=process.returncode, unix=time.time()))
        if process.returncode:
            raise RuntimeError(action+' failed; inspect log; no automatic retry')
        return self.check_receipts(action, cfg)

    def check_receipts(self, action, cfg):
        from tools.coopmwm import launch_training as t, clean4_budget as b
        from tools.coopmwm.exact_grade import verify_sealed
        from tools.coopmwm.protocol import digest
        from tools.coopmwm.storage import ledger
        root = self.root
        def receipt(name, status):
            value = t.read(root/name); verify_sealed(value, 'receipt_sha256')
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
        spent = b.smoke_gate_spent(ledger(root))
        if action == 'gate':
            gate = receipt('GATE_PASS.json', 'PASS')
            if (gate['smoke_sha256'] != smoke['receipt_sha256'] or
                    gate['source_manifest_sha256'] != cfg['source_manifest_sha256'] or
                    gate['safety']['collisions'] != 0 or gate['safety']['teleports'] != 0):
                raise ValueError('gate final-source/smoke binding or zero-event rule failed')
            t.release(cfg)
            return dict(gate_sha256=gate['receipt_sha256'], smoke_sha256=smoke['receipt_sha256'],
                        safety=gate['safety'], newmap2_smoke=spent)
        return dict(smoke_sha256=smoke['receipt_sha256'], fixes_sha256=fixes['receipt_sha256'],
                    levers=levers['levers'], newmap2_smoke=spent)

    def assert_coverage(self):
        from tools.coopmwm import launch_training as t
        cfg = t.checked_config(self.root/'CONFIG.json')
        result = coverage(cfg)
        publish(self.root/'PHASE_COVERAGE_PASS.json', result)
        return result

    def launch(self):
        from tools.coopmwm import launch_training as t
        from tools.coopmwm.protocol import digest
        root = self.root
        cfg = t.checked_config(root/'CONFIG.json')
        self.check_receipts('gate', cfg); t.health(root)
        if not (root/'PHASE_COVERAGE_PASS.json').is_file():
            raise ValueError('u1-u24 phase coverage must pass before launch')
        if any((root/name).exists() for name in ('LAUNCHED.json', 'LAUNCH_DISPATCH_STARTED.json', 'owners')):
            raise ValueError('existing training dispatch; never duplicate')
        publish(root/'LAUNCH_DISPATCH_STARTED.json', dict(config_sha256=digest(cfg), unix=time.time()))
        logs = root/'launch_logs'; logs.mkdir(); owners = {}
        try:
            for role in ('A', 'B', 'R', 'monitor'):
                command = ['nice', '-n', '19', PYTHON, '-B', '-u', '-m', 'tools.coopmwm.launch_training',
                           'monitor' if role == 'monitor' else 'arm', '--config', str(root/'CONFIG.json')]
                if role != 'monitor':
                    command += ['--arm', role]
                with (logs/(role+'.log')).open('x') as log:
                    p = subprocess.Popen(command, cwd='/home/ubuntu', env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                owners[role] = dict(pid=p.pid, start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21])
                publish(root/('DISPATCH_'+role+'.json'), owners[role])
            time.sleep(3)
            for role, owner in owners.items():
                fields = Path(f"/proc/{owner['pid']}/stat").read_text().split()
                if fields[2] == 'Z' or fields[21] != owner['start_ticks']:
                    raise ValueError('training owner exited at launch: '+role)
            t.health(root)
        except BaseException:
            publish(root/'PARTIAL_LAUNCH.json', dict(owners=owners, unix=time.time(), reason='never automatic retry'))
            raise
        value = dict(schema='coopmwm.clean5a.launched.v1', timestamp=datetime.now(timezone.utc).isoformat(),
            unix=time.time(), root=str(root), source_root=str(self.source), runtime_commit=self.manifest['commit'],
            runtime_sha256=digest(cfg), config_file_sha256=hashlib.sha256((root/'CONFIG.json').read_bytes()).hexdigest(),
            source_manifest_sha256=cfg['source_manifest_sha256'], owners=owners, updates=UPDATES,
            start_updates=dict(A=0, B=0, R=0), smoke_sha256=t.read(root/'SMOKE_PASS.json')['receipt_sha256'],
            gate_sha256=t.read(root/'GATE_PASS.json')['receipt_sha256'], newmap=cfg['newmap'],
            guidance=cfg['clean4_restart']['guidance'], remaining_cap_sim_s=cfg['clean4_budget']['remaining_sim_s'],
            approved_sim_s=cfg['clean4_budget']['approved_sim_s'], safety_rule=cfg['clean4_restart']['inherited_safety'])
        publish(root/'LAUNCHED.json', value)
        return value

    def monitor(self):
        from tools.coopmwm import launch_training as t
        from tools.coopmwm.exact_grade import verify_sealed
        from tools.coopmwm.protocol import digest
        from tools.coopmwm.storage import ledger
        root = self.root
        cfg = t.checked_config(root/'CONFIG.json')
        launched = read(root/'LAUNCHED.json'); seen = set()
        while True:
            t.health(root)
            for u in range(1, UPDATES+1):
                path = root/f'LEARNING_MONITOR_u{u:03}.json'
                if u in seen or not path.is_file():
                    continue
                row = read(path); verify_sealed(row, 'receipt_sha256')
                if row['config_sha256'] != digest(cfg):
                    raise ValueError('monitor config binding changed')
                diagnostic = None
                if u in (2, 4, 6, 12, 24):
                    kind = 'probe' if u in (2, 4) else 'replicated'
                    p = root/f'track2_diagnostics/{kind}_u{u:03}_live.json'
                    if not p.is_file():
                        continue
                    diagnostic = read(p); verify_sealed(diagnostic, 'receipt_sha256')
                safety = {k: sum(a['metrics']['safety'][k] for a in row['arms']) for k in ('collisions', 'teleports')}
                if any(safety.values()):
                    raise ValueError('absolute training safety event: '+json.dumps(safety))
                spent = sum(r['sim_s'] for r in ledger(root))
                if spent > cfg['clean4_budget']['remaining_sim_s']:
                    raise ValueError('clean-training envelope exceeded')
                lines = [datetime.now(timezone.utc).isoformat()+f' CLEAN5A update {u}; all arms sealed.',
                    'CLEAN5A Learning '+json.dumps({a['arm']: dict(action=a['action'], diagnose=a['diagnose'],
                        bank_ce=a['metrics'].get('bank_move_ce'), layer_kl=a['metrics'].get('layer_kl')) for a in row['arms']}, sort_keys=True),
                    'CLEAN5A Safety '+json.dumps(safety, sort_keys=True)+'; A full rule; B/R composite report-only; any collision/teleport pauses.',
                    ('CLEAN5A Check '+json.dumps({k: diagnostic[k] for k in ('metric_reports', 'verdict') if k in diagnostic}, sort_keys=True)
                     if diagnostic else 'CLEAN5A Check not scheduled at this update.'),
                    f'CLEAN5A Budget root={spent}/{cfg["clean4_budget"]["remaining_sim_s"]} sim-s; u24 criterion unchanged.']
                try:
                    from tools.coopmwm import clean4_diag
                    lines.append(clean4_diag.status_line(clean4_diag.diagnose_update(root, u, tapes=True)))
                except Exception as error:   # report-only diagnostics never stop training
                    lines.append(f'CLEAN5A diag u{u} unavailable: {error!r}')
                with (root/'CLEAN5A_STATUS.md').open('a', encoding='utf-8') as f:
                    f.write('\n'.join(lines)+'\n\n'); f.flush(); os.fsync(f.fileno())
                publish(self.control/f'MONITORED_u{u:03}.json', dict(update=u, unix=time.time(),
                        monitor_sha256=row['receipt_sha256'], spent_sim_s=spent))
                seen.add(u)
            complete = []
            for role, owner in launched['owners'].items():
                done = (root/('MONITOR_COMPLETE.json' if role == 'monitor' else role+'_COMPLETE.json')).is_file()
                complete.append(done)
                if done:
                    continue
                p = Path(f"/proc/{owner['pid']}/stat")
                fields = p.read_text().split() if p.exists() else []
                if not fields or fields[2] == 'Z' or fields[21] != owner['start_ticks']:
                    raise ValueError('owner exited before completion: '+role)
            if all(complete):
                return dict(updates=UPDATES, monitored_updates=sorted(seen))
            time.sleep(30)

    def run(self):
        sequence([('QUALIFIED', self.qualified), ('BUDGET', self.budget),
                  ('PREPARE', self.prepare), ('SMOKE', lambda: self.command('smoke')),
                  ('GATE', lambda: self.command('gate')), ('PHASE_COVERAGE', self.assert_coverage),
                  ('LAUNCH', self.launch)], self.status)
        sequence([('MONITOR', self.monitor)], self.status)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--control', required=True)
    args = p.parse_args()
    import fcntl
    os.environ['COOPMWM_G5_RUNTIME'] = '1'
    control = Path(args.control)
    with (control/'OWNER.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish(control/'CHAIN_STARTED.json', dict(pid=os.getpid(), unix=time.time(),
                start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
        Chain(args.source, control).run()


if __name__ == '__main__':
    main()
