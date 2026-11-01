"""Bounded retries of idempotent CPU-queue transport, never inference/physics.

An SSH process timeout alone is not proof of a connection failure. Only explicit
SSH connection diagnostics qualify; executor errors, tracebacks and parity errors
remain fatal. After reconnect, owner identities and immutable job receipts must
be reverified before replaying a metadata read or identical response write.
"""
import json
import re
import time


def connection_error(error):
    if not isinstance(error,RuntimeError):return False
    text=str(error).strip()
    if any(word in text.lower() for word in ('traceback','assertionerror','parity','executor','outofmemory')):
        return False
    patterns=(r'Connection (?:closed|reset) by [^\r\n]+ port \d+',
        r'kex_exchange_identification: (?:read: Connection reset by peer|Connection closed by remote host)',
        r'ssh: connect to host \S+ port \d+: (?:Connection timed out|Connection refused|Connection reset by peer)',
        r'client_loop: send disconnect: (?:Broken pipe|Connection reset)',
        r'Connection to \S+ (?:closed by remote host\.|closed\.)')
    return bool(text) and all(any(re.fullmatch(p,line) for p in patterns)
        for line in text.splitlines() if line.strip())


def validate_snapshot(before,after):
    if before['boot_id']!=after['boot_id']:raise ValueError('cpuQ reboot; inspect jobs before transport recovery')
    for name,h in before['receipts'].items():
        if after['receipts'].get(name)!=h:raise ValueError('immutable queue receipt changed: '+name)
    for pid,identity in before['live_processes'].items():
        current=after['live_processes'].get(pid)
        if current is not None and current!=identity:raise ValueError('owned PID identity changed: '+pid)
    return after


def retry(operation,verify,record,*,delays=(15,30,60,120),sleep=time.sleep,clock=time.monotonic):
    """At most five attempts /300s; failed reverification never replays operation."""
    deadline=clock()+300
    reconnect=False
    for attempt in range(len(delays)+1):
        try:
            if reconnect:verify()
            if clock()>=deadline:raise TimeoutError('bounded SSH recovery deadline; inspect receipts')
            return operation()
        except RuntimeError as error:
            if not connection_error(error) or attempt==len(delays):raise
            delay=delays[attempt]
            if clock()+delay>=deadline:raise
            record(dict(attempt=attempt+1,error=str(error),backoff_s=delay,new_sumo_sim_s=0))
            sleep(delay);reconnect=True
    raise AssertionError('unreachable')


def snapshot(remote,root):
    """Read only queue metadata, owner /proc identities and allowed job receipts."""
    return json.loads(remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({root!r});assert r.is_dir()
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
receipts={{}};live={{}}
def receipt(p):
 if p.exists():receipts[str(p.relative_to(r))]=hashlib.sha256(p.read_bytes()).hexdigest()
for arm in ('A','B','R'):
 p=r/f'WARM_{{arm}}_DISPATCH.json'
 if not p.exists():continue
 receipt(p);d=json.loads(p.read_text());assert d['boot_id']==boot
 proc=pathlib.Path('/proc')/str(d['pid'])
 if (proc/'stat').exists():
  s=(proc/'stat').read_text().split();assert s[21]==d['start_ticks']
  live[str(d['pid'])]=dict(start_ticks=s[21],cmd_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())
charges=[json.loads(x) for x in (r/'CHARGES.jsonl').read_text().splitlines()]
for c in charges:
 ident=c['id'];assert '/' not in ident and '..' not in ident
 receipts['charge:'+ident]=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
 p=r/'jobs'/ident
 for name in ('CLAIM.json','CLOSE.json'):receipt(p/name)
 if (p/'CLAIM.json').exists() and not (p/'CLOSE.json').exists():
  d=json.loads((p/'CLAIM.json').read_text());proc=pathlib.Path('/proc')/str(d['pid'])
  if (proc/'stat').exists():
   s=(proc/'stat').read_text().split()
   live[str(d['pid'])]=dict(start_ticks=s[21],cmd_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())
for a in ('A','B','R'):
 for p in r.glob(a+'_UPDATE_*.json'):receipt(p)
print(json.dumps(dict(boot_id=boot,receipts=receipts,live_processes=live,
 charged_jobs=len(charges),closed_jobs=sum((r/'jobs'/c['id']/'CLOSE.json').exists() for c in charges))))''',timeout=30))


class QueueTransport:
    def __init__(self,remote,root,record):
        self.remote=remote;self.root=root;self.record=record
        self.baseline=retry(lambda:snapshot(remote,root),lambda:None,record)

    def verify(self):
        current=snapshot(self.remote,self.root)
        self.baseline=validate_snapshot(self.baseline,current)
        self.record(dict(reverified=True,snapshot=current,new_sumo_sim_s=0))

    def call(self,code,timeout=30):
        return retry(lambda:self.remote('cpuQ',code,timeout=timeout),self.verify,self.record)


def qualify():
    """CPU fault injection + real receipt-only reconnect verification; no dispatch."""
    import io
    import unittest
    import uuid
    from . import ops
    from .storage import write
    from .protocol import checked,sha,REGISTRY
    from .warm_deploy import ROOT
    from .warm_protocol import budget
    from .resume_gate import RESUME_CAP
    target=ops.EVIDENCE/'WARM_TRANSPORT_REGISTRATION.json'
    assert not target.exists(),'existing registration must be adopted, never duplicated'
    start=time.monotonic();stream=io.StringIO()
    tests=unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.test_warm_transport_cpu')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(tests)
    cpu=dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),
        wall_s=time.monotonic()-start,log=stream.getvalue(),new_sumo_sim_s=0)
    assert cpu['pass_'] and not cpu['skips']
    events=[];transport=QueueTransport(ops.ssh.remote,ROOT,events.append)
    transport.verify()
    value=dict(cpu=cpu,snapshot=transport.baseline,verification_events=events,
        helpers={name:sha(ops.LOCAL/'tools/coopmwm'/name) for name in
            ('warm_transport.py','test_warm_transport_cpu.py','resume_gate.py','warm_bridge.py','warm_deploy.py','warm_release_ops.py')},
        explicit_cap_sim_s=RESUME_CAP,registered_plan=budget(),new_sumo_sim_s=0,
        original_stop_preserved=True,physics_dispatched=False,
        contract='5attempts/300s,backoff15/30/60/120; CPUqueue metadata and identical response persistence only; '
            'reverify boot/PID/start/command and charge/claim/close/update receipts before replay; '
            'no physics/executor/parity/service-call retry; existing STOP remains durable')
    evidence=ops.EVIDENCE/('WARM_TRANSPORT_QUAL_'+uuid.uuid4().hex+'.json')
    write(evidence,value);h=sha(evidence)
    line=('\nS-COOPMWM MAIN0129 TRANSPORT ENGINEERING REGISTERED evidenceSHA'+h+
        ': '+value['contract']+'; '+str(cpu['tests'])+'CPU PASS/0skips/zeroSUMO; no new dispatch. '
        'Explicit resume6000000 conflicts with existing6473440 allocation; dispatch budget guard blocks until owner cap/scope resolution.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(target,dict(line=line,evidence_path=str(evidence),evidence_sha256=h,new_sumo_sim_s=0))
    print(json.dumps(dict(cpu=cpu,snapshot_receipts=len(transport.baseline['receipts']),
        live_processes=transport.baseline['live_processes'],registered=True,physics_dispatched=False)))


if __name__=='__main__':qualify()
