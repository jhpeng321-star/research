"""Forced-command SSH RPC restricted to one owned warm inference spool."""
import json
from pathlib import Path
import sys
from .warm_spool import key_of,submit


GATES=('STARTUP_WARM_BATCH_SINGLE_K4_H120.json','STARTUP_WARM_BATCH_SINGLE_K8_H300.json',
    'STARTUP_PARITY_K4_H120.json','STARTUP_PARITY_K8_H300.json',
    'STARTUP_WARM_PARITY_K4_H120.json','STARTUP_WARM_PARITY_K8_H300.json')
# Concurrent-plan background gates: exposed for hot-add admission (hotadd.GATES
# requires them; a route on this module alone stalled the 14:24 gpuA hot-add on
# 'missing/nonexact new-worker gate STARTUP_COMMITTED_K4_H300.json'). They do not
# change this module's 'ready' predicate.
EXTRA_GATES=('STARTUP_COMMITTED_K4_H300.json','STARTUP_COMMITTED_K8_H420.json')


def status(root):
    root=Path(root);receipts={}
    for name in (*GATES,*EXTRA_GATES,'SERVICE_OWNER.json','SPOOL_OWNER.json','SPOOL_FAILED.json'):
        p=root/name
        if p.exists():receipts[name]=json.loads(p.read_text())
    d=receipts.get('SPOOL_OWNER.json');state=None
    if d:
        p=Path('/proc')/str(d['pid']);s=(p/'stat').read_text().split() if (p/'stat').exists() else None
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=d['boot_id']:
            raise ValueError('GPU boot identity changed')
        if s and s[21]!=d['start_ticks']:raise ValueError('GPU worker PID identity changed')
        state=s[2] if s else 'exited'
    ready=all(receipts.get(n,{}).get('pass_') and receipts[n]['max_cost_error']==0. for n in GATES)
    return dict(receipts=receipts,state=state,ready=ready and 'SERVICE_OWNER.json' in receipts,
        alive=state in ('S','R','D'),failed='SPOOL_FAILED.json' in receipts)


def handle(root,value):
    root=Path(root);assert root.is_dir() and (root/'GPU_BINDING.json').is_file()
    action=value.get('action')
    if action=='status' and set(value)=={'action'}:return status(root)
    if action=='submit' and set(value)=={'action','record'}:
        s=status(root)
        if not s['alive'] or s['failed'] or not s['ready']:raise ValueError('unqualified or dead GPU spool')
        return dict(key=submit(root,value['record']),owner=s['receipts']['SPOOL_OWNER.json'])
    if action=='forecast' and set(value)=={'action','key'}:
        key=key_of(dict(state_id=value['key']));p=root/'forecasts'/(key+'.json')
        result=json.loads(p.read_text()) if p.exists() else None
        if result is not None and result['key']!=key:raise ValueError('forecast identity mismatch')
        return dict(forecast=result,status=status(root))
    raise ValueError('restricted warm RPC action/fields')


def main(root):
    # Authorized key fixes root and command; input cannot execute code or access
    # arbitrary paths. Private authentication material never leaves cpuQ.
    line=sys.stdin.buffer.read(32*1024*1024+1)
    if len(line)>32*1024*1024:raise ValueError('oversized warm RPC')
    print(json.dumps(handle(root,json.loads(line)),allow_nan=False),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
