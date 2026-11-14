"""Immutable receipts and append-only conservative accounting; no replay/refunds."""
import json
import os
from .protocol import checked, canonical, CAPS, CAP


def write(path, value):
    path = checked(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(canonical(value)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def read(path):
    return json.loads(checked(path).read_text(encoding='utf-8'))


def ledger(root):
    path = checked(root)/'CHARGES.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def reserve(root, jobs, phase):
    # Only the exclusive supervisor writes; reserve the WHOLE wave before dispatch.
    old = ledger(root)
    ids = [j['id'] for j in jobs]
    if len(set(ids)) != len(ids) or set(ids) & {r['id'] for r in old}:
        raise ValueError('duplicate charge/attempt; inspect completed state')
    amount = sum(j.get('sim_s',1200) for j in jobs)
    if sum(r['sim_s'] for r in old)+amount > CAP or sum(r['sim_s'] for r in old if r['phase'] == phase)+amount > CAPS[phase]:
        raise ValueError('whole-wave budget blocked')
    with (root/'CHARGES.jsonl').open('a',encoding='utf-8') as stream:
        for job in jobs:
            stream.write(canonical(dict(id=job['id'],phase=phase,sim_s=job.get('sim_s',1200)))+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def reconciled(root):
    outstanding = [r['id'] for r in ledger(root) if not (root/'jobs'/r['id']/'RESULT.json').exists()]
    if outstanding:
        raise ValueError('unfinished charges block resume: '+str(outstanding))
    return sum(r['sim_s'] for r in ledger(root))
