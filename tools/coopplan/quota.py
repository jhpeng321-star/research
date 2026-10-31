"""Atomic prospective reservations: all live pairs and complete solo twins get priority."""
import json
import os
import time

from .storage import read
from .protocol import canonical
from .planner import HORIZON, CADENCE


def credit_key(ident):
    # A fixed two-solo pool is reserved before we know which group action will win.
    return ident.split('_t')[0]+'_t'+ident.split('_t')[1].split('_')[0]+'_CG'


def outstanding(root, charges=None):
    pending = dict(read(root/'ALLOCATION.json').get('reserved',{}))
    path = root/'RESERVATIONS.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            event = json.loads(line)
            for key,value in event.get('reserve',{}).items():
                if key in pending:
                    raise ValueError('duplicate reservation '+key)
                pending[key] = value
            for key in event.get('release',[]):
                pending[key] = 0
    if charges is None:
        ledger = root/'CHARGES.jsonl'
        charges = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    for c in charges:
        key = c.get('reserve_key',c['id'])
        # Released pools have no charges. Consumed pools stay present at zero.
        pending[key] = pending.get(key,0)-c['sim_s']
    if any(v < 0 for v in pending.values()):
        raise ValueError('unreserved or overconsumed physics')
    return pending


def append(root,event):
    with (root/'RESERVATIONS.jsonl').open('a',encoding='utf-8') as f:
        f.write(canonical(dict(event,unix=time.time()))+'\n');f.flush();os.fsync(f.fileno())


def reserve_actions(root,state,actions):
    """Prune only for affordability, with a fixed outcome-independent cyclic priority."""
    from .storage import verify,ledger_lock
    binding = verify(root)
    assert actions[0]['id'] == 'O0'
    if len(actions) == 1:
        return [],dict(reason='only_MAPPO_available',pruned=[])
    with ledger_lock(root):
        ledger = root/'CHARGES.jsonl'
        charges = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
        allocation = read(root/'ALLOCATION.json')
        pending = outstanding(root,charges)
        remaining = allocation['cap']-sum(c['sim_s'] for c in charges)-sum(pending.values())
        if allocation.get('world_caps'):
            prefix = state.replay_key.split('_t')[0]+'_'
            world_left = allocation['world_caps'][prefix[:-1]]-sum(c['sim_s'] for c in charges if c['id'].startswith(prefix))-sum(v for k,v in pending.items() if k.startswith(prefix))
            remaining = min(remaining,world_left)
        alternatives = actions[1:]
        rotation = ((state.time_s-300)//CADENCE) % len(alternatives)
        priority = alternatives[rotation:]+alternatives[:rotation]
        cost = state.time_s+binding.get('horizon_s',HORIZON)
        # O0 + n candidates + BOTH solos (regardless of predicted winner).
        n = min(len(priority),max(0,remaining//cost-3))
        admitted = [actions[0]]+priority[:n] if n else []
        if admitted:
            reserves = {state.replay_key+'_'+a['id']:cost for a in admitted}
            reserves[state.replay_key+'_CG'] = 2*cost
            if any(k in outstanding(root,charges) for k in reserves):
                raise ValueError('decision was already reserved')
            append(root,dict(decision=state.replay_key,reserve=reserves))
        return admitted,dict(reason='full' if n == len(priority) else 'budget_pruned',
                             available_sim_s=remaining,pruned=[a['id'] for a in priority[n:]])


def release_credit(root,key):
    from .storage import ledger_lock
    with ledger_lock(root):
        pending = outstanding(root)
        if pending.get(key,0):
            append(root,dict(release=[key]))
