"""MAIN05:08 live transport handover; never repeat a model claim or SUMO job.

Existing clients receive a prompt capacity-declined response when all GPU slots
are occupied. Assigned searches are always adopted, including unfinished ones.
The Bernoulli query probability in legacy behaviour tapes remains historical;
query_admission explicitly records the subsequent deterministic capacity gate.
"""
import argparse
import json
import os
from pathlib import Path
import threading
import time
from .warm_broker import Remote,worker_endpoints
from .warm_spool import immutable
from .warm_health import assert_running,HealthPause
from .protocol import digest


class WorkerUnavailable(RuntimeError):
    """A dead worker needs an owned replacement, not a global training STOP."""


def check_worker(root,endpoint,status,key=None):
    if status['alive'] and not status['failed']:return
    failure=status.get('receipts',{}).get('SPOOL_FAILED.json',{}).get('error','')
    if any(word in failure.lower() for word in ('parity','immutable','accounting','physics','safety','budget')):
        raise ValueError('fatal worker invariant: '+failure)
    # Preserve a failed in-flight claim. A replacement must never recompute it.
    value=dict(slot=endpoint['slot'],key=key,status=status,unix=time.time(),
        action='replace only this verified dead worker; qualify before admission; never replay inference claim',
        new_sumo_sim_s=0)
    target=root/'worker_restart_requests';target.mkdir(exist_ok=True)
    immutable(target/(digest(value)+'.json'),value)
    raise WorkerUnavailable(endpoint['slot'])


def resource_endpoints(root,endpoints,config):
    """Admit a migrated slot only after its fresh-service READY receipt."""
    if not config.get('resource_ready_only'):return endpoints
    result={}
    for slot,e in endpoints.items():
        p=Path(root)/('GPU_RESOURCE_READY_'+slot+'_1651.json')
        if not p.exists():continue
        v=json.loads(p.read_text())
        assert v['amendment_sha256'] in (Path(root)/'AUTHORITY.md').read_text()
        assert v['slot']==slot and v['owner']['pid']==v['service']['pid']
        result[slot]=e
    return result


def settle_unavailable(root,slot,items):
    """A failed model claim is terminally declined, never recomputed or scored."""
    settled=[]
    for kind,record in items:
        if record.get('mode')=='diagnostic':continue
        key=record['state_id'];assignment=Path(root)/'broker_assignments'/(key+'.json')
        target=Path(root)/('centre_responses' if kind=='centre' else 'responses')/(key+'.json')
        if target.exists() or not assignment.exists():continue
        if json.loads(assignment.read_text())['slot']!=slot:continue
        unavailable=dict(slot=slot,key=key,inference_retry=False)
        if kind=='centre':
            # HARDEN-3: an R centre claimed by a dead slot was never re-assignable
            # (assignments are slot-sticky), so its R episode waited to the deadline.
            from .query_errors import declined
            immutable(target,declined('centre',record,'assigned worker unavailable; inference never replayed',
                worker_unavailable=unavailable))
        else:
            immutable(target,dict(accepted=False,reason='assigned worker unavailable; policy continues; inference never replayed',
                record_sha256=digest(record),compute_counts_unknown=True,worker_unavailable=unavailable))
        settled.append(key)
    return settled


def pending(root):
    from .exploration import regime
    items=[]
    from .resource_config import retired
    excluded=set(retired(root)['query_keys'])
    for req,resp,kind in [('requests','responses','plan'),('centre_requests','centre_responses','centre')]:
        for p in (root/req).glob('*.json'):
            if p.stem in excluded:continue
            if (root/resp/p.name).exists():continue
            try:record=json.loads(p.read_text())
            except json.JSONDecodeError:
                # Historical writers create then fill a request (not atomic).
                # In-progress metadata can be observed but is never a forecast.
                if time.time()-p.stat().st_mtime<30:continue
                raise
            items.append((kind,record))
    order=batch_order(root)
    # HARDEN-3: within the same (dense-first, sim-time) priority, keep each
    # episode's parallel batch contiguous and in its inherited group order.
    # The previous hash (state_id) tie-break interleaved the 4-6 group queries
    # of ~10 episodes that all reach t%10==0 at the same sim time, so no
    # episode received its first-in-order answer until most of the whole
    # queue was searched (all SUMO episodes idle meanwhile).
    items.sort(key=lambda v:(v[0]=='centre',v[1].get('mode')=='diagnostic',
        regime(v[1]['rows'])!='merge-dense',v[1]['time_s'])+order.get(v[1]['state_id'],(float('inf'),0))+(v[1]['state_id'],))
    return items


_BATCH_CACHE={}


def batch_order(root):
    """state_id -> (reservation unix, inherited index) from immutable quota receipts."""
    out={}
    for p in Path(root).glob('quota_requests/u*/*_REQUEST.json'):
        key=str(p)
        if key not in _BATCH_CACHE:
            try:
                value=json.loads(p.read_text())
                reserve=json.loads(p.with_name(p.name.replace('_REQUEST','_RESERVE')).read_text())
            except (OSError,json.JSONDecodeError):continue
            _BATCH_CACHE[key]={r['state_id']:(reserve.get('unix',0.),i)
                for i,r in enumerate(value.get('records',[value]))}
        out.update(_BATCH_CACHE[key])
    return out


def withdrawn(root):
    folder=Path(root)/'request_withdrawals'
    return {p.stem for p in folder.glob('*.json')} if folder.is_dir() else set()


def capacity_declines(items,assigned,slots,blocking=(),drain=False,withdrawn=()):
    """Keep enough unassigned requests for free slots; preserve ALL old claims."""
    busy={assigned[r['state_id']]['slot'] for _,r in items if r['state_id'] in assigned}
    free=max(0,len(set(slots)-busy));declines=[]
    for kind,r in items:
        if r['state_id'] in assigned:continue
        # Diagnostic/centre requests wait, never become fabricated model outputs.
        if kind!='plan' or r.get('mode')=='diagnostic':continue
        # HARDEN-3: the requesting decision is already resolved (an earlier
        # inherited-order group was accepted); never spend a slot on it.
        if drain or r['state_id'] in withdrawn:
            declines.append(r);continue
        # MAIN05:48 quota reservations wait for real inference even while all
        # slots are occupied; they are never fabricated capacity declines.
        if r['state_id'] in blocking:
            if free:free-=1
            continue
        if free:free-=1
        else:declines.append(r)
    return declines,busy


def decline(root,record,slots,busy,amendment,drain=False,withdrawn=False):
    key=record['state_id']
    admission=dict(stage='live broker boundary',available=False,
        wm_admission_probability=0.,capacity_predicate=('registered old-wave boundary drain; existing assigned searches finish once' if drain else
            'client withdrew: decision already resolved by an earlier inherited-order group' if withdrawn else 'all registered slots occupied or reserved by earlier requests'),
        slots=sorted(slots),busy_slots=sorted(busy),amendment_sha256=amendment,
        legacy_query_probability_unchanged=True,unix=time.time())
    result=dict(accepted=False,reason='boundary drain; policy continues' if drain else
        'withdrawn by client; decision already resolved' if withdrawn else 'capacity unavailable; policy continues',
        record_sha256=digest(record),query_admission=admission,
        candidates_searched=0,imagined_futures=0)
    immutable(root/'capacity_declines'/(key+'.json'),dict(key=key,**admission))
    immutable(root/'responses'/(key+'.json'),result)


def new_item_order(items,last_kind,enabled):
    """Bound centre starvation at search boundaries; assigned claims stay first.

    At most one centre helper between plan searches when both queues are nonempty.
    Stable sort keeps the existing dense/time/state priority within either queue.
    Disabled is the original ordering, without changing a request or model call.
    """
    if not enabled:return items
    prefer_centre=last_kind=='plan'
    return sorted(items,key=lambda v:(v[0]=='centre')!=prefer_centre)


def drain_active(root,config):
    """One registered handover boundary; never abandon an assigned search.

    A tail handover may close NEW old-wave admissions only after its executed
    quota is met. The marker automatically restores ordinary admission before
    the replacement supervisor starts any new wave.
    """
    if config.get('drain_until_handover') and not (root/'MAIN0548_HANDOVER_COMPLETE.json').exists():return True
    marker=config.get('boundary_drain_until')
    if marker and not (root/marker).exists():
        from .quota_admission import state
        return state(root,config['boundary_drain_update'])['executed']>=config['boundary_drain_minimum']
    return False


def run(root,config_path):
    import fcntl
    import signal
    root=Path(root);config=json.loads(Path(config_path).read_text())
    binding=json.loads((root/'WARM_REMOTE_BINDING.json').read_text())
    from .resource_config import clocks
    config=clocks(root,config);binding=clocks(root,binding)
    assert config['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
    assert config['deadline_unix']==binding['deadline_unix']
    lock=(root/'WARM_REMOTE_BROKER.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    owner=dict(pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source=config['source_root'])
    tag=config.get('receipt_tag','')
    immutable(root/('CAPACITY_BROKER_OWNER'+tag+'.json'),owner)
    stop_path=root/('WARM_GUIDED_STOP_V5.json' if (root/'DNS_GUIDED_RELEASE_1651.json').exists() else
        'WARM_GUIDED_STOP_V4.json' if (root/'MATCH_GUIDED_RELEASE.json').exists() else
        'WARM_GUIDED_STOP_V3.json' if (root/'INCIDENT_GUIDED_RELEASE.json').exists() else
        'WARM_GUIDED_STOP_V2.json' if tag else 'WARM_GUIDED_STOP.json')
    assignments=root/'broker_assignments';assignments.mkdir(exist_ok=True)
    for name in ('broker_completed','capacity_declines','capacity_admissions'):(root/name).mkdir(exist_ok=True)
    stop=threading.Event();mutex=threading.Lock();inflight=set();errors=[];quarantined=set()

    def worker(endpoint):
        try:
            remote=Remote(root,endpoint)
            check_worker(root,endpoint,remote.call(dict(action='status')))
            remote.verify()
            last_kind='plan'
            while not stop.is_set():
                assert_running(root,'B')
                if time.time()>binding['deadline_unix']:raise ValueError('original GPU deadline')
                with mutex:
                    item=None
                    items=pending(root)
                    # An existing assignment MUST be adopted before new work on
                    # this slot, even when the new request is dense/preferred.
                    assigned_items=[v for v in items if (assignments/(v[1]['state_id']+'.json')).exists()]
                    new_items=[v for v in items if not (assignments/(v[1]['state_id']+'.json')).exists()]
                    if config.get('decline_withdrawn',True):
                        gone=withdrawn(root)
                        new_items=[v for v in new_items if v[0]!='plan' or v[1]['state_id'] not in gone]
                    if drain_active(root,config):
                        new_items=[v for v in new_items if v[0]!='plan' or v[1].get('mode')=='diagnostic']
                    new_items=new_item_order(new_items,last_kind,config.get('pipeline_centre_fairness',False))
                    for kind,record in assigned_items+new_items:
                        key=record['state_id'];p=assignments/(key+'.json')
                        previous=json.loads(p.read_text()) if p.exists() else None
                        if key not in inflight and (previous is None or previous['slot']==endpoint['slot']):
                            immutable(p,dict(slot=endpoint['slot'],kind=kind,key=key,record_sha256=digest(record)))
                            inflight.add(key);item=(kind,record,previous is not None);break
                if item is None:stop.wait(.1);continue
                kind,record,adopted=item;key=record['state_id'];begin=time.monotonic()
                admission=root/'capacity_admissions'/(key+'.json')
                if admission.exists():
                    old=json.loads(admission.read_text());assert old['key']==key and old['slot']==endpoint['slot']
                else:immutable(admission,dict(key=key,slot=endpoint['slot'],
                    adopted=adopted,capacity_admission_probability=1.,unix=time.time()))
                # Identical transport submission is idempotent in the durable
                # GPU spool. An existing inference claim is NEVER recomputed.
                remote.call(dict(action='submit',record=dict(record,mode='centre') if kind=='centre' else record))
                while True:
                    value=remote.call(dict(action='forecast',key=key))
                    if value['forecast'] is not None:break
                    check_worker(root,endpoint,value['status'],key)
                    assert_running(root,'B');stop.wait(3)
                payload=value['forecast']
                from .query_errors import validate_response,malformed,declined
                bad=malformed(kind,record,payload['result'])
                if bad:
                    # HARDEN-3: a malformed result is a per-query decline, never a STOP.
                    remote.event(dict(event='malformed_response_declined',key=key,kind=kind,reason=bad,global_stop=False))
                    payload=dict(payload,result=declined(kind,record,'malformed GPU response: '+bad))
                else:validate_response(record,payload['result'])
                if payload['result'].get('query_error'):
                    remote.event(dict(event='query_code_declined',key=key,
                        query_error=payload['result']['query_error'],global_stop=False))
                immutable(root/('centre_responses' if kind=='centre' else 'responses')/(key+'.json'),payload['result'])
                immutable(root/'broker_completed'/(key+'.json'),dict(slot=endpoint['slot'],kind=kind,key=key,
                    inference_wall_s=payload['wall_s'],delivery_wall_s=time.monotonic()-begin,unix=time.time()))
                last_kind=kind
                with mutex:inflight.remove(key)
        except WorkerUnavailable:
            # This slot is quarantined until an explicit owned replacement is
            # admitted. Other slots and policy/SUMO episodes remain independent.
            with mutex:
                quarantined.add(endpoint['slot'])
                for key in settle_unavailable(root,endpoint['slot'],pending(root)):inflight.discard(key)
            return
        except BaseException as error:
            errors.append(repr(error));stop.set()
            if not isinstance(error,HealthPause) and not stop_path.exists():
                immutable(stop_path,dict(error=repr(error),source='capacity broker',unix=time.time()))

    threads={};endpoints={};last_slots=None
    try:
        while not stop.is_set():
            endpoints=resource_endpoints(root,worker_endpoints(root,binding,endpoints),config)
            endpoints={s:e for s,e in endpoints.items() if s not in quarantined}
            for slot,e in endpoints.items():
                if slot not in threads:
                    threads[slot]=threading.Thread(target=worker,args=(e,));threads[slot].start()
            with mutex:
                items=pending(root)
                assigned={r['state_id']:json.loads((assignments/(r['state_id']+'.json')).read_text())
                    for _,r in items if (assignments/(r['state_id']+'.json')).exists()}
                blocking={p.stem for p in (root/'quota_admissions').glob('*.json')
                    if json.loads(p.read_text()).get('blocking')}
                drain=drain_active(root,config)
                gone=withdrawn(root) if config.get('decline_withdrawn',True) else set()
                declines,busy=capacity_declines(items,assigned,endpoints,blocking,drain,gone)
                for r in declines:decline(root,r,endpoints,busy,config['amendment_sha256'],drain,r['state_id'] in gone)
                slots=tuple(sorted(endpoints))
                if slots!=last_slots:
                    # Existing live clients dynamically read this file; truthful
                    # actual capacity releases their stale two-slot backpressure.
                    immutable(root/f'WORKER_ADMISSION_Z_MAIN0508_{time.time_ns()}.json',
                        dict(workers=[endpoints[s] for s in slots],amendment_sha256=config['amendment_sha256']))
                    last_slots=slots
            stop.wait(.1)
    except BaseException as error:
        errors.append(repr(error));stop.set()
        if not stop_path.exists():
            immutable(stop_path,dict(error=repr(error),source='capacity broker control',unix=time.time()))
    finally:
        for t in threads.values():t.join()
        lock.close()
    if errors:raise ValueError(errors)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('config',type=Path)
    a=p.parse_args();run(a.root,a.config)
