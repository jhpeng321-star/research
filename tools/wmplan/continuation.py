"""Finite, exclusive collection of the already-dispatched approved restart.

No launch, retry, signal, model choice, or behaviour changes are made here.
"""
from copy import deepcopy
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.coopbandit.protocol import canonical, digest, sha
from tools.wmplan import ops, restart, report
from tools.wmplan.design import panel, decision, admit, H_CREDIT
from tools.coopplan.planner import TIMES, candidates
from tools.coopplan.predictor import Prediction, Outcome, ObservableState


def configure():
    restart.configure()
    report.OUT = ops.OUT


def smoke_collect():
    configure()
    r = json.loads(ops.remote(f'''import pathlib,json,sys,hashlib,time
root=pathlib.Path({ops.ROOT!r});sys.path.insert(0,str(root/'pytools'))
from tools.wmplan.live import verify
binding=verify(root)
def load(p):return json.loads(p.read_text())
assert (root/'SMOKE_COMPLETE.json').exists() and not (root/'STOP.json').exists()
jobs=list((root/'jobs').iterdir());assert all((p/'RESULT.json').exists() and (p/'CLOSE.json').exists() for p in jobs)
assert [p.name for p in jobs if '_t' not in p.name]==['s984000_expected']
charges=[json.loads(s) for s in (root/'CHARGES.jsonl').read_text().splitlines()]
print(json.dumps(dict(binding=binding,smoke=load(root/'SMOKE_COMPLETE.json'),charges=charges,
 results={{p.name:load(p/'RESULT.json') for p in jobs}},closed={{p.name:load(p/'CLOSE.json') for p in jobs}},
 baseline=load(pathlib.Path(binding['baseline_root'])/'jobs/s984000_mappo/RESULT.json'),
 observations={{p.stem:load(p) for p in (root/'requests').glob('*.json')}},
 forecasts={{p.stem:load(p) for p in (root/'responses').glob('*.json')}},
 reservations={{p.stem:load(p) for p in (root/'reservations').glob('*.json')}},
 result_hashes={{p.name:hashlib.sha256((p/'RESULT.json').read_bytes()).hexdigest() for p in jobs}},unix=time.time())))
''',timeout=180))
    ops.save('SMOKE_RECEIPTS.json',r)
    return r


def check_smoke(r):
    """Recompute every smoke decision, quota, exact twin and endpoint identity."""
    w = panel()[0]
    s = r['results']['s984000_expected']
    b = r['baseline']
    assert s['world'] == b['world'] == w and s['arm'] == 'expected' and s['warmup_pass']
    assert s['actual_sim_s'] == b['actual_sim_s'] == 1200
    assert r['smoke']['result_sha256'] == r['result_hashes']['s984000_expected']
    assert [e['time_s'] for e in s['history']] == list(TIMES)
    assert set(s['clocks']) == {'600','900'}
    assert all(math.isfinite(v) for clock in s['clocks'].values() for v in clock.values())
    predictions = {}
    for key,f in r['forecasts'].items():
        p = Prediction({a:tuple(Outcome(**v) for v in fs) for a,fs in f['futures'].items()},False,'WM-PHYS-'+f['backend'])
        predictions[digest({a:[vars(v) for v in fs] for a,fs in p.actions.items()})] = (p,r['observations'][key])
    bytime = {c['time_s']:c for c in s['credits']}
    assert len(bytime) == len(s['credits'])
    expected_jobs = {'s984000_expected'}
    spent = 0
    for e in s['history']:
        t = e['time_s'];key = f's984000_expected_t{t}'
        p,observation = predictions[e['prediction_id']]
        assert observation['world_id'] == 984000 and observation['time_s'] == t
        actions = candidates(ObservableState(t,tuple(observation['rows']),key))
        assert canonical(actions) == canonical(e['candidates'])
        assert canonical(observation['rows']) == canonical(e['initial_rows'])
        p.validate(actions,8)
        d = decision(p,'expected')
        assert canonical(d) == canonical(e['decision'])
        chosen = next(a for a in actions if a['id'] == d['selected'])
        event = dict(time_s=t,executed_action=deepcopy(chosen),group=chosen['members'])
        cost = admit(event,spent)
        for field in ('proposed_action','executed_action','budget_pruned'):
            assert canonical(event[field]) == canonical(e[field])
        name = event['executed_action']['id']
        if name == 'O0':
            assert t not in bytime
            continue
        c = bytime[t]
        ids = [key+'_O0',key+'_'+name,key+'_'+name+'_solo0',key+'_'+name+'_solo1']
        branches = [r['results'][i] for i in ids]
        assert all(v['prefix_pass'] and v['actual_sim_s'] == t+H_CREDIT for v in branches)
        js = [v['metrics']['team_delay_vs'] for v in branches]
        dg = js[0]-js[1];ds = [js[0]-v for v in js[2:]]
        assert c['action'] == chosen and c['horizon_s'] == H_CREDIT and c['D_G'] == dg and c['member_alone_D'] == ds and c['C_G'] == dg-sum(ds)
        assert r['reservations'][key] == dict(cost=cost,world=w,time_s=t,action=name)
        expected_jobs.update(ids);spent += cost
    assert spent == s['credit_sim_s'] and len(r['reservations']) == len(s['credits'])
    assert set(r['results']) == set(r['closed']) == expected_jobs
    assert len(r['charges']) == len(expected_jobs) and {c['id'] for c in r['charges']} == expected_jobs
    for c in r['charges']:
        close = r['closed'][c['id']]
        assert c['sim_s'] == close['actual_sim_s'] == close['charged_sim_s']
    if not s['credits']:
        assert s['all_O0_bitwise_pass'] and s['final_prefix'] == b['final_prefix'] and s['clocks'] == b['clocks']
    return dict(pass_=True,seed=984000,arm='expected',decisions=12,executed_groups=len(s['credits']),
                exact_prefixes=4*len(s['credits']),actual_sim_s=1200+spent,charged_sim_s=1200+spent,
                combined_actual_sim_s=900+1200+spent,combined_charged_sim_s=3600+1200+spent,
                smoke_included_in_primary=True,no_efficacy_based_gate=True)


def smoke():
    configure()
    r = (json.loads((ops.OUT/'SMOKE_RECEIPTS.json').read_text()) if (ops.OUT/'SMOKE_RECEIPTS.json').exists() else smoke_collect())
    qa = check_smoke(r)
    if not (ops.OUT/'SMOKE_QA.json').exists():
        ops.save('SMOKE_QA.json',dict(**qa,receipt_sha256=sha(ops.OUT/'SMOKE_RECEIPTS.json')))
    print(json.dumps(qa),flush=True)
    return qa


def release_smoke():
    configure()
    from tools.wmplan.watch import git
    assert not git('status','--porcelain').strip(),'commit the smoke receipt before releasing the panel'
    qa = json.loads((ops.OUT/'SMOKE_QA.json').read_text())
    assert qa['pass_']
    print(ops.remote(f'''import pathlib,json,time
root=pathlib.Path({ops.ROOT!r})
assert (root/'SMOKE_COMPLETE.json').exists() and not (root/'STOP.json').exists()
with (root/'SMOKE_APPROVED.json').open('x') as f:json.dump(dict(unix=time.time(),qa_sha256={sha(ops.OUT/'SMOKE_QA.json')!r}),f)
print(json.dumps(dict(smoke_pass=True,remaining_primary_released=23,strict_released=False)))
'''),flush=True)


def publish(primary=False,verify=False):
    configure()
    prefix = 'PRIMARY_' if primary else ''
    if not verify and not (ops.OUT/(prefix+'RECEIPTS.json')).exists():
        ops.status(collect=True,primary=primary)
    report.run(verify=verify,primary=primary)
    if not verify:
        report.run(verify=True,primary=primary)
        report.publish(primary=primary)


def inspect_gate():
    result = json.loads(ops.remote(f'''import pathlib,json,time
root=pathlib.Path({ops.ROOT!r})
owners=json.loads((root/'DISPATCH.json').read_text())['owners'];alive={{}}
for owner in owners:
 p=pathlib.Path('/proc')/str(owner['pid'])/'stat';s=p.read_text().split() if p.exists() else None
 alive[owner['name']]=bool(s and s[21]==owner['start_ticks'] and s[2]!='Z' and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id'])
print(json.dumps(dict(unix=time.time(),smoke_complete=(root/'SMOKE_COMPLETE.json').exists(),
 smoke_released=(root/'SMOKE_APPROVED.json').exists(),primary_complete=(root/'ARM_COMPLETE_expected.json').exists(),
 strict_released=(root/'PRIMARY_REPORTED.json').exists(),complete=(root/'COMPLETE.json').exists(),
 stopped=(root/'STOP.json').exists(),owned_services_alive=alive)))
'''))
    if (ops.OUT/'GPU_R_READY.json').exists():
        owner = json.loads((ops.OUT/'GPU_R_READY.json').read_text())
        gpu_r = json.loads(ops.remote(f'''import pathlib,json
owner={owner!r};p=pathlib.Path('/proc')/str(owner['pid'])/'stat';s=p.read_text().split() if p.exists() else None
alive=bool(s and s[21]==owner['start_ticks'] and s[2]!='Z' and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id'])
print(json.dumps(dict(alive=alive)))
''',host='gpuR'))
        result['owned_services_alive']['gpuR'] = gpu_r['alive']
    return result


def fault(error):
    """Durable stop of this root only, then preserve all partial evidence."""
    configure()
    print(ops.remote(f'''import pathlib,sys,json
root=pathlib.Path({ops.ROOT!r});sys.path.insert(0,str(root/'pytools'))
from tools.coopplan.storage import stop
stop(root,{error!r})
print(json.dumps(dict(stopped=True,retry=False)))
'''),flush=True)
    ops.save('CONTINUATION_FAULT.json',dict(unix=time.time(),error=error,retry=False))
    receipt = json.loads(ops.remote(f'''import pathlib,json,time,hashlib
root=pathlib.Path({ops.ROOT!r})
def load(p):return json.loads(p.read_text())
charges=[json.loads(s) for s in (root/'CHARGES.jsonl').read_text().splitlines()] if (root/'CHARGES.jsonl').exists() else []
jobs={{p.name:dict(close=load(p/'CLOSE.json') if (p/'CLOSE.json').exists() else None,
 result=load(p/'RESULT.json') if (p/'RESULT.json').exists() else None,
 decisions={{f.name:load(f) for f in p.glob('DECISION_*.json')}}) for p in (root/'jobs').iterdir()}}
actual=sum(j['close']['actual_sim_s'] for j in jobs.values() if j['close'])
print(json.dumps(dict(unix=time.time(),root=str(root),binding=load(root/'BINDING.json'),stop=load(root/'STOP.json'),
 charges=charges,jobs=jobs,closed_actual=actual,combined_closed_actual=actual+900,
 charged=sum(c['sim_s'] for c in charges),combined_charged=sum(c['sim_s'] for c in charges)+3600,
 open_jobs=[k for k,v in jobs.items() if not v['close']],complete_episodes=sum(v['result'] is not None and '_t' not in k for k,v in jobs.items()),
 forecasts={{p.name:dict(response=load(root/'responses'/p.name) if (root/'responses'/p.name).exists() else None,
 request=load(p)) for p in (root/'requests').glob('*.json')}},
 logs={{p.name:p.read_text() for p in root.glob('*.log')}},retry=False)))
''',timeout=180))
    ops.save('FAULT_RECEIPTS.json',receipt)
    return receipt


def main():
    configure()
    from tools.wmplan.watch import commit,progress
    ops.save('CONTINUATION_ONCE.json',dict(pid=os.getpid(),unix=time.time(),hours=168,
        role='collect_existing_dispatch_only',new_dispatches=0))
    frozen = {p:sha(p) for p in (Path(__file__),Path(report.__file__),Path(restart.__file__),Path(ops.__file__))}
    deadline = time.monotonic()+168*3600
    last_progress = 0
    try:
        while time.monotonic() < deadline:
            if any(sha(p) != h for p,h in frozen.items()):
                raise RuntimeError('continuation/report source changed; stop/report, no retry')
            gate = inspect_gate()
            if gate['stopped']:
                raise RuntimeError('new remote durable STOP; no retry')
            if not all(gate['owned_services_alive'].values()) and not gate['complete']:
                raise RuntimeError('owned service ended incomplete; stop/report, no retry')
            if gate['smoke_complete'] and not gate['smoke_released']:
                qa = smoke()
                progress(f'- Approved v2 smoke PASS:12 decisions, {qa["executed_groups"]} complete group quartet(s), '
                         f'{qa["exact_prefixes"]} exact prefix gates. Smoke counted once in primary.\n'
                         f'- Combined actual{qa["combined_actual_sim_s"]}/charged{qa["combined_charged_sim_s"]}sim-s; '
                         'FULL-v4 fixed. Remaining23 primary worlds next; strict waits for committed primary report.')
                commit('wmplan: verify completed smoke and exact twins before primary panel')
                release_smoke()
            if gate['primary_complete'] and not gate['strict_released']:
                publish(primary=True)
                commit('wmplan: publish verified primary expected-arm results before strict')
                print(ops.remote(f'''import pathlib,json,time
root=pathlib.Path({ops.ROOT!r})
assert not (root/'STOP.json').exists()
with (root/'PRIMARY_REPORTED.json').open('x') as f:json.dump(dict(unix=time.time(),report_sha256={sha(ops.LOCAL/'docs/paper_results/WMPLAN.md')!r}),f)
print(json.dumps(dict(primary_published=True,strict_released=True)))
'''),flush=True)
            if gate['complete']:
                publish()
                ops.save('CONTINUATION_COMPLETE.json',dict(unix=time.time(),new_dispatches=0,
                    report_sha256=sha(ops.LOCAL/'docs/paper_results/WMPLAN.md')))
                commit('wmplan: publish verified final paired outcomes and retained restart charges')
                return
            if time.monotonic()-last_progress >= 1800:
                s = ops.status()
                live = next(v for v in s['owners'] if v['name']=='LIVE')
                if not live['identity_match'] or live['state'] in (None,'Z'):
                    raise RuntimeError('owned supervisor ended incomplete; stop/report, no retry')
                progress(f'- Approved v2 active: {s["episodes"]}/48 complete; '
                         f'closed combined actual{s["actual"]+900:g}/charged{s["charged"]+3600:g}sim-s '
                         f'({len(s["incomplete"])} open jobs excluded from closed actual).\n'
                         '- Combined upper199440charged/196740actual; no refund. FULL-v4, expected first on both GPUs; '
                         'strict waits for committed primary report. No full-panel efficacy conclusion yet.')
                commit('wmplan: preserve approved restart budget and completion progress')
                last_progress = time.monotonic()
            time.sleep(60)
        raise TimeoutError('finite168h continuation deadline; stop/report, no retry')
    except BaseException:
        error = traceback.format_exc()
        receipt = fault(error)
        progress('- Approved v2 continuation STOPPED on a new fault; no retry or new dispatch.\n'
                 f'- Combined charged{receipt["combined_charged"]:g}; closed actual{receipt["combined_closed_actual"]:g}sim-s, '
                 f'{len(receipt["open_jobs"])} open jobs at receipt time. All prior charges retained.\n'
                 '- See `wmplan/restart_v2/CONTINUATION_FAULT.json` and `FAULT_RECEIPTS.json`; '
                 'complete paired efficacy cannot be asserted from an incomplete arm.')
        line = (f'- S-WMPLAN APPROVED RESTART NEW FAULT / STOP, NO RETRY: '
                f'completeepisodes{receipt["complete_episodes"]}; combinedcharged{receipt["combined_charged"]:g}/'
                f'closedactual{receipt["combined_closed_actual"]:g}sim-s, openjobs{len(receipt["open_jobs"])}; '
                'prior3600charged/900actual retained; FULL-v4 unchanged; '
                f'faultSHA256{sha(ops.OUT/"FAULT_RECEIPTS.json")}; '
                'E:/agent-work/c4-wmplan/docs/paper_results/WMPLAN.md.\n')
        with ops.REG.open('a',encoding='utf-8') as f:
            f.write(line)
        ops.save('FAULT_REGISTRATION.json',dict(line=line))
        commit('wmplan: preserve new restart fault without retry')
        raise


def launch():
    configure()
    assert (ops.OUT/'DISPATCH.json').is_file() and not (ops.OUT/'CONTINUATION_LAUNCH.json').exists()
    logpath = ops.LOCAL/'_out/wmplan_v2_continuation.log'
    with logpath.open('x',encoding='utf-8') as log:
        p = subprocess.Popen([sys.executable,'-B','-u','-m','tools.wmplan.continuation'],cwd=ops.LOCAL,
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
    receipt = dict(pid=p.pid,unix=time.time(),log=str(logpath),role='existing_dispatch_only')
    ops.save('CONTINUATION_LAUNCH.json',receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--launch',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--release-smoke',action='store_true')
    p.add_argument('--verify',action='store_true')
    p.add_argument('--gate',action='store_true')
    p.add_argument('--primary',action='store_true')
    a = p.parse_args()
    if a.gate:
        configure()
        print(json.dumps(inspect_gate()))
    elif a.launch:
        launch()
    elif a.smoke:
        smoke()
    elif a.release_smoke:
        release_smoke()
    elif a.verify:
        publish(primary=a.primary,verify=True)
    else:
        main()
