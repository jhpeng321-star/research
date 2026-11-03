"""Read-only hourly null/diversity report for CURRENT B/R banks, not legacy LS only."""
import argparse
import json
import os
import subprocess
import sys
import time
from . import ops
from .ls_validity_ops import remote
from .ls_collect_ops import ROOT, COLLECTION
from .ls_label_ops import variation_summary
from .storage import read, write
from .protocol import sha, checked, REGISTRY

OWNER = ops.EVIDENCE/'LIVE_BANK_REPORT_DISPATCH.json'
REG = ops.EVIDENCE/'LIVE_BANK_REPORT_REGISTRATION_1340.json'


def report():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time,sys
sys.path.insert(0,'/home/ubuntu/coopmwm20261103_compositereleasecpu1/pytools')
from tools.coopmwm.protocol import digest
r=Path({ROOT!r});c=Path({COLLECTION!r});out=dict(unix=time.time(),banks={{}},activation_config_present=(c/'LS_NULLFIX_CONFIG.json').exists())
for arm in ('B','R'):
 b=r/'replay_bank'/arm;rows=[];failed=0
 for p in (b/'decisions').glob('*.json'):
  link=json.loads(p.read_text());vpath=b/'sumo_verification/verifications'/(link['verification_id']+'.json')
  if not vpath.exists():continue
  v=json.loads(vpath.read_text())
  if v.get('receipt_sha256')!=digest({{k:x for k,x in v.items() if k!='receipt_sha256'}}):raise ValueError('bank verification receipt digest mismatch')
  if v['status'] not in ('accepted','rejected'):
   failed+=1;continue
  key=link['state_id'];response=r/('responses' if arm=='B' else 'centre_responses')/(key+'.json');request=r/('requests' if arm=='B' else 'centre_requests')/(key+'.json')
  a=json.loads(response.read_text()) if response.exists() else {{}};q=json.loads(request.read_text()) if request.exists() else {{}}
  variation=a.get('variation') or {{}};config=q.get('search_config') or a.get('config') or {{}}
  costs=v.get('J_candidates_veh_s') or [];policy=v.get('J_policy_veh_s')
  null=all(x==policy for x in costs) if len(costs)==8 and policy is not None and all(x is not None for x in costs) else None
  rows.append(dict(state_id=key,verification_id=v['verification_id'],arm=arm,status=v['status'],policy_update=link.get('policy_update'),candidate_count=len(costs),exact_null=null,
   niches=config.get('niches') if arm=='B' else None,search_config=config if arm=='B' else {{'proposer':'random_unscored'}},
   search_clamp_enabled=bool((a.get('nullfix') or {{}}).get('enabled')) if arm=='B' else False,
   mean_pairwise_distance=variation.get('mean_pairwise_distance') if arm=='B' else None,
   occupied_niches=variation.get('occupied_niches') if arm=='B' else None,
   share_with_net_deceleration=variation.get('share_with_net_deceleration') if arm=='B' else None,
   mirror_active=False,best_margin_veh_s=v.get('best_margin_veh_s')))
 quality=b/'BANK_QUALITY.json';quality=json.loads(quality.read_text()) if quality.exists() else {{}}
 out['banks'][arm]=dict(rows=rows,quality={{k:quality.get(k) for k in ('unix','decisions','first_seed_positives','confirmed','pending_confirmation','admitted','cooperative')}},failed_or_ungraded_verifications=failed,
  inactionable_skipped=sum(json.loads(p.read_text()).get('reason')=='all candidates inactionable' for p in (b/'skipped').glob('*.json')))
print(json.dumps(out))''',timeout=45))
    value['banks']['B']['cohorts']=variation_summary(value['banks']['B']['rows'])
    value['banks']['R']['cohorts']={'random_unscored':variation_summary(value['banks']['R']['rows'])['before_release']}
    for bank in value['banks'].values():
        assert sum(c['labelled_lists'] for c in bank['cohorts'].values())==len(bank['rows'])
        assert all(0<=c['exact_null_lists']<=c['complete_eight_lists']<=c['labelled_lists'] for c in bank['cohorts'].values())
    value.update(new_sim_s=0,new_inference=0,source_sha256=sha(__file__),
        receipt_checksums_verified=True,count_invariants_passed=True,
        scope='current own-bank verifications only; legacy LS labels remain in NULL_DIVERSITY_BY_CONFIG.json',
        denominator='exact-null fraction over complete eight-candidate graded lists only; partial lists excluded',
        diversity='original returned B WM shortlist; R shortlist diversity not recorded here, never imputed',
        null_definition='all eight exact candidate delays equal policy delay; exact equality, no behavioural threshold')
    p=ops.EVIDENCE/f'LIVE_BANK_NULL_DIVERSITY_{time.time_ns()}.json';write(p,value)
    target=ops.EVIDENCE/'LIVE_BANK_NULL_DIVERSITY_BY_CONFIG.json'
    temp=target.with_name(f'.LIVE_BANK_NULL_DIVERSITY.{os.getpid()}.publishing')
    with temp.open('x',encoding='utf-8') as f:json.dump(value,f)
    os.replace(temp,target)
    print(json.dumps(dict(receipt=str(p),unix=value['unix'],activation_config_present=value['activation_config_present'],
        banks={a:{k:v for k,v in b.items() if k!='rows'} for a,b in value['banks'].items()},new_sim_s=0,new_inference=0)))
    return value


def alive(d):
    import psutil
    try:
        p=psutil.Process(d['pid'])
        return p.create_time()==d['create_time'] and 'tools.coopmwm.live_bank_report' in p.cmdline()
    except (psutil.NoSuchProcess,psutil.AccessDenied):return False


def run():
    with (ops.EVIDENCE/'LIVE_BANK_REPORT.claim').open('x') as f:
        json.dump(dict(pid=os.getpid(),unix=time.time(),read_only=True),f)
    while time.time()<1793894340:
        try:report()
        except Exception as e:
            write(ops.EVIDENCE/f'LIVE_BANK_REPORT_ERROR_{time.time_ns()}.json',dict(error=repr(e),read_only=True,unix=time.time()))
        time.sleep(3600)


def start():
    import psutil
    if OWNER.exists():
        d=read(OWNER);print(json.dumps(dict(owner=d,alive=alive(d),adopt=True)));return
    registration=read(REG)
    assert registration['source_sha256']==sha(__file__)
    assert sha(REG) in checked(REGISTRY).read_text(encoding='utf-8')
    assert ops.EVIDENCE.is_dir()
    with (ops.EVIDENCE/'LIVE_BANK_REPORT.log').open('x',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-X','utf8','-u','-m','tools.coopmwm.live_bank_report','run'],
            cwd=str(ops.LOCAL),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=(subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0,
            start_new_session=os.name!='nt')
    d=dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time(),unix=time.time(),read_only=True,
        interval_s=3600,source_sha256=sha(__file__),new_sim_s=0,new_inference=0)
    write(OWNER,d);print(json.dumps(d))


def register():
    if not REG.exists():
        p=max(ops.EVIDENCE.glob('LIVE_BANK_NULL_DIVERSITY_[0-9]*.json'),key=lambda p:p.stat().st_mtime_ns)
        v=read(p)
        assert v['source_sha256']==sha(__file__) and v['receipt_checksums_verified'] and v['count_invariants_passed']
        line=('\nS-COOPMWM MAIN13:40 hourly publication engineering amendment: '
              'existing LS-only variation report omits current B/R bank labels. Add '
              'a distinct read-only hourly current-bank reporter, same exact-null '
              'definition/complete-eight denominator and existing diversity helper; '
              'preserve LS report and missing R diversity as null. No search/physics/'
              'training/selection changes; owned local detached observer, zero SUMO/inference.\n')
        write(REG,dict(authority='MAIN2026-11-03 13:40',line=line,source_sha256=sha(__file__),
            smoke_receipt=p.name,smoke_sha256=sha(p),interval_s=3600,new_sim_s=0,new_inference=0))
    h=sha(REG);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(read(REG)['line']+'amendmentSHA'+h+'\n')
    print(json.dumps(dict(registered=True,receipt=str(REG),sha256=h,new_sim_s=0,new_inference=0)))


def status():
    d=read(OWNER);print(json.dumps(dict(owner=d,alive=alive(d))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('report','register','start','run','status'))
    globals()[p.parse_args().action]()
