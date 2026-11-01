"""TRAIN-only exact plan evidence, never consumed by a learner or proposer."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import time
import numpy as np
from .accel_storage import read,write
from .protocol import digest,sha
from .warm_protocol import worlds,SEED


def bootstrap(rows):
    if not rows:return dict(plans=0,worlds=0,mean=None,ci95=None,positive=None)
    clusters=defaultdict(list)
    for r in rows:
        if not math.isfinite(r['exact_saving']):raise ValueError('nonfinite exact grade')
        clusters[r['world']].append(r['exact_saving'])
    values=list(clusters.values());sums=np.array([sum(v) for v in values]);ns=np.array([len(v) for v in values])
    ci=None
    if len(values)>1:
        ix=np.random.default_rng(SEED).integers(0,len(values),(10000,len(values)))
        samples=sums[ix].sum(1)/ns[ix].sum(1)
        ci=list(map(float,np.quantile(samples,[.025,.975])))
    return dict(plans=len(rows),worlds=len(values),mean=float(sums.sum()/ns.sum()),ci95=ci,
                positive=sum(r['exact_saving']>0 for r in rows))


def summarize(rows,excluded):
    groups={}
    for label in ('type','regime','predicted_rank','known_label','group_size','update'):
        keys=sorted({str(r[label]) for r in rows})
        groups[label]={k:bootstrap([r for r in rows if str(r[label])==k]) for k in keys}
    return dict(overall=bootstrap(rows),by=groups,excluded=excluded,rows=rows,
        estimator='mean over exact-graded executed TRAIN plans; 10000 whole-world cluster resamples, RNG20261101; one-world CI unavailable; selection-conditional marginal same-state effects',
        prediction_comparison='WM mean saving at registered Hfinal versus exact executedD+120 marginal SUMO saving; horizons may differ; no assumption that predictions equal causal effects',
        training_effect=False)


def collect(root):
    allowed={w['seed'] for w in worlds('train')}
    receipts=[];excluded=defaultdict(int)
    for p in sorted((root/'exact_grades/receipts').glob('*.json')):
        r=read(p)
        if r.get('arm')!='B' or r.get('training_seed')!=SEED:continue
        if r.get('world_seed') not in allowed:raise ValueError('grade outside registered TRAIN panel')
        if r['status']!='graded':excluded[r['status']]+=1;continue
        if r.get('receipt_sha256')!=digest({k:v for k,v in r.items() if k!='receipt_sha256'}):
            raise ValueError('exact-grade receipt SHA mismatch')
        if not r.get('prefix',{}).get('pass_') or not r.get('noise',{}).get('crn_verified_t0'):
            raise ValueError('unqualified exact grade; never estimate or impute')
        receipts.append(r)
    assert len({r['grade_id'] for r in receipts})==len(receipts),'duplicate exact grade'
    rows=[];by_episode=defaultdict(list)
    for r in receipts:by_episode[r['episode_id']].append(r)
    for ident,items in sorted(by_episode.items()):
        bits=ident.split('_')
        assert len(bits)==4 and bits[0]=='warm' and bits[1].startswith('u') and bits[1][1:].isdigit() and bits[2]=='B'
        assert int(bits[3]) in allowed
        path=root/'jobs'/ident/'RESULT.json'
        ep=read(path)
        for r in items:
            assert Path(r['result_path'])==path and ep['id']==ident
            assert ep['world']['seed']==r['world_seed'] and ep['arm']=='B'
            step=next(s for s in ep['steps'] if int(s['time_s'])==r['window']['start_s'])
            proposal=step['proposal'];b=step['sample']['behaviour']
            assert b['executed'] and b['plan_start'] and proposal['plan']['id']==r['plan']['id']
            assert digest(proposal['plan'])==r['plan']['plan_sha256']
            rows.append(dict(grade_id=r['grade_id'],receipt_sha256=r['receipt_sha256'],world=r['world_seed'],
                update=r['update'],exact_saving=r['grade_veh_s'],type=r['plan']['option'],
                regime=r['plan']['regime'],predicted_rank=proposal.get('predicted_rank'),
                known_label='known-like' if proposal.get('known_family') else 'new',
                group_size=len(r['plan']['members']),duration_s=r['plan']['duration_s'],
                predicted_saving=proposal.get('expected_saving'),
                posthoc_label=proposal.get('posthoc'),
                descriptors=proposal['plan'].get('descriptors',proposal['plan'].get('shape'))))
    key=digest(dict(receipts=[r['receipt_sha256'] for r in receipts],excluded=dict(excluded)))
    return key,summarize(rows,dict(excluded))


def run(root,config):
    import fcntl
    import os
    lock=(root/'EXACT_VIABILITY_REPORT.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert config['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
    for n,h in config['source_hashes'].items():assert sha(Path(config['source_root'])/n)==h,n
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    try:
        while True:
            key,report=collect(root)
            target=root/f'B_EXACT_VIABILITY_{key}.json'
            if report['overall']['plans'] and not target.exists():
                summary=report['overall']
                write(target,dict(timestamp_utc=time.time(),arm='B',update=max(r['update'] for r in report['rows']),
                    exact_trust=dict(mean=summary['mean'],ci95=summary['ci95'],graded=summary['plans']),
                    exact_plan_evidence=report,report_only=True,source='qualified exact marginal SUMO grades only'))
            if (root/'EXACT_VIABILITY_REPORT_STOP.json').exists():return
            if (root/'FINAL_DISCOVERY_COMPLETE.json').exists():
                latest=sorted(root.glob('EXACT_GRADER_STATUS_*.json'))
                status=read(latest[-1]) if latest else {}
                if status and not status['pending'] and not status['running']:
                    write(root/'EXACT_VIABILITY_REPORT_COMPLETE.json',dict(unix=time.time(),graded=report['overall']['plans']))
                    return
            time.sleep(300)
    except BaseException as error:
        write(root/'EXACT_VIABILITY_REPORT_FAILED.json',dict(error=repr(error),unix=time.time(),training_unaffected=True))
        raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('config',type=Path)
    a=p.parse_args();run(a.root,read(a.config))
