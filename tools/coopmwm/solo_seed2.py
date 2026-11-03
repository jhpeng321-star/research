"""MAIN17:55 second-seed solos, using the existing qualified exact executor.

No policy update, search or new seed worlds. All solos share one policy branch.
Outputs are supplements; original first/replication/solo receipts stay immutable.
"""
import json
import os
from pathlib import Path
import time


def eligible(c):
    co=c.get('cooperation') or {}
    return bool(co.get('complete') and (co.get('cooperation_gain_veh_s') or 0)>0
        and c.get('checks',{}).get('replication') and c.get('checks',{}).get('no_later_harm'))


def cost(t0,duration,n,end=1200):return (n+1)*min(end,t0+duration+120)


def selected(root):
    from tools.coopmwm import replay_bank as rb
    from tools.coopmwm.storage import read
    root=Path(root);config=read(root/'B_REPLAY_EXPOSURE_V2.json')
    from tools.coopmwm.replay_start import binding
    cfg=binding(root,config,'B')
    conf=rb.confirmations(root,'B')
    rows=[r for r in rb.bank_decisions(root,'B',sources=cfg.get('replay_sources',{}).get('B',()))
        if eligible(conf.get(r['verification_id'],{}))]
    priority=(986066,986048,986003)
    rows.sort(key=lambda r:(next((i for i,s in enumerate(priority) if str(s) in r['verification_id']),3),r['verification_id']))
    return rows,conf


def quote(root):
    rows,conf=selected(root);entries=[]
    for r in rows:
        p=r['chosen_plan'];c=conf[r['verification_id']]
        entries.append(dict(verification_id=r['verification_id'],policy_version=r['_policy_version'],
            group=p['members'],duration_s=p['duration_s'],t0=r['t0'],
            sim_s=cost(r['t0'],p['duration_s'],len(p['members'])),
            first_seed_cooperation_gain_veh_s=c['cooperation']['cooperation_gain_veh_s']))
    return dict(entries=entries,total_sim_s=sum(e['sim_s'] for e in entries),
        phase='bank_verification_ext',new_worlds=0,new_wm_calls=0,
        rule='second CRN seed matches existing group replication; n solos + one shared policy per decision')


def run(root,control):
    import fcntl,hashlib
    from tools.coopmwm import replay_bank as rb,sumo_verify as sv
    from tools.coopmwm.storage import read,write
    from tools.coopmwm.exact_grade import _seal
    from tools.coopmwm.replay_start import owner
    root=Path(root);control=Path(control)
    authority=read(control/'AUTHORITY.json')
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==authority['source_sha256']
    lock=(control/'RUN.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (control/'OWNER.json').exists():raise ValueError('adopt existing seed2 solo owner')
    write(control/'OWNER.json',owner());os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    planned=authority['quote'];rows,conf=selected(root);by_vid={r['verification_id']:r for r in rows}
    links={read(p)['verification_id']:read(p) for p in (rb.bank_root(root,'B')/'decisions').glob('*.json')}
    try:
        for item in planned['entries']:
            vid=item['verification_id'];r=by_vid[vid];c=conf[vid]
            path=control/(vid+'.json')
            if path.exists():continue
            snapshot=rb._decision_snapshot(root,'B',r,links)
            replica=rb.replicate(snapshot,vid)
            original=read(rb.bank_root(root,'B')/'confirm_replication/verifications'/(c['replication']['verification_id']+'.json'))
            assert original['snapshot_sha256']==sv.snapshot_sha(replica),'replication snapshot identity differs'
            plans=[rb.solo_candidate(r['chosen_plan'],m) for m in item['group']]
            specs=sv.build_branches(replica,plans,r['proposer'])
            assert sum(s['sim_s'] for s in specs)==item['sim_s']
            base=control/'verification';runner=sv.SubprocessBranchRunner(base/'branches',workers=2,timeout_s=3600)
            result=sv.verify(replica,plans,runner,
                rb._confirm_charge(root,'bank_verification_ext','confirm_solo_seed2_1755'),
                r['proposer'],workers=2,root=base)
            if result['status'] not in ('accepted','rejected'):
                write(path,dict(status='incomplete',original_verification_id=vid,verification=result));raise RuntimeError('second-seed solos incomplete')
            assert result['J_policy_veh_s']==original['J_policy_veh_s'],'second-seed policy branch differs from original group replication'
            solos=[result['J_policy_veh_s']-v for v in result['J_candidates_veh_s']]
            gain=c['replication_advantage_veh_s']-max(solos)
            supplement=_seal(dict(status='complete',arm='B',original_verification_id=vid,
                policy_version=r['_policy_version'],group=item['group'],
                first_seed_cooperation_gain_veh_s=c['cooperation']['cooperation_gain_veh_s'],
                replication_cooperation_gain_veh_s=gain,replication_group_advantage_veh_s=c['replication_advantage_veh_s'],
                replication_solo_advantages_veh_s=solos,positive_both_seeds=c['cooperation']['cooperation_gain_veh_s']>0 and gain>0,
                verification=result,original_group_replication_receipt_sha256=original['receipt_sha256'],
                phase='bank_verification_ext',unix=time.time(),ci95=None,
                rule='raw second-seed evidence; BANK_STAT_V2 per-vehicle bootstrap must certify synergy separately'),
                'supplement_sha256')
            write(path,supplement)
        write(control/'COMPLETE.json',dict(unix=time.time(),entries=len(planned['entries']),quoted_sim_s=planned['total_sim_s']))
    except BaseException as e:
        write(control/'FAILED.json',dict(unix=time.time(),error=repr(e),global_stop_written=False));raise


if __name__=='__main__':
    import sys
    run(sys.argv[1],sys.argv[2])
