"""SHA-linked per-member views of already graded batched second-seed solos."""
import copy
import json
from pathlib import Path


def views(supplement):
    from .exact_grade import _seal,verify_sealed
    verify_sealed(supplement,'supplement_sha256')
    source=supplement['verification'];verify_sealed(source,'receipt_sha256')
    assert len(source['J_candidates_veh_s'])==len(supplement['group'])
    out=[]
    for i,member in enumerate(supplement['group']):
        grade=source['J_policy_veh_s']-source['J_candidates_veh_s'][i]
        body=dict(schema='coopmwm/solo-seed2-receipt-view/v1',
            verification_id=source['verification_id']+f'__member{i}',
            source_verification_id=source['verification_id'],source_receipt_sha256=source['receipt_sha256'],
            source_candidate_index=i,member=member,snapshot_sha256=source['snapshot_sha256'],
            status='accepted' if grade>0 else 'rejected',window=source['window'],
            branches=[source['branches'][0],source['branches'][i+1]],
            J_policy_veh_s=source['J_policy_veh_s'],J_candidates_veh_s=[source['J_candidates_veh_s'][i]],
            grade_veh_s=grade,charged_sim_s=0,new_sim_s=0,
            rule='analytical per-member view of one already charged batched verification; original branch IDs/digests unchanged')
        out.append(_seal(body,'receipt_sha256'))
    return out


def merge(bank,confirmations):
    from .exact_grade import _seal,verify_sealed
    out=dict(confirmations)
    for p in (Path(bank)/'confirm_solo_replication/links').glob('*.json'):
        link=json.loads(p.read_text());verify_sealed(link,'link_sha256');vid=link['verification_id']
        if vid not in out:raise ValueError('second-seed link has no original confirmation')
        original=out[vid]
        if original['confirmation_sha256']!=link['original_confirmation_sha256']:
            raise ValueError('second-seed link original confirmation SHA mismatch')
        c=copy.deepcopy(original);c.pop('confirmation_sha256')
        c['original_confirmation_before_seed2_sha256']=original['confirmation_sha256']
        c['second_seed_solo_link_sha256']=link['link_sha256']
        c.setdefault('cooperation',{})['replication_solo']=link['replication_solo']
        c['cooperation']['replication_cooperation_gain_veh_s']=link['replication_cooperation_gain_veh_s']
        out[vid]=_seal(c,'confirmation_sha256')
    return out


def wire(root,controls):
    from . import replay_bank as rb
    from .exact_grade import _seal,verify_sealed
    from .storage import write
    bank=rb.bank_root(root,'B');confirmed=rb.confirmations(root,'B');base=bank/'confirm_solo_replication'
    (base/'verifications').mkdir(parents=True,exist_ok=True);(base/'branches').mkdir(exist_ok=True)
    (base/'links').mkdir(exist_ok=True);wired=[]
    for control in map(Path,controls):
        for p in sorted(control.glob('sv_*.json')):
            s=json.loads(p.read_text())
            if s.get('status')!='complete':continue
            verify_sealed(s,'supplement_sha256');vid=s['original_verification_id']
            target=base/'links'/(vid+'.json')
            if target.exists():
                old=json.loads(target.read_text());verify_sealed(old,'link_sha256')
                assert old['source_supplement_sha256']==s['supplement_sha256'];wired.append(vid);continue
            rows=[]
            for r in views(s):
                target_receipt=base/'verifications'/(r['verification_id']+'.json')
                if target_receipt.exists():assert json.loads(target_receipt.read_text())==r
                else:write(target_receipt,r)
                branch=base/'branches'/r['verification_id']
                source=(control/'verification/branches'/r['source_verification_id']).resolve()
                assert source.is_dir()
                if not branch.exists():branch.symlink_to(source,target_is_directory=True)
                assert branch.is_symlink() and branch.resolve()==source
                rows.append(dict(member=r['member'],verification_id=r['verification_id'],
                    status=r['status'],advantage_veh_s=r['grade_veh_s'],receipt_sha256=r['receipt_sha256']))
            link=_seal(dict(verification_id=vid,original_confirmation_sha256=confirmed[vid]['confirmation_sha256'],
                source_supplement_sha256=s['supplement_sha256'],replication_solo=rows,
                replication_cooperation_gain_veh_s=s['replication_cooperation_gain_veh_s'],new_sim_s=0),
                'link_sha256')
            write(target,link);wired.append(vid)
    return dict(wired=wired,new_sim_s=0)
