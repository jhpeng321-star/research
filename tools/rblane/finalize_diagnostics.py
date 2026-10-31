"""Assemble verified final conditional metrics; no repeated inference."""
import hashlib
import json

from .operate import RECEIPTS,ROOT


def main():
    out=RECEIPTS/'FINAL_DIAGNOSTICS.json'
    assert out.parent.is_dir() and not out.exists()
    proof=json.loads((RECEIPTS/'MESSAGE_DECISION_EQUIVALENCE.json').read_text())
    assert proof['status']=='PASS'
    assert proof['current_lc_sha256']==hashlib.sha256((ROOT/'src/wms/physics/lc2013.py').read_bytes()).hexdigest()
    variants=[]
    for kind,label in (('panel14','population'),('panel15','frozen_encoder'),('panel16','published_native_points')):
        path=RECEIPTS/(kind+'_results/SUMMARY.json')
        verified=json.loads((RECEIPTS/(kind.upper()+'_VERIFIED.json')).read_text())
        assert verified['status']=='PASS' and verified['summary_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        summary=json.loads(path.read_text()); score=summary['aggregate']['lc2013']
        exact=score['exact']; false=exact['predicted']-exact['matched']
        motives={}
        for name,value in score['by_motivation'].items():
            cell=value['exact']
            motives[name]=dict(cell,false_positives=cell['predicted']-cell['matched'])
        result=dict(variant=label,receipt=kind,episodes=summary['episodes'],worlds=summary['worlds'],
            arms=summary.get('diagnostic_arms'),exact=exact,false_positives=false,
            by_motivation=motives,run_window_2s=score['run_window_2s'],
            by_transition=score['by_transition'],branches=score.get('branch_counts',{}),
            mobil=summary['aggregate']['mobil']['exact'],summary_sha256=verified['summary_sha256'])
        variants.append(result)
        print(json.dumps(dict(variant=label,exact=exact,motives=motives),indent=2))
    result=dict(status='VERIFIED_CONDITIONAL_DIAGNOSTIC',decision_source_commit=proof['baseline_commit'],
        message_upgrade_decision_equivalence=proof,current_lc_sha256=proof['current_lc_sha256'],
        snapshot='v2',driver_training_performed=False,distributional_qualification='NOT ESTABLISHED',
        variants=variants)
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
