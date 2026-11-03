"""One-use B-u4 diagnostic correction; original lesson and valid branches retained."""
from pathlib import Path
import json
import time
from .storage import read,write


def run(root,control):
    import fcntl,torch
    from .replay_start import config,accounting
    from .accel_learner import load
    from .replay_bank import record_lesson
    from .sumo_verify import SubprocessBranchRunner
    root=Path(root);control=Path(control);v=config(root);accounting(v);torch.set_num_threads(1)
    lock=(control/'CORRECTION.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (control/'CORRECTION_STARTED.json').exists(),'adopt one-use diagnostic correction'
    old=read(root/'LESSON_CHECK_u004_B.json')
    write(control/'CORRECTION_STARTED.json',dict(unix=time.time(),update=4,arm='B',
        original_receipt_sha256=old['lesson_check_sha256'],extra_sim_s_upper=4260))
    policy,_,_,payload=load(root/'B_checkpoint_004.pt');assert payload['update']==4
    out=record_lesson(root,'B',policy,root,4,
        lambda work:SubprocessBranchRunner(Path(work),workers=2,timeout_s=3600),
        sources=[v['collection_root']],adopt_entries=old['entries'],receipt_suffix='_MATCHED_WINDOW_V1')
    write(control/'CORRECTION_COMPLETE.json',dict(unix=time.time(),list_size=out['list_size'],
        graded_entries=out.get('graded_entries'),error=out.get('error'),
        adopted_branches=sum(e.get('adopted_existing_branch',False) for e in out['entries']),
        primary={k:out.get(k) for k in ('delta_mean_veh_s','delta_ci95_veh_s','share_delta_positive',
            'recapture_median','recapture_median_ci95','recapture_eligible_entries')}))


if __name__=='__main__':
    import sys
    run(sys.argv[1],sys.argv[2])
