"""Shared current-policy centring for R; candidate perturbations stay unscored."""
from pathlib import Path
import time
from .protocol import digest
from .storage import read,write


def trajectory(root,record,deadline):
    root=Path(root);key=record['state_id']
    request=root/'centre_requests'/(key+'.json');response=root/'centre_responses'/request.name
    write(request,record)
    while not response.exists():
        from .warm_health import assert_running
        assert_running(root,'R')
        if time.time()>deadline:raise ValueError('original GPU deadline while waiting for centre')
        time.sleep(.5)
    value=read(response)
    if value['record_sha256']!=digest(record):raise ValueError('centre response identity mismatch')
    if value['candidate_scores']!=0:raise ValueError('R must never score a candidate with the WM')
    return value['policy_traj'],value['receipt']
