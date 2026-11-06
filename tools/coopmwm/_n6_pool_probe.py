"""Test-only runner stand-in for N6's pool regression test (no SUMO): emulates the G5 binding rule that a policy
(plant) episode refuses a worker process in which a NATIVE episode already imported the env."""
import os

MAP = 'wz3to2'
NATIVE = True
_NATIVE_SEEN = []


def episode(spec, cfg, work):
    if spec['arm'] == 'NATIVE':
        _NATIVE_SEEN.append(spec['job_id'])
    elif _NATIVE_SEEN:
        raise ValueError('traffic_shepherd.c4.env was imported before the G5 runtime binding')
    return dict(job_id=spec['job_id'], arm=spec['arm'], pid=os.getpid())


logcheck = episode
