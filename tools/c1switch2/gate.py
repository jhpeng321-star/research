"""Approval parser and CPU-only polling: budget authority is never self-issued."""
import argparse
import json
import re
import time

from tools.c1switch.design import checked, sha
from tools.c1switch.storage import canonical, write

from .design import CONTROLLERS, DESIGN, GPU_SECONDS, PROTOCOL, REGISTRATION, budget


def approval(text):
    lines = []
    for line in text.splitlines():
        # Require an actual MAIN-authored approval entry. Quoted requests, future
        # conditions, our design line, and S-C1SWITCH's old approval do not match.
        if re.match(r'^-\s+(?:\d\d:\d\d\s+)?MAIN\b[^\n]*\bS-C1SWITCH2 BUDGET APPROVED\b', line):
            if not re.search(r'\b(?:NOT|NO|AWAIT|PENDING|REVOKED|EXAMPLE)\b', line):
                lines.append(line)
    return lines[-1] if lines else None


REQUIRED = ('frozen_v7_sources', 'seed_guards', 'exposure_cpuQ', 'exposure_gpuR',
            'exposure_gpuA', 'registry_intervals', 'wall_sumo', 'wall_model',
            'team_source_wm', 'team_source_bb', 'fleet_ppo', 'cav_mappo',
            'gpuA_parity', 'gpuR_parity', 'cpu_tests', 'canary_repeat')


def admission(text, binding, evidence, *, phase='main'):
    include_phys = binding.get('include_phys') is True
    expected = budget(include_phys)
    if binding.get('protocol') != PROTOCOL or binding.get('sim_s') != expected:
        raise ValueError('protocol/per-arm budget binding mismatch')
    if binding.get('library') != list(CONTROLLERS) or binding.get('gpu_seconds') != {
            a: GPU_SECONDS[a] for a in expected}:
        raise ValueError('library/GPU budget binding mismatch')
    if binding.get('design_sha256') != sha(DESIGN):
        raise ValueError('design changed after binding')
    needed = REQUIRED + (('wm_phys_qualification', 'team_source_wm_phys') if include_phys else ())
    if phase == 'canary':
        needed = tuple(k for k in needed if k != 'canary_repeat')
    elif phase != 'main':
        raise ValueError('unregistered admission phase')
    missing = [key for key in needed if evidence.get(key, {}).get('status') != 'PASS'
               or evidence[key].get('binding_sha256') != canonical(binding)]
    line = approval(text)
    return dict(status='READY' if line and not missing else 'BLOCKED',
                dispatch_authorized=bool(line and not missing), approval_line=line,
                missing_gates=missing, new_sim_s=0, new_gpu_s=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--registration', default=REGISTRATION)
    p.add_argument('--binding')
    p.add_argument('--evidence')
    p.add_argument('--output')
    p.add_argument('--poll-seconds', type=float, default=60)
    p.add_argument('--polls', type=int, default=1)
    a = p.parse_args()
    if a.polls < 1 or a.poll_seconds <= 0:
        raise ValueError('positive bounded poll settings required')
    for index in range(a.polls):
        text = checked(a.registration).read_text(encoding='utf-8')
        if a.binding:
            if not a.evidence:
                raise ValueError('bound polling requires gate evidence')
            status = admission(text, json.loads(checked(a.binding).read_text()),
                               json.loads(checked(a.evidence).read_text()))
        else:
            status = dict(status='APPROVAL_PRESENT' if approval(text) else 'WAIT_MAIN_APPROVAL',
                          approval_line=approval(text), dispatch_authorized=False,
                          reason='polling alone cannot authorize an unbound runtime')
        print(json.dumps(status), flush=True)
        if a.output:
            write(checked(a.output), status)
        if status.get('dispatch_authorized') or index+1 == a.polls:
            break
        time.sleep(a.poll_seconds)


if __name__ == '__main__':
    main()
