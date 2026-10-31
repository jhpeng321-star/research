"""Approval parser and CPU-only polling: budget authority is never self-issued."""
import argparse
import json
import re
import time

from tools.c1switch.design import checked, sha
from tools.c1switch.storage import canonical, write

from .design import CONTROLLERS, CPU_ARMS, DESIGN, GPU_SECONDS, PROTOCOL, REGISTRATION, budget


def approval(text):
    lines = []
    for line in text.splitlines():
        # Require an actual MAIN-authored approval entry. Quoted requests, future
        # conditions, our design line, and S-C1SWITCH's old approval do not match.
        main = re.match(r'^-\s+(?:\d\d:\d\d\s+)?MAIN\b[^\n]*\bS-C1SWITCH2 BUDGET APPROVED\b', line)
        owner = re.match(r'^-\s+\d\d:\d\d\s+S-C1SWITCH2 BUDGET APPROVED\s+\(owner,\s*chat:', line)
        if main or owner:
            if not re.search(r'\b(?:NOT|NO|AWAIT|PENDING|REVOKED|EXAMPLE)\b', line):
                lines.append(line)
    return lines[-1] if lines else None


REQUIRED = ('frozen_v7_sources', 'seed_guards', 'exposure_cpuQ', 'exposure_gpuR',
            'exposure_gpuA', 'registry_intervals', 'wall_sumo', 'wall_model',
            'inference_wm', 'inference_bb', 'fleet_ppo', 'cav_mappo',
            'gpuA_parity', 'gpuR_parity', 'cpu_tests', 'canary_repeat')
CPU_REQUIRED = tuple(k for k in REQUIRED if k not in
                     ('inference_wm', 'inference_bb', 'fleet_ppo', 'cav_mappo',
                      'gpuA_parity', 'gpuR_parity'))


def cpu_authority(text):
    return any(re.match(r'^-\s+01:48\s+MAIN\b', line) and
               'S-C1SWITCH2: do not wait for RL checkpoints' in line and
               'start the CPU-only arms now (6 fixed controllers + random)' in line
               for line in text.splitlines())


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
    if binding.get('prediction_objective') == 'on_road_fleet_delay':
        if not any('S-C1SWITCH2 DESIGN REGISTERED ' in line and
                   binding['design_sha256'] in line and 'prediction_objective=on_road_fleet_delay' in line
                   for line in text.splitlines()):
            raise ValueError('road-only objective amendment not registered')
    needed = REQUIRED + (('wm_phys_qualification', 'inference_wm_phys') if include_phys else ())
    if binding.get('dispatch_phase') == 'cpu' and phase == 'main':
        if binding.get('enabled_arms') != list(CPU_ARMS) or include_phys or not cpu_authority(text):
            raise ValueError('CPU subset requires MAIN 01:48 authority and exact seven-arm binding')
        needed = CPU_REQUIRED
    elif binding.get('dispatch_phase') == 'model' and phase == 'main':
        arms = binding.get('enabled_arms')
        if arms not in (['wm'], ['bb']) or include_phys or not cpu_authority(text):
            raise ValueError('model subset requires MAIN 01:48 authority and one frozen model arm')
        needed = CPU_REQUIRED + ('inference_'+arms[0], 'gpuA_parity', 'gpuR_parity')
    elif phase == 'main' and 'enabled_arms' in binding:
        raise ValueError('unregistered partial-arm admission')
    if phase == 'canary':
        # No model or selector is used by the already budgeted mechanical canary.
        needed = tuple(k for k in REQUIRED if k not in
                       ('inference_wm', 'inference_bb', 'fleet_ppo', 'cav_mappo',
                        'gpuA_parity', 'gpuR_parity', 'canary_repeat'))
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
