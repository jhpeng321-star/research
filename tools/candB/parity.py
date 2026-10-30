"""Registered S-CAPREL-PARITY only: four serial training-seed SUMO episodes.

Budget is precharged BEFORE reset, including failures; no automatic retries.
Replay reads only public episode arrays and the executed-command ledger.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

JOBS = (
    dict(id='s963110_ctrl0_base', seed=963110, reference='base', residual=0.),
    dict(id='s963110_ctrl0_caprel', seed=963110, reference='ctrl0', residual=0.),
    dict(id='s963111_caprel_probe', seed=963111, reference='ctrl0', residual=-.01),
    dict(id='s963112_caprel_probe', seed=963112, reference='ctrl0', residual=-.01),
)
BOUND = .1603
CAP = 3600


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=1, allow_nan=False)
        stream.write('\n')


def law_sources():
    return {name: sha(Path('src/wms/control')/name)
            for name in ('residual.py', 'alwayscmd.py', 'sumo_idm.py')}


def validate_registration(doc):
    if (doc.get('allocation') != 'S-CAPREL-PARITY' or doc.get('host') != 'gpuR'
            or doc.get('cap_sim_s') != CAP or doc.get('pooled_rmse_bound') != BOUND
            or doc.get('jobs') != list(JOBS) or not doc.get('registration_sha256')):
        raise ValueError('registration does not match the fixed training-seed allocation')
    if doc.get('source_sha256') != sha(__file__):
        raise ValueError('registered parity worker changed')
    if doc.get('caprel_source_sha256') != law_sources():
        raise ValueError('registered caprel law changed')


def identity(root):
    # Compare scientific arrays bitwise, excluding only descriptive metadata,
    # requested residual-law labels, command ledgers and wall-clock timings.
    names = ('time_s', 's_m', 'v_mps', 'a_mps2', 'lane', 'lane_carry', 'present',
             'is_cav', 'junction', 'vid_index', 'vehicle_ids', 'control_time_s',
             'controller', 'issued_acceleration_mps2', 'issued_target_speed_mps')
    paths = [root/job['id']/'episode.npz' for job in JOBS[:2]]
    hashes = []
    for path in paths:
        with np.load(path, allow_pickle=False) as tape:
            hashes.append({name: hashlib.sha256(tape[name].tobytes()).hexdigest() for name in names})
    return dict(bit_identical=hashes[0] == hashes[1], array_sha256=hashes,
                compared_arrays=list(names), residual_reference='ctrl0',
                caprel_source_sha256=law_sources())


def replay(path):
    from tools.execparity.replay import states_from_episode
    from wms.control.alwayscmd import executed
    from wms.control.native_cav import head_native_accel
    with np.load(path/'episode.npz', allow_pickle=False) as source:
        tape = dict(source)
    at = states_from_episode(tape)
    ledger = {}
    with gzip.open(path/'run/residual_execution.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            ledger.setdefault(row['timestamp_s'], []).append(row)
    errors, heads, actual = [], [], []
    excluded = dict(not_stable=0, speed_mismatch=0, missing_id=0)
    for time, rows in sorted(ledger.items()):
        i = int(np.searchsorted(tape['time_s'], time))
        if i+1 >= len(tape['time_s']):
            continue
        if abs(float(tape['time_s'][i])-time) > 1e-6:
            raise ValueError('physics ledger/tape grid mismatch')
        state = at(i)
        keep = torch.nonzero(state['present'][0]).flatten()
        state = {key: value.index_select(1, keep) for key, value in state.items()}
        with torch.no_grad():
            prediction = executed(state, torch.full_like(state['v_mps'], -.01),
                                  residual_reference='ctrl0')[0].numpy()
            native = head_native_accel(state)[0].numpy()
        ids = tape['vehicle_ids'][tape['vid_index'][i]]
        lookup = {str(ids[int(k)]): (local, int(k)) for local, k in enumerate(keep)}
        for row in rows:
            if row['vehicle_id'] not in lookup:
                excluded['missing_id'] += 1
                continue
            local, k = lookup[row['vehicle_id']]
            if not (tape['is_cav'][i, k] and tape['lane'][i, k] >= 0
                    and tape['present'][i+1, k]
                    and tape['vid_index'][i+1, k] == tape['vid_index'][i, k]):
                excluded['not_stable'] += 1
                continue
            v0 = float(tape['v_mps'][i, k])
            if abs(v0-row['speed_before_mps']) > 1e-6:
                excluded['speed_mismatch'] += 1
                continue
            realised = (float(tape['v_mps'][i+1, k])-v0)/.5
            errors.append(float(prediction[local])-realised)
            heads.append(float(native[local])-realised)
            actual.append(realised)
    if not errors or not np.isfinite([errors, heads, actual]).all():
        raise ValueError('missing/nonfinite stable executed rows')
    result = dict(n=len(errors), excluded=excluded,
        rmse=float(np.sqrt(np.mean(np.square(errors)))),
        native_head_rmse=float(np.sqrt(np.mean(np.square(heads)))),
        episode_sha256=sha(path/'episode.npz'),
        ledger_sha256=sha(path/'run/residual_execution.jsonl.gz'))
    return errors, heads, result


def run(root, registration):
    from traffic_shepherd.c4.env import CavSumoEnv
    from wms.control.nativeinit import NativeResidualCommand
    if os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise ValueError('S-CAPREL-PARITY must run nice 19')
    if importlib.metadata.version('eclipse-sumo') != '1.27.1':
        raise ValueError('registered eclipse-sumo 1.27.1 is required')
    validate_registration(registration)
    manifest = json.loads(Path('SOURCE_MANIFEST.json').read_text())
    if not manifest or any(sha(name) != digest for name, digest in manifest.items()):
        raise ValueError('staged source manifest mismatch')
    if root.exists() or not root.parent.is_dir():
        raise ValueError('parity output root must be fresh')
    root.mkdir()
    torch.set_num_threads(1)
    from tools.distill.runtime import configure_execution
    configure_execution('configs/execparity_cav_idm_v1.json')
    binary = str(Path(importlib.util.find_spec('sumo').origin).parent/'bin/sumo')
    if not Path(binary).is_file():
        raise ValueError('registered SUMO binary missing')
    write(root/'START.json', dict(pid=os.getpid(), nice=19, workers=1,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        registration=registration, sumo_version=subprocess.check_output([binary, '--version'], text=True)))
    charged = 0
    for index, job in enumerate(JOBS):
        if index == 2:
            proof = identity(root)
            write(root/'CTRL0_IDENTITY.json', proof)
            if not proof['bit_identical']:
                raise ValueError('fresh same-training-seed zero identity failed; stop before probes')
        charged += 900
        if charged > CAP:
            raise ValueError('parity budget exhausted')
        with (root/'budget.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(job=job, charged_sim_s=900, cumulative_sim_s=charged))+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        env = CavSumoEnv(controller='direct', population='textbook', k_slots=512,
                         physics_log=False, sumo_binary=binary, direct_accel_max_mps2=1.2)
        try:
            state = env.reset(seed=job['seed'], demand_vph=2400, penetration=.2,
                warmup_s=300, horizon_s=600, episode_dir=root/job['id'],
                meta=dict(allocation='S-CAPREL-PARITY', residual_reference=job['reference']))
            while not env.done:
                command = NativeResidualCommand(np.full_like(state['v_mps'].cpu().numpy(),
                    job['residual']), -3., 1.2, job['reference'])
                state, info = env.step(command)
            write(root/(job['id']+'_RESULT.json'), info['summary'])
            print('COMPLETE', job['id'], charged, flush=True)
        finally:
            env.close()
    errors, heads, worlds = [], [], {}
    for job in JOBS[2:]:
        e, h, row = replay(root/job['id'])
        errors.extend(e)
        heads.extend(h)
        worlds[str(job['seed'])] = row
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    head_rmse = float(np.sqrt(np.mean(np.square(heads))))
    report = dict(schema='candB_caprel_parity/1', allocation='S-CAPREL-PARITY',
        pooled_rmse=rmse, registered_bound=BOUND, passed=rmse <= BOUND,
        zero_ctrl0_identity=True, worlds=worlds, charged_sim_s=charged,
        actual_sim_s=3600, workers=1, nice=19, caprel_source_sha256=law_sources(),
        registration_sha256=registration['registration_sha256'])
    write(root/'REPORT.json', report)
    # Preserve the existing executor contract's additional same-state head-band
    # check. A fixed-bound PASS alone is not silently promoted past that guard.
    if report['passed'] and rmse <= head_rmse:
        write(root/'CAPREL_PARITY.json', dict(schema='execparity_always_commanded_acceleration/1',
            backend_contract='always_commanded_ctrl0/1', measured_real_sumo=True,
            zero_ctrl0_identity=True, passed=True, residual_reference='ctrl0',
            caprel_source_sha256=law_sources(), worlds=list(worlds),
            driver_sha256=sha('configs/execparity_cav_idm_v1.json'),
            native_arm_head_error=dict(rmse=head_rmse), c1_residual_execution_error=dict(rmse=rmse),
            error=dict(ours=dict(rmse=rmse)), registered_bound=BOUND,
            allocation='S-CAPREL-PARITY', report_sha256=sha(root/'REPORT.json')))
    print(json.dumps(report, indent=1), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    run(a.out, json.loads(a.registration.read_text()))


if __name__ == '__main__':
    main()
