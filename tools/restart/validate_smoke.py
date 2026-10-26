"""Verify only registered mechanics artifacts/observable arrays; zero simulator calls."""
from __future__ import annotations
import copy
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

RELEASE = Path('/root/autodl-tmp/c4/restart20261026')
ROOT = RELEASE/'runtime'
ARM = RELEASE/'arm'
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  C4_ALLOWED_ROOTS=str(RELEASE)+':'+str(RELEASE.with_name(RELEASE.name+'_inputs')))
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts/c4')]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import loop14 as loop  # noqa: E402
from tools.fastloop.world import verify  # noqa: E402
from wms.control.policy import load_policy_components, PolicyExecutor  # noqa: E402
from wms.control.nativeinit import sumo_residual_commands, verify_collection_gate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = loop.read_json(ARM/'CONFIG.json')
    main_state = loop.load_state(cfg)
    mini = Path(cfg.get('resumeprep', {}).get('mini_root', ARM/'mini'))
    state = loop.read_json(mini/'loop/LOOP14_STATE.json')
    phases = state['cycles'][0]['phases']
    complete = all(phases[p]['status'] == 'done' for p in ('policy', 'collect', 'convert', 'wm_ft'))
    if not complete and not args.observations:
        raise RuntimeError('mini phases incomplete; no mechanics PASS')
    expected_total = 87399 + 1922 * (1 + cfg.get('resumeprep', {}).get('smoke_setup_retry', {}).get('retry_number', 0))
    if state['sim_s_charged'] != 1922 or main_state['sim_s_charged'] != expected_total:
        raise RuntimeError('registered smoke accounting mismatch')
    gate = verify_collection_gate(state['policy'])
    training = loop.read_json(Path(phases['policy']['receipt']['out'])/'RUN_SUMMARY.json')
    initial = training['native_curve'][0]
    if initial['update'] != 0 or any(x != 0.0 for x in initial['delta_per_start']):
        raise RuntimeError('scientific update-zero native imagination identity failed')
    policy, encoder, rules, _ = load_policy_components(state['policy']['checkpoint'])
    zero = copy.deepcopy(policy)
    with torch.no_grad():
        zero.residual_head.weight.zero_()
        zero.residual_head.bias.zero_()
    rows, zero_count = [], 0
    for world in phases['collect']['receipt']['worlds']:
        artifact = Path(world['artifact'])
        verify(artifact)
        manifest = loop.read_json(artifact/'pairs/C4_EPISODES_MANIFEST.json')
        paired = {}
        for entry in manifest['episodes']:
            if entry['seed'] not in (968142, 968143):
                raise RuntimeError('unregistered mechanics world')
            kind = entry.get('branch', {}).get('kind')
            if kind not in ('B', 'POLICY'):
                raise RuntimeError('unexpected extra mechanics arm')
            directory = artifact/'pairs'/entry['dir']
            def array(name):
                return np.load(directory/(name+'.npy'), mmap_mode='r', allow_pickle=False)
            kin, present, cav, vid, time = (array(n) for n in ('veh_kin', 'veh_mask', 'is_cav', 'vid', 'time_s'))
            control_time = array('control_time_s')
            start = float(entry['warmup_s'])
            lane = np.where(kin[..., 1:4].sum(-1) > 0, kin[..., 1:4].argmax(-1), -1)
            mask = present.astype(bool) & cav.astype(bool) & (lane >= 0) & (time[:, None] > start)
            speed = kin[..., 4][mask]
            same = (present[1:].astype(bool) & present[:-1].astype(bool) & cav[:-1].astype(bool)
                    & (vid[1:] == vid[:-1]) & (vid[1:] >= 0) & (time[1:, None] > start))
            changes = same & (lane[1:] != lane[:-1]) & (lane[1:] >= 0) & (lane[:-1] >= 0)
            lane_requests = int((array('lane_request_target')[control_time >= start] >= 0).sum())
            lane_commands = int((array('lane_command_target')[control_time >= start] >= 0).sum())
            summary = loop.read_json(artifact/entry['source'].removeprefix('@artifact/')/'summary.json')
            if (not speed.size or not np.isfinite(speed).all() or lane_requests or lane_commands
                    or summary.get('collisions', 0) or summary.get('teleports', 0)
                    or summary.get('dropped_vehicle_steps', 0)):
                raise RuntimeError('scientific mechanics behavior failure: lanes/incidents/drops/nonfinite')
            row = {'seed': entry['seed'], 'arm': kind, 'cav_share': entry['penetration'],
                   'mean_cav_speed_mps': float(speed.mean()), 'lane_changes': int(changes.sum()),
                   'ramp_merges': int((changes & (lane[:-1] == 2) & (lane[1:] == 1)).sum()),
                   'policy_lane_requests': lane_requests, 'policy_lane_commands': lane_commands,
                   'J_vehicle_seconds': summary['ledger']['J_vehicle_seconds'],
                   'source_wait_vehicle_seconds': summary['ledger']['source_wait_vehicle_seconds'],
                   'arrivals': summary['ledger']['arrivals'],
                   'collisions': summary.get('collisions', 0), 'teleports': summary.get('teleports', 0)}
            rows.append(row)
            t0 = int(np.searchsorted(time, start))
            paired[kind] = {'kin': np.asarray(kin[t0]).copy(), 'present': np.asarray(present[t0]).copy(),
                            'vid': np.asarray(vid[t0]).copy(), 'summary': summary, 'row': row}
            if kind == 'B':
                # Exact command-level replay of the fresh zero-head executor on
                # real native observations. This is NOT an extra SUMO branch or
                # a claim that the trained u10 trajectory must equal native.
                executor = PolicyExecutor(zero, encoder, lane_rules=rules, deterministic=True)
                for tick in control_time[control_time >= start]:
                    i = int(np.searchsorted(time, tick))
                    st = {'s_m': torch.tensor(np.asarray(kin[i:i+1, :, 0])),
                          'v_mps': torch.tensor(np.asarray(kin[i:i+1, :, 4])),
                          'a_mps2': torch.tensor(np.asarray(kin[i:i+1, :, 5])),
                          'length_m': torch.tensor(np.asarray(kin[i:i+1, :, 6])),
                          'lane': torch.tensor(lane[i:i+1]),
                          'present': torch.tensor(np.asarray(present[i:i+1]), dtype=torch.bool),
                          'is_cav': torch.tensor(np.asarray(cav[i:i+1]), dtype=torch.bool)}
                    command, lanes = executor(st, with_lanes=True)
                    finite = np.isfinite(command.residual)
                    if lanes is not None or np.any(command.residual[finite] != 0):
                        raise RuntimeError('scientific native initialization command identity failed')
                    live = (st['present'] & st['is_cav'])[0].numpy()
                    residuals = {str(j): float(v) for j, v in enumerate(command.residual[0]) if live[j]}
                    if sumo_residual_commands(object(), residuals, command.accel_min, command.accel_max):
                        raise RuntimeError('zero initialization added SUMO commands')
                    zero_count += int(live.sum())
        if set(paired) != {'B', 'POLICY'}:
            raise RuntimeError('incomplete mechanics pair')
        if any(not np.array_equal(paired['B'][k], paired['POLICY'][k]) for k in ('kin', 'present', 'vid')):
            raise RuntimeError('scientific mechanics saved-state pairing failed')
    result = {'status': 'OBSERVATIONS_ONLY' if args.observations else 'PASS',
              'mini_complete': complete, 'registration': 'S-C0DIAG-SMOKE-1', 'charged_sim_s': 1922,
              'total_charged_sim_s': main_state['sim_s_charged'], 'rows': rows,
              'wm_update_zero_native_identity_starts': len(initial['delta_per_start']),
              'rejected_checkpoint_blocked': loop.read_json(ARM/'READINESS.json')['rejected_checkpoint_blocked'],
              'zero_init_observable_replay_native_identical': True, 'zero_init_replayed_cav_commands': zero_count,
              'zero_init_additional_sumo_branches': 0,
              'trajectory_scope': 'native branch observed speeds/lanes/merges; exact zero-head executor abstention replay; real POLICY is trained update10',
              'native_gate': {'checkpoint_sha256': gate['checkpoint_sha256'], 'mean_delta': gate['mean_delta']},
              'efficacy_certification': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    loop.write_atomic(ARM/('SMOKE_OBSERVATIONS.json' if args.observations else 'SMOKE_RESULT.json'), result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
