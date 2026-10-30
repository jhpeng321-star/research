"""S-INTERIM-A: prepare by default; --run is the explicit one-shot CPU simulation entry.

World identities live ONLY here. No training, checkpoint selection or global seed
registration is performed. Preparation imports no simulator and charges zero.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

SEEDS = tuple(range(977324, 977332))
CELLS = ((2400, .2), (2400, .4), (2700, .2), (2700, .4))
WARMUP_S, HORIZON_S, CAP_SIM_S = 300, 600, 14400


def checked(path):
    from wms.control.antiexploit import checked_path
    return checked_path(path)


def digest(path):
    return hashlib.sha256(checked(path).read_bytes()).hexdigest()


def plan():
    worlds = [dict(seed=seed, demand_vph=CELLS[i % len(CELLS)][0],
                   penetration=CELLS[i % len(CELLS)][1]) for i, seed in enumerate(SEEDS)]
    return dict(schema='S-INTERIM-A/1', worlds=worlds, arms=['ours', 'CTRL0'],
                warmup_s=WARMUP_S, horizon_s=HORIZON_S, cap_sim_s=CAP_SIM_S,
                episodes=len(worlds) * 2, population='textbook', device='cpu',
                execution='always_commanded_ctrl0/1', role='evaluation_only',
                allocation='two fresh worlds per fixed demand/penetration cell')


def plan_sha():
    return hashlib.sha256(json.dumps(plan(), sort_keys=True).encode()).hexdigest()


def registration_text(checkpoint_sha, audit_sha):
    return ('## S-INTERIM-A — prospective registration (owner-approved 2026-10-30 14:00)\n'
        f'Checkpoint SHA256: {checkpoint_sha}\nExposure audit SHA256: {audit_sha}\n'
        f'Script SHA256: {digest(__file__)}\nPlan SHA256: {plan_sha()}\n'
        f'Worlds and fixed cells: {json.dumps(plan()["worlds"], sort_keys=True)}\n'
        'Two arms: deterministic ours and CTRL0 (identical checkpoint/encoder, output residual exactly zero).\n'
        'Native warmup 300 s + intervention 600 s per arm/world; 16 episodes = 14,400 sim-s.\n'
        'Same mode-2 executor, native lanes, exogenous entrants and world seeds; textbook population.\n'
        'Append BEFORE running, after identity-only exposure audit confirms all eight worlds fresh.\n'
        'Primary: full-fleet-plus-source raw J(ours)-J(CTRL0), paired equal-world mean; lower is better.\n'
        'Report all eight pairs, paired-world bootstrap 95% CI (5000 draws, RNG 20261030), safety/drop counts.\n'
        'No checkpoint choice, training, tuning, seed replacement or automatic retry from these outcomes.\n'
        'Charge each attempted episode 900 sim-s before launch, failures retained within the cap.\n')


def validate_audit(path):
    receipt = json.loads(checked(path).read_text(encoding='utf-8'))
    if (receipt.get('schema') != 'interim_exposure_audit/1' or receipt.get('passed') is not True
            or receipt.get('seeds') != list(SEEDS) or receipt.get('fresh') is not True
            or receipt.get('excluded_from_training_and_selection') is not True
            or not receipt.get('surfaces') or receipt.get('plan_sha256') != plan_sha()):
        raise ValueError('missing fresh-world identity exposure audit bound to this fixed plan')
    # Audit surfaces are explicit identity-only provenance hashes. Do not open
    # their datasets or recursively scan evaluation/outcome directories here.
    for surface in receipt['surfaces']:
        checked(surface['path'])
        if len(surface.get('sha256', '')) != 64 or surface.get('identity_only') is not True:
            raise ValueError('audit surface must carry identity-only provenance and SHA256')
    return receipt


@contextmanager
def evaluation_cases():
    """Process-local exact-case authority; production/training seed guards stay intact."""
    import traffic_shepherd.c4.env as env
    import traffic_shepherd.scenarios as scenarios
    old_case, old_development = env.check_case, scenarios.require_development_case
    cases = {(w['seed'], w['demand_vph']) for w in plan()['worlds']}
    def check(seed, loading):
        if (seed, loading) not in cases:
            raise ValueError('outside registered S-INTERIM-A cases')
    def development(seed, loading, role):
        if role != 'exposed':
            raise ValueError('S-INTERIM-A only authorizes its explicit evaluation adapter')
        check(seed, loading)
    env.check_case, scenarios.require_development_case = check, development
    try:
        yield
    finally:
        env.check_case, scenarios.require_development_case = old_case, old_development


def observable_sha(state):
    import numpy as np
    h = hashlib.sha256()
    for key in ('s_m', 'v_mps', 'a_mps2', 'lane', 'present', 'is_cav', 'length_m'):
        value = state[key]
        a = np.ascontiguousarray(value.detach().cpu().numpy() if hasattr(value, 'detach') else value)
        h.update(key.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def run(checkpoint, output, audit, registration, driver, parity_receipt):
    """Explicit future operator entry; never invoked by preparation or CPU tests."""
    validate_audit(audit)
    text = registration_text(digest(checkpoint), digest(audit))
    if text not in checked(registration).read_text(encoding='utf-8'):
        raise ValueError('append the exact hash-bound registration text before running')
    os.environ.update(CUDA_VISIBLE_DEVICES='', C4_EXECUTION_PARITY='2', C4_SUMO_LEADER_RULE='1',
        C4_EXECUTION_DRIVER=str(checked(driver)), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1')
    from wms.control.execution import verify_parity, check_contract
    from wms.control.policy import load_policy_components, PolicyExecutor
    verify_parity(checked(parity_receipt))
    policy, encoder, rules, payload = load_policy_components(checked(checkpoint), device='cpu')
    check_contract(payload)
    if not policy.residual_on_native or rules is not None:
        raise ValueError('interim requires a native-lane residual checkpoint')
    import copy
    import numpy as np
    import torch
    ctrl0 = copy.deepcopy(policy)
    with torch.no_grad():
        ctrl0.residual_head.weight.zero_()
        ctrl0.residual_head.bias.zero_()
    actors = {'ours': PolicyExecutor(policy, encoder, deterministic=True, device='cpu'),
              'CTRL0': PolicyExecutor(ctrl0, copy.deepcopy(encoder), deterministic=True, device='cpu')}
    # Diagnostic evaluator constructs components directly: it does not promote
    # the candidate through the deployment/native-acceptance selection gate.
    output = checked(output)
    if not output.parent.is_dir():
        raise ValueError('output parent must already exist')
    output.mkdir()  # Exclusive one-shot output. No implicit retries or overwrite.
    (output / 'REGISTRATION.md').write_text(text, encoding='utf-8')
    (output / 'PLAN.json').write_text(json.dumps(dict(plan(), checkpoint_sha256=digest(checkpoint),
        exposure_audit_sha256=digest(audit), driver_sha256=digest(driver),
        parity_receipt_sha256=digest(parity_receipt)), indent=2), encoding='utf-8')
    from traffic_shepherd.c4.env import CavSumoEnv
    from traffic_shepherd.c4.baselines import run_policy_episode
    rows, charged = [], 0
    with evaluation_cases():
        for world in plan()['worlds']:
            hashes = []
            for arm, actor in actors.items():
                charged += WARMUP_S + HORIZON_S
                if charged > CAP_SIM_S:
                    raise ValueError('interim simulation budget exceeded')
                with (output / 'budget.jsonl').open('a', encoding='utf-8') as ledger:
                    ledger.write(json.dumps(dict(world=world, arm=arm, charged_sim_s=900,
                                                 cumulative_sim_s=charged)) + '\n')
                    ledger.flush()
                    os.fsync(ledger.fileno())
                initial = []
                actor.reset()
                def controller(state):
                    if not initial:
                        initial.append(observable_sha(state))
                    return actor(state)
                env = CavSumoEnv(population='textbook', device='cpu', physics_log=False,
                                 scenario_root=output / 'scenarios', entry_queue_in_state=True)
                try:
                    result = run_policy_episode(env, controller, seed=world['seed'],
                        demand_vph=world['demand_vph'], penetration=world['penetration'],
                        warmup_s=WARMUP_S, horizon_s=HORIZON_S,
                        episode_dir=output / f"w{world['seed']}_{arm}", meta=dict(task='S-INTERIM-A', arm=arm))
                finally:
                    env.close()
                hashes.append(initial[0])
                window = result['window']
                row = dict(world=world, arm=arm, initial_observable_sha256=initial[0],
                    J=window['speed_deficit_vehicle_seconds'] + window['source_wait_vehicle_seconds'],
                    summary=result)
                rows.append(row)
                with (output / 'results.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
            if hashes[0] != hashes[1]:
                raise ValueError('paired warmup initial state mismatch; retain both receipts')
    differences = np.array([rows[i]['J'] - rows[i + 1]['J'] for i in range(0, len(rows), 2)])
    boot = np.random.default_rng(20261030).choice(differences, (5000, len(differences))).mean(1)
    receipt = dict(status='COMPLETE', charged_sim_s=charged, n_worlds=len(differences),
        raw_dJ_ours_minus_CTRL0=differences.tolist(), mean_raw_dJ=float(differences.mean()),
        paired_world_bootstrap_ci95=np.quantile(boot, [.025, .975]).tolist(), selection_authorized=False,
        full_fleet_accounting_valid=all(r['summary'].get('dropped_vehicle_steps') == 0 for r in rows),
        safety_and_drop_receipts=[dict(world=r['world'], arm=r['arm'],
            **{key: r['summary'].get(key) for key in
               ('collisions', 'teleports', 'dropped_vehicle_steps', 'dropped_cav_steps')}) for r in rows])
    (output / 'INTERIM_RESULT.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=checked)
    parser.add_argument('--exposure-audit', type=checked)
    parser.add_argument('--registration', type=checked)
    parser.add_argument('--output', type=checked)
    parser.add_argument('--driver', type=checked)
    parser.add_argument('--parity-receipt', type=checked)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    if not args.run:
        audit_sha = digest(args.exposure_audit) if args.exposure_audit else '<EXPOSURE_AUDIT_SHA256>'
        print(registration_text(digest(args.checkpoint), audit_sha))
        return 0
    if any(getattr(args, k) is None for k in ('exposure_audit', 'registration', 'output', 'driver', 'parity_receipt')):
        parser.error('--run requires exposure-audit, registration, output, driver and parity-receipt')
    print(json.dumps(run(args.checkpoint, args.output, args.exposure_audit,
                         args.registration, args.driver, args.parity_receipt)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
