"""S-LOOPLONG opt-in boundary hooks. No launches on import or config preparation."""
from __future__ import annotations

import copy
import math
from pathlib import Path

FLAGS = ('wm_loss', 'auto_horizon', 'long_credit', 'speed', 'hot_swap', 'exploration')
HORIZONS = (10, 30, 60, 90, 120, 180, 300)


def enabled(cfg, flag=None):
    section = cfg.get('looplong', {})
    return section.get('enabled') is True and (flag is None or section.get(flag) is True)


def validate(cfg, refusal):
    section = cfg.get('looplong', {})
    for name in ('enabled', *FLAGS):
        if name in section and type(section[name]) is not bool:
            raise refusal('looplong flags must be boolean')
    if not enabled(cfg):
        return
    if section.get('enable_cycle', 3) < 3:
        raise refusal('looplong enables only at cycle-2 -> 3 or later boundary')
    if enabled(cfg, 'auto_horizon'):
        for key in ('pairs300', 'threshold', 'threshold_sha256'):
            if not section.get(key):
                raise refusal('looplong requires registered ' + key)
    if enabled(cfg, 'hot_swap') and not (
            cfg.get('fastloop', {}).get('overlap_policy') and
            cfg.get('resumeprep', {}).get('enabled')):
        raise refusal('hot swap requires owned-child overlap coordinator')
    if enabled(cfg, 'long_credit') and not enabled(cfg, 'auto_horizon'):
        raise refusal('long credit requires registered auto horizon')


def active(cfg, state, flag):
    # FT/check of cycle2 prepares policy3; policy2 itself is never changed.
    return enabled(cfg, flag) and state['cycles'][-1]['cycle'] + 1 >= cfg['looplong'].get('enable_cycle', 3)


def wm_sets(cfg, state):
    if not active(cfg, state, 'wm_loss'):
        return []
    return ['loss.longwm_enabled=true', 'curriculum.unroll_s=[300.0]',
            'curriculum.truncation_steps_per_stage=[600]',
            'optim.batch_size_per_stage=[2]', 'loss.paired_diff.couple_randomness=true',
            'loss.paired_diff.horizon_mode="stage"']


def needs_pair_proof(cfg, state):
    return active(cfg, state, 'wm_loss') or active(cfg, state, 'long_credit')


def verify_pairs(loop, state, converted):
    path = converted.get('pairing_receipt')
    if not path or loop.sha256(path) != converted.get('pairing_receipt_sha256'):
        raise loop.Refused('looplong requires the current cycle paired proof/hash')
    proof = loop.read_json(path)
    if (proof.get('schema') != 'paircrit_pairs/1' or proof.get('split') != 'TRAIN' or
            proof.get('policy_checkpoint_sha256') != state['policy']['sha256']):
        raise loop.Refused('looplong pair proof policy/role mismatch')
    for key, field in (('returns', 'policy_sha256'), ('native_returns', 'native_sha256')):
        if not converted.get(key) or loop.sha256(converted[key]) != proof.get(field):
            raise loop.Refused('looplong paired return bytes changed')
    if not proof.get('pairs') or any(p.get('horizon_s') != 300 or not p.get('same_saved_start')
                                    or not p.get('observable_start_sha256') for p in proof['pairs']):
        raise loop.Refused('looplong needs proven common-start real 300s pairs')
    return converted['pairing_receipt_sha256']


def horizon_selection(doc, previous):
    threshold = doc['threshold']
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise ValueError('nonfinite registered horizon threshold')
    results = doc['horizons']
    short_ok = all(isinstance(results.get(str(h), {}).get('sign_agreement'), (int, float))
                   and results[str(h)]['sign_agreement'] >= threshold for h in (10, 30, 60))
    passing = [h for h in HORIZONS if h >= 60 and results.get(str(h), {}).get('verdict') == 'PASS']
    target = max(passing, default=10) if short_ok else 10
    # A regressing previous horizon is never grandfathered into the new WM.
    previous = min(previous, target)
    if previous > 10 and results.get(str(previous), {}).get('verdict') != 'PASS':
        previous = 10
    return short_ok, previous, target


def gate(loop, cfg, state, wm, out, *, previous=None):
    section = cfg['looplong']
    if loop.sha256(section['threshold']) != section['threshold_sha256']:
        raise loop.Refused('registered horizon threshold hash changed')
    h = loop.host(cfg, 'gpu')
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    argv = [h['python'], '-B', str(Path(h['code_root']) / 'tools/coopw/horizon_gate.py'),
            '--wm-ckpt', wm['checkpoint'], '--wm-config', wm['config'],
            '--pairs', section['pairs300'], '--threshold', section['threshold'],
            '--groups', str(section.get('groups', 96)), '--panel-seed', str(section.get('panel_seed', 20261026)),
            '--horizons', *map(str, HORIZONS), '--out', str(output)]
    env = {'PYTHONPATH': f"{h['code_root']}/src:{h['code_root']}", 'CUDA_VISIBLE_DEVICES': '',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'WMS_LOOPLONG_SPEED': '1' if enabled(cfg, 'speed') else '0'}
    code = loop.run_cmd(h, argv, env=env, log=output.with_suffix('.log'),
                        timeout_s=3600 * section.get('gate_timeout_h', 4.))
    if code or not output.is_file():
        raise RuntimeError('looplong horizon measurement failed; no verdict/promotion')
    doc = loop.read_json(output)
    if doc.get('schema') != 'coopw_horizon_gate/1' or doc.get('wm_sha256') != wm['sha256']:
        raise loop.Refused('horizon gate WM/schema mismatch')
    # The original script labels 10s BASE; compare its actual number as well.
    if doc.get('threshold') != loop.read_json(section['threshold'])['threshold']:
        raise loop.Refused('horizon gate threshold changed')
    previous = previous or state.get('looplong_policy_horizon_s',
                                    state.get('looplong_boundary', {}).get('horizon_s', 60))
    short_ok, previous, target = horizon_selection(doc, previous)
    return {'short_nonregression': short_ok, 'previous_horizon_s': previous, 'horizon_s': target,
            'gate_file': str(output), 'gate_sha256': loop.sha256(output), 'wm': dict(wm)}


def boundary_descriptor(loop, cfg, state, boundary):
    converted = next((cycle['phases']['convert']['receipt'] for cycle in reversed(state['cycles'])
                      if cycle['phases']['convert'].get('receipt')), {})
    source = boundary['wm'].get('source', '')
    generation = int(source.split()[-1]) if source.startswith('cycle ') else state['cycles'][-1]['cycle'] - 1
    return {**boundary, 'schema': 'looplong_boundary/1', 'accepted_after_cycle': generation,
        'flags': {flag: enabled(cfg, flag) for flag in FLAGS},
        'coop_credit': cfg['looplong'].get('coop_credit', True),
        'pairs': converted.get('pairs'), 'pairing_receipt': converted.get('pairing_receipt'),
        'policy_returns': converted.get('returns'), 'native_returns': converted.get('native_returns'),
        'source_policy_sha256': state['policy']['sha256'],
        'source_policy_checkpoint': state['policy']['checkpoint'],
        'threshold_sha256': cfg['looplong']['threshold_sha256']}


def check_candidate(loop, cfg, state, ft, out):
    wm = {'checkpoint': ft['candidate'], 'config': cfg['wm']['config'], 'sha256': ft['candidate_sha256'],
          'source': f"cycle {state['cycles'][-1]['cycle']}"}
    return gate(loop, cfg, state, wm, Path(out) / 'HORIZON_GATE.json')


def initial_boundary(loop, cfg, state, wm, root, k):
    args = cfg['policy'].get('extra_args', [])
    if '--horizon-gate' in args:
        path = args[args.index('--horizon-gate') + 1]
        doc = loop.read_json(path)
        if (doc.get('wm_sha256') == wm['sha256'] and
                doc.get('threshold') == loop.read_json(cfg['looplong']['threshold'])['threshold']):
            ok, previous, target = horizon_selection(doc, 60)
            if ok:
                return {'short_nonregression': True, 'previous_horizon_s': previous,
                        'horizon_s': target, 'gate_file': path, 'gate_sha256': loop.sha256(path), 'wm': dict(wm)}
    return gate(loop, cfg, state, wm, root / f'initial_gate_c{k:02d}.json')


def publish(loop, cfg, state, boundary, out):
    descriptor = boundary_descriptor(loop, cfg, state, boundary)
    descriptor['acceptance'] = str(Path(out) / 'ACCEPT.json')
    descriptor['acceptance_sha256'] = loop.sha256(descriptor['acceptance'])
    state['looplong_boundary'] = descriptor
    loop.append(Path(cfg['loop_root']) / 'cycle_ledger.jsonl', {
        'event': 'looplong_boundary', 'cycle': state['cycles'][-1]['cycle'],
        'wm_sha256': boundary['wm']['sha256'], 'horizon_s': boundary['horizon_s'],
        'gate_sha256': boundary['gate_sha256'], 'threshold_sha256': descriptor['threshold_sha256']})
    if active(cfg, state, 'hot_swap'):
        loop.write_atomic(Path(cfg['loop_root']) / 'WM_SWAP.json', descriptor)


def policy_args(loop, cfg, state, out):
    k = state['cycles'][-1]['cycle']
    if not enabled(cfg) or k < cfg['looplong'].get('enable_cycle', 3):
        return []
    boundary = state.get('looplong_boundary')
    if not boundary or boundary['wm']['sha256'] != state['accepted_wm']['sha256']:
        raise loop.Refused('looplong needs a boundary gate for the accepted WM')
    descriptor = copy.deepcopy(boundary)
    # Overlap uses this cycle's just-converted real cohort, even on the old WM.
    cohort = state.get('looplong_cohort')
    if cohort:
        descriptor.update(cohort)
    path = Path(out).with_suffix('.looplong.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    loop.write_atomic(path, descriptor)
    args = ['--looplong-boundary', str(path)]
    if enabled(cfg, 'hot_swap'):
        args += ['--wm-swap-file', str(Path(cfg['loop_root']) / 'WM_SWAP.json')]
    return args
