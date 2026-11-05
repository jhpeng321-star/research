"""Explicit operator entrypoint; importing this module never boots or dispatches.

Usage on the GPU, ONLY when MAIN elects to run it:
 python -B -m tools.mining.wm_prescreen --worker-root FRESH_STAGED_ROOT \
   --query CAPTURED_QUERY.json --manifest SOURCE_AND_ASSET_HASHES.json

The worker root/overlay/pending GPU_BINDING.json must be staged by MAIN. This
program never stages, spawns a service, invokes SSH, starts SUMO, or charges.
It runs the original eight gates in this very foreground process, then a fixed
longitudinal grid. G5 candidates explicitly refuse BEFORE bootstrap.
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from unittest.mock import patch

from .wm_contract import (HORIZONS, LONG_FAMILIES, U0_SHA, check_sources, checked_path,
                          digest, file_sha, grid, read_json, validate_query)
from .wm_adapter import dynamic_executor, require_longitudinal

ASSET_PINS = {
    '/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt':
        'b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436',
    '/root/autodl-tmp/c4/speedhead20261030/mappospeed_final169.pt':
        'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca',
    '/root/autodl-tmp/c4/setupaudit20261030/policy_000190.pt':
        'de235a787024bb0f79d43937132eefcea69267b38205bc020f95bc98d4e8c7d1',
    '/root/wmcalib/expected_hashes.json':
        '1bc35cd92c2ebbfb4726566e6362201fda37791890a1643d81e7bb4de647973f',
    '/root/wmcalib/data/design.json':
        '09394b100f82a8e8bde7dbff131fc222f71807250086b15b34f9e19506047ba5',
}


def require_bound(path, manifest):
    p = checked_path(path).resolve()
    found = [k for k in manifest if checked_path(k).resolve() == p]
    if len(found) != 1 or file_sha(p) != manifest[found[0]]:
        raise ValueError('runtime dependency absent from exact source manifest: ' + str(p))


def qualify(root):
    """Original gates, original bootstrap and same-process qualification seals."""
    from tools.coopmwm import warm_service, plan_service, fresh_worker
    from tools.coopmwm.hotadd import GATES
    original = plan_service.configured_runtime
    cached = []
    def configured(bootstrap, fraction):
        if not cached:
            cached.append(original(bootstrap, fraction))
        return cached[0]
    # warm_service runs all eight original gates and hands the SAME rt to
    # plan_service. Empty stdin returns after qualification, no request service.
    with patch.object(plan_service, 'configured_runtime', configured), patch.object(sys, 'stdin', []):
        warm_service.main(root)
    receipt = read_json(root / fresh_worker.RECEIPT)
    if not receipt['pass_'] or receipt['pid'] != os.getpid():
        raise ValueError('same-process eight-gate qualification missing')
    for name in GATES:
        gate = read_json(root / name)
        if gate.get('pid') != os.getpid() or gate.get('pass_') is not True or gate.get('max_cost_error') != 0.:
            raise ValueError('nonexact startup gate: ' + name)
    return cached[0], receipt


def load_base(query):
    import torch
    b = query['base_continuation']
    if file_sha(b['checkpoint']) != U0_SHA:
        raise ValueError('frozen u0 checkpoint hash mismatch')
    saved = torch.load(checked_path(b['checkpoint']), map_location='cpu', weights_only=False)
    if saved.get('update') != 0 or not isinstance(saved.get('policy'), dict):
        raise ValueError('not an original u0 policy checkpoint')
    weights = saved['policy']
    if not all(k in weights and not bool(weights[k].count_nonzero())
               for k in ('actor.4.weight', 'actor.4.bias')):
        raise ValueError('nonzero actor: exact observable V2V forward adapter required')
    return weights


def validate_scores(scores, actions):
    ids = [a['id'] for a in actions]
    for horizon in HORIZONS:
        values = scores[str(horizon)]
        if set(values) != set(ids):
            raise ValueError('incomplete candidate scores')
        expected = [v['draw_id'] for v in values['policy']]
        if len(expected) != 8 or len(set(expected)) != 8:
            raise ValueError('complete indexed K8 required')
        for key in ids:
            if [v['draw_id'] for v in values[key]] != expected:
                raise ValueError('candidate CRN draw mismatch')
            if any(not math.isfinite(v['team_delay']) for v in values[key]):
                raise ValueError('nonfinite WM cost')


def rankings(scores, actions):
    validate_scores(scores, actions)
    def mean(values):
        return sum(v['team_delay'] for v in values) / len(values)
    out = []
    for action in actions:
        if 'wm_move' not in action or 'solo_of' in action:
            continue
        row = dict(action=action, horizons={})
        for h in HORIZONS:
            costs = scores[str(h)]; joint = costs[action['id']]
            solo_ids = [a['id'] for a in actions if a.get('solo_of') == action['id']]
            paired = [min(costs[s][k]['team_delay'] for s in solo_ids) - joint[k]['team_delay']
                      for k in range(8)] if solo_ids else None
            row['horizons'][str(h)] = dict(mean_team_delay=mean(joint),
                saving_vs_policy=mean(costs['policy']) - mean(joint),
                saving_vs_MAPPO=mean(costs['MAPPO']) - mean(joint),
                paired_best_solo_minus_joint=paired,
                mean_synergy=sum(paired) / 8 if paired is not None else None)
        out.append(row)
    return sorted(out, key=lambda x: (x['horizons']['300']['mean_team_delay'], x['action']['id']))


def score_grid(rt, query, actions, weights):
    """K8/H600 full-roster reference execution; no unqualified speed2 shortcut."""
    require_longitudinal(actions)
    from tools.coopmwm.plan_wm import evaluate
    from tools.coopmwm.plan_service import request_memory
    record = copy.deepcopy(query['record'])
    # Validated explicit history (including []); use the original warm adapter.
    memory = request_memory(rt, record, {'memory_mode': 'warm'})
    stats = {'active_member_seconds': 0}
    with dynamic_executor(rt, query, weights, telemetry=stats):
        # Every action has the same virtual future IDs and source sampler. No
        # trimming/padding/speed optimization introduced by this new decoder.
        scores = evaluate(rt, record, actions, 8, 600, chunk_actions=1, trim=False,
                          pad_actions=False, horizons=HORIZONS, memory=memory,
                          speed2=False, transfers=False)
    return scores, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker-root', required=True)
    ap.add_argument('--query', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--families', nargs='+', default=list(LONG_FAMILIES))
    a = ap.parse_args(argv)
    root = checked_path(a.worker_root)
    if not root.is_dir():
        raise ValueError('MAIN must stage a fresh source-bound worker root')
    query = validate_query(read_json(a.query))
    actions, rejected = grid(query, tuple(a.families))
    require_longitudinal(actions)  # Refuse G5 before any model or gate execution.
    if set(a.families) & {'yield_lane', 'open_gap_merge'}:
        from .wm_adapter import UnsupportedLaneABI
        raise UnsupportedLaneABI('G5 requested; no qualified lane injection ABI, even if geometry unavailable')
    manifest = read_json(a.manifest)
    check_sources(manifest)
    check_sources(ASSET_PINS)
    require_bound(a.query, manifest)
    for module in (sys.modules[__name__], sys.modules['tools.mining.wm_adapter'],
                   sys.modules['tools.mining.wm_contract']):
        require_bound(module.__file__, manifest)
    from tools.wmteach import runtime, scale
    from tools.coopmwm import plan_wm, wm, acceleration, moves
    from tools.coopmwm import warm_service, plan_service, fresh_worker
    for module in (runtime, scale, plan_wm, wm, acceleration, moves,
                   warm_service, plan_service, fresh_worker):
        require_bound(module.__file__, manifest)
    if (runtime.BACKEND != 'FULL-v4'
            or str(runtime.PHYS) != '/root/rbeval20261031/phys_code_22cc32f65a80'
            or file_sha(scale.__file__) != '98c63848a054cf2ec99f57acd3cdbb6975217317e80e0ec142827007741f698c'):
        raise ValueError('original FULL-v4 bootstrap/scale binding required')
    for rel in ('src/wms/control/bridge.py', 'src/wms/control/lanes.py',
                'tools/rbint/recipe.py', 'tools/rbeval/references.json', 'configs/wm_phys.toml'):
        require_bound(runtime.PHYS / rel, manifest)
    binding = read_json(root / 'GPU_BINDING.json')
    if binding.get('qualification', {}).get('mode') != 'fresh_startup_v1':
        raise ValueError('fresh pending binding required; no adopted startup receipts')
    require_bound(root / 'GPU_BINDING.json', manifest)
    require_bound(root / 'FROZEN_COSTS.json', manifest)
    if file_sha(root / 'FROZEN_COSTS.json') != '37e17257ca5c5fd1eddbd8db0770594fa06b88e95fbbd6f74430d2c115ac9bb5':
        raise ValueError('original frozen team-cost source changed')
    if time.time() >= binding.get('deadline_unix', 0):
        raise ValueError('worker binding deadline has expired')
    weights = load_base(query)
    # One exclusive foreground attempt; no retries or background dispatch.
    output = root / 'WM_GRID_RESULT.json'
    started = root / 'WM_GRID_STARTED.json'
    with started.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), query_sha256=digest(query), source_manifest=manifest,
                       new_sumo_sim_s=0, unix=time.time()), stream, sort_keys=True)
    rt, qualification = qualify(root)
    # plan_service already installs the ORIGINAL explicit reducer source with
    # (60,120,300,360,420,600). Re-adapting that transformed function against
    # the untransformed source would correctly fail its bytecode identity test.
    if time.time() >= binding['deadline_unix']:
        raise ValueError('deadline reached during original startup qualification')
    begin = time.monotonic()
    scores, stats = score_grid(rt, query, actions, weights)
    result = dict(schema='mining-wm-grid-v1', status='LONGITUDINAL_ONLY',
        query_sha256=digest(query), source_sha256=query['source_sha256'],
        base_checkpoint_sha256=U0_SHA, source_manifest=manifest,
        bootstrap_asset_pins=ASSET_PINS,
        qualification=qualification, K=8, H=600, horizons=list(HORIZONS),
        actions=actions, geometry_rejections=rejected, scores=scores,
        ranked=rankings(scores, actions), telemetry=stats, wall_s=time.monotonic()-begin,
        native_admissions=0, new_sumo_sim_s=0,
        lane_status='UNSUPPORTED: native target overwrite/post-motion ramp delegation',
        interpretation='WM ranking only; exact sampler on imagined local roster, not SUMO evidence')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
    print(json.dumps(dict(output=str(output), candidates=len(result['ranked']),
                          new_sumo_sim_s=0, status=result['status'])))


if __name__ == '__main__':
    main()
