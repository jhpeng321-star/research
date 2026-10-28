"""Genuine K8 current-WM futures on historical paired starts; no simulator.

Only authoritative 300s totals are available without queue clocks. Missing
request histories are NOT silently replaced by insertion rates: preflight
records them and refuses launch-compatible fitting until supplied.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

from common import ACTORS, CODE, OUT, REL, WM, WM_CONFIG, WM_SHA, accepted, read, safe, sha, write_new

sys.path[:0] = [str(CODE / 'src'), str(CODE)]


def inventory():
    """Examine every registered pair, not just a representative episode."""
    import numpy as np
    accepted()
    cycles, blockers = [], []
    for cycle in (1, 2, 3):
        root = REL / f'arm/loop/data/pairs_c{cycle:02}.a1'
        proof_path = REL / f'arm/loop/returns/pairing_c{cycle:02}.a1.json'
        proof = read(proof_path)
        actor, digest = ACTORS[cycle]
        assert proof['policy_checkpoint_sha256'] == digest
        actor_ok = actor.is_file() and sha(actor) == digest
        if not actor_ok:
            blockers.append(f'C{cycle} historical actor pathname is missing/stale; recover by SHA, never use a replacement actor')
        manifest_path = root / 'C4_EPISODES_MANIFEST.json'
        manifest = read(manifest_path)
        episodes = []
        for ep in manifest['episodes']:
            assert str(ep.get('split_role', '')).lower() == 'train'
            assert not 964000 <= int(ep['seed']) <= 964199
            p = safe(root / ep['dir'])
            missing = [name for name in ('entry_backlog', 'request_counter', 'entry_cav_counter')
                       if not safe(p / (name + '.npy')).is_file()]
            episodes.append({'episode': ep['episode'], 'missing_observable_arrays': missing})
        with np.load(safe(REL / f'arm/loop/returns/returns_c{cycle:02}.a1.npz'), allow_pickle=False) as z:
            assert str(z['policy_checkpoint']) == digest
            assert float(z['gamma']) == 1.0
            assert str(z['accounting']) == 'source_integral_plus_road_delay'
            queues = z['state__entry_backlog'].tolist()
        if any(e['missing_observable_arrays'] for e in episodes):
            blockers.append(f'C{cycle} missing recorded demand/CAV history and per-tick queue clocks; t0 backlog alone does not identify demand')
        cycles.append({'cycle': cycle, 'actor': str(actor), 'actor_sha256': digest,
                       'actor_verified': actor_ok, 'pairing_sha256': sha(proof_path),
                       'manifest_sha256': sha(manifest_path), 'pairs': len(proof['pairs']),
                       'start_backlogs': queues, 'episodes': episodes})
    return {'schema': 'c4prep_input_preflight/1', 'wm_sha256': WM_SHA,
            'source_commit': read(CODE / 'SOURCE.json')['commit'], 'cycles': cycles,
            'ready': not blockers, 'blockers': blockers,
            'no_inference_performed': True, 'no_sumo': True}


def replay(entrant, dest):
    """Use exact queue-aware policy imagination and recorded real costs."""
    import numpy as np
    import torch
    from wms.config.load import load_train_config
    from wms.control.antiexploit import inference_signature
    from wms.control.counterfactual import CounterfactualConfig, Variants, KIND_ALL, KIND_BASE, acting_cavs
    from wms.control.dreamer import DreamerConfig
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.control.policy14_cli import (Policy14Trainer, GroupSignal, ScaledTeamCritic,
        RewardWeights, authoritative_reward, load_members, make_controller_factory, _read_config,
        NPZRealReturnProvider)
    from wms.control.obs import build_observations
    from wms.control.safety import safe_accel
    from wms.control.start_states import StartStateSampler
    from wms.data.c4_pairs import PairedBranchSource
    from tools.antiexploit.offline import calibrate

    proof = inventory()
    write_new(dest / 'INPUTS.json', proof)
    if not proof['ready']:
        write_new(dest / 'BLOCKED.json', {'blockers': proof['blockers'], 'inference_jobs': 0})
        raise ValueError('observable queue/request/CAV clocks missing; see INPUTS.json')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.45)
    raw = _read_config(CODE / f'configs/c4_policy14_c4prep_{"on" if entrant else "off"}.toml')
    cfg = replace(DreamerConfig(**raw['trainer']), native_anchor=False)
    models, _ = load_members([str(WM)], [str(WM_CONFIG)], 'cuda')
    data_cfg = load_train_config(str(WM_CONFIG))
    payload = None
    for cycle in (1, 2, 3):
        source = REL / f'arm/loop/data/pairs_c{cycle:02}.a1'
        pairs = PairedBranchSource(source, horizon_steps=600,
            history_s=data_cfg.data.history_s, history_stride_s=data_cfg.data.history_stride_s,
            entries_v2=True, junction_fix=data_cfg.data.junction_fix,
            merge_pivot_history=data_cfg.model.merge_decision == 'first_passage', name='loop14_pairs',
            arm_kinds=('B', 'POLICY'))
        sampler = StartStateSampler(pairs.dataset)
        policy = CavPolicy(residual_on_native=True, lane_head=False)
        trainer = Policy14Trainer(models, policy, GroupSignal(RsuGroupEncoder()),
            ScaledTeamCritic(entry_backlog=True), sampler,
            observation_builder=build_observations, reward_fn=authoritative_reward,
            safety=safe_accel, controller_factory=make_controller_factory(
                native_lane_change=True, native_ramp_merge='wm_law'), config=cfg,
            reward_weights=RewardWeights(**raw['reward']), device='cuda',
            counterfactual=CounterfactualConfig(**raw['counterfactual']),
            real_return_provider=NPZRealReturnProvider(
                str(REL / f'arm/loop/returns/returns_c{cycle:02}.a1.npz')))
        trainer.load_checkpoint(ACTORS[cycle][0], policy_only=True)
        trainer.policy.eval()
        trainer.group_encoder.eval()
        trainer._training_imagination = False
        signature = inference_signature(cfg)
        if payload is None:
            payload = {'world_model_ids': trainer.world_model_ids, 'inference_signature': signature, 'rows': []}
        assert payload['world_model_ids'] == trainer.world_model_ids
        pairing = read(REL / f'arm/loop/returns/pairing_c{cycle:02}.a1.json')
        groups = {g['pair_key']: g for g in pairs.groups}
        with np.load(safe(REL / f'arm/loop/returns/returns_c{cycle:02}.a1.npz'), allow_pickle=False) as z:
            real_policy = z['returns'].copy()
        with np.load(safe(REL / f'arm/loop/returns/native_returns_c{cycle:02}.a1.npz'), allow_pickle=False) as z:
            real_native = z['returns'].copy()
        for i, item in enumerate(pairing['pairs']):
            group = groups[item['pair_key']]
            row = next(r for r in group['rows'][1:] if
                       pairs.dataset._loader.entries[r]['branch']['kind'] == 'POLICY')
            curves = []
            for r, target in ((row, real_policy[i]), (group['rows'][0], real_native[i])):
                loader = pairs.dataset._loader
                ep_index, tick = loader.index[r]
                ep = loader.episode(ep_index)
                kin = ep.array('veh_kin')[tick+1:tick+601]
                mask = ep.array('veh_mask')[tick+1:tick+601]
                queues = ep.array('entry_backlog')[tick:tick+600]
                costs = .5 * ((np.maximum(0., 1.-kin[..., 4]/31.29)*mask).sum(-1) + queues.sum(-1))
                assert len(costs) == 600 and np.isclose(costs.sum(), -float(target), rtol=1e-6, atol=1e-3)
                curves.append(costs)
            seen_horizons = set()
            for h in (10, 30, 60, 90, 120, 180, 300):
                seed = 710003 + 100003 * cycle + 1009 * h + i
                trainer.config = replace(cfg, horizon=h, entry_pad_max=max(32, (32*h+9)//10))
                torch.manual_seed(seed)
                with torch.no_grad():
                    start = sampler._batch([sampler._load(row)], 'cuda')
                    actual = trainer._imagination_config(start).horizon
                    if actual in seen_horizons:
                        continue
                    seen_horizons.add(actual)
                    start = trainer._entry_slots(start, torch.Generator().manual_seed(seed))
                    cav = acting_cavs(start.state)
                    variants = Variants(torch.tensor([0, 0], device='cuda'),
                        torch.cat((cav, torch.zeros_like(cav))),
                        torch.tensor([KIND_ALL, KIND_BASE], device='cuda'), 1, {0: 1}, {}, [])
                    image = trainer.imagine(start, variants=variants, deterministic=True,
                                            future_seed=seed)
                    assert int(image['imagination_horizon_s']) == actual
                    c = image['components']
                    costs = (c['delay'] + c['source_wait']).sum(1)
                    samples = (costs[:, 0] - costs[:, 1]).cpu().tolist()
                    assert len(samples) == 8
                    commands = image['antiexploit_commands']
                    deviation, mask = commands['deviation'][..., :1, :], commands['mask'][..., :1, :]
                    mse = float((deviation.square() * mask).sum() / mask.sum().clamp_min(1))
                result = {'cycle': cycle, 'world': item['seed'], 'start_s': int(item['pair_key'].split('@t')[1]),
                          'horizon_s': actual, 'real_dJ': float((curves[0]-curves[1])[:2*actual].sum()),
                          'paired_cost_samples': samples, 'native_residual_mse': mse,
                          'actor_sha256': ACTORS[cycle][1], 'seed': seed}
                write_new(dest / f'rows/c{cycle}_{i:02}_h{actual}.json', result)
                payload['rows'].append(result)
                print(json.dumps({'cycle': cycle, 'pair': i, 'horizon': actual, 'K': len(samples)}), flush=True)
                del image
    write_new(dest / 'PAIRED_FUTURES.json', payload)
    calibrate(dest / 'PAIRED_FUTURES.json', dest / 'ANCHOR.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('preflight', 'replay'))
    p.add_argument('--entrant', choices=('on', 'off'), default='off')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    target = safe(a.output)
    if a.action == 'preflight':
        doc = inventory()
        write_new(target, doc)
        print(json.dumps({k: v for k, v in doc.items() if k != 'cycles'}))
    else:
        replay(a.entrant == 'on', target)
