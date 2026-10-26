"""RSYNC-gated, bounded gpuA production-update timing. Never starts SUMO."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transfer-receipt", required=True, help="owner-provided gpuB transfer log receipt")
    ap.add_argument("--wm-ckpt", required=True)
    ap.add_argument("--wm-config", required=True)
    ap.add_argument("--native-returns", required=True)
    ap.add_argument("--reliability", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument('--real-returns')
    ap.add_argument('--policy-state')
    ap.add_argument('--calibrate-share', action='store_true')
    args = ap.parse_args()
    root = Path('/root/autodl-tmp/c4/longcritic').resolve()
    output, receipt = Path(args.output).resolve(), Path(args.transfer_receipt).resolve()
    if not output.is_relative_to(root) or not receipt.is_relative_to(root):
        raise ValueError("measurement output and supplied transfer receipt must be in the owned directory")
    if output.exists():
        raise ValueError("never overwrite a measurement")
    text = receipt.read_text(encoding='utf-8')
    if 'RSYNC_RC=0' not in text:
        raise ValueError("wait for RSYNC_RC=0; no training/measurement before transfer completion")
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '4'
    (root / 'tmp').mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(root / 'tmp')
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.cuda.set_per_process_memory_fraction(.3, device=0)
    from dataclasses import replace
    from wms.control.policy14_cli import load_members, check_train_seeds
    from wms.control.train_cli import _load_starts, _read_config, GroupSignal, policy_settings
    from wms.control.dreamer import DreamerConfig
    from wms.control.policy14 import Policy14Trainer, make_controller_factory
    from wms.control.real_returns14 import NPZRealReturnProvider, ScaledTeamCritic
    from wms.control.longcritic import LongCritic, LongCriticConfig
    from wms.control.coopw import ExplorationConfig, UncertaintyConfig
    from wms.control.counterfactual import CounterfactualConfig
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.control.obs import build_observations
    from wms.control.objective14 import authoritative_reward
    from wms.control.safety import safe_accel
    from wms.control.lanes import LaneRules
    from wms.control.contract import RewardWeights
    raw = _read_config('configs/c4_policy14_longcritic.toml')
    models, model_receipts = load_members([args.wm_ckpt], [args.wm_config], 'cuda')
    native = NPZRealReturnProvider(args.native_returns, seed=965140)
    from wms.control.real_share import split_provider, exclude_development_starts, calibrate_share
    real = NPZRealReturnProvider(args.real_returns, seed=965140) if args.real_returns else native
    fit, dev = split_provider(real) if args.calibrate_share else (real, None)
    native_fit, _ = split_provider(native) if args.calibrate_share else (native, None)
    results = {}
    for enabled in (False, True):
        random.seed(965140)
        np.random.seed(965140)
        torch.manual_seed(965140)
        train, _ = _load_starts(raw['data'], models, 965140, entries=True)
        check_train_seeds(train)
        if dev is not None:
            train = exclude_development_starts(train, set(dev.provenance['seeds'].tolist()))
        settings = policy_settings(raw)
        policy = CavPolicy(accel_max=settings['accel_max'], lane_head=settings.get('lane_control') != 'native',
                           init_zero_accel=settings.get('init_zero_accel', False))
        lane = dict(raw['lanes'])
        rules = LaneRules.from_mapping({k: v for k, v in lane.items() if k != 'enabled'}) if lane.get('enabled', True) else None
        exploration = ExplorationConfig(**{**raw['exploration'], 'long_persistence': enabled})
        switches = raw['policy14']
        trainer = Policy14Trainer(models, policy, GroupSignal(RsuGroupEncoder()),
            ScaledTeamCritic(entry_backlog=True, value_scale=float(switches['critic_value_scale'])), train,
            observation_builder=build_observations, reward_fn=authoritative_reward, safety=safe_accel,
            controller_factory=make_controller_factory(**{k: switches[k] for k in
                ('policy_ramp_merge', 'native_ramp_merge', 'native_lane_change') if k in switches}),
            config=replace(DreamerConfig(**raw['trainer']), validation_every=0, checkpoint_every=0),
            reward_weights=RewardWeights(**raw['reward']), device='cuda', lane_rules=rules,
            counterfactual=CounterfactualConfig(**raw['counterfactual']), real_return_provider=fit,
            exploration=exploration, uncertainty=UncertaintyConfig(**raw['uncertainty']))
        # Freeze at the same fully-ramped exploration/credit position in both arms.
        trainer.coop_anchor, trainer.update_count = 0, 150
        if args.policy_state:
            if args.calibrate_share:
                provenance = json.loads((Path(args.policy_state).parent/'CURRENT_REAL_SOURCE.json').read_text())
                if provenance['policy']['sha256'] != real.provenance.get('policy_checkpoint'):
                    raise ValueError('policy snapshot and real returns are not paired')
            state = torch.load(args.policy_state, map_location='cuda', weights_only=True)
            for name in ('policy', 'group_encoder', 'critic', 'critic_optimizer'):
                getattr(trainer, name).load_state_dict(state[name])
            trainer.target_critic.load_state_dict(state['critic'])
            del state
        selection, selection_s = None, None
        if enabled:
            cfg = LongCriticConfig(**{**raw['longcritic'], 'reliability_file': args.reliability})
            trainer.longcritic = LongCritic(trainer, cfg, native_fit,
                                            [r['checkpoint_sha256'] for r in model_receipts])
            if not any(trainer.longcritic.weights.get(h, 0) > 0 for h in cfg.horizons_s):
                raise ValueError("no compatible measured nonzero long weights; timing would omit item 1")
            trainer.longcritic.warmup(trainer)
            if args.calibrate_share:
                torch.cuda.synchronize()
                selection_start = time.perf_counter()
                selection = calibrate_share(trainer, fit, dev,
                    policy_sha256=real.provenance.get('policy_checkpoint'))
                torch.cuda.synchronize()
                selection_s = time.perf_counter() - selection_start
                (root/'REAL_SHARE_MEASURED.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
                print(json.dumps({'selection_s': selection_s, 'selection': selection}), flush=True)
        trainer.update()  # one unmeasured warm update; then one complete horizon rotation.
        elapsed, rows = [], []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            torch.cuda.synchronize()
            started = time.perf_counter()
            metrics = trainer.update()
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - started)
            rows.append({k: v for k, v in metrics.items() if k.startswith('longcritic')})
        results['on' if enabled else 'off'] = {'seconds': elapsed, 'median_s': statistics.median(elapsed),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'updates': rows,
            'real_share': trainer.config.real_return_mix, 'selection': selection, 'selection_s': selection_s}
        del trainer
        torch.cuda.empty_cache()
    payload = {'schema': 'longcritic_update_cost/1', 'device': torch.cuda.get_device_name(),
        'gpu_fraction': .3, 'threads': 4, 'new_sumo_sim_s': 0, 'models': model_receipts,
        'transfer_receipt_sha256': hashlib.sha256(text.encode()).hexdigest(), 'results': results}
    payload['speed2'] = {k: os.environ.get(k) for k in ('WMS_SPEED_PIVOT_DIAG_OFF', 'WMS_SPEED_GRAD_AUDIT_EVERY')}
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
