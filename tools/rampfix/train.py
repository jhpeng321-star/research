"""One isolated S-RAMPFIX variant; fixed original validation, <=3h including assembly.

Variant override sets (applied over the S-LONGLOSSTEST base config; the resolved
config of each variant is written to ``<variant>/`` by ``write_resolved``):

* ``mres``  : ``model.mainline_interaction_residual=true`` (WM1 + the new zero-initialised
              mainline head, own optimizer group ``mainline_residual`` at scale 1.0) —
              everything else keeps the loop fine-tune freeze contract;
* ``lane5`` : ``optim.lane_v5_long=true`` (the EXISTING lane_v5 network unfrozen into its
              own group at scale 0.05; no architecture change);
* ``both``  : the two together.
"""
import argparse
from dataclasses import asdict
import json
import math
import shutil
import signal
import time

from common import CONFIG, DEEP, OUT, VARIANTS, disk, event, mirror, write


def variant_overrides(variant):
    sets = []
    if variant in ('mres', 'both'):
        sets += ['model.mainline_interaction_residual=true',
                 'model.mainline_residual_hidden=64']
    if variant in ('lane5', 'both'):
        sets += ['optim.lane_v5_long=true', 'optim.lane_v5_long_lr_scale=0.05']
    return sets


def apply_lr_scales(overrides, extra):
    """Merge ``extra`` into the base config's optim.lr_scale inline-table override."""
    from wms.config.load import load_train_config

    cfg0 = load_train_config(CONFIG)
    scales = {**dict(cfg0.optim.lr_scale or {}), **extra}
    if scales:
        overrides.append('optim.lr_scale={'
                         + ','.join(json.dumps(k)+'='+str(v) for k, v in scales.items()) + '}')
    return overrides


def train(variant):
    import numpy as np
    import torch
    from torch.utils.data import WeightedRandomSampler
    from wms.cli.assemble import assemble, build_datasets
    from wms.config.load import load_train_config, write_resolved
    from wms.train.trainer import WmSTrainer
    from pathlib import Path
    import sys

    sys.path.insert(0, str(OUT/'runtime/scripts/c4'))
    from loop14_long import wm_sets
    design = json.loads((OUT/'DESIGN.json').read_text())
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.85)
    base = ['runtime.device="cuda"', 'runtime.threads=4', 'runtime.num_workers=0',
            'loss.loss_share_logging=false',
            'model.init_from='+json.dumps(str(OUT/'wm1/wm_s.pt'))]
    sets = wm_sets({'looplong': {'enabled': True, 'wm_loss': True}},
                   {'cycles': [{'cycle': 2}]})
    assert len(sets) == 6
    overrides = base + sets + ['curriculum.stage_steps=[400]', 'curriculum.max_hours=3.0']
    overrides += variant_overrides(variant)
    if variant in ('mres', 'both'):
        # The fresh zero-init head trains at the full stage lr (its own parameter group).
        apply_lr_scales(overrides, {'mainline_residual': 1.0})
    cfg = load_train_config(CONFIG, overrides)
    dest = OUT/variant
    dest.mkdir()
    write_resolved(cfg, dest)
    start = time.monotonic()
    history, best, best_step, stale = [], float('inf'), 0, 0
    reason = 'max_steps'

    def alarm(signum, frame):
        raise TimeoutError('10800s assembly/training/validation cap')

    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, 10800)
    try:
        inputs = assemble(cfg, dest, resolver=lambda n: Path(design['sources'].get(n, n)))
        # Selection must use exactly the original 120s validation dataset, not the
        # training assembly's longer horizon or newly added cycle2 worlds.
        val_cfg = load_train_config(CONFIG, base)
        val_dir = dest/'original_validation'
        val_dir.mkdir()
        _, original_val, _ = build_datasets(
            val_cfg, val_dir,
            resolver=lambda n: Path(design['validation_sources'].get(n, n)))
        assert original_val is not None
        if variant in ('mres', 'both'):
            head = getattr(inputs.model, 'mainline_residual', None)
            assert head is not None, 'mainline interaction residual did not build'
            last = head.net[-1]
            assert float(last.weight.detach().abs().sum()) == 0.0 \
                and float(last.bias.detach().abs().sum()) == 0.0, \
                'mainline head must be exactly zero at init (identity contract)'
        trainer = WmSTrainer(cfg, inputs.model, inputs.train,
                             val_dataset=original_val, out_dir=dest)
        trainer.set_stage_lr(3e-5)
        names = trainer.optimizer_groups['trainable_tensor_names']
        # Contract: this job unfreezes ONLY its variant groups on top of the loop FT set.
        if variant in ('mres', 'both'):
            assert any(n.startswith('mainline_residual') for n in names)
            groups = {g['prefix']: g for g in trainer.optimizer_groups['groups']}
            assert 'mainline_residual' in groups
        if variant in ('lane5', 'both'):
            assert any(n.startswith('lane_v5') for n in names)
            assert trainer.optimizer_groups['lane_v5_long']['trainable'] is True
        frozen_deep = tuple(p for p in DEEP
                            if not (variant in ('lane5', 'both') and p == 'lane_v5'))
        assert not any(n.startswith(frozen_deep) for n in names)
        write(variant+'/OPTIMIZER.json', {**trainer.optimizer_groups,
              'actual_stage_learning_rates': [g['lr'] for g in trainer.optimizer.param_groups],
              'longwm_sets': sets, 'variant_overrides': variant_overrides(variant),
              'validation_windows_total': len(original_val)})
        def save(name, **extra):
            temporary = dest/(name+'.writing.pt')
            trainer._save(temporary, **extra)
            temporary.replace(dest/name)

        save('wm_s_initial.pt', step=0)
        write(variant+'/physics.json', asdict(trainer.inference_physics()))

        def validate(step):
            nonlocal best, best_step, stale
            t = time.monotonic()
            result = trainer.evaluate(original_val, unroll_steps=240, require_full_horizon=True)
            value = float(result['mean_position'])
            assert result['n_batches'] and math.isfinite(value)
            if value < best:
                best, best_step, stale = value, step, 0
                save('wm_s_best.pt', step=step, val=result)
            else:
                stale += 1
            event(variant=variant, validation_step=step, validation=result,
                  validation_wall_s=time.monotonic()-t, best_step=best_step, stale=stale)

        validate(0)
        inputs.train.set_paired_horizon(600)
        weights = inputs.train.stage_weights(300)
        gen = torch.Generator().manual_seed(int(cfg.runtime.seed))
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(inputs.train), replacement=True, generator=gen)
        loader = trainer._make_loader(sampler, batch_size=2)
        stop = False
        while len(history) < 400 and not stop:
            for batch in loader:
                disk()
                t = time.monotonic()
                event(variant=variant, phase='batch_start', step=len(history)+1,
                      paired='pair_ref' in batch)
                row = trainer.train_batch(batch, unroll_steps=600, truncation_steps=600)
                torch.cuda.synchronize()
                row.update(step=len(history)+1, wall_s=time.monotonic()-t)
                assert np.isfinite(row['loss'])
                history.append(row)
                with (dest/'train_log.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                save('wm_s_latest.pt', step=len(history))
                write('PROGRESS.json', {'variant': variant, 'phase': 'training',
                      'steps': len(history), 'last_step_s': row['wall_s'],
                      'elapsed_s': time.monotonic()-start,
                      'remaining_s': 10800-(time.monotonic()-start)})
                if len(history) % 5 == 0:
                    validate(len(history))
                    mirror()
                    if stale >= 3:
                        reason, stop = 'validation_early_stop', True
                        break
                if len(history) >= 400:
                    break
    except TimeoutError:
        reason = 'wall_time_cap'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    assert (dest/'wm_s_best.pt').exists(), 'no successfully validated checkpoint'
    shutil.copy2(dest/'wm_s_best.pt', dest/'wm_s.pt')
    write(variant+'/TRAINING.json', {'steps_reached': len(history), 'best_step': best_step,
          'best_mean_position': best, 'reason': reason, 'elapsed_s': time.monotonic()-start,
          'cap_s': 10800, 'longwm_updates': sum('longwm' in r for r in history),
          'step_wall_s': [r['wall_s'] for r in history]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=list(VARIANTS))
    train(parser.parse_args().variant)
