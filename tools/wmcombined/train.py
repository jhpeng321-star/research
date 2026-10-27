"""One isolated S-WMCOMBINED variant; LONG-HORIZON checkpoint selection, <=3.5 h.

Variant ``combined`` = the loop FT recipe (c4_wm_combined.toml = c4_wm_loop14_r16_ft
verbatim + the S-E3DATA-1 fit-worlds paired source) + WM2 init +
``model.mainline_interaction_residual`` (zero-init head, own group at scale 1.0) +
``optim.lane_v5_long`` (existing lane_v5 net, own group at 0.05x) + the wired longwm
paired loss (300 s full-path unroll, 600-tick truncation, batch 2, coupled paired
randomness) over the CONCATENATED real paired pool (loop cycles 0-2 + 156 E3 fit
worlds).

S-LONGLOSSTEST-1 lesson: checkpoint selection on the 120 s position metric rejected
every long-loss update. Selection here is LONG-HORIZON ONLY: the held-out 300 s
paired-dJ error (|mean| primary, MAE reported) on the E3DATA 'calibration' worlds
(36; never fitted, never the verdict worlds), every 10 steps (and step 0 and the
final step); the original fixed-16 120 s mean_position is recorded as a DIAGNOSTIC
only. The final-step checkpoint is saved and scored too.
"""
import argparse
from dataclasses import asdict
import json
import math
import shutil
import signal
import time

from common import (CONFIG, E3, OUT, SELECT_EVERY, SELECT_HORIZON_S,
                    SELECT_PATIENCE, SELECT_ROLE, THRESH, WM2, VARIANTS, disk, event,
                    mirror, write)

CAP_S = 12600  # 3.5 h wall-time cap for assembly + training + selection


def variant_overrides(variant):
    assert variant == 'combined'
    return ['model.mainline_interaction_residual=true', 'model.mainline_residual_hidden=64',
            'optim.lane_v5_long=true', 'optim.lane_v5_long_lr_scale=0.05',
            'loss.longwm_extra_paired="e3_pairs"']


def apply_lr_scales(overrides, extra):
    from wms.config.load import load_train_config

    cfg0 = load_train_config(CONFIG)
    scales = {**dict(cfg0.optim.lr_scale or {}), **extra}
    if scales:
        overrides.append('optim.lr_scale={'
                         + ','.join(json.dumps(k)+'='+str(v) for k, v in scales.items()) + '}')
    return overrides


def train(variant, dry_run=False):
    import numpy as np
    import torch
    from torch.utils.data import WeightedRandomSampler
    from wms.cli.assemble import assemble, build_datasets
    from wms.config.load import load_train_config, write_resolved
    from wms.train.settings import TrainSettings
    from wms.train.trainer import WmSTrainer
    from pathlib import Path
    import sys

    sys.path.insert(0, str(OUT/'code'))
    from e3_score import evaluate as e3_evaluate
    sys.path.insert(0, str(OUT/'runtime/scripts/c4'))
    from loop14_long import wm_sets
    design = json.loads((OUT/'DESIGN.json').read_text())
    torch.set_num_threads(4)
    if not dry_run:
        torch.cuda.set_per_process_memory_fraction(.85)
    base = ['runtime.device="cpu"' if dry_run else 'runtime.device="cuda"',
            'runtime.threads=4', 'runtime.num_workers=0',
            'loss.loss_share_logging=false', 'model.init_from='+json.dumps(str(WM2))]
    sets = wm_sets({'looplong': {'enabled': True, 'wm_loss': True}},
                   {'cycles': [{'cycle': 2}]})
    assert len(sets) == 6
    overrides = base + sets + ['curriculum.stage_steps=[400]', 'curriculum.max_hours=3.5']
    overrides += variant_overrides(variant)
    apply_lr_scales(overrides, {'mainline_residual': 1.0})
    cfg = load_train_config(CONFIG, overrides)
    dest = OUT/('dryrun' if dry_run else variant)
    dest.mkdir(exist_ok=dry_run)
    write_resolved(cfg, dest)
    start = time.monotonic()
    history, best, best_step, stale = [], float('inf'), 0, 0
    reason = 'max_steps'

    def alarm(signum, frame):
        raise TimeoutError(f'{CAP_S}s assembly/training/validation cap')

    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, CAP_S)
    try:
        inputs = assemble(cfg, dest, resolver=lambda n: Path(design['sources'].get(n, n)))
        paired_receipt = inputs.train.pairs.receipt
        assert paired_receipt['mode'] == 'c4-paired-branches-concat', paired_receipt
        assert paired_receipt['n_groups'] == 46 + 156, paired_receipt
        assert paired_receipt['horizon_steps'] == 600, paired_receipt
        # Selection must use the E3DATA calibration worlds (36), never fit/validation.
        val_cfg = load_train_config(CONFIG, base+['loss.paired_diff.enabled=false'])
        val_dir = dest/'original_validation'
        val_dir.mkdir()
        _, original_val, _ = build_datasets(
            val_cfg, val_dir,
            resolver=lambda n: Path(design['validation_sources'].get(
                n, design['sources'].get(n, n))))
        assert original_val is not None
        head = getattr(inputs.model, 'mainline_residual', None)
        assert head is not None, 'mainline interaction residual did not build'
        last = head.net[-1]
        assert float(last.weight.detach().abs().sum()) == 0.0 \
            and float(last.bias.detach().abs().sum()) == 0.0, \
            'mainline head must be exactly zero at init (identity contract)'
        trainer = WmSTrainer(cfg, inputs.model, inputs.train,
                             val_dataset=original_val, out_dir=dest)
        trainer.set_stage_lr(3e-5)
        # wm_s_initial must equal WM2 EXACTLY (init_from is applied by the trainer).
        payload = torch.load(str(WM2), map_location='cpu', weights_only=True)
        own = inputs.model.state_dict()
        rep = trainer.init_report
        assert not rep['shape_mismatch'] and not rep['expanded'], rep['shape_mismatch'][:3]
        missing = [k for k in payload['model'] if k not in own]
        assert not missing, missing[:3]
        delta = max(float((own[k].float()-v.float()).abs().max())
                    for k, v in payload['model'].items())
        assert delta == 0.0, f'init_from is not exactly WM2 (max |delta| {delta})'
        write(('dryrun' if dry_run else variant)+'/INIT_PROOF.json',
              {'wm2_tensors': len(payload['model']), 'model_tensors': len(own),
               'max_abs_delta': delta, 'new_tensors': sorted(set(own)-set(payload['model']))})
        names = trainer.optimizer_groups['trainable_tensor_names']
        groups = {g['prefix']: g for g in trainer.optimizer_groups['groups']}
        assert any(n.startswith('mainline_residual') for n in names) \
            and 'mainline_residual' in groups
        assert any(n.startswith('lane_v5') for n in names) \
            and trainer.optimizer_groups['lane_v5_long']['trainable'] is True
        frozen_deep = ('merge_core', 'merge_pivot', 'ramp_speed_adapt',
                       'driver_encoder', 'driver_updater', 'residual_net.lane')
        assert not any(n.startswith(frozen_deep) for n in names)
        if dry_run:
            write('dryrun/PREFLIGHT_ASSEMBLE.json', {
                'paired_batches': paired_receipt, 'validation_windows': len(original_val),
                'optimizer_groups': [g['prefix'] for g in trainer.optimizer_groups['groups']],
                't': __import__('common').now(), 'free_gib': disk()})
            return
        write(variant+'/OPTIMIZER.json', {**trainer.optimizer_groups,
              'actual_stage_learning_rates': [g['lr'] for g in trainer.optimizer.param_groups],
              'longwm_sets': sets, 'variant_overrides': variant_overrides(variant),
              'validation_windows_total': len(original_val),
              'paired_batches': paired_receipt})

        def save(name, **extra):
            temporary = dest/(name+'.writing.pt')
            trainer._save(temporary, **extra)
            temporary.replace(dest/name)

        save('wm_s_initial.pt', step=0)
        write(variant+'/physics.json', asdict(trainer.inference_physics()))
        settings = TrainSettings.from_config(cfg)
        assert settings.paired_diff_couple_randomness
        threshold = json.loads(THRESH.read_text())['threshold']
        device = next(inputs.model.parameters()).device

        def select_long(step):
            """The registered LONG-HORIZON criterion: held-out 300 s paired dJ error."""
            nonlocal best, best_step, stale
            t = time.monotonic()
            doc = e3_evaluate(inputs.model, settings, str(E3), SELECT_ROLE, device,
                              threshold=threshold)
            h = doc['horizons'][str(int(SELECT_HORIZON_S))]
            value = abs(h['mean_signed_error'])
            assert math.isfinite(value) and math.isfinite(h['mae'])
            if value < best:
                best, best_step, stale = value, step, 0
                save('wm_s_best.pt', step=step, selection=h)
            else:
                stale += 1
            event(variant=variant, selection_step=step, selection=h,
                  selection_wall_s=time.monotonic()-t, best_step=best_step, stale=stale)
            return doc

        def diagnostic_120s(step):
            t = time.monotonic()
            result = trainer.evaluate(original_val, unroll_steps=240,
                                      require_full_horizon=True)
            value = float(result['mean_position'])
            assert result['n_batches'] and math.isfinite(value)
            event(variant=variant, diagnostic_120s_step=step, diagnostic=result,
                  diagnostic_wall_s=time.monotonic()-t)
            return value

        baseline = select_long(0)
        write(variant+'/SELECTION_WM2_BASELINE.json', baseline)
        diagnostic_120s(0)
        mirror()
        inputs.train.set_paired_horizon(600)
        weights = inputs.train.stage_weights(300)
        gen = torch.Generator().manual_seed(int(cfg.runtime.seed))
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(inputs.train), replacement=True,
                                        generator=gen)
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
                      'remaining_s': CAP_S-(time.monotonic()-start)})
                if len(history) % SELECT_EVERY == 0:
                    select_long(len(history))
                    diagnostic_120s(len(history))
                    mirror()
                    if stale >= SELECT_PATIENCE:
                        reason, stop = 'long_selection_early_stop', True
                        break
                if len(history) >= 400:
                    break
        if not stop and len(history) % SELECT_EVERY != 0:
            select_long(len(history))  # the final step is always selected on
    except TimeoutError:
        reason = 'wall_time_cap'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    assert (dest/'wm_s_best.pt').exists(), 'no successfully validated checkpoint'
    save('wm_s_final.pt', step=len(history), reason=reason)
    shutil.copy2(dest/'wm_s_best.pt', dest/'wm_s.pt')
    write(variant+'/TRAINING.json', {'steps_reached': len(history), 'best_step': best_step,
          'best_abs_mean_300s_paired_error': best, 'reason': reason,
          'elapsed_s': time.monotonic()-start, 'cap_s': CAP_S,
          'longwm_updates': sum('longwm' in r for r in history),
          'step_wall_s': [r['wall_s'] for r in history],
          'selection': {'role': SELECT_ROLE, 'horizon_s': SELECT_HORIZON_S,
                        'every': SELECT_EVERY, 'patience': SELECT_PATIENCE,
                        'criterion': 'abs(mean 300s paired dJ error), strict improvement'},
          'start_checkpoint': str(WM2)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=list(VARIANTS))
    parser.add_argument('--dry-run', action='store_true')
    train(parser.parse_args().variant, parser.parse_args().dry_run)
