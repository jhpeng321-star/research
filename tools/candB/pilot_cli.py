"""Isolated opt-in pilot adapter: unchanged Policy14 algebra, explicit evidence waiver."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from functools import wraps
import json
import math
import os
from pathlib import Path
import time
from unittest.mock import patch

import torch

from tools.candB import pilot_authority as authority


def install_exploratory(trainer, cfg, wm_sha256s, receipt):
    """Bypass qualification evidence only; never write a false CCT-2 PASS."""
    from tools.distill.runtime import sha256
    if (list(wm_sha256s) != [authority.WM_SHA] or not cfg.require_objective_match
            or not cfg.ctrl0_reference or cfg.k16 or cfg.estimator != 'antithetic'
            or cfg.es_directions != 2 or trainer.config.future_samples != 8
            or not trainer.config.pessimistic_credit or not trainer.config.deterministic_crn
            or not trainer.config.entrant_control or not trainer.config.entrant_learned_mix
            or trainer.policy.residual_reference != 'ctrl0'):
        raise ValueError('EXPLORATORY long-credit identity mismatch')
    receipt.update(H_long=300, H_G=300, label='EXPLORATORY', policy_authorized=False,
        qualification_passed=False, reason='owner 20261030 exploratory evidence waiver',
        artifact_sha256=sha256(cfg.cct2_file.format(wm_sha256=authority.WM_SHA)))


def command(doc, output, mode):
    from wms.control.phase_resume import resume_diff
    from wms.control.dreamer import DreamerConfig
    from wms.control.contract import RewardWeights
    raw, i = authority.config(doc['inputs']), doc['inputs']
    payload = torch.load(i['current_policy'], map_location='cpu', weights_only=True)
    args = ['--config', authority.ENTRY, '--wm-ckpt', i['wm'], '--wm-config', i['wm_config'],
        '--real-returns', i['real_returns'], '--longcritic-native-returns', i['native_returns'],
        '--paircrit-receipt', i['pairing_receipt'], '--paircrit-policy-sha256',
        '0978aa7288cf9e36a0b61949b1e236b3fe48a0d220dada64f67a4797f969dc5e',
        '--gate-panel', i['gate_panel'], '--horizon-gate', i['horizon_gate'],
        '--looplong-boundary', i['boundary'], '--critic-warmup-updates', '0',
        '--native-init', '--phase-relative-signal-gate', '--resume', i['current_policy'],
        '--new-phase', '--device', 'cuda', '--seed', '965140', '--max-updates', '150',
        '--output-dir', output, '--compact']
    for key in sorted(resume_diff(DreamerConfig(**raw['trainer']), RewardWeights(**raw['reward']), payload)):
        args += ['--resume-allow', key]
    if mode == 'dry':
        args += ['--prep-dry-run-updates', '2']
    return args


def run(path, output, mode, *, exploratory=False):
    authority.require_opt_in(exploratory)
    if 'EXPLORATORY' not in Path(output).name or Path(output).exists():
        raise ValueError('pilot output must be fresh and named EXPLORATORY')
    doc = authority.verify(path, exploratory=True)
    from wms.control import policy14_cli as cli, gated300, execution, policy14
    from wms.control.dreamer import DreamerTrainer
    from wms.control import parallel_futures, eslong
    from tools.candB import rankcheck
    args = command(doc, output, mode) + ['--rankcheck', str(path)]
    # Standard CLI checks the resumed initialization and pairing/boundary hashes.
    doc.update(current_policy_sha256=doc['input_sha256']['current_policy'],
               boundary_sha256=doc['input_sha256']['boundary'],
               pairing_receipt_sha256=doc['input_sha256']['pairing_receipt'])
    original_load, original_update = DreamerTrainer.load_checkpoint, DreamerTrainer.update
    original_score, original_execute, dumps = cli.independent_score, execution.executed_residual, json.dumps
    original_controller = policy14.make_cav_controller
    state = dict(capture=None, phase_update=0, baseline=None, trainer=None, began=None)

    class SerialPool:
        """Same ordered ES tasks on the coordinator; avoids a second CUDA context."""
        def __init__(self, trainer, common):
            if not common.get('eslong') or common.get('kind') != 'forward':
                raise ValueError('pilot serial pool is restricted to ES forward tasks')
            self.trainer, self.common = trainer, common

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, tasks):
            for task in tasks:
                yield eslong.forward_task(self.trainer, self.common, task)

    def load(trainer, checkpoint, *a, **kw):
        with Path(output, 'RUN_SUMMARY.json').open('w', encoding='utf-8') as stream:
            json.dump(dict(status='initializing', label='EXPLORATORY', deployment_authorized=False,
                source_commit=doc['source_commit'], input_sha256=doc['input_sha256']), stream, indent=1)
        current = os.environ['C4_EXECUTION_DRIVER']
        try:
            os.environ['C4_EXECUTION_DRIVER'] = doc['inputs']['current_policy_driver']
            return original_load(trainer, checkpoint, *a, **kw)
        finally:
            os.environ['C4_EXECUTION_DRIVER'] = current

    def controller(*a, **kw):
        callback = original_controller(*a, **kw)
        @wraps(callback)
        def measured(frame):
            result = callback(frame)
            capture = state['capture']
            if capture is not None:
                with torch.no_grad():
                    n = capture['n']
                    zeros = torch.zeros_like(frame['s_m'])
                    zero = original_execute(frame, zeros, zeros, residual_reference='ctrl0')
                    zero = ((frame['v_mps'] + .5*zero).clamp_min(0.)-frame['v_mps'])/.5
                    mask = (frame['present'] & frame['is_cav'] & (frame['lane'] >= 0))[:n]
                    if kw.get('act_mask') is not None:
                        mask = mask & kw['act_mask'][:n]
                    # Observe the FINAL callback acceleration, after acting masks
                    # and nonnegative-speed Euler execution, not requested actions.
                    delta = (result[:n, :, 3] - zero[:n])[mask]
                    values = torch.stack((mask.sum(), (delta != 0).sum(),
                        (-delta).clamp_min(0).sum(), delta.abs().sum(), delta.sum())).double()
                    capture['totals'] = values if capture['totals'] is None else capture['totals'] + values
            return result
        return measured

    def score(trainer, batch, *a, **kw):
        capture = dict(n=batch.state['s_m'].shape[0], totals=None)
        state['capture'] = capture
        try:
            result = original_score(trainer, batch, *a, **kw)
        finally:
            state['capture'] = None
        totals = capture['totals']
        if totals is not None:
            count, nonzero, brake, absolute, signed = totals.cpu().tolist()
            result['executed_residual'] = dict(scope='deterministic paired rescore, group rows, acting CAV physics ticks',
                count=count, fraction_nonzero=nonzero/max(count, 1), mean_braking=brake/max(count, 1),
                mean_abs_delta=absolute/max(count, 1), mean_delta=signed/max(count, 1))
        if state['baseline'] is None:
            state['baseline'] = {k: result[k] for k in ('raw_dJ_vs_CTRL0_dense', 'raw_dJ_vs_CTRL0_long')}
            if any(v != 0. for v in state['baseline'].values()):
                raise ValueError('EXPLORATORY zero-start != CTRL0 on frozen scoring start')
        return result

    def update(trainer, *a, **kw):
        state.update(trainer=trainer, began=time.monotonic())
        metrics = original_update(trainer, *a, **kw)
        state['phase_update'] += 1
        metrics.update(label='EXPLORATORY', phase_update=state['phase_update'],
                       optimizer_update_s=time.monotonic()-state['began'])
        return metrics

    def serialize(value, *a, **kw):
        if isinstance(value, dict) and ('actor_loss' in value or 'status' in value):
            value = dict(value, label='EXPLORATORY', deployment_authorized=False,
                         exploratory_authority_sha256=authority.sha256(path))
            if 'actor_loss' in value and 'independent_J' in value:
                value['update_including_rescore_s'] = time.monotonic()-state['began']
                if value.get('phase_update') == 10:
                    score = value['independent_J']
                    dense, long = (score[k] for k in ('raw_dJ_vs_CTRL0_dense', 'raw_dJ_vs_CTRL0_long'))
                    passed = all(math.isfinite(x) for x in (dense, long)) and dense < state['baseline']['raw_dJ_vs_CTRL0_dense']
                    value['R4'] = dict(passed=passed, rule='finite dense+300s; raw dense dJ below frozen phase zero')
                    if not passed:
                        Path(output, 'STOP').touch(exist_ok=True)
        return dumps(value, *a, **kw)

    def check_rank(document, raw, **kw):
        if document.get('schema') != authority.SCHEMA:
            raise ValueError('EXPLORATORY adapter accepts only its own distinct authority schema')
        return authority.verify_runtime(doc, raw, **kw)

    with ExitStack() as stack:
        for target, name, replacement in ((gated300, '_install_gain_only', install_exploratory),
                (rankcheck, 'verify_receipt', check_rank), (DreamerTrainer, 'load_checkpoint', load),
                (DreamerTrainer, 'update', update), (cli, 'independent_score', score),
                (policy14, 'make_cav_controller', controller), (json, 'dumps', serialize)):
            stack.enter_context(patch.object(target, name, replacement))
        stack.enter_context(patch.object(parallel_futures, 'FuturePool', SerialPool))
        return cli.main(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(authority.FLAG, action='store_true', required=True)
    p.add_argument('--authority', required=True)
    p.add_argument('output')
    p.add_argument('mode', choices=('dry', 'pilot'))
    a = p.parse_args()
    raise SystemExit(run(a.authority, a.output, a.mode, exploratory=True))


if __name__ == '__main__':
    main()
