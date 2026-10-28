"""Policy-side boundary binding and accepted-WM hot swap, all opt-in."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch

from .start_states import checked_path


def digest(path):
    return hashlib.sha256(checked_path(path).read_bytes()).hexdigest()


def read_boundary(path, *, swap=False):
    doc = json.loads(checked_path(path).read_text(encoding='utf-8'))
    if doc.get('schema') != 'looplong_boundary/1':
        raise ValueError('looplong boundary schema mismatch')
    wm = doc['wm']
    if digest(wm['checkpoint']) != wm['sha256'] or digest(doc['gate_file']) != doc['gate_sha256']:
        raise ValueError('looplong WM/gate binding changed')
    gate = json.loads(checked_path(doc['gate_file']).read_text())
    inherited = gate.get('schema') == 'c4launch_inherited_horizon/1'
    if inherited:
        from .c4launch_authority import inherited_gate
        wrapper = gate
        gate = inherited_gate(wrapper, [wm['sha256']])
        if doc['horizon_s'] != 90 or doc.get('imagination_horizon_s') != 90:
            raise ValueError('MAIN inherited horizon must stay exactly90s')
    if (not inherited and gate['wm_sha256'] != wm['sha256']) or not doc['short_nonregression']:
        raise ValueError('looplong WM/short gate mismatch')
    if not inherited and gate.get('wm_config_sha256') != digest(wm['config']):
        raise ValueError('looplong WM config binding mismatch')
    if not all(gate['horizons'].get(str(h), {}).get('sign_agreement', -1) >= gate['threshold']
               for h in (10, 30, 60)):
        raise ValueError('looplong short horizon numbers failed')
    if doc['horizon_s'] > 10 and gate['horizons'].get(str(doc['horizon_s']), {}).get('verdict') != 'PASS':
        raise ValueError('looplong selected horizon did not pass')
    target = doc.get('imagination_horizon_s', doc['horizon_s'])
    if (type(target) is not int or target not in (10, 30, 60, 90, 120, 180, 300)
            or target < doc['horizon_s']):
        raise ValueError('invalid looplong imagination horizon')
    if target > doc['horizon_s'] and not doc['flags'].get('long_credit'):
        raise ValueError('beyond-gate imagination requires calibrated long credit')
    if gate.get('threshold_source', {}).get('sha256') != doc['threshold_sha256']:
        raise ValueError('looplong threshold binding mismatch')
    if swap:
        if digest(doc['acceptance']) != doc['acceptance_sha256']:
            raise ValueError('WM acceptance hash changed')
        accepted = json.loads(checked_path(doc['acceptance']).read_text())
        if (accepted.get('decision') != 'ACCEPT' or
                accepted.get('inputs', {}).get('candidate_sha256') != wm['sha256'] or
                accepted.get('inputs', {}).get('config_sha256') != digest(wm['config'])):
            raise ValueError('only ACCEPTED WMs may hot swap')
    return doc


def configure(trainer, doc, raw, output):
    from .coopw import HorizonConfig, ExplorationConfig
    flags = doc['flags']
    target = doc.get('imagination_horizon_s', doc['horizon_s'])
    hs = [10]
    if flags.get('auto_horizon'):
        previous = doc['previous_horizon_s']
        hs = sorted({10, previous, doc['horizon_s'], target})
        position = max(1, trainer._curriculum_position(advance=False))
        inherited = raw.get('horizon', {}).get('curriculum_updates', [0, 150])
        spacing = max(1, int(inherited[-1] - inherited[-2])) if len(inherited) > 1 else 150
        knots = [0] + [position + spacing * i for i in range(len(hs) - 1)]
        base_pad = trainer.config.entry_pad_max
        pads = [max(base_pad, math.ceil(base_pad * h / trainer.config.horizon)) for h in hs]
        trainer.horizon_curriculum = HorizonConfig(enabled=True, horizons_s=hs,
            curriculum_updates=knots, entry_pad_max=pads, gate_file=doc['gate_file'])
        trainer.horizon_curriculum.bind([doc['wm']['sha256']])
    if flags.get('long_credit'):
        # Existing uncalibrated reward disagreement is superseded, not added to
        # the paired-CQR width (which would penalize unrelated absolute spread).
        trainer.long_credit_requested = True
        trainer.long_credit_trusted_s = int(doc['horizon_s'])
        from .looplong_real import RealPairCredit
        trainer.looplong_real_credit = RealPairCredit.from_boundary(doc)
        if trainer.looplong_real_credit.excluded_worlds:
            from .real_share import exclude_development_starts
            trainer.start_states = exclude_development_starts(
                trainer.start_states, trainer.looplong_real_credit.excluded_worlds)
        if target > trainer.long_credit_trusted_s:
            from .long_credit import fit_from_loop, MissingCostClocks
            path = checked_path(doc['source_policy_checkpoint'])
            if digest(path) != doc['source_policy_sha256']:
                raise ValueError('CQR source policy changed')
            payload = torch.load(path, map_location=trainer.device, weights_only=True)
            policy_state = copy.deepcopy(trainer.policy.state_dict())
            group_state = copy.deepcopy(trainer.group_encoder.state_dict())
            modes = trainer.policy.training, trainer.group_encoder.training
            refusal = None
            try:
                trainer.policy.load_state_dict(payload['policy'])
                trainer.group_encoder.load_state_dict(payload['group_encoder'])
                trainer.policy.eval()
                trainer.group_encoder.eval()
                trainer.long_credit = fit_from_loop(trainer, doc['pairs'], doc['pairing_receipt'],
                    doc['policy_returns'], doc['native_returns'], doc['source_policy_sha256'],
                    doc['wm']['sha256'], [h for h in hs if h > trainer.long_credit_trusted_s], doc['wm']['config'])
            except MissingCostClocks as error:
                refusal = {'schema': 'looplong_cqr_refusal/1', 'qualified': False, 'reason': str(error),
                           'rule': 'no CQR -> zero beyond-trusted credit; imagination capped at the '
                                   'trusted gate horizon (never raw K-sample spread)',
                           'trusted_horizon_s': trainer.long_credit_trusted_s, 'requested_horizon_s': target,
                           'pairs': doc['pairs'], 'wm_sha256': doc['wm']['sha256']}
            finally:
                trainer.policy.load_state_dict(policy_state)
                trainer.group_encoder.load_state_dict(group_state)
                trainer.policy.train(modes[0])
                trainer.group_encoder.train(modes[1])
            if refusal is not None:
                artifact = Path(output) / f"CQR_{doc['wm']['sha256'][:16]}_REFUSED.json"
                with artifact.open('x', encoding='utf-8') as stream:
                    json.dump(refusal, stream, indent=1)
                trainer.long_credit = None
                trainer.long_credit_refusal = {**refusal, 'artifact_sha256': digest(artifact)}
                if trainer.horizon_curriculum is not None:
                    kept = [h for h in hs if h <= trainer.long_credit_trusted_s]
                    trainer.horizon_curriculum = HorizonConfig(enabled=True, horizons_s=kept,
                        curriculum_updates=knots[:len(kept)], entry_pad_max=pads[:len(kept)],
                        gate_file=doc['gate_file'])
                    trainer.horizon_curriculum.bind([doc['wm']['sha256']])
                trainer.looplong_boundary = doc
                return _finish(trainer, flags, raw, doc)
            artifact = Path(output) / f"CQR_{doc['wm']['sha256'][:16]}.json"
            with artifact.open('x', encoding='utf-8') as stream:
                json.dump(trainer.long_credit, stream, indent=1, allow_nan=False)
            trainer.long_credit['artifact_sha256'] = digest(artifact)
            # Qualification belongs to the frozen collection actor, not its
            # subsequent updates (or a fresh candidate initialized elsewhere).
            trainer.long_credit_on_policy = all(
                torch.equal(value, payload[name][key])
                for name, module in (('policy', trainer.policy), ('group_encoder', trainer.group_encoder))
                for key, value in module.state_dict().items())
            trainer.long_credit_bound_update = trainer.update_count
            # This explicit bounded extension does not relabel a failed gate
            # PASS. Every beyond-gate contribution now passes through CQR.
            trainer.horizon_curriculum.allowed_stage = len(hs) - 1
            trainer.horizon_curriculum.gate_record.update(
                allowed_stage=len(hs) - 1, trusted_horizon_s=trainer.long_credit_trusted_s,
                beyond_gate_rule='paired CQR; unqualified/off-policy pessimistic-only')
        else:
            trainer.long_credit = None
    return _finish(trainer, flags, raw, doc)


def _finish(trainer, flags, raw, doc):
    from .coopw import ExplorationConfig
    if flags.get('exploration'):
        section = dict(raw.get('exploration', {}))
        section.update(enabled=True, long_persistence=True)
        trainer.exploration = ExplorationConfig(**section)
        if trainer.exploration.rho_at(trainer._curriculum_position(advance=False)) <= 0:
            raise ValueError('registered correlated exploration is not active at this boundary')
        trainer.looplong_logging = True
    trainer.looplong_boundary = doc


def maybe_swap(trainer, path, current, raw, output, load_members):
    if not path or not checked_path(path).is_file():
        return None
    request_sha = digest(path)
    if request_sha == getattr(trainer, '_looplong_swap_request_sha256', None):
        return None
    proposed = read_boundary(path, swap=True)
    if proposed['wm']['sha256'] == current['wm']['sha256']:
        trainer._looplong_swap_request_sha256 = request_sha
        return None
    if proposed.get('accepted_after_cycle', -1) <= current.get('accepted_after_cycle', -1):
        trainer._looplong_swap_request_sha256 = request_sha
        return None
    # Never change actor/optimizer/update count. Load on CPU first to avoid two
    # resident GPU WMs. All checks precede mutation; a failure stops own phase.
    from .dreamer import _rng_state, _restore_rng, _model_identity, apply_wm_physics_overrides
    rng = _rng_state(trainer.device)
    try:
        models, receipts = load_members([proposed['wm']['checkpoint']], [proposed['wm']['config']], 'cpu')
        if receipts[0]['physics'] != raw.get('_looplong_physics', receipts[0]['physics']):
            raise ValueError('hot swap physics changed')
        # CLI/sampler references may still point at the old modules. Move them
        # off device before installing the accepted member; do not rely on GC.
        for old in trainer.world_models:
            old.to('cpu')
        trainer.wm_physics_overrides = [apply_wm_physics_overrides(m, getattr(
            trainer.config, 'wm_physics_overrides', {})) for m in models]
        trainer.world_models = [m.to(trainer.device).eval().requires_grad_(False) for m in models]
        trainer.world_model_ids = [_model_identity(m, trainer.config.imagination_grad_decay)
                                   for m in trainer.world_models]
        trainer.return_low = trainer.return_high = None
        paircrit = getattr(trainer, 'paircrit', None)
        if paircrit is not None:
            from .longcritic import reliability_weights
            weights = {}
            if paircrit.config.reliability_file:
                doc = json.loads(checked_path(paircrit.config.reliability_file).read_text())
                weights = reliability_weights(doc, [proposed['wm']['sha256']])
            paircrit.engine.weights = weights
            paircrit.receipt['wm_weights'] = weights
        configure(trainer, proposed, raw, output)
    finally:
        _restore_rng(rng, trainer.device)
    event = {'event': 'accepted_wm_swap', 'swap_update': trainer.update_count,
             'first_update_on_new_wm': trainer.update_count + 1,
             'previous_wm_sha256': current['wm']['sha256'], 'wm_sha256': proposed['wm']['sha256'],
             'gate_sha256': proposed['gate_sha256']}
    with (Path(output) / 'wm_swaps.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event) + '\n')
    trainer._looplong_swap_request_sha256 = request_sha
    return proposed, receipts, event
