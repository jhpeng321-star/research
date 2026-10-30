"""Required WM-only pre-training ranking gate; never updates an optimizer or runs SUMO.

Run from the repository root with PYTHONPATH=src:. See README.md. A matched
CCT-2 authority is required for launch: measurement-only receipts never qualify training.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
import time

import numpy as np
import torch

from tools.distill.runtime import (MAPPO_SHA, WM_SHA, check_metadata,
                                  load_training_starts, safe_path, sha256)
from tools.setupaudit.summarize2 import cluster_plan, estimate
from wms.control import gated300
from wms.control.contract import RewardWeights
from wms.control.counterfactual import CounterfactualConfig, build_variants
from wms.control.coopw import ExplorationConfig, HorizonConfig
from wms.control.dreamer import DreamerConfig, _rng_state, _restore_rng
from wms.control.objective14 import authoritative_reward
from wms.control.obs import build_observations
from wms.control.policy import load_policy_components
from wms.control.policy14 import Policy14Trainer, make_controller_factory
from wms.control.real_returns14 import ScaledTeamCritic
from wms.control.safety import safe_accel
from wms.control.train_cli import GroupSignal, _read_config

ARMS = ('native', 'CTRL0', 'MAPPO_native_lanes', 'CIRCLES', 'PI', 'current', 'candB_zero')
SCHEMA = 'candB_rankcheck/1'


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def binding(raw):
    """The full resolved config, not a hand-picked subset of objective switches."""
    normalized = copy.deepcopy(raw)
    for section, key in (('horizon', 'gate_file'), ('long_credit', 'cct2_file')):
        path = normalized.get(section, {}).get(key)
        if path:
            normalized[section][key] = sha256(path.format(wm_sha256=WM_SHA))
    return canonical(normalized)


def runtime_contract(trainer):
    long = dict(trainer.gated300['config'])
    for key in ('cct1_file', 'cct2_file'):
        if long.get(key):
            long[key] = sha256(long[key].format(wm_sha256=WM_SHA))
    horizon = trainer.horizon_curriculum.as_dict()
    if horizon.get('gate_file'):
        horizon['gate_file'] = sha256(horizon['gate_file'])
    return dict(trainer=asdict(trainer.config), counterfactual=asdict(trainer.counterfactual),
        reward=asdict(trainer.reward_weights), long_credit=long,
        objective=gated300.objective_contract(trainer),
        residual_reference=trainer.policy.residual_reference,
        horizon=horizon,
        exploration=trainer.exploration.as_dict(),
        sampler_support_sha256=canonical(trainer.start_states.groups),
        long_credit_requested=trainer.long_credit_requested)


def verify_receipt(document, raw, *, wm, wm_config, trainer=None):
    if (document.get('measure_only', False)
            or document.get('schema') != SCHEMA or document.get('complete') is not True
            or document.get('passed') is not True or document.get('n_starts', 0) < 16
            or document.get('K') != 8 or document.get('horizons_s') != [90, 300]
            or document.get('mix_connected') is not True
            or document.get('config_sha256') != binding(raw)
            or document.get('wm_sha256') != wm
            or document.get('wm_config_sha256') != wm_config):
        raise ValueError('missing/failed/mismatched candidate-B ranking qualification')
    if trainer is not None:
        if runtime_contract(trainer) != document['runtime_contract']:
            raise ValueError('trained configuration differs from qualified runtime/objective')
        if not trainer.config.entrant_learned_mix or trainer.gated300['H_long'] != 300:
            raise ValueError('qualified mix/long-horizon execution is disconnected')
    sources = document.get('source_manifest', {})
    if not set(runtime_sources()).issubset(sources):
        raise ValueError('ranking source manifest is incomplete')
    for name, digest in sources.items():
        if sha256(name) != digest:
            raise ValueError('ranking source manifest mismatch: '+name)
    return document


def comparator_factory(arm, model, teacher):
    """Aligned mode-2 absolute comparator; all arms retain WM-native lanes.

    PI/CIRCLES reuse the published verdict-panel TimelineControllers bridge,
    including its control clock, sensor frames, PI memory and abstentions.
    """
    from wms.control.alwayscmd import executed, inferred_cap
    from wms.control.bridge import make_cav_controller
    from wms.control.continuous import project_native
    from wms.model.cav import EXEC_CAP_KEY, WM_LANE_TARGET_KEY, WM_MERGE_KEY

    def factory(policy, encoder, safety, record=None, **kwargs):
        acting = kwargs['act_mask']
        bridge = timeline = None
        if arm in ('MAPPO_native_lanes', 'MAPPO_speed_partial'):
            def deterministic(obs, signal, **kw):
                return teacher.policy(obs, signal, deterministic=True, **kw)
            deterministic.accel_max = teacher.policy.accel_max
            bridge = make_cav_controller(deterministic, GroupSignal(teacher.group_encoder),
                safe_accel, lane_rules=teacher.lane_rules, act_mask=acting,
                native=kwargs['native'], native_lane_change=True, native_ramp_merge='wm_law',
                policy_native_lanes=True, policy_ramp_merge=False, symmetric_exec_cap=True)

        def callback(state):
            nonlocal timeline
            # Keep DreamerTrainer's decision/CRN/telemetry bookkeeping on the
            # identical clock even though this arm supplies the executed motion.
            # Its residual output is deliberately discarded, never blended in.
            if callable(policy):
                zeros = torch.zeros_like(state['s_m'])
                obs = build_observations({**state, 'cmd_accel_mps2': zeros, 'lane_intent': zeros})
                policy(obs, encoder(obs, state))
            mask = state['present'].bool() & state['is_cav'].bool() & (state['lane'] >= 0)
            act = mask & acting
            native = project_native(state, kwargs['native'](state))
            zero = executed(state, torch.zeros_like(native), residual_reference='ctrl0')
            ctrl_rows = policy.ctrl0_rows[:, None]
            background = torch.where(ctrl_rows, zero, native)
            lane = torch.where(state['lane'] != 2, state[WM_LANE_TARGET_KEY], state['lane'])
            state[WM_MERGE_KEY] = mask & (state['lane'] == 2)
            if arm in ('MAPPO_native_lanes', 'MAPPO_speed_partial'):
                ms = {**state, EXEC_CAP_KEY: inferred_cap(state)}
                output = bridge(ms)
                acceleration = torch.maximum(-state['v_mps'] / .5,
                                              torch.minimum(output[..., 3], ms[EXEC_CAP_KEY]))
            elif arm in ('CIRCLES', 'PI'):
                from wms.train.cav_drive import TimelineControllers
                if timeline is None:
                    ids = torch.full((mask.shape[0], 301, mask.shape[1]),
                                     1 if arm == 'CIRCLES' else 2, dtype=torch.long)
                    timeline = TimelineControllers(model, dict(controller=ids,
                        acceleration_mps2=torch.zeros_like(ids, dtype=torch.float32),
                        target_speed_mps=torch.zeros_like(ids, dtype=torch.float32)))
                acceleration = timeline(state)[..., 3]
            else:
                acceleration = native if arm == 'native' else zero
            acceleration = torch.where(act, acceleration, background)
            speed = (state['v_mps'] + .5 * acceleration).clamp_min(0.)
            applied = (speed - state['v_mps']) / .5
            return torch.stack((state['s_m'] + .5 * speed, lane, speed, applied), -1)

        callback.native_lane_mask = lambda state: (state['present'].bool() & state['is_cav'].bool())
        return callback
    return factory


def install_measurement(trainer, section):
    """Use the exact requested loss without manufacturing a CCT-2 PASS artifact."""
    cfg = gated300.LongCreditConfig(**section)
    if (not cfg.enabled or cfg.mode != 'gain_only' or not cfg.ctrl0_reference
            or not cfg.require_objective_match or cfg.k16
            or not trainer.config.entrant_learned_mix
            or trainer.config.future_samples != 8 or not trainer.config.pessimistic_credit
            or not trainer.config.deterministic_crn
            or trainer.policy.residual_reference != 'ctrl0'):
        raise ValueError('measurement requires exact candidate-B config/mix/K8/CTRL0 identity')
    receipt = gated300.install(trainer, {**section, 'enabled': False}, [WM_SHA])
    receipt.update(config=asdict(cfg), H_long=300, measure_only=True,
                   policy_authorized=False, reason='measurement only; CCT-2 unqualified',
                   artifact_sha256=sha256(cfg.cct2_file.format(wm_sha256=WM_SHA)))
    return receipt


def load_initial_components(policy_path, device, checkpoint_driver=None):
    """Verify historical CRLF bytes at load; retain the staged LF driver for execution."""
    if checkpoint_driver is None:
        return load_policy_components(safe_path(policy_path), device=device)
    current = os.environ['C4_EXECUTION_DRIVER']
    old = safe_path(checkpoint_driver)
    if old.read_bytes().replace(b'\r\n', b'\n') != safe_path(current).read_bytes():
        raise ValueError('checkpoint driver differs beyond CRLF normalization')
    try:
        os.environ['C4_EXECUTION_DRIVER'] = str(old)
        return load_policy_components(safe_path(policy_path), device=device)
    finally:
        os.environ['C4_EXECUTION_DRIVER'] = current


def make_trainer(raw, model, policy_path, starts, *, device, cct2, horizon_gate,
                 measure_only=False, checkpoint_driver=None):
    policy, encoder, _, _ = load_initial_components(policy_path, device, checkpoint_driver)
    if not policy.residual_on_native:
        raise ValueError('current policy must be a native-residual checkpoint')
    policy.residual_reference = raw['policy']['residual_reference']
    policy.residual_additive_exploration = True
    encoder = encoder.encoder if isinstance(encoder, GroupSignal) else encoder
    horizon = HorizonConfig(**{**raw['horizon'], 'gate_file': str(horizon_gate)})
    horizon.bind([WM_SHA])
    if horizon.allowed_stage != len(horizon.horizons_s) - 1:
        raise ValueError('90s training curriculum is not qualified')
    trainer = Policy14Trainer([model], policy.eval(), GroupSignal(encoder).eval(),
        ScaledTeamCritic(entry_backlog=True, value_scale=raw['policy14']['critic_value_scale']),
        starts, observation_builder=build_observations, reward_fn=authoritative_reward,
        safety=safe_accel, controller_factory=make_controller_factory(
            native_ramp_merge='wm_law', native_lane_change=True),
        config=DreamerConfig(**raw['trainer']), reward_weights=RewardWeights(**raw['reward']),
        counterfactual=CounterfactualConfig(**raw['counterfactual']),
        exploration=ExplorationConfig(**raw['exploration']), horizon=horizon, device=device,
        real_return_provider=lambda **kw: None)
    trainer.long_credit_requested = True
    trainer.coop_anchor = 0
    # Prospective matched 90s regime; low-share starts still resolve to the exact
    # training low-share cap. Record that resolution rather than dropping starts.
    trainer.update_count = 1
    section = {**raw['long_credit'], 'cct2_file': str(cct2)}
    if measure_only:
        install_measurement(trainer, section)
    else:
        gated300.install(trainer, section, [WM_SHA])
    if trainer.gated300['H_long'] != 300:
        raise ValueError('matching all-population/source/K8/CTRL0/mix CCT-2 is required: '
                         + trainer.gated300['reason'])
    return trainer


def reduce_image(trainer, image):
    """Evaluate the actual dense-suffix/joint-q10 + gain-gated long objective."""
    weights = trainer.counterfactual.weights(trainer._curriculum_position(advance=False))
    loss, metrics = trainer.actor_objective(dict(image, lanes=None), 1., weights)
    # Per-future long J is recorded before gated300 truncates the trajectory.
    return dict(J90=image['rankcheck_J90'][:, 0].cpu().tolist(),
        J300=image['rankcheck_J300'][:, 0].cpu().tolist(),
        objective_loss=float(loss), objective90_loss=-metrics['gated300_dense_credit'],
        objective300_loss=float(loss), dense_credit=metrics['gated300_dense_credit'],
        long_credit=metrics['gated300_long_credit'],
        dense_horizon_s=image['imagination_horizon_s'])


def summarize(rows, arms=ARMS):
    worlds = [str(row['metadata'][0]['world']) for row in rows]
    unique, plan = cluster_plan(worlds, draws=5000, seed=20261030)
    summary = {}
    metrics = ['J90', 'J300', 'objective_loss', 'dense_credit', 'long_credit']
    if 'objective90_loss' in rows[0]['arms'][arms[0]]:
        metrics += ['objective90_loss', 'objective300_loss']
    for metric in metrics:
        values = np.array([[np.mean(row['arms'][arm][metric]) for arm in arms] for row in rows])
        if not np.isfinite(values).all():
            raise ValueError('nonfinite ranking measurement')
        summary[metric] = dict(arms=estimate(values, worlds, unique, plan),
            vs_CTRL0=estimate(values-values[:, 1:2], worlds, unique, plan))
    return summary, len(unique)


def ranking_gate(summary, real_evidence=None):
    # Owner R2, registered 2026-10-30 16:30: speed-only proxy at/below CTRL0,
    # 300s CI on the correct side of zero. 90s and full objective stay reported.
    proxy = ARMS.index('MAPPO_native_lanes')
    if summary['J300']['vs_CTRL0']['hi'][proxy] <= 0:
        return True, 'owner R2: MAPPO native-lanes upper95 dJ300 <= 0'
    if real_evidence:
        # Alternative needs hash-bound *real* speed-only baseline evidence, not
        # the historical lane-on MAPPO contrast or a WM-only assertion.
        for arm in ('CIRCLES', 'PI', 'MAPPO_native_lanes'):
            row = real_evidence.get('arms', {}).get(arm, {})
            if (real_evidence.get('measured_real_sumo') is True
                    and row.get('reference') == 'CTRL0' and row.get('native_lanes') is True
                    and row.get('upper95_dJ', float('inf')) < 0
                    and summary['J300']['vs_CTRL0']['hi'][ARMS.index(arm)] < 0):
                return True, 'owner R2 matched real speed-only evidence: ' + arm
    return False, 'owner R2 ranking not demonstrated'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/c4_policy14_candB.toml')
    for name in ('wm', 'wm-config', 'mappo', 'current-policy', 'cct2', 'horizon-gate',
                 'boundary', 'pairing-receipt', 'out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--real-evidence')
    p.add_argument('--measure-only', action='store_true',
                   help='score the requested objective without launch authority; receipt cannot launch')
    p.add_argument('--mappo-speed', help='additional partially trained speed-only exported actor')
    p.add_argument('--current-policy-driver', help='original checkpoint-bound driver; CRLF-only migration')
    p.add_argument('--gpu-memory-fraction', type=float, default=.40)
    p.add_argument('--n-starts', type=int, default=16)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    a = p.parse_args(argv)
    if a.n_starts < 16:
        p.error('at least 16 consecutive TRAIN sampler starts are required')
    if a.current_policy_driver and not a.measure_only:
        p.error('--current-policy-driver is only available for unlaunchable measurements')
    if not 0 < a.gpu_memory_fraction <= .40:
        p.error('ranking GPU memory fraction must be in (0, .40]')
    paths = [safe_path(v) for k, v in vars(a).items() if isinstance(v, str)
             and k not in ('device',)]
    del paths
    if sha256(a.wm) != WM_SHA or sha256(a.mappo) != MAPPO_SHA:
        raise ValueError('registered WM2/MAPPO checkpoint hash mismatch')
    qualification = json.loads(safe_path(a.cct2).read_text())
    if not a.measure_only and qualification.get('wm_config_sha256') != sha256(a.wm_config):
        raise ValueError('WM qualification must bind the exact inference config SHA256')
    out = safe_path(a.out)
    if out.exists() or not out.parent.is_dir():
        raise ValueError('out must be a fresh file in an existing directory')
    from tools.distill.runtime import configure_execution
    from wms.control.mappo.eval_bridge import load_policy_fn
    from wms.control.policy14_cli import load_members
    configure_execution('configs/execparity_cav_idm_v1.json')
    torch.set_num_threads(2)
    torch.manual_seed(965140)
    if a.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction)
    sources = {name: sha256(name) for name in runtime_sources()}
    from pathlib import Path
    if Path('SOURCE_MANIFEST.json').is_file():
        if sources != json.loads(safe_path('SOURCE_MANIFEST.json').read_text()):
            raise ValueError('staged source manifest mismatch')
    raw = _read_config(safe_path(a.config))
    raw['horizon']['gate_file'] = str(safe_path(a.horizon_gate))
    from wms.control.looplong import configure, read_boundary
    prepared = read_boundary(a.boundary)
    if (prepared['wm']['sha256'] != WM_SHA
            or sha256(prepared['wm']['config']) != sha256(a.wm_config)):
        raise ValueError('boundary WM/config identity mismatch')
    if prepared.get('coop_credit') is False:
        for key in ('w_individual', 'w_group', 'w_coop', 'w_team'):
            raw['counterfactual'][key] = [0.] * len(raw['counterfactual'][key])
    if sha256(raw['long_credit']['cct2_file'].format(wm_sha256=WM_SHA)) != sha256(a.cct2):
        raise ValueError('explicit CCT-2 must equal the preset training authority')
    model = load_members([a.wm], [a.wm_config], a.device)[0][0]
    starts, _ = load_training_starts(raw, model, 965140)
    trainer = make_trainer(raw, model, a.current_policy, starts, device=a.device,
                           cct2=a.cct2, horizon_gate=a.horizon_gate,
                           measure_only=a.measure_only, checkpoint_driver=a.current_policy_driver)
    payload = torch.load(safe_path(a.current_policy), map_location='cpu', weights_only=False)
    trainer.update_count = int(payload['update_count'])
    trainer.coop_anchor = payload.get('coopw', {}).get('anchor', 0)
    from wms.control.real_share import exclude_development_starts
    pairing = json.loads(safe_path(a.pairing_receipt).read_text())
    if pairing.get('schema') != 'paircrit_pairs/1' or pairing.get('split') != 'TRAIN':
        raise ValueError('TRAIN paired-return sampler authority required')
    excluded = {int(row['seed']) for row in pairing['pairs']
                if int(hashlib.sha256(f"965140:{int(row['seed'])}".encode()).hexdigest(), 16) % 5 == 0}
    trainer.start_states = exclude_development_starts(starts, excluded)
    from wms.control.dreamer import _same_groups
    sampler_restored = (payload.get('sampler') is not None
                        and _same_groups(trainer.start_states, payload['sampler']))
    if sampler_restored:
        trainer.start_states.load_state_dict(payload['sampler'])
    boundary_raw = copy.deepcopy(raw)
    if a.measure_only:
        boundary_raw['long_credit']['enabled'] = False
    configure(trainer, prepared, boundary_raw, out.parent)
    if a.measure_only:
        install_measurement(trainer, raw['long_credit'])
    if trainer.gated300['H_long'] != 300:
        raise ValueError('boundary rebind lost objective-qualified 300s credit')
    if trainer._tilt_sampler is not None and payload.get('coopw', {}).get('tilt_sampler'):
        trainer._tilt_sampler.load_state_dict(payload['coopw']['tilt_sampler'])
    trainer.start_states.configure_focus_cells(raw['data']['focus_cell_fraction'])
    first_update = trainer.update_count
    teacher = load_policy_fn(a.mappo, device=a.device)
    speed = load_policy_fn(a.mappo_speed, device=a.device) if a.mappo_speed else None
    speed_payload = (torch.load(safe_path(a.mappo_speed), map_location='cpu', weights_only=False)
                     if speed is not None else None)
    if speed is not None and (speed_payload.get('lane_control') != 'native'
            or any(k.startswith('lane_head.') for k in speed_payload['policy'])):
        raise ValueError('additional MAPPO checkpoint must be genuinely speed-only')
    arms = (*ARMS, 'MAPPO_speed_partial') if speed is not None else ARMS
    signature = runtime_contract(trainer)
    from wms.core.speed_flags import active as speed_flags
    doc = dict(schema=SCHEMA, complete=False, passed=False, measure_only=a.measure_only,
        launch_authorized=False, gpu_memory_fraction=a.gpu_memory_fraction,
        execution_speed_flags=speed_flags(),
        K=8, horizons_s=[90, 300],
        n_starts=a.n_starts, wm_sha256=WM_SHA, wm_config_sha256=sha256(a.wm_config),
        mappo_sha256=MAPPO_SHA, current_policy_sha256=sha256(a.current_policy),
        config_sha256=binding(raw), runtime_contract=signature,
        cct2_sha256=sha256(a.cct2), horizon_gate_sha256=sha256(a.horizon_gate),
        boundary_sha256=sha256(a.boundary), pairing_receipt_sha256=sha256(a.pairing_receipt),
        source_manifest=sources,
        sampler_provenance=starts.distill_provenance, sampler_restored=sampler_restored,
        arms=list(arms), rows=[])
    if speed is not None:
        doc['mappo_speed_partial'] = dict(path=a.mappo_speed, sha256=sha256(a.mappo_speed),
            iteration=speed_payload.get('iteration'), partially_trained=True)
    if a.current_policy_driver:
        doc['initialization_driver'] = dict(path=a.current_policy_driver,
            sha256=sha256(a.current_policy_driver), normalization='CRLF to LF, byte-identical otherwise',
            execution_sha256=sha256(os.environ['C4_EXECUTION_DRIVER']))
    # Preserve full-horizon per-future physical J before prefix truncation.
    original_prefix = gated300.prefix
    original_rollout = model.rollout
    mix_calls = []
    def rollout(*args, **kwargs):
        sampler = kwargs.get('entrant_driver_sampler')
        if sampler is None:
            raise ValueError('learned mix is not connected to the WM rollout')
        began = time.monotonic()
        result = original_rollout(*args, **kwargs)
        mix_calls.append(list(sampler.assigned))
        print('FUTURE', len(mix_calls), 'rollout_s', round(time.monotonic()-began, 3), flush=True)
        return result
    model.rollout = rollout
    def prefix(owner, imagined, dense_steps):
        cost = imagined['components']['delay'] + imagined['components']['source_wait']
        full = torch.stack([cost[..., r].contiguous().sum(1)
                            for r in range(cost.shape[-1])], -1)
        ninety = torch.stack([cost[:, :90, r].contiguous().sum(1)
                              for r in range(cost.shape[-1])], -1)
        return {**original_prefix(owner, imagined, dense_steps),
                'rankcheck_J300': full, 'rankcheck_J90': ninety}
    gated300.prefix = prefix
    original_factory = trainer.controller_factory
    saved_head = copy.deepcopy(trainer.policy.residual_head.state_dict())
    try:
        _restore_rng(payload['rng'], trainer.device)
        trainer._training_imagination = True
        for index in range(a.n_starts):
            # Exactly the training sampler/headroom rejection path, prospective
            # update-index seeds; no sorting, support filtering or outcome selection.
            trainer.update_count = first_update + index
            batch = trainer._sample_starts(trainer.start_states)
            check_metadata(batch.metadata)
            batch = trainer._entry_slots(batch, torch.Generator().manual_seed(
                1_000_003*(trainer.update_count+1)))
            variants = build_variants(batch.state, trainer.counterfactual)
            before = _rng_state(trainer.device)
            row = dict(metadata=batch.metadata, sampler_info=trainer._last_start_info, arms={})
            doc['rows'].append(row)
            for arm in arms:
                began = time.monotonic()
                _restore_rng(before, trainer.device)
                trainer.policy.residual_head.load_state_dict(saved_head)
                if arm in ('current', 'candB_zero'):
                    trainer.controller_factory = original_factory
                    if arm == 'candB_zero':
                        for parameter in trainer.policy.residual_head.parameters():
                            torch.nn.init.zeros_(parameter)
                else:
                    trainer.controller_factory = comparator_factory(
                        arm, model, speed if arm == 'MAPPO_speed_partial' else teacher)
                with torch.no_grad():
                    image = trainer.imagine(copy.deepcopy(batch), variants=variants,
                        deterministic=True, future_seed=trainer.update_count,
                        _memory_replay=True)
                    row['arms'][arm] = reduce_image(trainer, image)
                del image
                row['arms'][arm]['elapsed_s'] = time.monotonic()-began
                out.write_text(json.dumps(doc, indent=1, allow_nan=False)+'\n', encoding='utf-8')
                print('ARM', index+1, arm, 'elapsed_s', round(time.monotonic()-began, 3), flush=True)
            print('START', index+1, batch.metadata, flush=True)
    finally:
        gated300.prefix = original_prefix
        model.rollout = original_rollout
        trainer.policy.residual_head.load_state_dict(saved_head)
    doc['summary'], doc['n_worlds'] = summarize(doc['rows'], arms)
    real = json.loads(safe_path(a.real_evidence).read_text()) if a.real_evidence else None
    doc['ranking_passed'], doc['gate_reason'] = ranking_gate(doc['summary'], real)
    doc['passed'] = doc['ranking_passed'] and not a.measure_only
    if real:
        doc['real_evidence_sha256'] = sha256(a.real_evidence)
    doc['complete'] = True
    doc['mix_connected'] = bool(mix_calls)
    doc['mix_assignment_receipts'] = mix_calls
    out.write_text(json.dumps(doc, indent=1, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['arm | J90 | dJ90 [95% CI] | J300 | dJ300 [95% CI] | objective loss']
    for i, arm in enumerate(arms):
        fields = [arm]
        for h in ('J90', 'J300'):
            s = doc['summary'][h]
            d = s['vs_CTRL0']
            fields.extend((f"{s['arms']['mean'][i]:.3f}",
                f"{d['mean'][i]:+.3f} [{d['lo'][i]:+.3f},{d['hi'][i]:+.3f}]"))
        fields.append(f"{doc['summary']['objective_loss']['arms']['mean'][i]:.3f}")
        lines.append(' | '.join(fields))
    text = '\n'.join(lines)+'\n'+doc['gate_reason']+'\n'
    safe_path(str(out)+'.txt').write_text(text, encoding='utf-8')
    print(text)
    return 0 if doc['ranking_passed'] else 2


def runtime_sources():
    from pathlib import Path
    staged = Path('SOURCE_MANIFEST.json')
    if staged.is_file():
        return sorted(json.loads(safe_path(staged).read_text()))
    from tools.candA.authority import MANIFEST_FILES, LEADER_RULE_FILES
    return sorted(set(MANIFEST_FILES + LEADER_RULE_FILES + [
        'src/wms/control/residual.py', 'src/wms/control/entrant_mix.py',
        'src/wms/train/cav_drive.py', 'tools/errsrc/inference.py',
        'output/controller_verification_v2/pi_saturation_v1.py',
        'output/controller_verification_v4/pi_external_u.py',
        'output/controller_verification_v7/pi_owner_adapter.py',
        'tools/candB/rankcheck.py', 'configs/execparity_cav_idm_v1.json']))


if __name__ == '__main__':
    raise SystemExit(main())
