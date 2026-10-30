"""Required WM-only pre-training ranking gate; never updates an optimizer or runs SUMO.

Run from the repository root with PYTHONPATH=src:. See README.md. A matched
CCT-2 authority is required: a ranking receipt cannot manufacture WM qualification.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json

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
        long_credit_requested=trainer.long_credit_requested)


def verify_receipt(document, raw, *, wm, wm_config, trainer=None):
    if (document.get('schema') != SCHEMA or document.get('complete') is not True
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
    for name, digest in document.get('source_manifest', {}).items():
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
        if arm == 'MAPPO_native_lanes':
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
            if arm == 'MAPPO_native_lanes':
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


def make_trainer(raw, model, policy_path, starts, *, device, cct2, horizon_gate):
    policy, encoder, _, _ = load_policy_components(safe_path(policy_path), device=device)
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
    gated300.install(trainer, {**raw['long_credit'], 'cct2_file': str(cct2)}, [WM_SHA])
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
        objective_loss=float(loss), dense_credit=metrics['gated300_dense_credit'],
        long_credit=metrics['gated300_long_credit'],
        dense_horizon_s=image['imagination_horizon_s'])


def summarize(rows):
    worlds = [str(row['metadata'][0]['world']) for row in rows]
    unique, plan = cluster_plan(worlds, draws=5000, seed=20261030)
    summary = {}
    for metric in ('J90', 'J300', 'objective_loss', 'dense_credit', 'long_credit'):
        values = np.array([[np.mean(row['arms'][arm][metric]) for arm in ARMS] for row in rows])
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
    p.add_argument('--n-starts', type=int, default=16)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    a = p.parse_args(argv)
    if a.n_starts < 16:
        p.error('at least 16 consecutive TRAIN sampler starts are required')
    paths = [safe_path(v) for k, v in vars(a).items() if isinstance(v, str)
             and k not in ('device',)]
    del paths
    if sha256(a.wm) != WM_SHA or sha256(a.mappo) != MAPPO_SHA:
        raise ValueError('registered WM2/MAPPO checkpoint hash mismatch')
    out = safe_path(a.out)
    if out.exists() or not out.parent.is_dir():
        raise ValueError('out must be a fresh file in an existing directory')
    from tools.distill.runtime import configure_execution
    from wms.control.mappo.eval_bridge import load_policy_fn
    from wms.control.policy14_cli import load_members
    configure_execution('configs/execparity_cav_idm_v1.json')
    torch.set_num_threads(2)
    torch.manual_seed(965140)
    raw = _read_config(safe_path(a.config))
    raw['horizon']['gate_file'] = str(safe_path(a.horizon_gate))
    if sha256(raw['long_credit']['cct2_file'].format(wm_sha256=WM_SHA)) != sha256(a.cct2):
        raise ValueError('explicit CCT-2 must equal the preset training authority')
    model = load_members([a.wm], [a.wm_config], a.device)[0][0]
    starts, _ = load_training_starts(raw, model, 965140)
    starts.configure_focus_cells(raw['data']['focus_cell_fraction'])
    trainer = make_trainer(raw, model, a.current_policy, starts, device=a.device,
                           cct2=a.cct2, horizon_gate=a.horizon_gate)
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
    if payload.get('sampler') is not None and _same_groups(trainer.start_states, payload['sampler']):
        trainer.start_states.load_state_dict(payload['sampler'])
    from wms.control.looplong import configure, read_boundary
    configure(trainer, read_boundary(a.boundary), raw, out.parent)
    if trainer.gated300['H_long'] != 300:
        raise ValueError('boundary rebind lost objective-qualified 300s credit')
    if trainer._tilt_sampler is not None and payload.get('coopw', {}).get('tilt_sampler'):
        trainer._tilt_sampler.load_state_dict(payload['coopw']['tilt_sampler'])
    first_update = trainer.update_count
    teacher = load_policy_fn(a.mappo, device=a.device)
    signature = runtime_contract(trainer)
    doc = dict(schema=SCHEMA, complete=False, passed=False, K=8, horizons_s=[90, 300],
        n_starts=a.n_starts, wm_sha256=WM_SHA, wm_config_sha256=sha256(a.wm_config),
        mappo_sha256=MAPPO_SHA, current_policy_sha256=sha256(a.current_policy),
        config_sha256=binding(raw), runtime_contract=signature,
        cct2_sha256=sha256(a.cct2), horizon_gate_sha256=sha256(a.horizon_gate),
        boundary_sha256=sha256(a.boundary), pairing_receipt_sha256=sha256(a.pairing_receipt),
        source_manifest={name: sha256(name) for name in runtime_sources()},
        sampler_provenance=starts.distill_provenance, arms=list(ARMS), rows=[])
    # Preserve full-horizon per-future physical J before prefix truncation.
    original_prefix = gated300.prefix
    original_rollout = model.rollout
    mix_calls = []
    def rollout(*args, **kwargs):
        sampler = kwargs.get('entrant_driver_sampler')
        if sampler is None:
            raise ValueError('learned mix is not connected to the WM rollout')
        result = original_rollout(*args, **kwargs)
        mix_calls.append(list(sampler.assigned))
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
            for arm in ARMS:
                _restore_rng(before, trainer.device)
                trainer.policy.residual_head.load_state_dict(saved_head)
                if arm in ('current', 'candB_zero'):
                    trainer.controller_factory = original_factory
                    if arm == 'candB_zero':
                        for parameter in trainer.policy.residual_head.parameters():
                            torch.nn.init.zeros_(parameter)
                else:
                    trainer.controller_factory = comparator_factory(arm, model, teacher)
                with torch.no_grad():
                    image = trainer.imagine(copy.deepcopy(batch), variants=variants,
                        deterministic=True, future_seed=trainer.update_count,
                        _memory_replay=True)
                    row['arms'][arm] = reduce_image(trainer, image)
                del image
            doc['rows'].append(row)
            out.write_text(json.dumps(doc, indent=1, allow_nan=False)+'\n', encoding='utf-8')
            print('START', index+1, batch.metadata, flush=True)
    finally:
        gated300.prefix = original_prefix
        model.rollout = original_rollout
        trainer.policy.residual_head.load_state_dict(saved_head)
    doc['summary'], doc['n_worlds'] = summarize(doc['rows'])
    real = json.loads(safe_path(a.real_evidence).read_text()) if a.real_evidence else None
    doc['passed'], doc['gate_reason'] = ranking_gate(doc['summary'], real)
    if real:
        doc['real_evidence_sha256'] = sha256(a.real_evidence)
    doc['complete'] = True
    doc['mix_connected'] = bool(mix_calls)
    doc['mix_assignment_receipts'] = mix_calls
    out.write_text(json.dumps(doc, indent=1, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['arm | J90 | dJ90 [95% CI] | J300 | dJ300 [95% CI] | objective loss']
    for i, arm in enumerate(ARMS):
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
    return 0 if doc['passed'] else 2


def runtime_sources():
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
