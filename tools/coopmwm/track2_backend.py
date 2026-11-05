"""P1 production adapter for the clean Track-2 run (explicit opt-in only).

Public API: build_arm_runtime(cfg, arm) -> hooks; collect_update(cfg, arm, u)
-> episode receipts. ``hooks['train_update'](episodes, rewards=...)`` is called
AFTER hooks['safety']. P2 track2_reward supplies per-CAV observable-neighbourhood
SUMO rewards; dense WM teaching and counterfactual WM credit are disabled.
A uses the unchanged MAPPO PPO path.

Required config: run_root, training_seed, worlds_by_update (TRAIN worlds),
runtime/checkpoint/accel_bounds/gpu_deadline/workers/load_max, budget={total,
phases}, and track2_backend=True. B/R additionally require layer_config (the
LayerConfig fields except feature_dim/enabled), trigger_candidates,
trigger_partner_slots, decision_interval_s, trigger_lr, trigger_ppo,
layer_epochs. Configuration is common to B and R. A has no cooperation/bank.

The real accel_runner wave retains reservation, admission, spawned workers and
failure replacement. Its accel_plant worker installs episode_components below.
Bank pairs, effective imitation, and frozen practice plans are the coop3 readers,
not reconstructed labels. All physics stays behind OWNER_APPROVAL.json and the
parent's admission gates. This module never launches during import/build.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
import types

import numpy as np
import torch

from . import moves
from .acceleration import PHI
from .accel_learner import GAMMA, UNIT
from .coop_layer import CooperationLayer, LayerConfig
from .merge_features import (MERGE_DIM, MERGE_FEATURES, MergeFeaturesConfig,
                             ObservedVehicle, SENSOR_RANGE_M, SensorReport,
                             merge_features, _valid_report, _local_rows,
                             CONTROL_DT_S, V2V_K, V2V_RANGE_M)
from .trigger_policy import MoveExecutor, TriggerBatch, TriggerPolicy, ppo_update

SCHEMA = 'coopmwm-track2-backend-v1'
FEATURE_DIM = 29 + MERGE_DIM + 1  # local sensor, separate merge block, active move offset
ARMS = ('A', 'B', 'R')
REWARD_SCHEMA = 'coopmwm-track2-reward-v1'


class TriggerMove:
    """Bridge T1A onset/hold to T2A's onset_s/hold_s without editing either API."""
    def __init__(self, move):
        self.move = move

    @property
    def onset_s(self):
        return self.move.onset

    @property
    def hold_s(self):
        return self.move.hold

    def __getattr__(self, name):
        # deepcopy probes uninitialised objects, so don't recurse through getattr.
        if 'move' not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__['move'], name)


class MoveAPI:
    FAMILIES = moves.FAMILIES

    @staticmethod
    def candidates(obs_state, rng, n):
        return [TriggerMove(m) for m in moves.candidates(obs_state, rng, n)]

    @staticmethod
    def offsets(move, t):
        return moves.offsets(move.move, t)


def _path(value):
    path = Path(value)
    for candidate in (str(path), str(path.resolve())):
        text = candidate.replace('\\', '/').lower()
        if (any(s in text for s in ('holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
                                     'hetdata_truth', 'vehicle_params', 'vtype', 'driver'))
                or 'sealed' in text.split('/')):
            raise ValueError('restricted input path')
    return path


def _check(cfg, arm):
    if arm not in ARMS:
        raise ValueError('arm must be A, B or R')
    if not cfg.get('track2_backend'):
        raise ValueError('track2_backend explicit opt-in required')
    if cfg.get('device', 'cpu') != 'cpu' or cfg.get('coop_reward'):
        raise ValueError('CPU backend; additional coop_reward must be off')
    if not cfg.get('v2v'):
        raise ValueError('production backend requires the real V2V runtime')
    if cfg.get('host', 'cpuV') != 'cpuV':
        raise ValueError('registered host is cpuV')
    if cfg.get('dense_teacher') or cfg.get('group_credit'):
        raise ValueError('clean run requires dense teaching and WM credit OFF')
    _path(cfg['run_root'])
    if arm != 'A':
        if not cfg.get('merge_features', False):
            raise ValueError('B/R require observable merge features')
        if type(cfg['decision_interval_s']) is not int or cfg['decision_interval_s'] < 1:
            raise ValueError('positive registered decision interval required')
        reward = cfg.get('track2_reward', {})
        if (set(reward) != {'weight', 'provenance'}
                or type(reward['weight']) not in (int, float)
                or not math.isfinite(reward['weight'])
                or not isinstance(reward['provenance'], str) or not reward['provenance'].strip()):
            raise ValueError('common B/R track2_reward weight and registration provenance required')


def _batch_dict(batch):
    return dict(context=batch.context,
        tables=[[None if m is None else m.to_dict() for m in table] for table in batch.tables],
        descriptors=batch.descriptors.tolist(), mask=batch.mask.tolist(),
        actions=batch.actions.tolist(), logp=batch.logp.tolist(), values=batch.values.tolist())


def _batch(record):
    return TriggerBatch(record['context'], tuple(tuple(None if m is None else moves.Move.from_dict(m)
        for m in table) for table in record['tables']),
        torch.tensor(record['descriptors'], dtype=torch.float32),
        torch.tensor(record['mask'], dtype=torch.bool),
        torch.tensor(record['actions'], dtype=torch.long),
        torch.tensor(record['logp'], dtype=torch.float32),
        torch.tensor(record['values'], dtype=torch.float32))


def _features(context):
    """No RSU/broadcast or globally conditioned base intent enters the layer."""
    rows = []
    for vid, x in zip(context['members'], context['x']):
        block = context.get('merge_features', {}).get(vid)
        if block is None:
            # Bank records without sensor packets remain explicitly missing.
            block = merge_features(None, vid, now_s=0, cfg=MergeFeaturesConfig(True)).tolist()
        rows.append(list(x[:29]) + list(block) + [context.get('move_offsets', {}).get(vid, 0.)])
    result = torch.tensor(rows, dtype=torch.float32).reshape(len(rows), FEATURE_DIM)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('nonfinite local cooperation features')
    return result


class ProductionLayer(CooperationLayer):
    """Full-strength committed moves plus a zero-initialized learned residual.

    F1: the last feature records moves.offsets for the already selected move.
    That offset is added with fixed coefficient 1 from update 0; it is never
    gated by the legacy trigger_gain checkpoint slot. Only the additive residual
    is learned. Sampling, forward/imitation, scores and KL use the same mean.
    T2A's categorical PPO is updated separately on decision samples only.
    """
    def _features(self, context):
        return _features(context)

    def _distribution(self, features, move_offsets=None):
        residual, logits, _, offsets = super()._distribution(features, move_offsets)
        return residual + features[:, -1], logits, residual.new_ones(()), offsets

    def forward(self, context):
        """The same committed-offset + residual mean used by live sampling."""
        self._check_frozen()
        with torch.no_grad():
            mean, value = self.base(context)
        if not self.enabled:
            return mean, value
        correction = self._distribution(self._features(context))[0][:, None]
        return torch.where(correction == 0, mean, mean + correction), value

    def member_scores(self, sample, *, old=False):
        """Marginal Gaussian scores, retaining each CAV's own learning target.

        The inherited exact correlated JOINT KL remains the update trust region.
        A joint score times a mean reward would discard neighbourhood alignment.
        """
        record = sample['coop_layer']
        features = self._features(sample['context'])
        if old:
            residual = features.new_tensor(record['old_residual'])
        else:
            residual = self._distribution(features, record['offsets'])[0]
        with torch.no_grad():
            mean = self.base(sample['context'])[0].reshape(-1)
        mean = mean + residual + PHI * features.new_tensor(sample['previous_noise']).reshape(-1)
        sigma = self.base.log_std.detach().exp().reshape(()) * math.sqrt(1-PHI**2)
        action = features.new_tensor(sample['own_action']).reshape(-1)
        return torch.distributions.Normal(mean, sigma).log_prob(action)

    def update_rewards(self, records, *, epochs):
        """Layer-only per-CAV return PPO; transactional inherited joint KL cap."""
        self._check_frozen()
        self._check_optimizer()
        if type(epochs) is not int or epochs < 1 or not records:
            raise ValueError('positive epochs and per-CAV reward records required')
        parameters = self.layer_parameters()
        def kl():
            return torch.stack([self.joint_kl(r['sample']) for r in records]).mean()
        with torch.no_grad():
            initial = float(kl())
        if not math.isfinite(initial) or initial > self.config.kl_budget:
            raise ValueError('rollout already exceeds registered layer KL budget')
        metrics = []
        for _ in range(epochs):
            weights = [p.detach().clone() for p in parameters]
            optimizer = deepcopy(self.optimizer.state_dict())
            losses = []
            for record in records:
                sample = record['sample']
                scores = self.member_scores(sample)
                old_scores = self.member_scores(sample, old=True).detach()
                advantage = scores.new_tensor(record['returns'])
                if advantage.shape != scores.shape or not bool(torch.isfinite(advantage).all()):
                    raise ValueError('per-CAV reward alignment/nonfinite target')
                ratio = (scores-old_scores).exp()
                losses.append(-torch.minimum(ratio*advantage,
                    ratio.clamp(1-self.config.clip_ratio, 1+self.config.clip_ratio)*advantage).mean())
            loss = torch.stack(losses).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm,
                                                  error_if_nonfinite=True)
            try:
                self.optimizer.step()
                with torch.no_grad():
                    current = float(kl())
                accepted = math.isfinite(current) and current <= self.config.kl_budget
            except Exception:
                with torch.no_grad():
                    for p, old in zip(parameters, weights):
                        p.copy_(old)
                self.optimizer.load_state_dict(optimizer)
                self.optimizer.zero_grad(set_to_none=True)
                raise
            if not accepted:
                with torch.no_grad():
                    for p, old in zip(parameters, weights):
                        p.copy_(old)
                self.optimizer.load_state_dict(optimizer)
            self.optimizer.zero_grad(set_to_none=True)
            metrics.append(dict(loss=float(loss.detach()), kl=current, accepted=accepted,
                                gradient_norm=float(norm), target='per_cav_observable_sumo_return'))
            if not accepted:
                break
        self._check_frozen()
        return metrics


class EpisodePolicy(torch.nn.Module):
    def __init__(self, base, cfg, arm):
        super().__init__()
        self.cfg, self.arm = deepcopy(cfg), arm
        config = LayerConfig(**dict(cfg['layer_config'], feature_dim=FEATURE_DIM, enabled=True))
        self.layer = ProductionLayer(base, config)
        # Retain the legacy slot/optimizer layout for existing sidecars. It is
        # unused by ProductionLayer, gets no gradient and cannot gate actuation.
        self.layer.trigger_gain.requires_grad_(True)
        self.layer.optimizer = torch.optim.Adam(self.layer.layer_parameters(), lr=config.learning_rate)
        self.trigger = TriggerPolicy(self.layer.base, v2v_dim=cfg['v2v_dim'],
            n_candidates=cfg['trigger_candidates'], partner_slots=cfg['trigger_partner_slots'],
            freeze_base=True, moves_api=MoveAPI())
        self.trigger_optimizer = self.trigger.optimizer(lr=cfg['trigger_lr'])
        self.layer_rng = torch.Generator().manual_seed(int(cfg['training_seed']))
        self.candidate_rng = np.random.default_rng(int(cfg['training_seed']))
        self.executor = MoveExecutor(moves_api=MoveAPI())
        self.now, self.last_decision = None, None
        self.local_states, self.visible, self.blocks = {}, {}, {}
        self.reports, self.trigger_board = {}, {}

    @property
    def log_std(self):
        return self.layer.base.log_std

    def intent_base_policy(self):
        return self.layer.base

    def forward(self, context):
        return self.layer(context)

    def score(self, sample):
        return self.layer.score(sample)

    def observe(self, rows, now):
        """Whitelist local geometry before passing anything to move/merge APIs."""
        from .bank_v2v import _intent_features, DEFAULT_CFG
        table = {r['vid']: {k: r[k] for k in ('vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m')}
                 for r in rows}
        cavs = sorted(v for v, r in table.items() if r['is_cav'])
        reports, states, visible, blocks, local_contexts = {}, {}, {}, {}, {}
        execution_positions, neighbourhoods = {}, {}
        for vid in cavs:
            ego = table[vid]
            sensed = [r for r in table.values() if abs(r['s_m']-ego['s_m']) <= SENSOR_RANGE_M]
            partners = sorted((r['vid'] for r in sensed if r['is_cav'] and r['vid'] != vid),
                key=lambda v: (abs(table[v]['s_m']-ego['s_m']), v))[:self.trigger.partner_slots-1]
            # Only selected CAVs enter candidates; sensed HDVs are geometry only.
            local = [r for r in sensed if not r['is_cav'] or r['vid'] in [vid]+partners]
            states[vid], visible[vid] = dict(self=vid, rows=local), [vid]+partners
            vehicle = lambda r: ObservedVehicle(r['vid'], r['s_m'], int(r['lane']), r['v_mps'], r['length_m'])
            reports[vid] = SensorReport(float(now), vehicle(ego), (0, 1, 2),
                                       tuple(vehicle(r) for r in sensed if r['vid'] != vid))
            # Exactly the merge-feature packet authorization: t-1, selected
            # publisher, registered radio range/count, direct sensing only.
            accepted = []
            for partner in partners:
                report = self.reports.get(partner)
                if _valid_report(report, partner, float(now)-CONTROL_DT_S):
                    distance = abs(report.source.s_m-ego['s_m'])
                    if distance <= V2V_RANGE_M:
                        accepted.append((distance, partner, report))
            accepted.sort(key=lambda entry: (entry[0], entry[1]))
            positions = {r['vid']: r['s_m'] for r in sensed}
            for _, _, report in accepted[:V2V_K]:
                for observed in _local_rows(report):
                    positions.setdefault(observed.vehicle_id, observed.s_m)
            execution_positions[vid] = positions
            neighbourhoods[vid] = sorted(set(positions)-{vid})
            blocks[vid] = merge_features(reports[vid], vid, self.reports, now_s=float(now),
                partner_ids=partners, cfg=MergeFeaturesConfig(True)).tolist()
            if getattr(self.layer.base, 'intent_horizon', 0):
                # Local-only, t-1 intent board, distinct from base's RSU-conditioned board.
                local_contexts[vid] = _intent_features(dict(time_s=now, vehicles=local), vid, {},
                    DEFAULT_CFG, dict(horizon=self.layer.base.intent_horizon, board=self.trigger_board))
        self.now, self.reports = now, reports
        self.local_states, self.visible, self.blocks = states, visible, blocks
        self.local_intents = local_contexts
        self.execution_positions, self.neighbourhoods = execution_positions, neighbourhoods

    @torch.no_grad()
    def sample(self, context, generator, deterministic=False, memory=None):
        if self.now is None:
            raise ValueError('production layer requires controller observation before sampling')
        context = dict(context, merge_features={v: self.blocks[v] for v in context['members']})
        if getattr(self.layer.base, 'intent_horizon', 0):
            context['trigger_v2v'] = {v: self.local_intents[v] for v in context['members']}
        decision = None
        if (context['members'] and (self.last_decision is None
                or self.now-self.last_decision >= self.cfg['decision_interval_s'])):
            decision = self.trigger.decide(context, obs_states=self.local_states,
                observable_cavs=self.visible, rng=self.candidate_rng, generator=self.layer_rng,
                deterministic=deterministic)
            accepted = {}
            for vid, move in sorted(decision.selected().items()):
                accepted[vid] = self.executor.start(move, ego=vid, observable_cav_ids=self.visible[vid],
                    now=self.now, until=self.now+move.onset+move.hold) if move is not None else False
            self.last_decision = self.now
        self.executor._advance(self.now)
        offsets = {}
        for entry in self.executor.active:
            for vid in entry['move'].members:
                if vid in context['members']:
                    # Missing partner evidence does not trigger early release:
                    # keep the registered commitment until locally witnessed
                    # release or its original expiry; never extrapolate.
                    local = moves.offsets(entry['move'].move, self.now-entry['start'],
                        state={'s_m': self.execution_positions[vid]})
                    offsets[vid] = local[vid]
        context['move_offsets'] = {v: offsets.get(v, 0.) for v in context['members']}
        sample, memory = self.layer.sample(context, generator, deterministic, memory)
        if getattr(self.layer.base, 'intent_horizon', 0) and context['members']:
            published = self.trigger.publish_trigger_intent(context)
            self.trigger_board = {v: (int(self.now), values) for v, values in published.items()}
        if decision is not None:
            sample['track2_trigger'] = dict(_batch_dict(decision), accepted=accepted)
        sample['track2'] = dict(schema=SCHEMA, arm=self.arm, time_s=self.now,
            active_moves=[dict(move=e['move'].to_dict(), start=e['start'], until=e['until'])
                          for e in self.executor.active], merge_features=context['merge_features'],
            reward_neighbourhoods={v: self.neighbourhoods[v] for v in context['members']})
        return sample, memory

    def worker_state(self):
        return dict(schema=SCHEMA, layer=self.layer.state_dict(), trigger=self.trigger.state_dict())

    def restore_worker(self, state):
        if state['schema'] != SCHEMA:
            raise ValueError('unsupported track2 state schema')
        self.layer.load_state_dict(state['layer'], strict=True)
        self.trigger.load_state_dict(state['trigger'], strict=True)


def episode_components(job, base, controller_type):
    """Spawn-safe accel_plant hook. Absent flag returns the exact input objects."""
    if not job.get('track2_backend'):
        return base, controller_type
    cfg, arm = job['track2_config'], job['arm']
    _check(cfg, arm)
    if arm == 'A':
        return base, controller_type
    policy = EpisodePolicy(base, cfg, arm)
    policy.restore_worker(job['track2_state'])
    policy.layer_rng.manual_seed(int(job['noise_seed']))
    policy.candidate_rng = np.random.default_rng(int(job['noise_seed']))

    def controller(*args, **kwargs):
        # Practice uses a controller factory rather than a class. Wrap the
        # instance's apply so prefix replay/CRN restore stays exactly inherited.
        real = controller_type(*args, **kwargs)
        original = real.apply

        def apply(this, rows, commands, now):
            policy.observe(rows, now)
            return original(rows, commands, now)

        real.apply = types.MethodType(apply, real)
        return real

    return policy.eval(), controller


def record_outcomes(policy, event, post_rows):
    """Record P2's actual t+1 observations, never pre-action reward fallbacks.

    Own sensing is clipped before any outcome is read. Selected V2V packets
    remain t-1 relative to the outcome timestamp. Recording is offline telemetry;
    it does not advance the policy's sensing/intent boards or its random streams.
    """
    from .track2_reward import _sensor_packet, neighbourhood_reward
    now = event['time_s'] + CONTROL_DT_S
    packets, neighbourhoods = {}, {}
    present = {r['vid'] for r in post_rows}
    for vid in event['sample']['context']['members']:
        if vid not in present:
            # The public SUMO observation transition ended this agent lifetime.
            # This is a measured terminal occupancy state, not a fabricated
            # speed or a pre-action-frame reward fallback.
            packets[vid] = dict(terminal_exit=dict(cav_id=vid, time_s=now,
                before_time_s=event['time_s'], present_before=True, present_after=False,
                source='sumo_observation_transition'))
            neighbourhoods[vid] = []
            continue
        own = _sensor_packet(post_rows, vid, now)
        partners = [p for p in policy.visible.get(vid, ()) if p != vid]
        packet = dict(own=own, partner_reports={p: policy.reports[p] for p in partners
            if p in policy.reports}, partner_ids=partners)
        outcome = neighbourhood_reward(**packet, cav_id=vid, now_s=now,
                                       weight=policy.cfg['track2_reward']['weight'])
        # Dataclasses are encoded by accel_storage's serializer only if flattened;
        # keep explicit primitive packets for immutable JSON episode receipts.
        from dataclasses import asdict
        packets[vid] = dict(own=asdict(own), partner_reports={p: asdict(r)
            for p, r in packet['partner_reports'].items()}, partner_ids=partners)
        neighbourhoods[vid] = outcome['neighbour_ids']
    event['reward_observations'] = packets
    telemetry = event['sample'].setdefault('track2', dict(schema=SCHEMA,
        arm=policy.arm, time_s=event['time_s'], reward_neighbourhoods=neighbourhoods))
    telemetry['outcome_neighbourhoods'] = neighbourhoods


@contextmanager
def _bank_view(cfg):
    from . import bank_statistics, replay_bank, effective_imitation
    from functools import partial
    original = replay_bank.confirmations
    attach = effective_imitation.attach_limits
    try:
        bank_statistics.install(replay_bank, bank_rule='per_seed_v1', sources=cfg.get('replay_sources'))
        # The shared move contract registers this cap. Pass it explicitly to
        # the existing effective-target reader so no population truth lookup
        # is performed by this backend.
        effective_imitation.attach_limits = partial(attach, cap=moves.CAV_ACCEL_CAP_MPS2)
        yield replay_bank
    finally:
        replay_bank.confirmations = original
        effective_imitation.attach_limits = attach


def _load_base(cfg, arm, update):
    from .v2v_runtime import load_checkpoint
    path = _path(Path(cfg['run_root'])/f'{arm}_checkpoint_{update:03}.pt')
    if not path.exists() and update == 0:
        path = _path(cfg['initial_checkpoint'])
    return load_checkpoint(path, config=cfg)


def _reserve(cfg, jobs, phase):
    """The clean run's OWN ledger, with the parent's <=5M phase allocations."""
    import fcntl
    from tools.coopnative import storage
    budget = cfg['budget']
    total = budget['total']
    if type(total) is not int or total > 5000000 or total <= 0:
        raise ValueError('registered <=5M budget required')
    prior = cfg.get('prior_charged_sim_s', 960)
    if type(prior) is not int or prior < 960:
        raise ValueError('carry the already charged parity960')
    if any(type(v) is not int or v < 0 for v in budget['phases'].values()):
        raise ValueError('invalid phase allocation')
    if sum(budget['phases'].values()) > total:
        raise ValueError('phase allocations exceed total')
    root = _path(cfg.get('accounting_root', cfg['run_root']))
    artifacts = _path(cfg['run_root']).resolve()
    if artifacts != root.resolve() and artifacts.parent != root.resolve():
        raise ValueError('accounting parent must own the isolated smoke artifacts')
    reserve = types.FunctionType(storage.reserve.__code__, dict(storage.reserve.__globals__,
        CAP=total-prior, CAPS=budget['phases']))
    with (root/'ACCOUNTING.claim').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return reserve(root, jobs, phase)


def _wave(cfg, policies, worlds, ident, extras, *, used_seeds=None):
    from . import accel_runner
    from .v2v_runner import patched
    from .storage import ledger
    accounting_root = cfg.get('accounting_root', cfg['run_root'])
    with patched(accel_runner, dict(reserve=lambda r, jobs, phase: _reserve(cfg, jobs, phase),
            ledger=lambda r: ledger(accounting_root))):
        return accel_runner.wave(_path(cfg['run_root']), cfg, policies, worlds, 'train', ident,
            cfg.get('phase_map', {}).get('train', 'track2_train'), False, extras, used_seeds=used_seeds)


def _approved(cfg):
    root = _path(cfg['run_root'])
    path = _path(cfg.get('owner_approval_path', root/'OWNER_APPROVAL.json'))
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('approved') is not True:
        raise ValueError('OWNER_APPROVAL.json must explicitly approve launch')
    # Parent supplies the SHA binding after its full preflight; never accept an
    # approval copied from another configuration when a digest is registered.
    if cfg.get('owner_approval_sha256'):
        from .protocol import sha
        if sha(path) != cfg['owner_approval_sha256']:
            raise ValueError('OWNER_APPROVAL.json SHA mismatch')


class ArmRuntime:
    def __init__(self, cfg, arm, update):
        _check(cfg, arm)
        self.cfg, self.arm, self.update = deepcopy(cfg), arm, update
        self.base, self.optimizer, self.generator, payload = _load_base(cfg, arm, update-1)
        if payload.get('update') != update-1:
            raise ValueError('base checkpoint update mismatch')
        self.policy = self.base if arm == 'A' else EpisodePolicy(self.base, cfg, arm)
        if arm != 'A' and (update > 1 or cfg.get('initial_sidecars_required')):
            path = _path(Path(cfg['run_root'])/f'{arm}_track2_{update-1:03}.pt')
            state = torch.load(path, map_location='cpu', weights_only=False)
            if state.get('arm') != arm or state.get('update') != update-1:
                raise ValueError('cooperation checkpoint arm/update mismatch')
            self.policy.restore_worker(state['worker'])
            self.policy.layer.optimizer.load_state_dict(state['layer_optimizer'])
            self.policy.trigger_optimizer.load_state_dict(state['trigger_optimizer'])
        if arm != 'A':
            self.base = self.policy.layer.base  # runner serializes the immutable low-level model

    def bank_pairs(self):
        if self.arm == 'A':
            return [], dict(status='A_no_bank', pairs=0)
        with _bank_view(self.cfg) as rb:
            return rb.bank_pairs(_path(self.cfg['run_root']), self.arm, self.cfg['training_seed'],
                sources=self.cfg.get('replay_sources', {}).get(self.arm, ()), v2v=True,
                v2v_intent=self.cfg if self.cfg.get('v2v_intent') else None,
                imitation_bank_filter='cooperative_only', imitation_target_space='effective')

    def collect(self):
        cfg, arm, u = self.cfg, self.arm, self.update
        _approved(cfg)
        worlds = cfg['worlds_by_update'][str(u)]
        if not worlds:
            raise ValueError('registered TRAIN worlds required')
        from .accel_protocol import guard, install_seed_scope
        install_seed_scope()  # same registered-world setup used by native episode workers
        for world in worlds:
            guard(world, 'train')
        extras = dict(update=u, v2v=True, v2v_dim=cfg['v2v_dim'], b_replay=arm != 'A', health_guards=True)
        if arm != 'A':
            extras.update(track2_backend=True, track2_config=cfg, track2_state=self.policy.worker_state())
        ident = cfg.get('job_namespace', 'clean1')+f'_warm_u{u:03}'
        if arm != 'A':
            from .warm_health import safety_reference
            from .accel_storage import read
            references = cfg.get('safety_reference')
            if references is None:
                references = safety_reference([read(_path(Path(cfg['run_root'])/'jobs'/
                    f'{ident}_A_{w["seed"]}'/'RESULT.json')) for w in worlds])
            extras['safety_reference'] = references
        policies = {arm: self.base}
        from . import practice_starts as ps
        from .practice_weighting import effective_fraction
        with _bank_view(cfg):
            if ps.enabled(cfg, arm):
                if cfg.get('practice_weighting'):
                    from . import practice_weighting
                    if not getattr(practice_weighting, 'VERIFIED_CHILDREN', False):
                        raise ValueError('SUMO-verified augmentation dependency required before collection')
                fraction = effective_fraction(cfg)
                want = (ps.realised_count(cfg['run_root'], 'B', u) if arm == 'R'
                        else ps.practice_count(len(worlds), fraction))
                plan = ps.plan_for_update(cfg['run_root'], arm, u, fraction=fraction,
                    repeat_cap=cfg['practice_repeat_cap'], count=want, training_seed=cfg['training_seed'],
                    **(dict(weighting=True) if cfg.get('practice_weighting') else {}))
                rows = plan['rows']
                for row in rows:
                    if row.get('source') == 'augmented' or row.get('parent_example_id'):
                        if (not cfg.get('augmentation_prelaunch')
                                or row.get('prelaunch') != cfg['augmentation_prelaunch']):
                            raise ValueError('augmented practice needs parent registered prelaunch provenance')
                        ps.load_snapshot(cfg['run_root'], arm, row)
                episodes = (_wave(cfg, policies, worlds[:len(worlds)-len(rows)], ident, extras)
                            if len(rows) < len(worlds) else [])
                if rows:
                    calls = [ps.wave_call(cfg['run_root'], arm, ident, cfg.get('phase_map', {}).get(
                        'train', 'track2_train'), row, i, extras) for i, row in enumerate(rows)]
                    episodes += _wave(cfg, policies, [c['worlds'][0] for c in calls], ident+'_practice',
                        dict(extras, practice_batch=[dict(practice=c['extras']['practice'],
                                                        branch=c['extras']['branch']) for c in calls]),
                        used_seeds={e['world']['seed'] for e in episodes})
                return [ps.strip_steps(e) if e.get('practice') else e for e in episodes]
        return _wave(cfg, policies, worlds, ident, extras)

    def safety(self, episodes):
        """Coop3 composite/hard-brake guard, with exact practice A references.

        The parent calls this after collection and BEFORE train_update. A must
        have completed collection for the same update first. Matched practice
        references use A's PRE-update policy and one charged wave; they never
        replace normal same-world references when seeds collide.
        """
        if self.arm == 'A':
            return dict(status='A_reference', new_reference_sim_s=0)
        from . import practice_starts as ps, warm_health
        from .accel_storage import read
        from .practice_weighting import effective_fraction
        cfg, arm, u = self.cfg, self.arm, self.update
        root = _path(cfg['run_root'])
        ident = cfg.get('job_namespace', 'clean1')+f'_warm_u{u:03}'
        normal = [read(_path(root/'jobs'/f'{ident}_A_{e["world"]["seed"]}'/'RESULT.json'))
                  for e in episodes if not e.get('practice')]
        practice = [e for e in episodes if e.get('practice')]
        pairs = {}
        if practice:
            with _bank_view(cfg):
                plan = ps.plan_for_update(root, arm, u, fraction=effective_fraction(cfg),
                    repeat_cap=cfg['practice_repeat_cap'], count=len(practice),
                    training_seed=cfg['training_seed'],
                    **(dict(weighting=True) if cfg.get('practice_weighting') else {}))
                calls = []
                for episode in practice:
                    slot = episode['practice']['slot']
                    index = int(slot.rsplit('_p', 1)[1])-1
                    row = plan['rows'][index]
                    if row['example_id'] != episode['practice']['example_id']:
                        raise ValueError('practice receipt does not match frozen plan')
                    calls.append(ps.a_reference_call(root, arm, ident+f'_{arm}_practiceref', row,
                        index, dict(update=u, v2v=True, v2v_dim=cfg['v2v_dim'], b_replay=True,
                                    health_guards=True)))
                base, _, _, _ = _load_base(cfg, 'A', u-1)
                references = _wave(cfg, {'A': base}, [c['worlds'][0] for c in calls],
                    ident+f'_{arm}_practiceref', dict(update=u, v2v=True, v2v_dim=cfg['v2v_dim'],
                        b_replay=True, health_guards=True, practice_batch=[
                            dict(practice=c['extras']['practice'], branch=c['extras']['branch']) for c in calls]))
            by_id = {r['id']: r for r in references}
            for episode, call in zip(practice, calls):
                reference_id = call['ident']+'_A_'+str(call['worlds'][0]['seed'])
                if reference_id not in by_id:
                    raise ValueError('missing matched practice A reference')
                pairs[episode['id']] = by_id[reference_id]
        if len(normal)+len(pairs) != len(episodes):
            raise ValueError('missing paired safety reference')
        warm_health.braking(root, arm, u, episodes, normal, pairs)
        self.safety_checked = True
        return dict(status='checked', normal_references=len(normal), practice_references=len(pairs),
                    new_reference_sim_s=1200*len(pairs))

    def train(self, episodes, *, rewards=None, credit=(), labels=()):
        """Consume P2 reward_update(cfg, arm, update, episodes) output.

        The integration envelope is {schema=REWARD_SCHEMA, arm, update, rows}.
        Rows carry episode_id,time_s,cav_id,source='sumo',own_reward,
        neighbour_rewards (ID -> real reward), reward, weight, weight_provenance.
        Neighbour IDs must match the recorded authorized observation exactly.
        Outcomes cover EVERY control tick, including non-optimizable actions;
        only own stochastic actions get PPO ratios. Returns use the entire
        received complete episode, never only decision/onset/hold ticks.
        Callers may pass the envelope explicitly; otherwise import P2 lazily.
        P2 is an external integration dependency, never a production stub.
        """
        cfg, arm = self.cfg, self.arm
        if credit or labels:
            raise ValueError('clean run requires dense labels and WM credit OFF')
        if any(e['arm'] != arm for e in episodes):
            raise ValueError('episode arm mismatch')
        if arm == 'A':
            from .accel_learner import update
            return dict(ppo=update(self.base, self.optimizer, episodes, self.generator),
                        bank=dict(status='A_no_bank'))
        if not getattr(self, 'safety_checked', False):
            raise ValueError('paired safety hook must complete before B/R optimization')
        by_key = {}
        episode_ids = set()
        for episode in episodes:
            if episode['id'] in episode_ids:
                raise ValueError('duplicate episode identity for temporal credit')
            episode_ids.add(episode['id'])
            for step in episode['steps']:
                key = (episode['id'], step['time_s'])
                if key in by_key:
                    raise ValueError('duplicate episode/time sample')
                by_key[key] = step['sample']
        eligible = {k for k, s in by_key.items() if 'coop_layer' in s and s['context']['members']
                    and not s.get('deterministic')
                    and not s.get('behaviour', {}).get('executed')}
        if not eligible:
            raise ValueError('B/R layer update requires stochastic own-action samples')
        if rewards is None:
            try:
                from .track2_reward import reward_update
            except ImportError as error:
                raise RuntimeError('P2 track2_reward.reward_update integration dependency unavailable') from error
            rewards = reward_update(cfg, arm, self.update, episodes)
        targets = self._reward_targets(rewards, by_key, set(by_key))
        returns, advantages = self._temporal_targets(episodes, targets)
        records = [dict(sample=by_key[k], returns=returns[k]) for k in sorted(eligible)]
        layer = self.policy.layer.update_rewards(records, epochs=cfg['layer_epochs'])
        reference = []
        for key in sorted(eligible):
            if 'track2_trigger' in by_key[key]:
                batch = _batch(by_key[key]['track2_trigger'])
                with torch.no_grad():
                    logits = self.policy.trigger.distribution(batch.context, batch.descriptors, batch.mask)[0]
                reference.append((batch, logits.detach()))
        trigger_metrics = []
        for key in sorted(eligible):
            sample = by_key[key]
            if 'track2_trigger' not in sample:
                continue
            batch = _batch(sample['track2_trigger'])
            before = deepcopy(self.policy.trigger.state_dict())
            optimizer = deepcopy(self.policy.trigger_optimizer.state_dict())
            metrics = ppo_update(self.policy.trigger, self.policy.trigger_optimizer, batch,
                advantages[key], returns[key], **cfg['trigger_ppo'])
            kl = self._trigger_kl(reference)
            accepted = math.isfinite(kl) and kl <= self.policy.layer.config.kl_budget
            if not accepted:
                self.policy.trigger.load_state_dict(before)
                self.policy.trigger_optimizer.load_state_dict(optimizer)
            trigger_metrics.append(dict(metrics, exact_kl=kl, accepted=accepted))
        pairs, bank_summary = self.bank_pairs()
        from . import bank_imitation as bi
        if cfg.get('v2v_intent'):
            from .v2v_intent import refresh_bank_intents
            refresh_bank_intents(self.base, pairs)
        # Frozen base: existing full-pass/effective likelihood updates ONLY the
        # cooperation residual, with the existing independent .045 KL cap.
        bank = bi.run_separate(self.policy, self.policy.layer.optimizer, pairs, self.generator,
            deepcopy(self.policy).eval(), passes=cfg.get('imitation_epochs', 5),
            step_mode=cfg.get('imitation_step_mode', 'kl_targeted'), intent_bank_term=False,
            imitation_target_space='effective')
        self.policy.layer._check_frozen()
        self.policy.trigger_optimizer.zero_grad(set_to_none=True)
        return dict(layer=layer, trigger=trigger_metrics, dense=dict(enabled=False),
            wm_credit=dict(enabled=False), reward=dict(schema=REWARD_SCHEMA,
                registration=deepcopy(cfg['track2_reward']), cav_rows=sum(map(len, targets.values())),
                temporal=dict(method='full_rollout_discounted_return', gamma=GAMMA,
                    reward_unit=UNIT, registration='tools.coopmwm.accel_learner.GAMMA,UNIT',
                    trigger_advantage='return minus recorded per-CAV trigger value',
                    terminal='zero at episode end, terminal tick or CAV departure')),
            bank=bank, bank_summary=bank_summary)

    def _reward_targets(self, envelope, by_key, eligible):
        if (envelope.get('schema') != REWARD_SCHEMA or envelope.get('arm') != self.arm
                or envelope.get('update') != self.update):
            raise ValueError('P2 reward schema/arm/update mismatch')
        expected = {(key[0], key[1], vid) for key in eligible
                    for vid in by_key[key]['context']['members']}
        values = {}
        registration = self.cfg['track2_reward']
        for row in envelope['rows']:
            key = (row['episode_id'], row['time_s'], row['cav_id'])
            if key not in expected or key in values:
                raise ValueError('duplicate/unmatched per-CAV reward')
            neighbours = row['neighbour_rewards']
            observed = by_key[key[:2]]['track2']['reward_neighbourhoods'][key[2]]
            # Production workers bind the exact post-action sensor domain.
            # Legacy/fixture envelopes retain their pre-recorded strict domain.
            outcome_ids = by_key[key[:2]]['track2'].get('outcome_neighbourhoods')
            if outcome_ids is not None:
                observed = outcome_ids[key[2]]
            if (row.get('source') != 'sumo' or row.get('weight') != registration['weight']
                    or row.get('weight_provenance') != registration['provenance']
                    or set(neighbours) != set(observed)):
                raise ValueError('reward source/registration/observable neighbourhood mismatch')
            numbers = [row['own_reward'], row['reward'], *neighbours.values()]
            if not all(type(v) in (int, float) and math.isfinite(v) for v in numbers):
                raise ValueError('nonfinite per-CAV SUMO reward')
            target = row['own_reward'] + registration['weight'] * (
                sum(neighbours.values())/len(neighbours) if neighbours else 0.)
            if not math.isclose(row['reward'], target, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('observable SUMO reward formula mismatch')
            values[key] = float(row['reward'])
        if set(values) != expected:
            raise ValueError('missing aligned per-CAV SUMO rewards')
        for key in eligible:
            sample = by_key[key]
            trigger = sample.get('track2_trigger')
            if trigger is not None and trigger['context']['members'] != sample['context']['members']:
                raise ValueError('trigger/reward per-CAV member alignment mismatch')
        return {key: [values[(*key, vid)] for vid in by_key[key]['context']['members']]
                for key in sorted(eligible)}

    def _temporal_targets(self, episodes, rewards):
        """Full finite-episode Monte Carlo returns, in inherited MAPPO units.

        GAMMA/UNIT are the existing acceleration learner registration, applied
        once per physical control second. This is return-to-go (not GAE), so
        there is no lambda decay or new credit horizon. The residual has no
        per-CAV critic and uses the return with a zero baseline. Trigger PPO
        uses its own *recorded* value, never the global frozen MAPPO critic.

        accel_plant receipts end at the registered finite episode horizon and
        therefore bootstrap to zero, just like accel_learner. Explicit terminal
        ticks also stop credit. Absence from the next tick's member IDs is a
        departure; later reuse of the ID starts a new lifetime. A partial tape
        cannot silently become a shorter learning horizon: truncated receipts,
        missing ticks and a suffix inconsistent with actual_sim_s refuse.
        """
        returns, advantages = {}, {}
        for episode in episodes:
            steps = sorted(episode['steps'], key=lambda step: step['time_s'])
            if episode.get('truncated') or any(step.get('truncated') for step in steps):
                raise ValueError('truncated rollout requires terminal outcomes; no shortened credit horizon')
            if not steps:
                continue
            times = [step['time_s'] for step in steps]
            if (any(type(t) is not int for t in times)
                    or any(b-a != 1 for a, b in zip(times, times[1:]))):
                raise ValueError('temporal credit requires every consecutive control second')
            if ('actual_sim_s' in episode and times[-1]+1 != episode['actual_sim_s']):
                raise ValueError('incomplete rollout suffix for temporal credit')
            future = {}
            for step in reversed(steps):
                if step.get('terminated') or step.get('done'):
                    future = {}
                key = (episode['id'], step['time_s'])
                sample = step['sample']
                members = sample['context']['members']
                if len(set(members)) != len(members):
                    raise ValueError('duplicate per-CAV temporal member identity')
                current = {vid: reward/UNIT + GAMMA*future.get(vid, 0.)
                           for vid, reward in zip(members, rewards[key])}
                returns[key] = [current[vid] for vid in members]
                if not all(math.isfinite(value) for value in returns[key]):
                    raise ValueError('nonfinite per-CAV temporal return')
                if 'track2_trigger' in sample:
                    values = sample['track2_trigger']['values']
                    if (len(values) != len(members) or any(type(v) not in (int, float)
                            or not math.isfinite(v) for v in values)):
                        raise ValueError('invalid recorded per-CAV trigger values')
                    advantages[key] = [current[vid]-value for vid, value in zip(members, values)]
                    if not all(math.isfinite(value) for value in advantages[key]):
                        raise ValueError('nonfinite per-CAV temporal advantage')
                future = current  # drop departed IDs; never carry across episodes
        return returns, advantages

    @torch.no_grad()
    def _trigger_kl(self, reference):
        values = []
        for batch, logits in reference:
            after = self.policy.trigger.distribution(batch.context, batch.descriptors, batch.mask)[0]
            values.append(torch.distributions.kl_divergence(torch.distributions.Categorical(logits=logits),
                torch.distributions.Categorical(logits=after)).sum())
        return float(torch.stack(values).mean()) if values else 0.

    def save(self):
        """Own immutable base checkpoint plus layer/optimizer sidecar per update."""
        from .accel_learner import save
        root = _path(self.cfg['run_root'])
        target = root/f'{self.arm}_checkpoint_{self.update:03}.pt'
        sidecar = root/f'{self.arm}_track2_{self.update:03}.pt'
        if target.exists() or (self.arm != 'A' and sidecar.exists()):
            raise ValueError('checkpoint already exists')
        from .storage import ledger
        charged = int(self.cfg.get('prior_charged_sim_s', 960))+sum(r['sim_s'] for r in
            ledger(self.cfg.get('accounting_root', root)))
        save(target, self.base, self.optimizer, self.generator, self.update, charged)
        if self.arm != 'A':
            state = dict(arm=self.arm, update=self.update, worker=self.policy.worker_state(),
                         layer_optimizer=self.policy.layer.optimizer.state_dict(),
                         trigger_optimizer=self.policy.trigger_optimizer.state_dict())
            with sidecar.open('xb') as stream:
                torch.save(state, stream)
        return str(target)


def build_arm_runtime(cfg, arm):
    """Build one update's hooks without dispatch/reservation; caller owns gates.

    Set cfg['update'] before build (default 1). Keep this hook mapping until P2
    returns observable SUMO rewards; call safety, train_update and save_checkpoint. For cfg-only entry
    points collect_update reconstructs the same pre-update checkpoint weights.
    """
    runtime = ArmRuntime(cfg, arm, int(cfg.get('update', 1)))
    return dict(schema=SCHEMA, production=True, runtime=runtime, policy=runtime.policy,
        policy_factory=lambda: runtime.policy, bank_pairs=runtime.bank_pairs,
        collect_update=runtime.collect, train_update=runtime.train, save_checkpoint=runtime.save,
        safety=runtime.safety, episode_components=episode_components)


def collect_update(cfg, arm, u):
    return build_arm_runtime(dict(cfg, update=u), arm)['collect_update']()
