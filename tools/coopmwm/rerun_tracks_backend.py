"""Production Track-2 adapter onto the reviewed coop3 V2V plant and learner.

No physics runs at import/configuration. Every episode/branch is synchronous from
the coordinator's point of view and stays inside its reservation and worker permit.
Sibling factories are explicit approval-bound entry points: no synthetic fallback.
"""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import importlib
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import socket
import threading

import torch

from . import rerun_cpus as rc
from .exact_grade import _seal

COMPONENTS = ('merge_features', 'trigger_head', 'coop_layer', 'track2_reward',
              'plan_augment', 'practice_weighting')


def entrypoint(name):
    module, symbol = name.split(':', 1)
    if (not module.startswith('tools.coopmwm.') or '.test_' in module
            or not symbol.isidentifier()):
        raise ValueError('REFUSED: production entry point must be a coopmwm sibling adapter')
    value = getattr(importlib.import_module(module), symbol)
    if not callable(value):
        raise ValueError('REFUSED: component entry point is not callable')
    return value


def factories(cfg, base):
    """Siblings supply the documented consumed adapter, not guessed class APIs."""
    bindings = cfg['production']['component_factories']
    names = rc.component_names(cfg)
    if set(bindings) != set(names):
        raise ValueError('REFUSED: enabled production component factories required')
    def bind(name):
        constructor = entrypoint(bindings[name])
        return lambda **kw: constructor(base=base, **kw)
    return {name: bind(name) for name in names}


def base_weights(weights):
    return {k: v for k, v in weights.items() if not k.startswith('t2c.')}


def load_policy(weights):
    from .v2v_runtime import from_weights
    return from_weights(base_weights(weights), enabled=True)


def native_move_plan(move, obs, cfg):
    """Retain the native plan/window ABI, with exact *relative* move commands.

    Static absolute profiles cannot represent offsets on a state-dependent base.
    Verification therefore reuses native replay/CRN and adds these offsets to
    the branch's own base action each second, just like the live controller.
    """
    from . import moves
    plan = deepcopy(entrypoint(cfg['production']['move_to_plan'])(move, observation=obs, cfg=cfg))
    if set(plan['members']) != set(move.members) or type(plan['duration_s']) is not int or plan['duration_s'] <= 0:
        raise ValueError('REFUSED: native move adapter changed members/duration')
    plan['track2_offsets'] = {v: [] for v in move.members}
    for elapsed in range(plan['duration_s']):
        offsets = moves.offsets(move, elapsed)
        if not set(offsets).issubset(move.members):
            raise ValueError('REFUSED: native move offset commands foreign member')
        for vid in move.members:
            value = float(offsets.get(vid, 0.))
            if not math.isfinite(value):
                raise ValueError('REFUSED: nonfinite native move offset')
            plan['track2_offsets'][vid].append(value)
    return plan


def relative_plan_hook(controller, job, prefixes):
    """Native all8 replay keeps its sole safety writer and dynamic base actions."""
    original = controller.apply
    def apply(rows, commands, now):
        desired = original(rows, commands, now)
        event = controller.events[-1]
        sample = event['sample']
        members = sample['context']['members']
        for commitment in controller.active_plans:
            plan = commitment['plan']
            offsets = plan.get('track2_offsets')
            elapsed = int(now)-int(commitment['start_s'])
            if offsets is None or not 0 <= elapsed < plan['duration_s']:
                continue
            for vid in plan['members']:
                if vid not in commands or vid not in members:
                    continue
                residual = float(sample['own_action'][members.index(vid)][0])
                raw = commands[vid]+residual+offsets[vid][elapsed]
                raw = min(1.2, float(job['accel_bounds'][1]), max(float(job['accel_bounds'][0]), raw))
                desired[vid] = dict(requested_raw=raw, acceleration_residual=raw-commands[vid])
                sample.setdefault('absolute_accels', {})[vid] = raw
        event['requests'] = deepcopy(desired)
        return desired
    controller.apply = apply


def component_weights(components):
    return {f't2c.{prefix}.{k}': v.detach().cpu().clone()
            for prefix, name in (('layer', 'coop_layer'), ('trigger', 'trigger_head'))
            for k, v in components[name].state_dict().items()}


def restore_components(cfg, arm, model, weights):
    components = rc.track2_components(cfg, arm, adapters=factories(cfg, model))
    for prefix, name in (('layer', 'coop_layer'), ('trigger', 'trigger_head')):
        state = {k[len(prefix)+5:]: v for k, v in weights.items()
                 if k.startswith('t2c.'+prefix+'.')}
        if components:
            components[name].load_state_dict(state, strict=True)
    return components


def observation(rows, previous, now, bounds, *, ego=None):
    """Local sensing only; project an observable trace to one ego's radio range.

    Never forward the plant's all-road rows, latent types, or current commands to
    a component. The native MAPPO/V2V controller retains its reviewed encoder.
    """
    from .v2v_obs import V2V_RANGE_M
    fields = ('vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m')
    cavs = sorted((r for r in rows if r.get('is_cav')), key=lambda r: r['vid'])
    me = next((r for r in cavs if r['vid'] == ego), None) if ego is not None else (cavs[0] if cavs else None)
    visible = ([r for r in rows if abs(float(r['s_m'])-float(me['s_m'])) <= V2V_RANGE_M]
               if me is not None else [])
    def project(row):
        return {k: deepcopy(row[k]) for k in fields if k in row}
    visible = sorted(visible, key=lambda r: r['vid'])
    members = [r['vid'] for r in visible if r.get('is_cav')]
    reports = []
    if previous and previous['time_s'] == now-1:
        for row in sorted(previous['vehicles'], key=lambda r: r['vid']):
            if row.get('is_cav') and row['vid'] in members and row['vid'] != me['vid']:
                reports.append(dict(project(row), time_s=now-1,
                    last_command=row.get('issued_acceleration_mps2')))
    return dict(time_s=int(now), own_sensing=dict(ego_id=me['vid'] if me else ego,
                vehicles=[project(r) for r in visible]), v2v_reports=reports,
                cav_ids=members, bounds={v: [float(bounds[0]), min(1.2, float(bounds[1]))] for v in members})


def safety(result, trace, *, start=300, end=1200):
    from .replay_bank import ttc_events
    totals = result['totals']
    return dict(collisions=int(totals['collisions']), teleports=int(totals['teleports']),
                hard_brakes=int(totals['hard_brake']), ttc_events=int(ttc_events(trace, start, end)))


class EpisodeHook:
    """Add offsets after native V2V sampling, before the existing request/safety writer."""
    def __init__(self, cfg, arm, model, components, *, selected=None, takeover=299):
        self.cfg, self.arm, self.model, self.components = cfg, arm, model, components
        self.selected, self.takeover = selected, takeover
        self.rows, self.missing = [], []
        self.rng = random.Random(20261104)
        self.label_rng = random.Random(20261105)
        self.actors, self.pending, self.reward_rows = {}, {}, {}

    def __call__(self, controller, job, prefixes):
        if not self.components:
            return
        from . import moves, sumo_verify as sv
        from .rerun_tracks_runtime import dense_label
        original = controller.apply
        def apply(rows, commands, now):
            desired = original(rows, commands, now)
            if now <= self.takeover or (job.get('practice') and now <= job['practice']['snapshot']['t0']):
                return desired
            cavs = {r['vid'] for r in rows if r.get('is_cav') and r['vid'] in commands}
            acting = cavs if self.selected is None or isinstance(self.selected, AllMembers) else cavs & set(self.selected)
            self.actors = {v: state for v, state in self.actors.items() if v in acting}
            self.pending = {}
            intents = {}
            for ego in sorted(acting):
                obs = observation(rows, controller.previous_frame, int(now), job['accel_bounds'], ego=ego)
                features = self.components['merge_features'].encode(obs)
                self.missing.append(float(features['missing_share']))
                if ego not in self.actors:
                    salt = int(hashlib.sha256((str(job.get('noise_seed', 20261104))+':'+ego).encode()).hexdigest()[:16], 16)
                    self.actors[ego] = dict(move=None, onset=None, rng=random.Random(salt))
                state = self.actors[ego]
                if state['move'] is None or int(now) % self.cfg['trigger_head']['decision_interval_s'] == 0:
                    proposals = moves.candidates(obs, state['rng'], self.cfg['dense_teacher']['candidates'])
                    if not proposals or not any(m.family == 'none' for m in proposals):
                        raise ValueError('REFUSED: production move slate requires none')
                    active = self.components['trigger_head'].choose(obs, features, proposals, state['rng'])
                    if active not in proposals or not set(active.members).issubset(obs['cav_ids']):
                        raise ValueError('REFUSED: production trigger selected nonobservable CAV')
                    state.update(move=active, onset=int(now))
                    if self.selected is None:
                        self.reward_rows[ego] = len(self.rows)
                        self.rows.append(dict(observation=obs, features=features, move=active,
                            ego_id=ego, row_id=job['id']+f'_t{int(now):04}_{ego}', family=active.family,
                            realised_team_reward=0., reward_steps=0))
                offsets = moves.offsets(state['move'], int(now)-state['onset'])
                residual = self.components['coop_layer'].residual(obs, features)
                if not (set(offsets) | set(residual)).issubset(obs['cav_ids']):
                    raise ValueError('REFUSED: HDV/nonobservable production command')
                for vid in (set(offsets) | set(residual)) & acting & set(obs['cav_ids']):
                    # Deterministic single ownership: a CAV's own policy context
                    # wins its actuator; remaining ties use the stable actor ID.
                    priority = (vid != ego, ego)
                    if vid not in intents or priority < intents[vid][0]:
                        intents[vid] = (priority, offsets.get(vid, 0.)+residual.get(vid, 0.), obs['bounds'][vid])
                self.pending[ego] = obs
            sample = controller.events[-1]['sample']
            for vid, (_, correction, bounds) in sorted(intents.items()):
                raw = desired.get(vid, {}).get('requested_raw', commands[vid])
                lo, hi = bounds
                raw = min(hi, max(lo, raw+correction))
                desired[vid] = dict(requested_raw=raw, acceleration_residual=raw-commands[vid])
                sample.setdefault('absolute_accels', {})[vid] = raw
            controller.events[-1]['requests'] = deepcopy(desired)
            return desired
        controller.apply = apply

    def after_step(self, now, rows):
        if not self.components or self.selected is not None:
            return
        from .rerun_tracks_reward import team_reward
        for ego, obs in self.pending.items():
            after = observation(rows, None, int(now)+1, (-3., 1.2), ego=ego)
            receipt = team_reward(self.components['track2_reward'], obs, after, self.cfg)
            row = self.rows[self.reward_rows[ego]]
            row['realised_team_reward'] += receipt['team_reward']
            row['reward_steps'] += 1
            row['reward_receipt'] = receipt

    def _wm_row(self, controller, job, prefixes, now, obs, features, active, proposals, sv, dense_label):
                    sample = controller.events[-1]['sample']
                    snapshot = sv.capture(controller, prefixes, job, now, sample, controller.generator)
                    torch.rand((), generator=controller.generator)  # native verification pad draw
                    label = dense_label(self.components['dense_teacher'].label(obs, proposals,
                        kind=rc.teacher_kind(self.cfg, self.arm), rng=self.label_rng), proposals, self.cfg)
                    advantage = float(self.components['group_credit'].estimate(obs, active, self.model))
                    if not math.isfinite(advantage):
                        raise ValueError('REFUSED: nonfinite production group credit')
                    # The sibling move adapter supplies the native full-window
                    # plan representation; parameters/horizons are never guessed.
                    plans = [native_move_plan(m, obs, self.cfg) for m in proposals if m.family != 'none']
                    if not plans:
                        raise ValueError('REFUSED: no native move plans for verification')
                    window = sv.build_branches(snapshot, plans)[0]['window']
                    snapshot_path = Path(job['path'])/f't2c_snapshot_{int(now):04}.pt'
                    torch.save(snapshot, snapshot_path)
                    self.rows.append(dict(observation=obs, features=features, move=active, **label,
                        proposals=proposals, native_plans=plans, snapshot_path=str(snapshot_path),
                        snapshot_file_sha256=rc.sha(snapshot_path), group_advantage=advantage,
                        row_id=job['id']+f'_t{int(now):04}', family=active.family,
                        window=window, episode=job['id']))


def episode_worker(job, cfg, weights):
    """A spawned CPU process owns exactly one complete native SUMO episode."""
    from . import accel_plant
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    model = load_policy(job['weights'])
    if job['arm'] != 'A':
        model.requires_grad_(False)
    components = restore_components(cfg, job['arm'], model, weights)
    hook = EpisodeHook(cfg, job['arm'], model, components)
    hook.rng.seed(20261104+10000*job['update']+job['episode_index'])
    hook.label_rng.seed(20261105+10000*job['update']+job['episode_index'])
    result = accel_plant.episode(job, controller_hook=hook)
    from .practice_starts import strip_steps
    training = strip_steps(result) if result.get('practice') else result
    counters = safety(result, Path(job['path'])/'observable.jsonl.gz')
    return dict(rows=hook.rows, missing=hook.missing,
        reward=sum(s['reward'] for s in training['steps']), safety=counters,
        safety_stop=bool(counters['collisions'] or counters['teleports']), training=training)


class Backend:
    cpu_plant = False

    def __init__(self, cfg, *, prep_dir, stage_dir):
        self.cfg = deepcopy(cfg)
        self.prep_dir, self.stage_dir = Path(prep_dir), Path(stage_dir)
        self.host = socket.gethostname().split('.')[0]
        self.release = deepcopy(cfg['production']['release'])
        self.pool = None
        self.pool_lock = threading.Lock()
        self.models, self.optimizers, self.generators, self.components = {}, {}, {}, {}

    def configure(self, cfg, registration):
        rc.check_pins(cfg)
        self.cfg, self.registration = deepcopy(cfg), deepcopy(registration)
        from . import v2v_live
        # Reuse reviewed per_seed_v1, effective, intent and bank hooks. Own-ledger
        # accounting is owned by Runtime, not the historical shared-ledger hook.
        self.hooks = v2v_live.hooks(Path(cfg['run_root']), cfg)
        for value in cfg['production']['component_factories'].values():
            entrypoint(value)
        entrypoint(cfg['production']['move_to_plan'])
        harness = importlib.import_module(cfg['final_evaluation']['module'])
        if any(not callable(getattr(harness, n, None)) for n in ('prepare_track2', 'analyse_track2')):
            raise ValueError('REFUSED: shared production T1D adapter is incomplete')

    def adapters(self, base):
        native = factories(self.cfg, base)
        def bind(name):
            def build(**kw):
                value = native[name](**kw)
                self.components.setdefault(kw['arm'], {})[name] = value
                return value
            return build
        return {name: bind(name) for name in native}

    def load_base(self, path, *, arm, cfg):
        model, opt, generator, _ = self.hooks['load_checkpoint'](path, config=cfg)
        self.models[arm], self.optimizers[arm], self.generators[arm] = model, opt, generator
        return model

    def _gate(self, phase_costs):
        from . import v2v_launch3
        p = self.cfg['production']
        # Same launch3 immutable source/gate/HOLD/resources checks, immediately
        # before every actual process dispatch. Track2 requotes its own ledger.
        v2v_launch3.before_launch(p['launch_binding_path'], binding_sha256=p['launch_binding_sha256'],
            root=self.cfg['run_root'], control=p['control_root'], mover=p['mover_root'],
            phase_costs=phase_costs)

    def _dispatch(self, function, *args):
        with self.pool_lock:
            if self.pool is None:
                self.pool = ProcessPoolExecutor(max_workers=rc.CPU['max_sumo_workers'],
                    mp_context=multiprocessing.get_context('spawn'))
        return self.pool.submit(function, *args).result()  # permit covers entire lifetime

    def run_episode(self, *, arm, update, index, start, model, components, cfg):
        from . import accel_runner
        from .protocol import batch_worlds
        world = batch_worlds(update)[index]
        extras = dict(update=update, episode_index=index, v2v=True, v2v_dim=48, b_replay=arm != 'A')
        ident = f't2c_u{update:03}_{arm}_episode_{index:03}'
        if start is not None:
            from .practice_starts import wave_call
            call = wave_call(start.get('practice_root', cfg['run_root']), arm, ident, 'v2v_train', start['practice_row'], index, extras)
            world, extras = call['worlds'][0], call['extras']
        raw_id = f'{ident}_{arm}_{world["seed"]}'
        if start is not None:
            raw_id = f'{extras["practice"]["slot"]}_{arm}_{world["seed"]}'
        binding = dict(cfg, job_id_overrides={raw_id: ident}, precharged_unattempted=[ident])
        jobs, reused = accel_runner.wave_jobs(Path(cfg['run_root']), binding, {arm: model},
            [world], 'train', ident, 'v2v_train', False, extras)
        if reused or len(jobs) != 1:
            raise ValueError('REFUSED: production runtime never adopts/replays attempted jobs')
        phase = 'track2_smoke' if cfg.get('runtime_smoke') else 'v2v_train'
        # Runtime already charged this exact job. Launch3 is read-only; requote
        # validates remaining work, and the dispatch guard validates this charge.
        self._gate_reserved(ident, rc.EPISODE_SIM_S, phase)
        return self._dispatch(episode_worker, jobs[0], cfg,
                              component_weights(components) if components else {})

    def _gate_reserved(self, ident, amount, phase):
        from .storage import ledger
        rows = [r for r in ledger(self.cfg['run_root']) if r['id'] == ident]
        if len(rows) != 1 or rows[0]['sim_s'] != amount or rows[0]['phase'] != phase:
            raise ValueError('REFUSED: exact production dispatch reservation missing')
        # Reservation is already included in spent. Check remaining + spent,
        # rather than requiring this wave's cost a second time.
        from . import v2v_launch3
        p = self.cfg['production']
        v2v_launch3.before_launch(p['launch_binding_path'], binding_sha256=p['launch_binding_sha256'],
            root=self.cfg['run_root'], control=p['control_root'], mover=p['mover_root'],
            phase_costs={phase: 0})

    def ppo_update(self, model, batches):
        from .accel_learner import update
        arm = next(a for a, p in self.models.items() if p is model)
        if arm != 'A':
            raise ValueError('REFUSED: only A enters native base PPO')
        return update(model, self.optimizers[arm], batches, self.generators[arm])

    def save_checkpoint(self, path, model, components):
        arm = next(a for a, p in self.models.items() if p is model)
        payload = dict(base=model.state_dict(), optimizer=(components['coop_layer'].optimizer.state_dict()
            if components else self.optimizers[arm].state_dict()), generator=self.generators[arm].get_state())
        if components:
            payload.update(layer=components['coop_layer'].state_dict(), trigger=components['trigger_head'].state_dict())
        with Path(path).open('xb') as stream:
            torch.save(payload, stream)

    def checkpoint_components(self, path):
        data = torch.load(Path(rc.guard_path(path)), map_location='cpu', weights_only=False)
        return [k for k in ('base', 'layer', 'trigger', 'optimizer') if k in data]

    def diagnostic_weights(self, model, components):
        return dict({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    **(component_weights(components) if components else {}))

    def bank_examples(self, arm, update, *, cfg):
        from . import replay_bank as rb, bank_v2v, practice_starts
        from .warm_protocol import SEED
        from .v2v_intent import refresh_bank_intents
        root = Path(cfg['run_root'])
        links = practice_starts._links(root, arm)
        encoder = self.components[arm]['merge_features']
        out = []
        def consume(*, receipt, example, plan, pairs, trace):
            if example is None:
                raise ValueError('REFUSED: production bank requires real per_seed_v1 examples')
            frames = bank_v2v.trace_frames(trace)
            snapshot = rb._decision_snapshot(root, arm, receipt, links)
            seed_index = int(example['seed_index'])
            if seed_index > 1:
                salt = receipt['verification_id'] if seed_index == 2 else f'{receipt["verification_id"]}#s{seed_index}'
                snapshot = rb.replicate(snapshot, salt)
            snapshot_path = root/'components'/'bank_snapshots'/f'{arm}_{receipt["verification_id"]}_s{seed_index}.pt'
            rc.guard_path(snapshot_path)
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            if not snapshot_path.exists():
                torch.save(snapshot, snapshot_path)
            # Native intent refresh uses the frozen current base and the same
            # per-pair entry/time links as reviewed V2V bank imitation.
            refresh_bank_intents(self.models[arm], pairs)
            for i, (sample, weight) in enumerate(pairs):
                t = int(sample['intent_link']['t'])
                group = list(plan['members'])
                obs = observation(frames[t]['vehicles'], frames.get(t-1), t,
                                  snapshot['sumo']['accel_bounds'], ego=group[0])
                if not set(group).issubset(obs['cav_ids']):
                    continue  # impossible observable group never becomes a target
                features = encoder.encode(obs)
                native = deepcopy(sample['context'])
                indices = [native['members'].index(v) for v in group]
                effective = dict(action=[float(sample['action'][j][0]) for j in indices],
                    binding_limits={v: sample['binding_limits'][v] for v in group},
                    behaviour=dict(executed=True, proposal_members=group),
                    context=dict(members=group, observation=obs, features=features),
                    native_sample=deepcopy(sample),
                    plan_sha256=_seal(dict(plan=plan), 'sha256')['sha256'])
                ident = f'{arm}_{example["example_id"]}_t{t:04}'
                link = links.get(receipt['verification_id'])
                practice_row = (dict(verification_id=receipt['verification_id'],
                    example_id=example['example_id'], seed_index=seed_index,
                    salt=receipt['verification_id'], state_id=link['state_id'], fallback=False) if link else None)
                out.append(dict(id=ident, source_id=ident, arm=arm, plan=deepcopy(plan),
                    observation=obs, seed=snapshot['episode']['world']['seed'], weight=float(weight),
                    statistics='per_seed_v1', effective_sample=effective,
                    provenance=dict(verification_sha256=receipt['receipt_sha256'],
                        weight_sha256=_seal(dict(example=example, weight=weight), 'sha256')['sha256'],
                        effective_sha256=_seal(dict(sample=sample), 'sha256')['sha256']),
                    window=dict(receipt['window'], end_s=1200, truncated=False), practice_row=practice_row,
                    snapshot_path=str(snapshot_path), snapshot_file_sha256=rc.sha(snapshot_path)))
        self.hooks['bank_pairs'](root, arm, SEED, v2v=True, example_consumer=consume)
        if not out:
            raise ValueError('REFUSED: production bank has no observable effective per-seed examples')
        return out

    def resolve_states(self, job):
        row = next(r for r in self.registration['replicates'] if r['world_seed'] == job['world_seed'])
        out = []
        for state in row['states']:
            snapshot_path = Path(rc.guard_path(state['snapshot_path']))
            receipt_path = Path(rc.guard_path(state['receipt_path']))
            if rc.sha(snapshot_path) != state['snapshot_file_sha256'] or rc.sha(receipt_path) != state['receipt_file_sha256']:
                raise ValueError('REFUSED: registered diagnostic state bytes changed')
            out.append((torch.load(snapshot_path, map_location='cpu', weights_only=False),
                        json.loads(receipt_path.read_text(encoding='utf-8'))))
        return out

    def expand_verification(self, kind, payload, *, cfg):
        from . import sumo_verify as sv, replay_bank as rb
        path = Path(rc.guard_path(payload['snapshot_path']))
        if rc.sha(path) != payload['snapshot_file_sha256']:
            raise ValueError('REFUSED: verification snapshot bytes changed')
        snapshot = torch.load(path, map_location='cpu', weights_only=False)
        if kind == 'teacher':
            plans = payload['native_plans']
        elif kind == 'group_credit':
            plan = native_move_plan(payload['move'], payload['observation'], cfg)
            plans = [plan]+[rb.solo_candidate(plan, m) for m in plan['members']]
        else:
            plans = [payload['plan']]
        specs = sv.build_branches(snapshot, plans, 'random' if payload['arm'] == 'R' else 'wm')
        out = []
        for spec in specs:
            body = {k: v for k, v in spec.items() if k != 'branch_sha256'}
            body.update(window=deepcopy(payload['window']), sim_s=int(payload['window']['end_s']),
                        verification_id=payload['id'])
            out.append((_seal(body, 'branch_sha256'), snapshot))
        return out

    def grade_verification(self, kind, payload, results):
        if any(r['status'] != 'graded' for r in results):
            raise ValueError('REFUSED: incomplete native verification')
        costs = [r['J_veh_s'] for r in results]
        out = dict(status='graded', advantage=costs[0]-costs[1], branches=len(results))
        if kind == 'group_credit':
            out['advantage'] = min(costs[2:])-costs[1]  # group vs best solo
        if kind == 'teacher':
            proposals = payload['proposals']
            target = payload['target']
            if target is None:
                out['label_correct'] = None
            else:
                expected = 0 if target.family == 'none' else 1+[p for p in proposals if p.family != 'none'].index(target)
                out['label_correct'] = expected == min(range(len(costs)), key=lambda i: (costs[i], i))
        return _seal(out, 'verification_sha256')

    def verify_augmentation(self, payload, *, cfg, execute):
        from .rerun_tracks_augment import verify
        return verify(self, payload, cfg, execute)

    def practice_reference(self, start, *, common_weights):
        from . import replay_bank as rb
        path = Path(rc.guard_path(start['snapshot_path']))
        if rc.sha(path) != start['snapshot_file_sha256']:
            raise ValueError('REFUSED: practice reference snapshot bytes changed')
        snap = torch.load(path, map_location='cpu', weights_only=False)
        snap['weights'] = deepcopy(common_weights)
        from . import sumo_verify as sv
        snap['policy'] = dict(state_dict_sha256=sv._state_sha(common_weights))
        spec = rb.lesson_policy_spec(snap, start['plan'], dict(window=start['window'], proposer='wm'))
        # Full training horizon, same snapshot and CRN, just like native practice.
        body = {k: v for k, v in spec.items() if k != 'branch_sha256'}
        body.update(window=deepcopy(start['window']), sim_s=rc.EPISODE_SIM_S)
        return _seal(body, 'branch_sha256'), snap

    def run_reserved_branch(self, spec, snapshot, *, ident, phase):
        self._gate_reserved(ident, spec['sim_s'], phase)
        return self._dispatch(branch_worker, spec, snapshot, self.cfg,
                              spec.get('t2c_artifact_root', str(Path(self.cfg['run_root'])/'branches'/ident)))

    def expand_scheduled(self, kind, *, arm, update, cfg):
        if kind == 'fixed_test':
            worlds = self.registration['final_evaluation']['worlds']
            checkpoint = Path(cfg['run_root'])/f'{arm}_checkpoint_{update:03}.pt'
            return [(dict(branch='fixed_test', arm=arm, world_id=w['world_id'], world=w['world'],
                checkpoint=dict(path=str(checkpoint), file_sha256=rc.sha(checkpoint)),
                sim_s=1200, window=dict(start_s=0, end_s=1200)), {}) for w in worlds]
        from . import replay_bank as rb, sumo_verify as sv
        frozen = rb.lesson_list(cfg['run_root'], arm, cfg.get('replay_sources', {}).get(arm, ()))
        decisions = {r['verification_id']: r for r in rb.bank_decisions(cfg['run_root'], arm,
                     sources=cfg.get('replay_sources', {}).get(arm, ()))}
        from .practice_starts import _links
        links = _links(cfg['run_root'], arm)
        # Diagnostics can overlap the next update. Always read this immutable
        # update's complete checkpoint, never the now-mutating live component.
        weights = checkpoint_weights(Path(cfg['run_root'])/f'{arm}_checkpoint_{update:03}.pt')
        pairs = []
        for vid in frozen['verification_ids']:
            receipt = decisions[vid]
            snap = rb._decision_snapshot(cfg['run_root'], arm, receipt, links)
            snap['weights'], snap['policy'] = weights, dict(state_dict_sha256=sv._state_sha(weights))
            pairs.append((rb.lesson_policy_spec(snap, receipt['chosen_plan'], receipt), snap))
        return pairs

    def record_scheduled(self, kind, *, results, **kwargs):
        if not results or any(r['status'] != 'graded' for r in results):
            raise ValueError('REFUSED: incomplete scheduled native checks')
        return _seal(dict(status='graded', kind=kind, records=results), 'receipt_sha256')

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)


def backend_factory(cfg, *, prep_dir, stage_dir):
    return Backend(cfg, prep_dir=prep_dir, stage_dir=stage_dir)


def branch_worker(spec, snapshot, cfg, work):
    """Existing native branch replay/CRN/grade machinery, isolated in a CPU worker."""
    from . import sumo_verify as sv, bank_v2v
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    from .policy_synergy import MixedPolicy
    work = Path(work)
    if spec['branch'] in ('fixed_test', 'shared_final') or 'checkpoint' in spec:
        from . import accel_plant
        checkpoint = Path(rc.guard_path(spec['checkpoint']['path']))
        if rc.sha(checkpoint) != spec['checkpoint']['file_sha256']:
            raise ValueError('REFUSED: paired evaluation checkpoint changed')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        weights = dict(saved['base'])
        for prefix in ('layer', 'trigger'):
            weights.update({f't2c.{prefix}.{k}': v for k, v in saved.get(prefix, {}).items()})
        model = load_policy(weights)
        comp = restore_components(cfg, spec['arm'], model, weights)
        hook = EpisodeHook(cfg, spec['arm'], model, comp, selected=[])  # set below for all visible members
        hook.selected = None if not comp else AllMembers()
        job = dict(id=work.name, path=str(work), world=spec['world'], role='train', arm=spec['arm'],
            weights=saved['base'], runtime=cfg['runtime'], checkpoint=cfg['checkpoint'], noise_seed=spec['world']['seed'],
            deterministic=True, guided=False, v2v=True, sim_s=1200, accel_bounds=cfg['accel_bounds'],
            warm_history=cfg.get('warm_history', False))
        result = accel_plant.episode(job, controller_hook=hook)
        counters = safety(result, work/'observable.jsonl.gz')
        return dict({k: deepcopy(spec[k]) for k in ('arm', 'world_id', 'world', 'checkpoint', 'window',
                    'final_worlds_sha256') if k in spec}, status='graded',
                    J_veh_s=float(result['totals']['team_delay_vs']), safety=counters)
    model = load_policy(snapshot['weights'])
    arm = snapshot.get('policy_synergy', {}).get('arm', snapshot['episode']['arm'])
    comp = restore_components(cfg, arm, model, snapshot['weights']) if any(k.startswith('t2c.') for k in snapshot['weights']) else {}
    synergy = snapshot.get('policy_synergy')
    selected = synergy['new_members'] if synergy else AllMembers()
    layer_hook = EpisodeHook(cfg, arm, model, comp, selected=selected, takeover=snapshot['t0'])
    def hook(controller, job, prefixes):
        if synergy:
            old = load_policy(snapshot['u3_weights'])
            mixed = MixedPolicy(model, old, set(synergy['group'])-set(synergy['new_members']),
                                synergy['new_members'] if synergy.get('background_old') else None)
            controller.policy = controller.base_policy = mixed
        relative_plan_hook(controller, job, prefixes)
        layer_hook(controller, job, prefixes)
    runner = sv.LocalBranchRunner(work, controller_hook=hook, policy_loader=load_policy)
    with bank_v2v.enabled(True):
        result = sv.run_branch(spec, snapshot, runner)
    if result['status'] == 'graded':
        w = result['window']
        trace = work/spec['verification_id']/spec['branch']/'observable.jsonl.gz'
        from .replay_bank import ttc_events
        result['safety'] = dict(collisions=int(w['collisions']), teleports=int(w['teleports']),
            hard_brakes=int(w['hard_brake']),
             ttc_events=int(ttc_events(trace, spec['window']['start_s'], spec['window']['end_s'])))
        result['result_path'] = str(trace.parent/'RESULT.json')
    return result


class AllMembers:
    """Marker selecting every observable CAV without enabling training-time teaching."""
    pass


def checkpoint_weights(path):
    rc.guard_path(Path(path).resolve())
    saved = torch.load(path, map_location='cpu', weights_only=False)
    weights = dict(saved['base'])
    for prefix in ('layer', 'trigger'):
        weights.update({f't2c.{prefix}.{k}': v for k, v in saved.get(prefix, {}).items()})
    return weights
