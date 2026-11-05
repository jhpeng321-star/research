"""Default-off G4 production capture, paired barrier, ledger and fixed-macro worker.

Physics is exclusively accel_plant's native requests/CommandOwnership path. The
ledger is the reservation journal: the complete paired plan and its successor
state are committed in ONE atomic CHARGES.jsonl replacement under the existing
ACCOUNTING.claim lock. A missing completion after reservation is never retried.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path

import torch

from . import sparse_synergy_credit as credit
from . import track2_backend as tb


def enabled(cfg, update, *, smoke=False):
    binding = credit.settings(cfg)
    return binding is not None and not smoke and update >= binding['first_update']


def allocate_budget(cfg):
    """Derive an owner-bound native phase from existing slack/explicit debits.

    Called after canonical budget validation. The full source config (including
    this prospective allocation) is covered by launch_training.checked_config's
    OWNER_APPROVAL hash. Never increase the owner's total or protected prior.
    """
    binding = credit.settings(cfg)
    if binding is None:
        return
    allocation = cfg.get('g4_budget')
    if (not isinstance(allocation, dict) or set(allocation) !=
            {'phase_id', 'allocation_sim_s', 'debits', 'provenance'}
            or allocation['phase_id'] != binding['phase_id']
            or allocation['allocation_sim_s'] != credit.REGISTRATION['budget_cap_sim_s']
            or not isinstance(allocation['debits'], dict)
            or not isinstance(allocation['provenance'], str) or not allocation['provenance'].strip()):
        raise ValueError('G4 requires owner-bound fixed-cap budget allocation/provenance')
    phases = cfg['budget']['phases']
    if 'g4_synergy_credit' in phases:
        raise ValueError('G4 phase already allocated')
    for phase, debit in allocation['debits'].items():
        if (phase not in phases or type(debit) is not int or not 0 <= debit <= phases[phase]):
            raise ValueError('G4 budget debit exceeds registered source allocation')
        phases[phase] -= debit
    phases['g4_synergy_credit'] = allocation['allocation_sim_s']
    if sum(phases.values())+cfg['prior_charged_sim_s'] > cfg['budget']['total']:
        raise ValueError('G4 allocation exceeds remaining owner budget')
    if 'allocated' in cfg['budget']:
        cfg['budget']['allocated'] = sum(phases.values())


def _bytes_sha(value):
    return hashlib.sha256(value).hexdigest()


def state_digest(value):
    """Content hash, independent of torch pickle storage IDs."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return credit.digest(dict(dtype=str(tensor.dtype), shape=list(tensor.shape),
            bytes=_bytes_sha(tensor.numpy().tobytes())))
    if isinstance(value, dict):
        return credit.digest({k: state_digest(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return credit.digest([state_digest(v) for v in value])
    return credit.digest(value)


def _publish(path, data, *, replace=False):
    path = tb._path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temp, path)
    else:
        os.link(temp, path)
        temp.unlink()
    if os.name == 'posix':
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def publish(path, value):
    _publish(path, (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode())


def read(path):
    return json.loads(tb._path(path).read_text(encoding='utf-8'))


@contextmanager
def accounting_lock(root):
    with (tb._path(root)/'ACCOUNTING.claim').open('a+b') as stream:
        if os.name == 'posix':
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        else:
            import msvcrt
            if stream.tell() == 0:
                stream.write(b'\0'); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if os.name == 'posix':
                fcntl.flock(stream, fcntl.LOCK_UN)
            else:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


class BoundaryCapture:
    """Capture BEFORE policy.sample; publish only sampled accepted joint starts.

    Prefix fingerprints are native plant telemetry, not policy inputs. No SUMO
    save/load or parameter/truth reader is used: replay restores physical state
    via the checked prefix and causal controller state via this capture.
    """
    @classmethod
    def for_job(cls, job, controller, prefixes):
        if (job.get('practice') or job.get('deterministic') or job['arm'] == 'A'
                or credit.settings(job['track2_config']) is None):
            return None
        if not enabled(job['track2_config'], job['update']):
            return None
        return cls(job, controller, prefixes)

    def __init__(self, job, controller, prefixes):
        self.job, self.controller, self.prefixes = job, controller, prefixes
        self.captures, self.pending = {}, None

    def before_tick(self, now, rows):
        self.pending = None
        policy, controller = self.controller.policy, self.controller
        if (not any(r['is_cav'] for r in rows) or (policy.last_decision is not None
                and now-policy.last_decision < policy.cfg['decision_interval_s'])):
            return
        # The payload is captured before observe/sample mutates boards or RNG.
        self.pending = dict(time_s=now, job={k: deepcopy(self.job[k]) for k in
            ('id', 'world', 'arm', 'runtime', 'checkpoint', 'noise_seed', 'accel_bounds',
             'warm_history', 'weights', 'track2_config', 'track2_state')},
            prefixes=deepcopy(self.prefixes),
            samples={str(e['time_s']): deepcopy(e['sample']) for e in controller.events},
            generator_state=controller.generator.get_state().clone(),
            memory=deepcopy(controller.memory), reports=deepcopy(policy.reports),
            previous_frame=deepcopy(getattr(controller, 'previous_frame', None)),
            intent_board=deepcopy(getattr(controller, 'intent_board', {})),
            trigger_intent_board=deepcopy(getattr(controller, 'trigger_intent_board', {})))

    def after_tick(self, event):
        if self.pending is None:
            return
        episode = dict(id=self.job['id'], arm=self.job['arm'], world=self.job['world'], steps=[event])
        rows = [r for r in credit.opportunities([episode]) if credit._sampled(
            r, episode, self.job['track2_config'])]
        if not rows:
            self.pending = None
            return
        payload = self.pending
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        data = buffer.getvalue()
        capture = dict(stage='pre_action', time_s=payload['time_s'],
            snapshot_sha256=_bytes_sha(data), prefix_sha256=credit.digest(payload['prefixes']),
            crn_sha256=state_digest({k: payload[k] for k in ('generator_state', 'memory',
                'previous_frame', 'intent_board', 'trigger_intent_board')}),
            policy_sha256=state_digest(payload['job']['track2_state']),
            base_policy_sha256=state_digest(payload['job']['weights']))
        for row in rows:
            key = row['opportunity_id']
            _publish(Path(self.job['path'])/'g4_captures'/f'{key}.pt', data)
            self.captures[key] = deepcopy(capture)
        self.pending = None


def input_path(cfg, arm, update):
    return tb._path(Path(cfg['run_root'])/'g4'/f'{arm}_INPUT_{update:03}.json')


def _envelope_path(cfg, update):
    return tb._path(Path(cfg['run_root'])/'g4'/f'ENVELOPE_{update:03}.json')


def _ordinary(episodes):
    return [e for e in episodes if not e.get('practice')]


def paired_practice_count(cfg, update):
    value = read(input_path(cfg, 'B', update))
    if (value['binding'] != credit.settings(cfg) or value['update'] != update
            or value['arm'] != 'B'):
        raise ValueError('G4 practice count lacks paired collection binding')
    return sum(bool(e.get('practice')) for e in value['episodes'])


class Coordinator:
    def __init__(self, cfg, *, worker=None):
        self.cfg = cfg
        self.root, self.cap, self.phases = tb.reservation_limits(cfg)
        self.worker = worker or run_branch

    def measure(self, update, episodes_by_arm):
        """One complete paired transaction. Failed/incomplete attempts stay spent."""
        tb._approved(self.cfg)
        ordinary = {a: _ordinary(es) for a, es in episodes_by_arm.items()}
        captures = {a: {key: capture for e in es for key, capture in e['g4_captures'].items()}
                    for a, es in ordinary.items()}
        target = _envelope_path(self.cfg, update)
        with accounting_lock(self.root):
            ledger_path = self.root/'CHARGES.jsonl'
            prior_bytes = ledger_path.read_bytes() if ledger_path.exists() else b''
            if prior_bytes and not prior_bytes.endswith(b'\n'):
                raise ValueError('G4 refuses an incomplete ledger tail')
            old = [json.loads(line) for line in prior_bytes.splitlines()]
            plans = [r['g4_plan'] for r in old if 'g4_plan' in r]
            state = credit.initial_state(self.cfg)
            previous = None
            for plan in plans:
                credit._validate_plan(plan)
                if plan['binding'] != state['binding']:
                    # Each successor must use its own ledger/approval, not reset
                    # the sparse-start allowance by renaming a phase in place.
                    raise ValueError('G4 ledger already bound to another phase')
                if plan['reservation_before'] != state:
                    raise ValueError('G4 reservation history discontinuity')
                state = plan['reservation_after']
                if plan['update'] == update:
                    previous = plan
            if previous is not None:
                expected, _ = credit.plan_update(self.cfg, update, ordinary, captures,
                                                  previous['reservation_before'])
                if expected != previous:
                    raise ValueError('G4 restart inputs changed')
                if not target.exists():
                    raise ValueError('G4 reserved attempt incomplete; no retry or recycling')
                envelope = read(target)
                if envelope['plan'] != previous or envelope != credit.measured_envelope(previous, envelope['results']):
                    raise ValueError('G4 persisted envelope changed')
                return envelope
            plan, _ = credit.plan_update(self.cfg, update, ordinary, captures, state)
            phase = 'g4_synergy_credit'
            amount = plan['quote_sim_s']
            if (phase not in self.phases or sum(r['sim_s'] for r in old)+amount > self.cap
                    or sum(r['sim_s'] for r in old if r['phase'] == phase)+amount > self.phases[phase]
                    or any(sum(r['sim_s'] for r in old if r['phase'] == p) > limit
                           for p, limit in self.phases.items())):
                raise ValueError('G4 complete paired reservation exceeds remaining owner/phase budget')
            jobs = [dict(id='g4_'+j['job_sha256'], sim_s=j['sim_s'], phase=phase) for j in plan['jobs']]
            # A zero-cost marker journals empty updates too, without physics.
            if not jobs:
                jobs = [dict(id='g4_empty_'+plan['plan_sha256'], sim_s=0, phase=phase)]
            if {j['id'] for j in jobs} & {r['id'] for r in old}:
                raise ValueError('duplicate G4 reservation')
            jobs[0]['g4_plan'] = plan
            # Preserve the existing append-only ledger prefix byte-for-byte;
            # atomically publish just one complete additional paired batch.
            suffix = ''.join(json.dumps(r, sort_keys=True, allow_nan=False)+'\n' for r in jobs)
            _publish(ledger_path, prior_bytes+suffix.encode(), replace=True)
        # Everything is charged durably BEFORE admission or the first worker.
        results = [self.worker(job, self.cfg) for job in plan['jobs']]
        envelope = credit.measured_envelope(plan, results)
        if not plan['jobs']:
            # Keep the existing ledger reconciliation contract, even for a
            # zero-cost journal marker. This is explicitly not a physics row.
            publish(Path(self.cfg['run_root'])/'jobs'/jobs[0]['id']/'RESULT.json',
                dict(id=jobs[0]['id'], source='g4_empty_reservation', actual_sim_s=0,
                     charged_sim_s=0, plan_sha256=plan['plan_sha256']))
        publish(target, envelope)
        return envelope


def paired_envelope(cfg, arm, update, episodes, *, wait):
    """Called by BOTH detached owners before either optimizer. B owns dispatch."""
    if arm not in ('B', 'R') or not enabled(cfg, update):
        raise ValueError('G4 paired barrier requires enabled ordinary B/R update')
    path = input_path(cfg, arm, update)
    value = dict(binding=credit.settings(cfg), update=update, arm=arm, episodes=episodes)
    if path.exists():
        if read(path) != value:
            raise ValueError('G4 published collection changed')
    else:
        publish(path, value)
    if arm == 'B':
        other = wait(input_path(cfg, 'R', update))
        if any(other[k] != value[k] for k in ('binding', 'update')) or other['arm'] != 'R':
            raise ValueError('G4 paired collection binding mismatch')
        return Coordinator(cfg).measure(update, dict(B=episodes, R=other['episodes']))
    envelope = wait(_envelope_path(cfg, update))
    # Validate local on-policy binding before handing it to the learner too.
    credit.trigger_terms(cfg, arm, update, episodes, envelope)
    return envelope


def load_capture(job, cfg):
    start = job['start']
    episode_id, key = start['episode_id'], start['opportunity_id']
    if Path(episode_id).name != episode_id or not credit._sha(key):
        raise ValueError('G4 capture identity/path mismatch')
    data = tb._path(Path(cfg['run_root'])/'jobs'/episode_id/'g4_captures'/f'{key}.pt').read_bytes()
    capture = start['capture']
    if _bytes_sha(data) != capture['snapshot_sha256']:
        raise ValueError('G4 immutable capture bytes changed')
    payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
    origin = payload['job']
    if (origin['id'] != episode_id or origin['arm'] != start['arm']
            or payload['time_s'] != start['time_s']
            or credit.settings(origin['track2_config']) != job['binding']
            or credit.digest(payload['prefixes']) != capture['prefix_sha256']
            or state_digest(origin['weights']) != capture['base_policy_sha256']
            or state_digest(origin['track2_state']) != capture['policy_sha256']
            or state_digest({k: payload[k] for k in ('generator_state', 'memory',
                'previous_frame', 'intent_board', 'trigger_intent_board')}) != capture['crn_sha256']):
        raise ValueError('G4 pre-action capture binding mismatch')
    return payload


def branch_hook(job, payload, observer):
    """Use the real V2V controller and requests/safety writer, never TraCI writes.

    Replayed samples retain every prefix command. At t0 restore the causal
    boards and AR memory, then sample ONLY the frozen base for all future ticks.
    Mask offsets on the ORIGINAL macro using EACH actor's sensing/t-1 V2V view.
    """
    def install(controller, plant_job, prefixes):
        original = controller.apply
        t0 = job['start']['time_s']
        observer.reports = deepcopy(payload['reports'])
        base = controller.policy

        class FixedMacro:
            base_policy = base

            def __getattr__(self, key):
                return getattr(base, key)

            def sample(self, context, generator, deterministic=False, memory=None):
                sample, memory = base.sample(context, generator, False, memory)
                offsets = {}
                for i, vid in enumerate(context['members']):
                    value = credit.command_offsets(job, int(observer.now)-t0,
                        observable_state={'s_m': observer.execution_positions[vid]}).get(vid, 0.)
                    sample['action'][i][0] += value
                    offsets[vid] = value
                sample['g4_fixed_macro'] = dict(job_sha256=job['job_sha256'], offsets=offsets)
                return sample, memory

        fixed = FixedMacro()

        def apply(rows, commands, now):
            t = int(now)
            if t < t0:
                controller.replay = {str(t): payload['samples'][str(t)]}
            else:
                controller.replay = None
                if t == t0:
                    # Common stochastic continuation, conditional on the SAME
                    # pre-action capture. Seed IDs alter only suffix noise;
                    # never the world or replay prefix. Identical across masks.
                    seed = int(credit.digest([job['start']['capture']['crn_sha256'],
                                             job['verification_seed']])[:15], 16)
                    controller.generator.set_state(payload['generator_state'])
                    if job['verification_seed'] != credit.REGISTRATION['verification_seeds'][0]:
                        controller.generator.manual_seed(seed)
                    controller.memory = deepcopy(payload['memory'])
                    for key in ('previous_frame', 'intent_board', 'trigger_intent_board'):
                        setattr(controller, key, deepcopy(payload[key]))
                    controller.active_plans = []
                    controller.guide = None
                    controller.policy = fixed
                observer.observe(rows, now)
            result = original(rows, commands, now)
            if t >= t0:
                event = controller.events[-1]
                if ('g4_fixed_macro' not in event['sample'] or event.get('proposal') is not None
                        or set(event['requests'])-set(event['sample']['context']['members'])):
                    raise ValueError('G4 intervention/command ownership mismatch')
            return result

        controller.apply = apply
    return install


def run_branch(job, cfg):
    """Production worker adapter; invoked only after the complete ledger commit."""
    from . import accel_plant
    from .branch_eval import totals_at, check_prefix
    from .v2v_runtime import from_weights
    from .replay_bank import ttc_events
    from .runner import admission
    tb._approved(cfg)
    admission(cfg)
    payload = load_capture(job, cfg)
    source = payload['job']
    base = from_weights(source['weights'], enabled=True).eval()
    # Reuse ONLY the production locality builder; no trigger/layer sampling.
    observer = tb.EpisodePolicy(deepcopy(base), source['track2_config'], source['arm'])
    path = tb._path(Path(cfg['run_root'])/'jobs'/('g4_'+job['job_sha256']))
    native = dict(id='g4_'+job['job_sha256'], path=str(path), world=source['world'],
        arm=source['arm'], role='train', weights=source['weights'],
        runtime=source['runtime'], checkpoint=source['checkpoint'],
        noise_seed=source['noise_seed'], accel_bounds=source['accel_bounds'],
        warm_history=source['warm_history'], sim_s=1200, stop=1200, deterministic=False,
        v2v=True, b_replay=True, guided=False, queue_root=cfg['run_root'],
        branch=dict(time_s=job['start']['time_s'], revert=[], prefixes=payload['prefixes']))
    # accel_plant's v2v wrapper remains the sole wrapper; no global monkeypatch.
    result = accel_plant.episode(native, policy_loader=lambda weights: base,
        controller_hook=branch_hook(job, payload, observer))
    check_prefix(result, dict(prefixes=payload['prefixes']), job['start']['time_s'])
    start, end = totals_at(result, job['window']['start_s']), totals_at(result, 1200)
    if result['actual_sim_s'] != 1200:
        raise ValueError('G4 incomplete common window')
    return dict(job_sha256=job['job_sha256'], source='sumo', status='graded',
        capture=job['start']['capture'], window=job['window'],
        retained_members=job['retained_members'], verification_seed=job['verification_seed'],
        actual_sim_s=result['actual_sim_s'], J_veh_s=end['team_delay_vs']-start['team_delay_vs'],
        safety=dict(collisions=end['collisions']-start['collisions'],
            teleports=end['teleports']-start['teleports'],
            hard_brakes=end['hard_brake']-start['hard_brake'],
            ttc_events=int(ttc_events(path/'observable.jsonl.gz', job['window']['start_s'], 1200))))
