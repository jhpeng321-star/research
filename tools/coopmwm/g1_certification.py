"""Native G1 collection: exact F3 restarts, fixed library masks, no WM substitute.

Preparation only reads verified TRAIN sources. Execution uses the existing
snapshot controller and final safety/ownership writer, in isolated workers.
Every full-prefix branch is reserved before any worker starts; no retry path.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import torch

from . import executable_labels as labels, moves
from .protocol import digest


def innovation(seed_sha256, time_s, members):
    """Covariance RHO, keyed by identity rather than active-population order."""
    from .acceleration import RHO
    def normal(key):
        seed = int(digest([seed_sha256, time_s, key])[:15], 16)
        return float(torch.randn((), generator=torch.Generator().manual_seed(seed)))
    shared = normal(['shared'])
    return {v: math.sqrt(RHO)*shared+math.sqrt(1-RHO)*normal(['member', v]) for v in members}


class KeyedPolicy:
    def __init__(self, base, seed_sha256):
        self.base_policy, self.seed_sha256, self.now = base, seed_sha256, None

    def __getattr__(self, key):
        return getattr(self.base_policy, key)

    def __call__(self, *args, **kwargs):
        return self.base_policy(*args, **kwargs)

    @torch.no_grad()
    def sample(self, context, generator, deterministic=False, memory=None):
        from .acceleration import PHI, RHO
        from tools.coopfree.policy import gaussian_log_prob
        if deterministic or self.now is None:
            raise ValueError('G1 requires stochastic, time-keyed native base continuation')
        # Keep any native intent metadata; its unkeyed throwaway noise is never
        # issued and does not advance the capture generator or another branch.
        throwaway = torch.Generator().manual_seed(int(digest([self.seed_sha256, self.now])[:15], 16))
        sample, _ = self.base_policy.sample(context, throwaway, False, memory)
        mean, value = self.base_policy(context)
        members = context['members']
        previous = torch.tensor([[(memory or {}).get(v, 0.)] for v in members]).reshape(-1, 1)
        draws = innovation(self.seed_sha256, self.now, members)
        sigma = self.base_policy.log_std.exp()*math.sqrt(1-PHI**2)
        noise = PHI*previous+sigma*torch.tensor([[draws[v]] for v in members]).reshape(-1, 1)
        action = mean+noise
        logp = gaussian_log_prob(action, mean+PHI*previous, RHO, sigma) if len(mean) else mean.sum()
        sample.update(action=action.tolist(), own_action=action.tolist(), previous_noise=previous.tolist(),
            logp=float(logp), value=float(value), g1_crn=dict(time_s=self.now,
                seed_sha256=self.seed_sha256, innovations=draws))
        return sample, dict(zip(members, noise.flatten().tolist()))


def local_view(frame, macro, decision_s, partner_slots):
    """The same causal sensing/selected-partner geometry as P1.observe."""
    if frame['time_s'] != decision_s-1:
        raise ValueError('G1 causal frame must precede execution by exactly one second')
    from .track2_backend import SENSOR_RANGE_M
    table = {r['vid']: r for r in frame['vehicles']}
    for ego in macro.members:
        if ego not in table or not table[ego]['is_cav']:
            continue
        sensed = [r for r in table.values()
                  if abs(r['s_m']-table[ego]['s_m']) <= SENSOR_RANGE_M]
        partners = sorted((r['vid'] for r in sensed if r['is_cav'] and r['vid'] != ego),
            key=lambda v: (abs(table[v]['s_m']-table[ego]['s_m']), v))[:partner_slots-1]
        available = [ego]+partners
        rows = [{k: r[k] for k in ('vid', 's_m', 'v_mps', 'lane', 'is_cav')}
                for r in sensed if not r['is_cav'] or r['vid'] in available]
        local = dict(self=ego, rows=rows, observed_at_s=decision_s-1,
            acquirable_cav_ids=available, offered_moves=[macro.to_dict()])
        try:
            labels._local(macro, local, decision_s)
        except ValueError:
            continue
        return local
    raise ValueError('G1 exact macro is not executable in any proposer causal local view')


def select_matched(candidates):
    """Equal B/R family, arity, seed and horizon effort, before any outcomes."""
    pools = {a: {} for a in ('B', 'R')}
    for item in candidates:
        lesson = item['job']['lesson']
        key = (lesson['macro']['family'], len(lesson['macro']['members']),
               lesson['window']['end_s'])
        pools[lesson['arm']].setdefault(key, []).append(item)
    result, seen = [], {a: set() for a in pools}
    keys = sorted(set(pools['B']) & set(pools['R']))
    for key in keys:
        ranked = {}
        for arm in pools:
            ordered = labels.prioritize([i['job']['lesson'] for i in pools[arm][key]])
            by_sha = {i['job']['lesson']['lesson_sha256']: i for i in pools[arm][key]}
            ranked[arm] = []
            for lesson in ordered:
                identity = (lesson['snapshot_sha256'], lesson['decision_s'])
                if identity in seen[arm]:
                    continue
                seen[arm].add(identity)
                ranked[arm].append(by_sha[lesson['lesson_sha256']])
        n = min(len(ranked['B']), len(ranked['R']), 8-len(result)//2)
        for i in range(n):
            result.extend([ranked['B'][i], ranked['R'][i]])
    labels.budget_quote([item['job'] for item in result])
    return result


def prepare(cfg, runtime_sha256):
    from . import bank_breadth as breadth, bank_v2v, replay_bank as rb, sumo_verify as sv
    from .track2_backend import _bank_view
    candidates, diagnostics = [], []
    for arm in ('B', 'R'):
        def consume(*, cfg, receipt, example, plan, pairs, trace):
            identity = dict(arm=arm, verification_id=receipt['verification_id'],
                example_id=(example or {}).get('example_id'), source_plan_sha256=digest(plan))
            try:
                with _bank_view(cfg):
                    snapshot, _ = breadth.verified_restart(cfg['run_root'], arm, receipt, example, plan)
                macro = moves.from_plan(plan)
                if len(macro.actors) < 2:
                    raise ValueError('source projects to fewer than two executing actors')
                decision = snapshot['t0']+1
                occupied = {v for c in snapshot['committed_plans']
                    if c['start_s']+c['plan']['duration_s'] > decision for v in c['plan']['members']}
                if set(macro.members) & occupied:
                    raise ValueError('source macro overlaps an existing command owner')
                frame = bank_v2v.trace_frames(trace)[decision-1]
                local = local_view(frame, macro, decision, cfg['trigger_partner_slots'])
                lesson = labels.bind_lesson(arm=arm, receipt=receipt, example=example, plan=plan,
                    **labels.restart_binding(snapshot), runtime_sha256=runtime_sha256,
                    local=local, decision_s=decision, window=receipt['window'])
                snapshots = [snapshot, rb.replicate(snapshot, 'g1_'+lesson['lesson_sha256']+'_seed2')]
                seeds = [dict(verification_seed=i+1, snapshot_sha256=sv.snapshot_sha(s),
                    rng_sha256=s['rng']['generator_state_sha256']) for i, s in enumerate(snapshots)]
                candidates.append(dict(job=labels.job_spec(lesson, seeds), snapshots=snapshots))
            except (ValueError, KeyError, TypeError) as error:
                diagnostics.append(dict(identity, reason=str(error)))
        breadth.visit(cfg, arm, consume)
    selected = select_matched(candidates)
    return selected, dict(candidates=len(candidates), selected=len(selected), diagnostics=diagnostics,
        quote=labels.budget_quote([item['job'] for item in selected]),
        selection='native projection; two executing actors; matching B/R family/arity/horizon; no outcome ranking')


def install_macro(controller, spec, snapshot):
    """Add ONLY selected macro offsets to native base requests after the capture.

    The snapshot's prefix, live t0 action, generator/pad and AR state remain the
    native VerifyController's responsibility. No newly proposed later move.
    """
    macro = moves.Move.from_dict(spec['macro'])
    original = controller.apply
    selected = set(spec['selected_actors'])
    keyed = KeyedPolicy(controller.policy, spec['seed']['rng_sha256'])

    def apply(rows, commands, now):
        if int(now) > snapshot['t0']:
            keyed.now = int(now)
            controller.policy = keyed
        desired = original(rows, commands, now)
        t = int(now)
        if t < spec['decision_s']:
            return desired
        event = controller.events[-1]
        # No mutation of the snapshot or a shared branch sample.
        event['requests'] = desired = deepcopy(desired)
        table = {r['vid']: r for r in rows if r['is_cav']}
        positions = {}
        from .track2_backend import SENSOR_RANGE_M
        for vid in macro.members:
            if vid in table:
                positions[vid] = {v: r['s_m'] for v, r in table.items()
                    if abs(r['s_m']-table[vid]['s_m']) <= SENSOR_RANGE_M}
        event['g1_observable_positions'] = positions
        for vid in macro.members:
            if vid not in selected or vid not in commands or vid not in table:
                continue
            state = dict(s_m=positions[vid])
            offset = moves.offsets(macro, t-spec['decision_s'], state)[vid]
            if offset == 0:
                continue
            without = desired.get(vid, {}).get('requested_raw', commands[vid])
            raw = without+offset
            desired[vid] = dict(requested_raw=raw, acceleration_residual=raw-commands[vid],
                library_move_offset_mps2=offset, without_library_raw=without)
        return desired
    controller.apply = apply


def run_branch(spec, snapshot, root):
    from . import accel_plant, sumo_verify as sv
    from .acceleration import Policy
    from .replay_bank import ttc_events
    from .branch_eval import totals_at
    labels.check(spec, 'branch_sha256')
    if sv.snapshot_sha(snapshot) != spec['seed']['snapshot_sha256']:
        raise ValueError('G1 immutable verification snapshot changed')
    if snapshot['rng']['generator_state_sha256'] != spec['seed']['rng_sha256']:
        raise ValueError('G1 verification RNG binding changed')
    if snapshot['policy']['state_dict_sha256'] != spec['base_policy_sha256']:
        raise ValueError('G1 base policy binding changed')
    if sv._state_sha(snapshot['weights']) != spec['base_policy_sha256']:
        raise ValueError('G1 actual base weights differ from the immutable source binding')
    holder = {}
    def hook(controller, job, prefixes):
        # accel_plant practice factory installs the unchanged native verifier.
        holder['controller'] = controller
        install_macro(controller, spec, snapshot)
    path = Path(root)/'jobs'/spec['branch_sha256']
    ep, sumo = snapshot['episode'], snapshot['sumo']
    native = dict(id='g1_'+spec['branch_sha256'], path=str(path),
        world=ep['world'], role='train', arm=ep['arm'], weights=snapshot['weights'],
        runtime=sumo['runtime'], checkpoint=sumo['checkpoint'],
        noise_seed=sumo.get('noise_seed') or 0, deterministic=False,
        guided=False, b_replay=True, health_guards=False,
        sim_s=spec['sim_s'], stop=spec['window']['end_s'],
        warm_history=sumo['warm_history'], accel_bounds=sumo['accel_bounds'],
        v2v=int(snapshot['weights']['actor.0.weight'].shape[1]) != 33,
        practice=dict(snapshot=snapshot),
        branch=dict(time_s=snapshot['t0'], revert=[], prefixes=snapshot['prefixes']))
    def load(weights):
        # Historical verified sources use the original acceleration actor. A
        # V2V source, if encountered, must use its explicit native state shape.
        if any(k.startswith('actor.') for k in weights):
            from .v2v_runtime import from_weights
            return from_weights(weights, enabled=True).eval()
        base = Policy()
        base.load_state_dict(weights, strict=True)
        return base.eval()
    result = accel_plant.episode(native, controller_hook=hook, policy_loader=load)
    sv.check_prefix(result, snapshot)
    start, end = spec['window']['start_s'], spec['window']['end_s']
    first, last = totals_at(result, start), totals_at(result, end)
    steps = [s for s in result['steps'] if start <= s['time_s'] < end]
    if len(steps) != end-start or result['actual_sim_s'] != spec['sim_s']:
        raise ValueError('G1 incomplete native full-prefix/window result')
    for step in steps:
        if step['time_s'] > snapshot['t0']:
            crn = step['sample']['g1_crn']
            if crn != dict(time_s=step['time_s'], seed_sha256=spec['seed']['rng_sha256'],
                    innovations=innovation(spec['seed']['rng_sha256'], step['time_s'],
                        step['sample']['context']['members'])):
                raise ValueError('G1 member-keyed CRN evidence mismatch')
    audit = []
    for step in steps:
        for row in step.get('library_actuation_audit', []):
            if row.get('vid') in spec['selected_actors'] and 'library_issued_delta_mps2' in row:
                audit.append(dict(row, time_s=step['time_s'],
                    observable_state=dict(s_m=step['g1_observable_positions'][row['vid']])))
    return labels.seal(dict(branch_sha256=spec['branch_sha256'], status='graded',
        window=spec['window'], prefix_matched=True, crn_aligned_through_s=end,
        actual_sim_s=result['actual_sim_s'], delay_increments=[-s['reward'] for s in steps],
        safety=dict(collisions=int(last['collisions']-first['collisions']),
            teleports=int(last['teleports']-first['teleports']),
            hard_brake=int(last['hard_brake']-first['hard_brake']),
            ttc_below_1p5_events=int(ttc_events(path/'observable.jsonl.gz', start, end))),
        library_actuation_audit=audit, native_result_sha256=digest(result),
        native_trace_sha256=hashlib.sha256((path/'observable.jsonl.gz').read_bytes()).hexdigest()),
        'result_sha256')
