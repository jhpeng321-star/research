"""Explicit bridges between the accepted P1/P2/P3/P4 production contracts.

The canonical T2C registration is immutable. Narrow sibling configurations are
derived, never written back to it. No import/build operation dispatches physics.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
from statistics import fmean

import torch

from . import rerun_cpus as rc, track2_backend as p1, track2_monitor as p3
from . import track2_safety_reward as safety_reward


# Proposed architecture/optimizer settings, bound by the final owner-approval
# document before use. Width/lr/clip/epochs/norm reuse the acceleration actor/PPO;
# the separate KL cap is the registered T2D .045 cap. No driving gate is added.
OPTIMIZATION = dict(layer_config=dict(hidden_dim=64, learning_rate=.0003,
    kl_budget=.045, clip_ratio=.2, max_grad_norm=.5), trigger_candidates=8,
    trigger_partner_slots=4, decision_interval_s=60, trigger_lr=.0003,
    trigger_ppo=dict(clip_ratio=.2, value_weight=.5, entropy_weight=.01,
                     max_grad_norm=.5), layer_epochs=4)
CONTINGENCY = dict(smoke_sim_s=10400, environment_gate_rerun_sim_s=35820)


def backend_config(cfg, *, root=None, smoke=False):
    """Translate T2C's structured slots to P1 without changing default-off APIs."""
    rc.check_pins(cfg)
    safety_reward.settings(cfg)
    if cfg.get('launch_optimization') != OPTIMIZATION:
        raise ValueError('explicit approval-bound production optimization required')
    out = deepcopy(cfg)
    out.update(deepcopy(OPTIMIZATION))
    out.update(track2_backend=True, host='cpuV', dense_teacher=False,
        group_credit=False, coop_reward=False, merge_features=True,
        practice_weighting=True, practice_weighted_fraction=cfg['practice_weighting']['fraction'],
        track2_reward={k: cfg['track2_reward'][k] for k in ('weight', 'provenance')},
        prior_charged_sim_s=960)
    out['budget'] = dict(total=rc.TOTAL_CAP_SIM_S, phases=deepcopy(cfg['budget']['phases']))
    if cfg.get('launch_contingency') is not None:
        if cfg['launch_contingency'] != CONTINGENCY:
            raise ValueError('unregistered smoke/gate contingency')
        out['budget']['phases']['track2_smoke'] += CONTINGENCY['smoke_sim_s']
        out['budget']['phases']['gate_bank_verification_ext'] += CONTINGENCY['environment_gate_rerun_sim_s']
    if 'clean2_budget' in cfg:
        from .clean2_budget import checked
        restart = checked(cfg)
        # P1's native reserve subtracts prior_charged_sim_s from this GLOBAL
        # total. The registration's budget.total is the disjoint NEW-ledger cap.
        out['budget'] = dict(restart['budget'], total=restart['owner_cap_sim_s'],
                             remaining_cap_sim_s=restart['budget']['total'])
        out['prior_charged_sim_s'] = restart['protected_prior_sim_s']
    if 'clean3_budget' in cfg:
        from .clean3_restart import native_budget
        out['budget'], out['prior_charged_sim_s'] = native_budget(cfg)
    if cfg.get('g4_synergy_credit') and not smoke:
        from .g4_runtime import allocate_budget
        allocate_budget(out)
    prior = out['prior_charged_sim_s']
    cap = out['budget']['total'] if 'clean3_budget' in cfg else rc.TOTAL_CAP_SIM_S
    if sum(out['budget']['phases'].values())+prior > cap:
        raise ValueError('complete production allocations plus prior parity exceed owner cap')
    if 'clean2_budget' not in cfg and cfg['budget']['total']+960 > out['budget']['total']:
        raise ValueError('production quote plus prior parity exceeds owner cap')
    out['run_root'] = str(root or cfg['run_root'])
    out['workers'] = cfg['cpu']['workers']//3  # 3 x 8 episodes + 16 diagnostics <= 40
    out['owner_approval_path'] = str(Path(cfg['run_root'])/'OWNER_APPROVAL.json')
    if smoke:
        if cfg.get('g4_synergy_credit'):
            # The fixed G4 registration samples ordinary TRAIN only.
            out['g4_synergy_credit'] = False
        attempt = cfg.get('smoke_attempt', 1)
        if type(attempt) is not int or attempt not in (1, 2):
            raise ValueError('at most two budgeted qualification smoke attempts')
        out['phase_map'] = dict(train='track2_smoke')
        out['job_namespace'] = f'clean1_smoke{attempt}'
        out['worlds_by_update'] = {'1': cfg['worlds_by_update']['1'][:2]}
        if 'clean3_budget' in cfg and cfg.get('mined_moves'):
            out['practice_priority_mined'] = True  # registered clean3 smoke rule (B=R)
    return out


def bank_move_supervision(runtime):
    """State-bound verified-plan CE for the trigger, with a separate .045 KL.

    Reuses the per_seed_v1/effective bank reader and each example's own observed
    trace. The exact move label is inserted into its local candidate slate; it
    is never an executable instruction or a substitute for online sensing.
    """
    import numpy as np
    from . import bank_v2v, bank_breadth
    from .trigger_policy import TriggerBatch
    if runtime.arm == 'A':
        return dict(examples=0, source='A_no_bank')
    cfg, policy, records = runtime.cfg, runtime.policy, []
    contexts = []
    from .wall_w2 import settings as w2_settings
    w2 = w2_settings(cfg)
    observer = policy
    if w2['wall_pressure_features']:
        # W2: bank readers observe from a copy WITHOUT this process's wall memory.
        from .wall_negatives import observer as clean_observer
        observer = clean_observer(policy)
    def consume(*, cfg, receipt, example, plan, pairs, trace):
        frames = bank_v2v.trace_frames(trace)
        for context, label, reader, weight, time in bank_breadth.observable_contexts(
                observer, plan, pairs, frames, t0=receipt['t0'],
                certification=cfg.get('_executable_lesson')):
            if w2['wall_pressure_features']:
                # Same causal observer as online: the reader's own merge-pressure blocks.
                context = dict(context, merge_pressure={v: reader.pressure_blocks[v]
                                                        for v in context['members']
                                                        if v in reader.pressure_blocks})
            # Same acquirability contract as online feasible_proposals: candidates only over
            # the ego's observable CAV slots. With G5 on, local rows also carry sensed CAVs
            # outside the slots (gap geometry), which must never become move members.
            states = {v: (dict(reader.local_states[v], acquirable_cav_ids=list(reader.visible[v]))
                          if 'acquirable_cav_ids' not in reader.local_states[v] else reader.local_states[v])
                      for v in context['members']}
            batch = policy.trigger.decide(context, obs_states=states,
                observable_cavs=reader.visible, rng=np.random.default_rng(cfg['training_seed']),
                generator=torch.Generator().manual_seed(cfg['training_seed']), deterministic=True)
            tables, descriptors, mask = list(batch.tables), batch.descriptors.clone(), batch.mask.clone()
            targets = []
            for i, vid in enumerate(context['members']):
                index = 0 if label.family == 'none' else 1
                if index:
                    target = p1.TriggerMove(label)
                    table = list(tables[i]); table[index] = target; tables[i] = tuple(table)
                    descriptors[i, index] = torch.tensor(policy.trigger._descriptor(target, reader.visible[vid]))
                    mask[i, index] = True
                targets.append(index)
            batch = TriggerBatch(context, tuple(tables), descriptors, mask,
                torch.tensor(targets), batch.logp, batch.values)
            records.append((batch, weight))
            contexts.append(dict(verification_id=receipt['verification_id'],
                example_id=(example or {}).get('example_id', receipt['verification_id']+'#s1'),
                family=label.family, time_s=time, members=list(context['members'])))
    bank_breadth.visit(cfg, runtime.arm, consume)
    if not records:
        raise ValueError('nonempty observable verified-plan trigger supervision required')
    # Equal mass per supported family, then per context (not legacy AWR weight).
    # This uses every usable context without letting a few high-weight old plans
    # dominate. R uses precisely the same distribution over its verified families.
    from collections import Counter
    families = Counter(c['family'] for c in contexts)
    other_families = set()
    def other_support(*, cfg, receipt, example, plan, pairs, trace):
        frames = bank_v2v.trace_frames(trace)
        other_families.update(label.family for _, label, _, _, _ in bank_breadth.observable_contexts(
            policy, plan, pairs, frames, t0=receipt['t0'], certification=cfg.get('_executable_lesson')))
    bank_breadth.visit(cfg, 'R' if runtime.arm == 'B' else 'B', other_support)
    if set(families) != other_families:
        raise ValueError('F3 B/R observable trigger family support differs; add verified guidance')
    negatives = None
    if w2['wall_negative_labels']:
        # W2: this arm's OWN per-seed losing walls as do-not-wall labels (own family-mass bucket).
        from . import wall_negatives
        extra, extra_contexts, negatives = wall_negatives.trigger_records(cfg, runtime.arm, policy)
        records += extra
        contexts += extra_contexts
        families = Counter(c['family'] for c in contexts)
    weights = [1./families[c['family']] for c in contexts]
    records = [(b, w) for (b, _), w in zip(records, weights)]
    reference = []
    for batch, weight in records:
        with torch.no_grad():
            reference.append(policy.trigger.distribution(batch.context, batch.descriptors, batch.mask)[0])
    def loss():
        return torch.stack([w*torch.nn.functional.cross_entropy(
            policy.trigger.distribution(b.context, b.descriptors, b.mask)[0], b.actions)
            for b, w in records]).sum()/sum(w for _, w in records)
    initial = float(loss().detach())
    passes = []
    for _ in range(cfg['imitation_epochs']):
        weights = deepcopy(policy.trigger.state_dict())
        optimizer = deepcopy(policy.trigger_optimizer.state_dict())
        objective = loss()
        policy.trigger_optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(policy.trigger.learning_parameters(),
                                       cfg['trigger_ppo']['max_grad_norm'], error_if_nonfinite=True)
        policy.trigger_optimizer.step()
        with torch.no_grad():
            kl = fmean(float(torch.distributions.kl_divergence(
                torch.distributions.Categorical(logits=before), torch.distributions.Categorical(
                logits=policy.trigger.distribution(b.context, b.descriptors, b.mask)[0])).sum())
                for (b, _), before in zip(records, reference))
        accepted = math.isfinite(kl) and kl <= cfg['layer_config']['kl_budget']
        if not accepted:
            policy.trigger.load_state_dict(weights)
            policy.trigger_optimizer.load_state_dict(optimizer)
        passes.append(dict(accepted=accepted, kl=kl))
        if not accepted:
            break
    policy.trigger_optimizer.zero_grad(set_to_none=True)
    policy.layer._check_frozen()
    with torch.no_grad():
        # Per-context fit, aligned with `contexts` (reporting only; e.g. mined-example fit).
        context_ce = [float(torch.nn.functional.cross_entropy(
            policy.trigger.distribution(b.context, b.descriptors, b.mask)[0], b.actions)) for b, _ in records]
    out = dict(initial_ce=initial, final_ce=float(loss().detach()), context_final_ce=context_ce,
        examples=len(records), passes=passes, contexts=contexts, family_counts=dict(families),
        family_mass={f: 1./len(families) for f in families}, rule=bank_breadth.RULE,
        source='all usable verified bank plans and observable traces, including PLAN_FIX outputs')
    if negatives is not None:
        out['wall_negatives'] = negatives
    return out


def monitor_config(cfg, *, root=None):
    """P2/P3 use the same boolean reward switch and registered scalar fields."""
    safety_reward.settings(cfg)
    out = deepcopy(cfg)
    reward = cfg['track2_reward']
    out.update(run_root=str(root or cfg['run_root']), track2_reward=reward['enabled'],
        track2_reward_weight=reward['weight'], track2_reward_provenance=reward['provenance'],
        track2_reward_arms=['B', 'R'])
    return out


def update_metrics(arm, episodes, learning, reward_envelope, *, augmented_examples):
    """Measure P3 fields from actual P1 executions; unavailable levers are N/A."""
    from . import moves
    from .merge_features import MERGE_FEATURES
    metrics = dict.fromkeys(p3.METRICS, None)
    metrics.update(dict.fromkeys(p3.COUNTS, 0))
    metrics.update(families=list(moves.FAMILIES),
        trigger_counts=dict.fromkeys(moves.FAMILIES, 0), safety_stop=False,
        safety=dict.fromkeys(p3.SAFETY, 0), track2_reward=None)
    missing, kls = [], []
    proposal_receipts = []
    rewards = []
    for episode in episodes:
        total = episode['totals']
        for key, raw in (('collisions', 'collisions'), ('teleports', 'teleports'),
                         ('hard_brakes', 'hard_brake')):
            value = float(total[raw])
            # P3 counts events; the inherited counter is vehicle-seconds at .5s.
            # Preserve the unrounded physical vector in the update receipt; the
            # conservative integer ceiling is explicitly labelled below.
            metrics['safety'][key] += math.ceil(value)
        if 'ttc_events' not in episode:
            raise ValueError('actual episode TTC event count required')
        metrics['safety']['ttc_events'] += episode['ttc_events']
        metrics['weighted_starts'] += bool(episode.get('practice'))
        rewards.append(sum(s['reward'] for s in episode['steps']))
        for step in episode['steps']:
            sample = step['sample']
            trigger = sample.get('track2_trigger')
            if trigger is not None:
                if arm != 'A' and 'proposals' in trigger:
                    proposal_receipts.append(trigger['proposals'])
                for table, selected in zip(trigger['tables'], trigger['actions']):
                    move = table[selected]
                    family = 'none' if move is None else move['family']
                    if family not in metrics['trigger_counts']:
                        # Opt-in option families (W1/N2 protocols, N2 joint moves) are counted
                        # in their own key; the default family list is unchanged when OFF.
                        metrics['families'].append(family)
                        metrics['trigger_counts'][family] = 0
                    metrics['trigger_counts'][family] += 1
                    metrics['decisions'] += 1
            for block in sample.get('track2', {}).get('merge_features', {}).values():
                missing.append(float(block[MERGE_FEATURES.index('own_missing')]))
    metrics['reward'] = fmean(rewards)
    metrics['safety_stop'] = any(metrics['safety'][k] for k in ('collisions', 'teleports'))
    if arm != 'A':
        metrics['move_proposals'] = p1.feasible_proposals.aggregate(proposal_receipts)
        if reward_envelope is None:
            raise ValueError('native P2 envelope required for B/R monitor')
        metrics['track2_reward'] = reward_envelope['receipt']
        metrics['augmented_examples'] = augmented_examples
        metrics['merge_missing_share'] = fmean(missing) if missing else 1.
        for row in learning['layer']:
            if row['accepted']:
                kls.append(row['kl'])
        metrics['layer_kl'] = max(kls, default=0.)
        # P1 bank likelihood is effective/Tobit, not discrete move CE. P3's
        # historical field is filled only by an explicitly measured CE supplied
        # by the cross-branch bank head adapter, never relabelled action NLL.
        metrics['bank_move_ce'] = learning['bank_move_ce']
    return metrics


def checkpoint_weights(cfg, arm, update):
    """Bind immutable P1 base AND layer/trigger state into isolated-test weights."""
    path = Path(rc.guard_path(Path(cfg['run_root'])/f'{arm}_checkpoint_{update:03}.pt'))
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['update'] != update:
        raise ValueError('diagnostic checkpoint update mismatch')
    weights = deepcopy(saved['policy'])
    if cfg.get('v2v_intent') and 'intent_head.weight' not in weights:
        # Common u0 is stored as raw MAPPO. Diagnostics must reconstruct it
        # through the SAME registered widening/intent loader as live episodes.
        base, _, _, payload = p1._load_base(cfg, arm, update)
        if payload['update'] != update:
            raise ValueError('diagnostic native-loader update mismatch')
        weights = deepcopy(base.state_dict())
    if arm != 'A':
        side = Path(cfg['run_root'])/f'{arm}_track2_{update:03}.pt'
        state = torch.load(side, map_location='cpu', weights_only=False)
        if state['arm'] != arm or state['update'] != update:
            raise ValueError('diagnostic layer arm/update mismatch')
        for name in ('layer', 'trigger'):
            weights.update({f'p1.{name}.{k}': v for k, v in state['worker'][name].items()})
        weights['p1.arm'] = torch.tensor(1 if arm == 'B' else 2)
    return weights


def load_policy(weights, cfg):
    from .v2v_runtime import from_weights
    base = from_weights({k: v for k, v in weights.items() if not k.startswith('p1.')}, enabled=True)
    if 'p1.arm' not in weights:
        return base.eval()
    arm = {1: 'B', 2: 'R'}[int(weights['p1.arm'])]
    policy = p1.EpisodePolicy(base, cfg, arm)
    state = dict(schema=p1.SCHEMA)
    for name in ('layer', 'trigger'):
        prefix = f'p1.{name}.'
        state[name] = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
    policy.restore_worker(state)
    return policy.eval()


def run_branch(spec, snapshot, cfg, work):
    """Full-window native CRN replay, with P1 policies on the selected members."""
    from . import sumo_verify as sv, bank_v2v
    from .policy_synergy import MixedPolicy
    from .rerun_tracks_backend import relative_plan_hook
    def loader(weights):
        value = load_policy(weights, cfg)
        if isinstance(value, p1.EpisodePolicy):
            seed = int(snapshot['sumo'].get('noise_seed') or 0)
            value.layer_rng.manual_seed(seed)
            import numpy as np
            value.candidate_rng = np.random.default_rng(seed)
        return value
    def hook(controller, job, prefixes):
        relative_plan_hook(controller, job, prefixes)
        new = controller.policy
        synergy = snapshot.get('policy_synergy')
        if synergy:
            old = loader(snapshot['u3_weights'])
            mixed = MixedPolicy(new, old, set(synergy['group'])-set(synergy['new_members']),
                synergy['new_members'] if synergy.get('background_old') else None)
            controller.policy = controller.base_policy = mixed
        original = controller.apply
        def apply(rows, commands, now):
            if isinstance(new, p1.EpisodePolicy):
                # VerifyController restores its ledger at t0; refresh from the
                # live instance so the next decision sees original expiries.
                new.observe(rows, now, command_commitments=controller.active_plans)
            return original(rows, commands, now)
        controller.apply = apply
    runner = sv.LocalBranchRunner(work, controller_hook=hook, policy_loader=loader)
    with bank_v2v.enabled(True):
        result = sv.run_branch(spec, snapshot, runner)
    if result['status'] == 'graded':
        from .replay_bank import ttc_events
        trace = Path(work)/spec['verification_id']/spec['branch']/'observable.jsonl.gz'
        window = result['window']
        result['safety'] = dict(collisions=math.ceil(window['collisions']),
            teleports=math.ceil(window['teleports']), hard_brakes=math.ceil(window['hard_brake']),
            ttc_events=int(ttc_events(trace, spec['window']['start_s'], spec['window']['end_s'])))
        result['result_path'] = str(trace.parent/'RESULT.json')
    return result


def staging_closure(paths):
    """Use accepted P4 on the exact reader-selected transitive input set."""
    from .staging_closure import closure
    files, manifest = closure(paths)
    for path in files:
        rc.guard_path(path)
        if rc.sha(path) != manifest['entries'][path.as_posix()]['sha256']:
            raise ValueError('staging input changed during closure')
    return files, manifest
