"""Effective (censored) imitation target space for the bank AWR phase (c4/effective-imitation).

DIAGNOSIS (update-6 isolated synergy, real tapes): the MAPPO-initialised policy commands
>= 2.0 m/s^2 on 95% of steps while the CAV vType ``cav_typical`` accel cap is 1.2 m/s^2
(traffic_shepherd/c4/population.py, speedMode 27 also respects the leader safe speed), so
most commands lie in a DEAD ZONE: anything above u_t = min(1.2, leader-allowed acceleration
at step t) has identical physical effect. A raw AWR likelihood then rewards commands the
plant cannot execute, and the learned shift never crosses the binding limit.

Option ``imitation_target_space`` ('raw' = default, byte-identical; 'effective'):

- BINDING LIMIT u_t: for every imitation pair/step and proposal member, u_t (absolute
  m/s^2) is derived from the winning plan branch's recorded observable tape
  (observable.jsonl.gz): the safety-boundary issued command ``actions[].issued_safe`` and
  the realised acceleration (v_mps difference between consecutive seconds), each counted
  ONLY where the plan's own command exceeded it, under the vType accel cap read from the
  ``cav_typical`` vType definition. Never imputed from another vehicle, second or branch.
  Residual space: the policy acts on acceleration residuals (command - native), so the
  sample carries ``binding_limits`` = {vid: u_t - native_command}.
- CENSORED (Tobit) LIKELIHOOD: where the plan command is below its limit the pair
  contributes the exact Gaussian log-density at the plan command (identical to the raw
  term); where the plan command reached the limit the pair contributes log P(policy
  command >= limit) under the same AR(1)-correlated Gaussian (shared factor integrated by
  Gauss-Hermite), so ANY policy action above the limit is equally correct there and the
  likelihood never pulls the policy into the dead zone. AWR weights are unchanged.
- EFFECTIVE ERROR REPORT: L1 error between min(policy, u_t) and min(plan command, u_t),
  plus the share of binding target steps and, on binding steps, the mean policy command
  vs the mean limit.
- INTENT TARGETS: the plan members' future plan commands are clipped to their limits
  (v2v_intent.attach_bank_targets(..., limits=...)).

The KL trust region, the AWR weights and the kl_targeted step mode are untouched.
"""
import math

import numpy as np
import torch

from .acceleration import RHO
from tools.coopfree.policy import gaussian_log_prob

TARGET_SPACES = ('raw', 'effective')
#: The plan command counts as exceeding a recorded observable only above float noise.
BIND_EPS = 1e-9
#: Residual-vs-absolute command consistency tolerance (tape data integrity).
COMMAND_MATCH_TOL = 1e-6
#: Gauss-Hermite nodes for the shared-factor integral (fixed; differentiable evaluations).
GH_NODES = 64


def cav_accel_cap():
    """The CAV vType accel cap (m/s^2) read from the population's ``cav_typical`` vType
    definition - the same params the scenario writer emits for the CAV vType
    (traffic_shepherd/c4/population.py; never hard-coded, never a per-vehicle truth table)."""
    from traffic_shepherd.c4.population import TEXTBOOK_CAV_PARAMS
    cap = float(TEXTBOOK_CAV_PARAMS['accel_mps2'])
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError('cav_typical accel cap must be positive and finite')
    return cap


def binding_limit(plan_command, cap, issued_safe=None, realised_accel=None, eps=BIND_EPS):
    """Effective upper limit u_t (absolute m/s^2) of one vehicle at one step.

    The cap is the vType ceiling; each RECORDED observable (safety-boundary issued command,
    realised acceleration) lowers the limit only where the plan's own command exceeded it:
    anything above u_t had identical effect at that step."""
    u = float(cap)
    for observable in (issued_safe, realised_accel):
        if observable is None:
            continue
        observable = float(observable)
        if not math.isfinite(observable):
            continue
        if plan_command > observable+eps:
            u = min(u, observable)
    return u


def _log_phi(x):
    """log standard-normal CDF, computed in double precision (tails) and cast back."""
    cdf = torch.erfc((-x.double())/math.sqrt(2))*.5
    return torch.log(cdf.clamp_min(1e-300)).to(x.dtype)


_NODES, _WEIGHTS = np.polynomial.hermite.hermgauss(GH_NODES)


def censored_log_prob(action, mean, rho, sigma, limits):
    """Joint AR(1)-correlated Gaussian likelihood with Tobit censoring.

    action/mean/limits: (n,1) tensors; limits carry the binding residual threshold of every
    proposal member (inf = uncensored). A coordinate whose target lies BELOW its limit
    contributes the normal density at the target; a coordinate whose target reached the
    limit (target >= limit) contributes P(policy command >= limit). Both terms are
    integrated over the shared factor z0 (a = mean + sigma*(sqrt(rho) z0 + sqrt(1-rho) eps))
    by fixed-node Gauss-Hermite. With no censored coordinate the result equals
    gaussian_log_prob(action, mean, rho, sigma) exactly."""
    if not 0 <= rho < 1:
        raise ValueError('nonsingular shared mixing required')
    limits = torch.as_tensor(limits, dtype=mean.dtype).reshape(-1, 1)
    censored = torch.isfinite(limits) & (torch.as_tensor(action, dtype=mean.dtype).reshape(-1, 1) >= limits)
    if not bool(censored.any()):
        return gaussian_log_prob(action, mean, rho, sigma)
    scale = torch.as_tensor(sigma, dtype=mean.dtype, device=mean.device)
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError('finite positive sigma required')
    if not bool(torch.isfinite(mean).all()):
        raise ValueError('finite mean required')
    n = len(mean)
    if not n:
        raise ValueError('empty search density')
    nodes = torch.as_tensor(_NODES*math.sqrt(2.), dtype=torch.float64).reshape(1, 1, -1)
    log_weights = torch.as_tensor(np.log(_WEIGHTS/math.sqrt(math.pi)), dtype=torch.float64).reshape(1, -1)
    mean_d, action_d = mean.double().reshape(-1, 1, 1), torch.as_tensor(action).double().reshape(-1, 1, 1)
    cond_mean = mean_d+scale.double()*math.sqrt(rho)*nodes          # (n,1,K)
    cond_scale = scale.double()*math.sqrt(1-rho)                    # scalar
    residual = (action_d-cond_mean)/cond_scale
    log_density = (-.5*residual.square()-torch.log(cond_scale)-.5*math.log(2*math.pi))
    threshold = limits.double().reshape(-1, 1, 1)
    log_survival = _log_phi((cond_mean-threshold)/cond_scale)
    per_node = torch.where(censored.reshape(-1, 1, 1), log_survival, log_density).sum(0)
    value = torch.logsumexp(per_node+log_weights, dim=1)
    if not bool(torch.isfinite(value).all()):
        raise ValueError('nonfinite censored log likelihood')
    return value.to(mean.dtype).sum()


def pair_imitation(policy, sample, weight=0.):
    """The imitation loss term in the effective target space (same shape/normalisation as
    accel_learner.imitation; AWR weight unchanged; requires attached ``binding_limits``)."""
    b = sample.get('behaviour', {})
    mean, _ = policy(sample['context'])
    if not b.get('executed') or weight <= 0:
        return mean.sum()*0
    members = sample['context']['members']
    indices = [members.index(v) for v in b.get('proposal_members', []) if v in members]
    if not indices:
        return mean.sum()*0
    limits = sample.get('binding_limits')
    if not isinstance(limits, dict):
        raise ValueError('effective imitation pair lacks binding_limits')
    missing = [members[j] for j in indices if members[j] not in limits]
    if missing:
        raise ValueError('effective imitation pair lacks binding limits for '+repr(missing))
    action = torch.tensor(sample['action']).reshape(-1, 1)
    thresholds = torch.tensor([[float(limits[members[j]])] for j in indices], dtype=mean.dtype)
    return -weight*censored_log_prob(action[indices], mean[indices], RHO,
                                     policy.log_std.exp(), thresholds)/len(indices)


@torch.no_grad()
def _member_rows(policy, pairs):
    """Yield (mean_j, target_j, limit_j) for every weighted trainable member action."""
    for sample, weight in pairs:
        if weight <= 0:
            continue
        if not sample.get('behaviour', {}).get('executed'):
            continue
        members = sample['context']['members']
        js = [members.index(v) for v in sample['behaviour'].get('proposal_members', []) if v in members]
        if not js:
            continue
        limits = sample.get('binding_limits')
        if not isinstance(limits, dict):
            continue
        mean, _ = policy(sample['context'])
        action = torch.as_tensor(sample['action']).reshape(-1, 1)
        for j in js:
            if members[j] not in limits:
                continue
            yield float(mean[j]), float(action[j]), float(limits[members[j]])


@torch.no_grad()
def effective_action_error(policy, pairs):
    """Unweighted mean L1 between min(policy command, u_t) and min(plan command, u_t) over
    weighted trainable member actions (None when there is none)."""
    total = count = 0.
    for m, target, limit in _member_rows(policy, pairs):
        total += abs(min(m, limit)-min(target, limit))
        count += 1.
    return total/count if count else None


@torch.no_grad()
def effective_diagnostics(policy, pairs):
    """Per-update effective-space diagnostics over weighted trainable pairs: the share of
    binding target steps (plan command at/above its limit), the effective action error on
    those steps, and the mean policy command vs the mean limit on them."""
    binding_error = binding_count = 0.
    policy_sum = limit_sum = 0.
    count = 0
    for m, target, limit in _member_rows(policy, pairs):
        count += 1
        if target >= limit:
            binding_count += 1.
            binding_error += abs(min(m, limit)-limit)
            policy_sum += m
            limit_sum += limit
    return dict(binding_share=binding_count/count if count else None,
                binding_member_actions=int(binding_count), member_actions=count,
                effective_action_error_binding_mps2=binding_error/binding_count if binding_count else None,
                mean_policy_action_on_binding_mps2=policy_sum/binding_count if binding_count else None,
                mean_limit_on_binding_mps2=limit_sum/binding_count if binding_count else None,
                rule='effective space min(command, u_t); binding = plan command at/above its '
                     'recorded limit; u_t from the plan branch observable tape under the '
                     'cav_typical vType cap')


# ---------------------------------------------------------------- tape attachment

def _vehicle(frame, vid):
    for row in frame.get('vehicles', []):
        if row.get('vid') == vid:
            return row
    return None


def _issued_safe(frame, vid):
    """The safety-boundary issued command of vid recorded in this frame (None if absent)."""
    rows = [row for row in frame.get('actions', []) if row.get('vid') == vid]
    for row in reversed(rows):                      # one audit row per tick; last wins
        value = row.get('issued_safe')
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return None


def _plan_command(step, vid):
    """The plan's absolute pre-safety command of vid at this tape step (None if absent)."""
    requests = step.get('requests') or {}
    if vid in requests and requests[vid].get('requested_raw') is not None:
        return float(requests[vid]['requested_raw'])
    accels = (step.get('sample') or {}).get('absolute_accels') or {}
    return float(accels[vid]) if vid in accels and accels[vid] is not None else None


def _native_command(step, vid):
    commands = step.get('commands') or {}
    return float(commands[vid]) if vid in commands else None


def _realised_accel(previous, frame, vid):
    """Realised acceleration between the previous and this second (None if unavailable)."""
    if previous is None:
        return None
    dt = float(frame.get('time_s', 0))-float(previous.get('time_s', 0))
    if dt <= 0:
        return None
    before, now = _vehicle(previous, vid), _vehicle(frame, vid)
    if before is None or now is None:
        return None
    return (float(now['v_mps'])-float(before['v_mps']))/dt


def attach_limits(pairs, tape, trace_path, entry, cap=None):
    """(kept pairs, counts, limits by second): attach ``sample['binding_limits']``
    {vid: u_t - native command} in residual space to every pair whose proposal members all
    have a derivable recorded limit at their second of the plan branch observable tape.
    Pairs without one are EXCLUDED and counted (never imputed). The third return maps
    absolute seconds to {vid: u_t} in ABSOLUTE space for the intent-target clipping.
    Returns ([], counts, {}) when the trace is unreadable."""
    counts = dict(pairs=len(pairs), kept=0, excluded_missing_trace=False,
                  excluded_missing_limit=0, excluded_command_mismatch=0, cap_mps2=None)
    try:
        from .bank_v2v import trace_frames
        frames = trace_frames(trace_path)
    except (FileNotFoundError, OSError):
        counts['excluded_missing_trace'] = True
        return [], counts, {}
    cap = cav_accel_cap() if cap is None else float(cap)
    counts['cap_mps2'] = float(cap)
    steps = {int(s['time_s']): s for s in tape.get('steps', [])}
    t0 = int(entry['start_s'])
    limits_by_time = {}
    kept = []
    for i, (sample, w) in enumerate(pairs):
        t = t0+i
        frame, previous = frames.get(t), frames.get(t-1)
        if frame is None:
            counts['excluded_missing_limit'] += 1
            continue
        members = list(sample['context'].get('members') or sample['behaviour']['proposal_members'])
        step = steps.get(t)
        if step is None:
            counts['excluded_missing_limit'] += 1
            continue
        proposed = [v for v in sample['behaviour'].get('proposal_members', []) if v in members]
        residual, absolute, reason = {}, {}, None
        for vid in proposed:
            command = _plan_command(step, vid)
            native = _native_command(step, vid)
            if command is None or native is None or _vehicle(frame, vid) is None:
                reason = 'missing'
                break
            recorded = float(sample['action'][members.index(vid)][0])
            if abs(recorded-(command-native)) > COMMAND_MATCH_TOL:
                reason = 'mismatch'
                break
            u = binding_limit(command, cap, _issued_safe(frame, vid),
                              _realised_accel(previous, frame, vid))
            residual[vid] = u-native
            absolute[vid] = u
        if reason is not None:
            counts['excluded_command_mismatch' if reason == 'mismatch'
                   else 'excluded_missing_limit'] += 1
            continue
        for t_vid, u in absolute.items():          # only kept pairs publish their limits
            limits_by_time.setdefault(t, {})[t_vid] = u
        sample['binding_limits'] = residual
        kept.append((sample, w))
    counts['kept'] = len(kept)
    return kept, counts, limits_by_time
