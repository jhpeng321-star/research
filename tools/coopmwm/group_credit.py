"""T2E belief-only counterfactual GROUP advantage (opt-in, CPU only).

Outcome utility is negative team delay: credit = utility(move) - utility(base
MAPPO) = base_delay - joint_delay. Broadcast that group credit to the cooperation
layer only; the caller keeps the frozen MAPPO actions/rewards/optimizer untouched.
Solo-vs-joint synergy and member-reversion algebra are diagnostics, not weighted
substitutes for the group reward. Nothing here loads a model or dispatches SUMO.
"""
import copy
import dataclasses
import math
from collections.abc import Mapping
import numpy as np
from . import credit as cr

REGISTRATION = dict(
    metric='team_delay', direction='lower', wm_samples=4,
    sumo_verify_fraction=0.25, verify_mode='all8_both_seeds', isolated=True,
    truth_patterns=('truth', 'hidden', 'oracle', 'global_state',
                                   'vtype', 'driver'),
)
CONFIG_KEYS = frozenset(REGISTRATION) - {'truth_patterns'}


def credit_config(cfg=None):
    cfg = dict(cfg or {})
    unknown = sorted(set(cfg) - CONFIG_KEYS)
    if unknown:
        raise ValueError('unregistered group_credit keys: %r' % (unknown,))
    out = {k: REGISTRATION[k] for k in CONFIG_KEYS}
    for k, v in cfg.items():
        out[k] = v
    if out['metric'] != 'team_delay' or out['direction'] != 'lower':
        raise ValueError("registered metric team_delay, direction 'lower' only")
    for k in ('wm_samples',):
        if type(out[k]) is not int or out[k] < 1:
            raise ValueError('%s positive int required' % k)
    for k in ('sumo_verify_fraction',):
        v = out[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool) \
                or not math.isfinite(v) or not 0. <= float(v) <= 1.:
            raise ValueError('%s in [0,1] required' % k)
    if out['verify_mode'] != 'all8_both_seeds' or out['isolated'] is not True:
        raise ValueError("verify_mode 'all8_both_seeds' with isolated=True required")
    return out


def assert_observable(payload, patterns=None):
    """Reject truth-shaped keys, including nested mappings/dataclass moves.

    This is a defensive schema check, not a provenance oracle: callers must build
    belief and move inputs from own sensing and observable V2V reports only.
    """
    pats = REGISTRATION['truth_patterns'] if patterns is None else patterns
    def walk(node, path):
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            walk({f.name: getattr(node, f.name) for f in dataclasses.fields(node)}, path)
        elif isinstance(node, Mapping):
            for k, v in node.items():
                kl = str(k).lower()
                if any(p in kl for p in pats):
                    raise ValueError('non-observable key %r at %s' % (k, path))
                walk(v, path + '/' + str(k))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, path + '[%d]' % i)
    walk(payload, '')


def _mean(rows):
    for x in rows:
        if not math.isfinite(float(x)):
            raise ValueError('non-finite WM outcome')
    return float(np.mean(rows))


def _members(members):
    if isinstance(members, (str, bytes)):
        raise ValueError('member list required')
    members = list(members)
    if not 1 <= len(members) <= 4 or any(not isinstance(m, str) or not m for m in members) \
            or len(set(members)) != len(members):
        raise ValueError('1..4 unique observable CAV ids required')
    return members


def wm_estimate(scorer, belief, members, move, rng, cfg=None, draw_id=None,
                *, horizon_s=None):
    """Score base, joint, solos and member reversions on paired WM draws.

    scorer(belief, intervention, rng_child) -> outcome (team_delay, lower better).
    Each intervention specifies active_members; all other CAVs use base MAPPO
    (HDVs are never controlled). 'alone' activates only member; 'revert' activates
    everyone in the group EXCEPT member. Scores must measure the SAME full group
    outcome, never just the active subset. Input and RNG copies isolate branches.

    The scorer owns the full registered look-ahead. If supplied, horizon_s is
    forwarded unchanged on EVERY branch; it is never clamped to the move's hold.
    No module-level WM state, global simulator state or truth inputs are used.
    """
    cfg = credit_config(cfg)
    assert_observable(belief)
    assert_observable(move)
    members = _members(members)
    move_members = move.get('members') if isinstance(move, Mapping) \
        else getattr(move, 'members', None)
    if move_members is not None and list(move_members) != members:
        raise ValueError('move members/order must match counterfactual group')
    if horizon_s is not None and (type(horizon_s) is not int or horizon_s <= 0):
        raise ValueError('full positive integer horizon_s required')
    interventions = [dict(kind='base', active_members=[]),
                     dict(kind='joint', active_members=members, move=move)]
    interventions += [dict(kind='alone', member=m, active_members=[m], move=move)
                      for m in members]
    interventions += [dict(kind='revert', member=m,
                           active_members=[x for x in members if x != m], move=move)
                      for m in members]
    for itv in interventions:
        itv.update(group_members=members, background='base_mappo', horizon_s=horizon_s)
    rows = [[] for _ in interventions]
    sample_seeds = []
    for _ in range(cfg['wm_samples']):
        seed = int(rng.integers(0, 2 ** 63 - 1))
        sample_seeds.append(seed)
        for result, itv in zip(rows, interventions):
            value = scorer(copy.deepcopy(belief), copy.deepcopy(itv),
                           np.random.default_rng(seed))
            result.append(_mean([value]))
    b, j, *rest = [_mean(row) for row in rows]
    a, reverted = rest[:len(members)], rest[len(members):]
    algebra = cr.algebra(j, reverted, b)
    alone_gain = [b - x for x in a]
    return dict(draw_id=draw_id, members=members, move=move,
                horizon_s=horizon_s, sample_seeds=sample_seeds,
                outcome=dict(base=b, joint=j, alone=a, reverted=reverted),
                credit=dict(group_credit=b - j, improvement=b - j, delay_delta=j - b,
                            **algebra),
                isolated=dict(delta_all_new_veh_s=b - j,
                              delta_only_member_new_veh_s=dict(zip(members, alone_gain)),
                              policy_cooperation_gain_veh_s=min(a) - j,
                              positive_policy_cooperation=bool(min(a) > j)))


def layer_advantage(estimate, cfg=None):
    """Broadcast group utility improvement as the LAYER-ONLY advantage.

    Pass per_member (member order preserved) to the cooperation layer PPO loss in
    place of per-CAV advantages. Do not add this to base rewards or update MAPPO.
    SUMO verification records cannot enter this path.
    """
    credit_config(cfg)
    members = _members(estimate['members'])
    team = _mean([estimate['credit']['group_credit']])
    return dict(members=members, per_member=[team] * len(members), team=team)


def verify_requests(estimates, rng, cfg=None):
    """Bernoulli(sumo_verify_fraction) selection -> requests for the existing
    all8/isolated SUMO verification machinery (consumed later; never run here)."""
    cfg = credit_config(cfg)
    out = []
    for e in estimates:
        if not bool(rng.random() < cfg['sumo_verify_fraction']):
            continue
        assert_observable(e['move'])
        out.append(dict(kind='group_credit_verification',
                         verify_mode=cfg['verify_mode'], isolated=cfg['isolated'],
                         draw_id=e['draw_id'], members=_members(e['members']),
                         move=copy.deepcopy(e['move']), wm=copy.deepcopy(e['credit']),
                         horizon_s=e['horizon_s'], sample_seeds=list(e['sample_seeds']),
                         background='base_mappo',
                         machinery=dict(all8='verify_perseed.confirm_all8',
                                        isolated='policy_synergy.specs_for')))
    return out


def verification_candidates(request, plan, snapshot):
    """Dry adapter to the existing sumo_verify/verify_perseed plan schema.

    The caller converts the move to its ordinary feasible plan first. Pass the
    returned candidates to sumo_verify.verify and then verify_perseed.confirm_all8
    (with the SAME candidates). Existing runners, budget/health guards and receipt
    claims remain responsible for execution. No new dispatch or charge path.

    Branch indices map base/joint/solos/reversions to a single matched-window
    verification, so rejected/harmful moves can be measured too. Use a frozen base
    MAPPO snapshot with no active cooperation interventions in the background.
    The plan's full duration/profile is preserved, not shortened for solo masks.
    Existing branch specs are built without execution, and a truncated/shorter
    verification horizon is refused before a caller can dispatch it.
    """
    if request.get('kind') != 'group_credit_verification' \
            or request.get('verify_mode') != 'all8_both_seeds' \
            or request.get('isolated') is not True \
            or request.get('background') != 'base_mappo':
        raise ValueError('isolated all8 group credit request required')
    members = _members(request['members'])
    assert_observable(plan)
    if list(plan['members']) != members:
        raise ValueError('plan members/order must match group credit request')
    if snapshot.get('committed_plans'):
        raise ValueError('isolated base MAPPO snapshot must have no active cooperation plans')
    candidates = []
    masks = {}

    def subset(selected):
        key = tuple(selected)
        if not key:
            return 0  # sumo_verify's policy-continues branch
        if key not in masks:
            out = copy.deepcopy(plan)
            indices = [members.index(m) for m in selected]
            out['members'] = list(selected)
            # Same member-matrix contract as replay_bank.solo_candidate.
            for k, v in plan.items():
                if isinstance(v, list) and len(v) == len(members) \
                        and all(isinstance(row, list) for row in v):
                    out[k] = [copy.deepcopy(v[i]) for i in indices]
            out['id'] = '%s__group_credit_%d' % (plan['id'], len(candidates) + 1)
            candidates.append(out)
            masks[key] = len(candidates)
        return masks[key]

    branches = dict(base=0, joint=subset(members))
    branches['alone'] = [subset([m]) for m in members]
    branches['reverted'] = [subset([x for x in members if x != m]) for m in members]
    from . import sumo_verify as sv
    specs = sv.build_branches(snapshot, candidates)
    horizon = request.get('horizon_s')
    window = specs[0]['window']
    if horizon is not None and (type(horizon) is not int or horizon <= 0
                               or window['end_s'] - window['start_s'] < horizon):
        raise ValueError('existing all8 window cannot shorten full WM look-ahead')
    return dict(candidates=candidates, branch_indices=branches, specs=specs)


def verified_stats(records):
    """WM vs SUMO-exact credit monitoring (CPU, offline only - never a PPO input,
    same rule as credit.calibration). Records need wm.credit and exact.credit."""
    if not records:
        raise ValueError('verification records required')
    pairs, diffs = [], []
    for r in records:
        x, y = _mean([r['wm']['D_G']]), _mean([r['exact']['D_G']])
        if x != 0 and y != 0:
            pairs.append((x, y))
        diffs.append(y - x)
    hits = sum((x > 0) == (y > 0) for x, y in pairs)
    return dict(n=len(records), paired=len(pairs), ties=len(records) - len(pairs),
                sign_agreement=hits / len(pairs) if pairs else None,
                mae_D_G=float(np.mean(np.abs(diffs))))
