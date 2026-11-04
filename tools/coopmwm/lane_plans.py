"""Plan lane actions (c4/lane-plans, owner 2026-11-04).

Extends the group plan space from per-CAV acceleration sequences to
(acceleration, lane action keep/left/right) per second. The option is OFF
everywhere by default: plans carry no ``lane_*`` key and every output of the
search, the verifier, the bank and the imitation loss stays byte-identical
(the option key is popped from the config before anything else runs).

Lane encoding = ``wms.control.lanes`` (owner 2026-10-12): 0 keep, 1 left
(lane-1; WM lane 0 is the LEFT mainline lane), 2 right (lane+1). The only road
transitions the physics ever allows are lane 1 -> 0 (overtake), 0 -> 1 (return)
and ramp lane 2 -> 1 inside the merge zone (merge); 1 -> 2 is never allowed, so
a member has at most one feasible non-keep direction.

Physical envelope (no invented rules): the decision-time feasibility is exactly
``wms.control.lanes.lane_feasibility`` - the same SUMO-calibrated lane-change
safety model (LANECAL_NOTES) the WM imagination and the deployment executor
use: mainline 0<->1 only, ramp 2->1 inside the merge zone, the target-lane
gap rule, the dwell clock (unresolvable dwell = allowed, like a fresh CAV) and
the edge guard. Lane actions are SAMPLED only inside that envelope and are
ALWAYS re-judged by the same model when executed (WM rollout / SUMO
changeLane path); the envelope at t0 never overrides a later rejection.

Outcome prediction uses the learned WM lane model, in priority order:
1. ``wms.physics.lane_learnable`` (the c4/lane-policy lane model interface,
   used automatically once that branch is merged);
2. the lane_v5 network (``wms.model.lane_v5.LaneV5Net``) from the pre-fit
   checkpoint at ``lane_model`` config / ``COOPMWM_LANE_MODEL``;
3. an uncalibrated LaneV5Net with the registered head-bias prior
   (predictions are recorded on the plan as ``lane_predictions`` and are never
   a safety or sampling threshold).

SUMO execution (verification branches and live) goes through the same safe
changeLane path as c4/lane-policy. This module only emits per-second lane
requests (``Controller`` records them on the sample) and records per-request
outcomes {requested, issued, blocked, executed} through the bound executor
(``bind_executor`` - the merged lane-policy interface registers itself here;
until then requests are recorded and counted as ``unbound``, never claimed as
executed).

Registered exploration constants (search hyperparameters, NOT safety rules):
``lane_rate`` = probability that one member of a sampled plan carries one
lane-change attempt (mirrors the registered elite fraction scale), the attempt
second drawn uniformly over the plan, at most one attempt per member per plan
(exploration.ExPLORATION std 0.5 m/s^2 is the acceleration analogue).
"""
import math
import os

import numpy as np

from .mppi_search import _digest

LANE_KEEP, LANE_LEFT, LANE_RIGHT = 0, 1, 2   # wms.control.lanes LANE_STAY/LEFT/RIGHT
LANE_ACTIONS = (LANE_KEEP, LANE_LEFT, LANE_RIGHT)
LANE_NAMES = ('keep', 'left', 'right')

# Registered option/keys (default OFF). Unknown lane_* keys raise.
REGISTERED = dict(lane_actions=False, lane_rate=.25, lane_model=None)
MAX_ATTEMPTS_PER_MEMBER = 1
DEFAULT_EXECUTOR = None    # bound by bind_executor (c4/lane-policy safe changeLane path)


def split_config(cfg):
    """Pop every ``lane_*`` key from ``cfg`` (in place). None (option off) keeps the
    output byte-identical; a truthy ``lane_actions`` returns the merged lane config."""
    raw = {k: cfg.pop(k) for k in [k for k in cfg if k.startswith('lane_')]}
    unknown = set(raw)-set(REGISTERED)
    if unknown:
        raise ValueError(f'unregistered lane config keys {sorted(unknown)}')
    if not raw.get('lane_actions'):
        return None
    lane = dict(REGISTERED, **raw)
    rate = float(lane['lane_rate'])
    if not 0. <= rate <= 1. or not math.isfinite(rate):
        raise ValueError('lane_rate must be a finite probability')
    if lane['lane_model'] is not None and not isinstance(lane['lane_model'], str):
        raise ValueError('lane_model must be a checkpoint path or None')
    cfg.update({k: v for k, v in lane.items()})
    return lane


# --------------------------------------------------------------------------- envelope

def decision_envelope(record, members):
    """``{vid: {lane, left, right}}`` from ``wms.control.lanes.lane_feasibility`` on the
    recorded t0 state (all sensed vehicles - target-lane gaps need them), or None when the
    state/WM lane model cannot be resolved (degrade, never guess: no lane actions)."""
    rows = (record or {}).get('rows') or []
    by = {r['vid']: r for r in rows}
    if not rows or any(v not in by for v in members):
        return None
    if any(not isinstance(r.get(k), (int, float)) for r in rows for k in ('s_m', 'v_mps', 'lane')) \
            or any(not isinstance(r.get('is_cav'), bool) for r in rows):
        return None
    try:
        import torch
        from wms.control.lanes import LaneRules, lane_feasibility
        from wms.core.constants import DEFAULT_LENGTH_M
    except Exception:
        return None
    k = len(rows)
    state = dict(
        s_m=torch.tensor([[float(r['s_m']) for r in rows]], dtype=torch.float32),
        v_mps=torch.tensor([[float(r['v_mps']) for r in rows]], dtype=torch.float32),
        lane=torch.tensor([[int(r['lane']) for r in rows]], dtype=torch.long),
        present=torch.ones((1, k), dtype=torch.bool),
        is_cav=torch.tensor([[bool(r['is_cav']) for r in rows]]),
        length_m=torch.tensor([[float(r.get('length_m') or DEFAULT_LENGTH_M) for r in rows]],
                              dtype=torch.float32))
    try:
        feasible = lane_feasibility(state, None, LaneRules())[0]
    except Exception:
        return None
    out = {}
    for v in members:
        f = feasible[rows.index(by[v])]
        out[v] = dict(lane=int(by[v]['lane']), left=bool(f[LANE_LEFT]), right=bool(f[LANE_RIGHT]))
    return out


def _kinematics(rows):
    """lane_v5 rollout kinematics ``[1, K, 7]`` of the recorded state (see state.state_kin)."""
    import torch
    from wms.core.constants import DEFAULT_LENGTH_M
    k = len(rows)
    lane = torch.tensor([[int(r['lane']) for r in rows]], dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(lane.clamp_min(0), 3).to(torch.float32) \
        * (lane >= 0).unsqueeze(-1).to(torch.float32)
    return torch.cat((torch.tensor([[float(r['s_m']) for r in rows]], dtype=torch.float32).unsqueeze(-1),
                      one_hot,
                      torch.tensor([[float(r['v_mps']) for r in rows]], dtype=torch.float32).unsqueeze(-1),
                      torch.tensor([[float(r.get('a_mps2') or 0.) for r in rows]],
                                   dtype=torch.float32).unsqueeze(-1),
                      torch.tensor([[float(r.get('length_m') or DEFAULT_LENGTH_M) for r in rows]],
                                   dtype=torch.float32).unsqueeze(-1)), -1)


_MODEL_CACHE = {}


def lane_model(path=None):
    """The learned WM lane model: ``wms.physics.lane_learnable`` when the c4/lane-policy
    interface is merged, else the lane_v5 network (pre-fit ``path``, else the registered
    head-bias prior). Returns an object with ``.logits(rows) -> [K, 3]`` or None; built once
    per path per process."""
    key = path or os.environ.get('COOPMWM_LANE_MODEL') or ''
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    try:
        from wms.physics import lane_learnable
        make = getattr(lane_learnable, 'model', None)
        if make is not None:
            obj = make(path)
            _MODEL_CACHE[key] = obj
            return obj
    except Exception:
        pass
    try:
        import torch
        from wms.model.lane_v5 import LaneV5Net, load_pretrain
    except Exception:
        return None
    net = LaneV5Net().eval()
    if path:
        load_pretrain(net, path)
    obj = _LaneV5(net)
    _MODEL_CACHE[key] = obj
    return obj


def resolve_model(lane):
    """The lane config's model (config path wins, then ``COOPMWM_LANE_MODEL``)."""
    return lane_model((lane or {}).get('lane_model'))


class _LaneV5:
    """lane_v5 logits of the t0 state (features per LaneV5: step_features, registered priors)."""

    def __init__(self, net):
        self.net = net

    def logits(self, rows):
        import torch
        from wms.model.lane_v5 import ACC_F, step_features
        from wms.physics.config import PhysicsConfig
        kin = _kinematics(rows)
        present = torch.ones(kin.shape[:2], dtype=torch.bool)
        acc0 = torch.zeros((*kin.shape[:2], ACC_F))
        feats, _, _ = step_features(acc0, kin, present, PhysicsConfig(), None)
        with torch.no_grad():
            return self.net(feats)[0]


def predict(record, members, actions, model=None):
    """Outcome prediction of every sampled lane attempt via the learned WM lane model.

    Returns a list of {member, second_s, action, head, model_logit, model_prob}; the model
    fields are None (source 'envelope') when no lane model resolves. Never a gate: the
    physical envelope decided the support, the WM rollout / SUMO decide the outcome."""
    rows = (record or {}).get('rows') or []
    by = {r['vid']: r for r in rows}
    if model is None:
        model = lane_model(os.environ.get('COOPMWM_LANE_MODEL'))
    logits, heads = None, None
    if model is not None and rows:
        try:
            from wms.model.lane_v5 import HEAD_OF_LANE, TRANSITIONS
            heads = (HEAD_OF_LANE, TRANSITIONS)
            logits = model.logits(rows)
        except Exception:
            logits, heads = None, None
    out = []
    for i, vid in enumerate(members):
        row = actions[i]
        for t, action in enumerate(row):
            if not action:
                continue
            entry = dict(member=vid, second_s=int(t), action=int(action))
            if logits is not None:
                head = heads[0][int(by[vid]['lane'])]
                logit = float(logits[rows.index(by[vid])][head])
                entry.update(head=heads[1][head], model_logit=logit,
                             model_prob=float(1./(1.+math.exp(-max(-30., min(30., logit))))))
            else:
                entry.update(head=None, model_logit=None, model_prob=None)
            out.append(entry)
    return out


# --------------------------------------------------------------------------- sampling

def sample_lane_actions(envelope, members, duration, rng, rate=REGISTERED['lane_rate']):
    """Per member at most one lane-change attempt inside the decision-time envelope.

    ``rng.random() < rate`` draws the attempt, the second is uniform over the plan, the
    direction is the member's envelope-feasible one (at most one exists; both would be
    chosen uniformly). Returns ``members x duration`` ints."""
    actions = [[0]*int(duration) for _ in members]
    for i, v in enumerate(members):
        e = envelope.get(v) if envelope else None
        if e is None:
            continue
        allowed = [d for d, ok in ((LANE_LEFT, e['left']), (LANE_RIGHT, e['right'])) if ok]
        if not allowed or float(rng.random()) >= float(rate):
            continue
        action = allowed[int(rng.integers(0, len(allowed)))] if len(allowed) > 1 else allowed[0]
        actions[i][int(rng.integers(0, int(duration)))] = action
    return actions


def attach(plan, actions, envelope, predictions):
    """Attach lane fields to a built plan and re-digest its id (build_plan's scheme)."""
    plan['lane_actions'] = [[int(x) for x in row] for row in actions]
    plan['lane_envelope'] = dict(envelope or {})
    plan['lane_predictions'] = list(predictions or [])
    plan['uses_lane_actions'] = any(any(row) for row in plan['lane_actions'])
    plan['id'] = plan['id'][:-17]+'M'+_digest(plan)[:16]
    return plan


def validate(plan):
    """Structural lane fields of a plan (shape, values, one attempt per member, the direction
    inside the plan's recorded envelope). No safety judgement: execution re-checks."""
    actions = plan.get('lane_actions')
    if actions is None:
        if plan.get('uses_lane_actions'):
            raise ValueError('uses_lane_actions tag without lane_actions')
        return plan
    m, d = len(plan['members']), int(plan['duration_s'])
    a = np.asarray(actions, dtype=int)
    if a.shape != (m, d) or not np.isin(a, LANE_ACTIONS).all():
        raise ValueError('lane actions must be members x duration_s in {0,1,2}')
    for i, row in enumerate(a):
        if int((row != LANE_KEEP).sum()) > MAX_ATTEMPTS_PER_MEMBER:
            raise ValueError('more than the registered lane-change attempts per member')
    env = plan.get('lane_envelope') or {}
    for i, vid in enumerate(plan['members']):
        e = env.get(vid)
        if e is None:
            continue
        if a[i][a[i] == LANE_LEFT].size and not e.get('left'):
            raise ValueError(f'lane action outside the WM envelope: {vid} left')
        if a[i][a[i] == LANE_RIGHT].size and not e.get('right'):
            raise ValueError(f'lane action outside the WM envelope: {vid} right')
    if bool(plan.get('uses_lane_actions')) != bool((a != LANE_KEEP).any()):
        raise ValueError('uses_lane_actions tag disagrees with lane_actions')
    return plan


def at(plan, elapsed):
    """{vid: lane action} requested at this plan second; {} outside [0, D)."""
    t = int(math.floor(elapsed))
    if not 0 <= t < int(plan['duration_s']):
        return {}
    return {v: int(row[t]) for v, row in zip(plan['members'], plan['lane_actions']) if row[t]}


def requests_for(plan, elapsed):
    return at(plan, elapsed)


def solo_lane_fields(plan, member):
    """coop_search.solo_plan / replay_bank.solo_candidate fields for one member's lane row."""
    i = plan['members'].index(member)
    row = plan['lane_actions'][i]
    env = {member: plan['lane_envelope'][member]} if plan.get('lane_envelope', {}).get(member) else {}
    preds = [p for p in plan.get('lane_predictions') or [] if p.get('member') == member]
    return dict(lane_actions=[row], lane_envelope=env, lane_predictions=preds,
                uses_lane_actions=bool(any(row)))


# --------------------------------------------------------------------------- execution

def bind_executor(fn):
    """Register the safe changeLane executor (the merged c4/lane-policy interface calls this).

    ``fn(rows, requests, now) -> {vid: {requested, issued, blocked, executed}}``; None unbinds."""
    global DEFAULT_EXECUTOR
    DEFAULT_EXECUTOR = fn


def default_executor():
    return DEFAULT_EXECUTOR


def record_events(requests, outcomes):
    """Normalise executor output into the recorded per-request record (issued/blocked/executed)."""
    out = {}
    for vid, action in (requests or {}).items():
        o = (outcomes or {}).get(vid) or {}
        issued, blocked, executed = bool(o.get('issued')), o.get('blocked'), bool(o.get('executed'))
        if not issued and blocked is None and not executed:
            blocked = 'unbound'   # recorded, never claimed as executed
        out[vid] = dict(requested=int(action), issued=issued, blocked=blocked, executed=executed)
    return out


def outcome_counts(events):
    """{requests, issued, blocked, executed} of recorded lane events (audit summary)."""
    counts = dict(requests=len(events or {}), issued=0, blocked=0, executed=0)
    for e in (events or {}).values():
        counts['issued'] += bool(e.get('issued'))
        counts['blocked'] += bool(e.get('blocked'))
        counts['executed'] += bool(e.get('executed'))
    return counts


# --------------------------------------------------------------------------- imitation

def pair_lane_target(entry, members, second):
    """The plan's lane actions of one pair second: {vid: action} (targets may be keep)."""
    actions = entry.get('lane_actions')
    if actions is None:
        return None
    return {v: int(row[second]) for v, row in zip(members, actions)}


def intent_lane_targets(entry, members, second, horizon, end_s):
    """([H] future plan lane actions, [H] mask) for the intent lane channel (targets include
    lane): the plan's own lane actions from ``second``+1 on, 0 (keep) after the plan end."""
    actions = entry.get('lane_actions')
    if actions is None:
        return None, None
    end = int(end_s)
    targets = [[0]*int(horizon) for _ in members]
    mask = [[0]*int(horizon) for _ in members]
    for k in range(1, int(horizon)+1):
        t = int(second)+k
        if t >= end:
            break
        for j in range(len(members)):
            targets[j][k-1] = int(actions[j][t])
            mask[j][k-1] = 1.
    return targets, mask


def policy_lane_ce(policy, batch):
    """AWR-weighted cross-entropy of the policy's lane head on bank lane targets.

    Batch: (sample, weight) pairs; a pair trains when it carries ``lane_target`` and the
    policy exposes the c4/lane-policy lane head contract ``policy.lane_logits(context)``
    -> [N, 3] member logits. Per-pair term = weight x mean member CE; batch term = mean
    over trained pairs. None (inert) without a lane head or without lane targets, so the
    legacy imitation loss and its report stay byte-identical."""
    rows = [(s, w) for s, w in batch if s.get('lane_target')]
    if not rows or not hasattr(policy, 'lane_logits'):
        return None
    import torch
    terms = []
    for sample, weight in rows:
        members = list(sample['context']['members'])
        logits = policy.lane_logits(sample['context'])
        slot = {v: i for i, v in enumerate(members)}
        ce = 0.
        for vid, action in sample['lane_target'].items():
            j = slot.get(vid)
            if j is None:
                continue
            ce = ce+float(torch.nn.functional.cross_entropy(
                logits[j:j+1], torch.tensor([int(action)], dtype=torch.long)))
        terms.append(float(weight)*ce/max(1, len(sample['lane_target'])))
    return sum(terms)/len(terms)
