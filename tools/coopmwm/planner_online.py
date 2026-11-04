"""Belief-only, calibrated online move planning (explicit opt-in).

``score_moves`` is also the dense teacher's scoring boundary. ``rollout`` has the
existing mppi_runtime evaluator ABI: (record, actions, K, H) -> paired costs.
The production adapter below uses plan_wm.evaluate / policy_wm, not a second WM.
Calibration is mandatory: planner_calibration.calibrate_probability(p_raw,
calibration) -> float in [0, 1]. A caller may pass that API as ``calibrate``;
there is deliberately no identity/raw-probability fallback.
"""
from dataclasses import dataclass
import copy
import math
from collections.abc import Mapping

from .warm_history import sensor_rows, validate_history


@dataclass(frozen=True)
class Config:
    cadence_s: int = 60
    n_candidates: int = 32  # mppi_search.REGISTERED pop_min
    wm_samples: int = 8  # mppi_search.REGISTERED final_K
    horizon_s: int = 600  # full inherited look-ahead, never a move-length screen
    p_star: float = .8

    def __post_init__(self):
        for name, minimum in (('cadence_s', 1), ('n_candidates', 0),
                              ('wm_samples', 2), ('horizon_s', 600)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if not math.isfinite(self.p_star) or not 0 <= self.p_star <= 1:
            raise ValueError('p_star must be a probability')


def _field(move, key):
    return move[key] if isinstance(move, Mapping) else getattr(move, key)


def validate_observation(record):
    """Reject extra truth fields and observations outside the ego's radio/sensor view."""
    from .v2v_obs import V2VConfig
    allowed = {'ego_id', 'group', 'rows', 'time_s', 'world_id', 'state_id',
               'commands', 'policy_residuals', 'policy_state', 'history'}
    if set(record) - allowed:
        raise ValueError('planner observable record whitelist violation')
    required = {'ego_id', 'group', 'rows', 'time_s', 'commands',
                'policy_residuals', 'policy_state'}
    if not required <= set(record):
        raise ValueError('incomplete observable policy belief')
    rows = sensor_rows(record['rows'])
    by_id = {r['vid']: r for r in rows}
    ego = by_id.get(record['ego_id'])
    if ego is None or not ego['is_cav']:
        raise ValueError('planner ego must be an observed CAV')
    radius = V2VConfig().range_m
    if any(abs(r['s_m']-ego['s_m']) > radius for r in rows):
        raise ValueError('planner rows exceed observable sensing/V2V range')
    group = record['group']
    cavs = {r['vid'] for r in rows if r['is_cav']}
    if (not 1 <= len(group) <= 4 or len(set(group)) != len(group)
            or record['ego_id'] not in group or not set(group) <= cavs):
        raise ValueError('planner group must contain ego and 1..4 observable CAVs')
    now = record['time_s']
    if not math.isfinite(now) or now < 0:
        raise ValueError('invalid observation clock')
    for key in ('commands', 'policy_residuals'):
        if not set(record[key]) <= cavs or any(not math.isfinite(v) for v in record[key].values()):
            raise ValueError('only finite observed CAV commands/residuals are allowed')
    if not set(group) <= set(record['commands']):
        raise ValueError('group lacks base commands')
    if not isinstance(record['policy_state'], Mapping):
        raise ValueError('dynamic base-policy actor state required')
    history = validate_history(record.get('history', []), now)
    for frame in history:
        past = {r['vid']: r for r in frame['rows']}
        own = past.get(record['ego_id'])
        if own is None or any(abs(r['s_m']-own['s_m']) > radius for r in past.values()):
            raise ValueError('history exceeds the ego observable view')
    if history and history[-1]['rows'] != rows:
        raise ValueError('history must end at the current observation')
    return record


def _profile(move, move_api, record, horizon_s):
    members = tuple(_field(move, 'members'))
    if (not members or len(set(members)) != len(members)
            or not set(members) <= set(record['group'])
            or record['ego_id'] not in members):
        raise ValueError('move members must be observable group CAVs including ego')
    if _field(move, 'family') not in move_api.FAMILIES:
        raise ValueError('unregistered move family')
    profile = []
    for t in range(horizon_s):
        values = dict(move_api.offsets(move, t))
        if not set(values) <= set(members) or any(not math.isfinite(v) for v in values.values()):
            raise ValueError('invalid move command offsets')
        profile.append({v: float(x) for v, x in values.items() if x != 0.})
    return profile


def _costs(values, samples):
    if len(values) != samples:
        raise ValueError('WM must return every paired sample for the full horizon')
    costs = [float(x['team_delay'] if isinstance(x, Mapping) else x) for x in values]
    if any(not math.isfinite(x) for x in costs):
        raise ValueError('nonfinite WM cost')
    return costs


def score_moves(record, rng, *, rollout, calibration, config=None, calibrate=None,
                move_api=None):
    """Score K moves + none on common WM futures, then gate calibrated P(gain>0).

    All actions use the current MAPPO/base actor for the *entire* horizon. Offsets
    are residuals, not replacement accelerations. Among positive-gain candidates
    meeting the calibrated gate, execute the greatest expected gain (stable ties).
    Missing/invalid calibration or incomplete WM samples raise, never open a gate.
    """
    config = config or Config()
    validate_observation(record)
    if calibration is None:
        raise ValueError('planner calibration is required')
    if calibrate is None:
        from .planner_calibration import calibrate_probability
        calibrate = calibrate_probability
    if move_api is None:
        from . import moves as move_api
    candidates = list(move_api.candidates(copy.deepcopy(record), rng, config.n_candidates))
    if len(candidates) > config.n_candidates:
        raise ValueError('move generator exceeded candidate budget')
    # "none" continues the SAME current actor, including when it has residuals.
    actions = [dict(id='policy', residuals={}, window=0)]
    actual = []
    for move in candidates:
        if _field(move, 'family') == 'none':
            continue  # one explicit, separately rolled comparator only
        profile = _profile(move, move_api, record, config.horizon_s)
        action = dict(id=f'move:{len(actual)}', offset_profile=profile)
        actions.append(action)
        actual.append(move)
    futures = rollout(copy.deepcopy(record), actions, config.wm_samples, config.horizon_s)
    if set(futures) != {a['id'] for a in actions}:
        raise ValueError('incomplete WM action results')
    base = _costs(futures['policy'], config.wm_samples)
    scores = []
    for move, action in zip(actual, actions[1:]):
        costs = _costs(futures[action['id']], config.wm_samples)
        gains = [a-b for a, b in zip(base, costs)]
        p_raw = sum(x > 0 for x in gains)/config.wm_samples
        p_calibrated = float(calibrate(p_raw, calibration))
        if not math.isfinite(p_calibrated) or not 0 <= p_calibrated <= 1:
            raise ValueError('calibration API returned an invalid probability')
        gain = sum(gains)/config.wm_samples
        scores.append(dict(id=action['id'], family=_field(move, 'family'),
                           members=list(_field(move, 'members')), p_raw=p_raw,
                           p_calibrated=p_calibrated, expected_gain=gain,
                           paired_gains=gains, eligible=p_calibrated >= config.p_star and gain > 0))
    eligible = [i for i, s in enumerate(scores) if s['eligible']]
    best = max(eligible, key=lambda i: (scores[i]['expected_gain'], -i)) if eligible else None
    return dict(accepted=best is not None, selected_move=actual[best] if best is not None else None,
                selected_id=scores[best]['id'] if best is not None else 'policy',
                offset_profile=actions[best+1]['offset_profile'] if best is not None else [],
                expected_gain=scores[best]['expected_gain'] if best is not None else 0.,
                scores=scores, wm_samples=config.wm_samples, horizon_s=config.horizon_s,
                p_star=config.p_star, comparator='current_base_policy',
                reason='calibrated_gate' if best is not None else 'base_policy')


class WMRollout:
    """CPU-only adapter to the already-qualified plan_wm / policy_wm rollout."""
    def __init__(self, runtime, memory=None, chunk_actions=4):
        if type(chunk_actions) is not int or chunk_actions < 1:
            raise ValueError('positive action chunk width required')
        for model in (runtime.model, runtime.mp.policy):
            if any(p.device.type != 'cpu' for p in model.parameters()):
                raise ValueError('online planner requires a CPU WM/runtime')
        self.runtime, self.memory, self.chunk_actions = runtime, memory, chunk_actions

    def __call__(self, record, actions, K, H):
        from .plan_wm import evaluate
        return evaluate(self.runtime, record, actions, K, H, trim=False,
                        chunk_actions=self.chunk_actions, memory=self.memory, final_chain=True)
