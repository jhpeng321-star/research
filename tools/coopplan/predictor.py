"""Public, observable-only prediction contract; simulator replay stays backend-private."""
from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Mapping, Protocol, Sequence

from tools.coopbandit.protocol import digest


@dataclass(frozen=True)
class ObservableState:
    time_s: int
    rows: tuple
    replay_key: str  # opaque identity, never a feature or a hidden-state payload
    commands: Mapping = field(default_factory=dict)
    history: tuple = ()

    def __post_init__(self):
        if type(self.time_s) is not int or self.time_s < 0 or not self.replay_key:
            raise ValueError('state needs an integer clock and opaque replay identity')
        allowed = {'vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m'}
        seen = set()
        for row in self.rows:
            if set(row)-allowed or not {'vid','s_m','v_mps','lane','is_cav'} <= row.keys():
                raise ValueError('observable row whitelist')
            if row['vid'] in seen or not all(math.isfinite(row[k]) for k in ('s_m','v_mps')):
                raise ValueError('duplicate identity or nonfinite observation')
            seen.add(row['vid'])


@dataclass(frozen=True)
class Outcome:
    draw_id: str
    team_delay: float
    breakdown: bool | None
    member_delay: Mapping = field(default_factory=dict)
    safety: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    actions: Mapping[str, Sequence[Outcome]]
    exact: bool
    backend: str

    def validate(self, candidates, K):
        if set(self.actions) != {a['id'] for a in candidates} or K < 1:
            raise ValueError('prediction action coverage')
        common = None
        for futures in self.actions.values():
            ids = [f.draw_id for f in futures]
            if len(ids) != K or len(set(ids)) != K or not all(ids):
                raise ValueError('missing or duplicate paired draws')
            if common is not None and set(ids) != common:
                raise ValueError('CRN draw identities differ across actions')
            common = set(ids)
            for f in futures:
                if not math.isfinite(f.team_delay) or f.team_delay < 0:
                    raise ValueError('invalid delay; failed futures cannot be dropped')
                if f.breakdown is not None and type(f.breakdown) is not bool:
                    raise ValueError('unscorable breakdown must be None')
                if not all(math.isfinite(v) and v >= 0 for v in f.member_delay.values()):
                    raise ValueError('invalid member delay')
                if not all(math.isfinite(v) and v >= 0 for v in f.safety.values()):
                    raise ValueError('invalid safety outcome')
        if self.exact and K != 1:
            raise ValueError('deterministic oracle requires K=1')
        return self


class Predictor(Protocol):
    def predict(self, observable_state: ObservableState, candidate_actions: Sequence[Mapping],
                horizon: int, K: int) -> Prediction: ...


def validate_request(state, actions, horizon, K):
    if type(horizon) is not int or not (60 <= horizon <= 120 or horizon == 300) or type(K) is not int or K < 1:
        raise ValueError('registered horizon: integer H60..120 or MAIN02:21 H300, positive K')
    if not actions or len({a['id'] for a in actions}) != len(actions):
        raise ValueError('unique nonempty action IDs required')
    cavs = {r['vid'] for r in state.rows if r['is_cav']}
    for a in actions:
        members = a['members']
        if len(set(members)) != len(members) or not set(members) <= cavs:
            raise ValueError('only observed CAV members may be commanded')


class SumoOraclePredictor:
    """run_full_rerun must replay ALL previous executed decisions, then gate t0.

    It returns horizon-only metrics after a bitwise physical/controller-prefix
    match. Full state, world identity and demand stay in the runner's private store.
    """
    def __init__(self, run_full_rerun):
        self.run = run_full_rerun

    def predict(self, observable_state, candidate_actions, horizon, K):
        validate_request(observable_state, candidate_actions, horizon, K)
        if K != 1:
            raise ValueError('SUMO is deterministic for the registered full-state continuation: K=1')
        draw = digest(dict(replay=observable_state.replay_key, horizon=horizon))
        outcomes = {}
        for action in candidate_actions:
            result = self.run(observable_state, deepcopy(action), horizon)
            outcomes[action['id']] = (Outcome(draw, **result),)
        return Prediction(outcomes, True, 'SUMO-ORACLE').validate(candidate_actions, K)


class BackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class CompleteDraw:
    """Identity-bound arbitrary WM payload (including tensor-backed future builders)."""
    draw_id: str
    payload: object


class WMPhysPredictor:
    """Rebuilt WM bridge. One complete hidden-driver/arrival draw per k, across actions.

    draw_factory(state,H,K) and rollout(state,action,H,draw) are supplied by the
    identity-frozen RB-INT runtime. Never manufacture independent action-local RNGs.
    A missing runtime is explicit, rather than silently substituting the oracle.
    """
    def __init__(self, rollout=None, draw_factory=None):
        self.rollout, self.draw_factory = rollout, draw_factory

    def predict(self, observable_state, candidate_actions, horizon, K):
        validate_request(observable_state, candidate_actions, horizon, K)
        if self.rollout is None or self.draw_factory is None:
            raise BackendUnavailable('bind RB-INT WM-PHYS rollout and complete CRN future builder')
        draws = list(self.draw_factory(observable_state, horizon, K))
        if len(draws) != K:
            raise ValueError('future builder must return exactly K complete draws')
        bound_ids = [d.draw_id for d in draws if isinstance(d,CompleteDraw)]
        if any(not isinstance(d,str) or not d for d in bound_ids) or len(set(bound_ids)) != len(bound_ids):
            raise ValueError('complete futures require unique nonempty bound draw identities')
        identities = [digest(dict(replay=observable_state.replay_key, horizon=horizon,
                                  ordinal=i, payload=d.draw_id if isinstance(d,CompleteDraw) else d))
                      for i,d in enumerate(draws)]
        draws = [d.payload if isinstance(d,CompleteDraw) else d for d in draws]
        outcomes = {}
        for action in candidate_actions:
            outcomes[action['id']] = tuple(Outcome(identity, **self.rollout(
                observable_state, deepcopy(action), horizon, deepcopy(draw)))
                for identity,draw in zip(identities,draws))
        return Prediction(outcomes, False, 'WM-PHYS').validate(candidate_actions, K)
