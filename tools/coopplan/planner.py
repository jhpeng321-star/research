"""Same receding-horizon planner for exact SUMO and sampled world-model futures."""
from copy import deepcopy
from dataclasses import dataclass
from statistics import fmean

from tools.coopbandit.protocol import option, digest
from .predictor import validate_request

CONFIDENCE = .8  # Owner's decision rule; not a fitted behavioural threshold.
CADENCE = 60
HORIZON = 120
TIMES = tuple(range(300, 961, CADENCE))


def candidates(state, learned=None):
    # Fixed, outcome-independent rotation through the inherited intervention grid.
    setting = ((state.time_s-300)//CADENCE) % 4
    actions = [option('O0', state.rows, setting)]
    for name in ('O1','O2','O3','O4'):
        a = option(name, state.rows, setting)
        if a['available']:
            actions.append(a)
    if learned is not None:
        a = learned(state)
        if a is not None:
            if a.get('id') != 'FREE' or len(a['members']) != 2 or a.get('window') not in (10,20,30):
                raise ValueError('one frozen COOPFREE two-member proposal in its original window set')
            actions.append(a)
    validate_request(state, actions, HORIZON, 1)
    return actions


def decide(prediction, incumbent='O0'):
    actions = prediction.actions
    # Validation here also protects direct users, not only Predictor implementations.
    prediction.validate([dict(id=a,members=[]) for a in actions], len(actions[incumbent]))
    base = {f.draw_id:f for f in actions[incumbent]}
    means = {a:fmean(f.team_delay for f in fs) for a,fs in actions.items()}
    probabilities, effects, eligible = {}, {}, []
    for a, fs in actions.items():
        probabilities[a] = sum(f.team_delay < base[f.draw_id].team_delay for f in fs)/len(fs)
        members = set.intersection(*(set(f.member_delay) & set(base[f.draw_id].member_delay) for f in fs))
        effects[a] = {m:fmean(base[f.draw_id].member_delay[m]-f.member_delay[m] for f in fs)
                      for m in sorted(members)}
        if a != incumbent and probabilities[a] >= CONFIDENCE and means[a] < means[incumbent]:
            eligible.append(a)
    selected = min(eligible,key=lambda a:(means[a],a)) if eligible else incumbent
    return dict(selected=selected, probability_beats=probabilities, expected_delay=means,
                member_savings=effects, eligible=sorted(eligible), confidence=CONFIDENCE,
                semantics='strict_paired_delay_win', exact=prediction.exact, backend=prediction.backend)


def wmdecide_check(prediction):
    """Optional S-RB-DECIDE parity check using its published Future/decide API."""
    from tools.wmdecide.decision import Future, decide as registered_decide
    return registered_decide({a:[Future(f.draw_id,f.team_delay,f.breakdown) for f in fs]
                              for a,fs in prediction.actions.items()}, 'O0')


@dataclass
class Planner:
    predictor: object
    K: int = 1
    learned: object = None
    log: object = None
    last_time: int | None = None
    admit: object = None
    horizon: int = HORIZON

    def step(self, state):
        if state.time_s not in TIMES or (self.last_time is not None and state.time_s <= self.last_time):
            raise ValueError('registered cadence; decisions must be strictly increasing')
        actions = candidates(state,self.learned)
        available = deepcopy(actions)
        admission = None
        if self.admit is not None:
            actions,admission = self.admit(state,actions)
            if not actions:
                chosen = option('O0',state.rows,0)
                event = dict(time_s=state.time_s,replay_key=state.replay_key,candidates=[chosen],
                             group=[],executed_action=chosen,admission=admission,prediction_id=None,
                             available_candidates=available,horizon_s=self.horizon,
                             decision=dict(selected='O0',eligible=[],expected_delay={},probability_beats={'O0':0.},
                                           exact=True,backend='SUMO-ORACLE',confidence=CONFIDENCE))
                self.last_time = state.time_s
                return event,None
        prediction = self.predictor.predict(state,actions,self.horizon,self.K)
        decision = decide(prediction)
        chosen = next(a for a in actions if a['id'] == decision['selected'])
        event = dict(time_s=state.time_s, replay_key=state.replay_key, candidates=actions,
                     group=list(chosen['members']), executed_action=deepcopy(chosen),
                     available_candidates=available,horizon_s=self.horizon,
                     decision=decision, prediction_id=digest({a:[vars(f) for f in fs]
                                                            for a,fs in prediction.actions.items()}))
        if self.log is not None:
            self.log(event)
        if admission is not None:
            event['admission'] = admission
        self.last_time = state.time_s
        return event, prediction


def cooperation_credit(prediction, group_id, solos):
    """C_G = (J0-JG) - sum_i(J0-J_i); positive is superadditive saving."""
    if len(solos) != 2 or len(set(solos)) != 2:
        raise ValueError('two distinct member-alone counterfactuals required')
    prediction.validate([dict(id=a,members=[]) for a in prediction.actions],
                        len(prediction.actions['O0']))
    rows = {a:{f.draw_id:f.team_delay for f in fs} for a,fs in prediction.actions.items()}
    return {d:(rows['O0'][d]-rows[group_id][d])-
            sum(rows['O0'][d]-rows[a][d] for a in solos) for d in rows['O0']}
