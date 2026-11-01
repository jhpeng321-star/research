"""Numpy-only frozen MLP inference; no remote sklearn/pickle/version dependency."""
import numpy as np

from .data import KEYS, OPTIONS, encode


def predict(policy, x):
    columns = [np.zeros(len(x))]
    for m in policy['models']:
        h = (x-np.array(m['x_mean']))/np.array(m['x_scale'])
        for i,(w,b) in enumerate(zip(m['weights'],m['biases'])):
            h = h @ np.array(w)+np.array(b)
            if i < len(m['weights'])-1:
                h = np.tanh(h)
        columns.append(h.ravel()*m['y_scale']+m['y_mean'])
    return np.column_stack(columns)


def choose(policy, pred, available):
    best = np.where(available,pred,-np.inf).argmax(1)
    i = np.arange(len(pred))
    positive = pred[i,best] > 0
    probabilities = np.zeros_like(pred)
    for k in range(1,5):
        e = policy['errors'][k]
        if len(e):
            probabilities[:,k] = np.searchsorted(e,pred[:,k],side='left')/len(e)
    if policy['confidence'] > 0:
        positive &= probabilities[i,best] >= policy['confidence']
    return np.where(positive,best,0),probabilities


def export(payload):
    if payload['family'] != 'mlp':
        raise ValueError('portable export requires frozen MLP; no silent family substitution')
    result = dict(keys=list(KEYS),options=list(OPTIONS),confidence=payload['confidence'],
                  family='mlp',label_horizon_s=600,train_worlds=payload['train_worlds'],
                  errors=[e.tolist() for e in payload['errors']],models=[])
    for model in payload['models']:
        pipe = model.regressor_
        scale, mlp = pipe.steps[0][1], pipe.steps[1][1]
        result['models'].append(dict(x_mean=scale.mean_.tolist(),x_scale=scale.scale_.tolist(),
            weights=[w.tolist() for w in mlp.coefs_],biases=[b.tolist() for b in mlp.intercepts_],
            y_mean=float(model.transformer_.mean_[0]),y_scale=float(model.transformer_.scale_[0])))
    return result


class Student:
    def __init__(self, policy):
        self.policy = policy
        self.last_time = None
        if tuple(policy['keys']) != KEYS:
            raise ValueError('frozen causal feature schema mismatch')

    def step(self, state):
        from tools.coopplan.planner import candidates, TIMES
        if state.time_s not in TIMES or self.last_time is not None and state.time_s <= self.last_time:
            raise ValueError('strict60s cadence required')
        setting = ((state.time_s-300)//60) % 4
        actions = candidates(state)
        available = np.array([[o in {a['id'] for a in actions} for o in OPTIONS]])
        x = encode(state.rows,state.time_s,setting)[None]
        pred = predict(self.policy,x)
        chosen, probabilities = choose(self.policy,pred,available)
        action = next(a for a in actions if a['id'] == OPTIONS[int(chosen[0])])
        self.last_time = state.time_s
        return dict(time_s=state.time_s,replay_key=state.replay_key,candidates=actions,
                    available_candidates=actions,executed_action=action,group=list(action['members']),
                    initial_rows=list(state.rows),horizon_s=600,
                    decision=dict(selected=action['id'],predicted_mean_D=pred[0].tolist(),
                        probability_positive=probabilities[0].tolist(),confidence=self.policy['confidence'],
                        backend='DISTILL_OBSERVABLE_NUMPY'))
