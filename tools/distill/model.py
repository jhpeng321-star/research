"""Mean-value student, signed residual confidence and direct-choice comparator."""
import warnings

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import OPTIONS, KEYS, encode

FAMILIES = ('gbt', 'mlp')
CONFIDENCES = (0., .5, .65, .8, .9)


def estimator(family):
    if family == 'gbt':
        return GradientBoostingRegressor(n_estimators=80, max_depth=2, min_samples_leaf=6,
                                         loss='squared_error', learning_rate=.05, random_state=20261101)
    if family != 'mlp':
        raise ValueError('unknown model family')
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(16,8), activation='tanh', solver='lbfgs', alpha=1.,
            max_iter=300, random_state=20261101)), transformer=StandardScaler())


def fit_values(x, y, available, family):
    models = []
    for k in range(1,5):
        mask = available[:,k]
        # Unseen options get a nonpositive mean and cannot trigger a move.
        m = estimator(family) if mask.sum() >= 3 else DummyRegressor(strategy='constant',constant=0.)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            m.fit(x[mask] if mask.any() else x[:1], y[mask,k] if mask.any() else np.zeros(1))
        m.distill_convergence_warnings_ = sum(isinstance(w.message,ConvergenceWarning) for w in caught)
        models.append(m)
    return models


def predict_values(models, x):
    return np.column_stack([np.zeros(len(x))]+[m.predict(x) for m in models])


def teacher(y, available):
    # np.argmax gives deterministic O0-first ties for exact zero effects.
    return np.where(available,y,-np.inf).argmax(1)


def residuals(pred, y, available):
    # Signed overprediction errors, option-specific, fit on TRAIN OOF only.
    return [np.sort(pred[available[:,k],k]-y[available[:,k],k]) for k in range(5)]


def confidence(pred, errors):
    result = np.zeros_like(pred)
    for k in range(1,5):
        if len(errors[k]):
            # P(D>0) = P(pred-D < pred); strict at observed residual ties.
            result[:,k] = np.searchsorted(errors[k],pred[:,k],side='left')/len(errors[k])
    return result


def choose(pred, available, errors, level):
    ranked = np.where(available,pred,-np.inf)
    best = ranked.argmax(1)
    i = np.arange(len(pred))
    positive = ranked[i,best] > 0
    if level > 0:
        positive &= confidence(pred,errors)[i,best] >= level
    return np.where(positive,best,0)


def fit_classifier(x, y, available):
    target = teacher(y,available)
    model = (GradientBoostingClassifier(n_estimators=60,max_depth=2,min_samples_leaf=6,
                                        learning_rate=.05,random_state=20261101)
             if len(set(target)) > 1 else DummyClassifier(strategy='most_frequent'))
    return model.fit(x,target)


def classify(model, x, available):
    probs = np.zeros((len(x),5))
    probs[:,np.asarray(model.classes_,int)] = model.predict_proba(x)
    return np.where(available,probs,-np.inf).argmax(1)


class Student:
    """Same observable option API as COOPPLAN; no teacher/runtime access at inference."""
    def __init__(self, payload):
        self.payload = payload
        self.last_time = None
        if tuple(payload['keys']) != KEYS:
            raise ValueError('student feature schema mismatch')

    def step(self, state):
        from tools.coopplan.planner import candidates, TIMES
        if state.time_s not in TIMES or self.last_time is not None and state.time_s <= self.last_time:
            raise ValueError('strict COOPPLAN60s cadence required')
        setting = ((state.time_s-300)//60) % 4
        actions = candidates(state)
        available = np.array([[o in {a['id'] for a in actions} for o in OPTIONS]])
        x = encode(state.rows,state.time_s,setting)[None]
        pred = predict_values(self.payload['models'],x)
        selected = int(choose(pred,available,self.payload['errors'],self.payload['confidence'])[0])
        name = OPTIONS[selected]
        action = next(a for a in actions if a['id'] == name)
        self.last_time = state.time_s
        return dict(time_s=state.time_s, replay_key=state.replay_key, candidates=actions,
                    available_candidates=actions, executed_action=action, group=list(action['members']),
                    initial_rows=list(state.rows), horizon_s=600,
                    decision=dict(selected=name, predicted_mean_D=pred[0].tolist(),
                                  probability_positive=confidence(pred,self.payload['errors'])[0].tolist(),
                                  confidence=self.payload['confidence'], backend='DISTILL_OBSERVABLE'))
