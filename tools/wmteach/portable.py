"""Numpy-only export of the SAME shared GBT/MLP student, without family changes."""
import numpy as np

from tools.distill.data import KEYS, OPTIONS, encode


def export(payload):
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.compose import TransformedTargetRegressor
    if tuple(payload['keys']) != KEYS or payload['family'] not in ('gbt','mlp'):
        raise ValueError('shared student feature/family contract')
    result={k:payload[k] for k in ('family','confidence','label_horizon_s','train_worlds')}
    result.update(schema='wmteach_student_numpy/1',keys=list(KEYS),options=list(OPTIONS),
                  errors=[e.tolist() for e in payload['errors']],models=[])
    for m in payload['models']:
        if isinstance(m,DummyRegressor):
            item=dict(kind='constant',value=float(m.constant_.ravel()[0]))
        elif isinstance(m,GradientBoostingRegressor):
            trees=[]
            for estimator in m.estimators_.ravel():
                t=estimator.tree_
                trees.append(dict(feature=t.feature.tolist(),threshold=t.threshold.tolist(),
                    left=t.children_left.tolist(),right=t.children_right.tolist(),value=t.value[:,0,0].tolist()))
            item=dict(kind='gbt',initial=float(m.init_.constant_.ravel()[0]),rate=m.learning_rate,trees=trees)
        elif isinstance(m,TransformedTargetRegressor):
            scale,mlp=m.regressor_.steps[0][1],m.regressor_.steps[1][1]
            if mlp.activation!='tanh' or mlp.out_activation_!='identity':
                raise ValueError('unsupported frozen shared MLP activation')
            item=dict(kind='mlp',x_mean=scale.mean_.tolist(),x_scale=scale.scale_.tolist(),
                weights=[w.tolist() for w in mlp.coefs_],biases=[b.tolist() for b in mlp.intercepts_],
                y_mean=float(m.transformer_.mean_[0]),y_scale=float(m.transformer_.scale_[0]))
        else:
            raise ValueError('unsupported estimator; never substitute a different policy')
        result['models'].append(item)
    return result


def predict(policy,x):
    columns=[np.zeros(len(x))]
    for m in policy['models']:
        if m['kind']=='constant':
            y=np.full(len(x),m['value'])
        elif m['kind']=='mlp':
            h=(x-np.asarray(m['x_mean']))/np.asarray(m['x_scale'])
            for i,(w,b) in enumerate(zip(m['weights'],m['biases'])):
                h=h@np.asarray(w)+np.asarray(b)
                if i<len(m['weights'])-1: h=np.tanh(h)
            y=h.ravel()*m['y_scale']+m['y_mean']
        elif m['kind']=='gbt':
            # sklearn trees route FLOAT32 inputs against FLOAT64 thresholds.
            xf=x.astype(np.float32)
            y=np.full(len(x),m['initial'],dtype=float)
            for tree in m['trees']:
                feature=np.asarray(tree['feature']); threshold=np.asarray(tree['threshold'])
                left=np.asarray(tree['left']); right=np.asarray(tree['right'])
                node=np.zeros(len(x),dtype=int)
                active=left[node]>=0
                while active.any():
                    rows=np.flatnonzero(active); here=node[rows]
                    node[rows]=np.where(xf[rows,feature[here]]<=threshold[here],left[here],right[here])
                    active=left[node]>=0
                y+=m['rate']*np.asarray(tree['value'])[node]
        else:
            raise ValueError('unknown portable estimator')
        columns.append(y)
    return np.column_stack(columns)


def choose(policy,pred,available):
    # Algebra is shared; this function imports no sklearn at inference time.
    ranked=np.where(available,pred,-np.inf)
    best=ranked.argmax(1); i=np.arange(len(pred))
    positive=ranked[i,best]>0
    probabilities=np.zeros_like(pred)
    for k in range(1,5):
        e=policy['errors'][k]
        if len(e): probabilities[:,k]=np.searchsorted(e,pred[:,k],side='left')/len(e)
    if policy['confidence']>0: positive &= probabilities[i,best]>=policy['confidence']
    return np.where(positive,best,0),probabilities


class Student:
    def __init__(self,policy):
        if policy['schema']!='wmteach_student_numpy/1' or tuple(policy['keys'])!=KEYS:
            raise ValueError('frozen student schema mismatch')
        self.policy,self.last_time=policy,None

    def step(self,state):
        from tools.coopplan.planner import candidates,TIMES
        if state.time_s not in TIMES or self.last_time is not None and state.time_s<=self.last_time:
            raise ValueError('strict shared60s cadence required')
        setting=((state.time_s-300)//60)%4
        actions=candidates(state)
        av=np.array([[o in {a['id'] for a in actions} for o in OPTIONS]])
        pred=predict(self.policy,encode(state.rows,state.time_s,setting)[None])
        selected,probabilities=choose(self.policy,pred,av)
        action=next(a for a in actions if a['id']==OPTIONS[int(selected[0])])
        self.last_time=state.time_s
        return dict(time_s=state.time_s,replay_key=state.replay_key,candidates=actions,
            available_candidates=actions,executed_action=action,group=list(action['members']),
            initial_rows=list(state.rows),horizon_s=self.policy['label_horizon_s'],
            decision=dict(selected=action['id'],predicted_mean_D=pred[0].tolist(),
                probability_positive=probabilities[0].tolist(),confidence=self.policy['confidence'],
                backend='WMTEACH_OBSERVABLE_NUMPY'))
