"""Meaningful estimand tests: cohort attrition, additive median, cluster resampling."""
import json
import ast
import numpy as np
import analyze as a


def main():
    for source in a.HERE.glob('*.py'):
        ast.parse(source.read_text(encoding='utf-8'),filename=str(source))
    real=dict(ids=[1,2,3],s=[10.,20.,30.],v=[1.,2.,3.])
    pred=dict(endpoints=[dict(ids=[1,2,3],s=[12.,25.,35.],v=[1.,3.,3.],present=[True,True,False]),
                         dict(ids=[1,2,3],s=[14.,27.,35.],v=[1.,4.,3.],present=[True,False,True])])
    # All-draw common survival is ID 1, not an unpaired median on 1+2 / 1+3.
    score,ids=a.endpoint(pred,real,[1,2,3])
    assert ids=={1} and score==dict(position=3.,speed=0.,n_hdv=1)
    missing,_=a.endpoint(pred,real,[3],cohort={2})
    assert missing==dict(position=None,speed=None,n_hdv=0)
    # Native-first cost ties; true pair ties are excluded, not counted as wins.
    d=a.decision([1.,1.,2.],[10.,0.,0.])
    assert d['regret']==10 and d['top_choice_correct']==0 and d['pairwise_accuracy']==0
    assert a.decision([0.,0.],[5.,5.])['direction_accuracy'] is None
    # Opposing components deliberately have different marginal median draws.
    draws=[]
    for j,s,u in [(0.,0.,0.),(10.,10.,0.),(20.,0.,20.),(100.,100.,0.)]:
        val=dict(J=[j+.0001],source=[s],upstream=[u],merge=[0.],discharge=[0.],
            crossings=[0],merge_queue=[0],mean_merge_queue=[0.],
            fields=dict(density=[[[0.]*15]],speed=[[[0.]*15]]),endpoints=[pred['endpoints'][0]])
        draws.append(dict(arms=['pi'],values={'10':val}))
    f=a.forecast(draws,'pi',10)
    assert f['J']==15. and f['source']==5. and f['upstream']==10.
    assert sum(f[k] for k in a.COMP)==f['J']
    assert np.isclose(f['J_rounding_correction'],-.0001)
    # Whole worlds, not vehicles/ticks/controllers, determine interval width.
    rows=[dict(model='test',plant='common',world=w,arm=arm,horizon_s=10,position=float(w))
          for w in range(40) for arm in ('pi','mappo')]
    s=[r for r in a.summarize(rows) if r['model']=='test']
    assert len(s)==2 and all(r['estimate']==19.5 and r['n_worlds']==40 for r in s)
    assert s[0]['lo']==s[1]['lo'] and s[0]['hi']==s[1]['hi']
    sample=np.random.default_rng(0).integers(0,40,(2000,40))
    lo,hi=np.percentile(np.median(sample,axis=1),[2.5,97.5])
    assert (s[0]['lo'],s[0]['hi'])==(lo,hi)
    print(json.dumps(dict(all_task_python_parses=True,cohort_attrition=True,unavailable_not_zero=True,
        native_first_ties=True,additive_order_weights=True,whole_world_bootstrap=True)))


if __name__=='__main__':main()
