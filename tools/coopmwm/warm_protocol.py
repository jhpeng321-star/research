"""Latest owner scope; isolated from every historical immutable cold run."""
from .resume_quote import quote
from .accel_protocol import worlds,SEED,SUPERSEDED_CHARGE
from .protocol import digest

UPDATES=dict(A=48,B=48,R=24)
CHECK_EVERY=4
DEV_EVERY=16
FOCUS_FRACTION=.5
VIABILITY_UPDATES=4


def budget():
    q=quote()
    return dict(total=q['cap_sim_s'],phases=q['selected']['phases'],allocated=q['allocated_sim_s'],
        unallocated=q['unallocated_sim_s'],updates=UPDATES,training_seeds=[SEED],
        episodes_per_update=24,final_panel_sha256=digest(worlds('final_test')),
        final_checkpoint_rule='fixed A/B48,R24; no outcome selection',R_branches='deferred by MAIN23:54')


def batch_worlds(update):
    """Exactly12 focus +12 other existing TRAIN worlds, same deterministic mix/arm.

    Focus: three repeats per each of four cells. Other: one/two repeats alternating
    across eight cells; identities rotate before outcomes, no extra seed exposure.
    A's completed balanced updates1..4 are retained; emphasis applies prospectively.
    """
    if type(update) is not int or not 1<=update<=48:raise ValueError('unregistered update')
    result=[];other=0
    for cell in range(12):
        pool=[w for w in worlds('train') if w['cell']==cell]
        focus=pool[0]['demand'] in (2400,2700) and pool[0]['penetration'] in (.2,.4)
        n=3 if focus else 1+int((other+update)%2==0)
        if not focus:other+=1
        start=((update-1)*(3 if focus else 2))%6
        repeats={(start+i)%6 for i in range(n)}
        result.extend(w for w in pool if w['repeat'] in repeats)
    assert len(result)==24 and sum(w['demand']>=2400 and w['penetration']>=.2 for w in result)==12
    return result


def install_accounting(root=None):
    """Only the new registered runtime calls this; old run roots stay immutable."""
    import types
    from tools.coopnative import storage
    from . import accel_storage
    b=budget()
    if root is not None:
        import json
        from pathlib import Path
        r=Path(root);p=r/'INCIDENT_RECOVERY_BINDING.json'
        if p.exists():
            v=json.loads(p.read_text())
            assert v['amendment_sha256'] in (r/'AUTHORITY.md').read_text()
            b=v['budget']
    accel_storage._reserve=types.FunctionType(storage.reserve.__code__,
        dict(storage.reserve.__globals__,CAP=b['total']-SUPERSEDED_CHARGE,CAPS=b['phases']))
