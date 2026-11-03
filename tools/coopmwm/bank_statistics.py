"""MAIN17:27 conservative statistical views; original receipts stay immutable."""
import copy
import math


def replicated_ci(first,second):
    """Two-sided 95% Student-t mean CI over two independent CRN seeds (df=1).

    The extended window is NOT an independent replicate and is never a third
    observation. This deliberately exposes the large uncertainty at n=2.
    """
    if first is None or second is None:return None
    a,b=float(first),float(second)
    if not math.isfinite(a) or not math.isfinite(b):return None
    mean=(a+b)/2;half=12.706204736432095*abs(a-b)/2
    return [mean-half,mean+half]


def view(confirmation):
    c=copy.deepcopy(confirmation)
    ci=replicated_ci(c.get('first_seed_advantage_veh_s'),c.get('replication_advantage_veh_s'))
    c['legacy_admitted']=c.get('admitted',False)
    c['replicated_advantage_ci95']=ci
    c['admitted']=bool(c['legacy_admitted'] and ci is not None and ci[0]>0)
    co=c.get('cooperation') or {}
    co['legacy_cooperative']=co.get('cooperative')
    # A first-seed solo comparison cannot establish a replicated synergy CI.
    # Accept only independently recorded seed-level gains, never member-level
    # pseudo-replicates or a CI across unrelated bank states.
    rep=co.get('replication_cooperation_gain_veh_s')
    sci=replicated_ci(co.get('cooperation_gain_veh_s'),rep)
    co['synergy_ci95']=sci
    co['cooperative']=bool(c['admitted'] and co.get('complete') and sci is not None and sci[0]>0)
    co['statistical_reason']=('replicated synergy CI lower > 0' if co['cooperative'] else
        'replicated synergy CI unavailable or does not exclude zero')
    c['cooperation']=co
    c['statistical_rule']='two independent seed Student-t CI95 lower > 0; extended/safety constraints retained'
    return c


def install(rb):
    """Process-local V2V reader adapter; never changes legacy training or files."""
    if getattr(rb.confirmations,'statistical_1727',False):return
    original=rb.confirmations
    def confirmations(root,arm):return {k:view(v) for k,v in original(root,arm).items()}
    confirmations.statistical_1727=True
    rb.confirmations=confirmations
