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


REQUIRED_KEYS=('first_seed_advantage_veh_s','replication_advantage_veh_s',
               'cooperation.replication_cooperation_gain_veh_s')


def _applicable(c,key):
    """Records whose producer would carry ``key``; failed confirmations / incomplete or
    single-CAV cooperation legitimately lack the replicated values."""
    if key=='first_seed_advantage_veh_s':return True
    if key=='replication_advantage_veh_s':
        return 'replication' in (c.get('checks') or {}) or 'extended_advantage_veh_s' in c
    co=c.get('cooperation') or {}
    return bool(co.get('complete')) and co.get('cooperation_gain_veh_s') is not None


def _has(c,key):
    if key.startswith('cooperation.'):return key.split('.',1)[1] in (c.get('cooperation') or {})
    return key in c


def check_keys(records):
    """Schema guard: a statistical key absent from EVERY applicable record is a schema
    mismatch, not evidence; raise instead of silently rejecting the whole bank."""
    records=list(records)
    for key in REQUIRED_KEYS:
        rows=[c for c in records if _applicable(c,key)]
        if rows and not any(_has(c,key) for c in rows):
            raise KeyError(f'bank_statistics: key {key!r} missing from all {len(rows)} applicable '
                           'confirmation records; refusing to reject the whole bank silently')


def install(rb):
    """Process-local V2V reader adapter; never changes legacy training or files."""
    if getattr(rb.confirmations,'statistical_1727',False):return
    original=rb.confirmations
    def confirmations(root,arm):
        records=original(root,arm);check_keys(records.values())
        return {k:view(v) for k,v in records.items()}
    confirmations.statistical_1727=True
    rb.confirmations=confirmations
