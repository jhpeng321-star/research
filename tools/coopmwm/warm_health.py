"""MAIN01:38 pre-registered pauses. No final-panel access or outcome selection."""
from pathlib import Path
import time
import numpy as np
from .storage import read,write
from .protocol import sha


class HealthPause(RuntimeError):
    pass


def active_stop(root):
    root=Path(root);p=root/'STOP.json'
    if not p.exists():return False
    release=root/'WARM_TRANSPORT_RELEASE.json'
    if not release.exists():return True
    v=read(release)
    # Only this diagnosed transport stop is releasable; original stays on disk.
    return not (v.get('original_stop_sha256')==sha(p) and
        read(p)==dict(error='Connection closed by 1.15.42.207 port 22\n',
            source='owned warm bridge; no automatic retry') and
        v['amendment_sha256'] in (root/'AUTHORITY.md').read_text())


def assert_running(root,arm):
    root=Path(root)
    if active_stop(root):raise ValueError('durable warm STOP')
    if arm in ('B','R'):
        if (root/'WARM_GUIDED_STOP_V5.json').exists():raise ValueError('new durable guided engineering STOP')
        p4=root/'WARM_GUIDED_STOP_V4.json'
        if p4.exists():
            release=root/'DNS_GUIDED_RELEASE_1651.json'
            v=read(release) if release.exists() else {}
            if not (v.get('original_stop_sha256')==sha(p4)
                and read(p4).get('source')=='capacity broker'
                and read(p4).get('error')=="RuntimeError('ssh: Could not resolve hostname connect.westb.seetacloud.com: Temporary failure in name resolution')"
                and v.get('amendment_sha256','UNREGISTERED') in (root/'AUTHORITY.md').read_text()):
                raise ValueError('new durable guided engineering STOP')
        p3=root/'WARM_GUIDED_STOP_V3.json'
        if p3.exists():
            release=root/'MATCH_GUIDED_RELEASE.json'
            v=read(release) if release.exists() else {}
            if not (v.get('original_stop_sha256')==sha(p3)
                and read(p3).get('error')=="AssertionError('R count/regime/group-size/duration gate')"
                and v.get('amendment_sha256','UNREGISTERED') in (root/'AUTHORITY.md').read_text()):
                raise ValueError('new durable guided engineering STOP')
        p2=root/'WARM_GUIDED_STOP_V2.json'
        if p2.exists():
            release=root/'INCIDENT_GUIDED_RELEASE.json'
            v=read(release) if release.exists() else {}
            if not (v.get('original_stop_sha256')==sha(p2)
                and read(p2).get('error')=="ValueError('GPU failed; no inference retry')"
                and v.get('amendment_sha256','UNREGISTERED') in (root/'AUTHORITY.md').read_text()):
                raise ValueError('new durable guided engineering STOP')
        from .composite_health import released
        pauses=sorted(p for p in root.glob('PAUSE_*.json') if not released(root,p))
        if pauses:raise HealthPause('registered health pause '+pauses[0].name)
        p=root/'WARM_GUIDED_STOP.json'
        if p.exists():
            release=root/'MAIN0508_GUIDED_RELEASE.json'
            v=read(release) if release.exists() else {}
            original=read(p)
            qualified=(v.get('original_stop_sha256')==sha(p)
                and original.get('source')=='capacity broker control'
                and original.get('error')=="JSONDecodeError('Expecting value: line 1 column 1 (char 0)')"
                and v.get('amendment_sha256','UNREGISTERED') in (root/'AUTHORITY.md').read_text())
            if not qualified:raise ValueError('durable guided engineering STOP')


def pause(root,rule,evidence):
    path=Path(root)/('PAUSE_'+rule+'.json')
    from .composite_health import released
    if path.exists() and released(root,path):
        path=path.with_name('PAUSE_'+rule+'_'+str(time.time_ns())+'.json')
    value=dict(rule=rule,evidence=evidence,unix=time.time(),arms=['B','R'],
        action='pause; preserve receipts/checkpoints/charges; diagnosis and registered release required',
        checkpoint_selection=False,final_panel_access=False)
    if not path.exists():write(path,value)
    raise HealthPause('registered health pause '+rule)


def safety_reference(records):
    # Existing MAPPO check episodes are the available no-new-physics benchmark.
    # For unmatched TRAIN worlds, a zero-event benchmark is conservative, never
    # an assertion that the MAPPO counterfactual has been run on that world.
    by_world={str(e['world']['seed']):{k:float(e['metrics']['900'][k])
        for k in ('collisions','teleports')} for e in records}
    if not by_world:raise ValueError('MAPPO safety reference absent')
    zero=all(v[k]==0 for v in by_world.values() for k in v)
    return dict(by_world=by_world,unmatched_zero_event_benchmark=zero,
        source='existing registered MAPPO TRAIN-check panel; unmatched TRAIN zero-event pause is conservative')


def live_safety(root,arm,totals,world,reference):
    if arm not in ('B','R'):return
    assert_running(root,arm)
    from .composite_health import r_report_only
    if r_report_only(root,arm) and any(totals[k]>0 for k in ('collisions','teleports')):
        pause(root,'SAFETY',dict(world=world,arm=arm,
            totals={k:totals[k] for k in ('collisions','teleports')},
            source='MAIN2026-11-05 12:30; R any collision/teleport still pauses'))
    baseline=reference['by_world'].get(str(world))
    if baseline is None:
        if not reference['unmatched_zero_event_benchmark']:
            raise ValueError('unmatched TRAIN safety reference is not zero; require MAIN decision')
        baseline=dict(collisions=0.,teleports=0.)
    delta={k:float(totals[k])-baseline[k] for k in baseline}
    if any(v>0 for v in delta.values()):
        pause(root,'SAFETY',dict(world=world,arm=arm,totals={k:totals[k] for k in baseline},
            baseline=baseline,difference=delta,source=reference['source']))


def before_update(root,arm,u,routes,cumulative,trust_state,n_exec,effective_guide_beta=None):
    assert_running(root,arm)
    if arm not in ('B','R'):return
    if arm=='B':
        executed=sum(r['executed'] for r in cumulative)
        interval=trust_state.get('ci95')
        if executed>=30 and interval is not None and interval[1]<0:
            pause(root,'HARMFUL_PLANS',dict(update=u,executed=executed,trust=trust_state))
        previous=Path(root)/f'B_UPDATE_{u-1:03}.json'
        if routes['executed']<n_exec and previous.exists() and read(previous)['proposal_routing']['executed']<n_exec:
            pause(root,'STARVATION',dict(update=u,previous_update=u-1,N_exec=n_exec,
                previous=read(previous)['proposal_routing'],current=routes))
        buckets=trust_state.get('buckets',{})
        if effective_guide_beta != 1. and u<8 and trust_state.get('beta')==0 and (not buckets or all(v['beta']==0 for v in buckets.values())):
            pause(root,'STARVATION',dict(update=u,reason='trust collapsed to zero before u8',trust=trust_state))


def after_check(root,arm,u,report):
    if arm!='B':return
    metrics=report['comparisons']['B_minus_A']['metrics']
    bad={k:v for k,v in metrics.items() if k in ('600_team_delay_vs','900_team_delay_vs') and v['ci95'][0]>0}
    previous=Path(root)/f'AB_CHECK_{u-4:03}.json'
    if bad and previous.exists():
        prior=read(previous)['comparisons']['B_minus_A']['metrics']
        if any(prior[k]['ci95'][0]>0 for k in bad):
            pause(root,'LEARNING',dict(update=u,current=bad,previous_update=u-4,
                previous={k:prior[k] for k in bad}))


def braking(root,arm,u,episodes,a_records,pairs=None):
    """Registered hard-brake + composite guard. ``pairs`` optionally maps an episode id to its
    EXACT matched reference record (practice_starts: the same-snapshot/same-CRN A-policy
    reference); episodes without an explicit pair fall back to the same-world reference exactly
    as before - the guard is never weakened."""
    if arm not in ('B','R'):return
    from .composite_health import braking as composite_braking
    if composite_braking(root,arm,u,episodes,a_records,pause,pairs=pairs):return
    a={e['world']['seed']:e for e in a_records}
    ref=lambda e:(pairs or {}).get(e.get('id')) or a.get(e['world']['seed'])
    common=[e for e in episodes if ref(e) is not None]
    if len(common)<2:return
    delta=np.array([e['metrics']['900']['hard_brake']-ref(e)['metrics']['900']['hard_brake'] for e in common])/900.
    idx=np.random.default_rng(20261101).integers(0,len(delta),(10000,len(delta)))
    ci=np.quantile(delta[idx].mean(1),[.025,.975]).tolist()
    if ci[0]>0:pause(root,'SAFETY',dict(update=u,arm=arm,metric='hard-brake vehicle-seconds / 900 control seconds',
        worlds=[e['world']['seed'] for e in common],mean=float(delta.mean()),ci95=ci,
        method='10000 paired whole-world bootstrap; RNG20261101; vs same-update A'))
