"""Descriptive exact-labelled warm/cold qualification; no outcome tuning."""
from datetime import datetime,timezone
from .storage import write
from .protocol import digest,checked
from . import ops


def summary(data):
    from .accel_diagnostic import agreement
    timing={};diagnostic={};memory={}
    for host,value in data.items():
        rs=value['receipts'];memory[host]=rs['CAUSAL_MEMORY.json']
        for size in (2,4,6):
            key=f'SEARCH_GROUP{size}.json'
            if key in rs:
                r=rs[key]
                timing[str(size)]=dict(host=host,wall_s=r['wall_s'],plans_per_hour=r['plans_per_hour'],
                    candidates=r['result']['candidates_searched'],futures=r['result']['imagined_futures'],
                    screen=r['result']['screen'],final=r['result']['final'],centre_open_loop_gap=r['result']['centre_open_loop_gap'])
    for mode in ('cold','warm'):
        records=[];choices=[];times=[]
        for value in data.values():
            r=value['receipts']['E2_'+mode.upper()+'.json']
            records.append(dict(world=dict(seed=r['world_id']),wm=r['credits'],exact=r['exact']))
            choices.append(dict(world=r['world_id'],horizons=r['choices']))
            times.append(r['wall_s'])
        diagnostic[mode]=dict(agreement=agreement(records),choices=choices,query_wall_s=times)
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),memory=memory,timing=timing,
        exact_labelled_E2=diagnostic,interpretation='two registered near-parity set-point windows, not novel MPPI plan labels; ties separate, two-world CIs uninformative; no established warm decision-quality improvement',
        new_sumo_sim_s=0)


def publish(value,qualified=False):
    target=ops.EVIDENCE/('WARM_MEMORY_QUALIFIED.json' if qualified else 'WARM_MEMORY_DIAGNOSTIC_'+digest(value)[:16]+'.json')
    write(target,value)
    doc=checked(ops.LOCAL/'docs/paper_results/COOPMWM.md')
    lines=['\n## '+('WARM MEMORY QUALIFIED' if qualified else 'Warm memory diagnostic')+' — '+value['timestamp_utc']+'\n']
    for size,t in sorted(value['timing'].items()):
        lines.append(f'- {size} CAVs, {t["host"]}: **{t["wall_s"]:.3f}s/query, {t["plans_per_hour"]:.3f}plans/hour**, '
            f'{t["candidates"]} candidates/{t["futures"]} futures; screen `{t["screen"]}`, final `{t["final"]}`; '
            f'open-loop centre cost gap {t["centre_open_loop_gap"]:.3f}veh-s (future0 centre vs K4 closed-loop comparison).\n')
    for mode,d in value['exact_labelled_E2'].items():
        lines.append(f'- {mode} E2 query times: `{d["query_wall_s"]}`s.\n')
        for h,v in d['agreement']['horizons'].items():
            lines.append(f'- {mode} H{h}: D_i sign `{v["D_i"]}`; C_G sign `{v["C_G"]}`.\n')
        lines.append(f'- {mode} exact-labelled chosen-action savings: `{d["choices"]}`.\n')
    lines.append('- '+value['interpretation']+'. All original exact SUMO labels reused; qualification adds0SUMO.\n')
    lines.append(f'- Immutable complete receipt: `coopmwm/{target.name}`.\n')
    with doc.open('a',encoding='utf-8') as f:f.write(''.join(lines))
    return [str(doc.relative_to(ops.LOCAL)),str(target.relative_to(ops.LOCAL))]


def qualify():
    """Publish completed model qualification independently of learner release."""
    import json
    from .warm_release_ops import collect
    from .warm_remote_ops import probe_routes
    from .storage import read
    from .protocol import sha,REGISTRY
    target=ops.EVIDENCE/'WARM_MEMORY_QUALIFIED.json'
    assert not target.exists(),'adopt published warm qualification'
    data=collect()
    assert all(v['state'] in ('exited','Z') and v['receipts']['COMPLETE.json']['pass_'] for v in data.values())
    routes=probe_routes()
    for host in ('gpuA','gpuR'):
        route=routes[host+'@0'];assert route['alive'] and route['ready'] and not route['failed']
        files=sorted(ops.EVIDENCE.glob('WARM_DURABLE_STATUS_'+host+'_*.json'))
        w=read(files[-1]);assert w['receipts']['SPOOL_OWNER.json']==route['owner']
        gates=[v for n,v in w['receipts'].items() if n.startswith('STARTUP_')]
        assert len(gates)==6 and all(v['pass_'] and v['max_cost_error']==0 for v in gates)
    value=summary(data)
    value.update(pass_=True,qualification_only=True,training_dispatch=False,routes=routes,
        runtime_cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_durablecpu2.json'),
        search_screen='K4/H300 always; MAIN02:36 conditional screen revocation',
        launch_requirement='separate exact-only learner/grader integration and immutable release; not authorized by model qualification')
    assert value['runtime_cpu']['pass_'] and value['runtime_cpu']['tests']==160
    publish(value,qualified=True)
    line=('\nS-COOPMWM WARM MEMORY QUALIFIED qualificationSHA'+sha(target)+
        ': completed original causal60s observable per-driver FULLv4 posterior/empty-cold/CPU160 gates adopted; '
        'both original persistent GPU spools READY, six startup gates each EXACT0; original E2 cold/warm labels '
        'and group2/4/6 timings retained, zero new SUMO. Two-window decision comparison is descriptive, '
        'no demonstrated accuracy improvement. Registered MPPI K4/H300 always, K8/Hfinal unchanged. '
        'This qualifies warm inference ONLY; exact-only learner/grader integration and separate runtime release remain before B/R learning.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'WARM_MEMORY_QUALIFICATION_REGISTRATION.json',dict(line=line,qualification_sha256=sha(target),new_sumo_sim_s=0))
    print(json.dumps(dict(qualified=True,timestamp_utc=value['timestamp_utc'],training_dispatch=False,
        timing=value['timing'],new_sumo_sim_s=0)))


if __name__=='__main__':qualify()
