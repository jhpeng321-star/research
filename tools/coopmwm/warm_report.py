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
