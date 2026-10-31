"""Explicit surviving controller-code binding, matching S-C1CLOSED's source API."""
from pathlib import Path
import worker as w


def install():
    from wms.sim_bridge import load_controller_dependencies
    import wms.train.cav_drive as cd
    root=w.frozen.OWN/'e3final/runtime'
    assert root.is_dir()
    deps=load_controller_dependencies(root)
    values=(deps.config,deps.circles_type,deps.direct_target,deps.neighbor_type,
            deps.frame_type,deps.pi_type,deps.u_report_type,deps.pi_params_type)
    cd.load_controller_dependencies=lambda controllers:values
    return root


def init():
    w.init()
    install()


def main():
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import inspect
    import json
    import os
    import time
    assert os.getpriority(os.PRIO_PROCESS,0)==19
    assert os.getloadavg()[0]<100
    assert not json.loads((w.EVIDENCE/'TF_DONE.json').read_text())['errors']
    w.UNITS=w.ROOT/'feedback_units'
    out=w.ROOT/'feedback_evidence'
    panel=json.loads((w.ROOT/'PANEL.json').read_text())['states']
    selected=json.loads((w.EVIDENCE/'SELECTED.json').read_text())
    units=[]
    for e in panel:
        hist='h5' if e['cohort']=='wall_same' else 'snapshot'
        for tag in ('c1struct','bbtraj','wm13e','wm2'):
            units.append((e,tag,hist,False,True,'original','clh'))
        for s in selected:
            if s['cohort']==e['cohort']:
                u=(e,s['tag'],s['history'],s['population'],s['residual'],s['lanes'],'clh')
                if u not in units:units.append(u)
    root=install()
    import wms.train.cav_drive as cd
    from wms.sim_bridge import load_controller_dependencies
    deps=load_controller_dependencies(root)
    paths=[Path(__file__),Path(w.__file__),Path(inspect.getfile(cd.TimelineControllers))]
    paths += [Path(inspect.getfile(x)) for x in (deps.pi_type,deps.pi_params_type,deps.direct_target,deps.circles_type)]
    sources={str(p):w.digest(p) for p in paths}
    w.write(out/'DISPATCH.json',dict(units=len(units),workers=24,sources=sources,controller_root=str(root)))
    begin=time.time();errors=[]
    with ProcessPoolExecutor(max_workers=24,initializer=init) as pool:
        checks=[u for u in units if u[0]['sid']==panel[0]['sid']]
        for f in as_completed([pool.submit(w.run,u) for u in checks]):
            error=f.result()
            if error:errors.append(error)
        w.write(out/'PREFLIGHT.json',dict(errors=errors,units=len(checks)))
        if errors:return
        futures=[pool.submit(w.run,u) for u in units]
        for i,f in enumerate(as_completed(futures),1):
            error=f.result()
            if error:errors.append(error)
            if i%25==0 or i==len(futures):print('feedback_recovery',i,'/',len(futures),'errors',len(errors),flush=True)
    assert all(w.digest(p)==h for p,h in sources.items())
    w.write(out/'COMPLETE.json',dict(units=len(units),errors=errors,seconds=time.time()-begin,sources=sources))


if __name__=='__main__':main()
