"""Source-bound fresh-eight-gate niche/nullfix worker; original bindings retained."""
import json
import os
from pathlib import Path
import time


def atomic(path,value):
    temporary=path.with_name(path.name+f'.nichenull.{os.getpid()}.publishing')
    with temporary.open('x') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def run(root,generation='NICHENULL1'):
    import fcntl,signal,sys,hashlib
    root=Path(root)
    lock=(root/'SPOOL_WORKER.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(root/'pytools'))
    from tools.coopmwm import warm_service,plan_service,mppi_search,mppi_runtime
    from tools.coopmwm.warm_spool import SpoolInput,immutable
    from tools.coopmwm.nullfix_geometry import install
    from tools.coopmwm.protocol import digest
    assert generation in ('NICHENULL1','NICHENULL2')
    binding=json.loads((root/('GPU_BINDING_'+generation+'.json')).read_text())
    for name,h in binding['hashes'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
    install(root/'FROZEN_GEOMETRY_SOURCE.py',root/'FROZEN_NETWORK.net.xml.gz')
    owner=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),
        amendment_sha256=binding['effective_amendment_sha256'])
    atomic(root/'SPOOL_OWNER.json',owner)
    immutable(root/(generation+'_OWNER.json'),owner)
    original_read=plan_service.read
    warm_service.read=plan_service.read=lambda p:binding if p==root/'GPU_BINDING.json' else original_read(p)
    original_write=plan_service.write
    def publish(path,value):
        if path.parent==root and path.name=='SERVICE_OWNER.json':return atomic(path,value)
        if path.parent==root and path.name.startswith('STARTUP_'):
            value=dict(value,effective_source_manifest_sha256=digest(binding['hashes']),
                amendment_sha256=binding['effective_amendment_sha256'])
            return original_write(root/(path.stem+'_LSCOLLECT_'+generation+'.json'),value)
        return original_write(path,value)
    warm_service.write=plan_service.write=publish
    from tools.coopmwm.ls_gpu import capture_search
    from tools.coopmwm.ls_fast_worker import configured_search
    mppi_search.search=configured_search(capture_search(mppi_search.search,label_list8=True))
    original_plan=mppi_runtime.plan
    def plan(*args,**kwargs):
        value=original_plan(*args,**kwargs)
        from tools.coopmwm.feasible import resolve_allowed_mps
        record=args[1]
        value['nullfix']=dict(enabled=True,allowed_speeds_resolved=
            resolve_allowed_mps(record['rows'],record['group']) is not None,
            mirror_active=False,source_manifest_sha256=digest(binding['hashes']))
        return value
    mppi_runtime.plan=plan
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    sys.stdin=SpoolInput(root,binding['deadline_unix'])
    try:warm_service.main(root)
    except BaseException as error:
        import traceback
        immutable(root/('LS_GPU_FAILED_'+generation+'.json'),dict(owner=owner,error=repr(error),
            traceback=traceback.format_exc(),unix=time.time(),inference_retry=False))
        raise


if __name__=='__main__':
    import sys
    run(sys.argv[1])
