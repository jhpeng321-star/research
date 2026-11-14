"""Additive MAIN16:51 resource extension; original bindings stay immutable."""
from pathlib import Path
from .storage import read

NAME='RESOURCE_EXTENSION_1651.json'


def extension(root):
    root=Path(root);path=root/NAME
    if not path.exists():return None
    value=read(path)
    assert value['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
    assert value['gpu_deadline']==1793894340.0
    return value


def clocks(root,binding):
    value=extension(root)
    if value is None:return binding
    def apply(v):
        if isinstance(v,dict):
            return {k:value['gpu_deadline'] if k in ('gpu_deadline','deadline_unix') else apply(x)
                    for k,x in v.items()}
        if isinstance(v,list):return [apply(x) for x in v]
        return v
    result=apply(binding)
    result['gpu_deadline']=value['gpu_deadline']
    return result


def retired(root):
    result=dict(jobs=[],query_keys=[])
    for name in ('INCIDENT_RETIRED_JOBS.json','DNS_RETIRED_JOBS_1651.json','FAST_RETIRED_JOBS_1844.json'):
        p=Path(root)/name
        if p.exists():
            for key in result:result[key].extend(read(p).get(key,[]))
    return result
