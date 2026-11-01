"""Immutable local queue shared by owned simulator workers and the GPU bridge."""
import time
from pathlib import Path
from .protocol import digest
from .storage import read,write


def request(root,record):
    root=Path(root);key=record['state_id']
    path=root/'requests'/(key+'.json');result=root/'responses'/(key+'.json')
    if not path.exists():write(path,record)
    elif read(path)!=record:raise ValueError('request identity collision')
    while not result.exists():
        if (root/'STOP.json').exists():raise ValueError('STOP while waiting for WM')
        time.sleep(2)
    value=read(result)
    if value['record_sha256']!=digest(record):raise ValueError('WM response identity mismatch')
    return value


def guide(root,world):
    def call(rows,commands,now,sample):
        from tools.coopbandit.protocol import group
        from .policy import targets
        members=group(rows)
        if len(members)!=2:
            return dict(accepted=False,option=None,targets={},expected_saving=0.)
        payload=dict(world_id=world['seed'],time_s=int(now),rows=rows,
            targets=targets(sample),group=members,mode='proposal',setting=(int(now)//10)%4)
        record=dict(payload,state_id=digest(payload))
        return request(root,record)
    return call
