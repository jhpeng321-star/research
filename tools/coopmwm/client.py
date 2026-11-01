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
        if (root/'STOP.json').exists() or (root.parent/'STOP.json').exists():raise ValueError('STOP while waiting for WM')
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


def random_guide(generator):
    """Uniform inherited available options, with no WM inference or saving filter."""
    def call(rows,commands,now,sample):
        from tools.coopbandit.protocol import group
        from .policy import targets
        from .wm import proposal_candidates
        members=group(rows)
        if len(members)!=2:
            return dict(accepted=False,option=None,targets={},expected_saving=None)
        record=dict(rows=rows,targets=targets(sample),group=members,setting=(int(now)//10)%4)
        options=proposal_candidates(record)[2:]
        if not options:
            return dict(accepted=False,option=None,targets={},expected_saving=None)
        import torch
        selected=options[int(torch.randint(len(options),(),generator=generator))]
        return dict(accepted=True,option=selected['id'],targets=selected['proposal_targets'],
                    expected_saving=None,option_probability=1/len(options),
                    available_options=[a['id'] for a in options],proposer='uniform-random')
    return call
