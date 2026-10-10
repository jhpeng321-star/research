import gzip
import json
import pytest
from output.multi_wall_probe_v1.prefix_v2 import verify


def fixture(path, *, changed_command=False, changed_state=False):
    path.mkdir()
    with gzip.open(path/'interactions.jsonl.gz','wt',encoding='utf-8') as f:
        f.write(json.dumps(dict(kind='header',path=str(path)))+'\n')
        for t in (0,.5,1,1.5,2):
            f.write(json.dumps(dict(kind='sample',time_s=t,vehicles=[int(changed_state and t==2)]))+'\n')
    rows=[dict(time_s=i,phase='hold',reason='law',member_ids=['a'],wall_commands={'a':int(changed_command and i==2)},
               proposal_sha256='p',evidence_sha256='e',assignment=str(path)) for i in range(3)]
    (path/'policy.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    (path/'intervals.jsonl').write_text('\n'.join(json.dumps(dict(start_s=i,end_s=i+1,value=0)) for i in range(3)))
    telemetry=[]
    for i in range(3):
        telemetry += [dict(kind='rsu_received_report',received_at_s=i,value=0),
                      dict(kind='command',timestamp_s=i,value=int(changed_command and i==2))]
    (path/'telemetry.jsonl').write_text('\n'.join(json.dumps(r) for r in telemetry))


def test_boundary_is_precommand_but_includes_measured_state(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'
    fixture(a)
    fixture(b,changed_command=True)
    result=verify(a,b,2)
    assert result['counts']==dict(physical=5,policy=2,intervals=2,telemetry=5)


def test_measured_difference_at_boundary_invalidates_pair(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'
    fixture(a)
    fixture(b,changed_state=True)
    with pytest.raises(AssertionError):
        verify(a,b,2)
