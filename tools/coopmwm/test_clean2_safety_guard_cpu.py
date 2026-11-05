"""Hard events cannot be silently replaced as generic failed TRAIN episodes."""
import json

import pytest

from .accel_plant import clean2_safety_failure
from .accel_runner import episode_local
from .warm_health import HealthPause


@pytest.mark.parametrize('arm',['A','B','R'])
@pytest.mark.parametrize('event',['collisions','teleports'])
def test_absolute_hard_event_publishes_durable_pause_and_cannot_be_replaced(tmp_path,arm,event):
    job=dict(clean2_safety_guard=True,queue_root=str(tmp_path),arm=arm,id='owned_episode',
             world=dict(seed=986000),safety_reference=dict(collisions=99,teleports=99))
    totals=dict(collisions=0.,teleports=0.);totals[event]=1.
    with pytest.raises(HealthPause) as error:
        clean2_safety_failure(job,totals)
    assert episode_local(error.value) is False
    receipt=json.loads((tmp_path/'PAUSE_SAFETY.json').read_text())
    assert receipt['evidence']['arm']==arm and receipt['evidence']['totals']==totals
    assert receipt['evidence']['job']=='owned_episode'
    assert receipt['checkpoint_selection'] is False


def test_no_hard_event_keeps_composite_report_only_and_legacy_off_unchanged(tmp_path):
    job=dict(clean2_safety_guard=True,queue_root=str(tmp_path),arm='R',id='episode',world=dict(seed=986000))
    clean2_safety_failure(job,dict(collisions=0.,teleports=0.,ttc_events=999))
    clean2_safety_failure(dict(job,clean2_safety_guard=False),dict(collisions=1.,teleports=0.))
    assert not list(tmp_path.glob('PAUSE_*.json'))
    assert episode_local(ValueError('environment worker error')) is True
