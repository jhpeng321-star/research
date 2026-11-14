import json
from pathlib import Path
import pytest
from . import composite_health as c,warm_health as h
from .protocol import sha
from .test_safety_diagnosis_cpu import rows


def put(p,v):p.write_text(json.dumps(v))


def fixture(root):
    put(root/'AUTHORITY.md','authorized-rule authorized-release')
    put(root/'PAUSE_SAFETY.json',dict(rule='SAFETY',evidence='original'))
    put(root/'SAFETY_RULE_0957.json',dict(authority=c.AUTHORITY,TTC_threshold=1.5,bootstrap=10000,
        bootstrap_seed=20261101,alpha=.05,amendment_sha256='authorized-rule'))
    put(root/'SAFETY_DIAGNOSIS_u6.json',dict(missing={'B':[],'R':[]},paired_world_rows={'B':rows(hard=4.,ttc=-10.),'R':rows()}))
    put(root/'SAFETY_RELEASE_0957.json',dict(authority=c.AUTHORITY,released=True,
        original_pause_sha256=sha(root/'PAUSE_SAFETY.json'),diagnosis_sha256=sha(root/'SAFETY_DIAGNOSIS_u6.json'),
        rule_sha256=sha(root/'SAFETY_RULE_0957.json'),amendment_sha256='authorized-release'))


def test_only_exact_signed_eligible_old_pause_is_released(tmp_path):
    fixture(tmp_path)
    assert c.released(tmp_path,tmp_path/'PAUSE_SAFETY.json')
    h.assert_running(tmp_path,'B')
    put(tmp_path/'PAUSE_HARMFUL_PLANS.json',dict(new=True))
    with pytest.raises(h.HealthPause):h.assert_running(tmp_path,'R')
    assert not c.released(tmp_path,tmp_path/'PAUSE_HARMFUL_PLANS.json')


def test_modified_evidence_or_new_collision_stays_paused(tmp_path):
    fixture(tmp_path)
    put(tmp_path/'PAUSE_SAFETY.json',dict(rule='SAFETY',evidence='changed'))
    assert not c.released(tmp_path,tmp_path/'PAUSE_SAFETY.json')
    with pytest.raises(h.HealthPause):h.assert_running(tmp_path,'B')


def test_new_safety_evidence_never_overwrites_released_original(tmp_path):
    fixture(tmp_path);original=(tmp_path/'PAUSE_SAFETY.json').read_bytes()
    with pytest.raises(h.HealthPause):h.pause(tmp_path,'SAFETY',dict(collisions=1))
    assert (tmp_path/'PAUSE_SAFETY.json').read_bytes()==original
    assert len(list(tmp_path.glob('PAUSE_SAFETY_*.json')))==1
    with pytest.raises(h.HealthPause):h.assert_running(tmp_path,'B')
