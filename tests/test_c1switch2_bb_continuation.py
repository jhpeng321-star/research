"""MAIN12:14 is startup-only, per-card debited, and refuses before claims on pressure."""
import copy
import json

import pytest

from tools.c1switch.storage import canonical
from tools.c1switch2 import bbcontinue, gpuhold
from tools.c1switch2.gpuallocation import AUTHORITY, PREDECESSOR, PRIOR, allocation_cap


@pytest.fixture(autouse=True)
def isolated_operational_hold(tmp_path,monkeypatch):
    monkeypatch.setattr(gpuhold,'HOLD',tmp_path/'GPU_HOLD.json')


def allocation(host):
    return dict(schema='S-C1SWITCH2/BB-MAIN1214-v1',host=host,authority=AUTHORITY,
        plan_sha256='a'*64,predecessor_root=PREDECESSOR,prior_gpu_s=PRIOR[host],
        remaining_gpu_s=5400-PRIOR[host],finished=dict(elapsed_s=PRIOR[host]))


def test_original_and_continuation_share_exact_original_per_card_cap():
    binding = dict(arm='bb',service_gpu_cap_s=5400,prior_gpu_s=0)
    assert allocation_cap(binding) == 5400
    assert sum(PRIOR.values()) == 43.02397724799812
    for host in PRIOR:
        assert allocation_cap(binding,allocation(host))+PRIOR[host] == 5400


@pytest.mark.parametrize('field,value',[
    ('prior_gpu_s',0),('remaining_gpu_s',5400),('authority',''),
    ('host','cpuQ'),('predecessor_root','/root/other'),('remaining_gpu_s',float('nan')),
])
def test_exception_cannot_refund_expand_or_retarget(field,value):
    row = copy.deepcopy(allocation('gpuR'));row[field] = value
    with pytest.raises(ValueError):
        allocation_cap(dict(arm='bb',service_gpu_cap_s=5400),row)


def test_exception_never_restarts_wm():
    with pytest.raises(ValueError):
        allocation_cap(dict(arm='wm',service_gpu_cap_s=28800),allocation('gpuA'))


def test_resource_refusal_leaves_no_claim_or_gpu_start(tmp_path,monkeypatch):
    plan = dict(original='unchanged')
    config = dict(continuation_plan_sha256=canonical(plan))
    plan_path,bound,claim = tmp_path/'plan.json',tmp_path/'bound.json',tmp_path/'claim.json'
    plan_path.write_text(json.dumps(plan));bound.write_text(json.dumps(config))
    monkeypatch.setattr(bbcontinue,'PLAN',plan_path)
    monkeypatch.setattr(bbcontinue,'BOUND',bound)
    monkeypatch.setattr(bbcontinue,'CLAIM',claim)
    monkeypatch.setattr(bbcontinue,'authority',lambda:(canonical(plan),AUTHORITY))
    monkeypatch.setattr(bbcontinue,'resources',lambda config:({'disk':{'free_bytes':100000000}},False))
    monkeypatch.setattr(bbcontinue,'proof',lambda config:pytest.fail('proof/claim/start before resource admission'))
    assert bbcontinue.start() is False
    assert not claim.exists()


def test_consumed_continuation_is_never_retried(tmp_path,monkeypatch):
    plan = dict(original='unchanged')
    plan_path,bound,claim = tmp_path/'plan.json',tmp_path/'bound.json',tmp_path/'claim.json'
    plan_path.write_text(json.dumps(plan))
    bound.write_text(json.dumps(dict(continuation_plan_sha256=canonical(plan))))
    claim.write_text('{}')
    monkeypatch.setattr(bbcontinue,'PLAN',plan_path)
    monkeypatch.setattr(bbcontinue,'BOUND',bound)
    monkeypatch.setattr(bbcontinue,'CLAIM',claim)
    monkeypatch.setattr(bbcontinue,'authority',lambda:(canonical(plan),AUTHORITY))
    monkeypatch.setattr(bbcontinue,'resources',lambda config:pytest.fail('consumed continuation retried'))
    with pytest.raises(ValueError,match='one-use'):
        bbcontinue.start()
