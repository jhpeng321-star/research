"""Unchanged CV12 oracle acceptance semantics, explicit exposed seed4108 only."""
from pathlib import Path

import pytest
from tests import test_cv12_review as prior
from output.controller_verification_v13 import independent_review as review


@pytest.fixture
def saved(tmp_path):
    root,expected=prior.saved.__wrapped__(tmp_path)
    expected['seed']=4108
    for name in ('manifest','summary'):
        value=review.read(root/(name+'.json'))
        value['seed']=4108
        prior.write(root/(name+'.json'),value)
    prior.refresh(root)
    return root,expected


def test_only_seed_and_experiment_label_change():
    root=Path(__file__).resolve().parents[1]
    old=(root/'output/controller_verification_v12/independent_review.py').read_text()
    new=(root/'output/controller_verification_v13/independent_review.py').read_text()
    assert new==old.replace('CV-12','CV-13').replace('CV12','CV13').replace('4107','4108')


def test_complete_4108_evidence_passes(saved):
    verdict=review.verify_run(*saved)
    assert verdict['integrity_pass'],verdict
    assert verdict['safety_pass']
    assert verdict['physical_steps']==4800 and verdict['intervals']==2400
    assert verdict['tracking']['counts']=={'delivered':1,'departure_censored':1}


@pytest.mark.parametrize('kind',['pi_source_evidence','pi_owner_receipt','command','response'])
def test_missing_rows_fail_after_receipt_refresh(saved,kind):
    root,expected=saved
    data=list(review.rows(root/'telemetry.jsonl'))
    data.pop(next(i for i,x in enumerate(data) if x['kind']==kind))
    prior.lines(root/'telemetry.jsonl',data)
    prior.refresh(root)
    assert not review.verify_run(root,expected)['integrity_pass']


@pytest.mark.parametrize('field',['previous_raw_command_mps','raw_proposal_mps','generation'])
def test_recurrence_corruption_fails(saved,field):
    root,expected=saved
    data=list(review.rows(root/'telemetry.jsonl'))
    next(x for x in data if x['kind']=='pi_source_evidence' and x['timestamp_s']==2.)[field]+=1
    prior.lines(root/'telemetry.jsonl',data)
    prior.refresh(root)
    assert not review.verify_run(root,expected)['integrity_pass']


def test_wrong_seed_cannot_be_relabelled_as_4108(saved):
    root,expected=saved
    expected['seed']=4107
    for name in ('manifest','summary'):
        item=review.read(root/(name+'.json'))
        item['seed']=4107
        prior.write(root/(name+'.json'),item)
    prior.refresh(root)
    assert not review.verify_run(root,expected)['integrity_pass']


def test_collision_is_safety_failure(saved):
    root,expected=saved
    intervals=list(review.rows(root/'intervals.jsonl'))
    intervals[1]['collision_involvements']=1
    prior.lines(root/'intervals.jsonl',intervals)
    summary=review.read(root/'summary.json')
    summary['totals']['collision_involvements']=1
    prior.write(root/'summary.json',summary)
    prior.refresh(root)
    verdict=review.verify_run(root,expected)
    assert verdict['integrity_pass'] and not verdict['safety_pass']
