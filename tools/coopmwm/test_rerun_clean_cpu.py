"""Prepare-only clean rerun package: quote arithmetic, pins, approval-gated dry preflight (no physics)."""
import hashlib
import json
import pytest
from . import rerun_clean as rc


def test_quote_from_measured_receipts():
    q = rc.quote()
    assert q['per_arm_update_sim_s'] == 24*1200 + 6*1200 == 36000
    assert q['v2v_train'] == 2*24*36000 == 1728000
    assert q['v2v_lessons'] == 4*5280 and q['v2v_isolated'] == 6*32610
    assert q['v2v_interim'] == q['v2v_final'] == 2*96*1200
    assert q['total_without_tests'] == 1728000 + 21120 + 195660 + 35820 == 1980600
    assert q['total'] == 1980600 + 4*96*1200 == 2441400


def test_wall_clock_is_paced():
    w = rc.wall_clock()
    assert w['per_update_s'] == 1800 and w['B_s'] == 24*1800 + 4*900 and w['R_s'] == w['B_s'] + 1800


def test_runtime_config_refuses_empty_effective_slot(tmp_path):
    with pytest.raises(ValueError, match='slot is empty'):
        rc.runtime_config({}, root=tmp_path, legacy_root=tmp_path, source_root=tmp_path,
                          common_checkpoint_sha256='a'*64)
    with pytest.raises(ValueError, match='pin mismatch'):
        rc.check_pins(dict(rc.PINS, imitation_step_mode='fixed'))
    assert rc.check_pins(dict(rc.PINS))


def test_preflight_refuses_without_matching_owner_approval(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match='budget request missing'):rc.preflight_dry(tmp_path)
    req = tmp_path/'RERUN_BUDGET_REQUEST_DRAFT.json';req.write_text(json.dumps(rc.quote()))
    with pytest.raises(SystemExit, match='OWNER_APPROVAL'):rc.preflight_dry(tmp_path)
    good = dict(budget_request_sha256=hashlib.sha256(req.read_bytes()).hexdigest(), owner_approved=True,
                evening_verdict='positive', effective_imitation_commit='c'*40)
    (tmp_path/'OWNER_APPROVAL.json').write_text(json.dumps(dict(good, budget_request_sha256='0'*64)))
    with pytest.raises(SystemExit, match='budget sha differs'):rc.preflight_dry(tmp_path)
    (tmp_path/'OWNER_APPROVAL.json').write_text(json.dumps(dict(good, evening_verdict='negative')))
    with pytest.raises(SystemExit, match='not positive'):rc.preflight_dry(tmp_path)
    (tmp_path/'OWNER_APPROVAL.json').write_text(json.dumps(good))
    with pytest.raises(SystemExit, match='not pinned'):rc.preflight_dry(tmp_path)
    monkeypatch.setitem(rc.EFFECTIVE_IMITATION_SLOT, 'commit', 'c'*40)
    assert rc.preflight_dry(tmp_path)['ready_for_registered_launch'] is True
