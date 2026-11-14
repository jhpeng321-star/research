"""CPU tests for the cpuS plan-checking glue (no SUMO, no ssh): landing plan, remote-reserve contract,
RPC server ops on a temporary run root."""
import io
import json
import sys
from pathlib import Path

import pytest

from tools.coopmwm import cpus_checking as cc
from tools.coopmwm import cpus_rpc


def _touch(p, data=b'x'):
    p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(data);return p


def test_push_plan_excludes_scenarios_and_returns_record_last(tmp_path):
    bank = tmp_path/'replay_bank'/'B'
    rec = dict(verification_id='sv_x', charges=[dict(kind='confirm_seed2_all8', verification_id='v2'),
                                               dict(kind='confirm_solo_all8_s1', verification_id='v3'),
                                               dict(kind='confirm_seed3_all8', verification_id='v4')])
    keep = [_touch(bank/'confirm_seed2_all8'/'claims'/'v2.json'),
            _touch(bank/'confirm_seed2_all8'/'verifications'/'v2.json'),
            _touch(bank/'confirm_seed2_all8'/'branches'/'v2'/'policy'/'RESULT.json'),
            _touch(bank/'confirm_seed2_all8'/'branches'/'v2'/'policy'/'observable.jsonl.gz'),
            _touch(bank/'confirm_seed2_all8'/'branches'/'v2'/'cand1'/'SLIM.json'),
            _touch(bank/'confirm_seed2_all8'/'branches'/'v2'/'SNAPSHOT.pt'),
            _touch(bank/'confirm_solo_all8_s1'/'verifications'/'v3.json'),
            _touch(bank/'confirm_seed3_all8'/'alerts'/'BUDGET_BLOCKED_v4.json')]
    _touch(bank/'confirm_seed2_all8'/'branches'/'v2'/'cand1'/'scenarios'/'s1'/'requests.rou.xml')
    _touch(bank/'confirm_seed2_all8'/'branches'/'WORKERS.jsonl')
    _touch(bank/'confirm_seed2_all8'/'claims'/'other.json')
    files, record = cc.push_plan(bank, rec)
    assert sorted(files) == sorted(keep)
    assert record == bank/'confirmations_all8'/'sv_x.json'
    assert not any('scenarios' in p.parts for p in files)


def test_remote_reserve_contract(monkeypatch):
    cfg = dict(run_root='/r')
    calls = []

    def fake(cfg_, op, payload=None, timeout=600):
        calls.append(op)
        if op == 'reserve':
            if payload['jobs'][0]['id'] == 'blocked':
                return dict(ok=False, error='whole-wave budget blocked')
            return dict(ok=True, rows=[dict(id=j['id'], phase=payload['phase'], sim_s=j['sim_s'])
                                       for j in payload['jobs']])
        raise AssertionError(op)
    monkeypatch.setattr(cc, 'rpc', fake)
    reserve = cc.remote_reserve(cfg)
    reserve(Path('/r'), [dict(id='confirm_seed2_all8_a_policy', sim_s=510)], 'bank_verification_ext')
    assert reserve.rows['confirm_seed2_all8_a_policy']['phase'] == 'bank_verification_ext'
    with pytest.raises(ValueError, match='budget blocked'):
        reserve(Path('/r'), [dict(id='blocked', sim_s=1)], 'bank_verification_ext')
    with pytest.raises(AssertionError):
        reserve(Path('/other'), [dict(id='x', sim_s=1)], 'bank_verification_ext')


def test_remote_reserve_transport_failure_resolves_from_ledger(monkeypatch):
    cfg = dict(run_root='/r')
    state = dict(charged=True)

    def fake(cfg_, op, payload=None, timeout=600):
        if op == 'reserve':
            raise cc.RpcError('ssh died')
        if op == 'ledger':
            return dict(ok=True, rows=[dict(id='confirm_seed2_all8_a_policy', phase='p', sim_s=5)]
                        if state['charged'] else [])
        raise AssertionError(op)
    monkeypatch.setattr(cc, 'rpc', fake)
    reserve = cc.remote_reserve(cfg)
    reserve(Path('/r'), [dict(id='confirm_seed2_all8_a_policy', sim_s=5)], 'p')     # charged -> success
    assert 'confirm_seed2_all8_a_policy' in reserve.rows
    state['charged'] = False
    with pytest.raises(ValueError, match='nothing charged'):
        reserve(Path('/r'), [dict(id='confirm_seed2_all8_b_policy', sim_s=5)], 'p')


def test_all8_regex_covers_every_all8_kind_and_no_legacy_kind():
    ok = ['confirm_seed2_all8_sv_a_policy', 'confirm_extended_all8_sv_a_cand1', 'confirm_extended_seed2_all8_x_p',
          'confirm_solo_all8_s1_x_p', 'confirm_solo_all8_s2_x_p', 'confirm_seed3_all8_x_p',
          'confirm_extended_seed6_all8_x_p', 'confirm_solo_all8_s5_x_p']
    no = ['confirm_replication_sv_a_policy', 'confirm_extended_sv_a_cand1', 'confirm_solo_sv_x_p',
          'confirm_solo_seed2_1755_x', 'coopfix_exact_coop8_u000_B_x', 'policy_synergy_u019_R_x']
    assert all(cpus_rpc.ALL8.match(x) for x in ok)
    assert not any(cpus_rpc.ALL8.match(x) for x in no)


def test_rpc_server_read_ops_and_path_guard(tmp_path, monkeypatch):
    root = tmp_path/'run';bank = root/'replay_bank'/'B'
    f = _touch(bank/'confirm_seed2_all8'/'verifications'/'v.json', b'abc')
    control = tmp_path/'ctl';control.mkdir()
    (control/'RPC_BINDING.json').write_text(json.dumps(dict(run_root=str(root), phase='bank_verification_ext',
                                                            delegate_cap_sim_s=10, delegate_locks=[])))

    def call(op, payload):
        monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(payload)))
        out = io.StringIO();monkeypatch.setattr(sys, 'stdout', out)
        cpus_rpc.main([str(control), op])
        return json.loads(out.getvalue())
    assert call('absent', dict(paths=[str(f), str(bank/'nope.json')]))['present'] == [str(f)]
    import hashlib
    good = dict(path=str(f), sha256=hashlib.sha256(b'abc').hexdigest(), size=3)
    assert call('verify', dict(files=[good]))['bad'] == []
    assert call('verify', dict(files=[dict(good, sha256='0'*64)]))['bad'][0]['problem'] == 'sha256'
    v = call('absent', dict(paths=[str(tmp_path/'elsewhere.json')]))
    assert v['ok'] is False and 'outside' in v['error']
    v = call('absent', dict(paths=[str(bank/'sealed'/'x.json')]))
    assert v['ok'] is False
    v = call('reserve', dict(jobs=[dict(id='confirm_replication_x', sim_s=1)], phase='bank_verification_ext'))
    assert v['ok'] is False and 'all8-kind' in v['error']
    v = call('reserve', dict(jobs=[dict(id='confirm_seed2_all8_x', sim_s=1)], phase='train'))
    assert v['ok'] is False and 'registered' in v['error']


def test_minimal_plan_lands_only_training_read_files(tmp_path):
    bank = tmp_path/'replay_bank'/'B'
    s2 = bank/'confirm_seed2_all8'
    (s2/'verifications').mkdir(parents=True)
    (s2/'verifications'/'v2.json').write_text(json.dumps(dict(branches=[dict(branch='policy'), dict(branch='cand1'),
                                                                     dict(branch='cand2')])))
    for b in ('policy', 'cand1', 'cand2'):
        for n in ('RESULT.json', 'RESULT.json.gz', 'observable.jsonl.gz', 'CLAIM.json'):
            _touch(s2/'branches'/'v2'/b/n)
    _touch(s2/'branches'/'v2'/'cand2'/'SLIM.json')
    (s2/'branches'/'v2'/'cand2'/'RESULT.json').unlink();(s2/'branches'/'v2'/'cand2'/'RESULT.json.gz').unlink()
    _touch(s2/'branches'/'v2'/'cand1.result.json');_touch(s2/'branches'/'v2'/'SNAPSHOT.pt')
    _touch(s2/'claims'/'v2.json')
    solo = bank/'confirm_solo_all8_s2'
    _touch(solo/'verifications'/'vs.json')
    for b in ('policy', 'cand1'):
        for n in ('RESULT.json', 'RESULT.json.gz', 'observable.jsonl.gz'):
            _touch(solo/'branches'/'vs'/b/n)
    rec = dict(verification_id='sv_x', charges=[dict(kind='confirm_seed2_all8', verification_id='v2'),
                                               dict(kind='confirm_solo_all8_s2', verification_id='vs')],
               per_seed_v1=dict(examples=[
                   dict(admitted=True, seed_index=2, solo_verification_id='vs',
                        tape=dict(branches_dir=str(s2/'branches'), verification_id='v2', branch=dict(branch='cand1'))),
                   dict(admitted=False, seed_index=2, status='rejected')]))
    land, keep, record = cc.minimal_plan(bank, rec)
    rel = sorted(p.relative_to(bank).as_posix() for p in land)
    assert rel == sorted([
        'confirm_seed2_all8/verifications/v2.json', 'confirm_solo_all8_s2/verifications/vs.json',
        'confirm_seed2_all8/branches/v2/cand1/RESULT.json', 'confirm_seed2_all8/branches/v2/cand1/RESULT.json.gz',
        'confirm_seed2_all8/branches/v2/cand1/observable.jsonl.gz',
        'confirm_seed2_all8/branches/v2/policy/RESULT.json', 'confirm_seed2_all8/branches/v2/policy/RESULT.json.gz',
        'confirm_seed2_all8/branches/v2/cand2/SLIM.json',
        'confirm_solo_all8_s2/branches/vs/policy/RESULT.json', 'confirm_solo_all8_s2/branches/vs/policy/RESULT.json.gz',
        'confirm_solo_all8_s2/branches/vs/cand1/RESULT.json', 'confirm_solo_all8_s2/branches/vs/cand1/RESULT.json.gz'])
    kept = {p.relative_to(bank).as_posix() for p in keep}
    assert 'confirm_seed2_all8/branches/v2/cand2/observable.jsonl.gz' in kept
    assert 'confirm_seed2_all8/branches/v2/policy/observable.jsonl.gz' in kept
    assert 'confirm_seed2_all8/branches/v2/cand1.result.json' in kept and 'confirm_seed2_all8/claims/v2.json' in kept
    assert not set(land) & set(keep)
    assert record == bank/'confirmations_all8'/'sv_x.json'
