"""harden-2 fault injection for the B/R resume path (CPU only; no SUMO, GPU or ssh).

Local faults (one query, one episode, one GPU worker, one ssh call, one SUMO branch)
must stay local: a decline, a quarantined slot or a failed decision, never a global
WARM_GUIDED_STOP, a dead worker or an episode exception. Invariant faults stay fatal.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from unittest.mock import patch

import pytest
import torch

from tools.coopmwm import capacity_broker as cb
from tools.coopmwm import sumo_verify as sv
from tools.coopmwm.protocol import digest
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_transport import SSHConnectionError, retry
from tools.coopmwm.test_sumo_verify_cpu import FakePlant, charge_log, plan, snapshot


# ------------------------------------------------------------ capacity broker harness

class Done(BaseException):
    """Harness sentinel: ends the broker control loop once the scenario resolved."""


def record(key, t=400):
    return dict(state_id=key, world_id=986001, time_s=t, rows=[], group=['a', 'b'])


def ok_result(rec):
    return dict(accepted=False, reason='no positive plan', record_sha256=digest(rec), candidates_searched=8)


class Script:
    """Per-slot fake Remote. ``on`` maps action -> callable(value, state) or exception."""

    def __init__(self, submit=None, forecast=None, status=None, verify=None):
        self.submit, self.forecast, self.status, self.verify = submit, forecast, status, verify
        self.events = [];self.submitted = []

    @staticmethod
    def alive():
        return dict(alive=True, failed=False, ready=True, receipts={})


def harness(tmp_path, scripts, records, done, kind='plan', timeout=40):
    root = tmp_path
    for name in ('requests', 'responses', 'quota_admissions', 'centre_requests', 'centre_responses'):
        (root/name).mkdir()
    (root/'AUTHORITY.md').write_text('registered')
    deadline = time.time()+600
    (root/'WARM_REMOTE_BINDING.json').write_text(json.dumps(dict(
        endpoints=[dict(slot=s) for s in scripts], deadline_unix=deadline, search_sha256='s')))
    config = root/'CONFIG.json'
    config.write_text(json.dumps(dict(amendment_sha256='registered', deadline_unix=deadline,
                                      source_root='src', receipt_tag='_T')))
    folder = 'centre_requests' if kind == 'centre' else 'requests'
    for r in records:
        immutable(root/folder/(r['state_id']+'.json'), r)
        # quota-blocking B requests: never capacity-declined, they wait for a real search
        immutable(root/'quota_admissions'/(r['state_id']+'.json'), dict(blocking=True))

    class FakeRemote:
        def __init__(self, root, endpoint):
            self.s = scripts[endpoint['slot']]

        def call(self, value):
            action = value['action']
            if action == 'status':
                return Script.alive() if self.s.status is None else self.s.status(value)
            if action == 'submit':
                self.s.submitted.append(value['record']['state_id'])
                if self.s.submit:
                    self.s.submit(value)
                return dict(key=value['record']['state_id'])
            return self.s.forecast(value)

        def verify(self):
            if self.s.verify:
                self.s.verify()
            return Script.alive()

        def event(self, e):
            self.s.events.append(e)

    original = cb.resource_endpoints

    def endpoints(root_, e, c):
        if done(root):
            raise Done()
        return original(root_, e, c)

    result = {}

    def target():
        try:
            cb.run(root, config)
        except BaseException as error:  # noqa: BLE001 - run() re-raises collected errors
            result['error'] = error
    with patch.object(cb, 'Remote', FakeRemote), patch.object(cb, 'resource_endpoints', endpoints), \
            patch.object(cb, 'assert_running'), patch.object(cb, 'owner_identity', return_value=dict(pid=1)), \
            patch.object(signal, 'SIGHUP', 1, create=True), patch.object(signal, 'signal'), \
            patch('tools.coopmwm.exploration.regime', return_value='ordinary'):
        t = threading.Thread(target=target, daemon=True);t.start();t.join(timeout)
    assert not t.is_alive(), 'broker harness did not settle'
    stops = sorted(p.name for p in root.glob('WARM_GUIDED_STOP*.json'))
    stop = json.loads((root/stops[0]).read_text()) if stops else None
    return root, stop


def only_harness_stop(stop):
    """The only STOP is the harness sentinel from the control loop (no worker STOP first)."""
    assert stop is not None and stop['source'] == 'capacity broker control' and 'Done' in stop['error'], stop


def forecasts(results):
    def forecast(value):
        r = results.get(value['key'])
        return dict(forecast=None if r is None else dict(key=value['key'], result=r, wall_s=1.),
                    status=Script.alive())
    return forecast


def test_worker_killed_mid_search_quarantines_slot_and_settles_claim(tmp_path):
    a, b = record('a'), record('b')
    dead = dict(alive=False, failed=True, ready=False,
                receipts={'SPOOL_FAILED.json': dict(error="ValueError('complete observable group required')")})
    s1 = Script(forecast=lambda v: dict(forecast=None, status=dead))
    s2 = Script(forecast=forecasts({'b': ok_result(b)}))
    root, stop = harness(tmp_path, {'gpuA@0': s1, 'gpuR@0': s2}, [a, b],
                         lambda r: (r/'responses'/'a.json').exists() and (r/'responses'/'b.json').exists())
    only_harness_stop(stop)
    settled = json.loads((root/'responses'/'a.json').read_text())
    assert settled['accepted'] is False and settled['worker_unavailable']['inference_retry'] is False
    assert settled['record_sha256'] == digest(a)
    assert len(list((root/'worker_restart_requests').glob('*.json'))) == 1


@pytest.mark.parametrize('error', [
    SSHConnectionError('Connection closed by 1.15.42.207 port 22'),
    SSHConnectionError("ssh: Could not resolve hostname connect.westb.seetacloud.com: Temporary failure in name resolution"),
    RuntimeError('Traceback (most recent call last):\nValueError: unqualified or dead GPU spool'),
    TimeoutError('bounded SSH recovery deadline; inspect receipts'),
])
def test_one_slot_transport_or_dead_spool_is_slot_local_not_global_stop(tmp_path, error):
    a, b = record('a'), record('b', t=401)
    def submit(value):
        raise error
    s1 = Script(submit=submit, forecast=forecasts({}))
    s2 = Script(forecast=forecasts({'a': ok_result(a), 'b': ok_result(b)}))
    root, stop = harness(tmp_path, {'gpuA@0': s1, 'gpuR@0': s2}, [a, b],
                         lambda r: (r/'responses'/'a.json').exists() and (r/'responses'/'b.json').exists())
    only_harness_stop(stop)
    assert any(e['event'] == 'slot_local_fault' and not e['global_stop'] for e in s1.events)
    assert (root/'worker_restart_requests').exists()


def test_invariant_fault_on_one_slot_still_stops_globally(tmp_path):
    a = record('a')
    def submit(value):
        raise RuntimeError('Traceback ...\nValueError: persistent worker batch parity failed')
    root, stop = harness(tmp_path, {'gpuA@0': Script(submit=submit, forecast=forecasts({}))}, [a],
                         lambda r: False, timeout=10)
    assert stop is not None and stop['source'] == 'capacity broker' and 'parity' in stop['error']


def test_malformed_forecast_is_a_query_decline_and_slot_keeps_serving(tmp_path):
    a, b = record('a'), record('b', t=401)
    bad = {'a': dict(accepted=True, record_sha256='not-this-record'), 'b': ok_result(b)}
    s1 = Script(forecast=forecasts(bad))
    root, stop = harness(tmp_path, {'gpuA@0': s1}, [a, b],
                         lambda r: (r/'responses'/'b.json').exists())
    only_harness_stop(stop)
    declined = json.loads((root/'responses'/'a.json').read_text())
    assert declined['accepted'] is False and declined['record_sha256'] == digest(a)
    assert declined['malformed_forecast']['inference_retry'] is False
    assert json.loads((root/'responses'/'b.json').read_text()) == ok_result(b)
    assert [e['event'] for e in s1.events] == ['malformed_forecast_declined']


def test_slow_search_is_waited_for_without_stop(tmp_path):
    a = record('a');calls = []
    def forecast(value):
        calls.append(1)
        return dict(forecast=None if len(calls) < 2 else dict(key='a', result=ok_result(a), wall_s=1107.),
                    status=Script.alive())
    root, stop = harness(tmp_path, {'gpuA@0': Script(forecast=forecast)}, [a],
                         lambda r: (r/'responses'/'a.json').exists())
    only_harness_stop(stop)
    assert len(calls) == 2


def test_all_slots_lost_writes_capacity_alert(tmp_path):
    a = record('a')
    dead = dict(alive=False, failed=True, ready=False, receipts={})
    s1 = Script(status=lambda v: dead, forecast=forecasts({}))
    root, stop = harness(tmp_path, {'gpuA@0': s1}, [a],
                         lambda r: (r/'capacity_alerts').exists() and any((r/'capacity_alerts').iterdir()))
    only_harness_stop(stop)
    alert = json.loads(next((root/'capacity_alerts').iterdir()).read_text())
    assert alert['live_slots'] == [] and alert['quarantined_slots'] == ['gpuA@0']
    assert not (root/'responses'/'a.json').exists()            # blocking request still waits for a real search


def test_dead_slot_settles_centre_claim_so_R_never_waits_forever(tmp_path):
    c = dict(record('c'), commands={})
    dead = dict(alive=False, failed=True, ready=False, receipts={})
    s1 = Script(forecast=lambda v: dict(forecast=None, status=dead))
    root, stop = harness(tmp_path, {'gpuA@0': s1}, [c], lambda r: (r/'centre_responses'/'c.json').exists(),
                         kind='centre')
    only_harness_stop(stop)
    v = json.loads((root/'centre_responses'/'c.json').read_text())
    assert v['unavailable'] and v['candidate_scores'] == 0 and v['record_sha256'] == digest(c)
    from tools.coopmwm.centre_client import trajectory
    client = tmp_path/'client';(client/'centre_responses').mkdir(parents=True)
    (client/'centre_responses'/'c.json').write_text(json.dumps(v))
    with patch('tools.coopmwm.warm_health.assert_running'):
        assert trajectory(client, c, time.time()+10)[0] is None


# ------------------------------------------------------------ transport / spool / RPC

def test_dns_failure_is_retried_with_backoff_then_succeeds():
    attempts, events, sleeps = [], [], []
    def op():
        attempts.append(1)
        if len(attempts) < 3:
            raise SSHConnectionError('ssh: Could not resolve hostname connect.westb.seetacloud.com: '
                                     'Temporary failure in name resolution')
        return 'ok'
    assert retry(op, lambda: None, events.append, sleep=sleeps.append) == 'ok'
    assert len(events) == 2 and sleeps == [10, 20] and all(e['new_sumo_sim_s'] == 0 for e in events)


def test_truncated_rpc_stdout_is_a_retryable_transport_fault():
    from tools.coopmwm.warm_broker import Remote
    remote = Remote('.', dict(slot='s', known_hosts='k', identity_file='i', port=1, destination='d'))
    done = subprocess.CompletedProcess([], 0, stdout='{"forecast": nul', stderr='')
    with patch('subprocess.run', return_value=done), pytest.raises(SSHConnectionError):
        remote.raw(dict(action='status'))


def test_stale_publishing_file_from_killed_writer_never_blocks_destination(tmp_path):
    target = tmp_path/'responses'/'a.json';target.parent.mkdir()
    (tmp_path/'responses'/'a.json.publishing').write_bytes(b'{"partial')  # old fixed-name temporary
    assert immutable(target, dict(x=1)) is True
    assert immutable(target, dict(x=1)) is False
    with pytest.raises(ValueError):
        immutable(target, dict(x=2))
    assert sorted(p.name for p in target.parent.iterdir()) == ['a.json', 'a.json.publishing']


def test_full_disk_during_publication_leaves_no_partial_target(tmp_path):
    target = tmp_path/'a.json'
    with patch('os.fsync', side_effect=OSError(28, 'No space left on device')), pytest.raises(OSError):
        immutable(target, dict(x=1))
    assert list(tmp_path.iterdir()) == []
    assert immutable(target, dict(x=1)) is True


# ------------------------------------------------------------ GPU per-query isolation

@pytest.mark.parametrize('message', ['group of 2..6 CAVs required', 'complete observable group of two or three required',
                                     'history encoder requires genuinely uniform observed ticks',
                                     'extra seed must be members x canonical knots with a registered duration'])
def test_per_query_value_error_is_a_decline_not_worker_death(message):
    from tools.coopmwm.query_errors import execute, validate_response
    rec = dict(state_id='q')
    def fault():
        raise ValueError(message)
    v = execute(rec, fault)
    assert v['accepted'] is False and v['query_error']['kind'] == 'code'
    validate_response(rec, v)  # accepted by already-deployed brokers


def test_learned_search_kwargs_against_plain_search_are_a_declined_type_error():
    """Hot-swap hazard: this tree's mppi_search.search has no ``warm_start``; a learned
    kwargs call is a TypeError -> EVERY query silently declined. resume_preflight flags it."""
    import inspect
    from tools.coopmwm import mppi_search
    params = inspect.signature(mppi_search.search).parameters
    problems = sv.resume_preflight(dict(k=8, workers=9, phase='sumo_verification_k8'),
                                   search_config=dict(top_k=8), learned=True)
    if 'warm_start' not in params:
        assert any('warm_start' in p for p in problems)
    else:
        assert not any('warm_start' in p for p in problems)


# ------------------------------------------------------------ SUMO verification (k=8)

def eight():
    return [plan(c, gain=g) for c, g in zip('abcdefgh', (1., 2., .5, -1., 3., 5., 0., 4.))]


def test_k8_receipt_schema_nested_best_of_4_vs_8(tmp_path):
    log, charge = charge_log()
    r = sv.verify(snapshot(), eight(), FakePlant(), charge, root=tmp_path)
    assert r['k'] == 8 and r['n_branches'] == 9 and len(r['J_candidates_veh_s']) == 8
    assert [b['branch'] for b in r['branches']] == ['policy']+[f'cand{i}' for i in range(1, 9)]
    assert r['chosen_plan']['id'] == 'f' and r['grade_veh_s'] == 50.
    assert r['charged_sim_s'] == 9*r['branch_sim_s'] == sum(log[0][1])
    best4 = r['J_policy_veh_s']-min(r['J_candidates_veh_s'][:4])
    best8 = r['J_policy_veh_s']-min(r['J_candidates_veh_s'])
    assert (best4, best8) == (20., 50.)                      # rank 5-8 winner is visible per decision


def test_k8_smoke_gate_and_grade_reuse_accept_nine_branches(tmp_path):
    from tools.coopmwm import verification_grades as vg
    r = sv.verify(snapshot(), eight(), FakePlant(), charge_log()[1], root=tmp_path)
    with patch.object(sv, 'check_live_continuation', return_value=dict(pass_=True)):
        with pytest.raises(ValueError, match='4 candidates'):
            vg.record_smoke(tmp_path, r, {})
        with pytest.raises(ValueError, match='CRN'):  # fake plant has no CRN audit; k now accepted
            vg.record_smoke(tmp_path, r, {}, k=8)


def test_one_crashed_candidate_branch_picks_best_graded_and_records_missing():
    # MAIN (harden-int): policy graded -> best among graded candidates; missing recorded.
    r = sv.verify(snapshot(), eight(), FakePlant(fail={'cand6'}), charge_log()[1])
    assert r['status'] == 'accepted' and r['chosen_plan']['id'] == 'h' and r['grade_veh_s'] == 40.
    assert r['missing_candidates'] == ['cand6'] and r['graded_candidates'] == 7 and r['available_k'] == 8
    assert r['J_candidates_veh_s'][5] is None


def test_crashed_policy_branch_voids_the_decision_never_the_episode():
    r = sv.verify(snapshot(), eight(), FakePlant(fail={'policy'}), charge_log()[1])
    assert r['status'] == 'failed' and r['chosen_plan'] is None and r['J_candidates_veh_s'] is None


def test_full_disk_snapshot_prepare_fails_decision_not_episode(tmp_path):
    class Plant(FakePlant):
        def prepare(self, specs, snap):
            raise OSError(28, 'No space left on device')
    log, charge = charge_log()
    r = sv.verify(snapshot(), eight(), Plant(), charge, root=tmp_path)
    assert r['status'] == 'failed' and 'prepare failed' in r['reason'] and r['actual_sim_s'] == 0
    assert r['charged_sim_s'] == sum(log[0][1])               # precharged, never refunded or replayed


def test_disk_guard_blocks_before_any_charge(tmp_path):
    runner = sv.SubprocessBranchRunner(tmp_path/'branches', workers=9)
    log, charge = charge_log()
    with patch('shutil.disk_usage', return_value=type('U', (), dict(free=1024**3))()):
        r = sv.verify(snapshot(), eight(), runner, charge, root=tmp_path)
    assert r['status'] == 'failed' and r['reason'].startswith('resource guard') and log == []
    assert r['charged_sim_s'] == 0


def test_random_candidates_terminate_when_draws_collide():
    with patch('tools.coopmwm.mppi_search.sample_unscored', return_value=dict(id='same')):
        out = sv.random_candidates(dict(members=['a', 'b']), [[0.]*300]*2, (-3., 2.), 'ordinary', 7, k=8)
    assert [p['id'] for p in out] == ['same']


def test_B_with_plain_top4_under_k8_records_shortfall():
    class Ctl:
        beta = 1.;events = [];memory = {};active_plans = []
        policy = torch.nn.Linear(1, 1)
    topk = [plan(c, gain=1.) for c in 'abcd']
    proposal = dict(accepted=True, plan=topk[0], topk=topk, option='MPPI',
                    behaviour=dict(regime='ordinary', sample_probability=.95))
    seen = {}
    def verifier(snap, cands, proposer):
        seen['k'] = len(cands)
        return dict(verification_id='v', status='rejected', receipt_sha256='h', chosen=None, grade_veh_s=None,
                    J_policy_veh_s=1., J_candidates_veh_s=[1.]*len(cands), charged_sim_s=0, timing=dict(wall_s=0.))
    job = dict(arm='B', sumo_verify=dict(guide_beta=1., k=8))
    with patch.object(sv, 'capture', return_value={}):
        call = sv.verified_guide(lambda *a: proposal, verifier, Ctl(), {}, job, torch.Generator(), k=8)
        out = call([], {}, 400, {})
    assert seen['k'] == 4 and out['behaviour']['verify_k'] == dict(requested=8, available=4)


def test_resume_preflight_flags_k8_hazards():
    assert sv.resume_preflight(dict(k=4, workers=5, phase='sumo_verification')) == []
    problems = sv.resume_preflight(dict(k=8, workers=5, phase='sumo_verification'), search_config=None)
    text = ' | '.join(problems)
    assert 'top_k' in text and 'workers' in text and 'sumo_verification_k8' in text


# ------------------------------------------------------------ episode side
# (parallel-group early return / batch ordering / withdrawal: harden-3 07dd3cd7 and its tests)

def test_R_window_with_unavailable_centre_declines_and_releases_its_ticket(tmp_path):
    from tools.coopmwm import plan_client as client
    quota = tmp_path/'B_RANDOM_QUOTA_004.json'
    quota.write_text(json.dumps(dict(sampling=dict(ordinary=1.), tickets=[dict(id=0, regime='ordinary',
                                                                                 group_size=2, duration_s=60)])))
    job = dict(queue_root=str(tmp_path), accel_bounds=[-3., 2.], arm='R', random_quota=str(quota), update=4,
               id='warm_u004_R_986000', world=dict(seed=986000), gpu_deadline=time.time()+10)
    call = client.guide(job, torch.Generator().manual_seed(1))
    sample = dict(context=dict(members=['a', 'b']), own_action=[[0.], [0.]])
    with patch.object(client, 'group_components', return_value=[dict(members=['a', 'b'], size=2)]), \
            patch.object(client, 'regime', return_value='ordinary'), \
            patch('tools.coopmwm.centre_client.trajectory', return_value=(None, dict(unavailable=True))):
        out = call([], {}, 400, sample)
    assert out['accepted'] is False and out['reason'].startswith('R centre unavailable')
    call.completed(out, False)
    from tools.coopmwm.plan_client import used_tickets
    assert used_tickets(tmp_path/'random_match'/'u004') == set()


def test_hotadd_replacement_uses_its_own_receipt_label():
    from tools.coopmwm.hotadd import receipt_label
    assert receipt_label(dict(host='gpuA')) == 'gpuA'                      # original names unchanged
    assert receipt_label(dict(host='gpuA', receipt_label='gpuA_r2')) == 'gpuA_r2'
    with pytest.raises(ValueError):
        receipt_label(dict(host='gpuA', receipt_label='../x'))
