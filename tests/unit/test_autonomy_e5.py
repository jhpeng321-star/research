"""E5 full-world pairing and sign of the registered coopON-minus-coopOFF contrast."""
import json
import subprocess

import pytest

from tools.autonomy_e5 import remote as e5


def fixture_rows(box, *, mismatch=False, omit=False):
    rows = {}
    for label in ('on', 'off'):
        own = []
        for seed in range(973000, 973228):
            if (seed-973000) % 2 != (0 if box == 'cpuP' else 1):
                continue
            if omit and seed == 973000:
                continue
            value = 7 if label == 'on' else 5
            metrics = {key: value for key in ('team_delay_vs', 'road_delay_vs', 'source_wait_vs',
                       'throughput_vph', 'mainline_delay_vs', 'ramp_delay_vs', 'collisions', 'teleports')}
            metrics.update(road_delay_vs=value/2, source_wait_vs=value/2,
                           mainline_delay_vs=value/2, ramp_delay_vs=value/2, dropped_vehicle_steps=0)
            own.append({'ok': True, 'job': {'seed': seed, 'cell': (seed-973000) % 12},
                        'metrics': {'initial_state_sha256': 'different' if mismatch and label == 'off' else str(seed),
                                    'windows': {str(h): {**metrics, 't0_s': 300, 't_end_s': 300+h}
                                                for h in (120, 300, 600)}}})
        rows[label] = own
    return {**rows, 'charged_sim_s': 102600}


def test_contrast_direction_and_world_cluster_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setattr(e5, 'R', tmp_path)
    calls = []

    def ssh(box, source, timeout=120):
        calls.append(box)
        return fixture_rows(box) if 'off=json.loads' in source else {}

    monkeypatch.setattr(e5, 'ssh', ssh)
    e5.analyze()
    rows = json.loads((tmp_path/'RESULTS.json').read_text())
    assert len(rows) == 24
    assert all(r['mean'] == (1 if r['metric'] in ('road_delay_vs', 'source_wait_vs',
               'mainline_delay_vs', 'ramp_delay_vs') else 2) and r['ci95'] == [r['mean'], r['mean']]
               and r['n_worlds'] == 228 for r in rows)
    assert {r['horizon_s'] for r in rows} == {120, 300, 600}
    assert calls == ['cpuP', 'cpuQ', 'cpuP', 'cpuQ']


@pytest.mark.parametrize('mismatch,omit', [(True, False), (False, True)])
def test_invalid_pair_or_incomplete_panel_produces_no_result(tmp_path, monkeypatch, mismatch, omit):
    monkeypatch.setattr(e5, 'R', tmp_path)
    monkeypatch.setattr(e5, 'ssh', lambda box, source, timeout=120: fixture_rows(box, mismatch=mismatch, omit=omit))
    with pytest.raises(AssertionError):
        e5.analyze()
    assert not (tmp_path/'RESULTS.json').exists()


def test_detached_chain_verifies_canary_before_main_and_analyzes(tmp_path, monkeypatch):
    monkeypatch.setattr(e5, 'R', tmp_path)
    events = []
    states = iter(({}, {}, {'status': 'canary_complete'},
                   {'status': 'complete'}, {'status': 'complete'}))
    monkeypatch.setattr(e5, 'heartbeat', lambda box: next(states))
    monkeypatch.setattr(e5, 'pull', lambda box: events.append(('mirror', box)))

    def start(box, shard, canary=False):
        events.append(('start', box, canary))
        return {'pid': 100 + shard, 'sid': 100 + shard}

    def ssh(box, source, timeout=120):
        events.append(('verify_canary', box))
        return {'passed': True, 'canary_seed': 973000, 'charged_sim_s': 900}

    monkeypatch.setattr(e5, 'start', start)
    monkeypatch.setattr(e5, 'ssh', ssh)
    monkeypatch.setattr(e5, 'analyze', lambda: events.append(('analyze',)))
    e5.drive()
    assert events == [('start', 'cpuP', True), ('mirror', 'cpuP'),
                      ('verify_canary', 'cpuP'), ('verify_canary', 'cpuQ'),
                      ('start', 'cpuP', False), ('start', 'cpuQ', False),
                      ('mirror', 'cpuP'), ('mirror', 'cpuQ'), ('analyze',)]
    assert json.loads((tmp_path/'STATUS.json').read_text())['charged_sim_s'] == 205200


def test_failed_canary_never_launches_remainder(tmp_path, monkeypatch):
    monkeypatch.setattr(e5, 'R', tmp_path)
    states = iter(({}, {}, {'status': 'failed'}))
    starts = []
    monkeypatch.setattr(e5, 'heartbeat', lambda box: next(states))
    monkeypatch.setattr(e5, 'pull', lambda box: None)
    monkeypatch.setattr(e5, 'start', lambda box, shard, canary=False: starts.append(canary) or {})
    with pytest.raises(RuntimeError, match='canary failed'):
        e5.drive()
    assert starts == [True]
    assert json.loads((tmp_path/'STATUS.json').read_text())['status'] == 'WAIT_OWNER'


def test_ssh_accepts_multiline_runner_status(monkeypatch):
    doc = {'status': 'canary_complete', 'pid': 123}
    monkeypatch.setattr(e5.subprocess, 'run', lambda *a, **k:
                        subprocess.CompletedProcess(a[0], 0, json.dumps(doc, indent=2)+'\n', ''))
    assert e5.ssh('cpuP', 'read status') == doc


def test_resume_reuses_completed_canary(tmp_path, monkeypatch):
    monkeypatch.setattr(e5, 'R', tmp_path)
    (tmp_path/'CANARY_PID.json').write_text('{}')
    states = iter(({}, {}, {'status': 'canary_complete'},
                   {'status': 'complete'}, {'status': 'complete'}))
    starts = []
    monkeypatch.setattr(e5, 'heartbeat', lambda box: next(states))
    monkeypatch.setattr(e5, 'pull', lambda box: None)
    monkeypatch.setattr(e5, 'ssh', lambda *a, **k: {'passed': True})
    monkeypatch.setattr(e5, 'start', lambda box, shard, canary=False: starts.append(canary) or {})
    monkeypatch.setattr(e5, 'analyze', lambda: None)
    e5.drive()
    assert starts == [False, False]
