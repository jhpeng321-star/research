"""Receipt-integrity and passive fault handling; no model or simulation imports."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean3_follow as follow
import clean3_follow_remote as remote


def seal(value, key='receipt_sha256'):
    return dict(value, **{key: remote.digest(value)})


def put(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def launched(tmp_path):
    root, control = tmp_path/'root', tmp_path/'control'
    root.mkdir(); control.mkdir()
    cfg = dict(clean3_budget=dict(owner_cap_sim_s=7300000, protected_prior_sim_s=3611840,
        phases=dict(track2_smoke=24000), g4_cap_sim_s=230400))
    put(root/'CONFIG.json', cfg)
    smoke = seal(dict(status='SMOKE_PASS'))
    gate = seal(dict(status='PASS', smoke_sha256=smoke['receipt_sha256'], safety=dict(collisions=0, teleports=0)))
    put(root/'SMOKE_PASS.json', smoke); put(root/'GATE_PASS.json', gate)
    launch = dict(root=str(root), config_file_sha256=remote.hashlib.sha256((root/'CONFIG.json').read_bytes()).hexdigest(),
        runtime_sha256=remote.digest(cfg), smoke_sha256=smoke['receipt_sha256'], gate_sha256=gate['receipt_sha256'],
        owners=dict(A=dict(pid=99999999, start_ticks='0')))
    put(root/'LAUNCHED.json', launch)
    (root/'CHARGES.jsonl').write_text(json.dumps(dict(phase='track2_smoke', sim_s=9600))+'\n')
    return root, control, cfg, launch


def test_tampered_config_and_gate_fail_closed(launched):
    root, control, cfg, launch = launched
    put(root/'CONFIG.json', dict(cfg, unregistered=True))
    with pytest.raises(ValueError, match='binding mismatch'):
        remote.probe(root, control)
    put(root/'CONFIG.json', cfg)
    gate = json.loads((root/'GATE_PASS.json').read_text())
    gate['safety']['collisions'] = 1
    put(root/'GATE_PASS.json', gate)
    with pytest.raises(ValueError, match='seal mismatch'):
        remote.probe(root, control)


def test_faults_and_exited_owner_are_observed_without_writes(launched):
    root, control, cfg, launch = launched
    put(root/'STOP.json', dict(reason='owner pause'))
    before = {p: p.read_bytes() for p in root.iterdir()}
    value = remote.probe(root, control)
    assert value['faults']['STOP.json']['reason'] == 'owner pause'
    assert value['faults']['OWNER_A']['pid'] == 99999999
    assert value['protected_total_sim_s'] == 3621440
    assert before == {p: p.read_bytes() for p in root.iterdir()}


def test_sealed_monitor_safety_is_a_fault_even_without_stop(launched):
    root, control, cfg, launch = launched
    monitor = dict(config_sha256=launch['runtime_sha256'], arms=[dict(arm='B',
        metrics=dict(safety=dict(collisions=1, teleports=0)))])
    put(root/'LEARNING_MONITOR_u001.json', seal(monitor))
    value = remote.probe(root, control)
    assert 'SAFETY_u001' in value['faults']
    monitor['arms'][0]['metrics']['safety']['collisions'] = 0
    put(root/'LEARNING_MONITOR_u001.json', monitor | {'receipt_sha256': 'tampered'})
    with pytest.raises(ValueError, match='seal mismatch'):
        remote.probe(root, control)


def test_only_exact_registered_sha_released_pause_is_ignored(launched):
    root, control, cfg, launch = launched
    pause = root/'PAUSE_SAFETY_COMPOSITE_B_u009.json'
    put(pause, dict(reason='historical clean1 B9 composite finding retained'))
    rule = seal(dict(arm='B', authority='MAIN2026-11-05 14:34', composite_action='report_only',
        collision_teleport_action='pause_on_any', original_pause_sha256=remote.hashlib.sha256(pause.read_bytes()).hexdigest(),
        amendment_sha256='a'*64))
    put(root/'B_SAFETY_REPORT_ONLY_1434.json', rule)
    (root/'AUTHORITY.md').write_text('owner clean3 inherited safety amendmentSHA'+'a'*64)
    assert pause.name not in remote.probe(root, control)['faults']
    put(pause, dict(reason='different new safety finding'))
    assert pause.name in remote.probe(root, control)['faults']
    put(root/'PAUSE_NEW.json', dict(reason='new unknown safety finding'))
    assert 'PAUSE_NEW.json' in remote.probe(root, control)['faults']


def test_summary_keeps_both_diagnostic_quantities_and_five_lines():
    snapshot = dict(monitors={'6': dict(arms=[dict(arm=a, action='CONTINUE', diagnose=[], metrics=dict(
        bank_move_ce=0.2, layer_kl=0.01, safety=dict(collisions=0, teleports=0))) for a in ('A', 'B', 'R')])},
        mined={}, diagnostics={'6': dict(metric_reports=dict(delta_all_new_veh_s=dict(arm_means=dict(B=10)),
            policy_cooperation_gain_veh_s=dict(arm_means=dict(B=-2))))}, charged_sim_s=100,
        protected_total_sim_s=200, cap_sim_s=7300000)
    lines = follow.summary(snapshot, 6)
    assert len(lines) == 5
    assert 'total vs common base' in lines[3] and 'group-over-best-single synergy' in lines[3]
    assert '"B": 10' in lines[3] and '"B": -2' in lines[3]
