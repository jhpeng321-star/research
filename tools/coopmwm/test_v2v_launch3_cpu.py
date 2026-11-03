"""CPU tests for the launch3 gate: commit/gate/HOLD refusals, config identity,
budget arithmetic. Pure stdlib; no SUMO, no GPU, no network, new_sim_s=0."""
import json
from pathlib import Path

from . import v2v_launch3 as l3
from .v2v_launch3 import Refused


def _gate(path, verdict='PASS'):
    path.write_text(json.dumps(dict(verdict=verdict, unix=1.0)), encoding='utf-8')
    return path


def _dirs(tmp):
    control = tmp / 'control'
    root = tmp / 'root'
    mover = tmp / 'mover'
    for d in (control, root, mover):
        d.mkdir()
    mover.joinpath('OWNER.json').write_text(json.dumps(
        dict(pid=1, start_ticks='x', boot_id='b')), encoding='utf-8')
    return control, root, mover


def test_refuses_without_pass_gate(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        control, root, mover = _dirs(tmp)
        missing = control / l3.GATE_NAME
        assert not missing.exists()
        for gate in (missing, _gate(control / 'g.json', 'FAIL'),
                     _gate(control / 'g.json', 'pass'), _gate(control / 'g.json', None)):
            try:
                l3.preflight(l3.MERGED_COMMIT, gate, control, root, mover)
                raise AssertionError('preflight accepted a non-PASS gate')
            except Refused:
                pass


def test_refuses_on_hold_file():
    import tempfile
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        control, root, mover = _dirs(tmp)
        gate = _gate(control / l3.GATE_NAME)
        (control / l3.HOLD_NAME).write_text('{"reason":"operator"}', encoding='utf-8')
        try:
            l3.preflight(l3.MERGED_COMMIT, gate, control, root, mover)
            raise AssertionError('preflight ignored the control HOLD file')
        except Refused as error:
            assert l3.HOLD_NAME in str(error)
        (control / l3.HOLD_NAME).unlink()
        (root / l3.HOLD_NAME).write_text('{}', encoding='utf-8')
        try:
            l3.preflight(l3.MERGED_COMMIT, gate, control, root, mover)
            raise AssertionError('preflight ignored the root HOLD file')
        except Refused:
            pass


def test_arm_config_identity():
    configs = l3.build_launch3_configs()
    binding = l3.check_config_binding(configs)
    assert binding['hashes']['B-v2v'] == binding['hashes']['R-v2v']
    assert binding['hashes']['A-v2v'] != binding['hashes']['B-v2v']
    for arm, cfg in configs.items():
        assert cfg['v2v_intent'] is True and cfg['v2v_intent_horizon'] == 5
        assert cfg['v2v_intent_weight'] == .1 and cfg['v2v'] is True
        assert cfg['schedule'] == json.loads(json.dumps(l3.SCHEDULE))
    for key in l3.BANK_OPTIONS:
        assert key not in configs['A-v2v']
        assert configs['B-v2v'][key] == configs['R-v2v'][key] == l3.BANK_OPTIONS[key]
    broken = l3.build_launch3_configs()
    broken['B-v2v']['imitation_kl_cap'] = .044
    try:
        l3.check_config_binding(broken)
        raise AssertionError('unequal B/R configs accepted')
    except Refused:
        pass
    intent_off = l3.build_launch3_configs()
    intent_off['A-v2v']['v2v_intent'] = False
    try:
        l3.check_config_binding(intent_off)
        raise AssertionError('intent-off config accepted')
    except Refused:
        pass


def test_budget_arithmetic():
    full = l3.SUPERSEDED_CHARGE + 7562280 + 89119
    assert l3.requote(full)['charged_to_date_sim_s'] == full
    quote = l3.requote(0)
    assert quote['quote_total_sim_s'] == 2971840
    assert quote['worst_case_compute_bound_sim_s'] == 2971840 + l3.V2V_PHASES['v2v_train']
    assert quote['fits'] and quote['shortfall_sim_s'] == 0
    assert quote['worst_case_shortfall_sim_s'] == 0
    quote = l3.requote(full)
    assert quote['remaining_headroom_sim_s'] == l3.CEILING_SIM_S - full
    assert quote['fits'] and quote['shortfall_sim_s'] == 0
    assert quote['worst_case_shortfall_sim_s'] == 0
    edge = l3.CEILING_SIM_S - quote['quote_total_sim_s']
    assert l3.requote(edge)['fits'] and l3.requote(edge)['shortfall_sim_s'] == 0
    over = l3.requote(edge + 1)
    assert not over['fits'] and over['shortfall_sim_s'] == 1
    assert l3.requote(l3.CEILING_SIM_S)['shortfall_sim_s'] == 2971840
    edge_worst = l3.CEILING_SIM_S - quote['worst_case_compute_bound_sim_s']
    assert l3.requote(edge_worst)['worst_case_shortfall_sim_s'] == 0
    assert l3.requote(edge_worst + 1)['worst_case_shortfall_sim_s'] == 1
    assert l3.requote(edge_worst + 1)['fits']
    assert l3.requote(edge_worst + 1)['shortfall_sim_s'] == 0


def test_ledger_charged_reads_existing_ledger_format():
    import tempfile
    with tempfile.TemporaryDirectory() as name:
        path = Path(name) / 'CHARGES.jsonl'
        assert l3.ledger_charged(path) == l3.SUPERSEDED_CHARGE
        path.write_text('{"id":"a","sim_s":100}\n{"id":"b","sim_s":230}\n',
            encoding='utf-8')
        assert l3.ledger_charged(path) == l3.SUPERSEDED_CHARGE + 330


def test_commit_and_gate_refusals():
    gate = {'verdict': 'PASS'}
    for commit in ('', 'deadbeef', l3.INTENT_INT_COMMIT, l3.MERGED_COMMIT[:-1] + '0'):
        try:
            l3.verify_commit(commit)
            raise AssertionError('accepted commit ' + str(commit))
        except Refused as error:
            assert l3.MERGED_COMMIT in str(error)
    ok = l3.verify_commit(l3.MERGED_COMMIT)
    assert ok['commit'] == l3.MERGED_COMMIT


def test_resource_decision_boundary():
    six = l3.REQUIRED_FREE_BYTES
    alive = {'ready': True}
    assert l3.decision(six, True, False)['ready']
    assert not l3.decision(six - 1, True, False)['ready']
    assert not l3.decision(six, False, False)['ready']
    assert not l3.decision(six, True, True)['ready']
    assert l3.decision(six + 1, True, False)['disk_free_bytes'] == six + 1


def test_bind_runtime_refuses_wrong_options():
    good = dict(config=dict(budget=dict(total=l3.CEILING_SIM_S),
        diagnostic_schedule=dict(A=dict(isolated=[]), B=dict(isolated=[6, 12, 18, 24]),
            R=dict(isolated=[12, 24])), **l3.INTENT_OPTIONS))
    assert l3.bind_runtime(good)['bound']
    bad = json.loads(json.dumps(good))
    bad['config']['v2v_intent_horizon'] = 4
    try:
        l3.bind_runtime(bad)
        raise AssertionError('horizon 4 accepted')
    except Refused:
        pass
    bad = json.loads(json.dumps(good))
    bad['config']['budget']['total'] = 17500000
    try:
        l3.bind_runtime(bad)
        raise AssertionError('changed cap accepted')
    except Refused:
        pass
