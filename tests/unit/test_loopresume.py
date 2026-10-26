"""Prospective scientific boundary and durable four-world dispatch contracts."""
import json
from pathlib import Path

import pytest

from .test_fastloop import fixture, fast, loop


def test_later_policy_length_requires_a_bound_measured_recommendation(tmp_path, monkeypatch):
    cfg = fixture(tmp_path)
    state = loop.load_state(cfg)
    cfg['policy']['updates_per_cycle'] = 150
    cfg['loopresume'] = {'measured_update_selection': True}
    state['cycles'][-1]['cycle'] = 1
    assert loop.measured_policy_updates(cfg, state) == 150
    checkpoint = tmp_path/'policy.pt'
    checkpoint.write_bytes(b'accepted')
    state['policy'] = {'checkpoint': str(checkpoint)}
    state['cycles'][-1]['cycle'] = 2
    from wms.control import nativeinit
    monkeypatch.setattr(nativeinit, 'verify_collection_gate', lambda p: {'passed': True})
    summary = checkpoint.with_name('RUN_SUMMARY.json')
    summary.write_text(json.dumps({'status': 'completed', 'recommended_updates_per_cycle': 70}))
    assert loop.measured_policy_updates(cfg, state) == 70
    summary.write_text(json.dumps({'status': 'completed', 'recommended_updates_per_cycle': 0}))
    assert loop.measured_policy_updates(cfg, state) == 10
    summary.write_text(json.dumps({'status': 'completed', 'recommended_updates_per_cycle': 37}))
    with pytest.raises(loop.Refused, match='measured'):
        loop.measured_policy_updates(cfg, state)


@pytest.mark.parametrize('cycle,amended,deltas,continues', [
    (0, True, [-1., 1., -1., 1.], False),
    (1, False, [-1., 1., -1., 1.], False),
    (1, True, [-1., 1., -1., 1.], True),
    (1, True, [1.] * 4, False),
    (1, True, [-1.] * 4, True),
])
def test_prospective_guard_preserves_verdict_and_dispatch_barrier(tmp_path, monkeypatch, cycle, amended, deltas, continues):
    cfg = fixture(tmp_path)
    cfg['policy']['native_init'] = True
    cfg['collect'].update(worlds_per_cycle=6, branch_points=2)
    if amended:
        cfg['collect']['native_smoke_harmful_only_from_cycle'] = 1
    state = loop.load_state(cfg)
    state['cycles'][-1]['cycle'] = cycle
    root = Path(cfg['loop_root'])
    dispatched = []
    def world(loop_, cfg_, state_, ph, root_, item, box, attempt):
        if item['index'] >= 4:
            guard = loop.read_json(root / f'worlds_c{cycle:02d}.json')
            assert len(dispatched) >= 4 and guard['native_smoke_collection']['continues']
        dispatched.append(item['index'])
        return {'artifact': str(root / str(item['index'])), 'case': item, 'collect': {}}
    monkeypatch.setattr(fast, '_world', world)
    from wms.control import nativeinit
    monkeypatch.setattr(nativeinit, 'artifact_world_delta', lambda path, n: deltas[int(Path(path).name)])
    if continues:
        result = fast.collect(loop, cfg, state, {'attempt': 1}, root)
        assert len(result['worlds']) == 6
    else:
        with pytest.raises(loop.Refused, match='native smoke'):
            fast.collect(loop, cfg, state, {'attempt': 1}, root)
    guard = loop.read_json(root / f'worlds_c{cycle:02d}.json')['native_smoke']
    if len(set(deltas)) > 1:
        assert guard['verdict'] == 'inconclusive' and not guard['passed']
    assert len(dispatched) == (6 if continues else 4)
    charges = [json.loads(line) for line in (root / 'seed_ledger.jsonl').read_text().splitlines()
               if 'sim_s_charged' in json.loads(line)]
    assert len(charges) == len(dispatched)
    assert state['sim_s_charged'] == sum(row['sim_s_charged'] for row in charges)
