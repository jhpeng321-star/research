"""W2 (c4/wall-tune) CPU contract tests: no SUMO, GPU or ssh; fakes only.

The OFF golden (fixtures/w2_off_golden.json) was written by ``off_fingerprint()``
on the unchanged base tree 3e5bc9f9 BEFORE any W2 code existed, so the OFF test
compares the W2 tree against the pre-W2 bytes, not against itself.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

GOLDEN = Path(__file__).with_name('fixtures')/'w2_off_golden.json'


def _h(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=repr).encode()).hexdigest()


def base_cfg(root='E:/unused-w2-root'):
    return dict(track2_backend=True, run_root=str(root), training_seed=123,
        v2v=True, v2v_dim=48, v2v_intent=True, merge_features=True, host='cpuV',
        layer_config=dict(hidden_dim=8, learning_rate=.0003, kl_budget=.045,
            clip_ratio=.2, max_grad_norm=.5), trigger_candidates=4, trigger_partner_slots=5,
        decision_interval_s=60, trigger_lr=.0003, layer_epochs=1,
        trigger_ppo=dict(clip_ratio=.2, value_weight=.5, entropy_weight=.01, max_grad_norm=.5),
        worlds_by_update={'1': [dict(seed=1), dict(seed=2)]}, practice_starts=False,
        runtime='unused', checkpoint='unused', accel_bounds=[-3, 2.6], gpu_deadline=9999999999,
        workers=2, load_max=40, budget=dict(total=5000000, phases=dict(track2_train=4000000)),
        safety_reference=dict(by_world={}, unmatched_zero_event_benchmark=True, source='CPU fixture'),
        track2_reward=dict(weight=.5, provenance='owner registration; CPU fixture'),
        imitation_epochs=5)


def scene(t=0):
    """Three mainline CAVs abreast upstream of the merge zone plus ramp/mainline HDVs."""
    out = [dict(vid='c1', s_m=-420.+12.*t, v_mps=12., lane=0, is_cav=True, length_m=5.),
           dict(vid='c2', s_m=-425.+12.*t, v_mps=12., lane=1, is_cav=True, length_m=5.),
           dict(vid='c3', s_m=-440.+12.*t, v_mps=12., lane=1, is_cav=True, length_m=5.),
           dict(vid='h1', s_m=-410.+12.*t, v_mps=11., lane=0, is_cav=False, length_m=5.),
           dict(vid='h2', s_m=-380.+12.*t, v_mps=3., lane=2, is_cav=False, length_m=5.)]
    return out


def run_policy(cfg, arm='B', ticks=8, seed=7):
    from tools.coopmwm import exploration, track2_backend as tb
    from tools.coopmwm.acceleration import Controller
    from tools.coopmwm.v2v_intent import IntentPolicy
    from tools.coopmwm.v2v_runtime import controller_class
    old_regime, threads = exploration.regime, torch.get_num_threads()
    exploration.regime = lambda rows: 'ordinary'
    torch.set_num_threads(1)
    try:
        torch.manual_seed(seed)
        p = tb.EpisodePolicy(IntentPolicy(), cfg, arm)
        controller = controller_class(Controller)(p, torch.Generator().manual_seed(9), False)
        samples = []
        for k in range(ticks):
            now = 300+k
            rows = scene(k)
            p.observe(rows, now)
            controller.apply(rows, {r['vid']: 0. for r in rows if r['is_cav']}, now)
            samples.append(controller.events[-1]['sample'])
        return p, samples
    finally:
        exploration.regime = old_regime
        torch.set_num_threads(threads)


def off_fingerprint(extra=None):
    """Digests of every OFF-path output W2 touches (policy samples, weights, moves)."""
    from tools.coopmwm import feasible_proposals, group_lane, moves
    cfg = base_cfg()
    cfg.update(extra or {})
    p, samples = run_policy(cfg)
    out = dict(samples=[_h(s) for s in samples],
               trigger_state=_h({k: v.tolist() for k, v in p.trigger.state_dict().items()}),
               layer_state=_h({k: v.tolist() for k, v in p.layer.state_dict().items()}),
               trigger_keys=sorted(p.trigger.state_dict()),
               executor=_h([dict(move=e['move'].to_dict(), start=e['start'], until=e['until'])
                            for e in p.executor.active]))
    rows = [{k: r[k] for k in moves._ROW_KEYS} for r in scene()]
    rng = np.random.default_rng(5)
    out['moves_candidates'] = _h([m.to_dict() for m in moves.candidates(dict(self='c1', rows=rows), rng, 8)])
    rng = np.random.default_rng(5)
    out['feasible_candidates'] = _h([m.to_dict() for m in feasible_proposals.candidates(
        dict(self='c1', rows=rows), rng, 8)])
    wall = moves.Move('wall', ('c1', 'c2', 'c3'), 3, 25, .7)
    out['wall_dict'] = wall.to_dict()
    out['offsets'] = _h([moves.offsets(wall, t) for t in range(40)])
    out['offsets_state'] = _h([moves.offsets(wall, t, state=dict(s_m={'c1': -150.+5*t}))
                               for t in range(40)])
    fields = group_lane.executable_fields(wall, 40, [[0.1*i]*40 for i in range(3)])
    out['executable_fields'] = _h(fields)
    plan = dict(id='p', members=['c1', 'c2', 'c3'], duration_s=40, **fields)
    out['from_plan'] = moves.from_plan(plan).to_dict()
    return out


def test_off_is_byte_identical_to_the_pre_w2_tree():
    golden = json.loads(GOLDEN.read_text(encoding='utf-8'))
    assert off_fingerprint() == golden
    # Explicitly-False W2 keys are the same OFF path (no new parameters, rng draws or bytes).
    explicit = dict(wall_pressure_features=False, wall_release=False, wall_negative_labels=False)
    assert off_fingerprint(explicit) == golden


def test_w2_settings_defaults_types_and_b_equals_r():
    from tools.coopmwm import wall_w2
    assert wall_w2.settings({}) == dict(wall_pressure_features=False, wall_release=False,
                                        wall_negative_labels=False, wall_negatives=None)
    assert not wall_w2.any_on({})
    for bad in (dict(wall_release=1), dict(wall_pressure_features='yes'),
                dict(wall_negatives='x.jsonl'),          # path without the option
                dict(wall_release=True, w2_by_arm=dict(B=dict(wall_release=True), R={})),
                dict(w2_by_arm=dict(B={})),
                dict(w2_by_arm=dict(B={}, R=dict(wall_negative_labels=True)))):
        with pytest.raises(ValueError):
            wall_w2.settings(bad)
    same = dict(wall_release=True, wall_pressure_features=True,
                w2_by_arm=dict(B=dict(wall_release=True, wall_pressure_features=True),
                               R=dict(wall_release=True, wall_pressure_features=True)))
    assert wall_w2.settings(same)['wall_release'] is True
    from tools.coopmwm import track2_backend as tb
    with pytest.raises(ValueError, match='bool'):
        tb._check(dict(base_cfg(), wall_release='on'), 'B')


def _strip(value, keys=('merge_pressure',)):
    if isinstance(value, dict):
        return {k: _strip(v, keys) for k, v in value.items() if k not in keys}
    if isinstance(value, (list, tuple)):
        return [_strip(v, keys) for v in value]
    return value


def test_pressure_on_is_additive_zero_initialised_and_keeps_the_frozen_observation():
    off_policy, off = run_policy(base_cfg())
    on_policy, on = run_policy(dict(base_cfg(), wall_pressure_features=True))
    from tools.coopmwm.wall_pressure import PRESSURE_DIM
    # Every recorded byte except the added observable block is identical: the frozen MAPPO
    # input (context x / v2v), the layer residual features, actions, trigger logp and draws.
    assert [_h(_strip(s)) for s in on] == [_h(s) for s in off]
    assert all(set(s['context']['merge_pressure']) == set(s['context']['members']) for s in on)
    assert all(len(b) == PRESSURE_DIM for s in on for b in s['context']['merge_pressure'].values())
    assert any('merge_pressure' in s['track2_trigger']['context'] for s in on if 'track2_trigger' in s)
    # Additive head only: all OFF parameters equal, the new output layer starts at zero.
    off_state, on_state = off_policy.trigger.state_dict(), on_policy.trigger.state_dict()
    assert set(on_state)-set(off_state) == {f'w2_head.{k}' for k in
                                           ('0.weight', '0.bias', '2.weight', '2.bias')}
    assert all(torch.equal(off_state[k], on_state[k]) for k in off_state)
    assert not on_state['w2_head.2.weight'].any() and not on_state['w2_head.2.bias'].any()
    assert _h({k: v.tolist() for k, v in on_policy.layer.state_dict().items()}) == \
        _h({k: v.tolist() for k, v in off_policy.layer.state_dict().items()})
    # The head is trainable from the recorded decision (gradient reaches the zero layer).
    from tools.coopmwm.track2_backend import _batch
    record = next(s['track2_trigger'] for s in on if 'track2_trigger' in s)
    batch = _batch(record)
    loss = on_policy.trigger.imitation_loss(batch, torch.zeros(len(batch.actions), dtype=torch.long))
    loss.backward()
    assert on_policy.trigger.w2_head[2].weight.grad.abs().sum() > 0
    assert any(p is on_policy.trigger.w2_head[2].weight for p in on_policy.trigger.learning_parameters())


@pytest.mark.parametrize('extra', [dict(wall_pressure_features=True, wall_release=True),
                                   dict(wall_release=True)])
def test_b_and_r_are_identical(extra):
    b_policy, b = run_policy(dict(base_cfg(), **extra), arm='B')
    r_policy, r = run_policy(dict(base_cfg(), **extra), arm='R')
    def relabel(s):
        return json.loads(json.dumps(s, default=repr).replace('"arm": "R"', '"arm": "B"'))
    assert [_h(relabel(s)) for s in r] == [_h(relabel(s)) for s in b]
    assert _h({k: v.tolist() for k, v in b_policy.trigger.state_dict().items()}) == \
        _h({k: v.tolist() for k, v in r_policy.trigger.state_dict().items()})
    from tools.coopmwm import wall_w2
    assert wall_w2.settings(dict(base_cfg(), **extra)) == b_policy.w2 == r_policy.w2


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--write-golden']:
        GOLDEN.write_text(json.dumps(off_fingerprint(), indent=1, sort_keys=True)+'\n', encoding='utf-8')
        print('wrote', GOLDEN)
