"""W2 (c4/wall-tune) CPU contract tests: no SUMO, GPU or ssh; fakes only.

The OFF golden (fixtures/w2_off_golden.json) was written by ``off_fingerprint()``
on the unchanged base tree 3e5bc9f9 BEFORE any W2 code existed, so the OFF test
compares the W2 tree against the pre-W2 bytes, not against itself.
"""
from copy import deepcopy
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


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--write-golden']:
        GOLDEN.write_text(json.dumps(off_fingerprint(), indent=1, sort_keys=True)+'\n', encoding='utf-8')
        print('wrote', GOLDEN)
