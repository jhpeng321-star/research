"""Native lesson windows and full prefixes must match dry quotes and execution."""
from contextlib import nullcontext
from copy import deepcopy

import torch

from . import launch_evaluation as le, replay_bank as rb, practice_starts as ps
from .clean3_budget import _lesson_costs
from .exact_grade import verify_sealed
from .test_replay_bank_confirm_cpu import snapshot, plan


def test_native_quote_and_scheduled_wave_share_exact_bank_expansion(tmp_path, monkeypatch):
    cfg = dict(run_root=str(tmp_path), replay_sources={'B': ['original-bank']})
    # The observed 12-entry imported/native bank, preserving original windows.
    ends = [480, 660, 720, 600, 600, 720, 720, 720, 1020, 900, 960, 660]
    records, snapshots = {}, {}
    for i, end in enumerate(ends):
        vid = f'original_{i}'
        start = 300 if end <= 720 else 600
        records[vid] = dict(verification_id=vid, proposer='wm', chosen_plan=plan(vid, .1),
            window=dict(start_s=start, end_s=end, tail_s=120, truncated=False, metric='team_delay_vs'))
        snapshots[vid] = snapshot(start)
    monkeypatch.setattr(le.production, 'backend_config', lambda c: c)
    monkeypatch.setattr(le.p1, '_bank_view', lambda c: nullcontext())
    monkeypatch.setattr(rb, 'lesson_list', lambda *a: dict(verification_ids=list(records)))
    monkeypatch.setattr(rb, 'bank_decisions', lambda *a, **k: list(records.values()))
    monkeypatch.setattr(ps, '_links', lambda *a: {})
    monkeypatch.setattr(rb, '_decision_snapshot', lambda root, arm, r, links:
        deepcopy(snapshots[r['verification_id']]))
    def forbidden(*args, **kwargs):
        raise AssertionError('dry quote cannot load current weights or dispatch/reserve physics')
    monkeypatch.setattr(le.production, 'checkpoint_weights', forbidden)
    monkeypatch.setattr(le.p1, '_reserve', forbidden)
    monkeypatch.setattr(le, 'ProcessPoolExecutor', forbidden)
    quotes = le.lesson_quotes(cfg)
    assert _lesson_costs(quotes) == {f'lesson_u{u:03}': 8760 for u in (6, 12, 18, 24)}
    for row in quotes.values():
        verify_sealed(row, 'receipt_sha256')
    loads, waves = [], []
    def weights(native, arm, update):
        loads.append((arm, update))
        return {'actual_policy': torch.tensor([1.])}
    monkeypatch.setattr(le.production, 'checkpoint_weights', weights)
    def wave(c, pairs, phase, prefix):
        waves.append((phase, prefix, [dict(id=f'{prefix}_{i:04}', sim_s=int(spec['sim_s']),
            window=spec['window']) for i, (spec, _) in enumerate(pairs)]))
        return [dict(status='graded') for _ in pairs]
    monkeypatch.setattr(le, '_wave', wave)
    le.scheduled(cfg, 18)
    assert loads == [('B', 18)]
    assert waves == [('v2v_lessons', 'lesson_u018', quotes['lesson_u018']['branch_manifest'])]
    assert sum(b['sim_s'] for b in waves[0][2]) == 8760
    assert waves[0][2][0]['sim_s'] == 480  # not suffix-only 180
