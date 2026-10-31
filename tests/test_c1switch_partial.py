"""Receipt-only regression gates for the MAIN01:28 partial-analysis amendment."""
import json
from pathlib import Path

import numpy as np
import pytest

from tools.c1switch.design import CONTROLLERS
from tools.c1switch.partial import MIRROR, paired_rows, summarize_partial


def test_terminal_intersection_and_independent_registered_estimates():
    result = summarize_partial()
    assert result['status'] == 'PARTIAL'
    assert [w['seed'] for w in result['worlds']] == list(range(969112,969117))
    assert set(result['episode_n'].values()) == {5}
    assert len(result['measured_decisions']) == 60  # 5 worlds x 2 arms x 3 x 2 clocks
    assert result['new_sim_s'] == result['new_gpu_s'] == 0
    assert result['charged_sim_s'] == 383190
    # Independently reconstruct paired world regret and best-fixed selection from
    # exported world rows; compare all 2000 resampled contrasts, not just means.
    rows = result['world_rows']
    delta = np.array([r['regret60']['wm']-r['regret60']['bb'] for r in rows])
    episodes = np.array([[r['J600'][c] for c in CONTROLLERS] for r in rows])
    contrasts = np.array([[r['wm_minus_each_fixed_regret60'][c] for c in CONTROLLERS] for r in rows])
    draws = np.random.default_rng(0).integers(0,5,(2000,5))
    best = episodes[draws].mean(1).argmin(1)
    boot_fixed = contrasts[draws].mean(1)[np.arange(2000),best]
    primary = result['primary']
    assert primary['wm_minus_bb_regret60']['mean'] == pytest.approx(delta.mean())
    assert primary['wm_minus_bb_regret60']['ci95'] == pytest.approx(np.percentile(delta[draws].mean(1),[2.5,97.5]))
    assert primary['wm_minus_best_fixed_regret60']['ci95'] == pytest.approx(np.percentile(boot_fixed,[2.5,97.5]))
    assert primary['best_fixed'] == CONTROLLERS[episodes.mean(0).argmin()]


def test_missing_random_result_excludes_world_instead_of_expanding_arm_cohort(monkeypatch):
    original = Path.exists
    target = MIRROR/'jobs/random_s969112_episode/RESULT.json'
    monkeypatch.setattr(Path,'exists',lambda p: False if p == target else original(p))
    panel,rows,availability,_ = paired_rows(MIRROR)
    assert [w['seed'] for w in panel] == list(range(969113,969117))
    assert all(len(v) == 4 for v in rows.values())
    assert 'random' not in availability['969112']


@pytest.mark.parametrize('corruption',['branch','close','warmup'])
def test_rejects_corrupt_complete_world_receipts(monkeypatch,corruption):
    original = Path.read_bytes

    def altered(path):
        raw = original(path)
        if path == MIRROR/'jobs/wm_s969112_episode/RESULT.json':
            row = json.loads(raw)
            if corruption == 'branch':
                d = next(d for d in row['decisions'] if 'branches' in d)
                d['branches']['native']['60']['team_delay_vs'] += 1
            elif corruption == 'warmup':
                row['decisions'][0]['prefix']['tape_sha256'] = 'corrupt'
            raw = json.dumps(row).encode()
        if corruption == 'close' and path == MIRROR/'jobs/wm_s969112_episode/CLOSE.json':
            raw = json.dumps(dict(actual_sim_s=899,charged_sim_s=900)).encode()
        return raw

    monkeypatch.setattr(Path,'read_bytes',altered)
    with pytest.raises(ValueError,match='mismatch|accounting'):
        paired_rows(MIRROR)
