"""MAIN 02:30 real-outcome reward and disabled-WM integration regressions."""
from copy import deepcopy
import sys

import pytest
import torch

from tools.coopmwm.test_t2c_runtime_cpu import prepare, CPUBackend, RealisedReward
from tools.coopmwm.rerun_tracks_runtime import Runtime, state_sha
from tools.coopmwm.rerun_tracks_reward import team_reward


def test_real_outcome_changes_layer_not_base_and_needs_no_wm(tmp_path, monkeypatch):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    for name in ('dense_teacher', 'group_credit', 'planner_calibration'):
        monkeypatch.setitem(sys.modules, 'tools.coopmwm.'+name, None)
    rt = Runtime(cfg, reg, backend, smoke=True)
    # Hold bank/moves/initial weights fixed; change only realised local outcomes.
    rt.components['R']['coop_layer'].load_state_dict(rt.components['B']['coop_layer'].state_dict())
    class ZeroReward(RealisedReward):
        def realised_rewards(self, *args, **kw):
            return {v: 0. for v in super().realised_rewards(*args, **kw)}
    rt.components['R']['track2_reward'] = ZeroReward()
    before = {a: state_sha(m) for a, m in rt.models.items()}
    b, r = rt.update('B', 1, episodes=2), rt.update('R', 1, episodes=2)
    assert b['metrics']['team_reward'] > 0 and r['metrics']['team_reward'] == 0
    assert not torch.equal(rt.components['B']['coop_layer'].correction,
                           rt.components['R']['coop_layer'].correction)
    assert all(state_sha(rt.models[a]) == before[a] for a in ('A', 'B', 'R'))
    assert backend.ppo_calls == 0
    rt.update('A', 1, episodes=2)
    assert backend.ppo_calls == 1 and state_sha(rt.models['A']) != before['A']
    assert rt.components['A'] == {}


def test_local_hdv_rewards_and_global_rejection():
    from tools.coopmwm import rerun_cpus as rc
    obs = dict(time_s=10, own_sensing=dict(ego_id='c', vehicles=[dict(vid='c'), dict(vid='h')]),
               cav_ids=['c'], bounds={'c': [-3, 1.2]}, v2v_reports=[])
    after = deepcopy(obs)
    after['time_s'] = 11
    after['own_sensing']['vehicles'].append(dict(vid='remote'))
    class Reward:
        def realised_rewards(self, before, after, **kw):
            assert {r['vid'] for r in after['own_sensing']['vehicles']} == {'c', 'h'}
            return {'c': 2., 'h': 4.}
    receipt = team_reward(Reward(), obs, after, dict(track2_reward=rc.TRACK_SLOTS['track2_reward']))
    assert receipt['team_reward'] == 6. and receipt['neighbours'] == ['h']
    class GlobalReward:
        def realised_rewards(self, *a, **kw):
            return {'c': 2., 'h': 4., 'remote': 10000.}
    with pytest.raises(ValueError, match='local observable neighbourhood'):
        team_reward(GlobalReward(), obs, after, dict(track2_reward=rc.TRACK_SLOTS['track2_reward']))
