"""Opt-in parity against the committed track-base, loaded in memory only."""
import json
import subprocess
import types

import torch

from . import rerun_cpus as rc


def test_absent_and_false_opt_in_match_track_base_without_component_imports(tmp_path, monkeypatch):
    import sys
    code = subprocess.check_output(
        ['git', 'show', '13f184b5:tools/coopmwm/rerun_cpus.py'], text=True)
    baseline = types.ModuleType('tools.coopmwm._t2c_baseline')
    exec(compile(code, '<committed-track-base>', 'exec'), baseline.__dict__)
    for name in ('moves', 'trigger_policy', 'coop_layer', 'group_credit', 'merge_features',
                 'dense_teacher', 'plan_augment', 'practice_weighting', 'coop_reward'):
        monkeypatch.setitem(sys.modules, 'tools.coopmwm.'+name, None)
    kwargs = dict(root=str(tmp_path/'run'), reference_root=str(tmp_path/'refs'),
                  source_root=str(tmp_path/'src'), common_checkpoint_sha256='a'*64)
    base = dict(workers=2, planner=dict(lookahead_s=720))
    expected = baseline.runtime_config(base, **kwargs)
    for opt in ({}, dict(track2=False)):
        before = torch.get_rng_state().clone()
        actual = rc.runtime_config(base, **kwargs, **opt)
        assert torch.equal(before, torch.get_rng_state())
        assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
        assert rc.check_pins(actual)
        assert rc.track2_components(actual, 'B', adapters={}) == {}
        assert rc.episode_options(actual, 'B') == {}


def test_clean_profile_has_fair_enabled_consumers_and_costs(tmp_path):
    cfg = rc.runtime_config({}, root=str(tmp_path/'run'), reference_root=str(tmp_path/'refs'),
        source_root=str(tmp_path/'src'), common_checkpoint_sha256='a'*64, track2=True)
    assert cfg['launch_profile'] == 'track2_clean_run'
    assert cfg['teacher']['B']['kind'] == 'wm_dense'
    assert cfg['teacher']['R']['kind'] == 'random'
    for name in ('dense_teacher', 'plan_augment', 'practice_weighting'):
        assert cfg[name]['enabled'] and cfg[name]['arms'] == ['B', 'R']
    assert not cfg['coop_reward']['enabled']
    assert rc.budget()['phases']['teacher_verify'] > 0
    assert rc.budget()['phases']['augment_verify'] > 0
    assert rc.budget()['phases']['early_probes'] == 18*rc.ISOLATED_SIM_S_MEASURED
