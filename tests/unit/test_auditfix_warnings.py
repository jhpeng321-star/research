import copy

from wms.control.auditfix_diagnostics import warn_unused_settings


def test_reports_audited_inactive_sections_without_mutation(capsys):
    raw = {'horizon': {'horizons_s': [10, 90], 'curriculum_updates': [0, 1]},
        'uncertainty': {'enabled': False, 'future_samples': 8},
        'lanes': {'enabled': True}, 'policy': {'lane_control': 'policy', 'init_zero_accel': True},
        'policy14': {'policy_ramp_merge': True, 'sampled_marginal_lane_weight': .2,
                     'entropy_floor': .001, 'critic_warmup_updates': 20, 'critic_value_scale': 1000},
        'trainer': {'real_return_mix': .3}, 'paircrit': {'enabled': True},
        'data': {'focus_cell_fraction': .6},
        'counterfactual': {'start_tilt': {'enabled': True}, 'validation_uniform': True},
        'run': {'seed': 12, 'updates': 10, 'validation_batches': 2}}
    original = copy.deepcopy(raw)
    rows = warn_unused_settings(raw, native_init=True,
        boundary={'flags': {'auto_horizon': True, 'long_credit': True}})
    sections = {r['section'] for r in rows}
    assert sections == {'horizon', 'uncertainty', 'lanes', 'policy', 'policy14', 'trainer',
                        'paircrit', 'counterfactual', 'run'}
    assert capsys.readouterr().err.count('WARNING: inactive config') == len(rows)
    assert raw == original


def test_active_sections_do_not_get_spurious_mode_warnings(capsys):
    raw = {'horizon': {'horizons_s': [10, 90]}, 'uncertainty': {'enabled': True, 'future_samples': 8},
           'lanes': {'enabled': True}, 'trainer': {'real_return_mix': .3},
           'paircrit': {'enabled': True}, 'counterfactual': {'start_tilt': {}},
           'policy14': {'critic_warmup_updates': 20}, 'run': {'threads': 1}}
    assert warn_unused_settings(raw) == []
    assert capsys.readouterr().err == ''
