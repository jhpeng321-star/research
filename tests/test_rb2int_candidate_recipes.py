import copy
import tomllib

from tools.rb2int.audit import ROOT
from wms.physics.learned_parameters import load_parameters


def test_lp_candidate_is_same_physics_plus_declared_development_departure_distribution():
    base = tomllib.loads((ROOT/'configs/wm_phys_v5.toml').read_text())
    lp = tomllib.loads((ROOT/'configs/wm_phys_v5_lp.toml').read_text())
    comparison = copy.deepcopy(lp)
    comparison['physics']['physics_params'] = 'off'
    artifact = comparison['physics'].pop('physics_params_file')
    assert comparison == base
    parameters = load_parameters(str(ROOT/artifact))
    assert 0 <= parameters.ramp_last_probability <= 1
    for recipe in (base, lp):
        assert recipe['model']['residual_accel_off'] is True
        assert recipe['model']['temporal_accel_off'] is True
        assert recipe['model']['lane_head'] == 'off'
        assert recipe['physics']['entry_learned'] is False
        assert recipe['inference']['future_observation_inputs'] is False
        assert all(recipe['inference'][key] is False for key in (
            'learned_acceleration_noise', 'learned_admission', 'learned_lane_corrections',
            'learned_merge_corrections', 'training'))
        assert recipe['variants']['FULL']['fitted_lc'] is True
        assert recipe['variants']['FULL']['match_cav_execution_settings'] is True
        assert recipe['variants']['FULL']['condition_motion'] is True
