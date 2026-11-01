"""Physical parameter sampling must preserve CRN and source safety semantics."""
import json
from types import SimpleNamespace

import pytest

from wms.physics.config import PhysicsConfig
from wms.physics.learned_parameters import DepartureParameters, departure_parameters, load_parameters
from wms.model.sumo_insertion import SumoInsertion


def test_default_off_does_not_open_a_parameter_file():
    assert departure_parameters(PhysicsConfig(physics_params_file='missing')) is None
    with pytest.raises(ValueError, match='explicit frozen'):
        departure_parameters(PhysicsConfig(physics_params='learned_v1'))


def test_paired_identity_modes_are_retry_and_row_invariant():
    profile = DepartureParameters(.37, .62, .18)
    mix = SimpleNamespace(seed=17, row_keys=[969420, 969420])
    observed = set()
    for ordinal in range(100):
        for entrance in (0, 1):
            value = profile.modes(mix, 0, entrance, ordinal)
            assert value == profile.modes(mix, 1, entrance, ordinal)
            assert value == profile.modes(mix, 0, entrance, ordinal)
            if entrance:
                assert value[0] == 'free'
            observed.add(value)
    assert observed == {('free', 'max'), ('free', 'last'), ('first', 'max'), ('first', 'last')}


@pytest.mark.parametrize('value', [-.01, 1.01, float('nan'), float('inf'), True])
def test_probability_domain(value):
    with pytest.raises(ValueError):
        DepartureParameters(value, 0, 0)


def test_artifact_refuses_reserved_or_undeclared_fields(tmp_path):
    data = dict(schema='wm_phys.learned_parameters.v1', name='learned_v1',
        fit_worlds=[969420], selection_worlds=[969431], departure_parameters=dict(
        lane_first_probability=.2, mainline_last_probability=.3, ramp_last_probability=.4))
    path = tmp_path/'good.json'
    path.write_text(json.dumps(data))
    assert load_parameters(str(path)).ramp_last_probability == .4
    data['selection_worlds'] = [969432]
    path = tmp_path/'reserved.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='development-only'):
        load_parameters(str(path))
    data['selection_worlds'] = [969431]
    data['departure_parameters']['acceleration_residual'] = 1
    path = tmp_path/'extra.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='only documented'):
        load_parameters(str(path))


def placement(profile=None, direct_lane='free', direct_speed='max', gap=30., leader=5.):
    mix = SimpleNamespace(seed=5, row_keys=[969420], candidate=lambda *args: {})
    insertion = SumoInsertion(PhysicsConfig(), mix, depart_lane=direct_lane,
                              depart_speed=direct_speed, departure_profile=profile)
    process = SimpleNamespace(cfg=SimpleNamespace(conserve_queue=True, learned_entry=None,
        learned_admission=None, sumo_lane_speed_mps=[31.29,31.29,22.22]), dt=.5, entry_s=[0.,0.,0.])
    process._lane_rear = lambda work,b,lane: (gap+10.1,5.,leader,False) if lane == 1 else (60.,5.,15.,False)
    work = SimpleNamespace(inserted=[], birth_ordinals={}, s={})
    accepted = []
    def place(work, b, entrance, lane, cav, speed, length):
        work.inserted.append((b,0));accepted.append((lane,speed));return True
    process._place = place
    result = insertion.place(process, work, 0, 0, [0,1], False, 3)
    return result, accepted, work.birth_ordinals


@pytest.mark.parametrize('profile,lane,speed', [
    (DepartureParameters(0,0,0),'free','max'),
    (DepartureParameters(1,0,0),'first','max'),
    (DepartureParameters(0,1,0),'free','last'),
    (DepartureParameters(1,1,0),'first','last')])
def test_learned_modes_equal_documented_insertion_forms(profile,lane,speed):
    assert placement(profile) == placement(None,lane,speed)


def test_last_mode_preserves_unsafe_refusal_and_zero_profile_preserves_baseline():
    assert placement(DepartureParameters(0,0,0)) == placement()
    assert placement(DepartureParameters(1,1,0), gap=.01)[0] is False
    assert placement(DepartureParameters(1,0,0), gap=.01)[0] is False


def test_development_filter_is_before_any_start_artifact_open():
    from tools.rb2lphys.evaluate import restrict_start_source
    from tools.rbeval.core import load
    from pathlib import Path
    code = load(Path(__file__).resolve().parents[1]/'tools/rbeval/references.json')['starts']['code']
    filtered = restrict_start_source(code, [969420,969431])
    assert filtered.index("if w['seed'] in (969420, 969431)") < filtered.index('for i, world in enumerate(worlds)')
    for world in (969419,969432,969439):
        with pytest.raises(ValueError, match='development panel'):
            restrict_start_source(code, [world])
