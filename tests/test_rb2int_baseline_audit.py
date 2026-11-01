import copy

import pytest

from tools.rb2int.baseline_audit import WM_SHA, contract


def identity():
    return dict(model='wm2', wm_sha256=WM_SHA, K=8, horizon=600, t0=300,
        arms=['native', 'ctrl0', 'mappo', 'pi', 'circles'], mix=True, native_lanes=True,
        seed=965140, update=190, sources={'/frozen/src/wms/model.py': 'physics',
        '/frozen/tools/rbeval/run_wm.py': 'old_orchestration',
        '/frozen/tools/rbeval/references.json': 'frozen_helpers'})


def test_extended_orchestration_does_not_replace_frozen_physics_or_reference_helpers():
    result = contract(identity(), 'WM2', '/frozen')
    assert result == {'/frozen/src/wms/model.py': 'physics', '/frozen/tools/rbeval/references.json': 'frozen_helpers'}


@pytest.mark.parametrize('key,value', [('K', 2), ('model', 'candidate'), ('wm_sha256', 'changed'), ('native_lanes', False)])
def test_wrong_original_baseline_contract_is_rejected(key, value):
    data = copy.deepcopy(identity())
    data[key] = value
    with pytest.raises(ValueError, match='frozen baseline identity'):
        contract(data, 'WM2', '/frozen')


def test_orchestration_only_source_manifest_cannot_stand_in_for_model():
    data = identity()
    data['sources'].pop('/frozen/src/wms/model.py')
    with pytest.raises(ValueError, match='physical implementation'):
        contract(data, 'WM2', '/frozen')
