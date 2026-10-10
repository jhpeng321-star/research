"""Registered owner-mode and wall-value checks precede lifecycle mutation."""
from types import SimpleNamespace

import pytest

from output.controller_verification_v7.pi_owner_adapter import PIOwnerAdapter
from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd.sensing import SensorFrame, NeighborMeasurement


@pytest.mark.parametrize('ids', [{''}, {1}, iter(())])
def test_invalid_cav_identity_rejected(ids):
    with pytest.raises(ValueError):
        PIOwnerAdapter(cav_ids=ids)


# WM-48: owner mode 27 is now accepted by the PI adapter (see
# output/pi_mode27_patch/pi_mode27.diff); any other mode stays refused.
@pytest.mark.parametrize('invalid', ['mode29', 'wall_nan'])
def test_invalid_owner_contract_preserves_prior_pi_state(invalid):
    writes = []
    api = SimpleNamespace(vehicle=SimpleNamespace(
        getSpeedMode=lambda v: 31,
        slowDown=lambda *args: writes.append(args),
        setSpeed=lambda *args: writes.append(args)))
    adapter = PIOwnerAdapter(cav_ids={'a'})
    owner = CommandOwnership(api, {'a'}, adapter)

    def frame(t):
        return SensorFrame('a', t, 10., 0., 31.29, 0,
            front=NeighborMeasurement(20., 0.), front_coverage_m=100., rear_coverage_m=100.)

    adapter.step(owner, {'a': frame(1.)}, now_s=1., live_ids={'a'})
    before = dict(adapter.pi._state)
    kwargs = {}
    if invalid == 'mode29':
        owner.cav_speed_mode = 29
    else:
        kwargs['wall_commands'] = {'a': float('nan')}
    with pytest.raises(ValueError):
        adapter.step(owner, {'a': frame(2.)}, now_s=2., live_ids={'a'}, **kwargs)
    assert adapter.pi._state == before
    assert len(writes) == 1 and adapter.stopped
