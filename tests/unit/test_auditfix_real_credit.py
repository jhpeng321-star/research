from types import SimpleNamespace

import pytest

from wms.control.auditfix_diagnostics import require_real_credit_effect
from wms.control.counterfactual import CounterfactualConfig


def trainer(multipliers, floor=True):
    return SimpleNamespace(credit_multipliers=multipliers,
        config=SimpleNamespace(real_credit_individual_floor=floor),
        counterfactual=CounterfactualConfig(enabled=True, curriculum_updates=[0, 1],
            w_individual=[8.9, 8.9], w_coop=[8.9, 8.9], w_group=[9.9, 9.9],
            w_team=[10.3, 10.3], w_all=[1., 1.]))


def test_audited_candidate_noop_errors_but_no_file_is_unchanged():
    assert require_real_credit_effect(trainer(None)) is None
    t = trainer({'individual': .40108381909949953, 'group': 1., 'coop': 1., 'team': 1.})
    with pytest.raises(ValueError, match='--real-credit has no effective weight change'):
        require_real_credit_effect(t)


def test_useful_multiplier_keeps_existing_meaning():
    result = require_real_credit_effect(trainer({'group': .5}))
    assert result[0]['effective']['group'] == 4.95
    assert result[0]['effective']['individual'] == 8.9
    result = require_real_credit_effect(trainer({'individual': .4}, floor=False))
    assert result[0]['effective']['individual'] == pytest.approx(3.56)
    assert result[0]['effective']['coop'] == pytest.approx(3.56)
