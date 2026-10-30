"""Budget/seed registration guards; CPU only, no simulator invocation."""
import copy

import pytest

from tools.candB import parity


def test_fixed_allocation_source_and_budget_guards():
    doc = dict(allocation='S-CAPREL-PARITY', host='gpuR', cap_sim_s=3600,
        pooled_rmse_bound=.1603, jobs=list(parity.JOBS), registration_sha256='registered',
        source_sha256=parity.sha(parity.__file__), caprel_source_sha256=parity.law_sources())
    parity.validate_registration(doc)
    amended = dict(doc, jobs=list(parity.JOBS[:3]), prior_charged_sim_s=900)
    parity.validate_registration(amended)
    for key, value in [('cap_sim_s', 4500), ('host', 'cpuQ'),
                       ('source_sha256', 'stale'), ('pooled_rmse_bound', .17)]:
        changed = copy.deepcopy(doc)
        changed[key] = value
        with pytest.raises(ValueError):
            parity.validate_registration(changed)
    assert len(parity.JOBS)*900 == parity.CAP
    assert all(963000 <= job['seed'] <= 963999 for job in parity.JOBS)
    assert [j['residual'] for j in parity.JOBS] == [0., 0., -.01, -.01]
