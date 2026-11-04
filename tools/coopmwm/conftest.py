"""Local (Windows) CPU tests only: provide flock when fcntl is absent; no-op on Linux."""
from tools.coopmwm.fcntl_compat import install

install()

import pytest


@pytest.fixture(autouse=True)
def _bank_imitation_runtime_binding():
    """v2v_live.hooks binds the runtime's imitation_step_mode process-wide (bank_imitation.install);
    with the owner 08:52 'kl_targeted' pin a hooks() call in one test must not leak into another
    test's 'fixed' (byte-identical) expectations: restore the binding after every test. The
    c4/imit-kl-option runtime limit (RUNTIME_KL_LIMIT) is restored the same way."""
    from tools.coopmwm import bank_imitation as bi
    saved = (bi.RUNTIME_STEP_MODE, bi.RUNTIME_INTENT_BANK_TERM, bi.RUNTIME_KL_LIMIT)
    yield
    bi.install(*saved)
