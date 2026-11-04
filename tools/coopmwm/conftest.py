"""Local (Windows) CPU tests only: provide flock when fcntl is absent; no-op on Linux."""
from tools.coopmwm.fcntl_compat import install

install()

import pytest


@pytest.fixture(autouse=True)
def _bank_imitation_runtime_binding():
    """v2v_live.hooks binds the runtime's imitation_step_mode and imitation_target_space
    process-wide (bank_imitation.install); with the owner 08:52 'kl_targeted' pin and the
    c4/effective-imitation 'effective' pin a hooks() call in one test must not leak into
    another test's 'fixed'/'raw' (byte-identical) expectations: restore the binding after
    every test."""
    from tools.coopmwm import bank_imitation as bi
    saved = (bi.RUNTIME_STEP_MODE, bi.RUNTIME_INTENT_BANK_TERM, bi.RUNTIME_TARGET_SPACE)
    yield
    bi.RUNTIME_STEP_MODE, bi.RUNTIME_INTENT_BANK_TERM, bi.RUNTIME_TARGET_SPACE = saved
    from tools.coopmwm import replay_bank as rb
    c = rb.confirmations
    while hasattr(c, 'original'):                        # bank_statistics.install wraps process-wide
        c = c.original                                   # (boot_v2/per_seed_v1; t1727 keeps no .original
    rb.confirmations = c                                 #  and self-restores inside its own tests)
