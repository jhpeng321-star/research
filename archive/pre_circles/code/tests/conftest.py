"""Shared pytest fixtures.

Several tests start real SUMO connections or activate the global traffic
snapshot; if a test crashes mid-episode the leaked connection/snapshot makes
later, unrelated tests fail in full-suite order only. Clear both around every
test so each test starts from a clean process state.
"""

import pytest

from env.snapshot import clear_active_snapshot
from env.sumo_utils import close_sumo


@pytest.fixture(autouse=True)
def _clean_traci_state(request):
    # test_snapshot manages one module-scoped SUMO connection shared by all
    # its tests; closing connections around each test would kill it.
    if request.module.__name__.rsplit(".", 1)[-1] == "test_snapshot":
        yield
        return
    clear_active_snapshot()
    close_sumo()
    yield
    clear_active_snapshot()
    close_sumo()
