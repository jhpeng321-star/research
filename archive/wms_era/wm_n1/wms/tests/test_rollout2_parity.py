"""T6: torch ``wms.physics_torch`` vs numpy ``rollout2`` on TRAIN states.

Bounded guard for the ``_out/T6.md`` evidence: a couple of the 8 FIDELITY
chunk-4 states, open road + ``sumo_fidelity`` (model-frame geometry, sumo IDM),
residual/arrivals/lane-changes off.  TRAIN-only, CPU, no SUMO.  The full
8-state 120 s sweep is run by ``python -m wms.parity``.
"""

import importlib.util

import pytest


if importlib.util.find_spec("torch") is None:
    pytest.skip("PyTorch is not installed in this workspace interpreter", allow_module_level=True)

from output.wd01.wm_n1.wms import parity  # noqa: E402


CASES = (
    ("w8676m2700p20t360", 360.0, 120),
    ("w8677m2700p20t360", 360.0, 60),
)


@pytest.mark.parametrize("world,t0,steps", CASES)
def test_torch_matches_numpy_rollout2(world, t0, steps):
    if not (parity.CHUNK / f"{world}__base").is_dir():
        pytest.skip(f"TRAIN chunk-4 state {world} is not available")

    state = parity.load_fidelity_state(world, t0)
    ref = parity.numpy_rollout(state, t0=t0, n_steps=steps)
    got = parity.torch_rollout(state, n_steps=steps)
    result = parity.compare(ref, got, check=(steps,))[steps]

    assert result["n"] > 0
    assert result["max_dpos"] < 1e-6, result
    assert result["max_dv"] < 1e-6, result
    assert result["lane_agree"] == 1.0, result
