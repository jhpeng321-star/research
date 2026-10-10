"""Small CLI-only model fixture; no simulator or external data."""

import numpy as np


class FixtureModel:
    def predict_states(self, state, plan, horizon_s, seed):
        times = np.arange(0.0, float(horizon_s) + 0.5, 0.5)
        free = np.full((len(times), 1), 10.0)
        speed = np.full_like(free, 10.0 - float(plan))
        return {"time_s": times, "speed_mps": speed, "free_speed_mps": free}


def factory():
    return FixtureModel()
