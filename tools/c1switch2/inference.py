"""Six controller-in-model futures with a strict full-team objective contract.

The registered model engine supplies frozen WM/BB dynamics and observable source
admissions. The old entries-OFF v7 adapter is deliberately NOT a valid engine.
"""
import copy

from .core import means
from .design import CONTROLLERS, HORIZON, K, PROTOCOL
from .observable import anonymous_payload, future_seed


def validate_engine(attestation):
    required = dict(protocol=PROTOCOL, library=list(CONTROLLERS), history_s=60,
                    horizon_s=HORIZON, K=K, objective='full_team_delay_fleet_plus_source',
                    controller_feedback=True, wall_start_stop=True,
                    observable_source_admissions=True, anonymous_seed=True)
    if any(attestation.get(k) != v for k, v in required.items()):
        raise ValueError('six-way full-team inference attestation required; v7 entries-OFF is insufficient')


class Predictor:
    def __init__(self, engine, wall):
        validate_engine(engine.attestation)
        self.engine, self.wall = engine, wall

    def samples(self, raw):
        payload = anonymous_payload(raw, self.wall.validate_memory)
        seed = future_seed(payload)
        samples = {}
        for controller in CONTROLLERS:
            # A separate causal controller state per candidate/future is mandatory.
            # Engine imagines a held candidate for 60s, not three new picker calls.
            samples[controller] = self.engine.imagine(
                copy.deepcopy(payload), controller=controller, K=K,
                horizon_s=HORIZON, seed=seed, wall=self.wall)
        means(samples)
        return samples

    def predict(self, payload, ident):
        # ident is transport metadata only, excluded from stochastic keys.
        return means(self.samples(payload))
