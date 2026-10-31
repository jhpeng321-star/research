"""Six-way inference; full-team or explicitly registered common road-only scoring."""
import copy
import re

from .core import means
from .design import CONTROLLERS, HORIZON, K, PROTOCOL
from .observable import anonymous_payload, future_seed


def validate_engine(attestation):
    required = dict(protocol=PROTOCOL, library=list(CONTROLLERS), history_s=60,
                    horizon_s=HORIZON, K=K,
                    controller_feedback=True, wall_start_stop=True,
                    anonymous_seed=True)
    if any(attestation.get(k) != v for k, v in required.items()):
        raise ValueError('six-way full-team inference attestation required; v7 entries-OFF is insufficient')
    objective = attestation.get('objective')
    if objective == 'full_team_delay_fleet_plus_source':
        if attestation.get('observable_source_admissions') is not True:
            raise ValueError('full-team inference needs observable source admissions')
    elif objective == 'on_road_fleet_delay':
        if (attestation.get('observable_source_admissions') is not False or
                attestation.get('entries') is not False or
                attestation.get('evaluation_objective') != 'full_team_delay_fleet_plus_source' or
                not re.fullmatch('[0-9a-f]{64}', attestation.get('objective_design_sha256', ''))):
            raise ValueError('road-only scoring needs a registered design binding and disclosure')
    else:
        raise ValueError('unregistered prediction objective')


class Predictor:
    def __init__(self, engine, wall):
        validate_engine(engine.attestation)
        self.engine, self.wall = engine, wall

    def samples(self, raw):
        payload = anonymous_payload(raw, self.wall.validate_memory)
        if hasattr(self.engine, 'imagine_all'):
            samples = self.engine.imagine_all(payload, K=K, horizon_s=HORIZON)
            means(samples)
            return samples
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
