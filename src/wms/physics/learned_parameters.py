"""Small physical departure-parameter distribution; no motion residuals.

Outputs are documented SUMO departLane/departSpeed values. Draws are keyed by
future/world/entrance/request ordinal, persistent across retries and actions.
IDM and LC population/posterior coordinates remain owned by FIT/LANE.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class DepartureParameters:
    lane_first_probability: float
    mainline_last_probability: float
    ramp_last_probability: float
    sha256: str = ''

    def __post_init__(self):
        for value in (self.lane_first_probability, self.mainline_last_probability,
                      self.ramp_last_probability):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('physical departure probabilities must lie in [0,1]')

    def modes(self, mix, row, entrance, ordinal):
        if not hasattr(mix, 'seed') or not hasattr(mix, 'row_keys'):
            raise ValueError('learned departures require identity-keyed paired futures')
        if entrance not in (0, 1) or int(ordinal) != ordinal or ordinal < 0:
            raise ValueError('invalid departure identity')
        prefix = f'physical-departure-v1:{mix.seed}:{mix.row_keys[row]}:{entrance}:{ordinal}'
        def uniform(channel):
            return int.from_bytes(hashlib.sha256(f'{prefix}:{channel}'.encode()).digest()[:8], 'big') / 2**64
        lane = 'first' if entrance == 0 and uniform('lane') < self.lane_first_probability else 'free'
        probability = self.ramp_last_probability if entrance else self.mainline_last_probability
        speed = 'last' if uniform('speed') < probability else 'max'
        return lane, speed


@lru_cache(maxsize=16)
def load_parameters(path):
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get('schema') != 'wm_phys.learned_parameters.v1' or data.get('name') != 'learned_v1':
        raise ValueError('unsupported learned physics artifact')
    worlds = data.get('fit_worlds', [])
    selected = data.get('selection_worlds', [])
    if not worlds or not selected or any(isinstance(w, bool) or not isinstance(w, int)
        or not 969400 <= w <= 969431 for w in worlds + selected):
        raise ValueError('learned physics artifact must declare development-only exposure')
    values = data['departure_parameters']
    names = {'lane_first_probability', 'mainline_last_probability', 'ramp_last_probability'}
    if set(values) != names:
        raise ValueError('only documented departure parameters may be learned here')
    return DepartureParameters(**values, sha256=hashlib.sha256(raw).hexdigest())


def departure_parameters(physics):
    name = getattr(physics, 'physics_params', 'off')
    if name == 'off':
        return None
    if name != 'learned_v1' or not getattr(physics, 'physics_params_file', None):
        raise ValueError('learned_v1 requires an explicit frozen parameter file')
    return load_parameters(str(Path(physics.physics_params_file).resolve()))
