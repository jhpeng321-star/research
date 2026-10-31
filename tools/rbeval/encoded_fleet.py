"""Couple the existing frozen motion-encoder point settings to real admissions.

Extraction is owned by S-RB-LANE tools/rblane/encoder.py (b192e0c8), not
reimplemented here. This adapter only preserves initial identity and ensures
admission and newborn motion use the SAME zero-history encoder centre (HDV)
or inherited population default (CAV). No fit, optimizer or random draw.
"""
import torch

from wms.model.driver_encoder import PARAMETER_NAMES


class EncodedPointBirths:
    names = PARAMETER_NAMES

    def __init__(self, encoded):
        self.encoded = encoded
        self.current = None
        self.births_checked = 0

    def initial(self, state):
        raw = self.encoded.settings(state['vehicle_id'], state['is_cav'])
        self.current = {name: value.to(state['v_mps']) for name, value in raw.items()}
        return self.current

    def pool_for(self, row, cav=False):
        if int(row) != 0:
            raise ValueError('this FIT diagnostic uses one world per process row')
        values = self.encoded.defaults if cav else self.encoded.population
        return values.view(1, -1)

    def candidate(self, row, entrance, ordinal, cav=False):
        # A point population has one support element. Ordinal is still recorded
        # and verified; no hidden type or independently keyed identity is drawn.
        return dict(zip(self.names, self.pool_for(row, cav)[0].tolist()))

    def __call__(self, state, newborn, previous, *, birth_ordinals=None):
        coordinates = set(map(tuple, newborn.nonzero(as_tuple=False).tolist()))
        if coordinates and (birth_ordinals is None or set(birth_ordinals) != coordinates):
            raise ValueError('newborn settings require exact admitted birth ordinals')
        if not coordinates:
            self.current = previous
            return previous
        values = torch.where(state['is_cav'][..., None],
            self.encoded.defaults.view(1, 1, -1),
            self.encoded.population.view(1, 1, -1)).to(state['v_mps'])
        self.current = {name: torch.where(newborn, values[..., i], previous[name])
                        for i, name in enumerate(self.names)}
        self.births_checked += len(coordinates)
        return self.current
