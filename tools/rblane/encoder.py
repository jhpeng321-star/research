"""Frozen WM2 motion encoder, isolated from learned motion/lane corrections.

No training. Uses the exact shared long-history featurizer and source encoder
weights. Infer once at t0, keep settings by observable VID, and use the frozen
encoder's zero-history population centre for post-t0 entrants. Replaying the
pre-t0 LC warmup with inferred static settings is retrospective initialization,
not observation of any post-t0 outcome.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from wms.data.history import long_history_features
from wms.model.driver_encoder import DriverEncoder, PARAMETER_NAMES
from wms.model.wms import WmSModel

REGISTERED_WM2_SHA = 'b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'


def encoder_source_hashes():
    paths = dict(wrapper=Path(__file__),
                 driver_encoder=Path(inspect.getsourcefile(DriverEncoder)),
                 history_features=Path(inspect.getsourcefile(long_history_features)))
    result = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    result['current_interaction'] = hashlib.sha256(
        inspect.getsource(WmSModel._driver_interaction).encode()).hexdigest()
    return result


class FrozenMotionEncoder:
    def __init__(self, checkpoint: Path, physics, *, required_sha=None):
        self.sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.source_sha256 = encoder_source_hashes()
        if required_sha is not None and self.sha256 != required_sha:
            raise ValueError('frozen encoder checkpoint identity mismatch')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload['format'] != 'wm_s/1':
            raise ValueError('unexpected source checkpoint format')
        cfg = payload['config']
        self.history_s = float(cfg['history_s'])
        self.stride_s = float(cfg.get('history_stride_s', 1.0))
        if self.history_s != 60:
            raise ValueError('this diagnostic supplies exactly 60s authentic history')
        weights = {key.removeprefix('driver_encoder.'): value
                   for key, value in payload['model'].items() if key.startswith('driver_encoder.')}
        self.encoder = DriverEncoder(int(cfg.get('history_features', 6)),
            hidden=int(weights['query'].numel()), long_history=True,
            driver_param_mode=cfg.get('driver_param_mode', 'legacy'), physics=physics)
        self.encoder.load_state_dict(weights, strict=True)
        self.encoder.eval().requires_grad_(False)
        self.physics = physics
        a, b, tau, s0, _ = physics.idm()
        self.defaults = torch.tensor([physics.v0_mps, tau, s0, a, b, .2,
            physics.mobil_threshold_mps2, physics.lane_safe_gap_m, 1.], dtype=torch.float32)
        n = int(self.history_s / self.stride_s) + 1
        with torch.no_grad():
            self.population = self.encoder(torch.zeros(1, 1, n, 9),
                torch.zeros(1, 1, n, dtype=torch.bool), torch.zeros(1, 1, 4),
                self.defaults.view(1, 1, -1))['values'][0, 0].clone()

    @torch.no_grad()
    def prepare(self, kin, mask, ids, times, t0_s):
        ticks = np.flatnonzero((times >= t0_s - self.history_s) & (times <= t0_s))
        if len(ticks) != int(self.history_s / self.physics.dt_s) + 1 or times[ticks[-1]] != t0_s:
            raise ValueError('missing exact causal encoder history')
        k = torch.from_numpy(np.array(kin[ticks], copy=True)).float()
        m = torch.from_numpy(np.array(mask[ticks], copy=True)).bool()
        identity = torch.from_numpy(np.array(ids[ticks], copy=True)).long()
        continuity = m[:-1] & m[1:] & (identity[:-1] == identity[1:]) & (identity[:-1] >= 0)
        history, history_mask = long_history_features(k, m, continuity, m[-1],
            history_s=self.history_s, history_stride_s=self.stride_s, dt_s=self.physics.dt_s)
        # Invoke the source model's stateless interaction method directly, so
        # there is no independently maintained leader/follower featurization.
        interaction = WmSModel._driver_interaction(SimpleNamespace(physics=self.physics),
                                                  k[-1:].float(), m[-1:])
        values = self.encoder(history[None], history_mask[None], interaction,
            self.defaults.view(1, 1, -1).expand(1, k.shape[1], -1))['values'][0]
        lookup = {int(identity[-1, slot]): values[slot].clone()
                  for slot in torch.where(m[-1] & (identity[-1] >= 0))[0].tolist()}
        return EncodedSettings(lookup, self.population, self.defaults)


class EncodedSettings:
    def __init__(self, lookup, population, defaults):
        self.lookup, self.population, self.defaults = lookup, population, defaults

    def settings(self, ids, is_cav):
        values = torch.stack([self.defaults if bool(cav) else
            self.lookup.get(int(vid), self.population)
            for vid, cav in zip(ids.flatten(), is_cav.flatten())]).reshape(*ids.shape, -1).double()
        return {name: values[..., index] for index, name in enumerate(PARAMETER_NAMES)}
