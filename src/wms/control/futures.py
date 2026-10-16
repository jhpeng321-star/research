"""Sampled futures: several stochastic world-model rollouts per start instead of one deterministic one.

Two sources of randomness, both OFF by default and both with common random numbers (CRN):

* epistemic driver uncertainty -- each HDV's encoded IDM/MOBIL parameters (and the driver
  embedding) are perturbed within the encoder's own uncertainty (:func:`sample_driver_encodings`;
  the RSU check's version, factored out);
* aleatoric acceleration noise -- standard-normal draws that the WM scales by the residual head's
  predicted ``exp(accel_log_sigma)`` (``WmSModel.rollout(accel_noise=...)``).

CRN: standard normals are drawn ONCE per start and sample index from a caller-owned generator and
reused for every ensemble member (member-specific encodings and sigmas stay distinct), for every
counterfactual variant row of the same start, and for every RSU candidate.  Differences between
candidates / variants / members therefore measure the decision (or the WM), not sampling noise.
Nothing here touches the global RNG stream, so with sampling off a run is bit-identical.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import torch
from torch import Tensor

HISTORY_KEYS = ("history", "history_mask", "driver_history", "driver_history_mask")


def seed_of(*parts) -> int:
    """Deterministic 60-bit seed of a tuple of printable parts."""
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:15], 16)


@dataclass(frozen=True)
class FuturesConfig:
    n_samples: int = 1               # futures per start (x ensemble members = pseudo-members)
    driver_noise_scale: float = 1.0  # multiplies the encoder uncertainty; 0 = no driver sampling
    embedding_noise_std: float = 0.05
    accel_noise: bool = False        # aleatoric HDV acceleration noise from the residual log-sigma
    accel_noise_scale: float = 1.0   # multiplies the standard normal draws
    accel_sigma_max: float = 1.0     # cap of exp(log_sigma) in m/s^2 (the head's clamp reaches e^2)
    seed: int = 0

    def __post_init__(self):
        if isinstance(self.n_samples, bool) or not isinstance(self.n_samples, int) or self.n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        for name in ("driver_noise_scale", "embedding_noise_std", "accel_noise_scale",
                     "accel_sigma_max"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.n_samples > 1 and self.driver_noise_scale == 0 and not self.accel_noise:
            raise ValueError("n_samples > 1 needs a randomness source (driver or accel noise)")

    @property
    def samples_drivers(self) -> bool:
        return self.driver_noise_scale > 0

    @property
    def active(self) -> bool:
        """Sampling machinery is used (otherwise the single deterministic rollout is untouched)."""
        return self.n_samples > 1 or self.accel_noise

    @classmethod
    def from_trainer(cls, config) -> "FuturesConfig":
        """From a ``DreamerConfig`` (``future_*`` fields)."""
        return cls(n_samples=config.future_samples, driver_noise_scale=config.future_driver_scale,
                   embedding_noise_std=config.future_embedding_std,
                   accel_noise=bool(config.future_accel_noise),
                   accel_noise_scale=config.future_accel_scale,
                   accel_sigma_max=config.future_accel_sigma_max, seed=int(config.future_seed))


class NormalDraws:
    """Standard-normal draws of a caller-owned CPU generator, cached by key.

    The first request of a key draws; later requests (other ensemble members, other candidates)
    return the SAME numbers, which must have the same shape."""

    def __init__(self, generator: Optional[torch.Generator] = None):
        self.generator = generator if generator is not None else torch.Generator(device="cpu")
        self.cache: dict = {}

    def __call__(self, shape, reference: Tensor, key) -> Tensor:
        if key not in self.cache:
            self.cache[key] = torch.randn(tuple(shape), generator=self.generator)
        if tuple(self.cache[key].shape) != tuple(shape):
            raise ValueError("ensemble driver encodings must have matching shapes/keys")
        return self.cache[key].to(reference)


#: The driver encoder's ``uncertainty`` output is NOT trained by any WM loss (driver settings are
#: learned from prediction error only, with no likelihood on the parameters), so it sits near its
#: initial value, ~0.7 x the parameter's full deviation bound.  Sampling from it at scale 1 with a
#: +/-50 % cap slowed free-flow traffic systematically (one halved-v0 car blocks its followers) and
#: raised the 30 s HDV median |ds| from 5.3 m to 27.8 m (HORIZON_CHECK.md, 2026-10-13).  Callers
#: therefore default to the POINT estimate (noise 0); sampling is an explicit, uncalibrated opt-in.
DRIVER_SAMPLE_MAX_REL_DEV = 0.15


def sample_driver_encodings(model, state: Mapping, history_args: Mapping, n_samples: int,
                            normal: Callable, *, driver_noise_scale: float = 1.0,
                            embedding_noise_std: float = 0.05,
                            max_rel_dev: float = DRIVER_SAMPLE_MAX_REL_DEV) -> dict:
    """``n_samples`` perturbed driver encodings of ``model`` for a start state ``[B, K]``.

    Returns ``{"driver_params": {name: [n, B, K]}, "driver_embedding": [n, B, K, D]}`` (keys only
    when the WM exposes them) for ``rollout(driver_params=..., driver_embedding=...)``.  Draws come
    from ``normal(shape, reference, key)`` so ensemble members share them (CRN); each member keeps
    its own encoded values and uncertainty.  Parameters are perturbed in physical units by the
    encoder uncertainty and capped to +/-``max_rel_dev`` of the encoded value so positive IDM
    parameters stay positive; the embedding by ``embedding_noise_std`` (hybrid residuals and history
    lane heads condition on it).  ``driver_noise_scale = embedding_noise_std = 0`` returns the point
    estimate exactly.  The encoder uncertainty is uncalibrated (see DRIVER_SAMPLE_MAX_REL_DEV).
    Call under ``torch.no_grad()``; samples are constants.
    """
    if not (0.0 <= float(max_rel_dev) < 1.0):
        raise ValueError("max_rel_dev must be in [0, 1)")
    if not hasattr(model, "encode_drivers"):
        raise ValueError("driver sampling needs a world model with encode_drivers")
    encoded, embedding, params = model.encode_drivers(dict(state), **history_args)
    sampled = {}
    if params is not None:
        names = tuple(params)
        values = torch.stack([params[name] for name in names], -1)          # [B, K, P]
        draws = normal((n_samples, *values.shape), values, ("parameters", names))
        # Physical-unit encoder uncertainty; cap excursions to +/-max_rel_dev of the encoded value.
        scale = encoded["uncertainty"] if encoded is not None else values.abs() * 0.1
        samples = values + draws * scale * driver_noise_scale
        samples = torch.maximum(torch.minimum(samples, values * (1.0 + max_rel_dev)),
                                values * (1.0 - max_rel_dev))
        sampled["driver_params"] = {name: samples[..., k] for k, name in enumerate(names)}
    if embedding is not None:
        draws = normal((n_samples, *embedding.shape), embedding, "embedding")
        sampled["driver_embedding"] = embedding + draws * embedding_noise_std * driver_noise_scale
    if not sampled:
        raise ValueError("world model must expose driver parameters or embedding")
    return sampled


def lane_uniform_draws(models, n_starts: int, n_steps: int, n_slots: int, seed,
                       device=None) -> Optional[Tensor]:
    """LANEFIX: per-start lane-decision uniforms ``[B, T, K]`` for world models that sample
    their HDV lane decisions (``lane_decision='sample'``), else ``None``.

    Drawn once per START from a private generator seeded by ``seed`` (the global RNG is not
    touched); the caller shares them across ensemble members and expands them to the
    counterfactual rows of a start (common random numbers, like the acceleration noise).
    """
    if not any(getattr(model, "_samples_lanes", False) for model in models):
        return None
    generator = torch.Generator(device="cpu").manual_seed(seed_of("lanes", seed))
    draws = torch.rand(int(n_starts), int(n_steps), int(n_slots), generator=generator)
    return draws.to(device) if device is not None else draws


def arm_major_lane_uniform(models, n_arms: int, n_samples: int, n_steps: int, n_slots: int,
                           generator: torch.Generator) -> dict:
    """LANEFIX: ``{"lane_uniform": [n_arms * n_samples, T, K]}`` for arm-major / sample-minor
    candidate batches (every arm sees the same draws per sample index: common random numbers)
    when a world model samples its lane decisions, else ``{}`` (no generator draw)."""
    if not any(getattr(model, "_samples_lanes", False) for model in models):
        return {}
    draws = torch.rand(int(n_samples), int(n_steps), int(n_slots), generator=generator)
    return {"lane_uniform": draws.repeat(int(n_arms), 1, 1)}


def draw_accel_noise(n_samples: int, shape, generator: torch.Generator, *, scale: float = 1.0,
                     device=None) -> Tensor:
    """Standard-normal HDV acceleration draws ``[n_samples, B, T, K]`` (``shape`` = (B, T, K))."""
    noise = torch.randn(n_samples, *shape, generator=generator) * scale
    return noise.to(device) if device is not None else noise


class SampledFutures:
    """Per-call plan of sampled futures for ONE batch of starts.

    ``plan(models, state, rollout_kwargs, n_steps)`` returns ``extras[member][sample]``: dicts of
    start-level (``[B, ...]``) keyword arguments for ``model.rollout`` -- ``driver_params`` /
    ``driver_embedding`` and, when ``accel_noise`` is on, ``accel_noise [B, T, K]`` +
    ``accel_noise_sigma_max``.  The caller expands them to counterfactual rows with the same row
    index it uses for the start state (so every variant of a start shares its sample), and every
    member shares the same noise draws.  ``seed`` fixes the generator: pass a per-update seed for
    training and a fixed one for validation.
    """

    def __init__(self, config: FuturesConfig, seed: Optional[int] = None):
        self.config = config
        self.generator = torch.Generator(device="cpu").manual_seed(
            seed_of("futures", config.seed, 0 if seed is None else seed))
        self.draws = NormalDraws(self.generator)

    @torch.no_grad()
    def plan(self, models, state: Mapping, rollout_kwargs: Mapping, n_steps: int):
        cfg = self.config
        n = cfg.n_samples
        history_args = {key: rollout_kwargs[key] for key in HISTORY_KEYS if key in rollout_kwargs}
        reference = state["s_m"]
        noise = None
        if cfg.accel_noise:
            b, k = reference.shape[:2]
            noise = draw_accel_noise(n, (b, int(n_steps), k), self.generator,
                                     scale=cfg.accel_noise_scale, device=reference.device)
        extras = []
        for model in models:
            per_sample = [{} for _ in range(n)]
            if cfg.samples_drivers:
                sampled = sample_driver_encodings(
                    model, state, history_args, n, self.draws,
                    driver_noise_scale=cfg.driver_noise_scale,
                    embedding_noise_std=cfg.embedding_noise_std)
                for s in range(n):
                    if "driver_params" in sampled:
                        per_sample[s]["driver_params"] = {
                            name: value[s] for name, value in sampled["driver_params"].items()}
                    if "driver_embedding" in sampled:
                        per_sample[s]["driver_embedding"] = sampled["driver_embedding"][s]
            if noise is not None:
                for s in range(n):
                    per_sample[s]["accel_noise"] = noise[s]
                    per_sample[s]["accel_noise_sigma_max"] = cfg.accel_sigma_max
            extras.append(per_sample)
        return extras
