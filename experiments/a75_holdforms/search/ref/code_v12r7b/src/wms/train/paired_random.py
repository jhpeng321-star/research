"""Stateless common random numbers for paired training arms.

``uniform(stream, times)`` / ``normal(stream, times)`` return ``[B, T, K]`` on
the input device/dtype. ``times`` is a 1-D sequence of absolute integer unroll
steps (a scalar means one step, NOT a length). Stream names are arbitrary stable
strings. Requests can be sliced, reordered or repeated without changing draws
or consuming torch/numpy global RNG state. The training seed fixes the realization.

Runner recipe (only when ``inp.paired_random is not None``)::

    pr = inp.paired_random
    pick = pr.uniform("reveal_pick", [0])[:, 0] < probability
    length = 1 + (pr.uniform("reveal_length", [0])[:, 0] * max_steps).long()
    times = range(c0, c1)
    lane_uniform = pr.uniform("lane", times)
    state_noise = torch.stack((pr.normal("state_s", times),
                               pr.normal("state_v", times)), dim=-1)

Pass the last two tensors to ``model.rollout`` for that chunk. t0 perturbations
are already handled by prepare_inputs using separate ``t0_s`` / ``t0_v`` streams.
The same object must cover the whole unroll, including synthetic entry padding.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor


def _hash(*parts: Any) -> int:
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _mix(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finalizer, with intentional unsigned wraparound."""
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


class PairedRandom:
    """Identity-keyed draws; construct with :meth:`from_batch`.

    C4's collator retains ``pair_group`` and numeric ``vehicle_ids [B,K]`` but
    drops source metadata. Its ID contract is a positive episode prefix in the
    millions and an episode-local remainder. A row whose valid IDs all share a
    positive million prefix is decoded under that contract; other IDs are used
    verbatim. Generic callers with large numeric IDs should supply
    ``vehicle_ids_episode_encoded=False`` to explicitly disable C4 decoding.

    Unknown (-1) original slots must be absent throughout the recorded window;
    they receive one inert sentinel, never an assumed physical slot identity.
    Appended entry slots receive disjoint synthetic identities by padding ordinal.
    Permuting physical slots/rows and their IDs/groups permutes the output exactly.
    """

    def __init__(self, identities: list[list[Any]], groups: list[Any], *, seed: int,
                 device: torch.device, dtype: torch.dtype) -> None:
        self.device, self.dtype = device, dtype
        self.row_seeds = [_hash("paired-entry", int(seed), group) for group in groups]
        self._keys = np.asarray([
            [_hash("paired-vehicle", int(seed), group, identity) for identity in row]
            for group, row in zip(groups, identities)], dtype=np.uint64)

    @classmethod
    def from_batch(cls, batch: Mapping[str, Any], *, seed: int,
                   device: torch.device, dtype: torch.dtype = torch.float32,
                   pad: int = 0) -> "PairedRandom":
        if "vehicle_ids" not in batch:
            raise ValueError("couple_randomness requires stable vehicle_ids [B, K]")
        ids = torch.as_tensor(batch["vehicle_ids"]).detach().cpu()
        present = torch.as_tensor(batch["present"]).detach().cpu().bool()
        if ids.ndim != 2 or ids.shape != (present.shape[0], present.shape[2]):
            raise ValueError("couple_randomness requires stable vehicle_ids [B, K]")
        if ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex():
            raise ValueError("couple_randomness requires integer vehicle_ids")
        b_n = ids.shape[0]
        ref = torch.as_tensor(batch["pair_ref"]).detach().cpu()
        if (ref.shape != (b_n,) or ref.is_floating_point() or ref.is_complex()
                or ref.dtype == torch.bool or bool(((ref < 0) | (ref >= b_n)).any())):
            raise ValueError("couple_randomness requires valid pair_ref [B]")
        ref = ref.long()
        if not torch.equal(ref[ref], ref):
            raise ValueError("pair_ref must point to a self-referencing group reference")
        groups = batch.get("pair_key", batch.get("pair_group"))
        if groups is None:
            raise ValueError("couple_randomness requires stable pair_group or pair_key [B]")
        if torch.is_tensor(groups):
            groups = groups.detach().cpu().tolist()
        elif isinstance(groups, np.ndarray):
            groups = groups.tolist()
        if len(groups) != b_n or any(g is None for g in groups):
            raise ValueError("couple_randomness requires stable pair_group or pair_key [B]")
        if any(groups[b] != groups[int(ref[b])] for b in range(b_n)):
            raise ValueError("pair_group/pair_key disagrees with pair_ref")
        identities = []
        for b, row in enumerate(ids.tolist()):
            valid = [i for i in row if i >= 0]
            encoded = batch.get("vehicle_ids_episode_encoded", None)
            if encoded is None:
                prefixes = {i // 1_000_000 for i in valid}
                encoded = len(prefixes) == 1 and next(iter(prefixes)) > 0
            if encoded:
                row = [i % 1_000_000 if i >= 0 else i for i in row]
            known = [i for i in row if i >= 0]
            if len(set(known)) != len(known):
                raise ValueError("couple_randomness requires unique stable vehicle_ids per row")
            if any(i < 0 and bool(present[b, :, k].any()) for k, i in enumerate(row)):
                raise ValueError("couple_randomness: present vehicle has no stable vehicle_ids")
            identities.append([("vehicle", i) if i >= 0 else ("empty",) for i in row]
                              + [("entry", i) for i in range(int(pad))])
        return cls(identities, list(groups), seed=seed, device=device, dtype=dtype)

    def _uniform(self, stream: str, times: Any, domain: str) -> np.ndarray:
        if not isinstance(stream, str) or not stream:
            raise ValueError("paired random stream must be a nonempty string")
        if torch.is_tensor(times):
            times = times.detach().cpu().numpy()
        times = np.asarray(times)
        if times.ndim == 0:
            times = times.reshape(1)
        if times.ndim != 1 or (times.size and times.dtype.kind not in "iu"):
            raise ValueError("paired random times must be absolute integer step indices")
        time_keys = _mix(times.astype(np.uint64) ^ np.uint64(_hash(domain, stream)))
        bits = _mix(self._keys[:, None, :] ^ time_keys[None, :, None])
        # 52 random bits and midpoint quantization keep log inputs strictly positive.
        return ((bits >> np.uint64(12)).astype(np.float64) + 0.5) * (2.0 ** -52)

    def uniform(self, stream: str, times: Any) -> Tensor:
        """Uniform draws in [0,1), shape [B,len(times),K], without global RNG use."""
        out = torch.as_tensor(self._uniform(stream, times, "uniform"),
                              device=self.device, dtype=self.dtype)
        # Float conversion can round the largest values to 1 (including fp32).
        return out.clamp_max(1.0 - torch.finfo(self.dtype).eps)

    def normal(self, stream: str, times: Any) -> Tensor:
        """Standard-normal draws, shape [B,len(times),K], without global RNG use."""
        u = self._uniform(stream, times, "normal-radius")
        v = self._uniform(stream, times, "normal-angle")
        values = np.sqrt(-2.0 * np.log(u)) * np.cos(2.0 * np.pi * v)
        return torch.as_tensor(values, device=self.device, dtype=self.dtype)
