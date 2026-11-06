"""In-memory, critic-only adapter for authoritative real negative-J returns.

No rewards are reconstructed here. Providers own the recorded continuation and
terminal/truncation semantics. A return is the sum of authoritative negative
total-vehicle-second rewards (including source waiting), discounted with ``gamma``
once per CONTROL step. Use gamma=1 for undiscounted -J and configure the trainer
likewise. No per-vehicle averaging, reward shaping, symlog or unit conversion is
applied. Imagined rewards must use this same authoritative reward definition.

Strict NPZ contract: all eight ``state__`` fields in ``_STATE_NAMES`` are required,
with vehicle fields [N,K] and entry_backlog [N,2]. Metadata declares TRAIN,
native/random_direct/current_policy, fixed exogenous requested_population [N],
300..600 second continuations, gamma=1 and the accounting/terminal constants below.
Each ledger column is ONE window's remaining continuation, including initially
queued, uninserted and later-entering requests through its declared endpoint.
Step costs are already vehicle-seconds: sum them, never multiply by dt again or
divide by admitted/network/requested counts. Optional paired raw ledgers permit
target recomputation. Seed metadata excludes 964000..964199. Source/provenance
declarations cannot prove population exogeneity, ledger completeness, current-
policy consistency, calibration or scientific qualification.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np
import torch
from torch import Tensor

from .policy import TeamCritic
from .start_states import checked_path


RETURN_UNITS = "negative_total_vehicle_seconds"
ACCOUNTING = "source_integral_plus_road_delay"
TERMINAL_SEMANTICS = "finite_horizon_charged_to_end"
_STATE_NAMES = ("s_m", "v_mps", "a_mps2", "lane", "present", "is_cav", "length_m",
                "entry_backlog")
_SOURCE_KINDS = ("native", "random_direct", "current_policy")
_PROHIBITED = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth", re.I)


@dataclass(frozen=True)
class RealReturnBatch:
    """Flat critic state tensors ``[B,...]`` and raw continuation returns ``[B]``.

    ``gamma`` is explicit to prevent fitting undiscounted -J to a discounted
    imagined value function. ``detached`` validates and cuts all upstream graphs,
    including when a provider accidentally supplies graph-bearing tensors.
    """

    states: Mapping[str, Tensor]
    returns: Tensor
    gamma: float
    units: str = RETURN_UNITS

    def detached(self, *, device, gamma: float) -> RealReturnBatch:
        if self.units != RETURN_UNITS:
            raise ValueError(f"real returns must have units {RETURN_UNITS!r}")
        if (isinstance(self.gamma, bool) or not isinstance(self.gamma, (int, float))
                or not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1
                or not math.isclose(self.gamma, gamma, rel_tol=0.0, abs_tol=1e-12)):
            raise ValueError("real-return gamma must match the trainer's control-step gamma")
        target = self.returns
        if (not torch.is_tensor(target) or not target.is_floating_point()
                or target.ndim != 1 or target.numel() == 0):
            raise ValueError("real returns must be a nonempty floating tensor [B]")
        target = target.detach().to(device)
        if not bool(torch.isfinite(target).all()) or bool((target > 0).any()):
            raise ValueError("authoritative negative-J returns must be finite and nonpositive")
        if not isinstance(self.states, Mapping) or not self.states:
            raise ValueError("real states must be a nonempty mapping of tensors [B,...]")
        states = {}
        for name, value in self.states.items():
            if (not isinstance(name, str) or not torch.is_tensor(value)
                    or value.ndim == 0 or value.shape[0] != target.shape[0]):
                raise ValueError("real state tensors must have the same batch dimension as returns")
            states[name] = value.detach().to(device)
        return RealReturnBatch(states, target, self.gamma, self.units)


class RealReturnProvider(Protocol):
    """Called once per enabled update under no_grad; provider state is caller-owned.

    ``update`` is one-based. ``batch_size`` is the requested size; smaller or larger
    nonempty batches are allowed and independently mean-reduced in the regression.
    Providers must return recorded critic states, never imagined/teacher actions.
    """

    def __call__(self, *, batch_size: int, device: torch.device,
                 update: int) -> RealReturnBatch: ...


class ScaledTeamCritic(TeamCritic):
    """TeamCritic with raw-vehicle-second output = base value * value_scale.

    The fixed, checkpointed buffer belongs only to this subclass. Scaling keeps
    the learned head near unit magnitude while retaining raw bootstrap values
    and analytic state derivatives. Pair with an appropriate residual scale.
    """

    def __init__(self, hidden_dim=64, n_heads=4, n_layers=2, entry_backlog=False, *,
                 value_scale=1.0):
        if (isinstance(value_scale, bool) or not isinstance(value_scale, (int, float))
                or not math.isfinite(value_scale) or value_scale <= 0):
            raise ValueError("value_scale must be a finite positive number")
        super().__init__(hidden_dim, n_heads, n_layers, entry_backlog)
        self.register_buffer("value_scale", torch.tensor(float(value_scale), dtype=torch.float64))

    def forward(self, state: Mapping[str, Tensor]) -> Tensor:
        return super().forward(state) * self.value_scale


def warm_start_real_critic(trainer, updates, batch_size=None):
    """Fit only the online critic, then hard-copy its complete state to the target.

    Calls the injected provider under no_grad at indices 1..updates, independently
    of actor update_count and real_return_mix. Uses the existing critic optimizer
    and clipping limit, dividing residuals ONLY by config.real_return_scale; it
    neither reads nor updates imagined-return statistics. Returns JSON-ready
    first/last raw and scaled training MSEs, last pre-clip norm and finiteness.
    First MSE is before fitting; last is after the final step on that step's batch
    (different sampled batches need not have comparable MSEs). Zero updates is a
    no-op. Critic module modes are restored; actor/WM/samplers/duals are untouched.
    """
    if isinstance(updates, bool) or not isinstance(updates, Integral) or updates < 0:
        raise ValueError("updates must be a nonnegative integer")
    size = trainer.config.batch_size if batch_size is None else batch_size
    if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
        raise ValueError("batch_size must be a positive integer")
    scale = trainer.config.real_return_scale
    if (isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError("real_return_scale must be a finite positive number")
    log = {"updates": int(updates), "batch_size": int(size), "real_return_scale": float(scale),
           "first_mse_raw": None, "first_mse_scaled": None,
           "last_mse_raw": None, "last_mse_scaled": None,
           "last_grad_norm": None, "finite": True}
    if updates == 0:
        return log
    if not callable(trainer.real_return_provider):
        raise ValueError("warm_start_real_critic requires a real_return_provider")

    def loss_and_metrics(batch):
        prediction = trainer.critic(batch.states)
        if prediction.shape != batch.returns.shape:
            raise ValueError("real critic predictions must have shape [B], matching real returns")
        residual = prediction - batch.returns.to(prediction)
        loss = (residual / scale).square().mean()
        raw = float(residual.detach().double().square().mean())
        scaled = float(loss.detach())
        if not math.isfinite(raw) or not math.isfinite(scaled):
            raise ValueError("nonfinite real critic warm-start MSE")
        return loss, raw, scaled

    modes = [(module, module.training) for module in trainer.critic.modules()]
    try:
        trainer.critic.train()
        for update in range(1, int(updates) + 1):
            with torch.no_grad():
                batch = trainer.real_return_provider(batch_size=int(size), device=trainer.device,
                                                       update=update)
                if not isinstance(batch, RealReturnBatch):
                    raise ValueError("real_return_provider must return a RealReturnBatch")
                batch = batch.detached(device=trainer.device, gamma=trainer.config.gamma)
            with torch.enable_grad():
                loss, raw, scaled = loss_and_metrics(batch)
                if update == 1:
                    log.update(first_mse_raw=raw, first_mse_scaled=scaled)
                trainer.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(trainer.critic.parameters(),
                                                      trainer.config.grad_clip,
                                                      error_if_nonfinite=True)
                trainer.critic_optimizer.step()
            if update == updates:
                with torch.no_grad():
                    _, raw, scaled = loss_and_metrics(batch)
                log.update(last_mse_raw=raw, last_mse_scaled=scaled, last_grad_norm=float(norm))
        trainer.target_critic.load_state_dict(trainer.critic.state_dict(), strict=True)
    finally:
        trainer.critic_optimizer.zero_grad(set_to_none=True)
        for module, mode in modes:
            module.training = mode
    return log


def _checked_npz_path(path):
    # Check both spellings, including resolved symlinks; the shared guard supplies
    # workspace/approved-root enforcement. No file is opened before these checks.
    _check_text(str(path))
    path = checked_path(path)
    _check_text(str(path))
    if path.suffix.lower() != ".npz":
        raise ValueError("real-return path must end in .npz")
    return path


def _check_text(value):
    if _PROHIBITED.search(value):
        raise ValueError("restricted real-return path/provenance")


def _scalar(data, name, *, text=False):
    value = np.asarray(data[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    if text:
        if value.dtype.kind not in "US":
            raise ValueError(f"{name} must be a text scalar")
        value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        _check_text(value)
        return value
    if value.dtype.kind not in "fiu" or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric scalar")
    return value.item()


def _numeric(value, name, shape=None):
    value = np.asarray(value)
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite real numbers")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return value


def _ledger(data, n):
    road = _numeric(data["road_delay_steps"], "road_delay_steps")
    source = _numeric(data["source_wait_steps"], "source_wait_steps")
    if road.ndim != 2 or not 600 <= road.shape[0] <= 1200 or road.shape[1] != n:
        raise ValueError("road_delay_steps must have shape [T,N], T in 600..1200")
    if source.shape != road.shape:
        raise ValueError("source_wait_steps must match road_delay_steps [T,N]")
    if (road < 0).any() or (source < 0).any():
        raise ValueError("ledger step costs must be nonnegative vehicle-seconds")
    if _scalar(data, "physics_dt") != 0.5:
        raise ValueError("physics_dt must be 0.5 seconds")
    # float64 accumulation preserves raw totals and avoids integer accumulation
    # overflow. dt is metadata only: these arrays are already integrated costs.
    with np.errstate(over="ignore", invalid="ignore"):
        returns = -(road.sum(0, dtype=np.float64) + source.sum(0, dtype=np.float64))
    return returns, road.shape[0] * 0.5


def _validate_npz(data):
    required = {f"state__{name}" for name in _STATE_NAMES} | {
        "returns", "gamma", "units", "split", "source_kind", "requested_population", "horizon_s",
        "accounting", "terminal_semantics"}
    optional = {"policy_checkpoint", "seed", "seeds", "road_delay_steps", "source_wait_steps",
                "physics_dt"}
    missing, unknown = required - data.keys(), data.keys() - required - optional
    if missing or unknown:
        raise ValueError(f"invalid real-return keys: missing={sorted(missing)}, unknown={sorted(unknown)}")
    provenance = {}
    for name, expected in (("units", RETURN_UNITS), ("split", "TRAIN"),
                           ("accounting", ACCOUNTING), ("terminal_semantics", TERMINAL_SEMANTICS)):
        provenance[name] = _scalar(data, name, text=True)
        if provenance[name] != expected:
            raise ValueError(f"{name} must be {expected!r}")
    if _scalar(data, "gamma") != 1:
        raise ValueError("strict real returns require gamma=1")
    provenance["gamma"] = 1.0
    kind = provenance["source_kind"] = _scalar(data, "source_kind", text=True)
    if kind not in _SOURCE_KINDS:
        raise ValueError(f"source_kind must be one of {_SOURCE_KINDS}")
    if kind == "current_policy" and "policy_checkpoint" not in data:
        raise ValueError("current_policy requires policy_checkpoint")
    if "policy_checkpoint" in data:
        checkpoint = _scalar(data, "policy_checkpoint", text=True)
        if not checkpoint.strip():
            raise ValueError("policy_checkpoint must be nonempty")
        provenance["policy_checkpoint"] = checkpoint
    targets = _numeric(data["returns"], "returns")
    if targets.ndim != 1 or targets.size == 0 or not (targets < 0).all():
        raise ValueError("returns must be nonempty strictly negative [N]")
    n = targets.size
    for name in ("requested_population", "horizon_s"):
        value = _numeric(data[name], name, (n,))
        if name == "requested_population":
            if not ((value > 0) & (value == np.floor(value))).all():
                raise ValueError("requested_population must be positive exogenous request counts [N]")
        elif not ((value >= 300) & (value <= 600)).all():
            raise ValueError("horizon_s must be in [300,600]")
        provenance[name] = value.copy()
    for name in ("seed", "seeds"):
        if name in data:
            seeds = np.asarray(data[name])
            if (seeds.dtype.kind not in "iu" or seeds.shape not in ((), (n,))
                    or (seeds < 0).any()):
                raise ValueError(f"{name} metadata must be nonnegative integers, scalar or [N]")
            if ((seeds >= 964000) & (seeds <= 964199)).any():
                raise ValueError("forbidden traffic seed in real-return metadata")
            provenance[name] = seeds.copy()
    positions = _numeric(data["state__s_m"], "state__s_m")
    if positions.ndim != 2 or positions.shape[0] != n or positions.shape[1] < 1:
        raise ValueError("state__s_m must have shape [N,K], K > 0")
    states = {}
    for name in _STATE_NAMES:
        value = np.asarray(data[f"state__{name}"])
        shape = (n, 2) if name == "entry_backlog" else positions.shape
        if name in ("present", "is_cav"):
            if (value.shape != shape or value.dtype.kind not in "biu"
                    or not ((value == 0) | (value == 1)).all()):
                raise ValueError(f"state__{name} must be binary [N,K]")
            states[name] = torch.from_numpy(value.astype(np.bool_, copy=True))
        elif name == "lane":
            if (value.shape != shape or value.dtype.kind not in "iu"
                    or not ((value >= -1) & (value <= 2)).all()):
                raise ValueError("state__lane must be integer [N,K] in -1..2")
            states[name] = torch.from_numpy(value.astype(np.int64, copy=True))
        else:
            value = _numeric(value, f"state__{name}", shape)
            if (np.abs(value.astype(np.float64)) > np.finfo(np.float32).max).any():
                raise ValueError(f"state__{name} must be representable as finite float32")
            if name in ("v_mps", "length_m", "entry_backlog") and (value < 0).any():
                raise ValueError(f"state__{name} must be nonnegative")
            states[name] = torch.from_numpy(value.astype(np.float32, copy=True))
    present = states["present"]
    # Recorded junction transits are present with lane=-1; do not drop their cost
    # or fabricate a lane. TeamCritic explicitly supports a zero lane one-hot.
    if (states["length_m"][present] <= 0).any():
        raise ValueError("present vehicles need positive length_m")
    ledger_keys = {"road_delay_steps", "source_wait_steps"}
    if ledger_keys.intersection(data):
        if not ledger_keys.issubset(data) or "physics_dt" not in data:
            raise ValueError("raw ledgers require road_delay_steps, source_wait_steps and physics_dt")
        recomputed, horizon = _ledger(data, n)
        if not np.isfinite(recomputed).all() or not np.allclose(
                recomputed, targets, rtol=1e-7, atol=1e-8):
            raise ValueError("returns do not match raw vehicle-second ledger totals")
        if not (provenance["horizon_s"] == horizon).all():
            raise ValueError("horizon_s does not match ledger T * physics_dt")
    if "physics_dt" in data:
        if _scalar(data, "physics_dt") != 0.5:
            raise ValueError("physics_dt must be 0.5 seconds")
        provenance["physics_dt"] = 0.5
    provenance["ledger_recomputed"] = ledger_keys.issubset(data)
    return states, torch.from_numpy(targets.astype(np.float64, copy=True)), provenance


class NPZRealReturnProvider:
    """Strict TRAIN-only provider; detached CPU cache, stateless replacement sampling.

    ``default_rng(seed + update)`` makes every requested update reproducible without
    consuming global numpy/torch RNGs or saving provider state. Optional ``seed`` /
    ``seeds`` NPZ metadata describes traffic worlds, not this sampling RNG seed.
    ``provenance`` exposes validated declarations, not qualification evidence.
    """

    def __init__(self, path, seed=0):
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("sampling seed must be a nonnegative integer")
        self.path = _checked_npz_path(path)
        self.seed = int(seed)
        with np.load(self.path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError("duplicate real-return NPZ keys")
            data = {name: archive[name] for name in archive.files}
        self._states, self._returns, self.provenance = _validate_npz(data)

    def __call__(self, *, batch_size: int, device: torch.device,
                 update: int) -> RealReturnBatch:
        for name, value in (("batch_size", batch_size), ("update", update)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        rng = np.random.default_rng(self.seed + int(update))
        index = torch.from_numpy(rng.integers(self._returns.numel(), size=int(batch_size)))
        return RealReturnBatch({name: value.index_select(0, index).to(device)
                                for name, value in self._states.items()},
                               self._returns.index_select(0, index).to(device), 1.0)


def write_real_returns_npz(path, states, road_delay_steps, source_wait_steps,
                           requested_population, physics_dt=0.5, *, source_kind,
                           policy_checkpoint=None) -> Path:
    """Write validated gamma=1 targets and raw authoritative continuation ledgers.

    ``states`` uses unprefixed critic field names. Ledgers are nonnegative [T,N]
    vehicle-seconds, T=600..1200, physics_dt=0.5. Each column starts at its critic
    state and is charged through its finite endpoint. The caller supplies ALL
    source waiting (even never-inserted requests) and road delay (including later
    entrants), and fixed policy-independent requested counts. Nothing is inferred
    from the network population. Parent directories must already exist.
    """
    path = _checked_npz_path(path)
    if not isinstance(states, Mapping):
        raise ValueError("states must be a mapping of critic field names")

    def array(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

    data = {f"state__{name}": array(value) for name, value in states.items()}
    population = array(requested_population)
    if population.ndim != 1 or not population.size:
        raise ValueError("requested_population must have shape [N], N > 0")
    data.update(road_delay_steps=array(road_delay_steps), source_wait_steps=array(source_wait_steps),
                physics_dt=np.asarray(physics_dt))
    targets, horizon = _ledger(data, population.size)
    data.update(returns=targets, gamma=np.asarray(1.0), units=np.asarray(RETURN_UNITS),
                split=np.asarray("TRAIN"), source_kind=np.asarray(source_kind),
                requested_population=population, horizon_s=np.full(population.size, horizon),
                accounting=np.asarray(ACCOUNTING), terminal_semantics=np.asarray(TERMINAL_SEMANTICS))
    if policy_checkpoint is not None:
        data["policy_checkpoint"] = np.asarray(policy_checkpoint)
    _validate_npz(data)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **data)
    return path
