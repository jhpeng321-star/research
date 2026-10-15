"""Chunked truncated-BPTT unroll: one forward (and optionally backward) pass of a batch.

The runner owns the chunk loop and the log record; the loss arithmetic lives in
``wms.train.losses``.  Backward is called every ``truncation_steps`` steps so
activation memory is bounded by one chunk regardless of the unroll length, and all
per-car terms share one global valid-count normaliser.  Logged loss terms are
per-chunk contributions to that normaliser (horizon-invariant); the traffic term,
a per-chunk mean, is logged as the mean over chunks.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch
from torch import Tensor

from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.stability import ConservationConfig, ConservationTracker

from .diagnostics import GradientShares, driver_param_stats
from .inputs import UnrollInputs, build_targets, prepare_inputs
from .losses.base import LossContext, TermValue, UnrollContext, UnrollTargets
from .losses.compose import LossComposer, hybrid_residual_weight
from .losses.driver_prior import driver_prior_loss, prior_mask, prior_targets
from .settings import TrainSettings

#: Chunk-term names that always appear in the log record (zero until first added).
ALWAYS_LOGGED = ("position", "speed", "lane", "residual_mag", "lane_rate", "flip_rate",
                 "traffic")
#: Step offsets at which the monotone gate curve is logged.
GATE_PROBE_K = (0, 1, 2, 5, 10, 20, 40, 80, 160, 240)
GATE_LOG_K = (10, 20, 60, 120, 240)
_PROJECTION_ACTIVE_KEYS = ("gap_active", "speed_high", "speed_low", "invalid_lane")


@dataclass
class _Accumulators:
    """Running sums of one unroll (device scalars are summed in float64)."""

    sums: dict[str, Tensor] = field(default_factory=dict)
    total: float = 0.0
    raw_samples: list[Tensor] = field(default_factory=list)
    rate_samples: list[Tensor] = field(default_factory=list)
    projection: dict[str, float] = field(default_factory=dict)
    entrant_count: int = 0
    driver_encoded: Optional[Tensor] = None
    hybrid_residual: Optional[Tensor] = None
    hybrid_physics: Optional[Tensor] = None
    hybrid_count: int = 0

    def add(self, name: str, value: Tensor) -> None:
        value = value.detach().double()
        self.sums[name] = value if name not in self.sums else self.sums[name] + value


@dataclass
class _DriverGradLeaves:
    """FIX 1 bookkeeping: the one t0 encoder graph plus detached leaf tensors.

    Every chunk consumes the leaves; because ``backward`` accumulates into leaf
    ``.grad``, the sum of all chunk gradients w.r.t. the encoding is available after
    the last chunk and pushed through the single encoder graph exactly once.
    """

    values: Optional[Tensor]
    embedding: Optional[Tensor]
    values_leaf: Optional[Tensor] = None
    embedding_leaf: Optional[Tensor] = None

    @classmethod
    def encode_once(cls, model: Any, inp: UnrollInputs) -> Optional["_DriverGradLeaves"]:
        """Encode at t0 exactly as inference does; build leaves when grad is on."""
        encoded, embedding, _ = model.encode_drivers(
            inp.initial, driver_history=inp.observed_hist,
            driver_history_mask=inp.observed_hist_mask)
        if encoded is None:
            return None
        values = encoded["values"]
        leaves = cls(values=values, embedding=embedding)
        if torch.is_grad_enabled() and values.requires_grad:
            leaves.values_leaf = values.detach().requires_grad_(True)
            if embedding is not None and embedding.requires_grad:
                leaves.embedding_leaf = embedding.detach().requires_grad_(True)
        return leaves

    def chunk_inputs(self) -> tuple[Optional[dict[str, Tensor]], Optional[Tensor]]:
        """``(driver_params, driver_embedding)`` every chunk receives (leaves if grad)."""
        values = self.values_leaf if self.values_leaf is not None else self.values.detach()
        params = {name: values[..., i] for i, name in enumerate(PARAMETER_NAMES)}
        embedding = (self.embedding_leaf if self.embedding_leaf is not None
                     else (None if self.embedding is None else self.embedding.detach()))
        return params, embedding

    @property
    def leaf_parameters(self) -> list[Tensor]:
        return [leaf for leaf in (self.values_leaf, self.embedding_leaf)
                if leaf is not None]

    def backward_through_encoder(self) -> None:
        """Push the accumulated leaf gradients through the t0 encoder graph."""
        graphs: list[Tensor] = []
        grads: list[Tensor] = []
        for graph, leaf in ((self.values, self.values_leaf),
                            (self.embedding, self.embedding_leaf)):
            if leaf is None or leaf.grad is None:
                continue
            graphs.append(graph)
            grads.append(leaf.grad)
        if graphs:
            torch.autograd.backward(graphs, grads)


class UnrollRunner:
    """Run ``model.rollout`` over a batch and evaluate the composed loss."""

    def __init__(self, model: Any, settings: TrainSettings, composer: LossComposer, *,
                 projection: Optional[Any], device: torch.device) -> None:
        self.model, self.settings, self.composer = model, settings, composer
        self.projection, self.device = projection, device

    def run(self, batch: Mapping[str, Any], *, unroll_steps: int, backward: bool = True,
            share: bool = False, capture_shares: bool = False,
            truncation_steps: Optional[int] = None,
            _rate_probe: bool = False, _pair_probe: bool = False,
            _probe_training: bool = False) -> dict:
        """One pass; returns the log record (``loss`` plus every logged term and monitor).

        ``share`` (forward-only diagnostic) adds ``loss_share``: each term's share of the
        gradient norm on a single probe chunk.  ``capture_shares`` (with ``backward``)
        measures the shares during the real backward pass instead.  ``truncation_steps``
        overrides the configured chunk length (per-stage truncation, round 5).
        """
        if share and backward:
            raise ValueError("loss-share logging is a forward-only diagnostic (backward=False)")
        cfg = self.settings
        pair_loss = self.composer.paired_diff
        return_on = (pair_loss is not None and pair_loss.return_enabled and "pair_ref" in batch)
        pair_reference = None
        if return_on and torch.is_grad_enabled() and not _pair_probe:
            # The derivative of Huber/ranking is evaluated at the WHOLE return, while
            # replay retains only one TBPTT chunk's graph. Restore RNG for exact replay.
            devices = [self.device.index or 0] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                probe = self.run(batch, unroll_steps=unroll_steps, backward=False,
                                 truncation_steps=truncation_steps, _pair_probe=True,
                                 _probe_training=backward or _rate_probe)
            pair_reference = probe["_paired_return_difference"]
        rate_reference = None
        if cfg.lane_rate_weight and (backward or share) and not _rate_probe and not _pair_probe:
            # Replay identical RNG draws (dropout/state noise) without retaining a
            # whole-horizon graph: each BPTT chunk then receives the derivative of
            # the single whole-batch rate loss, not a mean of chunkwise squares.
            devices = [self.device.index or 0] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                probe = self.run(batch, unroll_steps=unroll_steps, backward=False,
                                 _rate_probe=backward)
            rate_reference = probe["predicted_lane_change_rate"]
        model = self.model
        inp = prepare_inputs(batch, model, cfg, self.device, unroll_steps)
        targets = build_targets(inp, batch, cfg, float(model.physics.dt_s), self.device)
        self.composer.prepare(targets)
        if pair_reference is not None:
            targets.prepared["paired_return_reference"] = pair_reference
        parameters = [p for p in model.parameters() if p.requires_grad]
        n_model_parameters = len(parameters)
        # Round 5 (FIX 1): encode the driver settings ONCE from the observed pre-t0
        # history at t0 -- exactly where inference encodes them -- and hand every
        # chunk detached leaf tensors (no per-chunk re-encode, no chunk-start
        # interaction drift); the accumulated leaf gradients go through the single
        # encoder graph after the last chunk (exact gradient, one encoder pass).
        driver_leaves = (_DriverGradLeaves.encode_once(model, inp)
                         if cfg.driver_grad_across_chunks else None)
        if driver_leaves is not None:
            parameters = parameters + driver_leaves.leaf_parameters
        if "lane_event" in self.composer.names:
            targets.prepared["lane_event"] = self._lane_event_inputs(inp, driver_leaves)
        # INFRAV5 (weight 0 = never evaluated): population-mean driver-settings prior.
        prior = (self._driver_prior(inp, driver_leaves, backward=backward)
                 if float(getattr(cfg, "driver_prior_weight", 0.0)) > 0.0
                 and driver_leaves is not None else None)
        # WMLANE C (training only): teacher-observed prefixes train the driver update.
        reveal_training = backward or _probe_training or (return_on and _rate_probe)
        reveal = (self._reveal_mask(inp) if reveal_training
                  and float(getattr(cfg, "driver_reveal_prob", 0.0)) > 0.0 else None)
        param_offset = embedding_offset = None
        measure = capture_shares and backward
        shares = GradientShares(parameters) if (measure or share) else None
        pair_grads = ([None] * len(parameters) if backward and targets.pair is not None
                      and pair_loss is not None else None)
        pair_difference = None
        return_total = 0.0
        conservation = (ConservationTracker(inp.present[:, 0], ConservationConfig(enabled=True))
                        if (cfg.conservation_enabled or cfg.conservation_weight) else None)
        acc = _Accumulators()
        for name in ALWAYS_LOGGED:
            if name in self.composer.names:
                acc.sums[name] = torch.zeros((), dtype=torch.float64, device=self.device)
        noisy = backward or _rate_probe or _probe_training
        chunk = max(1, int(truncation_steps if truncation_steps else cfg.truncation_steps))
        n_chunks = max(1, (targets.n_steps + chunk - 1) // chunk)
        probe_chunk = self._share_probe_chunk(targets.n_steps, chunk)
        current = inp.initial
        hist, hist_mask = inp.hist, inp.hist_mask
        driver_params = driver_embedding = None
        if driver_leaves is not None:
            acc.driver_encoded = driver_leaves.values.detach()
            driver_params, driver_embedding = driver_leaves.chunk_inputs()
        carry_driver = not bool(cfg.driver_grad_across_chunks)
        for c0 in range(0, targets.n_steps, chunk):
            c1 = min(targets.n_steps, c0 + chunk)
            # Forward-only share diagnostic: only the probe chunk needs its graph;
            # every other chunk runs without one (round-5 memory fix).
            grad_off = share and c0 != probe_chunk
            chunk_params, chunk_embedding = driver_params, driver_embedding
            observe_kwargs = {}
            if inp.paired_random is not None:
                pr = inp.paired_random
                times = range(c0, c1)
                if getattr(model, "_samples_lanes", False):
                    observe_kwargs["lane_uniform"] = pr.uniform("lane", times)
                if noisy and (cfg.state_noise_std_m or cfg.state_noise_std_mps):
                    observe_kwargs["state_noise"] = torch.stack(
                        (pr.normal("state_s", times), pr.normal("state_v", times)), -1)
            if reveal is not None:
                observe_kwargs.update(observe_states=inp.states[:, c0:c1 + 1],
                                      observe_mask=reveal[:, c0:c1 + 1])
                if param_offset is not None and chunk_params is not None:
                    chunk_params = {name: chunk_params[name] + param_offset[..., i]
                                    for i, name in enumerate(PARAMETER_NAMES)}
                if embedding_offset is not None and chunk_embedding is not None:
                    chunk_embedding = chunk_embedding + embedding_offset
            with torch.no_grad() if grad_off else contextlib.nullcontext():
                roll = model.rollout(
                    current, n_steps=c1 - c0, k0=c0,
                    cav_teacher=None if inp.teacher is None else inp.teacher[:, c0:c1 + 1],
                    present_seq=inp.present_seq[:, c0:c1 + 1],
                    is_cav_seq=inp.is_cav[:, c0:c1 + 1],
                    history=hist, history_mask=hist_mask, controller_id=inp.controller_id,
                    action_timeline=inp.action_timeline, cav_controller=inp.cav_controller,
                    state_noise_std_m=float(cfg.state_noise_std_m) if noisy else 0.0,
                    state_noise_std_mps=float(cfg.state_noise_std_mps) if noisy else 0.0,
                    projection=self.projection, entries=inp.entries,
                    driver_history=inp.observed_hist,
                    driver_history_mask=inp.observed_hist_mask,
                    driver_params=chunk_params, driver_embedding=chunk_embedding,
                    **observe_kwargs)
                self._collect_rollout(roll, acc, conservation, inp, c0, c1)
                if "driver_parameters_updated" in roll:
                    param_offset, embedding_offset = self._update_offsets(
                        roll, chunk_params, chunk_embedding, param_offset, embedding_offset)
                if acc.driver_encoded is None and "driver_parameters" in roll:
                    acc.driver_encoded = roll["driver_parameters"].detach()
                if carry_driver:
                    if driver_params is None and "driver_parameters" in roll:
                        driver_params = {name: acc.driver_encoded[..., i]
                                         for i, name in enumerate(PARAMETER_NAMES)}
                    if driver_embedding is None and "driver_embedding" in roll:
                        driver_embedding = roll["driver_embedding"].detach()
                ctx = LossContext(pred=roll, current=current, targets=targets, c0=c0, c1=c1,
                                  rate_reference=rate_reference)
                breakdown = self.composer(ctx)
                if return_on:
                    difference = pair_loss.predicted_return_difference(ctx).detach()
                    pair_difference = (difference if pair_difference is None
                                       else pair_difference + difference)
                    return_total += sum(float(breakdown.terms[n].value.detach())
                                        for n in pair_loss.return_names)
                raw = ctx.raw_residual
                raw = torch.zeros_like(roll["speeds"]) if raw is None else raw
                acc.raw_samples.append(raw.detach().masked_select(ctx.valid_aux).abs())
                acc.rate_samples.append(
                    ctx.change_probabilities.detach().masked_select(ctx.valid_aux))
                shareable = self.composer.shareable_chunk(breakdown)
                if measure:
                    for name, tensor in shareable.items():
                        shares.capture(name, tensor)
                if not bool(torch.isfinite(breakdown.total)):
                    raise RuntimeError("non-finite WM-S loss")
                if pair_grads is not None:
                    paired = sum(v.value for n, v in breakdown.terms.items()
                                 if n.startswith("pair_"))
                    if torch.is_tensor(paired) and paired.requires_grad:
                        grads = torch.autograd.grad(paired, parameters, retain_graph=True,
                                                    allow_unused=True)
                        for i, g in enumerate(grads):
                            if g is not None:
                                pair_grads[i] = (g.detach() if pair_grads[i] is None
                                                 else pair_grads[i] + g.detach())
                if backward:
                    breakdown.total.backward()
                for name, term_value in breakdown.terms.items():
                    acc.add(name, term_value.logged)
                if share and c0 == probe_chunk:
                    for name, tensor in shareable.items():
                        shares.probe(name, tensor)
                acc.total += float(breakdown.total.detach())
                current = {key: (value.detach() if torch.is_tensor(value) else value)
                           for key, value in roll["state"].items()}
                if inp.entries is None:
                    current["is_cav"] = inp.is_cav[:, c1]
                    current["length_m"] = inp.initial["length_m"]
                else:
                    # Recorded overrides own original slots only; the entry process owns
                    # newcomer type/length, including across BPTT boundaries.
                    newcomers = current["entrant"].bool()
                    current["is_cav"] = torch.where(newcomers, current["is_cav"],
                                                    inp.is_cav[:, c1])
                    current["length_m"] = torch.where(newcomers, current["length_m"],
                                                      inp.initial["length_m"])
                hist, hist_mask = roll["history"].detach(), roll["history_mask"]
        if return_on:
            values = pair_loss.return_losses(targets, pair_difference)
            acc.total -= return_total
            for name, value in values.items():
                acc.sums[name] = value.detach().double()
                acc.total += float(value.detach())
        record = self._finish(inp, targets, acc, conservation, shares, n_chunks,
                              backward=backward, share=share, measure=measure,
                              rate_reference=rate_reference)
        if targets.pair is not None and pair_loss is not None:
            self._pair_diagnostics(record, batch, targets, pair_loss)
            record["paired_grad_norm"] = (self._paired_grad_norm(
                pair_grads, parameters[:n_model_parameters], driver_leaves)
                if pair_grads is not None else None)
        if _pair_probe:
            record["_paired_return_difference"] = pair_difference
        if prior is not None:
            self._apply_driver_prior(prior, driver_leaves, record)
        if backward and driver_leaves is not None:
            driver_leaves.backward_through_encoder()
        return record

    @staticmethod
    def _paired_grad_norm(grads, parameters, leaves) -> float:
        """Norm of the summed paired gradient on MODEL parameters (including the encoder)."""
        n = len(parameters)
        if leaves is not None and leaves.leaf_parameters:
            graphs, upstream = [], []
            for graph, leaf in ((leaves.values, leaves.values_leaf),
                                (leaves.embedding, leaves.embedding_leaf)):
                if leaf is not None:
                    g = grads[n + len(upstream)]
                    graphs.append(graph)
                    upstream.append(torch.zeros_like(leaf) if g is None else g)
            extra = torch.autograd.grad(graphs, parameters, grad_outputs=upstream,
                                        retain_graph=True, allow_unused=True)
            for i, g in enumerate(extra):
                if g is not None:
                    grads[i] = g.detach() if grads[i] is None else grads[i] + g.detach()
        return sum(float(g.double().square().sum()) for g in grads[:n] if g is not None) ** 0.5

    @staticmethod
    def _pair_diagnostics(record, batch, targets, pair_loss) -> None:
        """Metadata only: no RNG draws or loss/gradient changes on the legacy paired path."""
        worlds = batch.get("pair_world")
        record["paired_unique_worlds"] = len(set(worlds)) if worlds is not None else None
        record["paired_groups"] = int(torch.unique(targets.pair.ref).numel())
        if targets.prepared.get("paired_return") is None:
            pair_loss._prepare_return(targets)
        effects = pair_loss.recorded_effect_bins(targets)
        bins = effects["effect_bin"][effects["active"]]
        record["paired_effect_bins"] = {
            "negative": int((bins < 0).sum()), "within_margin": int((bins == 0).sum()),
            "positive": int((bins > 0).sum())}
        record["paired_effect_margin_veh_s"] = pair_loss.cfg.rank_margin_veh_s

    def _driver_prior(self, inp: UnrollInputs, leaves: "_DriverGradLeaves", *,
                      backward: bool) -> dict:
        """INFRAV5 prior on the t0 encoding: value, batch means and (training) its leaf grad.

        The gradient is taken w.r.t. the encoder-output leaf and added to the chunk gradients
        before the single push through the encoder graph, so it reaches only the encoder."""
        cfg = self.settings
        leaf = leaves.values_leaf
        values = leaf if leaf is not None else leaves.values.detach()
        mask = prior_mask(inp.present[:, 0], inp.is_cav[:, 0], inp.observed_hist_mask)
        value, means = driver_prior_loss(
            values, mask, prior_targets(cfg), weight=float(cfg.driver_prior_weight),
            speed_limit_mps=float(self.model.physics.speed_limit_mps))
        grad = None
        if backward and leaf is not None and value.requires_grad:
            (grad,) = torch.autograd.grad(value, leaf)
        return {"value": float(value.detach()), "means": means, "grad": grad}

    @staticmethod
    def _apply_driver_prior(prior: dict, leaves: "_DriverGradLeaves", record: dict) -> None:
        """Log the prior (added to ``loss``) and accumulate its gradient into the leaf."""
        record["driver_prior"] = prior["value"]
        record["driver_prior_means"] = prior["means"]
        record["loss"] = float(record["loss"]) + prior["value"]
        grad = prior["grad"]
        if grad is None:
            return
        leaf = leaves.values_leaf
        total = grad if leaf.grad is None else leaf.grad + grad
        total_norm = float(total.norm())
        # share of the prior in the gradient arriving at the encoder output (t0 settings)
        record["driver_prior_grad_share"] = (float(grad.norm()) / total_norm
                                             if total_norm > 0.0 else 0.0)
        leaf.grad = total

    def _lane_event_inputs(self, inp: UnrollInputs,
                           driver_leaves: Optional[_DriverGradLeaves]) -> dict:
        """Driver inputs of the teacher-forced lane-event loss (the t0 encoding)."""
        model = self.model
        if getattr(model, "lane_v5", None) is not None:
            # LANEV5: no driver inputs; the pre-t0 lane age and accumulators only.
            kin0 = inp.states[:, 0]
            lane_age0 = model._initial_lane_age({}, inp.present[:, 0], kin0, inp.observed_hist,
                                                inp.observed_hist_mask)
            return {"lane_age0": lane_age0,
                    "lane_v5_acc0": model.initial_lane_v5_acc(
                        inp.present[:, 0], kin0, inp.observed_hist, inp.observed_hist_mask)}
        if driver_leaves is not None:
            params, embedding = driver_leaves.chunk_inputs()
        else:
            _, embedding, params = model.encode_drivers(
                inp.initial, driver_history=inp.observed_hist,
                driver_history_mask=inp.observed_hist_mask)
        kin0 = inp.states[:, 0]
        lane_age0 = model._initial_lane_age({}, inp.present[:, 0], kin0, inp.observed_hist,
                                            inp.observed_hist_mask)
        if lane_age0 is None:
            lane_age0 = torch.full(inp.present[:, 0].shape, 999.0, device=kin0.device,
                                   dtype=kin0.dtype)
        prep = {"driver_params": params, "driver_embedding": embedding,
                "lane_age0": lane_age0}
        if getattr(model, "_tracks_lane_blocked", False):
            # LANEFIX v2 features: pre-t0 blocked time from the same observed history.
            prep["lane_blocked0"] = model.initial_lane_blocked(
                inp.present[:, 0], kin0, inp.observed_hist, inp.observed_hist_mask, params)
        return prep

    def _reveal_mask(self, inp: UnrollInputs) -> Tensor:
        """Random teacher-observed prefixes ``[B, T+1, K]`` (WMLANE C training mode).

        Each t0 HDV is revealed with probability ``driver_reveal_prob`` for the first
        ``R ~ U{1..driver_reveal_max_s/dt}`` steps (never at index 0, never on CAVs
        or off-road/junction ticks).
        """
        cfg = self.settings
        b, t1, k = inp.states.shape[:3]
        device = inp.states.device
        dt = float(self.model.physics.dt_s)
        max_steps = max(1, int(round(float(cfg.driver_reveal_max_s) / dt)))
        hdv0 = inp.present[:, 0] & ~inp.is_cav[:, 0]
        if inp.paired_random is None:
            pick = (torch.rand(b, k, device=device) < float(cfg.driver_reveal_prob)) & hdv0
            length = torch.randint(1, max_steps + 1, (b, k), device=device)
        else:
            pick = (inp.paired_random.uniform("reveal_pick", [0])[:, 0]
                    < float(cfg.driver_reveal_prob)) & hdv0
            length = 1 + (inp.paired_random.uniform("reveal_length", [0])[:, 0]
                          * max_steps).long()
        steps = torch.arange(t1, device=device).view(1, t1, 1)
        mask = (steps >= 1) & (steps <= length.unsqueeze(1)) & pick.unsqueeze(1)
        return mask & inp.loss_valid[:, :t1] & ~inp.is_cav[:, :t1]

    @staticmethod
    def _update_offsets(roll: Mapping[str, Any], chunk_params, chunk_embedding,
                        param_offset, embedding_offset):
        """Carry in-rollout driver updates to the next BPTT chunk as detached offsets."""
        base = (torch.stack([chunk_params[name] for name in PARAMETER_NAMES], -1)
                if chunk_params is not None else roll["driver_parameters"])
        step = (roll["driver_parameters_updated"] - base).detach()
        param_offset = step if param_offset is None else param_offset + step
        if "driver_embedding_updated" in roll:
            base_e = chunk_embedding if chunk_embedding is not None \
                else roll.get("driver_embedding")
            if base_e is not None:
                step_e = (roll["driver_embedding_updated"] - base_e).detach()
                embedding_offset = step_e if embedding_offset is None \
                    else embedding_offset + step_e
        return param_offset, embedding_offset

    def _share_probe_chunk(self, n_steps: int, chunk: int) -> int:
        """First chunk at/after the traffic ramp saturates (every term live), else the last."""
        if self.settings.traffic_loss:
            ramp_end = int(round(float(self.settings.traffic_ramp_end_s)
                                 / float(self.model.physics.dt_s)))
            probe = (ramp_end // chunk) * chunk
            if 0 <= probe < n_steps:
                return probe
        return ((n_steps - 1) // chunk) * chunk

    def _collect_rollout(self, roll: Mapping[str, Any], acc: _Accumulators,
                         conservation: Optional[ConservationTracker], inp: UnrollInputs,
                         c0: int, c1: int) -> None:
        """Side statistics of one chunk: entrants, projection, conservation, hybrid magnitudes."""
        if "entrant" in roll:
            acc.entrant_count = max(acc.entrant_count, int(roll["entrant"].sum()))
        for key, value in (roll.get("projection_stats") or {}).items():
            acc.projection[key] = acc.projection.get(key, 0.0) + float(value)
        if conservation is not None:
            seq = None if inp.entrant_seq is None else inp.entrant_seq[:, c0:c1]
            conservation.track_rollout(roll["masks"], entrant_seq=seq)
        if self.model.rollout_mode == "hybrid" and roll.get("residual_magnitude") is not None:
            res, phys = roll["residual_magnitude"], roll["physics_accel_magnitude"]
            # Accumulate detached values only: these are diagnostics, and keeping
            # every chunk's graph alive through a whole stage (the forward-only
            # share pass runs no backward) was the ~20 GB retention found in the
            # round-5 review.  The chunk-local graph still carries the
            # ResidualPenalty term, which reads the roll tensors directly.
            acc.hybrid_residual = (res.detach() if acc.hybrid_residual is None
                                   else acc.hybrid_residual + res.detach())
            acc.hybrid_physics = (phys.detach() if acc.hybrid_physics is None
                                  else acc.hybrid_physics + phys.detach())
            acc.hybrid_count += int(roll.get("residual_count", 0))

    def _finish(self, inp: UnrollInputs, targets: UnrollTargets, acc: _Accumulators,
                conservation: Optional[ConservationTracker], shares: Optional[GradientShares],
                n_chunks: int, *, backward: bool, share: bool, measure: bool,
                rate_reference: Optional[float]) -> dict:
        """End-of-unroll terms, host transfer and the log record."""
        cfg, model, composer = self.settings, self.model, self.composer
        raw_values = torch.cat(acc.raw_samples)
        rate_values = torch.cat(acc.rate_samples)
        has_targets = targets.aux_count > 0
        predicted_rate = float(rate_values.mean()) if has_targets else 0.0
        late: dict[str, float] = {}
        if composer.lane_rate is not None and rate_reference is None:
            late["lane_rate"] = composer.lane_rate.final(predicted_rate, targets.aux_count)
            acc.total += late["lane_rate"]
        uctx = UnrollContext(model=model, initial=inp.initial, present0=inp.present[:, 0],
                             targets=targets, conservation=conservation)
        unroll_values = composer.evaluate_unroll(uctx)
        for name, value in unroll_values.items():
            if measure:
                shares.capture(name, value.value)
            if backward and composer.differentiable(name):
                value.value.backward()
            if share:
                shares.probe(name, value.value, keep_zero=True)
        extra = composer.extra_total(unroll_values, like=inp.states)
        logged = self._host_scalars(acc, targets, unroll_values)
        logged.update(late)
        if self.projection is not None:
            present_steps = inp.present[:, 1:targets.n_steps + 1]
            active = sum(acc.projection.get(k, 0.0) for k in _PROJECTION_ACTIVE_KEYS)
            denom_present = float(present_steps.sum().clamp_min(1.0).detach())
            logged.update(
                projection_active_total=float(active),
                projection_active_fraction=float(active) / denom_present,
                projection_gap_active=float(acc.projection.get("gap_active", 0.0)),
                projection_s_clamped=float(acc.projection.get("s_clamped", 0.0)),
                projection_invalid_lane=float(acc.projection.get("invalid_lane", 0.0)),
                projection_present_changed=float(acc.projection.get("present_changed", 0.0)))
        if conservation is not None:
            logged["conservation_metric"] = (max(conservation.history)
                                             if conservation.history else 0.0)
            logged["conservation_created"] = (float(conservation.created.max().detach())
                                              if conservation.created.numel() else 0.0)
        loss_logged = acc.total
        for name in composer.chunk_mean_terms():
            chunk_total = logged[name]
            logged[name] = chunk_total / n_chunks
            loss_logged = (loss_logged - chunk_total) + logged[name]
        loss_logged = loss_logged + float(extra.detach())
        gates = model.gate(torch.arange(targets.n_steps, device=self.device))
        curve = [float(x) for x in model.gate(torch.tensor(GATE_PROBE_K, device=self.device)
                                              ).detach()]
        record = {
            "loss": loss_logged, "gate_curve": curve, "predicted_lane_change_rate": predicted_rate,
            "raw_residual_abs_mean_mps2": float(raw_values.mean()) if has_targets else 0.0,
            "raw_residual_abs_p95_mps2": (float(torch.quantile(raw_values, .95))
                                          if has_targets else 0.0),
            **logged,
            "gate_g0": float(gates[0].detach()), "gate_g_last": float(gates[-1].detach()),
            "gate_monotone": bool(torch.all(gates[1:] >= gates[:-1])),
            "unroll_steps": int(targets.n_steps), "valid_targets": int(targets.hdv_mask.sum()),
            "driver_param_stats": driver_param_stats(acc.driver_encoded, targets.hdv_mask[:, 0]),
            "entrants": int(acc.entrant_count),
            "valid_hdv_slots_t0": int(targets.hdv_mask[:, 0].sum())}
        if getattr(model, "gate_kind", "cumsum") == "saturating":
            gate_log = model.gate(torch.tensor(GATE_LOG_K, device=self.device)).detach()
            record["gate_g_by_k"] = {
                str(k): float(value) for k, value in zip(GATE_LOG_K, gate_log)}
        if getattr(model, "lane_gate_kind", "shared") == "saturating":
            lane_gate_log = model.lane_gate(torch.tensor(GATE_LOG_K, device=self.device)
                                            ).detach()
            record["lane_gate_g_by_k"] = {
                str(k): float(value) for k, value in zip(GATE_LOG_K, lane_gate_log)}
        if cfg.traffic_loss:
            record["chunks"] = int(n_chunks)
        agg_names = [n for n in ("agg_field", "agg_reward", "agg_flow") if n in logged]
        if agg_names:
            # C4 aggregate loss: its share of the logged loss value (gradient shares are in
            # ``loss_share`` when loss-share logging is on).
            agg_total = sum(float(logged[n]) for n in agg_names)
            record["aggregate_value_share"] = (agg_total / loss_logged
                                               if loss_logged > 0.0 else 0.0)
        pair_names = [n for n in ("pair_pos", "pair_speed", "pair_agg", "pair_return", "pair_rank")
                      if n in logged]
        if pair_names:
            # PAIRLOSS: share of the logged loss value and whether this was a paired batch.
            pair_total = sum(float(logged[n]) for n in pair_names)
            record["paired_value_share"] = (pair_total / loss_logged
                                            if loss_logged > 0.0 else 0.0)
            record["paired_batch"] = targets.pair is not None
            if targets.pair is not None:
                record["paired_arms"] = int(targets.pair.arm.sum())
        if cfg.paired_diff_horizon_mode == "stage" and targets.pair is not None:
            # Use the actual horizon, including fallback to a shorter available bucket.
            record["paired_horizon_steps"] = int(targets.n_steps)
            record["paired_horizon_s"] = float(targets.n_steps * targets.dt_s)
        if model.rollout_mode == "hybrid" and acc.hybrid_count > 0:
            physics_mag, residual_mag = (float(acc.hybrid_physics.detach()),
                                         float(acc.hybrid_residual.detach()))
            total_mag = physics_mag + residual_mag
            record["physics_share"] = (physics_mag / total_mag) if total_mag > 0.0 else 0.0
            record["residual_share"] = (residual_mag / total_mag) if total_mag > 0.0 else 0.0
            record["residual_abs_mean_mps2"] = residual_mag / max(acc.hybrid_count, 1)
            record["residual_magnitude_weight"] = hybrid_residual_weight(cfg)
        if share:
            record["loss_share"] = shares.probed()
        if measure:
            shares_by_term, norms = shares.accumulated()
            record.update(loss_share=shares_by_term, loss_grad_norm=norms)
        return record

    def _host_scalars(self, acc: _Accumulators, targets: UnrollTargets,
                      unroll_values: Mapping[str, TermValue]) -> dict:
        """One host transfer for every summed term, the braking counts and unroll terms."""
        plan = targets.prepared.get("braking")
        names = list(acc.sums)
        tensors = [acc.sums[n] for n in names]
        for name, value in unroll_values.items():
            names.append(name)
            tensors.append(value.logged.detach().double())
        counts: list[str] = []
        if plan is not None:
            counts.append("brake_steps")
            tensors.append(plan.brake_steps.double())
            if plan.prop_mask is not None:
                counts.append("prop_steps")
                tensors.append(plan.prop_mask.sum().double())
        values = torch.stack(tensors).tolist() if tensors else []
        out = {name: float(v) for name, v in zip(names, values)}
        for name, v in zip(counts, values[len(names):]):
            out[name] = int(v)
        return out
