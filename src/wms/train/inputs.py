"""Batch -> unroll inputs: device placement, entries-v2 padding, t0 noise, recorded targets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from wms.data.action_timeline import CONTROLLERS

from .cav_drive import TimelineControllers
from .losses.base import UnrollTargets
from .losses.traffic import entrant_steps
from .paired_random import PairedRandom
from .settings import TrainSettings


def state_from_kin(kin: Tensor, present: Tensor) -> dict[str, Tensor]:
    """Scalar-field state mapping from a packed ``[B, K, 7]`` kinematics tensor."""
    lane = kin[..., 1:4].argmax(-1)
    lane = torch.where(kin[..., 1:4].sum(-1) > 0, lane, torch.full_like(lane, -1))
    return {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4], "a_mps2": kin[..., 5],
            "length_m": kin[..., 6], "lane": lane, "present": present}


def step_weights(steps: int, dtype: torch.dtype, device: torch.device, decay: float) -> Tensor:
    """Normalised exponentially decaying per-step weights."""
    w = torch.exp(-float(decay) * torch.arange(steps, device=device, dtype=dtype))
    return w / w.sum().clamp_min(1.0e-8)


#: Adaptive entry pad (``entry_pad_auto``): 1.5x the expected arrivals + 16, as the C4
#: imagination's ``wms.control.dreamer.with_entry_slots``.
ENTRY_PAD_SAFETY = 1.5
ENTRY_PAD_MARGIN = 16


def entry_pad_for(settings: TrainSettings, histories: list, horizon_s: float) -> int:
    """Entrant padding slots of one batch: fixed ``entry_pad`` or the adaptive size."""
    import numpy as np

    pad = int(settings.entry_pad)
    if not settings.entry_pad_auto:
        return pad
    need = max(int(np.ceil(float(np.sum(h.rate)) * float(horizon_s) * ENTRY_PAD_SAFETY))
               + ENTRY_PAD_MARGIN for h in histories)
    return max(1, min(int(settings.entry_pad_max), max(pad, need)))


def make_entry_process(settings: TrainSettings, batch: Mapping[str, Any], present: Tensor,
                       model: Any, device: torch.device,
                       unroll_steps: Optional[int] = None,
                       paired_random: Optional[PairedRandom] = None) -> Optional[Any]:
    """Seeded entries-v2 process over the free padding slots, or ``None``.

    Padding slots are the last ``pad`` columns (empty at t0); entrants occupy
    them and move by the model physics, but have no recorded target so ``loss_valid``
    excludes them from the per-car loss.  The fixed process seed makes a run
    deterministic; the per-window observed-history features keep the arrival
    process data-driven.  ``pad`` is ``entry_pad``, or with ``entry_pad_auto`` the
    rate x horizon size of :func:`entry_pad_for` (then blocked arrivals also stay pending
    without a cap, as in the C4 imagination); the process carries it as ``train_pad``.
    """
    if not settings.entries_v2 or "entry_feat" not in batch or int(settings.entry_pad) <= 0:
        return None
    import numpy as np

    from wms.model.entries import EntranceHistory, EntryConfig, TorchEntryProcessV2

    b_n, k_n = int(present.shape[0]), int(present.shape[2])
    feat = batch["entry_feat"].to(device).cpu().numpy()
    if feat.shape[1] != EntranceHistory.defaults().to_features().size:
        raise ValueError("entry_feat has an unexpected width")
    histories = [EntranceHistory.from_features(feat[b]) for b in range(b_n)]
    dt_s = float(model.physics.dt_s)
    steps = int(present.shape[1]) - 1 if unroll_steps is None else min(
        int(unroll_steps), int(present.shape[1]) - 1)
    pad = entry_pad_for(settings, histories, max(steps, 1) * dt_s)
    pool = np.zeros((b_n, k_n + pad), dtype=bool)
    pool[:, k_n:] = True
    cfg = EntryConfig(max_pending=10 ** 9) if settings.entry_pad_auto else None
    rule = str(getattr(settings, "entry_insert_rule", "v2"))
    if rule != "v2":                    # ENTRYFIX; "v2" keeps the legacy construction
        cfg = EntryConfig(**({"max_pending": 10 ** 9} if settings.entry_pad_auto else {}),
                          insert_rule=rule)
    process = TorchEntryProcessV2(histories=histories, dt_s=dt_s,
                                  seed=int(settings.seed), batch=b_n, pool=pool, cfg=cfg,
                                   **({} if paired_random is None else
                                      {"row_seeds": paired_random.row_seeds,
                                       "couple_draws": True}))
    process.train_pad = pad
    return process


def extend_for_entries(states: Tensor, present: Tensor, is_cav: Tensor,
                       teacher: Optional[Tensor], hist: Optional[Tensor],
                       hist_mask: Optional[Tensor], loss_valid: Tensor, pad: int):
    """Append ``pad`` padding slots to every per-slot tensor.

    Returns ``(states, present_init, present_seq, is_cav, teacher, hist, hist_mask,
    loss_valid)``.  ``present_init`` (the t0 snapshot) keeps the padding empty,
    ``present_seq`` marks it present so an entrant is not dropped by the recorded
    presence mask, and ``loss_valid`` is False there so entrants never enter the loss.
    """
    b_n, t_n = int(present.shape[0]), int(present.shape[1])
    device, dtype = present.device, states.dtype
    zeros_k = torch.zeros(b_n, t_n, pad, dtype=dtype, device=device)
    states = torch.cat((states, zeros_k.unsqueeze(-1).expand(-1, -1, -1, states.shape[-1])), dim=2)
    present_init = torch.cat((present, torch.zeros(b_n, t_n, pad, dtype=torch.bool,
                                                    device=device)), dim=2)
    present_seq = torch.cat((present, torch.ones(b_n, t_n, pad, dtype=torch.bool,
                                                 device=device)), dim=2)
    is_cav = torch.cat((is_cav, torch.zeros(b_n, t_n, pad, dtype=torch.bool, device=device)), dim=2)
    loss_valid = torch.cat((loss_valid, torch.zeros(b_n, t_n, pad, dtype=torch.bool,
                                                    device=device)), dim=2)
    if teacher is not None:
        teacher = torch.cat((teacher, torch.zeros(b_n, teacher.shape[1], pad, teacher.shape[-1],
                                                  dtype=teacher.dtype, device=device)), dim=2)
    if hist is not None:
        hist = torch.cat((hist, torch.zeros(b_n, pad, hist.shape[2], hist.shape[3],
                                            dtype=hist.dtype, device=device)), dim=1)
        hist_mask = torch.cat((hist_mask, torch.zeros(b_n, pad, hist_mask.shape[2],
                                                      dtype=torch.bool, device=device)), dim=1)
    return states, present_init, present_seq, is_cav, teacher, hist, hist_mask, loss_valid


@dataclass
class UnrollInputs:
    """Everything the chunk loop needs, on the training device."""

    states: Tensor
    present: Tensor                  # t0 snapshot (padding empty)
    present_seq: Tensor
    is_cav: Tensor
    teacher: Optional[Tensor]
    hist: Optional[Tensor]
    hist_mask: Optional[Tensor]
    observed_hist: Optional[Tensor]
    observed_hist_mask: Optional[Tensor]
    loss_valid: Tensor
    entries: Optional[Any]
    controller_id: Optional[Tensor]
    action_timeline: Optional[dict]
    cav_controller: Optional[Any]
    entrant_seq: Optional[Tensor]
    n_steps: int
    initial: dict[str, Tensor]
    paired_random: Optional[PairedRandom] = None


def prepare_inputs(batch: Mapping[str, Any], model: Any, settings: TrainSettings,
                   device: torch.device, unroll_steps: int) -> UnrollInputs:
    """Move ``batch`` to ``device`` and derive the t0 state, entries and CAV drive."""
    states = batch["states"].to(device).float()
    present = batch["present"].to(device).bool()
    is_cav = batch.get("is_cav", torch.zeros_like(present)).to(device).bool()
    teacher = batch.get("cav_teacher")
    if teacher is not None:
        teacher = teacher.to(device).float()
    hist = batch.get("hist")
    hist_mask = batch.get("hist_mask")
    if hist is not None:
        hist, hist_mask = hist.to(device), hist_mask.to(device)
    observed_hist = batch.get("driver_hist", hist)
    observed_hist_mask = batch.get("driver_hist_mask", hist_mask)
    if observed_hist is not None:
        observed_hist = observed_hist.to(device)
        observed_hist_mask = observed_hist_mask.to(device)
    if model.long_history and "driver_hist" not in batch:
        raise ValueError("long-history trainer requires pre-t0 driver_hist")
    loss_valid = batch.get("loss_valid")
    loss_valid = present if loss_valid is None else loss_valid.to(device).bool()
    coupled = bool(getattr(settings, "paired_diff_couple_randomness", False)) \
        and "pair_ref" in batch
    paired_random = (PairedRandom.from_batch(batch, seed=int(settings.seed), device=device,
                                            dtype=states.dtype) if coupled else None)
    entries = make_entry_process(settings, batch, present, model, device, unroll_steps,
                                 paired_random=paired_random)
    present_seq = present
    if entries is not None:
        pad = int(entries.train_pad)
        if paired_random is not None:
            paired_random = PairedRandom.from_batch(
                batch, seed=int(settings.seed), device=device, dtype=states.dtype, pad=pad)
        (states, present, present_seq, is_cav, teacher, hist, hist_mask,
         loss_valid) = extend_for_entries(states, present, is_cav, teacher, hist, hist_mask,
                                          loss_valid, pad)
        if model.long_history:
            observed_hist = F.pad(observed_hist, (0, 0, 0, 0, 0, pad))
            observed_hist_mask = F.pad(observed_hist_mask, (0, 0, 0, pad))
    n_steps = min(int(unroll_steps), states.shape[1] - 1)
    if n_steps < 1:
        raise ValueError("batch does not contain requested unroll")
    initial = state_from_kin(states[:, 0], present[:, 0])
    initial["is_cav"] = is_cav[:, 0]
    _perturb_t0(initial, states, present, is_cav, model, settings, paired_random)
    controller_id = batch.get("controller_id")
    if controller_id is not None:
        controller_id = controller_id.to(device)
    timeline = _device_timeline(batch.get("action_timeline"), device, entries)
    _refuse_unknown_commands(timeline, settings)
    cav_controller = None
    if settings.cav_drive == "closed_loop":
        if settings.black_box:
            raise ValueError("closed_loop CAV driving requires structured WM-S")
        cav_controller = TimelineControllers(model, timeline)
        teacher = None
    if not model.long_history:
        observed_hist, observed_hist_mask = hist, hist_mask
    entrant_seq = batch.get("entrant_seq")
    if entrant_seq is not None:
        entrant_seq = entrant_seq.to(device).bool()
    return UnrollInputs(states, present, present_seq, is_cav, teacher, hist, hist_mask,
                        observed_hist, observed_hist_mask, loss_valid, entries, controller_id,
                        timeline, cav_controller, entrant_seq, n_steps, initial, paired_random)


def _perturb_t0(initial: dict[str, Tensor], states: Tensor, present: Tensor, is_cav: Tensor,
                model: Any, settings: TrainSettings,
                paired_random: Optional[PairedRandom] = None) -> None:
    """WM-S2 t0 state noise (driver_params mode): perturb only t0 HDV slots, in place."""
    if not (model.rollout_mode == "driver_params"
            and (settings.state_noise_std_m or settings.state_noise_std_mps)):
        return
    hdv0 = present[:, 0] & ~is_cav[:, 0]
    noise_s = (torch.randn_like(initial["s_m"]) if paired_random is None
               else paired_random.normal("t0_s", [0])[:, 0])
    initial["s_m"] = (initial["s_m"] + noise_s
                       * settings.state_noise_std_m * hdv0)
    noise_v = (torch.randn_like(initial["v_mps"]) if paired_random is None
               else paired_random.normal("t0_v", [0])[:, 0])
    initial["v_mps"] = (initial["v_mps"] + noise_v
                        * settings.state_noise_std_mps * hdv0).clamp_min(0)
    kin = state_from_kin(states[:, 0], present[:, 0])["veh_kin"].clone()
    kin[..., 0] = initial["s_m"]
    kin[..., 4] = initial["v_mps"]
    initial["veh_kin"] = kin


def _refuse_unknown_commands(timeline: Optional[Mapping[str, Any]],
                             settings: TrainSettings) -> None:
    """DATAHYG: a ``direct`` step whose command is unknown may not drive a CAV or a controller
    embedding (closed-loop CAVs or the black box); teacher-driven structured WM-S is fine."""
    if timeline is None or "command_known" not in timeline:
        return
    if settings.cav_drive != "closed_loop" and not settings.black_box:
        return
    direct = timeline["controller"] == CONTROLLERS.index("direct")
    if bool((direct & ~timeline["command_known"].bool()).any()):
        raise ValueError("direct timeline steps with unknown issued commands (controller "
                         "hygiene relabelled dspeed windows) cannot drive closed-loop CAVs or "
                         "the black-box controller embedding")


def _device_timeline(timeline: Optional[Mapping[str, Any]], device: torch.device,
                     entries: Optional[Any]) -> Optional[dict]:
    """Action timeline on ``device``; entrant columns default to the documented native plan."""
    if timeline is None:
        return None
    out = {key: (value.to(device) if torch.is_tensor(value) else value)
           for key, value in timeline.items()}
    if entries is not None:
        for key in ("controller", "acceleration_mps2", "target_speed_mps", "command_known"):
            if key not in out:
                continue
            value = out[key]
            fill = CONTROLLERS.index("native") if key == "controller" else 0
            padding = value.new_full((*value.shape[:2], int(entries.train_pad)), fill)
            out[key] = torch.cat((value, padding), dim=2)
    return out


def build_targets(inputs: UnrollInputs, batch: Mapping[str, Any], settings: TrainSettings,
                  dt_s: float, device: torch.device) -> UnrollTargets:
    """Recorded per-unroll targets, masks and the global valid-count normaliser."""
    n = inputs.n_steps
    states = inputs.states
    weights = step_weights(n, states.dtype, device, settings.step_weight_decay).view(1, n, 1)
    # Structured WM-S supervises HDVs only (CAVs are teacher-forced); the black box
    # supervises every learned mover.  ``loss_valid`` keeps a slot through a junction
    # transit but skips the junction step itself (undefined lane target).
    hdv_mask = (inputs.loss_valid[:, 1:n + 1] if settings.black_box
                else inputs.loss_valid[:, 1:n + 1] & ~inputs.is_cav[:, 1:n + 1])
    entrant = entrant_steps(batch.get("entrant"), n, states.shape[1] - 1, device)
    if entrant is not None:
        hdv_mask = hdv_mask & ~entrant
    aux_mask = hdv_mask & ~inputs.is_cav[:, 1:n + 1] & inputs.present[:, :n]
    denom = (weights * hdv_mask.to(weights.dtype)).sum().clamp_min(1.0)
    road = None
    if settings.aggregate_weight > 0.0:
        # Aggregate loss (C4): recorded on-road presence -- the full road (incl. arrivals)
        # when the entry process runs and the windows carry it, else the t0 cohort.
        from .losses.aggregate import road_targets

        road = road_targets(batch, states, inputs.present, int(batch["states"].shape[2]),
                            entries_on=inputs.entries is not None, device=device)
    pair = None
    if getattr(settings, "paired_diff_weight", 0.0) > 0.0 and "pair_ref" in batch:
        # PAIRLOSS: paired batch (pair groups of GENRAND branches) -> t0-cohort pair targets.
        from .losses.paired_diff import pair_targets

        pair = pair_targets(batch, states, inputs.present, int(batch["states"].shape[2]),
                            device=device)
    targets = UnrollTargets(
        states=states, n_steps=n, weights=weights, hdv_mask=hdv_mask, aux_mask=aux_mask,
        aux_count=int(aux_mask.sum()), denom=denom,
        lane_target=states[:, 1:n + 1, :, 1:4].argmax(-1), loss_valid=inputs.loss_valid,
        is_cav=inputs.is_cav, dt_s=float(dt_s), position_delta_m=settings.position_huber_delta_m,
        speed_delta_mps=settings.speed_huber_delta_mps,
        history_lane_head=settings.lane_head == "history", road=road)
    if pair is not None:
        targets.pair = pair
    return targets
