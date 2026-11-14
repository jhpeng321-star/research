"""v0.3 net trainer: M concurrent members, per-tau targets, OOB selection.

BRIEF_B3_V03.md items implemented here (B3b; the GBM/baseline/persistence
sibling items are B3a's):

  A1  per-tau target normalisation (y / TRAIN median per tau & policy,
      data.V03TensorData) before symlog/twohot;
  A2  twohot grid over the realistic normalised range only
      (model.SYM_MIN_V03..SYM_MAX_V03);
  A3  tau conditioning in every head layer (FiLM) + learned per-tau
      output scale/bias (model.FiLMHead);
  A4  inputs standardised from TRAIN stats saved with the weights
      (data.fit_v03_stats -> stats.json) + the diag summary-feature token;
  A5  median point forecast (model.PmfGrid.quantile_torch, q=0.5);
  B6  selection on out-of-bag worlds: early stop on OOB mean CRPS
      averaged over ALL tau and both policies, eval every --eval-every
      (500) steps, patience 3, max 8000 steps; CAL is NEVER used for
      selection (cal_net guard in the tests; CAL only enters the final
      report);
  B7  --ablation: short base / wd / dropout 0.1 / d_model 128 runs scored
      on OOB, choice recorded and applied when --variant auto;
  B8  MC weight rises with tau; on-grid n-step TD targets (chained d30 +
      the tau=30 label) with one-step bootstrap fallback at the exact
      tau-30 through the continuous tau conditioning; explicit
      consistency penalty V(s,tau) ~= D30 + V(s+30, tau-30);
  B9  shared encoder trained on damping-fleet + switching-mix rows
      together (switching-mix metrics are scored only on switching-mix
      rows at evaluation);
  B10 M members trained CONCURRENTLY on one device via
      torch.func.stack_module_state + vmap (per-member batches, dropout
      randomness per member) and ONE elementwise AdamW over the stacked
      parameters (= M independent AdamWs, with a per-member active mask
      that freezes early-stopped members bitwise); resume is
      bitwise-exact per member (stacked params/moments + RNG state);
  B11 per-member budget re-planned at --replan-after steps from measured
      ms/step INCLUDING eval/checkpoint/logging overhead; cosine LR
      follows the re-planned total; device-resident dataset; bf16
      autocast on Ampere+, fp16 + manual loss scaling on T4;
  C13 paired Delta head trained on the t0 CRN restore pairs, predicting
      the damping-fleet minus switching-mix delay contrast directly
      (v0.3.2 fix #13: the campaign '__native' arms are random per-CAV
      switching schedules -- NOT a fleet-native policy, so the head's
      contrast is damping_fleet - switching_mix; REP-P replicate-mean
      hook: rebuild the pair frames with
      data.build_delta_pairs(label_fn=...)).

ZERO simulation; CPU runs are smoke only (owner rule: no CPU training).
"""
from __future__ import annotations

import base64
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, stack_module_state, vmap

from .data import V0Dataset, V03TensorData, load_manifest
from .model import (PmfGrid, V03ValueConfig, V03ValueNet, count_parameters,
                    nll_twohot, twohot_encode_v03)
from .train import CheckpointDataMismatch, resolve_device, seed_everything

FORMAT_V03 = "wd01_partB_v03/1"
CKPT_DIR = "ckpt"
TAU_MAX_S = 570.0
MIN_BATCH = 64
MIN_TD_BATCH = 32

_OOM_ERRORS = tuple(dict.fromkeys(
    e for e in (getattr(torch, "OutOfMemoryError", None),
                getattr(torch.cuda, "OutOfMemoryError", None))
    if e is not None))

# B7 ablation variants: overrides on top of the base config
VARIANTS: dict[str, dict] = {
    "base": {},
    "wd": {"weight_decay": 0.1},
    "dropout": {"dropout": 0.1},
    "d_model128": {"d_model": 128},
}

_PART_NAMES = ("mc_delay", "mc_exp", "mc_brake", "coll", "td", "cons",
               "delta")


@dataclass
class TrainNetConfig:
    data_root: str
    out_dir: str
    members: int = 5
    seed: int = 20260929
    batch_size: int = 512
    td_batch: int = 256
    pair_batch: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    dropout: float = 0.0
    d_model: int = 192
    grad_clip: float = 1.0
    eval_every_steps: int = 500           # B6
    patience: int = 3                     # B6
    max_steps_cap: int = 8000             # B6
    time_budget_min: float = 75.0
    ckpt_every_min: float = 10.0
    max_steps: int | None = None
    smoke: bool = False
    device: str = "auto"
    dtype: str = "auto"                   # auto | bf16 | fp16 | fp32
    w_cons: float = 0.25                  # B8 consistency penalty weight
    w_delta: float = 1.0                  # C13 delta head weight
    w_coll: float = 1.0
    mc_weight_min: float = 0.5            # B8: mc(tau) = min+(1-min)*tau/570
    tau_ema: float = 0.995
    replan_after_steps: int = 500         # B11
    verify: bool = True
    chunks: tuple[str, ...] | None = None
    exclude_superseded: bool = True
    resume: bool = True
    features: str = "oracle"              # "oracle" | "deployable" (v0.3.1)
    variant: str = "auto"                 # B7: base|wd|dropout|d_model128|auto
    ablation: bool = False                # B7
    ablation_steps: int = 1500
    eval_row_cap: int | None = None       # smoke: cap OOB rows per member
    probe_steps: int = 3
    deterministic: bool = True
    extras: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["chunks"] = list(self.chunks) if self.chunks else None
        return d


def mc_weight(tau: torch.Tensor, mc_min: float) -> torch.Tensor:
    """B8: the MC mixing weight rises with tau (1.0 at tau = 570 s)."""
    return mc_min + (1.0 - mc_min) * (tau / TAU_MAX_S).clamp(0.0, 1.0)


def resolve_dtype(preference: str, device: torch.device) -> str:
    """B11: bf16 autocast on Ampere+ (cap >= 8), fp16 (+ manual loss
    scaling) on older CUDA such as T4, fp32 on CPU."""
    if device.type != "cuda" or preference == "fp32":
        return "fp32"
    if preference in ("bf16", "fp16"):
        return preference
    major, _ = torch.cuda.get_device_capability(0)
    return "bf16" if major >= 8 else "fp16"


def _autocast_ctx(device: torch.device, dtype: str):
    if device.type != "cuda" or dtype == "fp32":
        import contextlib
        return contextlib.nullcontext()
    return torch.amp.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if dtype == "bf16" else torch.float16)


def _disable_mha_fastpath() -> None:
    """The nn.TransformerEncoder eval fast path is not vmap-safe; the
    classic implementation is (training uses it anyway)."""
    try:
        torch.backends.mha.set_fastpath_enabled(False)   # torch >= 2.2
    except AttributeError:
        pass


class LossScaler:
    """Manual fp16 loss-scale management (T4).  Explicit because the
    stacked-parameter optimizer is not a torch.optim.Optimizer, so
    torch.amp.GradScaler cannot wrap it.  bf16/fp32 need none."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.scale = 2.0 ** 16 if enabled else 1.0
        self._good_steps = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale if self.enabled else loss

    @torch.no_grad()
    def unscale_and_check(self, grads: dict) -> bool:
        """Divide grads by the scale; returns False (and halves the
        scale) when any grad overflowed: the step must be skipped."""
        if not self.enabled:
            return True
        found = any(not torch.isfinite(g).all() for g in grads.values())
        if found:
            self.scale *= 0.5
            self._good_steps = 0
            return False
        for g in grads.values():
            g.div_(self.scale)
        self._good_steps += 1
        if self._good_steps >= 2000:
            self.scale *= 2.0
            self._good_steps = 0
        return True


class StackedAdamW:
    """Elementwise AdamW over stacked [M, ...] parameters with a
    per-member active mask.

    Adam(W) is elementwise, so one update over the stacked parameters is
    exactly M independent AdamWs; the mask freezes early-stopped members
    BITWISE (grads AND the decoupled weight decay are masked), which a
    plain torch.optim.AdamW over stacked params could not do.
    """

    def __init__(self, params: dict, members: int, lr: float,
                 betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        self.params = params
        self.members = members
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {k: torch.zeros_like(v) for k, v in params.items()}
        self.v = {k: torch.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params.values():
            p.grad = None

    @torch.no_grad()
    def step(self, grads: dict, active: torch.Tensor, lr: float) -> None:
        self.t += 1
        b1, b2 = self.betas
        eps, wd = self.eps, self.weight_decay
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for k, p in self.params.items():
            a = active.to(p.dtype).view(-1, *([1] * (p.dim() - 1)))
            g = grads[k] * a
            m, v = self.m[k], self.v[k]
            m.mul_(b1).add_(g, alpha=1.0 - b1)
            v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
            mhat = m / bc1
            vhat = v / bc2
            upd = mhat / (vhat.sqrt() + eps) + wd * p
            p.add_((upd * (-lr)) * a)


def build_member_nets(cfg_model: V03ValueConfig, members: int, seed: int,
                      device: torch.device) -> list[V03ValueNet]:
    """Deterministic per-member init (member m seeded seed + 1000*m)."""
    nets = []
    for m in range(members):
        torch.manual_seed(seed + 1000 * m)
        nets.append(V03ValueNet(cfg_model).to(device))
    return nets


def stack_members(nets: list[V03ValueNet]):
    """B10: stacked (params, buffers) with a leading member dim."""
    params, buffers = stack_module_state(nets)
    params = {k: v.detach().clone().requires_grad_(True)
              for k, v in params.items()}
    buffers = {k: v.detach().clone() for k, v in buffers.items()}
    return params, buffers


def member_state(params: dict, buffers: dict, member: int) -> dict:
    """Unstack one member's full state_dict (params + buffers)."""
    return {k: v[member].detach().cpu().clone()
            for k, v in {**buffers, **params}.items()}


def _cosine_lr(lr: float, step: int, total: int) -> float:
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(step / max(total, 1),
                                                    1.0)))


def _torch_rng_b64() -> str:
    return base64.b64encode(torch.get_rng_state().numpy().tobytes()) \
        .decode("ascii")


def _restore_torch_rng(b64: str) -> None:
    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8).copy()
    torch.set_rng_state(torch.from_numpy(buf))


def _atomic_json(path: Path, obj: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _centers_t(cfg_model: V03ValueConfig, device) -> torch.Tensor:
    from .model import inv_symlog
    return inv_symlog(torch.linspace(cfg_model.sym_min, cfg_model.sym_max,
                                     cfg_model.n_bins, device=device))


# ----------------------------------------------------------- member loss --

def _member_losses(pb: dict, tpb: dict, mc: dict, td: dict, pair: dict,
                   template: V03ValueNet, tcfg: TrainNetConfig,
                   centers: torch.Tensor, live_pair: bool = True,
                   cons_w: float = 1.0):
    """Loss for ONE member (vmap maps the leading member dim).

    pb/tpb: params+buffers of the student / EMA target; mc: MC batch +
    targets; td: TD batches (t, t+30, d30, on-grid chain info, exact
    normalisers); pair: delta-pair batch (model batch already stacked).
    All value targets are per-tau-normalised (A1).  Returns (loss,
    parts[7]) with parts ordered as _PART_NAMES.
    """
    dev = centers.device
    out = functional_call(template, pb, (mc,))
    w = mc["w"]
    wm = w * mc_weight(mc["tau"], tcfg.mc_weight_min)
    wm = wm / wm.mean().clamp(min=1e-8)
    l_delay = nll_twohot(out["v_delay"], twohot_encode_v03(mc["y_delay"]),
                         wm)
    we = w / w.mean().clamp(min=1e-8)
    l_exp = nll_twohot(out["v_exp"], twohot_encode_v03(mc["y_exp"]), we)
    l_brake = nll_twohot(out["v_brake"], twohot_encode_v03(mc["y_brake"]),
                         we)
    l_coll = torch.nn.functional.binary_cross_entropy_with_logits(
        out["p_coll"].float(), mc["coll"], weight=we)

    # ---- TD (B8): on-grid n-step chain where the data allow, else a
    # one-step bootstrap at the EXACT tau-30 through the EMA target net
    # (continuous tau conditioning + exact piecewise-linear normaliser;
    # no label interpolation).  Plus the explicit consistency penalty
    # V(s, tau) ~= D30 + V(s+30, tau-30) in normalised units.
    if "t" in td:
        out_td = functional_call(template, pb, (td["t"],))
        with torch.no_grad():
            out_n = functional_call(template, tpb, (td["n"],))
            v_next = torch.softmax(out_n["v_delay"].float(), -1) @ centers
        one_step_norm = (td["d30"] + v_next * td["med_next"]) \
            / td["med_tau"]
        chain_norm = (td["chain_sum"] + td["term_y30"]) / td["med_tau"]
        td_target = torch.clamp(torch.where(td["chain_ok"], chain_norm,
                                            one_step_norm), -50.0, 50.0)
        wtd = td["w"] * (1.0 - mc_weight(td["tau"], tcfg.mc_weight_min))
        wtd = wtd / wtd.mean().clamp(min=1e-8)
        l_td = nll_twohot(out_td["v_delay"],
                          twohot_encode_v03(td_target), wtd)
        v_student = torch.softmax(out_td["v_delay"].float(), -1) @ centers
        cons = torch.nn.functional.huber_loss(v_student, one_step_norm,
                                              delta=1.0)
    else:                       # timing probe: no TD tensors provided
        l_td = torch.zeros((), device=dev)
        cons = torch.zeros((), device=dev)

    # ---- Delta head (C13): damping_fleet - switching_mix at the same
    # (world, t0, tau), normalised by the TRAIN per-tau delta scale
    # (v0.3.2 fix #13: the '__native' arms are random per-CAV switching
    # schedules, NOT a fleet-native policy -- no fleet-native contrast
    # claim anywhere); masked mean so members without pairs contribute
    # zero (vmap-safe, no indexing).
    if live_pair and tcfg.w_delta > 0 and "batch" in pair:
        out_p = functional_call(template, pb, (pair["batch"],))
        d_err = torch.nn.functional.huber_loss(out_p["delta"].float(),
                                               pair["delta_norm"],
                                               delta=1.0,
                                               reduction="none")
        msk = pair["mask"]
        l_delta = (d_err * msk).sum() / msk.sum().clamp(min=1.0)
    else:
        l_delta = torch.zeros((), device=dev)

    loss = l_delay + l_exp + l_brake + tcfg.w_coll * l_coll + l_td \
        + cons_w * cons + tcfg.w_delta * l_delta
    parts = torch.stack([l_delay.detach(), l_exp.detach(), l_brake.detach(),
                         l_coll.detach(), l_td.detach(), cons.detach(),
                         l_delta.detach()])
    return loss, parts


def _cons_weight(tcfg: TrainNetConfig, step: int, planned: int) -> float:
    """Ramp the consistency weight over the first quarter of training:
    at init the target-net values are still noisy, and an unreduced
    huber on them would swamp the MC anchors (observed: initial cons
    loss ~2.4e3 with a uniform-init head)."""
    ramp = min(1.0, step / max(1e-6, 0.25 * max(planned, 1)))
    return tcfg.w_cons * ramp


def _stack_dicts(dicts: list[dict]) -> dict:
    """[M] batch dicts -> leading-member-dim dict of stacked tensors
    (recursing into nested dicts such as the TD (t, t+30) pair)."""
    out = {}
    for k, v0 in dicts[0].items():
        vs = [d[k] for d in dicts]
        if isinstance(v0, dict):
            out[k] = _stack_dicts(vs)
        elif torch.is_tensor(v0):
            out[k] = torch.stack(vs, dim=0)
        else:
            out[k] = vs
    return out


# ------------------------------------------------------------------ probe --

def timing_probe(cfg: TrainNetConfig, device: torch.device, dtype: str,
                 dims: dict | None = None, log=print) -> dict:
    """Lockstep ms/step on a throwaway stacked ensemble (B10/B11).  One
    lockstep step advances ALL M members; per-member cost = lockstep / M.
    The probe is discarded and cannot perturb the trained members."""
    cfg_model = V03ValueConfig(d_model=cfg.d_model, dropout=cfg.dropout,
                               **(dims or {}))
    nets = build_member_nets(cfg_model, cfg.members, cfg.seed, device)
    n_params = count_parameters(nets[0])
    params, buffers = stack_members(nets)
    template = nets[0]
    opt = StackedAdamW(params, cfg.members, cfg.lr,
                       weight_decay=cfg.weight_decay)
    mc = _probe_batch(cfg, cfg_model, device)
    centers = _centers_t(cfg_model, device)
    active = torch.ones(cfg.members, device=device)

    def loss_fn(pb, tb, mc_i):
        return _member_losses(pb, tb, mc_i, {}, {}, template, cfg, centers,
                              live_pair=False)

    times = []
    try:
        for i in range(max(3, cfg.probe_steps)):   # >= 1 measured step
            t0 = time.perf_counter()
            with _autocast_ctx(device, dtype):
                loss, _ = vmap(loss_fn, in_dims=(0, 0, 0),
                               randomness="different")(
                    {**params, **buffers}, {**params, **buffers}, mc)
                opt.zero_grad()
                loss.sum().backward()
                grads = {k: (p.grad if p.grad is not None
                             else torch.zeros_like(p))
                         for k, p in params.items()}
                opt.step(grads, active, _cosine_lr(cfg.lr, 1, 100))
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= 2:
                times.append(time.perf_counter() - t0)
    finally:
        del nets, params, buffers, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ms = float(np.mean(times)) * 1e3 if times else float("nan")
    probe = dict(device=str(device), dtype=dtype, members=cfg.members,
                 batch_size=cfg.batch_size, td_batch=cfg.td_batch,
                 k_steps=max(3, cfg.probe_steps), warmup=2,
                 ms_per_lockstep_step=round(ms, 1),
                 ms_per_member_step=round(ms / max(1, cfg.members), 1),
                 n_params_per_member=n_params)
    log(f"probe: {n_params / 1e6:.2f} M params/member; {ms:.0f} ms per "
        f"lockstep step ({ms / max(1, cfg.members):.0f} ms per member-"
        f"step; M={cfg.members}, batch={cfg.batch_size}+{cfg.td_batch}, "
        f"dtype={dtype}, device={device})")
    return probe


def _probe_batch(cfg: TrainNetConfig, cfg_model: V03ValueConfig,
                 device: torch.device) -> dict:
    """A fake stacked MC batch with the real shapes (probe only)."""
    b = min(cfg.batch_size, 128)

    def t(*shape):
        return torch.zeros(cfg.members, *shape, device=device)
    return dict(
        lane_bins=t(b, 3, cfg_model.lmax_bins, 3),
        lane_mask=torch.ones(cfg.members, b, 3, cfg_model.lmax_bins,
                             device=device),
        lane_geom=torch.ones(cfg.members, 3, cfg_model.lmax_bins,
                             device=device),
        cav_tokens=t(b, cfg_model.max_cavs, cfg_model.cav_raw),
        cav_mask=torch.zeros(cfg.members, b, cfg_model.max_cavs,
                             device=device),
        wall_tokens=t(b, cfg_model.max_walls, cfg_model.wall_raw),
        wall_mask=torch.zeros(cfg.members, b, cfg_model.max_walls,
                              device=device),
        demand=t(b, cfg_model.demand_raw),
        context=t(b, cfg_model.ctx_raw),
        summary=t(b, cfg_model.summary_dim),
        tau=torch.full((cfg.members, b), 300.0, device=device),
        policy=torch.zeros(cfg.members, b, dtype=torch.long,
                           device=device),
        y_delay=torch.full((cfg.members, b), 1.0, device=device),
        y_exp=torch.full((cfg.members, b), 1.0, device=device),
        y_brake=torch.full((cfg.members, b), 1.0, device=device),
        coll=torch.zeros(cfg.members, b, device=device),
        w=torch.full((cfg.members, b), 1.0 / b, device=device))


# ---------------------------------------------------------------- trainer --

class ConcurrentNetTrainer:
    """B10: all M members advance in lockstep; per-member batches, best
    states, early stopping and bitwise per-member resume."""

    def __init__(self, cfg: TrainNetConfig, data: V03TensorData,
                 device: torch.device, dtype: str, log=print,
                 cfg_model: V03ValueConfig | None = None,
                 weight_decay: float | None = None,
                 out_dir: Path | None = None):
        self.cfg = cfg
        self.data = data
        self.device = device
        self.dtype = dtype
        self.log = log
        self.ds: V0Dataset = data.ds
        self.out = Path(out_dir or cfg.out_dir)
        self.ckpt = self.out / CKPT_DIR
        self.ckpt.mkdir(parents=True, exist_ok=True)
        self.cfg_model = cfg_model or V03ValueConfig(
            d_model=cfg.d_model, dropout=cfg.dropout,
            **data.model_dims())
        self.weight_decay = cfg.weight_decay if weight_decay is None \
            else weight_decay

        self.nets = build_member_nets(self.cfg_model, cfg.members,
                                      cfg.seed, device)
        self.params, self.buffers = stack_members(self.nets)
        self.template = self.nets[0]
        self.target = {k: v.detach().clone()
                       for k, v in {**self.buffers, **self.params}.items()}
        self.opt = StackedAdamW(self.params, cfg.members, cfg.lr,
                                weight_decay=self.weight_decay)
        self.scaler = LossScaler(enabled=(dtype == "fp16"))
        self.centers = _centers_t(self.cfg_model, device)
        self.grid = PmfGrid(self.cfg_model.sym_min, self.cfg_model.sym_max,
                            self.cfg_model.n_bins)
        self.active = torch.ones(cfg.members, device=device)
        self.step = 0
        self.best: list[tuple[float, int] | None] = [None] * cfg.members
        self.best_state: list[dict | None] = [None] * cfg.members
        self.patience_ctr = [0] * cfg.members
        self.completed = False
        self.oom_events: list[dict] = []
        self._build_pools()

    # ------------------------------------------------------------- pools --
    def _build_pools(self):
        """Per-member bootstrap pools (B6) + OOB row sets."""
        cfg = self.cfg
        self.data.set_members(cfg.seed, cfg.members)
        src_split = np.array(self.ds.src_split)
        src_world = np.array([n.rsplit("__", 1)[0].split("/")[-1]
                              for n in self.ds.src_names])
        row_src = self.ds.rows["src_idx"].to_numpy()
        self.pool, self.pool_w, self.td_pool, self.td_pool_w, \
            self.pair_pool = [], [], [], [], []
        for m in range(cfg.members):
            sampled, _ = self.data.bootstrap_worlds(cfg.seed, m)
            world_list = sorted({w.rsplit("__", 1)[0].split("/")[-1]
                                 for w in sampled})
            keep_src = (src_split == "TRAIN") & np.isin(src_world,
                                                        world_list)
            idx = np.flatnonzero(keep_src[row_src])
            w = self.ds.rows["row_weight"].to_numpy()[idx]
            self.pool.append(idx)
            self.pool_w.append((w / w.sum()).astype(np.float64))

            td = self.ds.td
            td_ok = ((td["split"] == "TRAIN") & (td["tau"] >= 60.0)
                     & (td["t_idx"] >= 0)
                     & (td["tnext_idx"] >= 0)).to_numpy()
            td_ok &= keep_src[td["src_idx"].to_numpy()]
            tidx = np.flatnonzero(td_ok)
            tw = self.ds.td_weights()[tidx]
            self.td_pool.append(tidx)
            self.td_pool_w.append((tw / tw.sum()).astype(np.float64))

            pf = self.data.pairs.get("TRAIN")
            if pf is not None and len(pf):
                pw = pf["world"].to_numpy().astype(str)
                self.pair_pool.append(
                    np.flatnonzero(np.isin(pw, world_list)))
            else:
                self.pair_pool.append(np.array([], dtype=np.int64))
        self.log(f"trainer: M={cfg.members} concurrent members "
                 f"({count_parameters(self.nets[0]) / 1e6:.2f} M params "
                 f"each, dtype={self.dtype}); OOB rows/member: "
                 f"{[len(x) for x in self.data.oob_rows]}")

    # ------------------------------------------------------------ batches --
    def _mc_idx(self, member: int, step: int) -> np.ndarray:
        rng = np.random.default_rng([self.cfg.seed, member, step, 11])
        bs = min(self.cfg.batch_size, len(self.pool[member]))
        return rng.choice(self.pool[member], size=bs,
                          replace=len(self.pool[member]) >= bs,
                          p=self.pool_w[member])

    def _td_idx(self, member: int, step: int) -> np.ndarray:
        rng = np.random.default_rng([self.cfg.seed, member, step, 13])
        bs = min(self.cfg.td_batch, len(self.td_pool[member]))
        return rng.choice(self.td_pool[member], size=bs,
                          replace=len(self.td_pool[member]) >= bs,
                          p=self.td_pool_w[member])

    def _pair_idx(self, member: int, step: int) -> np.ndarray:
        pool = self.pair_pool[member]
        if len(pool) == 0:
            return np.array([], dtype=np.int64)
        rng = np.random.default_rng([self.cfg.seed, member, step, 17])
        bs = min(self.cfg.pair_batch, len(pool))
        return rng.choice(pool, size=bs, replace=len(pool) >= bs)

    def _mc_batch(self, member: int, step: int) -> dict:
        idx = self._mc_idx(member, step)
        b = self.data.gather_rows(idx)
        yd, ye, yb, coll, w = self.data.row_targets(idx)
        b.update(y_delay=yd, y_exp=ye, y_brake=yb, coll=coll, w=w)
        return b

    def _td_batch(self, member: int, step: int) -> dict:
        idx = self._td_idx(member, step)
        itt = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        pre = self.data.t_td["pre"][itt]
        pol = self.data.t_td["policy_id"][itt]
        t = self.data.gather_global(
            self.data.t_td["t_idx"][itt], self.data.t_td["tau"][itt],
            pol, pre)
        n = self.data.gather_global(
            self.data.t_td["tnext_idx"][itt],
            self.data.t_td["tau"][itt] - 30.0, pol, pre)
        w = torch.from_numpy(
            self.ds.td_weights()[idx].astype(np.float32)).to(self.device)
        return dict(
            t=t, n=n, d30=self.data.t_td["d30"][itt],
            tau=self.data.t_td["tau"][itt],
            chain_ok=self.data.t_td_chain["ok"][itt],
            chain_sum=self.data.t_td_chain["sum"][itt],
            term_y30=self.data.t_td_chain["term_y30"][itt],
            med_tau=self.data.t_td_med_tau[itt],
            med_next=self.data.t_td_med_next[itt], w=w)

    def _pair_model_batch(self, g: torch.Tensor, tau: torch.Tensor,
                          pre: torch.Tensor):
        m = g.shape[0]
        return self.data.gather_global(
            g, tau, torch.zeros(m, dtype=torch.long, device=self.device),
            pre)

    def _pair_batch(self, member: int, step: int) -> dict:
        idx = self._pair_idx(member, step)
        if len(idx) == 0:
            return None
        g, tau, dnorm, _dreal, pre = self.data.pair_tensors("TRAIN")
        return dict(g=g[idx], tau=tau[idx], delta_norm=dnorm[idx],
                    pre=pre[idx])

    # ---------------------------------------------------------- one step --
    def _train_one_step(self, planned: int):
        cfg, dev = self.cfg, self.device
        s = self.step
        mc = _stack_dicts([self._mc_batch(m, s) for m in range(cfg.members)])
        td = _stack_dicts([self._td_batch(m, s) for m in range(cfg.members)])
        pair = self._stack_pairs([self._pair_batch(m, s)
                                  for m in range(cfg.members)])
        pb = {**self.params, **self.buffers}

        def loss_fn(pb_i, tb_i, mc_i, td_i, pair_i):
            return _member_losses(pb_i, tb_i, mc_i, td_i, pair_i,
                                  self.template, cfg, self.centers,
                                  cons_w=_cons_weight(cfg, s, planned))
        with _autocast_ctx(dev, self.dtype):
            loss, parts = vmap(loss_fn, in_dims=(0, 0, 0, 0, 0),
                               randomness="different")(pb, self.target,
                                                       mc, td, pair)
        total = self.scaler.scale_loss(loss.sum())
        self.opt.zero_grad()
        total.backward()
        grads = {k: (p.grad if p.grad is not None else torch.zeros_like(p))
                 for k, p in self.params.items()}
        ok = self.scaler.unscale_and_check(grads)
        if ok:
            self._clip_per_member(grads)
            lr = _cosine_lr(cfg.lr, self.step + 1, planned)
            self.opt.step(grads, self.active, lr)
        else:
            self.oom_events.append(dict(step=int(self.step),
                                        kind="fp16_overflow"))
        self.opt.zero_grad()
        with torch.no_grad():               # EMA target net
            for k, p in self.params.items():
                self.target[k].mul_(cfg.tau_ema).add_(
                    p, alpha=1.0 - cfg.tau_ema)
        return float(total.detach()), parts.detach()

    def _stack_pairs(self, pairs: list[dict | None]) -> dict:
        """Per-member pair batches -> stacked model batch + targets.

        Members with no pairs (or fewer rows than others) are padded to a
        common length with masked-out rows (vmap needs equal shapes; the
        delta loss is a masked mean, so padding contributes nothing)."""
        dev = self.device
        n_max = max([len(p["g"]) if p is not None else 0 for p in pairs]
                    + [1])

        def pad(t: torch.Tensor, pad_value=0.0):
            n = t.shape[0]
            if n == n_max:
                return t
            tail = torch.full((n_max - n,), pad_value, dtype=t.dtype,
                              device=dev)
            return torch.cat([t, tail])
        batches, dns, mask = [], [], []
        for p in pairs:
            if p is None:
                g0 = torch.zeros(n_max, dtype=torch.long, device=dev)
                b = self._pair_model_batch(g0, torch.zeros(n_max, device=dev),
                                           torch.ones(n_max, device=dev))
                batches.append(b)
                dns.append(torch.zeros(n_max, device=dev))
                mask.append(torch.zeros(n_max, device=dev))
                continue
            g = pad(p["g"].long(), 0)
            tau = pad(p["tau"], 30.0)
            pre = pad(p["pre"], 1.0)
            batches.append(self._pair_model_batch(g, tau, pre))
            dns.append(pad(p["delta_norm"], 0.0))
            mask.append(pad(torch.ones(len(p["g"]), device=dev), 0.0))
        return dict(batch=_stack_dicts(batches), delta_norm=torch.stack(dns),
                    mask=torch.stack(mask))

    def _clip_per_member(self, grads: dict) -> None:
        """Global-norm grad clip PER MEMBER (stacked)."""
        sq = None
        for g in grads.values():
            term = g.float().pow(2).sum(dim=tuple(range(1, g.dim())))
            sq = term if sq is None else sq + term
        norm = sq.sqrt().clamp(min=1e-12)
        scale = (self.cfg.grad_clip / norm).clamp(max=1.0)
        scale = scale * self.active.to(scale.dtype)
        for k, g in grads.items():
            g.mul_(scale.view(-1, *([1] * (g.dim() - 1))))

    # ---------------------------------------------------------- eval OOB --
    @torch.no_grad()
    def eval_oob(self, cap: int | None = None) -> list[float | None]:
        """B6: per-member mean CRPS (real units, via the per-tau TRAIN
        medians) over its OOB rows -- ALL tau, both policies.  Reads ONLY
        TRAIN OOB rows; CAL is never touched here (the cal_net guard
        test pins this)."""
        out = [None] * self.cfg.members
        self.template.eval()          # no dropout / no RNG in selection
        for m in range(self.cfg.members):
            if not bool(self.active[m]):
                continue
            idx = self.data.oob_rows[m]
            if len(idx) == 0:
                out[m] = float("nan")
                continue
            if cap is not None and len(idx) > cap:
                rng = np.random.default_rng([self.cfg.seed, m, 23])
                idx = np.sort(rng.choice(idx, size=cap, replace=False))
            pb = {k: v[m] for k, v in {**self.buffers,
                                       **self.params}.items()}
            probs = []
            bs = 512
            for i in range(0, len(idx), bs):
                o = functional_call(self.template, pb,
                                    (self.data.gather_rows(idx[i:i + bs]),))
                probs.append(torch.softmax(o["v_delay"].float(), -1)
                             .to(torch.float64).cpu().numpy())
            probs = np.concatenate(probs, axis=0)
            y_norm = self.data.y_norm["delay"][torch.as_tensor(
                idx, device=self.device)].to(torch.float64).cpu().numpy()
            scale = self.data.med_row["delay"][idx]
            out[m] = float(np.mean(self.grid.crps_real(probs, y_norm,
                                                       scale)))
        self.template.train()
        return out

    # ---------------------------------------------------------- ckpt I/O --
    def _ckpt_paths(self) -> dict:
        return dict(state=self.ckpt / "net_state.pt",
                    meta=self.ckpt / "net_meta.json")

    def _best_json(self) -> list:
        return [None if b is None else [float(b[0]), int(b[1])]
                for b in self.best]

    def save_ckpt(self, planned: int):
        self.ckpt.mkdir(parents=True, exist_ok=True)
        p = self._ckpt_paths()
        tmp = Path(str(p["state"]) + ".tmp")
        torch.save(dict(
            format=FORMAT_V03,
            params={k: v.cpu() for k, v in self.params.items()},
            buffers={k: v.cpu() for k, v in self.buffers.items()},
            target={k: v.cpu() for k, v in self.target.items()},
            opt_m={k: v.cpu() for k, v in self.opt.m.items()},
            opt_v={k: v.cpu() for k, v in self.opt.v.items()},
            opt_t=int(self.opt.t),
            scaler=(None if not self.scaler.enabled else
                    dict(scale=float(self.scaler.scale),
                         good_steps=int(self.scaler._good_steps))),
            active=self.active.cpu(),
            step=int(self.step),
            planned_total=int(planned),
            best=self._best_json(),
            patience_ctr=list(self.patience_ctr),
            completed=bool(self.completed),
            rng_torch=_torch_rng_b64(),
            data_manifest_sha256=self.ds.man.sha256,
            batch_size=int(self.cfg.batch_size),
            td_batch=int(self.cfg.td_batch),
            oom_events=list(self.oom_events),
            cfg_model=self.cfg_model.to_json(),
            weight_decay=float(self.weight_decay),
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ), tmp)
        os.replace(tmp, p["state"])
        _atomic_json(p["meta"], dict(
            format=FORMAT_V03, step=int(self.step),
            planned_total=int(planned),
            active=[float(a) for a in self.active.tolist()],
            best=self._best_json(),
            completed=bool(self.completed),
            data_manifest_sha256=self.ds.man.sha256))

    def try_resume(self) -> int:
        """Restore the stacked state exactly; refuses a foreign dataset."""
        p = self._ckpt_paths()
        if not p["state"].exists():
            return 0
        try:
            ck = torch.load(p["state"], map_location="cpu",
                            weights_only=True)
            if ck.get("data_manifest_sha256") != self.ds.man.sha256:
                raise CheckpointDataMismatch(
                    "net checkpoint was trained against a different data "
                    f"MANIFEST sha ({ck.get('data_manifest_sha256')} != "
                    f"{self.ds.man.sha256})")
            self.params = {k: v.to(self.device).requires_grad_(True)
                           for k, v in ck["params"].items()}
            self.buffers = {k: v.to(self.device)
                            for k, v in ck["buffers"].items()}
            self.target = {k: v.to(self.device)
                           for k, v in ck["target"].items()}
            self.opt = StackedAdamW(self.params, self.cfg.members,
                                    self.cfg.lr,
                                    weight_decay=self.weight_decay)
            self.opt.m = {k: v.to(self.device)
                          for k, v in ck["opt_m"].items()}
            self.opt.v = {k: v.to(self.device)
                          for k, v in ck["opt_v"].items()}
            self.opt.t = int(ck["opt_t"])
            if self.scaler.enabled and ck.get("scaler"):
                self.scaler.scale = float(ck["scaler"]["scale"])
                self.scaler._good_steps = int(
                    ck["scaler"].get("good_steps", 0))
            self.active = ck["active"].to(self.device)
            self.step = int(ck["step"])
            self.patience_ctr = list(ck["patience_ctr"])
            self.completed = bool(ck.get("completed", False))
            self.oom_events = list(ck.get("oom_events", []))
            self.cfg.batch_size = int(ck.get("batch_size",
                                             self.cfg.batch_size))
            self.cfg.td_batch = int(ck.get("td_batch", self.cfg.td_batch))
            self.best = [None if b is None else (float(b[0]), int(b[1]))
                         for b in ck.get("best", [None] * self.cfg.members)]
            self.best_state = []
            for m in range(self.cfg.members):
                bp = self.ckpt / f"m{m}_best.pt"
                self.best_state.append(
                    torch.load(bp, map_location="cpu", weights_only=True)
                    if bp.exists() else None)
            if ck.get("rng_torch"):
                _restore_torch_rng(ck["rng_torch"])
            self.log(f"trainer: resumed at lockstep step {self.step} "
                     f"from {p['state'].name}")
            return self.step
        except CheckpointDataMismatch:
            raise
        except Exception as e:                          # noqa: BLE001
            self.log(f"trainer: checkpoint unusable "
                     f"({type(e).__name__}: {e}); starting fresh")
            return 0

    # ------------------------------------------------------------ train --
    def train(self, planned: int, deadline: float | None,
              stop_after: int | None = None) -> dict:
        """Lockstep training to `planned` steps (or deadline/stop_after/
        all-members-early-stopped)."""
        cfg = self.cfg
        t_start = time.time()
        ckpt_every_s = max(cfg.ckpt_every_min * 60.0, 1.0)
        last_ckpt = time.time()
        log_every = max(5, planned // 10)
        eval_every = max(1, cfg.eval_every_steps)
        loss_hist: list[float] = []
        resumed_at = self.step
        replanned = False

        while self.step < planned and bool(self.active.any()):
            if deadline is not None and time.time() > deadline:
                self.log(f"trainer: time budget reached at lockstep "
                         f"step {self.step}")
                break
            if stop_after is not None and self.step - resumed_at \
                    >= stop_after:
                self.log(f"trainer: simulated interruption at lockstep "
                         f"step {self.step} (stop_after={stop_after})")
                break
            try:
                loss, parts = self._train_one_step(planned)
            except _OOM_ERRORS:
                self._handle_oom()
                continue
            loss_hist.append(loss)
            self.step += 1
            now = time.time()
            if now - last_ckpt >= ckpt_every_s:
                self.save_ckpt(planned)
                last_ckpt = now

            # B11: re-plan from the MEASURED rate (incl. eval/ckpt/logging
            # overhead); the cosine LR total follows the new plan.
            if (not replanned and cfg.max_steps is None and not cfg.smoke
                    and self.step >= cfg.replan_after_steps):
                replanned = True
                elapsed = now - t_start
                ms_eff = elapsed / max(1, self.step - resumed_at) * 1e3
                if deadline is None:
                    planned = min(cfg.max_steps_cap,
                                  self.step + 4 * cfg.replan_after_steps)
                else:
                    remaining = max(0.0, deadline - now)
                    extra = int(remaining * 1e3 / max(ms_eff, 1e-6))
                    planned = min(cfg.max_steps_cap, self.step + extra)
                self.log(f"plan: RE-PLANNED after {self.step} lockstep "
                         f"steps: measured {ms_eff:.0f} ms/step (incl. "
                         f"eval/ckpt/logging overhead) -> planned total "
                         f"{planned} steps/member; cosine LR follows")

            if self.step % log_every == 0 or self.step == planned:
                ms = 1e3 * (time.time() - t_start) / max(1,
                                                         self.step
                                                         - resumed_at)
                p = parts.mean(dim=0)
                self.log(f"trainer step {self.step}/{planned} "
                         f"loss {loss:.4f} (mc_d {p[0]:.3f} "
                         f"td {p[4]:.3f} cons {p[5]:.3f} "
                         f"delta {p[6]:.3f} "
                         f"active {int(self.active.sum())}) "
                         f"{ms:.0f} ms/lockstep-step")

            if self.step % eval_every == 0 or self.step == planned:
                self._eval_and_select(planned)

        self.completed = self.step >= planned or not bool(
            self.active.any())
        self.save_ckpt(planned)
        return dict(steps_done=self.step, planned=int(planned),
                    resumed_from=resumed_at,
                    completed=bool(self.completed),
                    active=[float(a) for a in self.active.tolist()],
                    best=self._best_json(),
                    oom_events=list(self.oom_events),
                    mean_loss_last100=float(np.mean(loss_hist[-100:]))
                    if loss_hist else None,
                    train_seconds=round(time.time() - t_start, 1))

    def _eval_and_select(self, planned: int) -> None:
        """B6: OOB CRPS every eval; per-member best tracking, patience
        and early stop; m{k}_best.pt saved on every new best."""
        cfg = self.cfg
        crps = self.eval_oob(cap=cfg.eval_row_cap)
        msg = []
        for m in range(cfg.members):
            if not bool(self.active[m]):
                msg.append(f"m{m}:frozen")
                continue
            c = crps[m]
            if c is None or not np.isfinite(c):
                msg.append(f"m{m}:nan")
                continue
            if self.best[m] is None or c < self.best[m][0]:
                self.best[m] = (float(c), int(self.step))
                self.best_state[m] = member_state(self.params,
                                                  self.buffers, m)
                bp = self.ckpt / f"m{m}_best.pt"
                tmp = Path(str(bp) + ".tmp")
                torch.save(self.best_state[m], tmp)
                os.replace(tmp, bp)
                self.patience_ctr[m] = 0
                msg.append(f"m{m}:{c:.1f}*")
            else:
                self.patience_ctr[m] += 1
                msg.append(f"m{m}:{c:.1f}({self.patience_ctr[m]})")
                if self.patience_ctr[m] >= cfg.patience:
                    self.active[m] = 0.0
                    b = self.best[m]
                    self.log(f"member {m}: EARLY STOP at step {self.step} "
                             f"(patience {cfg.patience} exhausted; best "
                             f"OOB CRPS {b[0]:.1f} at step {b[1]})")
        self.log("eval oob: " + " ".join(msg) + f" @ step {self.step}")
        self.save_ckpt(planned)

    def _handle_oom(self) -> None:
        new_bs = max(MIN_BATCH, self.cfg.batch_size // 2)
        new_tdb = max(MIN_TD_BATCH, self.cfg.td_batch // 2)
        if new_bs == self.cfg.batch_size and new_tdb == self.cfg.td_batch:
            raise
        self.oom_events.append(dict(
            step=int(self.step), batch_size=self.cfg.batch_size,
            td_batch=self.cfg.td_batch, new_batch_size=new_bs,
            new_td_batch=new_tdb))
        self.log(f"trainer: CUDA OOM at step {self.step}; halving batch "
                 f"{self.cfg.batch_size}+{self.cfg.td_batch} -> "
                 f"{new_bs}+{new_tdb} and retrying")
        self.cfg.batch_size, self.cfg.td_batch = new_bs, new_tdb
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------- write-out --
    def write_members(self, out_dir: Path | None = None) -> list[dict]:
        """member_k.pt from the BEST state on EVERY path (natural finish,
        budget stop or all members early-stopped); falls back to the
        final state when a member never improved."""
        out = Path(out_dir or self.out)
        out.mkdir(parents=True, exist_ok=True)
        sidecars = []
        for m in range(self.cfg.members):
            state = self.best_state[m]
            if state is None:
                state = member_state(self.params, self.buffers, m)
            torch.save(state, out / f"member_{m}.pt")
            b = self.best[m]
            sidecars.append(dict(
                format=FORMAT_V03, member=m,
                member_seed=self.cfg.seed + 1000 * m,
                steps=int(self.step),
                cfg_model=self.cfg_model.to_json(),
                weight_decay=float(self.weight_decay),
                best_oob_crps=(None if b is None else float(b[0])),
                best_step=(None if b is None else int(b[1])),
                data_manifest_sha256=self.ds.man.sha256,
                dtype=self.dtype, device=str(self.device),
                torch=torch.__version__,
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            _atomic_json(out / f"member_{m}.json", sidecars[-1])
        return sidecars


# --------------------------------------------------------------- ablation --

def run_ablation(cfg: TrainNetConfig, data: V03TensorData,
                 device: torch.device, dtype: str,
                 log=print) -> tuple[list, str]:
    """B7: short base / wd / dropout 0.1 / d_model 128 runs scored on OOB
    (mean over members of the members' best OOB CRPS).  Returns (table,
    chosen variant).  Artifacts live under out/ckpt/ablation/<variant>/
    (resume-only, excluded from the deliverable bundle)."""
    short = cfg.max_steps or cfg.ablation_steps
    table = []
    base_dir = Path(cfg.out_dir) / CKPT_DIR / "ablation"
    for name, over in VARIANTS.items():
        vcfg = dataclasses.replace(
            cfg, max_steps=short, ablation=False, resume=False,
            eval_row_cap=cfg.eval_row_cap or 2000,
            weight_decay=over.get("weight_decay", cfg.weight_decay),
            dropout=over.get("dropout", cfg.dropout),
            d_model=over.get("d_model", cfg.d_model))
        cfg_model = V03ValueConfig(d_model=vcfg.d_model,
                                   dropout=vcfg.dropout,
                                   **data.model_dims())
        tr = ConcurrentNetTrainer(vcfg, data, device, dtype, log=log,
                                  cfg_model=cfg_model,
                                  weight_decay=vcfg.weight_decay,
                                  out_dir=base_dir / name)
        tr.train(short, deadline=None)
        vals = [float("nan") if b is None else b[0] for b in tr.best]
        mean_best = float(np.nanmean(vals)) if vals else float("nan")
        table.append(dict(variant=name, steps=int(short),
                          oob_crps_per_member=[round(v, 1) for v in vals],
                          oob_crps_mean=round(mean_best, 1)))
        log(f"ablation: {name:10s} mean OOB CRPS {mean_best:.1f} "
            f"(members {[round(v, 1) for v in vals]})")
        del tr
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ok = [t for t in table if np.isfinite(t["oob_crps_mean"])]
    chosen = min(ok, key=lambda t: t["oob_crps_mean"])["variant"] if ok \
        else "base"
    _atomic_json(Path(cfg.out_dir) / "ablation.json",
                 dict(schema="wd01_partB_v03_ablation/1", table=table,
                      chosen=chosen, steps=int(short),
                      criterion="mean over members of the best OOB CRPS "
                                "(all tau, both policies)"))
    log(f"ablation: chosen variant {chosen!r} (by OOB)")
    return table, chosen


# ------------------------------------------------------------------- run --

def apply_variant(cfg: TrainNetConfig, variant: str,
                  dims: dict | None = None) -> tuple[
          TrainNetConfig, V03ValueConfig]:
    over = VARIANTS.get(variant, {})
    cfg = dataclasses.replace(
        cfg,
        weight_decay=over.get("weight_decay", cfg.weight_decay),
        dropout=over.get("dropout", cfg.dropout),
        d_model=over.get("d_model", cfg.d_model))
    return cfg, V03ValueConfig(d_model=cfg.d_model, dropout=cfg.dropout,
                               **(dims or {}))


def run_net_training(cfg: TrainNetConfig, log=print) -> tuple[
        dict, V03TensorData]:
    """v0.3 net train phase: verify data -> device-resident tensors ->
    (optional ablation) -> probe -> M concurrent members -> weights.

    Returns (metrics dict, V03TensorData).  Calibration (D14) and the
    net evaluation run separately (CLI), like the v0.2 train/eval split.
    """
    device = resolve_device(cfg.device)
    seed_everything(cfg.seed)
    if cfg.deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:                               # pragma: no cover
            pass
    _disable_mha_fastpath()
    dtype = resolve_dtype(cfg.dtype, device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.smoke:
        cfg.batch_size = min(cfg.batch_size, 64)
        cfg.td_batch = min(cfg.td_batch, 32)
        cfg.pair_batch = min(cfg.pair_batch, 32)
        cfg.eval_row_cap = cfg.eval_row_cap or 40
        cfg.probe_steps = min(cfg.probe_steps, 2)
        if cfg.max_steps is None:
            cfg.max_steps = 6
        cfg.eval_every_steps = min(cfg.eval_every_steps, 2)
    log(f"net train: device={device} dtype={dtype} "
        f"torch={torch.__version__} members={cfg.members} "
        f"seed={cfg.seed} variant={cfg.variant} features={cfg.features}")
    man = load_manifest(cfg.data_root, splits=("TRAIN", "CALIBRATION"),
                        chunks=cfg.chunks,
                        exclude_superseded=cfg.exclude_superseded,
                        verify_files=cfg.verify, log=log)
    ds = V0Dataset(man, feature_set=cfg.features, log=log)
    data = V03TensorData(ds, device, log=log)
    data.save_stats(out_dir / "stats.json")     # A4: ships with the weights
    log("net train: stats.json written (feature standardisation + per-tau "
        "medians + delta scales; ships with the weights)")

    ablation_table = None
    variant = cfg.variant
    if variant == "auto":
        aj = out_dir / "ablation.json"
        if aj.exists():
            variant = json.loads(aj.read_text(encoding="utf-8")).get(
                "chosen", "base")
            log(f"net train: variant {variant!r} loaded from "
                f"ablation.json")
        else:
            variant = "base"
    if cfg.ablation:
        ablation_table, chosen = run_ablation(cfg, data, device, dtype,
                                              log=log)
        if cfg.variant == "auto":
            variant = chosen

    cfg, cfg_model = apply_variant(cfg, variant, data.model_dims())
    probe = timing_probe(cfg, device, dtype, dims=data.model_dims(),
                         log=log)
    planned = _plan_steps(cfg, probe["ms_per_lockstep_step"], log)

    trainer = ConcurrentNetTrainer(cfg, data, device, dtype, log=log,
                                   cfg_model=cfg_model)
    if cfg.resume:
        trainer.try_resume()
    else:
        log("trainer: resume disabled (--no-resume) -> fresh start")
    deadline = (time.time() + cfg.time_budget_min * 60.0
                if cfg.time_budget_min and cfg.time_budget_min > 0
                else None)
    rep = None
    if trainer.step >= planned:
        # skip path (addendum 4c parity): the member files are written
        # from the persisted best states without further training
        log(f"trainer: already at lockstep step {trainer.step} >= "
            f"planned {planned}; skipping (members written from the "
            f"persisted best states)")
    else:
        rep = trainer.train(planned, deadline)
    sidecars = trainer.write_members(out_dir)

    metrics = dict(
        config=cfg.to_json(), device=str(device), dtype=dtype,
        variant=variant, format=FORMAT_V03,
        data=dict(manifest_sha256=man.sha256, n_sources=len(man),
                  files_verified=man.files_verified,
                  feature_set=cfg.features),
        probe=probe, planned_steps_per_member=int(planned),
        training=(rep if rep is not None
                  else _last_train_report(trainer, planned)),
        members=sidecars,
        ablation=ablation_table,
        oom_events=trainer.oom_events)
    return metrics, data


def _last_train_report(trainer: ConcurrentNetTrainer,
                       planned: int) -> dict:
    """A train() report for the metrics dict without re-running."""
    return dict(steps_done=int(trainer.step), planned=int(planned),
                completed=bool(trainer.completed),
                active=[float(a) for a in trainer.active.tolist()],
                best=trainer._best_json(),           # noqa: SLF001
                oom_events=list(trainer.oom_events))


def _plan_steps(cfg: TrainNetConfig, ms_per_lockstep: float,
                log=print) -> int:
    """B11: per-member steps from the measured lockstep rate; the cosine
    LR total follows (and is re-planned mid-run at --replan-after)."""
    if cfg.max_steps is not None:
        log(f"plan: fixed {cfg.max_steps} lockstep steps (= per member; "
            f"M={cfg.members} concurrent) -- --max-steps/--smoke")
        return int(cfg.max_steps)
    train_s = 0.85 * cfg.time_budget_min * 60.0   # rest: eval + ckpt + log
    steps = int(train_s * 1e3 / max(ms_per_lockstep, 1e-6))
    steps = max(min(steps, cfg.max_steps_cap), 50)
    log(f"plan: budget {cfg.time_budget_min:.0f} min -> {steps} lockstep "
        f"steps (= per member; M={cfg.members} concurrent; probe "
        f"{ms_per_lockstep:.0f} ms/lockstep-step, cap {cfg.max_steps_cap})")
    return steps
