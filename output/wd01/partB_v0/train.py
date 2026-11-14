"""V0 training: M bootstrap members of the tau/policy-conditioned value net.

Implements the plan's training section (PLAN_PARTB_V0_TRAINING.md +
addenda): AdamW lr 3e-4 cosine, batch 512 default, fp16 autocast +
GradScaler on CUDA (T4 has no bf16), full fp32 on CPU; TD at commit 30 s
against a slow EMA target net mixed with MC anchors by lambda_TD (0.5
annealed to 0.8); row weights equalise native vs damping per tau; a
timing probe runs first and scales the step count into --time-budget-min;
checkpoints every --ckpt-every-min, plain state_dicts + JSON sidecar,
auto-resume with corrupt-latest -> previous fallback.

Design notes (documented choices):
  * TD applies to rows with tau >= 60; the target net is queried at
    tau - 30 (values 90/270/540 are OFF the label grid -- this is the
    learned-interpolation choice the plan allows: tau conditioning is
    sinusoidal/continuous and MC anchors at the 5 grid taus supervise the
    tau axis; the tau=60 -> 30 case is on-grid/exact).  A "grid-chained"
    TD target would equal the MC label identically (the builder verified
    label[t, tau] == chained D30 + label[t+30k, 60] to float noise), so
    it adds no signal.
  * TD rows are used even where the MC label is incomplete
    (label_mc_complete == 0): the bootstrap needs only d30 and the next
    state, which is exactly how supervision extends past the data end.
  * lambda-returns are deferred: V0 mixes one-step TD with MC by
    lambda_TD (the plan's annealing) instead of forward-view lambda
    traces.
  * DreamerV3-style return-percentile normalisation is subsumed by the
    symlog twohot head (scale-free, fixed bins); demand regime enters via
    the context vector and per-demand climatologies are used in eval.

ZERO simulation; CPU runs are smoke only (owner rule: no CPU training).
"""
from __future__ import annotations

import base64
import dataclasses
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .data import V0Dataset, bootstrap_world_ids, load_manifest
from .model import (V0ValueConfig, V0ValueNet, count_parameters, decode_mean,
                    nll_twohot, twohot_encode)

CKPT_DIR = "ckpt"
FORMAT = "wd01_partB_v0/1"

# torch.OutOfMemoryError (torch >= 2.5) / torch.cuda.OutOfMemoryError
# (older) exist on CPU-only builds too -- the OOM path is unit-testable
# by raising the same class on CPU.
_OOM_ERRORS = tuple(dict.fromkeys(
    e for e in (getattr(torch, "OutOfMemoryError", None),
                getattr(torch.cuda, "OutOfMemoryError", None))
    if e is not None))
MIN_BATCH = 64
MIN_TD_BATCH = 32


class CheckpointDataMismatch(RuntimeError):
    """Checkpoint's data MANIFEST sha differs from the loaded dataset."""


# --------------------------------------------------------------- device --

def resolve_device(prefer: str = "auto") -> torch.device:
    """--device > WD_DEVICE env > cuda-if-available > cpu."""
    if prefer not in ("auto", None, ""):
        return torch.device(prefer)
    env = os.environ.get("WD_DEVICE", "").strip()
    if env:
        return torch.device(env)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_autocast(device: torch.device, enabled: bool):
    """fp16 autocast guarded across torch >= 2.1 (T4: fp16, no bf16)."""
    if device.type != "cuda" or not enabled:
        import contextlib
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch.float16)


def _make_grad_scaler(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return None
    if hasattr(torch.amp, "GradScaler"):          # torch >= 2.3
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)  # torch 2.1/2.2 fallback


# --------------------------------------------------------------- config --

@dataclass
class TrainConfig:
    data_root: str
    out_dir: str
    members: int = 5
    seed: int = 20260928
    batch_size: int = 512
    td_batch: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    time_budget_min: float = 75.0
    ckpt_every_min: float = 10.0
    eval_every_steps: int | None = None     # None -> auto (steps // 8)
    max_steps: int | None = None            # per member; overrides scaling
    smoke: bool = False
    device: str = "auto"
    exclude_superseded: bool = True
    verify: bool = True
    chunks: tuple[str, ...] | None = None
    lambda_td_start: float = 0.5
    lambda_td_end: float = 0.8
    lambda_td_anneal_frac: float = 0.5      # anneal over this fraction
    tau_ema: float = 0.995
    w_coll: float = 1.0
    probe_steps: int = 12
    resume: bool = True
    value_model: str = "net"                # "net" | "gbm" (B3 item D17)
    features: str = "oracle"                # "oracle" | "deployable_strict"
    extras: dict = field(default_factory=dict)   # | "deployable_detector"

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["chunks"] = list(self.chunks) if self.chunks else None
        return d


# ------------------------------------------------------------ rng helpers --

def _torch_rng_b64() -> str:
    return base64.b64encode(torch.get_rng_state().numpy().tobytes()) \
        .decode("ascii")


def _restore_torch_rng(b64: str) -> None:
    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8).copy()
    torch.set_rng_state(torch.from_numpy(buf))


def _numpy_rng_json() -> list:
    st = np.random.get_state()
    return [st[0], st[1].tolist(), int(st[2]), int(st[3]), int(st[4])]


def _restore_numpy_rng(js: list) -> None:
    np.random.set_state((js[0], np.asarray(js[1], dtype=np.uint32),
                         int(js[2]), int(js[3]), int(js[4])))


# ------------------------------------------------------------- selection --

class CalSelectionRemoved(RuntimeError):
    """v0.3.2 fix #12: CAL-based model selection was REMOVED.  Selection
    and early stopping use TRAIN world-grouped OOB only (oob_crps300);
    CAL is read once, for the final report tables."""


def cal_native_crps300(net: V0ValueNet, ds: V0Dataset) -> float:
    """HARD GUARD (v0.3.2 fix #12): the old CALIBRATION-based
    model-selection signal is removed -- calling it always raises.

    Selection must use TRAIN world-grouped OOB only (oob_crps300 over the
    member's OOB TRAIN rows); CAL is read once, for the final report."""
    raise CalSelectionRemoved(
        "cal_native_crps300 (CALIBRATION-based model selection) was "
        "removed in v0.3.2: model selection must use TRAIN world-grouped "
        "OOB only (see oob_crps300); CAL is read once, for the final "
        "report tables")


def oob_crps300(net: V0ValueNet, ds: V0Dataset, oob_rows) -> float:
    """TRAIN world-grouped OOB selection signal (v0.3.2 fixes #9+#12):
    mean CRPS at tau=300 of ONE member over `oob_rows` -- the TRAIN rows
    of the member's out-of-bag bootstrap WORLDS (world-grouped bootstrap:
    both arms of an OOB world are out together).  Never sees CALIBRATION
    rows: guard_no_calibration_rows raises on any non-TRAIN row."""
    from .evaluate import _softmax_np, met
    from .gbm import guard_no_calibration_rows
    net.eval()
    r = ds.rows
    idx = np.asarray(oob_rows, dtype=np.int64)
    if len(idx) == 0:
        return float("nan")
    sub = r.loc[idx]
    sub = sub[np.isclose(sub["tau"], 300.0)]
    if len(sub) == 0:
        return float("nan")
    idx300 = sub.index.to_numpy()
    guard_no_calibration_rows(sub["split"].to_numpy(),
                              "oob_crps300 model selection")
    probs = []
    with torch.no_grad():
        for s in range(0, len(idx300), 512):
            out = net(ds.gather(idx300[s:s + 512]))
            probs.append(_softmax_np(out["v_delay"]))
    probs = np.concatenate(probs, axis=0)
    y = r.loc[idx300, "total_delay_ref_s"].to_numpy()
    return float(np.mean(met().crps(probs, y)))


# ----------------------------------------------------------------- probe --

def timing_probe(cfg: TrainConfig, device: torch.device, log=print) -> dict:
    """K steps on a throwaway member-0 clone -> ms/step + projections.

    Prints the parameter count.  On CUDA the allocator peak is MEASURED
    at the real (cfg) batch via torch.cuda.max_memory_allocated.  On
    CUDA OOM (probe or training) the batch halves (min 64) and the run
    continues; the effective batch lands in metrics/env_report and the
    checkpoint sidecars, so resume keeps it.  On CPU-only hosts the T4
    figure is an explicitly-labelled analytic estimate.  The probe model
    is discarded, so it cannot perturb the trained members or resume
    determinism.
    """
    default_bs = cfg.batch_size
    attempts = 0
    while True:
        seed_everything(cfg.seed)                   # member-0 init seed
        net = V0ValueNet(V0ValueConfig()).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
        scaler = _make_grad_scaler(device, enabled=True)
        n_params = count_parameters(net)
        bs, tdb = cfg.batch_size, cfg.td_batch
        dummy = dict(
            lane_bins=torch.zeros(bs, 3, net.cfg.lmax_bins, 3),
            lane_mask=torch.ones(bs, 3, net.cfg.lmax_bins),
            lane_geom=torch.ones(bs, 3, net.cfg.lmax_bins),
            cav_tokens=torch.zeros(bs, net.cfg.max_cavs, net.cfg.cav_raw),
            cav_mask=torch.zeros(bs, net.cfg.max_cavs),
            wall_tokens=torch.zeros(bs, net.cfg.max_walls,
                                    net.cfg.wall_raw),
            wall_mask=torch.zeros(bs, net.cfg.max_walls),
            demand=torch.zeros(bs, net.cfg.demand_raw),
            context=torch.zeros(bs, net.cfg.ctx_raw),
            tau=torch.full((bs,), 300.0),
            policy=torch.zeros(bs, dtype=torch.long),
        )

        def one_step():
            opt.zero_grad(set_to_none=True)
            with _make_autocast(device, True):
                out = net(dummy)
                logits = out["v_delay"]
                target = torch.zeros(bs, net.cfg.n_bins, device=device)
                target[:, net.cfg.n_bins // 2] = 1.0
                loss = torch.nn.functional.cross_entropy(logits, target) \
                    + out["p_coll"].float().mean()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

        times, oomed = [], False
        cuda_peak = None
        try:
            for i in range(max(2, cfg.probe_steps)):
                t0 = time.perf_counter()
                one_step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if i >= 2:                          # warmup discarded
                    times.append(time.perf_counter() - t0)
            if device.type == "cuda":
                cuda_peak = torch.cuda.max_memory_allocated() / 2 ** 30
        except _OOM_ERRORS:
            oomed = True
        if not oomed:
            break
        new_bs = max(MIN_BATCH, bs // 2)
        new_tdb = max(MIN_TD_BATCH, tdb // 2)
        if attempts >= 4 or (new_bs == bs and new_tdb == tdb):
            raise
        attempts += 1
        del net, opt, dummy
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log(f"probe: CUDA OOM at batch {bs}+{tdb}; halving to "
            f"{new_bs}+{new_tdb} and restarting the probe")
        cfg.batch_size, cfg.td_batch = new_bs, new_tdb

    cpu_peak = None if device.type == "cuda" \
        else _measure_cpu_peak(net, dummy)
    ms = float(np.mean(times)) * 1e3 if times else float("nan")
    probe = dict(
        device=str(device), batch_size=bs, td_batch=tdb,
        default_batch_size=default_bs,
        k_steps=max(2, cfg.probe_steps), warmup=2,
        ms_per_step=round(ms, 1), n_params_per_member=n_params,
        cuda_peak_gb=None if cuda_peak is None else round(cuda_peak, 3),
        cpu_peak_gb=None if cpu_peak is None else round(cpu_peak, 3),
        oom_fallback=bool(attempts))
    log(f"probe: {n_params / 1e6:.2f} M params/member; "
        f"{ms:.0f} ms/step (K={max(2, cfg.probe_steps)}, "
        f"batch={bs}+{tdb}, device={device})")
    if cuda_peak is not None:
        log(f"probe: CUDA allocator peak MEASURED at batch {bs}+{tdb}: "
            f"{cuda_peak:.2f} GiB -> "
            f"{'fits comfortably' if cuda_peak < 13 else 'DOES NOT FIT'} "
            f"in 16 GiB T4")
    if cpu_peak is not None:
        w_gib = 16 * n_params / 2 ** 30          # fp32 w+g+AdamW moments
        scale = default_bs / bs
        est = w_gib + cpu_peak * scale * 0.5     # fp16 activations
        log(f"probe: T4 fit ESTIMATE for default batch {default_bs}: "
            f"{w_gib:.2f} GiB fp32 weights+grads+AdamW moments + "
            f"{cpu_peak:.2f} GiB activations (CPU fwd/bwd peak measured "
            f"at batch {bs}+{tdb}) x {scale:.1f} (batch scale) x 0.5 "
            f"(fp16) ~= {est:.2f} GiB -> "
            f"{'fits comfortably' if est < 13 else 'RISK'} in 16 GiB; "
            f"the real GPU run measures the peak directly and the OOM "
            f"fallback halves the batch (min {MIN_BATCH}) if needed")
    return probe


def _measure_cpu_peak(net, dummy) -> float | None:
    """Peak memory of one fwd+bwd on CPU via torch.profiler (best effort)."""
    try:
        from torch.profiler import ProfilerActivity, profile
        net.zero_grad(set_to_none=True)
        with profile(activities=[ProfilerActivity.CPU],
                     profile_memory=True) as prof:
            out = net(dummy)
            loss = out["v_delay"].float().pow(2).mean()
            loss.backward()
        total = sum(getattr(e, "cpu_memory_usage", 0)
                    for e in prof.key_averages())
        net.zero_grad(set_to_none=True)
        return max(total, 0) / 2 ** 30
    except Exception:                               # pragma: no cover
        return None


# ---------------------------------------------------------------- member --

class MemberTrainer:
    """One ensemble member: bootstrap worlds, train, checkpoint, resume."""

    def __init__(self, cfg: TrainConfig, ds: V0Dataset, member: int,
                 device: torch.device, log=print):
        self.cfg = cfg
        self.ds = ds
        self.member = member
        self.device = device
        self.log = log
        self.seed = cfg.seed + 1000 * member
        self.out = Path(cfg.out_dir)
        self.ckpt = self.out / CKPT_DIR
        self.ckpt.mkdir(parents=True, exist_ok=True)
        # token widths are DATA-driven (v0.4 action/wall inputs: pre-v0.4
        # roots carry 17/6, v0.4 roots 20/11); the other hyper-parameters
        # stay at the V0 defaults
        self.cfg_model = V0ValueConfig(
            cav_raw=int(ds.feats["cav_tokens"].shape[-1]),
            wall_raw=int(ds.feats["wall_tokens"].shape[-1]))

        # bootstrap over TRAIN WORLDS (v0.3.2 fix #9: the sampling unit is
        # the world id -- a world's base and switching arms always move
        # together, never independently)
        sampled_worlds, oob_worlds = bootstrap_world_ids(
            ds.src_names, ds.src_split, cfg.seed, member)
        self.worlds = sampled_worlds          # world ids ("<chunk>/<w>")
        self.oob_worlds = list(oob_worlds)
        world_set = {w.split("/")[-1] for w in self.worlds}
        oob_name_set = {w.split("/")[-1] for w in self.oob_worlds}
        # rows of the bootstrap sample (TRAIN split only)
        src_split = np.array(ds.src_split)
        src_world = np.array([n.rsplit("__", 1)[0].split("/")[-1]
                              for n in ds.src_names])
        keep_src = np.array([
            (src_split[i] == "TRAIN") and (src_world[i] in world_set)
            for i in range(len(ds.src_names))])
        oob_src = np.array([
            (src_split[i] == "TRAIN") and (src_world[i] in oob_name_set)
            for i in range(len(ds.src_names))])
        rows = ds.rows
        in_pool = keep_src[rows["src_idx"].to_numpy()]
        self.pool = np.flatnonzero(in_pool)
        w = rows["row_weight"].to_numpy()[self.pool]
        self.pool_w = (w / w.sum()).astype(np.float64)
        # this member's OOB TRAIN rows: the model-selection rows (fix #12)
        in_oob = oob_src[rows["src_idx"].to_numpy()]
        self.oob_rows = np.flatnonzero(in_oob)

        td = ds.td
        td_ok = ((td["tau"] >= 60.0) & (td["t_idx"] >= 0)
                 & (td["tnext_idx"] >= 0)
                 & (td["split"] == "TRAIN")).to_numpy()
        td_ok &= keep_src[td["src_idx"].to_numpy()]
        self.td_pool = np.flatnonzero(td_ok)
        tw = ds.td_weights()[self.td_pool]
        self.td_pool_w = (tw / tw.sum()).astype(np.float64)

        seed_everything(self.seed)
        self.net = V0ValueNet(self.cfg_model).to(device)
        self.target = V0ValueNet(self.cfg_model).to(device)
        self.target.load_state_dict(self.net.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self._planned_steps = None
        self._set_cosine_lr(0)
        self.scaler = _make_grad_scaler(device, enabled=True)
        self.step = 0
        self.best = None                             # (crps, state_dict)
        self.best_step = -1
        self.completed = False
        self.oom_events: list[dict] = []

    # ------------------------------------------------------------- steps --
    def _set_cosine_lr(self, step: int, total: int | None = None):
        total = total or self._planned_steps or 1
        import math
        lr = self.cfg.lr * 0.5 * (1 + math.cos(math.pi * min(step / total,
                                                             1.0)))
        for g in self.opt.param_groups:
            g["lr"] = lr

    def _lambda_td(self, total: int) -> float:
        c = self.cfg
        frac = min(1.0, self.step / max(1, int(c.lambda_td_anneal_frac
                                                * total)))
        return c.lambda_td_start + (c.lambda_td_end - c.lambda_td_start) \
            * frac

    def _mc_batch(self, s: int):
        rng = np.random.default_rng([self.seed, s, 11])
        idx = rng.choice(self.pool, size=min(self.cfg.batch_size,
                                             len(self.pool)),
                         replace=len(self.pool) >= self.cfg.batch_size,
                         p=self.pool_w)
        b = self.ds.gather(idx)
        r = self.ds.rows.iloc[idx]
        y = dict(
            delay=r["total_delay_ref_s"].to_numpy().astype(np.float32),
            exp=r["ttc_lt3_veh_s"].to_numpy().astype(np.float32),
            brake=r["hard_brake_s"].to_numpy().astype(np.float32),
            coll=r["collision_any"].to_numpy().astype(np.float32))
        w = r["row_weight"].to_numpy().astype(np.float32)
        return b, y, w

    def _td_batch(self, s: int):
        rng = np.random.default_rng([self.seed, s, 13])
        idx = rng.choice(self.td_pool,
                         size=min(self.cfg.td_batch, len(self.td_pool)),
                         replace=len(self.td_pool) >= self.cfg.td_batch,
                         p=self.td_pool_w)
        b_t, b_n, d30 = self.ds.gather_td(idx)
        w = self.ds.td_weights()[idx].astype(np.float32)
        return b_t, b_n, d30, w

    # ---------------------------------------------------------- ckpt I/O --
    def _paths(self, prev: bool = False) -> dict:
        suf = ".prev" if prev else ""
        return dict(
            state=self.ckpt / f"m{self.member}_state{suf}.pt",
            optim=self.ckpt / f"m{self.member}_optim{suf}.pt",
            meta=self.ckpt / f"m{self.member}_meta{suf}.json",
            best=self.ckpt / f"m{self.member}_best.pt",
            best_meta=self.ckpt / f"m{self.member}_best.json")

    def save_best(self):
        """Persist the best-state weights (addendum 4c item 1) at every
        new best: ckpt/m{k}_best.pt + m{k}_best.json.  This is what makes
        the member file recoverable on the skip path after a restart.
        Selection is TRAIN world-grouped OOB only (v0.3.2 fix #12)."""
        if self.best is None or self.best[1] is None:
            return
        self.ckpt.mkdir(parents=True, exist_ok=True)
        p = self._paths()
        tmp = Path(str(p["best"]) + ".tmp")
        torch.save(self.best[1], tmp)
        os.replace(tmp, p["best"])
        _atomic_json(p["best_meta"], dict(
            format=FORMAT, member=self.member,
            step=int(getattr(self, "best_step", -1)),
            oob_crps300=float(self.best[0]),
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))

    def save_ckpt(self, planned_total: int):
        self.ckpt.mkdir(parents=True, exist_ok=True)
        p = self._paths()
        # rotate current -> .prev (state+optim+meta move atomically enough
        # for the fallback contract: .prev is a complete older triple)
        cur = self._paths(False)
        if cur["meta"].exists():
            prev = self._paths(True)
            for k in ("state", "optim", "meta"):
                if cur[k].exists():
                    os.replace(cur[k], prev[k])
        tmp = Path(str(p["state"]) + ".tmp")
        torch.save({k: v.cpu() for k, v
                    in self.net.state_dict().items()}, tmp)
        os.replace(tmp, p["state"])
        opt_state = {"optim": _cpu_opt_state(self.opt),
                     "target": {k: v.cpu() for k, v in
                                self.target.state_dict().items()},
                     "scaler": (self.scaler.state_dict()
                                if self.scaler is not None else None)}
        tmp = Path(str(p["optim"]) + ".tmp")
        torch.save(opt_state, tmp)
        os.replace(tmp, p["optim"])
        meta = dict(
            format=FORMAT, member=self.member, step=int(self.step),
            member_seed=self.seed, worlds=self.worlds,
            planned_total=int(planned_total),
            batch_size=int(self.cfg.batch_size),
            td_batch=int(self.cfg.td_batch),
            oom_events=list(self.oom_events),
            lambda_td=self._lambda_td(planned_total),
            best_oob_crps300=(None if self.best is None
                              else float(self.best[0])),
            completed=self.completed,
            rng=dict(python=list(random.getstate()[1]),
                     numpy=_numpy_rng_json(),
                     torch=_torch_rng_b64()),
            data_manifest_sha256=self.ds.man.sha256,
            cfg_model=self.cfg_model.to_json(),
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        _atomic_json(p["meta"], meta)

    def try_resume(self) -> int:
        """Load latest ckpt; on any failure fall back to .prev; else 0.

        Returns the restored step.  Refuses a checkpoint whose
        data_manifest_sha256 differs from the current dataset.
        """
        for prev in (False, True):
            p = self._paths(prev)
            if not p["meta"].exists():
                continue
            try:
                meta = json.loads(p["meta"].read_text(encoding="utf-8"))
                if meta.get("data_manifest_sha256") != self.ds.man.sha256:
                    raise CheckpointDataMismatch(
                        "checkpoint was trained against a different data "
                        f"MANIFEST sha ({meta.get('data_manifest_sha256')}"
                        f" != {self.ds.man.sha256}); delete "
                        f"{self.ckpt} to restart")
                state = torch.load(p["state"], map_location="cpu",
                                   weights_only=True)
                opt_state = torch.load(p["optim"], map_location="cpu",
                                       weights_only=True)
                self.net.load_state_dict(state)
                self.target.load_state_dict(opt_state["target"])
                _load_opt_state(self.opt, opt_state["optim"], self.device)
                if self.scaler is not None and \
                        opt_state.get("scaler") is not None:
                    self.scaler.load_state_dict(opt_state["scaler"])
                self.step = int(meta["step"])
                self.worlds = list(meta["worlds"])
                self.completed = bool(meta.get("completed", False))
                # keep the OOM-reduced batch a resumed run started with
                self.cfg.batch_size = int(meta.get("batch_size",
                                                   self.cfg.batch_size))
                self.cfg.td_batch = int(meta.get("td_batch",
                                                 self.cfg.td_batch))
                self.oom_events = list(meta.get("oom_events", []))
                self._planned_steps = meta.get("planned_total")
                if meta.get("best_oob_crps300") is not None:
                    self.best = (float(meta["best_oob_crps300"]), None)
                elif meta.get("best_cal_crps300") is not None:
                    # legacy pre-v0.3.2 checkpoint: value kept for resume
                    # arithmetic only; it was a CAL number and selection
                    # no longer reads CAL (fix #12)
                    self.best = (float(meta["best_cal_crps300"]), None)
                    try:            # best_step from the best meta, if any
                        bm = json.loads(
                            (self.ckpt /
                             f"m{self.member}_best.json").read_text(
                                 "utf-8"))
                        self.best_step = int(bm.get("step", -1))
                    except Exception:                       # noqa: BLE001
                        pass
                _restore_numpy_rng(meta["rng"]["numpy"])
                _restore_torch_rng(meta["rng"]["torch"])
                random.setstate((2, tuple(meta["rng"]["python"]), None))
                what = ".prev fallback" if prev else "latest"
                self.log(f"member {self.member}: resumed at step "
                         f"{self.step} from {p['meta'].name} ({what})")
                return self.step
            except CheckpointDataMismatch:
                raise                       # never train against wrong data
            except Exception as e:                  # noqa: BLE001
                self.log(f"member {self.member}: checkpoint "
                         f"{p['meta'].name} unusable "
                         f"({type(e).__name__}: {e}); "
                         + ("trying previous" if not prev else
                            "no usable checkpoint; starting fresh"))
                continue
        return 0

    # ------------------------------------------------------------ training --
    def train(self, planned_total: int, deadline: float | None,
              eval_fn, stop_after: int | None = None) -> dict:
        """Train from self.step to planned_total (or deadline/keys).

        stop_after (test/ops hook): stop this many steps after entry --
        an in-run interruption that leaves a NOT-completed checkpoint,
        the state a killed cloud run resumes from.
        """
        cfg = self.cfg
        t_start = time.time()
        self._planned_steps = planned_total
        ckpt_every_s = max(cfg.ckpt_every_min * 60.0, 1.0)
        last_ckpt = time.time()
        eval_every = cfg.eval_every_steps or max(20, planned_total // 8)
        log_every = max(5, planned_total // 20)
        loss_hist: list[float] = []
        resumed_at = self.step

        while self.step < planned_total:
            if deadline is not None and time.time() > deadline:
                self.log(f"member {self.member}: time budget reached at "
                         f"step {self.step}")
                break
            if stop_after is not None and self.step - resumed_at \
                    >= stop_after:
                self.log(f"member {self.member}: simulated interruption "
                         f"at step {self.step} (stop_after={stop_after})")
                break
            try:
                loss, parts = self._train_one_step(planned_total)
            except _OOM_ERRORS:
                self._handle_oom()
                continue                     # retry the same, smaller step
            loss_hist.append(loss)
            self.step += 1
            self._set_cosine_lr(self.step, planned_total)
            if self.step % log_every == 0 or self.step == planned_total:
                ms = 1e3 * (time.time() - t_start) / max(1, self.step
                                                        - resumed_at)
                self.log(f"member {self.member} step {self.step}/"
                         f"{planned_total} loss {loss:.4f} "
                         f"(mc_d {parts['mc_d']:.3f} td {parts['td']:.3f} "
                         f"exp {parts['exp']:.3f} brk {parts['brk']:.3f} "
                         f"coll {parts['coll']:.4f} lam {parts['lam']:.2f}) "
                         f"{ms:.0f} ms/step")
            now = time.time()
            if now - last_ckpt >= ckpt_every_s:
                self.save_ckpt(planned_total)
                last_ckpt = now
            if self.step % eval_every == 0 or self.step == planned_total:
                crps = eval_fn(self.net)
                self.net.train()
                if self.best is None or crps < self.best[0]:
                    self.best = (crps,
                                 {k: v.detach().cpu().clone() for k, v in
                                  self.net.state_dict().items()})
                    self.best_step = self.step
                    self.save_best()             # addendum 4c: persist best
                    self.log(f"member {self.member}: new best TRAIN-OOB "
                             f"CRPS@300 {crps:.1f} at step {self.step} "
                             f"(world-grouped OOB; CAL never read, fix "
                             f"#12)")
        self.completed = self.step >= planned_total
        if self.best is not None and self.best[1] is not None:
            self.net.load_state_dict(self.best[1])   # model selection
        else:
            self.best = (eval_fn(self.net),
                         {k: v.detach().cpu().clone() for k, v in
                          self.net.state_dict().items()})
            self.best_step = self.step
            self.save_best()
        self.save_ckpt(planned_total)
        final = eval_fn(self.net)
        return dict(member=self.member, steps_done=self.step,
                    resumed_from=resumed_at,
                    batch_size=self.cfg.batch_size,
                    td_batch=self.cfg.td_batch,
                    oom_events=list(self.oom_events),
                    best_oob_crps300=(None if self.best is None
                                      else float(self.best[0])),
                    final_oob_crps300=float(final),
                    n_oob_rows=int(len(self.oob_rows)),
                    mean_loss_last100=float(np.mean(loss_hist[-100:]))
                    if loss_hist else None,
                    train_seconds=round(time.time() - t_start, 1))

    def _handle_oom(self) -> None:
        """CUDA OOM: halve the batch (floored) and retry the step."""
        new_bs = max(MIN_BATCH, self.cfg.batch_size // 2)
        new_tdb = max(MIN_TD_BATCH, self.cfg.td_batch // 2)
        if new_bs == self.cfg.batch_size \
                and new_tdb == self.cfg.td_batch:
            raise                        # already at the floor: not OOM-fixable
        self.oom_events.append(dict(
            step=int(self.step),
            batch_size=self.cfg.batch_size, td_batch=self.cfg.td_batch,
            new_batch_size=new_bs, new_td_batch=new_tdb))
        self.log(f"member {self.member}: CUDA OOM at step {self.step}; "
                 f"halving batch {self.cfg.batch_size}+{self.cfg.td_batch} "
                 f"-> {new_bs}+{new_tdb} and retrying "
                 f"(recorded in metrics/env_report)")
        self.cfg.batch_size, self.cfg.td_batch = new_bs, new_tdb
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _train_one_step(self, planned_total: int) -> tuple[float, dict]:
        cfg, dev = self.cfg, self.device
        lam = self._lambda_td(planned_total)
        b, y, w = self._mc_batch(self.step)
        b_t, b_n, d30, w_td = self._td_batch(self.step)

        with _make_autocast(dev, True):
            out = self.net(b)
        out = {k: v.float() for k, v in out.items()}
        wn = torch.as_tensor(w, device=dev)
        wn = wn / wn.mean().clamp(min=1e-8)
        l_delay = nll_twohot(out["v_delay"],
                             twohot_encode(torch.as_tensor(
                                 y["delay"], device=dev)), wn)
        l_exp = nll_twohot(out["v_exp"],
                           twohot_encode(torch.as_tensor(
                               y["exp"], device=dev)), wn)
        l_brake = nll_twohot(out["v_brake"],
                             twohot_encode(torch.as_tensor(
                                 y["brake"], device=dev)), wn)
        coll = torch.as_tensor(y["coll"], device=dev).float()
        l_coll = torch.nn.functional.binary_cross_entropy_with_logits(
            out["p_coll"], coll, weight=wn)

        with _make_autocast(dev, True):
            out_t = self.net(b_t)
        out_t = {k: v.float() for k, v in out_t.items()}
        with torch.no_grad():
            out_n = self.target(b_n)
            v_next = decode_mean(out_n["v_delay"].float())
            td_val = torch.as_tensor(d30, device=dev) + v_next
            td_target = twohot_encode(td_val)
        wtd = torch.as_tensor(w_td, device=dev)
        wtd = wtd / wtd.mean().clamp(min=1e-8)
        l_td = nll_twohot(out_t["v_delay"], td_target, wtd)

        loss = (1.0 - lam) * l_delay + lam * l_td + l_exp + l_brake \
            + cfg.w_coll * l_coll
        self.opt.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                           cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                           cfg.grad_clip)
            self.opt.step()
        with torch.no_grad():                        # EMA target net
            for p, tp in zip(self.net.parameters(),
                             self.target.parameters()):
                tp.mul_(cfg.tau_ema).add_(p, alpha=1.0 - cfg.tau_ema)
        return float(loss.detach()), dict(mc_d=float(l_delay.detach()),
                                          td=float(l_td.detach()),
                                          exp=float(l_exp.detach()),
                                          brk=float(l_brake.detach()),
                                          coll=float(l_coll.detach()),
                                          lam=lam)


def _cpu_opt_state(opt) -> dict:
    """Optimizer state_dict with all tensors moved to CPU."""
    sd = opt.state_dict()

    def to_cpu(x):
        if torch.is_tensor(x):
            return x.detach().cpu()
        if isinstance(x, dict):
            return {k: to_cpu(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(to_cpu(v) for v in x)
        return x

    return to_cpu(sd)


def _load_opt_state(opt, sd, device: torch.device) -> None:
    def to_dev(x):
        if torch.is_tensor(x):
            return x.to(device)
        if isinstance(x, dict):
            return {k: to_dev(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(to_dev(v) for v in x)
        return x

    opt.load_state_dict(to_dev(sd))


def _atomic_json(path: Path, obj: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def write_member_weights(tr: MemberTrainer, man, device: torch.device,
                         log=print) -> dict:
    """Write member_{k}.pt + sidecar from the member's best state
    (addendum 4c item 2) -- on EVERY path: natural finish, the
    'already >= planned; skipping' path, and deadline/budget stop.

    Best-state preference: in-memory best > persisted ckpt/m{k}_best.pt >
    ckpt/m{k}_state.pt (last state, when no best was ever materialized).
    Returns the per-member report record."""
    out = Path(tr.cfg.out_dir)
    state, src = None, "memory_best"
    if tr.best is not None and tr.best[1] is not None:
        state = tr.best[1]
    if state is None:
        best_meta_path = tr.ckpt / f"m{tr.member}_best.json"
        if (tr.ckpt / f"m{tr.member}_best.pt").exists():
            state = torch.load(tr.ckpt / f"m{tr.member}_best.pt",
                               map_location="cpu", weights_only=True)
            src = "ckpt_best"
            try:
                meta = json.loads(best_meta_path.read_text("utf-8"))
                if tr.best is None or tr.best[0] is None:
                    v = meta.get("oob_crps300",
                                 meta.get("cal_native_crps300"))
                    tr.best = (float(v), None) if v is not None else None
                tr.best_step = int(meta.get("step", -1))
            except Exception:                           # noqa: BLE001
                pass
    if state is None:
        state_path = tr.ckpt / f"m{tr.member}_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(
                f"member {tr.member}: no best and no state checkpoint in "
                f"{tr.ckpt} -- cannot write member_{tr.member}.pt")
        state = torch.load(state_path, map_location="cpu",
                           weights_only=True)
        src = "ckpt_state"
    torch.save({k: v.cpu() for k, v in state.items()},
               out / f"member_{tr.member}.pt")
    best_crps = float(tr.best[0]) if tr.best is not None and \
        tr.best[0] is not None else None
    sidecar = dict(
        format=FORMAT, member=tr.member, member_seed=tr.seed,
        worlds_bootstrap=tr.worlds, steps=int(tr.step),
        best_step=int(getattr(tr, "best_step", -1)),
        weights_source=src,
        selection="TRAIN world-grouped OOB CRPS@300 (v0.3.2 fixes #9+#12;"
                  " CAL never read)",
        cfg_model=tr.cfg_model.to_json(),
        batch_size=tr.cfg.batch_size, td_batch=tr.cfg.td_batch,
        oom_events=list(tr.oom_events),
        best_oob_crps300=best_crps,
        data_manifest_sha256=man.sha256,
        device=str(device), torch=torch.__version__,
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _atomic_json(out / f"member_{tr.member}.json", sidecar)
    log(f"member {tr.member}: weights written from {src} "
        f"(step {tr.step}, best OOB CRPS "
        f"{'n/a' if best_crps is None else f'{best_crps:.1f}'})")
    return dict(member=tr.member, steps_done=int(tr.step),
                resumed_from=int(tr.step), weights_source=src,
                best_oob_crps300=best_crps,
                final_oob_crps300=best_crps,
                already_done=src != "memory_best")


# ------------------------------------------------------------------- run --

def run_training(cfg: TrainConfig, log=print) -> tuple[dict, V0Dataset]:
    """Full B2 train phase: verify data -> probe -> M members -> ckpts.

    Returns (metrics dict, dataset).  Evaluation is a separate phase
    (evaluate.py) invoked by the CLI with the same dataset.
    """
    device = resolve_device(cfg.device)
    seed_everything(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.smoke:
        cfg.batch_size = min(cfg.batch_size, 128)
        cfg.td_batch = min(cfg.td_batch, 64)
        if cfg.max_steps is None:
            cfg.max_steps = 6
    log(f"train: device={device} torch={torch.__version__} "
        f"members={cfg.members} seed={cfg.seed}")
    man = load_manifest(cfg.data_root, splits=("TRAIN", "CALIBRATION"),
                        chunks=cfg.chunks,
                        exclude_superseded=cfg.exclude_superseded,
                        verify_files=cfg.verify, log=log)
    ds = V0Dataset(man, log=log)

    probe = timing_probe(cfg, device, log=log)
    planned = _plan_steps(cfg, probe["ms_per_step"], log)

    deadline = (time.time() + cfg.time_budget_min * 60.0
                if cfg.time_budget_min and cfg.time_budget_min > 0
                else None)
    members_report = []
    for m in range(cfg.members):
        tr = MemberTrainer(cfg, ds, m, device, log=log)
        if cfg.resume:
            tr.try_resume()
        else:
            log(f"member {m}: resume disabled (--no-resume) -> "
                f"starting fresh")
        if tr.step >= planned:
            # skip path (addendum 4c): the member file is (re)written from
            # the persisted best state even when no training happens here
            log(f"member {m}: already at step {tr.step} >= planned "
                f"{planned}; skipping (writing member_{m}.pt from the "
                f"persisted best state)")
            rep = write_member_weights(tr, man, device, log=log)
            members_report.append(rep)
            continue
        def eval_fn(net, _ds=ds, _rows=tr.oob_rows):
            # v0.3.2 fixes #9+#12: selection on TRAIN world-grouped OOB
            # rows only; CAL is never read (cal_native_crps300 raises)
            return oob_crps300(net, _ds, _rows)
        rep = tr.train(planned, deadline, eval_fn)
        # final weights = selected best state, via the same writer as the
        # skip/budget paths (addendum 4c item 2)
        rep.update(weights_source="memory_best", already_done=False)
        write_member_weights(tr, man, device, log=log)
        log(f"member {m}: done (steps={rep['steps_done']}, "
            f"best TRAIN-OOB CRPS@300 "
            f"{rep['best_oob_crps300']:.1f})")
        members_report.append(rep)
    oom_events = []
    for rep in members_report:
        for e in rep.get("oom_events", []):
            oom_events.append(dict(member=rep.get("member"), **e))
    metrics = dict(
        config=cfg.to_json(), device=str(device),
        data=dict(manifest_sha256=man.sha256, n_sources=len(man),
                  n_superseded_skipped=man.n_superseded_skipped,
                  files_verified=man.files_verified),
        probe=probe, planned_steps_per_member=planned,
        oom_events=oom_events,
        members=members_report)
    return metrics, ds


def _plan_steps(cfg: TrainConfig, ms_per_step: float, log=print) -> int:
    """Steps per member: explicit --max-steps, else budget-scaled."""
    if cfg.max_steps is not None:
        log(f"plan: fixed {cfg.max_steps} steps/member (--max-steps/"
            f"--smoke); full-M projection at probe speed: "
            f"{cfg.max_steps * cfg.members * ms_per_step / 6e4:.1f} min")
        return int(cfg.max_steps)
    train_s = 0.85 * cfg.time_budget_min * 60.0     # rest: eval + overhead
    steps = int(train_s * 1e3 / max(ms_per_step, 1e-6) / cfg.members)
    steps = max(steps, 100)
    log(f"plan: budget {cfg.time_budget_min:.0f} min -> {steps} "
        f"steps/member x {cfg.members} members ~= "
        f"{steps * cfg.members * ms_per_step / 6e4:.1f} min train "
        f"(probe {ms_per_step:.0f} ms/step)")
    return steps
