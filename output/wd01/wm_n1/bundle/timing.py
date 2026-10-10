"""Bundle-side timing gate on the REAL ``model.net.WmN1Net`` member.

INTERFACES.md Mismatch 2: :mod:`evalh.timing_gate` measures an ad-hoc ``Toy``
Transformer (``d_in=16`` etc.), not the frozen WM-N1 member, so it cannot
bound the real member's cost.  This module wires the REAL
:class:`model.net.WmN1Net` with :class:`model.net.WmN1Config`, while REUSING
the evalh gate's thresholds, extrapolation formula and JSON schema:

* ``evalh.timing_gate.{BUDGET_HOURS_MAX,GATE_LIMIT_HOURS,ROLLOUT_FRACTION,
  EVAL_HOURS}`` (design WM_N1_DESIGN.md:226-227,229-231);
* ``evalh.timing_gate.extrapolate`` and ``GATE_QUOTE``;
* output schema ``wm_n1_timing_gate/1``, plus ``measured_model`` /
  ``proxy`` / ``parameter_count`` / ``architecture`` fields recording that the
  real member -- not the toy -- was measured.

The measured training step is a real forward pass + :func:`model.losses.
total_loss` backward; the rollout throughput is the real
:func:`model.rollout.rollout`.  Runnable on CPU (``--device cpu --tiny``).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from evalh import timing_gate as T
from model import constants as C
from model.net import WmN1Config, WmN1Net, parameter_count
from model import losses
from model.rollout import rollout as model_rollout

MEASURED_MODEL = "train.idea_a.IdeaATrainer (IdeaACorpus canonical batch)"
V1_MEASURED_MODEL = "model.net.WmN1Net"
REPLACES = ("INTERFACES.md Mismatch 2: evalh.timing_gate measures a Toy "
            "Transformer; this bundle wrapper measures the real WmN1Net AND "
            "the registered Idea A corpus member (IdeaACorpus canonical batch "
            "+ real P2/P4/P5 bundle), so the gate runs on the real training "
            "path the GPU boxes execute.")

# K_max is a FORMAT parameter (amendment 5, "K = 192"): a v2 pack gates at
# 128 slots, a v3 pack at 192.  The scene builder takes K from the gate
# config (``n_vehicles``), so a v3 gate is ``make_config(format="v3")``.
N_VEHICLES_BY_FORMAT = {fmt: int(k)
                        for fmt, k in sorted(C.FORMAT_K_MAX.items())}

#: Amendment 5 ("K = 192"): the registered member-1 / held-out (i) timing gate
#: runs on the pack_v3 real-batch path, so the Idea A corpus measurement builds
#: its synthetic pack at K_max = 192 by default (``make_config(format="v3")``
#: still selects the v1 scene width).
DEFAULT_GATE_FORMAT = C.FORMAT_V3
DEFAULT_GATE_K_MAX = int(C.K_MAX_V3)


def n_vehicles_for_format(format_name: str) -> int:
    """Registered slot width of a gate format (v2 -> 128, v3 -> 192)."""
    try:
        return N_VEHICLES_BY_FORMAT[format_name]
    except KeyError:
        raise ValueError(
            f"unknown format {format_name!r}; registered: "
            f"{sorted(N_VEHICLES_BY_FORMAT)}") from None


def activation_memory_estimate_mb(k_max: int, batch: int = 8) -> float:
    """Saved-activation estimate (MiB) of the registered A3.2 loss.

    ``KMAX_OPTIONS.md`` section 2 fit at batch 8 over the 22 graph passes
    (1 residual + 20 rollout + 1 contrast): ``MiB ~= 0.291*K^2 + 5.44*K``
    (measured 5470 / 11782 / 20487 MiB at K = 128 / 192 / 256); activation
    is linear in the batch size, so other batch sizes scale linearly.
    """
    k = float(k_max)
    b = float(batch) / 8.0
    return float(b * (0.291 * k * k + 5.44 * k))


def pack_bytes_estimate(k_max: int, k128_bytes: float) -> float:
    """Pack-disk estimate at ``k_max`` from a K=128 pack size in bytes.

    The K-dependent share (96.8 % of the corpus, ``KMAX_OPTIONS.md``
    section 1) scales with K; the K-independent share does not.
    """
    k_dep = 0.968 * float(k128_bytes) * (float(k_max) / 128.0)
    k_indep = 0.032 * float(k128_bytes)
    return float(k_dep + k_indep)


@dataclass
class GateConfig:
    """Real-member architecture + the full-plan schedule to extrapolate.

    ``n_vehicles`` is the slot width K of the measured scene: the v2 format
    default is 128; a v3 (amendment 5) gate passes ``n_vehicles=192`` or is
    built with ``make_config(format="v3")``.
    """

    name: str = "real-member"
    n_members: int = 5
    epochs: int = 15
    steps_per_epoch: int = 2000
    batch_size: int = 8
    n_vehicles: int = 128
    rollout_horizon_steps: int = 20
    seed: int = 0
    # DataLoader workers for the measured Idea A corpus step; 0 = in-process
    # (unit/preflight).  A real gate passes None -> min(8, cpu-2).
    num_workers: Optional[int] = 0
    # architecture (model.net.WmN1Config)
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    ffn: int = 768
    n_mix: int = C.N_MIX
    radius_m: float = 150.0
    bias_hidden: int = 64
    counter_channels: int = 128
    dropout: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)

    def wmn1_config(self) -> WmN1Config:
        return WmN1Config(
            d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_layers,
            ffn=self.ffn, n_mix=self.n_mix, radius_m=self.radius_m,
            bias_hidden=self.bias_hidden, counter_channels=self.counter_channels,
            dropout=self.dropout)


def make_config(tiny: bool = False, format: str | None = None,
                **overrides) -> GateConfig:
    cfg = GateConfig()
    if tiny:
        cfg = GateConfig(n_members=5, epochs=2, steps_per_epoch=4, batch_size=2,
                         n_vehicles=8, rollout_horizon_steps=4, d_model=16,
                         n_heads=2, n_layers=1, ffn=32, bias_hidden=8,
                         counter_channels=8)
    elif format is not None and "n_vehicles" not in overrides:
        overrides["n_vehicles"] = n_vehicles_for_format(format)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _torch(device: str):
    torch.set_num_threads(1)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")
    return torch


def _build_model(cfg: GateConfig, device: str) -> WmN1Net:
    torch.manual_seed(int(cfg.seed))
    return WmN1Net(cfg.wmn1_config()).to(device)


def make_scene(cfg: GateConfig, device: str, generator: torch.Generator) -> dict:
    """A frozen-format batch (the six tensors of features/FORMAT.md)."""
    B, K = cfg.batch_size, cfg.n_vehicles

    def r(shape, scale=1.0):
        return (torch.rand(*shape, generator=generator) * scale).to(device)

    veh = torch.zeros(B, K, C.VEH_F, device=device)
    for b in range(B):
        for i in range(K):
            lane = i % C.N_LANES
            veh[b, i, C.S_DOWN] = -150.0 + 30.0 * i
            veh[b, i, C.LANE_LEFT + lane] = 1.0
            veh[b, i, C.V_IDX] = 8.0 + 3.0 * i
            veh[b, i, C.LENGTH_IDX] = C.NOMINAL_LENGTH_M
            veh[b, i, C.IS_CAV_IDX] = 1.0 if i % 3 == 0 else 0.0
            veh[b, i, C.ACT_NATIVE if i % 3 == 0 else C.ACT_UNKNOWN] = 1.0
            veh[b, i, C.PRESENT_IDX] = 1.0
    batch = dict(
        veh_step=veh.to(device),
        veh_mask=torch.ones(B, K, dtype=torch.bool, device=device),
        plan=torch.zeros(B, K, C.N_SEGMENTS, C.PLAN_F, device=device),
        wall=torch.zeros(B, C.MAX_WALLS, C.WALL_F, device=device),
        wall_mask=torch.zeros(B, C.MAX_WALLS, dtype=torch.bool, device=device),
        counter=r((B, C.N_ENTRANCES, C.COUNTER_BINS), 2.0))
    batch["wall"][:, 0, 0] = 1.0
    batch["wall"][:, 0, 5] = 0.5
    batch["wall_mask"][:, 0] = True
    return batch


def make_targets(cfg: GateConfig, device: str,
                 generator: torch.Generator) -> dict:
    B, K = cfg.batch_size, cfg.n_vehicles
    return dict(
        accel=torch.randn(B, K, generator=generator).to(device),
        target_mask=torch.ones(B, K, dtype=torch.bool, device=device),
        lane=torch.zeros(B, K, dtype=torch.long, device=device),
        arrivals=torch.zeros(B, C.N_ENTRANCES, device=device),
        outcome_quantiles=torch.rand(B, C.N_TAU, C.N_OUTCOME_TARGETS,
                                     generator=generator).to(device) * 10.0,
        collision=torch.zeros(B, C.N_TAU, device=device))


def _peak_mb() -> float:
    """Peak process memory (MiB): POSIX RSS or Windows working set."""
    try:
        import resource
        return float(resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:                                    # pragma: no cover
        pass
    try:                                                 # Windows
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = _PMC()
        counters.cb = ctypes.sizeof(_PMC)
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters),
            counters.cb)
        if not ok:
            return float("nan")
        return float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0)
    except Exception:                                    # pragma: no cover
        try:
            import tracemalloc
            _, peak = tracemalloc.get_traced_memory()
            return float(peak) / 1e6
        except Exception:
            return float("nan")


def measure_throughput(cfg: GateConfig, device: str, *,
                       warmup: int = 2, timed_steps: int = 5,
                       model: Optional[WmN1Net] = None) -> dict:
    """Real-member training steps/s and rollout scene-steps/s.

    ``model`` may be supplied pre-built (and already moved) so a strict
    device audit can exclude the one legitimate host-side module-construction
    ``nn.Parameter`` from the measured path.
    """
    _torch(device)
    if model is None:
        model = _build_model(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(int(cfg.seed))

    def scene():
        return make_scene(cfg, device, gen)

    def step_loss(batch, targets):
        return losses.total_loss(model(**batch), targets)

    batch = scene()
    targets = make_targets(cfg, device, gen)
    model.train()
    for _ in range(warmup):
        opt.zero_grad()
        step_loss(batch, targets).backward()
        opt.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(timed_steps):
        opt.zero_grad()
        step_loss(batch, targets).backward()
        opt.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    steps_per_sec = timed_steps / train_s if train_s > 0 else float("nan")

    model.eval()
    with torch.no_grad():
        t1 = time.perf_counter()
        model_rollout(model, batch, n_steps=cfg.rollout_horizon_steps,
                      seed=int(cfg.seed))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        roll_s = time.perf_counter() - t1
    scene_steps = cfg.batch_size * cfg.rollout_horizon_steps
    rollout_per_sec = scene_steps / roll_s if roll_s > 0 else float("nan")
    return dict(steps_per_sec=float(steps_per_sec),
                rollout_scene_steps_per_sec=float(rollout_per_sec),
                rollout_samples_per_sec=float(rollout_per_sec),
                train_seconds=float(train_s), rollout_seconds=float(roll_s),
                peak_memory_mb=_peak_mb(), device=device,
                n_params=parameter_count(model))


def loss_decrease_check(cfg: GateConfig, device: str, *,
                        steps: int = 30) -> dict:
    """Overfit one fixed real batch: the total loss must strictly decrease."""
    _torch(device)
    model = _build_model(cfg, device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(int(cfg.seed) + 1)
    batch = make_scene(cfg, device, gen)
    targets = make_targets(cfg, device, gen)
    first = last = float("nan")
    for i in range(steps):
        opt.zero_grad()
        loss = losses.total_loss(model(**batch), targets)
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss.item())
        last = float(loss.item())
    return dict(first_loss=first, last_loss=last,
                decreased=bool(last < first), steps=int(steps))


def _idea_a_cfg(steps: int, seed: int, out_dir: str, k_max=None,
                batch_size: int = 8, num_workers: Optional[int] = 0):
    from train.idea_a import IdeaATrainConfig

    return IdeaATrainConfig(epochs=1, steps_per_epoch=max(1, int(steps)),
                            batch_size=max(1, int(batch_size)),
                            seeds=tuple([int(seed)] * 5),
                            out_dir=str(out_dir), k_max=k_max,
                            num_workers=num_workers)


def _idea_a_trainer(pack_dir, steps: int = 6, seed: int = 0, out_dir="unused",
                    k_max=None, device="cpu", batch_size: int = 8,
                    num_workers: Optional[int] = 0):
    """A real Idea A trainer over ``pack_dir`` through the canonical batch."""
    from train.idea_a import (IdeaACorpus, IdeaATrainer,
                              build_idea_a_bundle)

    ia_cfg = _idea_a_cfg(steps, seed, out_dir, k_max, batch_size=batch_size,
                         num_workers=num_workers)
    corpus = IdeaACorpus(str(pack_dir), split="TRAIN", pair_sampling=True,
                         horizon_steps=ia_cfg.rollout_horizon_steps,
                         k_max=ia_cfg.k_max)
    trainer = IdeaATrainer(ia_cfg, corpus, member=1, seed=int(seed),
                           device=device, bundle=build_idea_a_bundle(ia_cfg))
    return ia_cfg, corpus, trainer


def _release_corpus_handles(*corpora) -> None:
    """Release memmap handles + GC only; NEVER remove a timing workdir.

    A timing workdir holds the ``out`` directory (TRAIN_RECEIPT.json, the
    ``train_log.jsonl`` and the member checkpoint) plus the log/checkpoint
    evidence of a failure.  Deleting it (the old ``shutil.rmtree``) destroyed
    that evidence without owner approval, so this only drops the open ``.npy``
    memmap handles (Windows keeps the files locked while they are mapped) and
    lets GC reclaim them.  Every workdir is PRESERVED on success and failure and
    its path is returned/bound by the caller.
    """
    import gc

    for corpus in corpora:
        for holder in (corpus, getattr(corpus, "_v1", None)):
            mm = getattr(holder, "_mm", None)
            if isinstance(mm, dict):
                mm.clear()
    gc.collect()


def _timing_workdir(work_dir=None, prefix: str = "wmn1_timing_out_") -> Path:
    """Resolve (or create) a PRESERVED timing workdir."""
    import tempfile

    p = (Path(work_dir) if work_dir is not None
         else Path(tempfile.mkdtemp(prefix=prefix)))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_device(device: str) -> None:
    """Refuse a CUDA request on a host without CUDA (never a CPU-only skip)."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")


def _fit_preparation(trainer, source):
    """Run the trainer's CANONICAL fit preparation, strictly fail-closed.

    The registered throughput measurement must execute the exact batches and
    normalization the box's ``fit`` runs, so BEFORE any batch iterator or
    optimizer is built the timing path makes ONE strict call to the trainer's
    canonical ``prepare_training_source(source, windows=None)`` (coordinator
    contrastfix contract).  That single method centralizes everything ``fit``
    does pre-sampler - the A6 gate, the TRAIN-only source gate (A3.2), the
    eligible sibling pair pool bound on ``trainer._contrast_pair_pool`` with
    the fail-closed empty-pool refusal, and the TRAIN-only
    ``fit_outcome_norm(source, windows=None, ...)`` outcome normalization -
    so the timing path DUPLICATES NONE of it and has NO legacy/raw fallback:
    a trainer without the canonical method is refused outright instead of
    silently measuring a pre-contrastfix trainer (raw unsupported pair pool,
    no empty-pool refusal, identity normalization).

    DEPENDENCY (verified 2026-10-02, read-only): contrastfix2 HEAD 25479ae
    does NOT yet expose ``prepare_training_source`` (coordinator branch A
    pending).  Until that merge the real timing gate FAILS CLOSED with the
    error below - never the old trainer's behavior.

    Returns the eligible pair pool for diagnostics, read from
    ``trainer._contrast_pair_pool`` (what the canonical method bound);
    A's own return value is the fallback because this helper assumes
    NOTHING about the canonical method's return type.  No loss floor,
    schedule or optimizer math is changed here: the measured step remains
    one real ``_optim_step`` on the shared batches.
    """
    prepare = getattr(trainer, "prepare_training_source", None)
    if prepare is None or not callable(prepare):
        raise RuntimeError(
            "the timing gate requires the trainer's canonical "
            "prepare_training_source(source, windows=None) (coordinator "
            "contrastfix merge); refusing a legacy/raw fallback "
            "preparation that would measure the pre-contrastfix trainer "
            "(fail closed)")
    prepared = prepare(source, windows=None)
    return getattr(trainer, "_contrast_pair_pool", prepared)


def measure_idea_a_corpus(cfg: GateConfig, device: str, *,
                          warmup: int = 2, timed_steps: int = 5,
                          k_max: int = DEFAULT_GATE_K_MAX,
                          pack_dir=None, tiny: bool = False,
                          work_dir=None) -> dict:
    """Real Idea A corpus step throughput on the requested device.

    With ``pack_dir`` given the REGISTERED gate measures the box's REAL pack
    through :class:`IdeaACorpus` + the pair-aware batch sampler + all five
    A3.2 losses (CUDA allowed).  Without it the synthetic pack is built at the
    FORMAT K_max (``DEFAULT_GATE_K_MAX`` = 192, v3, amendment 5) and the corpus
    is opened with the matching explicit ``k_max`` - the ``--tiny``/CPU test
    path.  A NON-tiny gate with no real pack FAILS CLOSED (it never silently
    measures a synthetic pack).  Records the median s/step after warmup, peak
    memory (CUDA ``max_memory_allocated`` on GPU, RSS on CPU), the active loss
    terms, the resolved K_max, the EXACT supplied pack path (``pack_source``),
    ``source_kind`` (``real-pack``/``synthetic``), the EFFECTIVE batching path
    (``batch_path`` = ``dataloader``/``in-process``, the same choice ``fit``
    makes for the resolved workers) and the actual ``num_workers``.  BEFORE
    the sampler/optimizer are built the measurement makes ONE strict call to
    the trainer's canonical ``prepare_training_source(source, windows=None)``
    - everything ``fit`` does pre-sampler (A6 gate, TRAIN-only source gate,
    the eligible sibling pair pool from the real masks, the fail-closed
    empty-pool refusal and the TRAIN-only outcome normalization) is
    centralized there; a trainer without the canonical method FAILS CLOSED
    (no legacy/raw fallback) - see :func:`_fit_preparation`.

    The workdir (``work_dir`` argument, else a new temp dir) is PRESERVED on
    success and failure; its path and the ``out`` dir are returned so a failure
    receipt/checkpoint/log is never destroyed.
    """
    import statistics

    from bundle.e2e import write_synthetic_pack
    from train.idea_a import _make_optimizer

    _resolve_device(device)
    if not tiny and pack_dir is None:
        raise RuntimeError(
            "the non-tiny timing gate requires a real converted+verified pack "
            "(pack_dir); refusing to fall back to a synthetic pack")
    d = _timing_workdir(work_dir, "wmn1_timing_measure_")
    out_dir = Path(d) / "out"
    trainer = None
    corpus = None
    try:
        pack = Path(d) / "pack"
        if pack_dir is None:
            write_synthetic_pack(pack, seed=int(cfg.seed), k=int(k_max))
            source_kind = "synthetic"
        else:
            pack = Path(pack_dir)
            source_kind = "real-pack"
        ia_cfg, corpus, trainer = _idea_a_trainer(
            pack, steps=int(warmup) + int(timed_steps) + 2,
            seed=int(cfg.seed), out_dir=str(out_dir),
            k_max=int(k_max), device=device, batch_size=int(cfg.batch_size),
            num_workers=cfg.num_workers)
        # the registered gate measures the REAL path: every A3.2 term must be
        # computable (fail closed) and the pair-aware sampler guarantees one.
        # A's canonical prepare_training_source reads this (fit parity).
        trainer._require_all_terms = True
        # fit's PRE-SAMPLER preparation is shared BEFORE sampler/optimizer via
        # the ONE canonical trainer call (A6 gate, TRAIN-only gate, eligible
        # sibling pair pool, fail-closed empty pool, TRAIN-only outcome
        # normalization); a trainer without it fails closed - no raw fallback.
        _fit_preparation(trainer, corpus)
        # The SAME loader/in-process choice ``fit`` makes for the resolved
        # workers (idea_a.fit: resolved_num_workers() > 0 -> DataLoader).
        resolved_workers = int(ia_cfg.resolved_num_workers())
        if resolved_workers > 0:
            batches = trainer._iter_batches_loader(corpus)
            batch_path = "dataloader"
        else:
            batches = trainer._iter_batches(corpus)
            batch_path = "in-process"
        optimizer, _ = _make_optimizer(trainer, ia_cfg.lr)
        for _ in range(int(warmup)):
            trainer._optim_step(next(batches), optimizer)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        step_times: list = []
        for _ in range(int(timed_steps)):
            t0 = time.perf_counter()
            trainer._optim_step(next(batches), optimizer)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
        train_s = float(sum(step_times))
        s_per_step = float(statistics.median(step_times))
        n_params = int(sum(p.numel() for p in trainer.parameters()))
        active = list(getattr(trainer, "_last_terms", []))
        if device.startswith("cuda"):
            peak_cuda = int(torch.cuda.max_memory_allocated())
            peak_mb = peak_cuda / (1024.0 * 1024.0)
        else:
            peak_cuda = None
            peak_mb = _peak_mb()
        resolved_k = int(corpus.k_max) if corpus.k_max is not None \
            else int(k_max)
    finally:
        # PRESERVE the workdir (logs/checkpoint/failure receipt); release only
        # the open memmap handles.
        _release_corpus_handles(corpus)
    sps = float(timed_steps) / train_s if train_s > 0 else float("nan")
    return dict(steps_per_sec=sps, s_per_step=s_per_step,
                median_step_seconds=s_per_step, train_seconds=train_s,
                n_params=n_params, device=device, k_max=resolved_k,
                format=C.format_of_k_max(resolved_k),
                # exact supplied pack path (owner 2.3); kind kept separately
                pack_source=str(pack), source_kind=source_kind,
                batch_path=batch_path,
                work_dir=str(d), out_dir=str(out_dir),
                num_workers=resolved_workers,
                active_loss_terms=active, peak_memory_mb=float(peak_mb),
                peak_cuda_memory_bytes=peak_cuda)


def loss_decrease_idea_a(cfg: GateConfig, device: str, *,
                         steps: int = 30,
                         k_max: int = DEFAULT_GATE_K_MAX,
                         pack_dir=None, tiny: bool = False,
                         work_dir=None) -> dict:
    """Real Idea A corpus overfit: the total loss must strictly decrease.

    Uses the box's REAL pack through the canonical batch + pair-aware sampler
    when ``pack_dir`` is given (all five A3.2 terms active, fail closed), and
    the synthetic pack otherwise.  A NON-tiny run with no real pack FAILS
    CLOSED.  Runs on the requested device.  The workdir is PRESERVED on success
    and failure and its path (plus the ``out`` dir) is returned.
    """
    from bundle.e2e import write_synthetic_pack

    _resolve_device(device)
    if not tiny and pack_dir is None:
        raise RuntimeError(
            "the non-tiny timing gate requires a real converted+verified pack "
            "(pack_dir); refusing to fall back to a synthetic pack")
    d = _timing_workdir(work_dir, "wmn1_timing_loss_")
    out_dir = Path(d) / "out"
    corpus = None
    try:
        pack = Path(d) / "pack"
        if pack_dir is None:
            write_synthetic_pack(pack, seed=int(cfg.seed), k=int(k_max))
        else:
            pack = Path(pack_dir)
        ia_cfg, corpus, trainer = _idea_a_trainer(
            pack, steps=int(steps), seed=int(cfg.seed),
            out_dir=str(out_dir), k_max=int(k_max), device=device,
            batch_size=int(cfg.batch_size), num_workers=cfg.num_workers)
        receipt = trainer.fit()
        losses = [float(x) for x in receipt["losses"]]
        active = list(receipt.get("active_loss_terms", []))
    finally:
        _release_corpus_handles(corpus)
    first = losses[0] if losses else float("nan")
    last = losses[-1] if losses else float("nan")
    return dict(first_loss=first, last_loss=last,
                decreased=bool(losses and last < first), steps=len(losses),
                active_loss_terms=active, work_dir=str(d), out_dir=str(out_dir))


def run_timing_gate(cfg: GateConfig, *, device: str = "cpu",
                    tiny: bool = False, pack_dir=None,
                    budget_hours_max: float = T.BUDGET_HOURS_MAX,
                    gate_limit_hours: float = T.GATE_LIMIT_HOURS,
                    warmup: int = 2, timed_steps: int = 5,
                    loss_steps: int = 30, format: str | None = None,
                    k_max: int | None = None, work_dir=None) -> dict:
    """Run the real-member gate; returns a ``wm_n1_timing_gate/1`` dict.

    The registered measurement is the REAL Idea A corpus member
    (:func:`measure_idea_a_corpus` / :func:`loss_decrease_idea_a`); the v1
    ``model.net.WmN1Net`` measurement is kept as ``v1_member``.

    ``pack_dir`` is the converted+verified pack dir from the queue stages: when
    given, the registered Idea A measurement (throughput, peak memory, active
    loss terms, K_max and the projected hours) runs on that REAL pack and the
    requested device (CUDA allowed).  ``None`` keeps the synthetic pack for the
    ``--tiny``/CPU test path.

    ``format`` / ``k_max`` select the FORMAT slot width of the synthetic
    measured Idea A corpus (amendment 5): they default to v3 / K_max=192
    (``pack_v3``), so the registered member-1 and held-out (i) gates run on the
    K=192 real-batch path.  ``make_config(format="v2")`` with ``k_max=128`` is
    the v2 fallback.
    """
    fmt = DEFAULT_GATE_FORMAT if format is None else str(format)
    resolved_k = (int(k_max) if k_max is not None
                  else n_vehicles_for_format(fmt))
    # ONE preserved gate workdir; the measure/loss subdirs (logs, receipts,
    # checkpoints, failure evidence) are never removed.
    gate_work = _timing_workdir(work_dir, "wmn1_timing_gate_")
    t0 = time.perf_counter()
    idea_a = measure_idea_a_corpus(cfg, device, warmup=warmup,
                                   timed_steps=timed_steps, k_max=resolved_k,
                                   pack_dir=pack_dir, tiny=tiny,
                                   work_dir=gate_work / "idea_a")
    loss = loss_decrease_idea_a(cfg, device, steps=max(2, loss_steps),
                                k_max=resolved_k, pack_dir=pack_dir,
                                tiny=tiny, work_dir=gate_work / "loss")
    v1 = measure_throughput(cfg, device, warmup=warmup,
                            timed_steps=timed_steps)
    v1_loss = loss_decrease_check(cfg, device, steps=max(2, loss_steps))
    gate_seconds = time.perf_counter() - t0
    throughput = dict(idea_a)
    throughput["rollout_scene_steps_per_sec"] = \
        v1["rollout_scene_steps_per_sec"]
    throughput["rollout_samples_per_sec"] = v1["rollout_samples_per_sec"]
    throughput["v1_peak_memory_mb"] = v1["peak_memory_mb"]
    extra = T.extrapolate(cfg, throughput)
    within_budget = bool(extra["feasible"]
                         and extra["total_hours"] <= budget_hours_max)
    within_gate = bool(gate_seconds <= gate_limit_hours * 3600.0)
    idea_a_ok = bool(loss["decreased"] and idea_a["steps_per_sec"] > 0)
    passed = bool(within_budget and within_gate and loss["decreased"]
                  and idea_a_ok and bool(v1_loss["decreased"]))
    return dict(
        schema="wm_n1_timing_gate/1",
        generated_utc=datetime.now(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        device=device, tiny=bool(tiny),
        format=fmt, k_max=int(idea_a.get("k_max", resolved_k)),
        # exact supplied pack path (owner 2.3) + its kind, and the EFFECTIVE
        # batching path/worker count actually measured.
        pack_source=idea_a.get("pack_source", str(pack_dir)
                               if pack_dir is not None else "synthetic"),
        source_kind=idea_a.get("source_kind",
                               "real-pack" if pack_dir is not None
                               else "synthetic"),
        batch_path=idea_a.get("batch_path"),
        num_workers=int(idea_a.get("num_workers", 0)),
        pack_dir=(str(pack_dir) if pack_dir is not None else None),
        work_dir=str(gate_work),
        artifacts=dict(
            work_dir=str(gate_work),
            idea_a_work_dir=idea_a.get("work_dir"),
            idea_a_out_dir=idea_a.get("out_dir"),
            loss_work_dir=loss.get("work_dir"),
            loss_out_dir=loss.get("out_dir"),
        ),
        peak_memory_mb=float(idea_a.get("peak_memory_mb", float("nan"))),
        peak_cuda_memory_bytes=idea_a.get("peak_cuda_memory_bytes"),
        measured_model=MEASURED_MODEL, proxy=False,
        replaces_mismatch=REPLACES,
        config=cfg.to_json(), architecture=asdict(cfg.wmn1_config()),
        parameter_count=throughput["n_params"],
        throughput=throughput, loss_decrease=loss, extrapolation=extra,
        active_loss_terms=list(loss.get("active_loss_terms", [])),
        idea_a=dict(throughput=idea_a, loss_decrease=loss),
        v1_member=dict(measured_model=V1_MEASURED_MODEL, throughput=v1,
                       loss_decrease=v1_loss),
        thresholds=dict(budget_hours_max=budget_hours_max,
                        gate_limit_hours=gate_limit_hours,
                        rollout_fraction=T.ROLLOUT_FRACTION,
                        eval_hours=T.EVAL_HOURS,
                        citation="WM_N1_DESIGN.md:226-227,229-231"),
        gate_seconds=float(gate_seconds),
        checks=dict(within_budget=within_budget, within_gate=within_gate,
                    loss_decreased=bool(loss["decreased"]),
                    idea_a_decreased=bool(loss["decreased"]),
                    v1_loss_decreased=bool(v1_loss["decreased"])),
        passed=passed, quote=T.GATE_QUOTE)


def write_timing_gate(path, gate: dict) -> str:
    """Write ``TIMING_GATE.json`` and return the raw-BYTES sha256."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(gate, indent=1).encode("utf-8")
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(body)
    tmp.replace(p)
    return hashlib.sha256(body).hexdigest()
