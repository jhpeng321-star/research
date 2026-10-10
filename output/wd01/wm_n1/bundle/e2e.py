"""Bundle integration entry points: GPU-box executor + G5 e2e smoke.

The owner pre-flight G5 gate (``PREFLIGHT_GATES_IDEA_A.md``) runs a 60 s J4
TRAIN world through the queue stages convert -> timing (>= 30 s) -> train ->
checkpoint -> resume -> eval on CPU, and resolves
``bundle.e2e.run_e2e_smoke``.  P10 ``bundle.multibox`` resolves its default
executor ``bundle.e2e.run_config_job(box_id, config, work_dir)`` here too.

This module is *wiring only*: it drives the REAL pieces - the packed-corpus
loader (``train.dataset.MemmapCorpus``), the Idea A trainer on the real
P2/P4/P5 bundle (:func:`train.idea_a.build_idea_a_bundle`), the real-member
timing wrapper (:mod:`bundle.timing`), and the A3.4 held-out splits
(:func:`train.idea_a.heldout_split_report`).  The CPU smoke
(:func:`run_e2e_smoke`) writes a small synthetic packed corpus in place of the
GPU-box featurised upload (the upload verify/convert needs the featurised
shards, which are produced on the box).  The PRODUCTION executor
(:func:`run_config_job`) does NOT synthesise anything: it requires the real
converted+verified ``$WORK/pack``, CUDA, the frozen A6 record, the full
registered Idea A schedule and writes ``$WORK/members/member_<k>``.

No SUMO, CPU only, TRAIN worlds only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model import constants as C

WM_N1_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TIMING_SECONDS = 30.0

_TAUS = tuple(float(t) for t in C.TAUS)


# ------------------------------------------------------------ synthetic pack --
def _fill_veh(rng, veh, mask, base, t, k):
    for s in range(k):
        veh[base + t, s, C.S_DOWN] = -150.0 + 20.0 * s
        lane = s % C.N_LANES
        veh[base + t, s, C.LANE_LEFT + lane] = 1.0
        veh[base + t, s, C.V_IDX] = 8.0 + 2.0 * s
        veh[base + t, s, C.LENGTH_IDX] = C.NOMINAL_LENGTH_M
        veh[base + t, s, C.IS_CAV_IDX] = 1.0 if s < 3 else 0.0
        veh[base + t, s, C.ACT_NATIVE] = 1.0 if s < 3 else 0.0
        veh[base + t, s, C.ACT_UNKNOWN] = 0.0 if s < 3 else 1.0
        veh[base + t, s, C.PRESENT_IDX] = 1.0
        mask[base + t, s] = 1


def write_synthetic_pack(root, *, n_runs: int = 4, k: int = 16,
                         n_ticks: int = 3, seed: int = 0, runs=None) -> dict:
    """A minimal packed TRAIN corpus (sibling arms) the loader can open.

    ``runs`` (optional) overrides the built-in 4-run template with an explicit
    list of run dicts (``chunk``/``world``/``arm``/``policy``/``split``/
    ``t0_s``), used by the many-group pair-sampler tests.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if runs is None:
        runs = [
            dict(chunk="chunk4", world="w1", arm="base", policy="damping",
                 split="TRAIN", t0_s=540.0),
            dict(chunk="chunk4", world="w1", arm="native", policy="native",
                 split="TRAIN", t0_s=540.0),
            dict(chunk="chunk4", world="w1", arm="pi", policy="pi",
                 split="TRAIN", t0_s=540.0),
            dict(chunk="chunk4", world="w2", arm="native", policy="native",
                 split="TRAIN", t0_s=600.0),
        ][:n_runs]
    runs = list(runs)
    n_runs = len(runs)
    n_tick = n_runs * n_ticks
    veh = np.zeros((n_tick, k, C.VEH_F), dtype=np.float16)
    mask = np.zeros((n_tick, k), dtype=np.uint8)
    plan = np.zeros((n_runs, k, C.N_SEGMENTS, C.PLAN_F), dtype=np.float16)
    wall = np.zeros((n_tick, C.MAX_WALLS, C.WALL_F), dtype=np.float16)
    wall_mask = np.zeros((n_tick, C.MAX_WALLS), dtype=np.uint8)
    counter = rng.random((n_tick, C.N_ENTRANCES, C.COUNTER_BINS)).astype(
        np.float16)
    accel = rng.standard_normal((n_tick, k)).astype(np.float16)
    target_mask = np.zeros((n_tick, k), dtype=np.uint8)
    lane_next = np.zeros((n_tick, k), dtype=np.int8)
    arrivals = np.zeros((n_tick, C.N_ENTRANCES), dtype=np.int16)
    # P9 Idea A channels (the same per-tick arrays the real featurizer writes)
    veh_kin = np.zeros((n_tick, k, 7), dtype=np.float16)
    hdv_base = np.zeros((n_tick, k, 6), dtype=np.float16)
    cav_plan = np.zeros((n_tick, k, 8), dtype=np.float16)
    event_tok = np.zeros((n_tick, 8, 8), dtype=np.float16)
    event_mask = np.zeros((n_tick, 8), dtype=np.uint8)
    items, outcome_rows = [], []
    run_ranges = []
    for r, run in enumerate(runs):
        base = r * n_ticks
        for t in range(n_ticks):
            _fill_veh(rng, veh, mask, base, t, k)
            target_mask[base + t] = mask[base + t]
            if t == 1:
                lane_next[base + t, :k] = 0
            for s in range(k):
                lane = s % C.N_LANES
                v = 8.0 + 2.0 * s
                length = C.NOMINAL_LENGTH_M
                veh_kin[base + t, s] = np.array(
                    [-150.0 + 20.0 * s, 0, 0, 0, v, 0.0, length],
                    dtype=np.float16)
                veh_kin[base + t, s, 1 + lane] = 1.0
                hdv_base[base + t, s, 0] = -0.1 * (s - k / 2.0)
                if s < 3:
                    cav_plan[base + t, s, 0] = v / 40.0
                    cav_plan[base + t, s, 1] = 0.0
                    cav_plan[base + t, s, 5 + lane] = 1.0
        run_ranges.append((int(base), int(base + n_ticks)))
        for s in range(min(3, k)):
            plan[r, s, :, 0] = 1.0
        rows = []
        for tau in _TAUS:
            rows.append({
                "world": run["world"], "source": run["arm"],
                "chunk": run["chunk"], "split": run["split"],
                "t0_s": run["t0_s"], "t": run["t0_s"], "tau": tau,
                "total_delay_ref_s": 10.0 + 3.0 * r + tau / 30.0,
                "ttc_lt3_veh_s": 1.0 + 0.1 * tau / 30.0,
                "hard_brake_s": 0.25 * r,
                "collision_any": 1.0 if run["arm"] == "pi" else 0.0,
                "tau_available": True,
            })
        first = len(outcome_rows)
        outcome_rows.extend(rows)
        items.append({
            "world": run["world"], "arm": run["arm"], "chunk": run["chunk"],
            "split": run["split"], "t0_s": float(run["t0_s"]),
            "pair_key": f"{run['world']}@t0={run['t0_s']:g}",
            "policy": run["policy"], "tick": base, "plan_row": r,
            "outcome_rows": list(range(first, first + len(_TAUS))),
        })
    arrays = {
        "veh_step": veh, "veh_mask": mask, "plan": plan, "wall": wall,
        "wall_mask": wall_mask, "counter": counter, "accel_next": accel,
        "target_mask": target_mask, "lane_next": lane_next,
        "arrivals": arrivals, "veh_kin": veh_kin, "hdv_base": hdv_base,
        "cav_plan_ch": cav_plan, "event_tok": event_tok,
        "event_mask": event_mask,
    }
    for name, arr in arrays.items():
        np.save(root / f"{name}.npy", arr)
    pd.DataFrame(outcome_rows).to_parquet(root / "outcomes.parquet")
    files = {f"{name}.npy": name for name in arrays}
    (root / "INDEX.json").write_text(json.dumps({
        "schema": "wm_n1_corpus_pack/2", "k_max": int(k),
        "runs": [dict(tick_start=a, tick_end=b) for a, b in run_ranges],
        "files": files,
    }, indent=1), encoding="utf-8")
    manifest = {
        "schema": "wm_n1_pack/1", "k_max": int(k),
        "arrays": {name: {"file": f"{name}.npy", "dtype": str(arr.dtype),
                          "shape": list(arr.shape)}
                   for name, arr in arrays.items()},
        "outcomes": {"file": "outcomes.parquet"},
        "items": items,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                        encoding="utf-8")
    return manifest


# ---------------------------------------------------------------- timing ------
def run_timing_stage(device: str = "cpu", *,
                     min_seconds: float = DEFAULT_TIMING_SECONDS,
                     tiny: bool = True) -> dict:
    """Real-member timing stage, sustained to the gate's >= 30 s window."""
    import torch

    from bundle import timing as T
    from model import losses

    cfg = T.make_config(tiny=tiny)
    t0 = time.perf_counter()
    gate = T.run_timing_gate(cfg, device=device, tiny=tiny)
    extra_steps = 0
    if time.perf_counter() - t0 < float(min_seconds):
        T._torch(device)
        model = T._build_model(cfg, device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        gen = torch.Generator().manual_seed(int(cfg.seed))
        batch = T.make_scene(cfg, device, gen)
        targets = T.make_targets(cfg, device, gen)
        model.train()
        while time.perf_counter() - t0 < float(min_seconds):
            opt.zero_grad()
            losses.total_loss(model(**batch), targets).backward()
            opt.step()
            extra_steps += 1
    return {"gate": gate, "seconds": float(time.perf_counter() - t0),
            "extra_steps": int(extra_steps)}


# ------------------------------------------------------------- e2e smoke ------
def run_e2e_smoke(*, work_dir=None, device: str = "cpu", seed: int = 0,
                  timing_seconds: float = DEFAULT_TIMING_SECONDS) -> dict:
    """G5 entry point: convert -> timing -> train -> ckpt -> resume -> eval.

    Runs on CPU with the real Idea A trainer over the real P2/P4/P5 bundle.
    The packed corpus is a small synthetic TRAIN pack (the GPU-box featurised
    upload is not available on the CPU box); the timing stage is the real
    member measured for >= 30 s.
    """
    import tempfile

    from train import idea_a as ia

    base = Path(work_dir) if work_dir else Path(
        tempfile.mkdtemp(prefix="wmn1_e2e_"))
    base.mkdir(parents=True, exist_ok=True)
    stages = []

    pack = base / "pack"
    manifest = write_synthetic_pack(pack, seed=seed)
    corpus = ia.IdeaACorpus(str(pack), split="TRAIN", pair_sampling=True,
                            k_max=None)
    stages.append("convert")

    timing = run_timing_stage(device, min_seconds=timing_seconds, tiny=True)
    stages.append("timing")

    members = base / "members"
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=3, batch_size=4,
                              out_dir=str(members))
    trainer = ia.IdeaATrainer(cfg, corpus, member=1, seed=101,
                              bundle=ia.build_idea_a_bundle(cfg))
    receipt = trainer.fit()
    stages.append("train")

    ckpt = Path(receipt["checkpoint"])
    if not ckpt.is_file():
        raise RuntimeError(f"no checkpoint written at {ckpt}")
    stages.append("checkpoint")

    resumed = ia.IdeaATrainer(cfg, corpus, member=1, seed=101,
                              bundle=ia.build_idea_a_bundle(cfg))
    resumed.resume(ckpt)
    stages.append("resume")

    eval_receipt = resumed.evaluate_cal()
    stages.append("eval")

    return {
        "ok": True,
        "device": device,
        "stages": stages,
        "timing_seconds": float(timing["seconds"]),
        "timing_gate_passed": bool(timing["gate"].get("passed")),
        "timing_extra_steps": int(timing["extra_steps"]),
        "train_steps": int(receipt["steps"]),
        "n_worlds": int(len(manifest["items"])),
        "pack_dir": str(pack),
        "checkpoint": str(ckpt),
        "eval_status": eval_receipt.get("status"),
    }


# ------------------------------------------------------- multibox executor ----
#: Production member launch device.  The registered GPU run is CUDA-only; a
#: host without CUDA must fail closed rather than silently train on CPU.
PRODUCTION_DEVICE = "cuda"
MEMBERS_DIRNAME = "members"
PACK_DIRNAME = "pack"
BOXES_DIRNAME = "boxes"
MANIFEST_NAME = "manifest.json"
INDEX_NAME = "INDEX.json"


class ProductionLaunchError(RuntimeError):
    """A production member launch refused (missing real pack / hardware...)."""


class ProductionConfigRefusal(ProductionLaunchError):
    """The config is not an authoriseable Idea A member launch."""


def _resolve_run_root(work_dir, box_id: str) -> Path:
    """Top-level ``$WORK`` behind the per-box sandbox ``$WORK/boxes/<box>``.

    ``bundle.multibox.run_config`` hands the executor ``<work_dir>/boxes/<box>``
    (its own per-box state sandbox), but the registered member-1 run reads the
    real ``$WORK/pack`` and writes the handoff checkpoint ``$WORK/members/
    member_1``.  When the given dir is exactly that sandbox we step back up to
    the run root; any other path (direct/test calls) is used as-is.
    """
    wd = Path(work_dir)
    if wd.name == str(box_id) and wd.parent.name == BOXES_DIRNAME:
        return wd.parent.parent
    return wd


def _real_pack_k_max(pack_dir: Path) -> int:
    """Registered K_max of a real converted+verified pack (metadata only).

    Reads ``INDEX.json`` / ``manifest.json`` K_max and format labels only - the
    packed arrays and the outcomes are never interpreted here.

    PRODUCTION is pack_v3 only (amendment 5): a pack whose K_max is not 192, or
    whose ``INDEX.json``/``manifest.json`` does not self-describe
    ``format="v3"``, is refused before any corpus/model work.  The v2/K128 pack
    is the registered fallback for the synthetic/CPU test path only, never the
    production member launch.
    """
    try:
        index = json.loads((pack_dir / INDEX_NAME).read_text(encoding="utf-8"))
        manifest = json.loads(
            (pack_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except Exception as e:                                # noqa: BLE001
        raise ProductionLaunchError(
            f"real pack at {pack_dir} has unreadable {INDEX_NAME}/"
            f"{MANIFEST_NAME}: {e}") from e
    if int(index.get("k_max", -1)) != int(manifest.get("k_max", -2)):
        raise ProductionLaunchError(
            f"real pack {pack_dir}: {INDEX_NAME} K_max {index.get('k_max')!r} "
            f"!= {MANIFEST_NAME} K_max {manifest.get('k_max')!r}")
    k_max = int(index.get("k_max", -1))
    if k_max != int(C.K_MAX_V3):
        raise ProductionLaunchError(
            f"real pack {pack_dir}: production requires the pack_v3 format "
            f"K_max={int(C.K_MAX_V3)} (amendment 5); got K_max {k_max!r}. "
            f"A v2/K{int(C.K_MAX_V2)} pack is refused for the production "
            "member launch")
    index_fmt, manifest_fmt = index.get("format"), manifest.get("format")
    if index_fmt != C.FORMAT_V3 or manifest_fmt != C.FORMAT_V3:
        raise ProductionLaunchError(
            f"real pack {pack_dir}: production requires format "
            f"{C.FORMAT_V3!r} in both {INDEX_NAME} and {MANIFEST_NAME}; got "
            f"{index_fmt!r}/{manifest_fmt!r}")
    return k_max


def _refuse_existing_output(out_member_dir: Path, member: int) -> None:
    """Refuse to launch onto an existing member output (no implicit resume).

    A prior attempt's ``members/member_<k>`` dir, checkpoint or ``train_log``
    must never be overwritten: that would destroy a previous run's evidence and
    silently resume/append.  Any existing non-empty dir, checkpoint or log is
    refused before the corpus/trainer is built.
    """
    p = Path(out_member_dir)
    ckpt = p / f"member_{member}.pt"
    log = p / "train_log.jsonl"
    if ckpt.exists() or log.exists():
        raise ProductionLaunchError(
            f"member {member} output already contains a checkpoint/log at {p}; "
            "refusing to overwrite a previous attempt (no implicit resume)")
    if p.is_symlink() or p.is_file():
        raise ProductionLaunchError(
            f"member {member} output path {p} already exists and is not a "
            "fresh directory; refusing to overwrite (no implicit resume)")
    if p.is_dir() and any(p.iterdir()):
        raise ProductionLaunchError(
            f"member {member} output dir {p} already exists and is not empty; "
            "refusing to overwrite a previous attempt (no implicit resume)")


def run_config_job(box_id: str, config: str, work_dir, *,
                   device: str = PRODUCTION_DEVICE,
                   cuda_available=None) -> dict:
    """Production executor: train ONE registered Idea A member, then STOP.

    Fail-closed production semantics (the old smoke that synthesised a pack,
    forced ``epochs=1/steps=2/batch=4`` and trained on CPU is gone):

    * the REAL converted+verified pack must already exist at ``$WORK/pack``
      (no synthetic fallback);
    * ``state_global_token`` (A6) must be frozen - checked before any data;
    * the requested device must be CUDA (a ``cpu`` request is refused); a host
      without CUDA is refused;
    * the real pack must be pack_v3 (K_max 192, ``format="v3"``); a v2/K128
      pack is refused before any corpus/model work;
    * the member output dir must not already exist/contain a checkpoint or
      ``train_log`` (no implicit resume, no overwriting a prior attempt);
    * the FULL registered :class:`train.idea_a.IdeaATrainConfig` defaults are
      authoritative (15 epochs x 2000 steps, batch 8), with the registered
      member seed, the pack's registered K_max (pack_v3 -> 192) and host-scaled
      DataLoader workers (``num_workers=None``);
    * the handoff checkpoint lands at ``$WORK/members/member_<k>/member_<k>.pt``;
    * the returned receipt is validated against the actual trainer device, the
      full step schedule and the checkpoint before being reported.

    Only Idea A ensemble members are launched.  Held-out, the variant-N
    comparator and any CAL/eval path are refused here so a member run cannot
    auto-launch them; member-1 ends after its checkpoint.
    """
    from bundle.multibox import CONFIG_SPECS, IDEA_A_FAMILY
    from train import idea_a as ia

    spec = CONFIG_SPECS.get(config)
    if spec is None:
        raise ProductionLaunchError(f"unknown config {config!r}")
    if spec.kind != "member" or spec.family != IDEA_A_FAMILY:
        raise ProductionConfigRefusal(
            f"production member launch refuses {config!r} "
            f"(kind={spec.kind!r}, family={spec.family!r}); only Idea A "
            "ensemble members are launched here (held-out/N/CAL are never "
            "auto-launched)")
    member = int(spec.member)

    root = _resolve_run_root(work_dir, box_id)
    root.mkdir(parents=True, exist_ok=True)
    pack_dir = root / PACK_DIRNAME
    out_member_dir = root / MEMBERS_DIRNAME / f"member_{member}"

    cfg = ia.IdeaATrainConfig(member=member, ensemble_k=5,
                              out_dir=str(out_member_dir), device=str(device),
                              num_workers=None)
    seed = cfg.seed_for_member(member)

    # A6 first: the registered run has state_global_token ON and must have the
    # frozen Amendment 6 record; refuse before touching a dataset.
    if bool(cfg.state_global_token):
        ia._seals.require_frozen_amendment_6()

    # Hardware fail-closed: the registered run is CUDA-ONLY.  A non-CUDA
    # request (e.g. device="cpu") is refused outright, before any data or model
    # work, rather than silently training on CPU.
    if not str(device).startswith("cuda"):
        raise ProductionLaunchError(
            f"device {device!r} is not CUDA; the registered production member "
            "launch is CUDA-only and a CPU run is refused before any data or "
            "model work")
    if cuda_available is None:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        raise ProductionLaunchError(
            f"device {device!r} requested but CUDA is not available; refusing "
            "to run the registered member on CPU")

    # Real converted+verified pack ONLY - no synthetic fallback.
    if not pack_dir.is_dir() or not (pack_dir / MANIFEST_NAME).is_file() \
            or not (pack_dir / INDEX_NAME).is_file():
        raise ProductionLaunchError(
            f"real converted+verified pack required at {pack_dir} "
            f"({MANIFEST_NAME} + {INDEX_NAME}); refusing to build a synthetic "
            "pack in production")
    k_max = _real_pack_k_max(pack_dir)
    cfg.k_max = int(k_max)

    # Refuse to overwrite a previous attempt: no implicit resume.  The member
    # output dir must not already exist (or must be empty), and no checkpoint /
    # train log may already be present.  Checked before any corpus/model work.
    _refuse_existing_output(out_member_dir, member)

    corpus = ia.IdeaACorpus(str(pack_dir), split="TRAIN", pair_sampling=True,
                            horizon_steps=cfg.rollout_horizon_steps,
                            k_max=cfg.k_max)
    # Seed torch from the EXACT registered member seed BEFORE the network /
    # bundle factory runs: ``build_idea_a_bundle`` initialises the weights, so
    # they must depend only on the registered seed and never on the ambient
    # torch RNG state the process happens to carry (the trainer's own later
    # seed reset is unchanged and still governs the training loop).  The local
    # torch import is UNCONDITIONAL because the hardware check above imports
    # torch only when no ``cuda_available`` bypass was supplied, which would
    # otherwise leave torch unseeded on that path.
    import torch

    torch.manual_seed(int(seed))
    trainer = ia.IdeaATrainer(cfg, corpus, member=member, seed=seed,
                              device=str(device),
                              bundle=ia.build_idea_a_bundle(cfg))
    receipt = trainer.fit()

    # Validate the ACTUAL receipt, device, full schedule and checkpoint.
    expected_steps = int(cfg.effective_epochs()) * int(cfg.steps_per_epoch)
    got_steps = int(receipt.get("steps", -1))
    if got_steps != expected_steps:
        raise ProductionLaunchError(
            f"member {member} ran {got_steps} steps; expected the full "
            f"registered schedule {expected_steps} "
            f"({cfg.effective_epochs()} x {cfg.steps_per_epoch})")
    if int(receipt.get("member", -1)) != member:
        raise ProductionLaunchError(
            f"trained member {receipt.get('member')!r} != requested {member}")
    if int(receipt.get("seed", -1)) != seed:
        raise ProductionLaunchError(
            f"trained seed {receipt.get('seed')!r} != registered {seed}")
    actual_device = str(getattr(trainer, "device",
                                getattr(cfg, "device", "")))
    if str(device).startswith("cuda") and not actual_device.startswith("cuda"):
        raise ProductionLaunchError(
            f"trainer device {actual_device!r} is not the requested CUDA "
            f"device {device!r}")
    ckpt = Path(receipt.get("checkpoint")
                or (out_member_dir / f"member_{member}.pt"))
    if not ckpt.is_file():
        raise ProductionLaunchError(
            f"member {member} produced no checkpoint at {ckpt}")
    return {
        "ok": True, "kind": "member", "config": config,
        "box_id": str(box_id), "member": member, "seed": seed,
        "device": actual_device, "steps": got_steps,
        "epochs": int(receipt.get("epochs", cfg.effective_epochs())),
        "expected_steps": expected_steps,
        "ensemble_k": int(cfg.ensemble_k),
        # owner 2.3 (same as the timing gate): pack_source is the EXACT
        # supplied pack path; the kind is kept separately.
        "pack_dir": str(pack_dir), "pack_source": str(pack_dir),
        "source_kind": "real-pack",
        "k_max": int(cfg.k_max), "format": C.format_of_k_max(int(cfg.k_max)),
        "num_workers": int(cfg.resolved_num_workers()),
        "state_global_token": bool(cfg.state_global_token),
        "checkpoint": str(ckpt),
    }


__all__ = ["run_e2e_smoke", "run_config_job", "run_timing_stage",
           "write_synthetic_pack", "ProductionLaunchError",
           "ProductionConfigRefusal", "PRODUCTION_DEVICE"]
