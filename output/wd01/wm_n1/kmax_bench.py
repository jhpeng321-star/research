"""K-slot overflow cost probe (ANALYSIS ONLY) for WM-N1 Idea A.

Synthetic tensors only; no corpus / SUMO / K change.  Builds the REAL Idea A
member at the CODE_READY training config (hdv d_model=128, 4 heads, 3 layers,
ffn=512; arrival counter_channels=128; real 64-d CavMotionSummary projection +
CorrectionHead d_model=256) and measures, per K in {128,192,256}:

* one training forward+backward wall time (median of --reps, CPU 2 threads,
  BELOW_NORMAL priority);
* the exact autograd saved-tensor footprint of that training graph
  (saved_tensors_hooks), which is the per-batch activation memory;
* process peak RSS.

Rollout horizon / batch size follow the registered config (20 steps, B=8).
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

torch.set_num_threads(2)


def _proc_handle():
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    return k32.GetCurrentProcess()


def below_normal() -> bool:
    try:
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k32.SetPriorityClass.restype = ctypes.c_int
        ok = bool(k32.SetPriorityClass(_proc_handle(), 0x00004000))
        k32.GetPriorityClass.argtypes = [ctypes.c_void_p]
        k32.GetPriorityClass.restype = ctypes.c_uint
        cls = k32.GetPriorityClass(_proc_handle())
        return ok and cls == 0x00004000
    except Exception:
        return False


class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def _mem_info() -> "_PMC | None":
    try:
        k32 = ctypes.windll.kernel32
        psa = ctypes.windll.psapi
        psa.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
                                             ctypes.POINTER(_PMC),
                                             ctypes.c_ulong]
        psa.GetProcessMemoryInfo.restype = ctypes.c_int
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        if psa.GetProcessMemoryInfo(_proc_handle(), ctypes.byref(pmc),
                                    pmc.cb):
            return pmc
    except Exception:
        pass
    return None


def peak_rss_mb() -> float:
    pmc = _mem_info()
    return pmc.PeakWorkingSetSize / (1024.0 * 1024.0) if pmc else float("nan")


def working_set_mb() -> float:
    pmc = _mem_info()
    return pmc.WorkingSetSize / (1024.0 * 1024.0) if pmc else float("nan")


def build_real_member():
    from arrival import ArrivalConfig, IdeaAArrivalHead
    from corr import CorrConfig
    from hdv import HdvConfig, HdvResidualNet
    from train.g4_real import RealCorrBundle
    from train.idea_a import IdeaABundle

    torch.manual_seed(101)
    hdv = HdvResidualNet(HdvConfig(d_model=128, n_heads=4, n_layers=3,
                                   ffn=512, use_cav_flag=False))
    arrival = IdeaAArrivalHead(ArrivalConfig(counter_channels=128,
                                             d_model=256))
    corr = RealCorrBundle(CorrConfig(summary_dim=64, d_model=256))
    return IdeaABundle(ctrl=object(), hdv=hdv, arrival=arrival, corr=corr)


def make_batch(b: int, k: int, horizon: int, seed: int = 7) -> dict:
    import torch.nn.functional as F
    from model import constants as C

    g = torch.Generator().manual_seed(seed)
    slots = torch.arange(k, dtype=torch.float32)
    lane_idx = (slots.long() % C.N_LANES)
    one_hot = F.one_hot(lane_idx, C.N_LANES).float()
    s0 = (-100.0 + 20.0 * slots).unsqueeze(0).repeat(b, 1)
    v0 = (6.0 + 0.75 * (slots % 40.0)).unsqueeze(0).repeat(b, 1)
    a0 = (0.4 * torch.sin(slots * 1.3)).unsqueeze(0).repeat(b, 1)
    length0 = torch.full((b, k), float(C.NOMINAL_LENGTH_M))
    kin = torch.stack([s0,
                       one_hot[..., 0].expand(b, k),
                       one_hot[..., 1].expand(b, k),
                       one_hot[..., 2].expand(b, k),
                       v0, a0, length0], dim=-1)
    is_cav = torch.zeros(b, k, dtype=torch.bool)
    is_cav[:, :max(1, k // 16)] = True
    target_mask = torch.ones(b, k, dtype=torch.bool)
    baseline_idm = (-0.1 * (slots - k / 2.0)).unsqueeze(0).repeat(b, 1)
    true_accel = a0
    target_accel = baseline_idm + true_accel
    lane_target = torch.full((b, k), -1, dtype=torch.long)
    lane_target[:, 2] = 1
    counter = torch.rand(b, C.N_ENTRANCES, C.COUNTER_BINS, generator=g)
    arrivals = torch.zeros(b, C.N_ENTRANCES)
    summary_raw = torch.rand(b, 16, generator=g) * 0.1
    global_token = torch.zeros(b, 256)
    tau = torch.tensor(C.TAUS, dtype=torch.float32)
    arm = torch.arange(b) % 2
    y = (10.0 + 4.0 * arm).view(b, 1, 1).repeat(1, C.N_TAU, 1) \
        + 0.02 * tau.view(1, -1, 1)
    y = y.repeat(1, 1, C.N_OUTCOME_TARGETS)
    outcome_mask = torch.ones(b, C.N_TAU, dtype=torch.bool)
    collision = torch.zeros(b, C.N_TAU)

    teacher = torch.zeros(horizon + 1, b, k, 4)
    teacher[0, :, :, 0] = s0
    teacher[0, :, :, 1] = lane_idx.float()
    teacher[0, :, :, 2] = v0
    teacher[0, :, :, 3] = a0
    for t in range(1, horizon + 1):
        vp = teacher[t - 1, :, :, 2]
        vt = (vp + teacher[t - 1, :, :, 3] * 0.5).clamp(min=0.0)
        st = teacher[t - 1, :, :, 0] + 0.5 * (vp + vt) * 0.5
        teacher[t, :, :, 0] = st
        teacher[t, :, :, 1] = teacher[t - 1, :, :, 1]
        teacher[t, :, :, 2] = vt
        teacher[t, :, :, 3] = teacher[t - 1, :, :, 3]
    rollout_accel = true_accel.unsqueeze(0).repeat(horizon, 1, 1)
    rollout_baseline = baseline_idm.unsqueeze(0).repeat(horizon, 1, 1)
    rollout_mask = torch.ones(horizon, b, k, dtype=torch.bool)

    meta = []
    for i in range(b):
        ref = bool(arm[i].item() == 0)
        pk = f"w{i // 2}@t0=100"
        meta.append({"world": "TRAIN", "arm": "native" if ref else "wall_v31",
                     "policy": "native" if ref else "wall", "pair_key": pk,
                     "split": "TRAIN"})
    return {
        "kin": kin, "kin_mask": target_mask, "is_cav": is_cav,
        "baseline_idm": baseline_idm, "target_accel": target_accel,
        "target_mask": target_mask, "lane_target": lane_target,
        "counter": counter, "arrivals": arrivals, "summary_raw": summary_raw,
        "global_token": global_token, "outcome_quantiles": y,
        "outcome_mask": outcome_mask, "collision": collision,
        "meta": meta, "teacher": teacher, "rollout_accel": rollout_accel,
        "rollout_baseline_idm": rollout_baseline, "rollout_mask": rollout_mask,
        "s0": s0, "v0": v0, "a0": a0, "lane0": lane_idx.float().expand(b, k),
        "length0": length0, "present0": target_mask,
        "wall_group": -torch.ones(b, k, dtype=torch.long),
        "slot_id": torch.arange(k).unsqueeze(0).expand(b, k).contiguous(),
    }


def measure_saved_bytes(trainer, batch) -> int:
    total = {"bytes": 0}
    seen = set()

    def pack(t):
        try:
            ptr = t.untyped_storage().data_ptr()
        except Exception:
            ptr = None
        key = (ptr, t.numel() * t.element_size())
        if ptr is None or key not in seen:
            seen.add(key)
            total["bytes"] += int(t.numel() * t.element_size())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = trainer.total_loss(batch)
    del loss
    gc.collect()
    return total["bytes"]


def run_k(trainer, k: int, b: int, horizon: int, reps: int) -> dict:
    batch = make_batch(b, k, horizon)
    from train.idea_a import IdeaATrainConfig
    params = list(trainer.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    base_rss = working_set_mb()
    saved = measure_saved_bytes(trainer, batch)
    peak_after_mem = peak_rss_mb()
    gc.collect()

    torch.set_num_threads(2)
    times = []
    for rep in range(reps + 1):                     # +1 warmup
        opt.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        loss = trainer.total_loss(batch)
        loss.backward()
        t1 = time.perf_counter()
        del loss
        if rep > 0:
            times.append(t1 - t0)
        gc.collect()
    median = statistics.median(times)
    return {
        "K": k, "batch": b, "horizon": horizon,
        "median_s": float(median),
        "times_s": [float(x) for x in times],
        "saved_activation_mb": saved / (1024.0 * 1024.0),
        "working_set_after_mem_mb": peak_after_mem,
        "base_working_set_mb": base_rss,
        "peak_rss_mb": peak_rss_mb(),
        "n_params": int(sum(p.numel() for p in params)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, action="append",
                    default=None, help="repeatable; default 128,192,256")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    ks = args.k or [128, 192, 256]

    print(json.dumps({"event": "start", "below_normal": below_normal(),
                      "threads": torch.get_num_threads(),
                      "torch": torch.__version__,
                      "batch": args.batch, "horizon": args.horizon,
                      "reps": args.reps, "ks": ks}, sort_keys=True),
          flush=True)

    from train.idea_a import IdeaATrainer, IdeaATrainConfig
    bundle = build_real_member()
    cfg = IdeaATrainConfig(out_dir="unused", device="cpu",
                           batch_size=args.batch,
                           rollout_horizon_steps=args.horizon)
    trainer = IdeaATrainer(cfg, corpus=None, bundle=bundle)

    results = []
    for k in ks:
        rec = run_k(trainer, k, args.batch, args.horizon, args.reps)
        results.append(rec)
        print(json.dumps({"event": "result", **rec}, sort_keys=True),
              flush=True)

    t128 = next((r["median_s"] for r in results if r["K"] == 128), None)
    for r in results:
        if t128:
            r["time_ratio_vs_128"] = r["median_s"] / t128
    out = {"results": results, "batch": args.batch, "horizon": args.horizon,
           "reps": args.reps}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
    print(json.dumps({"event": "done", **out}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
