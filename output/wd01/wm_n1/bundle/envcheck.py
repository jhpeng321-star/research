"""GPU-box environment self-check (``python -m bundle.envcheck``).

Run this **on the rented GPU box** right after installing the environment and
before any queue stage.  It refuses to pass when the box cannot support the
WM-N1 bundle:

* ``torch < 2.7`` (Blackwell-class cu128 wheels start at torch 2.7);
* a Blackwell-class GPU (compute capability major 12, e.g. sm_120) is present
  but ``torch.cuda.get_arch_list()`` lacks ``sm_120`` / ``compute_120``;
* ``--device cuda`` was requested but CUDA is unavailable.

The sm_120 arch requirement applies to Blackwell ONLY.  An Ada card such as
the RTX 4090 D (compute capability ``(8, 9)``, sm_89) is accepted without an
sm_120 arch token, as is any non-Blackwell CUDA GPU, as long as torch>=2.7.

When a GPU is usable it runs a tiny forward **and backward** of the REAL
``model.net.WmN1Net`` member (the exact module the bundle timing gate measures)
on the selected device via :mod:`bundle.timing`, so a CUDA kernel/arch problem
surfaces here rather than mid-run.  On a CPU-only box the same path runs on
``cpu`` (the local parity reference, torch 2.12.1+cpu).

    python -m bundle.envcheck                 # auto: cuda if present, else cpu
    python -m bundle.envcheck --device cpu    # force the local CPU fallback
    python -m bundle.envcheck --device cuda --json

Exit code 0 = PASS, 2 = refused (message on stdout/stderr).
"""
from __future__ import annotations

import argparse
import json
import sys

#: Minimum supported torch (major, minor).  cu128 Blackwell wheels: >= 2.7.
MIN_TORCH = (2, 7)
#: Arch-list tokens that prove the wheel carries Blackwell sm_120 kernels.
SM120_ARCHES = ("sm_120", "compute_120")


def parse_version(version: str):
    """Parse ``"2.12.1+cpu"`` -> ``(2, 12)``; ``None`` if unparseable."""
    head = str(version).split("+", 1)[0].split("-", 1)[0]
    parts = head.split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return None


def version_ok(version: str) -> bool:
    """True when ``version >= 2.7`` (Blackwell cu128 minimum)."""
    mm = parse_version(version)
    return mm is not None and mm >= MIN_TORCH


def is_blackwell(capability) -> bool:
    """A compute capability ``(major, minor)`` with major 12 (sm_120 class)."""
    if not capability:
        return False
    try:
        return int(capability[0]) >= 12
    except (TypeError, ValueError, IndexError):
        return False


def has_sm120_arch(arch_list) -> bool:
    """True when ``arch_list`` contains an sm_120 / compute_120 token."""
    for token in arch_list or ():
        text = str(token).strip().lower()
        if text in SM120_ARCHES:
            return True
        if text.endswith("120") and ("sm" in text or "compute" in text):
            return True
    return False


def evaluate_environment(*, version, cuda_available, device_requested,
                         capability=None, arch_list=()) -> dict:
    """Pure policy check (no torch calls) so it is testable off a GPU."""
    reasons = []
    checks = {}

    checks["torch_min_2_7"] = bool(version_ok(version))
    if not checks["torch_min_2_7"]:
        reasons.append(
            f"torch {version} < {MIN_TORCH[0]}.{MIN_TORCH[1]} (Blackwell cu128 "
            "requires torch>=2.7)")

    requested = str(device_requested).lower()
    if requested == "cuda" and not cuda_available:
        checks["device"] = False
        reasons.append("--device cuda requested but torch.cuda.is_available() "
                       "is False")
    else:
        checks["device"] = True
    if requested == "auto":
        selected = "cuda" if cuda_available else "cpu"
    elif requested == "cuda":
        selected = "cuda" if cuda_available else "cpu"
    else:
        selected = requested

    blackwell = bool(cuda_available) and is_blackwell(capability)
    checks["blackwell_detected"] = blackwell
    if blackwell:
        arch_ok = has_sm120_arch(arch_list)
        checks["blackwell_sm120_arch"] = arch_ok
        if not arch_ok:
            reasons.append(
                "Blackwell GPU detected (compute capability "
                f"{tuple(capability)}) but torch.cuda.get_arch_list()="
                f"{list(arch_list)} lacks sm_120/compute_120; install the "
                "cu128 wheel")
    else:
        checks["blackwell_sm120_arch"] = True

    return {"ok": not reasons, "reasons": reasons, "checks": checks,
            "device": selected, "blackwell": blackwell}


def _torch_info() -> dict:
    import torch
    info = {
        "version": str(torch.__version__),
        "compiled_cuda": (str(torch.version.cuda)
                          if torch.version.cuda is not None else None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": (str(torch.version.cuda)
                         if torch.version.cuda is not None else None),
        "device_count": (int(torch.cuda.device_count())
                         if torch.cuda.is_available() else 0),
        "device_name": None,
        "capability": None,
        "arch_list": [],
    }
    if hasattr(torch.cuda, "get_arch_list"):
        try:
            info["arch_list"] = list(torch.cuda.get_arch_list())
        except Exception:                                # noqa: BLE001
            pass
    if torch.cuda.is_available():
        try:
            info["device_name"] = str(torch.cuda.get_device_name())
            cap = torch.cuda.get_device_capability()
            info["capability"] = (int(cap[0]), int(cap[1]))
        except Exception:                                # noqa: BLE001
            pass
    return info


def _forward_backward(device: str, tiny: bool = True) -> dict:
    """A tiny real-member forward + backward on ``device``."""
    import torch

    from bundle import timing as T
    from model import losses

    try:
        cfg = T.make_config(tiny=tiny)
        model = T._build_model(cfg, device)
        gen = torch.Generator().manual_seed(int(cfg.seed))
        batch = T.make_scene(cfg, device, gen)
        targets = T.make_targets(cfg, device, gen)
        model.train()
        loss = losses.total_loss(model(**batch), targets)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        grad_finite = all(
            bool(torch.isfinite(g).all().item()) for g in grads)
        loss_finite = bool(torch.isfinite(loss).item())
        return {"ok": bool(loss_finite and grad_finite and grads),
                "device": device, "loss": float(loss.item()),
                "n_grad_tensors": len(grads), "grad_finite": grad_finite,
                "n_params": int(T.parameter_count(model)), "error": None}
    except Exception as e:                               # noqa: BLE001
        return {"ok": False, "device": device, "loss": None,
                "n_grad_tensors": 0, "grad_finite": False, "n_params": 0,
                "error": f"{type(e).__name__}: {e}"}


def run_selfcheck(device: str = "auto", *, tiny: bool = True,
                  with_forward_backward: bool = True,
                  format: str | None = None) -> dict:
    """Run the full self-check; return a JSON-serialisable report.

    ``format`` (v2/v3, amendment 5) selects the slot width K the report
    sizes for: it records ``k_max`` and the saved-activation memory
    estimate of the registered A3.2 loss at that K.  The default is now
    v3 (K_max=192, ``pack_v3``), the format the member-1 / held-out (i)
    c_m1only box configs run on; pass ``format="v2"`` for the frozen v2
    (K_max=128) estimate.
    """
    from bundle.timing import (activation_memory_estimate_mb,
                               n_vehicles_for_format)

    fmt = "v3" if format is None else str(format)
    k_max = n_vehicles_for_format(fmt)     # raises ValueError for unknown
    info = _torch_info()
    env = evaluate_environment(
        version=info["version"], cuda_available=info["cuda_available"],
        device_requested=device, capability=info["capability"],
        arch_list=info["arch_list"])
    report = {"ok": bool(env["ok"]), "torch": info, "environment": env,
              "format": fmt, "k_max": int(k_max),
              "memory_estimate_mb": {
                  "saved_activations_a32_b8":
                      round(activation_memory_estimate_mb(k_max, batch=8),
                            1),
                  "fit": "KMAX_OPTIONS.md section 2: MiB ~= 0.291*K^2 + 5.44*K "
                         "at batch 8 (linear in batch)",
              },
              "forward_backward": None}
    if env["ok"] and with_forward_backward:
        fb = _forward_backward(env["device"], tiny=tiny)
        report["forward_backward"] = fb
        report["ok"] = bool(fb["ok"])
    return report


def _print_report(report: dict) -> None:
    info = report["torch"]
    env = report["environment"]
    print(f"torch: {info['version']} (compiled cuda: {info['compiled_cuda']})")
    print(f"cuda available: {info['cuda_available']} "
          f"(devices: {info['device_count']}, "
          f"name: {info['device_name']}, capability: {info['capability']})")
    print(f"arch list: {info['arch_list']}")
    print(f"selected device: {env['device']} "
          f"(blackwell detected: {env['blackwell']})")
    print(f"format: {report.get('format')} k_max: {report.get('k_max')} "
          f"(saved-activation estimate A3.2 B=8: "
          f"{report.get('memory_estimate_mb', {}).get('saved_activations_a32_b8')} MB)")
    for key, value in env["checks"].items():
        print(f"  check {key}: {value}")
    fb = report.get("forward_backward")
    if fb is not None:
        print(f"forward/backward: ok={fb['ok']} device={fb['device']} "
              f"loss={fb['loss']} grad_finite={fb['grad_finite']} "
              f"n_params={fb['n_params']} error={fb['error']}")
    for reason in env["reasons"]:
        print(f"  refusal: {reason}")
    if report["ok"]:
        print("ENVCHECK PASS")
    else:
        print("ENVCHECK FAIL")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bundle.envcheck",
        description="WM-N1 GPU-box environment self-check (torch>=2.7 cu128, "
                    "Blackwell sm_120, real-member forward/backward)")
    p.add_argument("--device", default="auto",
                   choices=("auto", "cpu", "cuda"),
                   help="auto (default): cuda when available, else cpu")
    p.add_argument("--no-forward-backward", action="store_true",
                   help="skip the tiny real-member forward/backward")
    p.add_argument("--format", default=None, choices=("v2", "v3"),
                   help="slot-width format the memory estimate sizes for: "
                        "v3 K_max=192 (default, amendment 5 pack_v3), "
                        "v2 K_max=128")
    p.add_argument("--json", action="store_true",
                   help="print the report as JSON only")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = run_selfcheck(args.device, format=args.format,
                           with_forward_backward=not args.no_forward_backward)
    if args.json:
        print(json.dumps(report, indent=1, default=str))
    else:
        _print_report(report)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
