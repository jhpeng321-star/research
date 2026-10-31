"""gpuR off-critical-path panel evaluator (MAIN ADDENDUM 2026-10-31 11:52).

Evaluates the full 12x2x{120,300} s TRAIN panel paired-J MAE for one immutable
exported pilot checkpoint (step_XXXXXX.pt) with the SAME code, inputs and CRN
as the gpuA pilot (pilot.panel), so training steps on gpuA are never blocked by
panel evaluations. Writes a PANEL_{name}.json receipt byte-compatible with the
pilot's own record_panel format for relay back into the pilot out dir.
"""
from __future__ import annotations

import argparse
import json
import socket
import time

from tools.wmfixmerge.data import WM2_SHA, digest, install_guard, safe


def load_candidate(rt, base_checkpoint, candidate):
    """Strictly load an exported pilot checkpoint into the scoring runtime."""
    import torch
    payload = torch.load(safe(candidate), map_location="cpu", weights_only=True)
    original = torch.load(safe(base_checkpoint), map_location="cpu", weights_only=True)
    if payload.get("wmfix", {}).get("base_sha256") != WM2_SHA:
        raise ValueError("candidate does not carry the registered WM2 base")
    rt.model.load_state_dict(payload["model"], strict=True)
    for name, value in payload["model"].items():
        if not name.startswith("merge_response.") and not torch.equal(value, original["model"][name]):
            raise ValueError("frozen WM2 tensor changed in candidate: " + name)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "config", "checkpoint", "controllers", "candidate", "out", "name"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=None)
    args = parser.parse_args(argv)
    install_guard()
    if digest(args.checkpoint) != WM2_SHA:
        raise ValueError("base is not registered WM2")
    out = safe(args.out)
    target = out / f"PANEL_{args.name}.json"
    if target.exists():
        raise FileExistsError("panel receipt already exists: " + target.name)
    import torch
    torch.set_num_threads(2)
    if args.device == "cuda" and args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    from tools.wmfixmerge.pilot import panel
    from tools.wmfixmerge.runtime import Runtime
    rt = Runtime(args.prepared, args.config, args.checkpoint, args.controllers, args.device)
    payload = load_candidate(rt, args.checkpoint, args.candidate)
    step = payload["wmfix"]["step"]
    values, meta = panel(rt)
    with target.open("x") as stream:
        json.dump(dict(step=step, panel=values,
                       producer=dict(host=socket.gethostname(), device=str(rt.device),
                                     candidate_sha256=digest(args.candidate),
                                     base_sha256=WM2_SHA),
                       **meta), stream, indent=2, allow_nan=False)
    print(json.dumps(dict(name=args.name, step=step, seconds=meta["seconds"],
                          mae={h: values[str(h)]["paired_J_MAE"] for h in values},
                          wrote=str(target))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
