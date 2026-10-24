"""Round-13F smoke verdict: every round-12 check, then the round-13F switches in each reloaded checkpoint.

Round 13F = round 12 + repair A (``model.follower_courtesy_state_grad``, ``model.follower_courtesy_intent =
'pivot'``, ``loss.follower_courtesy.absolute_weight``) + ``model.temporal_accel_off`` (R13T_DESIGN.md).  The
round-12 verdict (:mod:`wms.train.round12_gate`) runs unchanged on the round-13F config and writes its report to
``<report>.r12.json``; this module then reloads ``wm_s_stage120.pt``, ``wm_s_stage300.pt`` and ``wm_s.pt``
strictly and requires the model spec to carry ``temporal_accel_off = True``, ``follower_courtesy_state_grad =
True`` and ``follower_courtesy_intent = 'pivot'``, and the saved legacy config to carry the config's
``follower_courtesy_absolute_weight`` (> 0).  The combined result goes to ``--report`` (exclusive create).
Exit 0 = pass, 3 = fail.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import round12_gate

CHECKPOINTS = ("wm_s_stage120.pt", "wm_s_stage300.pt", "wm_s.pt")


def check_r13f_checkpoints(smoke_dir: Path, absolute_weight: float) -> list[dict]:
    from wms.model.checkpoint import load_checkpoint

    if not absolute_weight > 0:
        raise ValueError(f"config absolute_weight must be > 0, got {absolute_weight!r}")
    out = []
    for name in CHECKPOINTS:
        loaded = load_checkpoint(Path(smoke_dir) / name, device="cpu", strict=True)
        spec, model = loaded.model.spec, loaded.model
        saved_abs = loaded.legacy_config.get("follower_courtesy_absolute_weight")
        problems = []
        if spec.temporal_accel_off is not True or getattr(model, "temporal_accel_off", None) is not True:
            problems.append("temporal_accel_off is not on after reload")
        if spec.follower_courtesy_state_grad is not True:
            problems.append("follower_courtesy_state_grad is not on")
        if spec.follower_courtesy_intent != "pivot":
            problems.append(f"follower_courtesy_intent is {spec.follower_courtesy_intent!r}, not 'pivot'")
        if saved_abs is None or float(saved_abs) != float(absolute_weight):
            problems.append(f"saved absolute_weight {saved_abs!r} != config {absolute_weight!r}")
        if problems:
            raise ValueError(f"{name}: " + "; ".join(problems))
        out.append({"file": name, "temporal_accel_off": True, "follower_courtesy_state_grad": True,
                    "follower_courtesy_intent": "pivot", "absolute_weight": float(saved_abs)})
    return out


def _arg(argv: list[str], flag: str) -> str:
    if flag not in argv or argv.index(flag) + 1 >= len(argv):
        raise SystemExit(f"{flag} is required")
    return argv[argv.index(flag) + 1]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    report = Path(_arg(argv, "--report"))
    config = Path(_arg(argv, "--config"))
    smoke_dir = Path(_arg(argv, "--smoke-dir"))
    r12_report = report.with_name(report.name + ".r12.json")
    r12_argv = list(argv)
    r12_argv[r12_argv.index("--report") + 1] = str(r12_report)
    code = round12_gate.main(r12_argv)
    try:
        result = json.loads(r12_report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"smoke: round-12 verdict report unreadable: {exc}")
        return 3
    result["round12_verdict_exit"] = code
    if code == 0 and result.get("ok"):
        try:
            from wms.config.load import load_train_config

            weight = float(load_train_config(config).loss.follower_courtesy.absolute_weight)
            result["r13f_checkpoints"] = check_r13f_checkpoints(smoke_dir, weight)
        except Exception as exc:
            result["ok"] = False
            result.setdefault("errors", []).append(f"round-13F switch check failed: {exc}")
    else:
        result["ok"] = False
    try:
        with report.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError) as exc:
        print(f"smoke: bad report: {exc}")
        return 3
    print("smoke: ok (round 13F)" if result["ok"] else "smoke: bad " + "; ".join(result.get("errors", [])))
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
