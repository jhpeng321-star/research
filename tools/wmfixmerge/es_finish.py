"""Complete scoring bindings for the already-running, immutable ES run.

The live driver's pre-fix export serialized PhysicsConfig into PhysicsOptions
JSON. Preserve that legacy artifact and create schema-correct scoring configs
that load physics independently from each immutable checkpoint. No rollout,
checkpoint mutation, process signalling or verdict-threshold changes.
"""
import argparse
from dataclasses import asdict
import json
import time


def finish(root):
    from tools.wmfixmerge.data import digest, safe
    from tools.wmfixmerge.es_ops import COMMON
    from tools.wmcombined.e3_score import load_for_scoring
    root = safe(root)
    if not (root / "FINAL.json").is_file():
        return False
    final = json.loads((root / "FINAL.json").read_text())
    if digest(final["checkpoint"]) != final["sha256"]:
        raise ValueError("selected immutable checkpoint changed")
    cfg, _, _ = load_for_scoring(COMMON[1], COMMON[3])
    resolved = asdict(cfg)
    resolved["model"]["merge_response"] = True
    resolved = dict(config=resolved, checkpoint_physics=True)
    gen = int(final["best"]["gen"])
    scoring = root / f"generation_{gen:03d}_scoring_config.json"
    if scoring.is_file():
        if json.loads(scoring.read_text()) != resolved:
            raise ValueError("immutable scoring config differs")
    else:
        with scoring.open("x") as stream:
            json.dump(resolved, stream, indent=2, allow_nan=False)
    target = root / "candidate_config.json"
    # The driver writes FINAL just before candidate_config; never race it.
    if not target.is_file():
        return False
    current = json.loads(target.read_text())
    if current != resolved:
        original = root / f"generation_{gen:03d}_config.json"
        if current != json.loads(original.read_text()):
            raise ValueError("unrecognized candidate config; preserving it")
        legacy = root / "candidate_config_legacy.json"
        if legacy.exists():
            raise ValueError("legacy config already present; reconcile before retry")
        target.rename(legacy)
        with target.open("x") as stream:
            json.dump(resolved, stream, indent=2, allow_nan=False)
    # Real scoring reload, not just canonical checkpoint tensor reload.
    _, _, candidate = load_for_scoring(target, final["checkpoint"])
    from tools.wmfixmerge.es import PARAMS
    for param in PARAMS:
        if param["where"] == "physics":
            if getattr(candidate.physics, param["name"]) != final["best"]["params"][param["name"]]:
                raise ValueError("scoring lost optimised physics: " + param["name"])
    receipt = root / "SCORING_READY.json"
    if not receipt.exists():
        with receipt.open("x") as stream:
            json.dump(dict(candidate_sha256=final["sha256"], config_sha256=digest(target),
                           checkpoint_physics=True, frozen_checkpoint_preserved=True,
                           canonical_scoring_reload=True), stream, indent=2)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=21600.)
    args = parser.parse_args()
    from tools.wmfixmerge.data import install_guard, safe
    install_guard()
    root = safe(args.root)
    if args.watch:
        import os
        with (root / "FINISH_WATCH_PID.json").open("x") as stream:
            json.dump(dict(pid=os.getpid(), started=time.time()), stream)
    began = time.monotonic()
    while True:
        if finish(root):
            import subprocess
            import sys
            final = json.loads((root / "FINAL.json").read_text())
            target = root / "VERDICT_PREFLIGHT_FINAL.json"
            if not target.exists():
                subprocess.run([sys.executable, "-m", "tools.wmfixmerge.es_verdict_preflight",
                    "--candidate", final["checkpoint"], "--config", str(root / "candidate_config.json"),
                    "--out", str(target)], check=True)
            print("SCORING_READY", flush=True)
            return
        if (root / "ES_ABORT.json").is_file():
            print("ES_ABORT; no scoring/adoption claim", flush=True)
            return
        if not args.watch:
            return
        if time.monotonic() - began >= args.max_seconds:
            print("finish watcher wall cap; no scoring/adoption claim", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
