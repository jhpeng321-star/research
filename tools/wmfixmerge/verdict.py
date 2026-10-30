"""One-command A/B/C verdict. Dry-run by default; missing adapters block all scoring."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from tools.wmfixmerge.data import WM2_SHA, digest, install_guard, safe


def preflight(plan, candidate, config):
    blockers = []
    if plan.get("schema") != "wmfix_verdict_plan/1" or set(plan.get("stages", {})) != {"A", "B", "C1", "SAFETY"}:
        return ["invalid/incomplete A+B+C1+SAFETY plan"]
    for label, path in {"candidate": candidate, "candidate_config": config,
            **{k: plan.get(k) for k in ("reference", "base_config", "controllers", "corrected_report",
                "harness_contract", "validation_real", "zero_reference")}}.items():
        try:
            if not path or not safe(path).is_file():
                blockers.append("missing " + label)
        except ValueError:
            blockers.append("restricted " + label)
    if not plan.get("accept_roots"):
        blockers.append("canonical accept.py fixed TRAIN roots/panel not bound")
    for stage in ("B", "C1"):
        spec = plan["stages"][stage]
        producer = spec.get("producer") or {}
        if spec.get("adapter") != "bound_external" or not producer.get("script") or not producer.get("files"):
            blockers.append(stage + " exact frozen producer/receipt binding missing")
            continue
        if str(producer["script"]) not in producer["files"] or not producer.get("args"):
            blockers.append(stage + " producer entrypoint/arguments not hash-bound")
        for path, sha in producer["files"].items():
            if not safe(path).is_file() or digest(path) != sha:
                blockers.append(stage + " producer closure missing/changed: " + str(path))
        if spec.get("receipt") and (not spec.get("sha256") or not safe(spec["receipt"]).is_file()
                                    or digest(spec["receipt"]) != spec["sha256"]):
            blockers.append(stage + " existing receipt missing/changed")
    if not blockers:
        if digest(plan["reference"]) != WM2_SHA:
            blockers.append("reference SHA is not WM2")
        contract = json.loads(safe(plan["harness_contract"]).read_text())
        if contract.get("corrected_report_sha256") != digest(plan["corrected_report"]) \
                or contract.get("source_clock_match") is not True or contract.get("residual_remains") is not True \
                or contract.get("scorer_sha256") != digest("tools/wmrequal/score.py") \
                or contract.get("native_lanes_sha256") != digest("tools/wmrequal/native_lanes.py"):
            blockers.append("corrected-harness clock/retarget contract not PASS")
    return blockers


def bound_panel(spec, binding, *, stage, out, candidate, config, reference, device):
    if spec.get("receipt"):
        if digest(spec["receipt"]) != spec["sha256"]:
            raise ValueError("external panel changed after preflight")
        path = safe(spec["receipt"])
    else:
        producer = spec["producer"]
        for file, sha in producer["files"].items():
            if digest(file) != sha:
                raise ValueError("external producer changed after preflight")
        path = safe(out / (stage + ".json"))
        values = dict(candidate=str(candidate), config=str(config), reference=str(reference),
                      out=str(path), device=device, binding=str(out / "BINDING.json"))
        argv = [sys.executable, "-m", "tools.wmfixmerge.producer_runner", producer["script"],
                *[str(token).format(**values) for token in producer["args"]]]
        with (out / (stage + ".log")).open("x") as log:
            completed = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False,
                                       timeout=producer.get("timeout_seconds", 7200))
        if completed.returncode != 0:
            raise ValueError(stage + " producer failed; retained log, no automatic retry")
    doc = json.loads(path.read_text())
    if doc.get("binding") != binding:
        raise ValueError("external panel candidate/reference/config/plan binding mismatch")
    if doc.get("producer") != spec["producer"]:
        raise ValueError("external panel is not from frozen registered producer")
    return doc


def run(plan, candidate, config, out, execute=False, device="cpu"):
    from tools.wmfixmerge import verdict_metrics as metrics
    out = safe(out)
    if not out.is_dir():
        raise ValueError("existing output directory required")
    blockers = preflight(plan, candidate, config)
    result = dict(schema="wmfix_verdict/1", status="BLOCKED" if blockers else "DRY_RUN", passed=False,
                  blockers=blockers, executed_stages=[], panels={})
    # Even a stale/interrupted run is never restarted automatically. No PID is signalled.
    if (out / "RUN.json").exists():
        result["status"] = "BLOCKED"
        result["blockers"].append("existing RUN receipt; running/interrupted dispatch must be reconciled")
    if not execute or result["blockers"]:
        return result
    binding = dict(candidate_sha256=digest(candidate), reference_sha256=digest(plan["reference"]),
                   config_sha256=digest(config), corrected_report_sha256=digest(plan["corrected_report"]))
    with (out / "RUN.json").open("x") as stream:
        json.dump(dict(pid=os.getpid(), binding=binding, device=device), stream, indent=2)
    with (out / "BINDING.json").open("x") as stream:
        json.dump(binding, stream, indent=2)
    from tools.wmfixmerge import verdict_panels as panels
    try:
        a = panels.validation(plan, candidate, config, out, device)
        a["binding"] = binding
        result["panels"]["A"] = metrics.validation(a)
        result["executed_stages"].append("A")
        with (out / "A.json").open("x") as stream:
            json.dump(a, stream, allow_nan=False)
        for stage, compare in (("B", metrics.speedhead), ("C1", metrics.c1)):
            doc = bound_panel(plan["stages"][stage], binding, stage=stage, out=out,
                              candidate=candidate, config=config, reference=plan["reference"], device=device)
            result["panels"][stage] = compare(doc)
            result["executed_stages"].append(stage)
        accept = panels.safety(plan, candidate, config, out)
        result["panels"]["SAFETY"] = dict(passed=accept["decision"] == "ACCEPT", receipt=accept)
        result["executed_stages"].append("SAFETY")
        result["passed"] = all(p["passed"] for p in result["panels"].values()) and len(result["panels"]) == 4
        result["status"] = "PASS" if result["passed"] else "REJECT"
    except (OSError, ValueError, RuntimeError, AssertionError, KeyError, ImportError, subprocess.TimeoutExpired) as exc:
        result.update(status="BLOCKED", passed=False)
        result["blockers"].append(f"{type(exc).__name__}: {exc}")
    result["binding"] = binding
    with (out / "VERDICT.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="tools/wmfixmerge/verdict_plan.json")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--receipt-out", help="optional immutable dry-run/blocker receipt")
    args = parser.parse_args(argv)
    install_guard()
    plan = json.loads(safe(args.plan).read_text())
    root = safe(args.train_dir)
    try:
        final = json.loads(safe(root / "FINAL.json").read_text())
        if final.get("step", 0) < 1 or final.get("status") != "COMPLETE" or digest(final["checkpoint"]) != final["sha256"]:
            raise ValueError("no immutable trained final checkpoint")
        result = run(plan, final["checkpoint"], root / "candidate_config.json", args.out, args.execute, args.device)
    except (OSError, ValueError, KeyError) as exc:
        result = dict(schema="wmfix_verdict/1", status="BLOCKED", passed=False, executed_stages=[],
                      blockers=[f"{type(exc).__name__}: {exc}"] +
                        preflight(plan, root / "missing_final.pt", root / "candidate_config.json"), panels={})
    if args.receipt_out:
        with safe(args.receipt_out).open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
