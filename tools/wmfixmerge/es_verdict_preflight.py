"""Read-only preflight of the unchanged registered A/B/C1/accept verdict."""
import argparse
import json
from pathlib import Path


def selected_plan(candidate, explicit=None):
    if explicit:
        return Path(explicit)
    private = Path(candidate).parent / "ES_VERDICT_PLAN.json"
    return private if private.is_file() else Path("tools/wmfixmerge/verdict_plan.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", help="defaults to the candidate's frozen ES plan if present")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from tools.wmfixmerge.data import digest, install_guard, safe
    from tools.wmfixmerge.verdict import preflight
    install_guard()
    plan_path = safe(selected_plan(args.candidate, args.plan))
    plan = json.loads(plan_path.read_text())
    blockers = preflight(plan, args.candidate, args.config)
    required = {"accept_panel": plan.get("accept_panel")}
    for stage, flags in (("C1", ("--cohort", "--runner")), ("B", ("--candidate-dir",))):
        argv = plan["stages"][stage]["producer"]["args"]
        for flag in flags:
            path = argv[argv.index(flag) + 1]
            if flag == "--candidate-dir":
                path = str(safe(path) / "COMPLETE.json")
            required[stage + flag] = path
    checks = {key: dict(path=value, exists=bool(value and safe(value).is_file()))
              for key, value in required.items()}
    blockers += ["missing " + key for key, value in checks.items() if not value["exists"]]
    blockers = list(dict.fromkeys(blockers))
    doc = dict(schema="wmfix_es_verdict_preflight/1", status="BLOCKED" if blockers else "READY",
               passed=False, executed_stages=[], blockers=blockers, checks=checks,
               candidate_sha256=digest(args.candidate), config_sha256=digest(args.config),
               plan_sha256=digest(plan_path), evaluation_only=True)
    with safe(args.out).open("x") as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)
    print(json.dumps(doc, indent=2), flush=True)


if __name__ == "__main__":
    main()
