"""Freeze the registered verdict with private ES artifact paths; no dispatch.

Only B producer source paths/hashes and its candidate output directory change.
All cohorts, metrics, thresholds, base quartet, C1 and accept bindings are kept.
"""
import argparse
import copy
import json
from pathlib import Path


def isolated_plan(plan, tree, root, digest):
    result = copy.deepcopy(plan)
    producer = result["stages"]["B"]["producer"]
    names = ("producer_b.py", "quartet_candidate.py")
    for name in names:
        old = [path for path in producer["files"] if Path(path).name == name]
        if len(old) != 1:
            raise ValueError("registered B producer closure is ambiguous: " + name)
        new = str(tree / "tools/wmfixmerge" / name)
        del producer["files"][old[0]]
        producer["files"][new] = digest(new)
        if name == "producer_b.py":
            producer["script"] = new
    argv = producer["args"]
    argv[argv.index("--candidate-dir") + 1] = str(root / "quartet_candidate")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--plan", default="tools/wmfixmerge/verdict_plan.json")
    args = parser.parse_args()
    from tools.wmfixmerge.data import digest, install_guard, safe
    install_guard()
    root = safe(args.root)
    if not root.is_dir():
        raise ValueError("existing ES run root required")
    tree = Path(__file__).resolve().parents[2]
    plan = json.loads(safe(args.plan).read_text())
    result = isolated_plan(plan, tree, root, digest)
    target = root / "ES_VERDICT_PLAN.json"
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    with (root / "ES_VERDICT_PLAN_BINDING.json").open("x") as stream:
        json.dump(dict(schema="wmfix_es_verdict_plan_binding/1",
                       source_plan_sha256=digest(args.plan), plan_sha256=digest(target),
                       amendments=["private B output and producer closure",
                                   "candidate/config/reference identity checks"],
                       metric_threshold_changes=False, dispatched=False), stream, indent=2)
    print(str(target))


if __name__ == "__main__":
    main()
