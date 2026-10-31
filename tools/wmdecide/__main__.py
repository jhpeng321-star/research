"""python -B -m tools.wmdecide {pick,credibility} --help"""

import argparse
import json
from pathlib import Path

from .decision import Future, decide
from .rbeval_adapter import load_harness, observations, safe_path, sha256
from .report import (credibility_section, reliability_svg, summarize_rows, write_csv, write_json)


def credibility(eval_root, inputs, out, allow_partial=False):
    if not __debug__:
        raise RuntimeError("Do not disable evaluator assertions with Python -O")
    out = safe_path(out)
    if out.exists() or not out.parent.is_dir():
        raise ValueError("Output must be a NEW directory under an existing parent")
    inputs = [safe_path(p) for p in inputs]
    core, scorer = load_harness(eval_root)
    real = core.load(core.HERE / "real_observables.json")["episodes"]
    # The evaluator rechecks every source identity, K8 membership, BIC flag,
    # prehistory, support and delay ledger before any statistical calculation.
    pred, audit = scorer.read_predictions(inputs, real, allow_partial)
    data = observations(pred, real, core.ARMS, core.HS, core.REGIONS)
    summaries, diagrams = summarize_rows(data)
    model = audit["model"]["model"] if audit["model"] else "no_complete_forecasts"
    provenance = dict(schema="wmdecide/1", model=model, audit=audit,
        status="DIAGNOSTIC_ONLY" if pred else "BLOCKED_NO_COMPLETE_K8_WORLD",
        qualification="NOT_ASSESSED_USE_RBEVAL_GATE", bootstrap=10000, bootstrap_seed=20261031,
        finite_K_resampling=False, intervals="pointwise_stratified_whole_world_percentile_95",
        switch_confidence=.8, paired_probability="strict_delay_less_than",
        evaluator_sources={name: sha256(core.HERE / name) for name in
            ("core.py", "score.py", "references.json", "real_observables.json")},
        decision_sources={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    out.mkdir()
    write_json(out / "audit.json", provenance)
    for name, rows in data.items():
        write_csv(out / (name + ".csv"), rows)
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "reliability.csv", diagrams)
    write_json(out / "credibility.json", dict(provenance=provenance, summary=summaries,
                                              reliability=diagrams))
    for target in ("breakdown", "preference"):
        with (out / f"reliability_{target}.svg").open("x", encoding="utf-8") as f:
            f.write(reliability_svg(diagrams, target, model))
    with (out / "CREDIBILITY.md").open("x", encoding="utf-8") as f:
        f.write(credibility_section(model, audit, summaries))
    write_json(out / "FILES.json", {p.name: sha256(p) for p in sorted(out.iterdir())})
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pick = commands.add_parser("pick", help="Choose from JSON action→Future-list under fixed 0.8 rule")
    pick.add_argument("--input", required=True)
    pick.add_argument("--incumbent", required=True)
    pick.add_argument("--out", required=True, help="New JSON file under an existing parent")
    report = commands.add_parser("credibility", help="Score validated S-RB-EVAL K8 outputs")
    report.add_argument("--eval-root", required=True, help="Trusted local S-RB-EVAL worktree/runtime")
    report.add_argument("--inputs", nargs="+", required=True, help="One model's shard directories")
    report.add_argument("--out", required=True, help="New report directory under an existing parent")
    report.add_argument("--allow-partial", action="store_true", help="Complete K8 worlds only; diagnostic")
    args = parser.parse_args()
    if args.command == "pick":
        raw = json.loads(safe_path(args.input).read_text(encoding="utf-8"))
        actions = {action: [Future(**row) for row in rows] for action, rows in raw.items()}
        output = safe_path(args.out)
        if not output.parent.is_dir():
            raise ValueError("Output parent must exist")
        result = decide(actions, args.incumbent)
        write_json(output, result)
        print(json.dumps({"selected": result["selected"], "switched": result["switched"]}))
    else:
        result = credibility(args.eval_root, args.inputs, args.out, args.allow_partial)
        print(json.dumps({"status": result["status"], "model": result["model"],
                          "complete_worlds": len(result["audit"]["complete_worlds"])}))


if __name__ == "__main__":
    main()
