"""Small explicit CLI for the adopted CIRCLES baseline."""
import argparse
import json

from .workflow import evaluate, inspect_baseline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="verify model/configuration and singleton inference without SUMO")
    run = commands.add_parser("evaluate", help="one fixed, previously exposed development case")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--loading", type=int, required=True)
    run.add_argument("--penetration", type=float, required=True)
    run.add_argument("--controller", choices=("native", "damping", "pi-external-u"), required=True)
    run.add_argument("--desired-speed-mps", type=float)
    run.add_argument("--role", choices=("exposed", "development"), default="exposed")
    run.add_argument("--output", required=True)
    run.add_argument("--duration", type=int, help="shorter than 2400 s marks mechanics-only; no efficacy acceptance")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        if args.controller == "pi-external-u" and args.desired_speed_mps is None:
            parser.error("--desired-speed-mps is required for selected controller")
        if args.controller == "pi-external-u" and args.desired_speed_mps != 31.29:
            parser.error("pi-external-u requires --desired-speed-mps 31.29")
        if args.controller != "pi-external-u" and args.desired_speed_mps is not None:
            parser.error("--desired-speed-mps is only valid for pi-external-u")
    result = inspect_baseline() if args.command == "inspect" else evaluate(
        seed=args.seed, loading=args.loading, penetration=args.penetration,
        controller_name=args.controller, output=args.output, role=args.role, duration=args.duration,
        desired_speed_mps=args.desired_speed_mps)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
