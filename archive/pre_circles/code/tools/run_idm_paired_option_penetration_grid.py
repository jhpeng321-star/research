"""Resumable first-risk-onset P30 grid across CAV penetration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from experiments.io import atomic_write_json


SCHEMA = "idm_paired_option_penetration_grid/1"
DEFAULT_PENETRATIONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def _float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def _int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def _slug(penetration: float) -> str:
    return f"p{int(round(100 * penetration)):02d}"


def cell_output(root: Path, demand: int, penetration: float, decision_index: int) -> Path:
    return root / (
        f"idm_paired_option_grid_{_slug(penetration)}_m{demand}_l{decision_index}_v1"
    )


def cell_command(
    *,
    python: Path,
    repo: Path,
    output: Path,
    demand: int,
    penetration: float,
    decision_index: int,
    seeds: list[int],
    workers: int,
    reward_refs: Path,
    evidence_role: str = "development",
) -> list[str]:
    command = [
        str(python),
        str(repo / "tools" / "run_idm_decentralized_authority_verification.py"),
        "--output",
        str(output),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--demand",
        str(demand),
        "--workers",
        str(workers),
        "--action-set",
        "activation_speed_ceiling_v1",
        "--option-steps",
        "20",
        "--safety-horizon-steps",
        "60",
        "--cav-penetration",
        f"{penetration:.2f}",
        "--decision-index",
        str(decision_index),
        "--maximum-landmarks-per-cell",
        "3",
        "--reward-refs",
        str(reward_refs),
        "--candidate-ranking",
        "runway",
        "--candidate-count",
        "1",
        "--landmark",
        "risk_onset",
    ]
    if evidence_role != "development":
        command.extend(["--evidence-role", evidence_role])
    return command


def _valid_complete(
    output: Path,
    demand: int,
    penetration: float,
    seeds: list[int],
    evidence_role: str = "development",
) -> bool:
    summary_path = output / "summary.json"
    progress_path = output / "progress.json"
    if not summary_path.exists() or not progress_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    protocol = summary.get("protocol") or {}
    role_matches = (
        protocol.get("evidence_role") == evidence_role
        if evidence_role != "development"
        else bool(protocol.get("development_only", False))
    )
    frozen_matches = True
    if evidence_role == "damping_acceptance":
        frozen_matches = bool(
            protocol.get("action_set") == "activation_speed_ceiling_v1"
            and protocol.get("terminal_reference_percentile") == "p30"
            and int(protocol.get("option_steps", -1)) == 20
            and int(protocol.get("minimum_dose_steps", -1)) == 16
            and int(protocol.get("decision_index", -1)) == 3
            and int(protocol.get("maximum_landmarks_per_cell", -1)) == 3
            and protocol.get("candidate_ranking") == "runway"
            and int(protocol.get("candidate_count", -1)) == 1
            and protocol.get("landmark") == "risk_onset"
            and int(protocol.get("safety_horizon_steps", -1)) == 60
            and int(protocol.get("service_horizon_steps", -1)) == 60
            and protocol.get("reward_refs_sha256")
            == "737b9ce4bb1945f7681b807941c86509ae3143128bd0729d2cb2aee5237c5b58"
            and protocol.get("decision_source_path") is None
        )
    return bool(
        int(protocol.get("demand", -1)) == demand
        and abs(float(protocol.get("cav_penetration", -1.0)) - penetration) <= 1e-12
        and sorted(int(seed) for seed in protocol.get("seeds", ())) == sorted(seeds)
        and role_matches
        and frozen_matches
        and int(progress.get("pending", -1)) == 0
        and int(progress.get("complete", -1)) == int(summary.get("branch_count", -2))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=_int_list, required=True)
    parser.add_argument("--demands", type=_int_list, default=[1900, 2000, 2100])
    parser.add_argument("--penetrations", type=_float_list, default=list(DEFAULT_PENETRATIONS))
    parser.add_argument("--decision-index", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--workers", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--reward-refs", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--evidence-role",
        choices=("development", "damping_acceptance"),
        default="development",
    )
    parser.add_argument(
        "--reuse-10pct-root",
        type=Path,
        default=None,
        help=(
            "Optional existing directory prefix. For each demand, the driver reuses "
            "<prefix>_mDEMAND_lINDEX_v1 after strict protocol checks."
        ),
    )
    args = parser.parse_args()
    if args.evidence_role == "damping_acceptance" and args.reuse_10pct_root is not None:
        raise ValueError("prospective damping acceptance prohibits artifact reuse")
    repo = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = sorted(set(args.seeds))
    demands = sorted(set(args.demands))
    penetrations = sorted(set(args.penetrations))
    cells = [
        (demand, penetration)
        for penetration in penetrations
        for demand in demands
    ]
    started = time.time()
    completed: list[dict] = []
    for index, (demand, penetration) in enumerate(cells, 1):
        if abs(penetration - 0.10) <= 1e-12 and args.reuse_10pct_root is not None:
            output = Path(
                f"{args.reuse_10pct_root.resolve()}_m{demand}_l{args.decision_index}_v1"
            )
            source = "reused"
        else:
            output = cell_output(output_root, demand, penetration, args.decision_index)
            source = "generated"
        if _valid_complete(output, demand, penetration, seeds, args.evidence_role):
            status = "complete"
        else:
            command = cell_command(
                python=args.python.resolve(),
                repo=repo,
                output=output,
                demand=demand,
                penetration=penetration,
                decision_index=args.decision_index,
                seeds=seeds,
                workers=args.workers,
                reward_refs=args.reward_refs.resolve(),
                evidence_role=args.evidence_role,
            )
            print(
                f"[{index}/{len(cells)}] m{demand} {_slug(penetration)} -> {output}",
                flush=True,
            )
            subprocess.run(command, cwd=repo, check=True)
            if not _valid_complete(
                output, demand, penetration, seeds, args.evidence_role
            ):
                raise RuntimeError(f"cell completed without a valid summary: {output}")
            status = "complete"
        completed.append(
            {
                "demand": demand,
                "penetration": penetration,
                "decision_index": args.decision_index,
                "output": str(output.resolve()),
                "source": source,
                "status": status,
            }
        )
        atomic_write_json(
            output_root / "grid_progress.json",
            {
                "schema": SCHEMA,
                "completed_cells": len(completed),
                "total_cells": len(cells),
                "elapsed_s": time.time() - started,
                "cells": completed,
            },
        )
    atomic_write_json(
        output_root / "grid_summary.json",
        {
            "schema": SCHEMA,
            "status": "complete",
            "development_only": args.evidence_role == "development",
            "evidence_role": args.evidence_role,
            "seeds": seeds,
            "demands": demands,
            "penetrations": penetrations,
            "decision_index": args.decision_index,
            "elapsed_s": time.time() - started,
            "cells": completed,
        },
    )


if __name__ == "__main__":
    main()
