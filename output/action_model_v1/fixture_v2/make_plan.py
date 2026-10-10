"""Regenerate PLAN.json for the WM-43B draft from wm43_plan (stdlib only).

Usage (from the repo root): python output/action_model_v1/fixture_v2/make_plan.py
Refuses to overwrite a plan that differs unless --force is given.  Runs
nothing else; zero simulation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wm43_plan as P  # noqa: E402


def build() -> dict:
    return {
        "schema": P.PLAN_SCHEMA,
        "plan_id": "wm43b_wall_closure_v1",
        "plan_seed": 20260930,
        "created_utc": "2026-09-30T00:00:00+00:00",
        "status": "DRAFT - NOT AUTHORIZED - never launched",
        "purpose": ("Re-validation of the WM-40 per-CAV dispatcher wall groups on the "
                    "REAL TraCI/ownership path under the wall-v2.1 assembly closure law "
                    "(owner decision 2026-09-30, Option B) plus the new assembly_timeout "
                    "abort.  NOT efficacy evidence (short mechanics runs are not efficacy "
                    "evidence; no controller comparison is made)."),
        "proposal_source": "output/action_model_v1/fixture_v2/REGISTRATION_WM43B_DRAFT.md",
        "network": ("assets/research/highway_merge_shepherd_long/osm.net.xml.gz "
                   "(bytes untouched; digest verified at preflight)"),
        "arms": {
            "A1": "unchanged spec vs WM-43 r7: assemble -> hold -> release_at_s=90 (members 1-4 m apart); re-validation under the closure law",
            "A2": "unchanged spec vs WM-43 r7: wall + damping non-member follower, release_at_s=300",
            "A3": "unchanged spec vs WM-43 r7: induced member_span_exceeded abort (pin to 25 m/s from t=90 s)",
            "A4": "NEW wide-pair closure: members 40-60 m apart, assemble within 60 s (span <= 6 m), hold, release_at_s=150; front paced to v_wall, rear closure formula, zero collisions, native handback",
            "A5": "NEW assembly timeout: wide pair, scripted actuator pins the front member at 10.5 m/s from assembly start so closure is impossible; assembly_timeout abort 60 +/- 1 s after assembly_start, same-tick native handback",
        },
        "constants": {
            "horizon_ticks": P.HORIZON_TICKS,
            "control_dt_s": P.CONTROL_DT_S,
            "physics_dt_s": P.PHYSICS_DT_S,
            "wall_speed_mps": P.WALL_SPEED_MPS,
            "law_version": P.WALL_LAW_VERSION,
            "loading": P.LOADING,
            "penetration": P.PENETRATION,
            "role": P.ROLE,
            "seeds_by_arm": {arm: list(seeds) for arm, seeds in P.SEEDS_BY_ARM.items()},
            "roster_derivation_prefix": P.SEEDS_DERIVATION_PREFIX,
            "runs_root_relative": P.RUNS_ROOT_RELATIVE,
            "workers_max": P.WORKERS_MAX,
            "retries": P.RETRIES,
            "wall_clock_budget_s": P.WALL_CLOCK_BUDGET_S,
            "simulated_seconds_total": P.SIMULATED_SECONDS_TOTAL,
            "interpreter": P.INTERPRETER,
            "sumo": "SUMO 1.27.1 (PATH)",
        },
        "budget": {
            "runs_total": P.RUNS_TOTAL,
            "runs_per_arm": 5,
            "simulated_seconds": P.SIMULATED_SECONDS_TOTAL,
            "replay": "single replay each; zero retries; zero training",
            "acceptance_sealed_data": "none touched",
            "wall_clock_request": "<= 1 h (authorization bound); micro-roster runs expected to finish materially faster",
        },
        "runs": P.expected_runs(),
        "smoke": {
            "arm": P.SMOKE_ARM,
            "seed": P.SMOKE_SEED,
            "horizon_ticks": P.SMOKE_HORIZON_TICKS,
            "simulated_seconds": P.SMOKE_HORIZON_TICKS,
            "runs_root_relative": P.RUNS_ROOT_RELATIVE + P.SMOKE_RUNS_ROOT_SUFFIX,
            "used_as_data": False,
            "purpose": ("pre-launch smoke: one A4 (wide-pair closure) run to t=70 s "
                        "exercises scenario/vType load, dispatcher wall assembly start, "
                        "closure to assembled and telemetry rows through the REAL TraCI "
                        "path; never used as data; zero retries; verdict write-once bound "
                        "to the registration sha256 + package code digest"),
        },
    }


if __name__ == "__main__":
    text = json.dumps(build(), indent=2) + "\n"
    target = P.PLAN_PATH
    if target.exists() and target.read_text(encoding="utf-8") != text \
            and "--force" not in sys.argv:
        print(f"refusing to overwrite differing {target} (use --force)")
        sys.exit(1)
    target.write_text(text, encoding="utf-8")
    print(f"wrote {target}")
