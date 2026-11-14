"""INC1 R4 adapter: R4 through the frozen v2.2 driver + the registered hook.

REGISTRATION_WM_N1_INC1_ADDENDUM_1 A1.2/A1.3.  R4 drives the patched frozen
driver ``walls_design/merge_release/copies/post_inc1/wall_driver.py`` (the
frozen v2.2 post + ``patches/wall_driver_inc1_hook.diff``) with:

* campaign modules imported READ-ONLY from ``E:/research-dense/output``
  (``campaign_pilot_v1`` / ``campaign_stage1_v3``; never written);
* the ``MixedAssignmentSession`` dispatcher;
* the INC1 stopped obstacle inserted by the new ``post_restore_hook``;
* ``branch_length_s_override = 300`` and the J4 mid speed factor 0.85
  (arm name ``W-mid-early``), with the wall request at the incident t0;
* the restore/world/arm dicts synthesized from the INC1 warm-up receipt.

Infeasibility (A1.3): a proposer refusal (``not_feasible``) or an assembly
that does not complete (``aborted_in_assembly``) is logged with the driver's
own reason, the run completes natively and is FLAGGED infeasible.  R1 with no
qualifying CAV is likewise flagged infeasible (see ``schedules``).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

try:
    from . import constants as C
except ImportError:  # script execution
    import constants as C  # type: ignore

#: the registered J4 mid speed factor (W-mid); the arm name encodes it.
R4_SPEED_FACTOR = C.R4_SPEED_FACTOR
#: the arm name parsed by ``campaign_walls_v1.common`` (prefix W-mid, timing
#: early) - the factor is read from THIS token, never a loose field.
R4_ARM_NAME = "W-mid-early"
#: the branch horizon override (the INC1 300 s arm).
R4_BRANCH_LENGTH_S = C.ARM_HORIZON_S
#: the registered fixed-arm hold draw used when the caller passes none
#: (H_min = 60 s, the smallest registered draw; releases well inside 300 s).
R4_HOLD_S = 60.0

#: outcomes that mean the wall assembled (assembly completed per the driver).
FEASIBLE_WALL_OUTCOMES = frozenset({
    "assembled_released", "merge_released", "aborted_in_hold",
})

SHIM_DIR = Path(__file__).resolve().parent / "r4shim"
LIVE_OUTPUT = Path("E:/research-dense/output")   # READ-ONLY dependency tree
POST_WALL_GROUP = (C.WD01_ROOT / "walls_design" / "merge_release" / "copies"
                   / "post" / "wall_group.py")


class R4DriverError(RuntimeError):
    """An R4 adapter contract violation; never silently swallowed."""


# --------------------------------------------------------------------- shims
def install_shims() -> dict:
    """Make the PATCHED wall law importable (read-only) for this process.

    * ``r4shim`` is prepended to ``sys.path`` so ``campaign_walls_v1``
      resolves ``wall_driver`` from ``copies/post_inc1`` and the rest from
      ``copies/post``;
    * ``E:/research-dense/output`` is appended for ``campaign_pilot_v1`` /
      ``campaign_stage1_v3``;
    * the patched v2.2 ``traffic_shepherd.wall_group`` is injected into
      ``sys.modules`` so ``mixed_assignment`` binds the v2.2 law class.

    Idempotent; never writes to any copy or the live tree.
    """
    if str(SHIM_DIR) not in sys.path:
        sys.path.insert(0, str(SHIM_DIR))
    if str(LIVE_OUTPUT) not in sys.path:
        sys.path.append(str(LIVE_OUTPUT))
    injected = False
    existing = sys.modules.get("traffic_shepherd.wall_group")
    existing_file = (getattr(existing, "__file__", None) or "").replace(
        "\\", "/")
    post_norm = str(POST_WALL_GROUP).replace("\\", "/")
    if post_norm not in existing_file:
        spec = importlib.util.spec_from_file_location(
            "traffic_shepherd.wall_group", str(POST_WALL_GROUP))
        module = importlib.util.module_from_spec(spec)
        sys.modules["traffic_shepherd.wall_group"] = module
        spec.loader.exec_module(module)
        injected = True
    # ``mixed_assignment`` binds ``WallGroupSession`` at import time; if it was
    # imported before the v2.2 wall_group injection, re-bind it.
    wg = sys.modules["traffic_shepherd.wall_group"]
    mixed = sys.modules.get("traffic_shepherd.mixed_assignment")
    if mixed is not None and getattr(mixed, "WallGroupSession", None) is not \
            getattr(wg, "WallGroupSession", None):
        sys.modules.pop("traffic_shepherd.mixed_assignment", None)
    return {"shim_dir": str(SHIM_DIR), "live_output": str(LIVE_OUTPUT),
            "wall_group_injected": injected,
            "wall_group_file": str(POST_WALL_GROUP)}


def load_patched_wall_driver():
    """Import the patched frozen driver; assert it is the post_inc1 copy."""
    install_shims()
    import campaign_walls_v1.wall_driver as driver  # noqa: PLC0415

    path = str(getattr(driver, "__file__", ""))
    if "post_inc1" not in path.replace("\\", "/"):
        raise R4DriverError(
            "campaign_walls_v1.wall_driver did not resolve to the INC1 hook "
            f"copy (got {path!r}); refusing to run an unpatched R4")
    return driver


def load_patched_wall_request():
    install_shims()
    import campaign_walls_v1.wall_request as wall_request  # noqa: PLC0415

    return wall_request


def build_r4_stack(api, cav_ids, *, config=None, cav_speed_mode=None):
    """Build the frozen dispatcher stack (MixedAssignmentSession + ownership).

    Read-only import of the frozen ``traffic_shepherd`` control stack; the
    patched v2.2 ``wall_group`` is bound first by :func:`install_shims`.
    Returns ``(session, ownership, sensor)``.
    """
    install_shims()
    from traffic_shepherd.controller_factory import create_controller  # noqa: PLC0415
    from traffic_shepherd.mixed_assignment import (  # noqa: PLC0415
        MixedAssignmentSession,
    )
    from traffic_shepherd.ownership import CommandOwnership  # noqa: PLC0415
    from traffic_shepherd.sensing import SumoLocalSensor  # noqa: PLC0415

    if config is None:
        from traffic_shepherd.config import load_config  # noqa: PLC0415
        config = load_config()
    controller = create_controller("damping", config)
    ownership = CommandOwnership(
        api, cav_ids, controller,
        cav_speed_mode=(C.CAV_SPEED_MODE if cav_speed_mode is None
                        else int(cav_speed_mode)))
    session = MixedAssignmentSession(
        circles_controller=ownership.controller, pi_session=None,
        cav_ids=sorted(str(v) for v in cav_ids))
    sensor = SumoLocalSensor(api, config)
    return session, ownership, sensor


# --------------------------------------------------------- manifest synthesis
def build_restore(warmup_receipt: dict, *, cav_ids, boundary_edges=None,
                  physics_row_at_t0=None) -> dict:
    """Synthesize the campaign ``restore`` manifest from an INC1 warm-up receipt.

    The driver reads ``manifest.harness_state.roster.cav_ids`` /
    ``boundary_edges``, the optional ``physics_row_at_t0`` and
    ``saved_state_id``; those are the only campaign manifest fields it needs
    (see INC1_R4_BLOCKER.md section 2.2).
    """
    digest = warmup_receipt.get("saved_state_digest") if warmup_receipt else None
    saved_state_id = str(digest) if digest else hashlib.sha256(
        json.dumps(sorted(str(v) for v in cav_ids)).encode("utf-8")
    ).hexdigest()
    return {
        "manifest": {
            "harness_state": {
                "roster": {
                    "cav_ids": sorted(str(v) for v in cav_ids),
                    "boundary_edges": dict(boundary_edges or {}),
                },
            },
            "physics_row_at_t0": physics_row_at_t0,
        },
        "saved_state_id": saved_state_id,
    }


def build_world(world: dict) -> dict:
    """The campaign ``world`` dict (branch length overridden at call time)."""
    return {"world": str(world["world"]), "seed": int(world["seed"]),
            "branch_length_s": float(R4_BRANCH_LENGTH_S)}


def build_arm(world: dict, *, t0_s: float = C.T0_S,
              hold_s: float = R4_HOLD_S) -> dict:
    """The campaign ``arm`` dict for R4 (W-mid early, request at t0)."""
    run_name = f"{world['world']}__r4"
    return {
        "arm": R4_ARM_NAME,
        "run": run_name,
        "mode": "restore_wall",
        "native_sibling_run": f"{world['world']}__r0",
        "branch_seed": int(world.get("seed", 0)),
        "restores_state": {"from_run": world["warmup_run"], "at_s": float(t0_s)},
        "wall": {"speed_factor": float(R4_SPEED_FACTOR), "timing": "early",
                 "target_time_s": float(t0_s), "request_window_s":
                 C.R4_REQUEST_WINDOW_S,
                 "deadline_s": float(t0_s) + C.R4_REQUEST_WINDOW_S,
                 "speed_floor_mps": 2.5, "hold_s": float(hold_s)},
    }


# ------------------------------------------------------------ infeasibility
def branch_events(run_dir) -> list:
    path = Path(run_dir) / "wall_events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def classify_branch(summary: dict, run_dir) -> tuple[bool, str | None]:
    """Return ``(feasible, infeasible_reason)`` per A1.3.

    Feasible iff the wall assembled (``assembled_at_s`` recorded) or the
    outcome is one of the assembled outcomes.  A proposer refusal
    (``not_feasible``) or an aborted assembly is infeasible, carrying the
    driver's own logged reason.
    """
    outcome = summary.get("wall_outcome")
    assembled = summary.get("assembled_at_s") is not None
    if assembled or outcome in FEASIBLE_WALL_OUTCOMES:
        return True, None
    if outcome == "not_feasible":
        rows = [e for e in branch_events(run_dir)
                if e.get("kind") == "not_feasible"]
        reason = rows[0].get("reason") if rows else "not_feasible"
        return False, f"proposer_refusal:{reason}"
    if outcome == "aborted_in_assembly":
        return False, f"assembly_not_completed:{summary.get('abort_reason')}"
    return False, f"infeasible_outcome:{outcome}"


def write_result(run_dir, result: dict) -> Path:
    path = Path(run_dir) / "r4_result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


# ------------------------------------------------------------------- driver
def run_r4_branch(api, world: dict, run_dir, warmup_receipt: dict, *,
                  session, ownership, sensor, cav_ids=None,
                  obstacle_hook=None, hold_s: float = R4_HOLD_S,
                  t0_s: float = C.T0_S, boundary_edges=None,
                  physics_row_at_t0=None, write: bool = True) -> dict:
    """Run one R4 branch through the patched frozen driver.

    ``obstacle_hook(api, t_s) -> dict`` inserts the INC1 stopped obstacle; its
    return value is logged by the driver in the branch record.  The caller
    injects the dispatcher ``session`` / ``ownership`` / ``sensor`` (real on
    the box, fakes in the tests).
    """
    driver = load_patched_wall_driver()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if cav_ids is None:
        cav_ids = list(api.vehicle.getIDList())
    restore = build_restore(warmup_receipt, cav_ids=cav_ids,
                            boundary_edges=boundary_edges,
                            physics_row_at_t0=physics_row_at_t0)
    world_dict = build_world(world)
    arm = build_arm(world, t0_s=t0_s, hold_s=hold_s)
    post_restore_hook = None
    if obstacle_hook is not None:
        def _obstacle_hook(_api, _t_s):
            return obstacle_hook(_api, _t_s)

        post_restore_hook = _obstacle_hook
    summary = driver.drive_wall_branch(
        api, world_dict, arm, run_dir, restore,
        session=session, ownership=ownership, sensor=sensor,
        post_restore_hook=post_restore_hook,
        branch_length_s_override=float(R4_BRANCH_LENGTH_S),
    )
    feasible, reason = classify_branch(summary, run_dir)
    result = {
        "schema": "wm_n1_inc1_r4_result/1",
        "world": str(world["world"]),
        "arm": "R4",
        "run": arm["run"],
        "feasible": bool(feasible),
        "infeasible_reason": reason,
        "wall_outcome": summary.get("wall_outcome"),
        "assembled_at_s": summary.get("assembled_at_s"),
        "wall_request_at_s": summary.get("request_s"),
        "speed_factor": float(R4_SPEED_FACTOR),
        "branch_length_s_override": float(R4_BRANCH_LENGTH_S),
        "wall_id": summary.get("wall_id"),
        "members": summary.get("members"),
        "horizon_s": summary.get("horizon_s"),
        "t0_s": summary.get("t0_s"),
    }
    if write:
        write_result(run_dir, result)
    return result


def main(argv=None) -> int:                                # pragma: no cover
    """Read-only evidence probe: print the patched-driver identity."""
    import argparse

    parser = argparse.ArgumentParser(prog="inc1.r4_driver")
    parser.add_argument("--identity", action="store_true")
    args = parser.parse_args(argv)
    if args.identity:
        info = install_shims()
        driver = load_patched_wall_driver()
        info["wall_driver_file"] = str(driver.__file__)
        print(json.dumps(info, indent=2))
        return 0
    parser.error("nothing to do; pass --identity")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
