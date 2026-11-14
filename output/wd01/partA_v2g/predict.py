"""WD-01 partA_v2g predict(): the deterministic Part A v2 general
predictor, contract 'wd01_predict/2' with M = 1 + the v2.1 tracked-scope
fields (PREDICT_CONTRACT_v2_1.md, frozen; LF-sha256 of the design
ba52820a6559cfe766daa1be4aa29c479ec7b5c9e9760c1b5ac21106b2b361d2).

- Input: ONE reconstructor sample per K (the Part-A view; the adapter
  rejects records lacking s_down_m/lane_phys).  StateSample.arrivals is
  IGNORED, never refused, and NO vehicle is inserted.
- Dynamics: deterministic Treiber IDM + MOBIL + the mandatory merge rule
  with ONE nominal parameter set (params_nominal.json,
  meta.param_set_id = its sha256).
- part_a_scope = 'tracked': delay_part_a_s / safety fields are computed
  over the tracked (recon-provided) vehicles only.  delay_part_b_s is
  0.0 -- this predictor has NO Part B head, so its plan/native totals
  cover the tracked scope only (the planner compares part_a values only
  under equal scopes, v2.1 section 3).
- horizon_s is capped at the 120 s Part A rollout (part_a_horizon_s);
  longer horizons are rejected loudly.
- continuation: 'native' switches every CAV to native after the commit
  window; 'planner' holds the commit-window actions (the documented
  stand-in for pi, identified by continuation_policy_id
  'parta_v2g/hold-commit-actions').
- future_* fields are realised deterministic quantities (p in {0, 1},
  q05 = the realised minimum), the perfect-simulator encoding.
- future_min_ttc_validated stays False (the future TTC head is not
  validated).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import numpy as np

from . import metrics as MX
from .adapter import parse_sample
from .engine import (PartAV2GEngine, free_road_flags, load_params,
                     param_set_id)

import sys

_WD01 = Path(__file__).resolve().parents[1]
# The shared predict kit (output/wd01/predict) must win over THIS file
# (partA_v2g/predict.py) whenever the part dir itself sits on sys.path;
# keeping output/wd01 first makes 'predict' resolve to the kit.
if not sys.path or str(sys.path[0]) != str(_WD01):
    sys.path.insert(0, str(_WD01))

from predict.contract import (  # noqa: E402
    CONTINUATION_NATIVE,
    CONTINUATION_PLANNER,
    CONTRACT_VERSION_V2,
    PART_A_SCOPE_TRACKED,
    Meta,
    PlanCandidate,
    Prediction,
    WallOutcome,
)
from predict.native_ref import native_reference  # noqa: E402
from predict.protocol import PredictorV2Mixin  # noqa: E402
from predict.validate import validate_plan, validate_state_samples  # noqa: E402

MODEL_ID = "parta_v2g/1"
MEMBER_ID = "parta_v2g_nominal"
RECON_SCHEMA_VERSION = "partA_view/1 (r0 SCHEMA_V1.md mapping)"
HOLD_ACTIONS_POLICY_ID = "parta_v2g/hold-commit-actions"
VALUE_MODEL_ID = "parta_v2g/none"
DT = 0.5


class PartAV2GeneralPredictor(PredictorV2Mixin):
    """Deterministic general-rules predictor, contract wd01_predict/2,
    M = 1, part_a_scope 'tracked'."""

    model_id = MODEL_ID
    part_a_horizon_s = 120.0

    def __init__(self, params: Optional[Mapping] = None,
                 params_path: Path = None):
        from .engine import PARAMS_PATH
        self._params_path = Path(params_path) if params_path else PARAMS_PATH
        self.params = dict(params) if params else load_params(self._params_path)
        self.pid = param_set_id(self._params_path)
        self.dt = float(self.params["dt_s"])

    def weights_sha256(self) -> str:
        return self.pid

    # ------------------------------------------------------------------

    def predict_many(self, state_samples, plans, controller_state, *,
                     horizon_s: float = 120.0, commit_s: float = 30.0,
                     continuation: str = CONTINUATION_NATIVE,
                     trajectory_s: Optional[float] = None):
        t_start = time.perf_counter()
        horizon_s = float(horizon_s)
        commit_s = float(commit_s)
        if continuation not in (CONTINUATION_NATIVE, CONTINUATION_PLANNER):
            raise ValueError(
                f"continuation must be 'native' or 'planner', got "
                f"{continuation!r}")
        part_a = float(self.part_a_horizon_s)
        if horizon_s > part_a + 1e-9:
            raise ValueError(
                f"horizon_s {horizon_s} exceeds the Part A rollout "
                f"({part_a} s); Part A v2 general is a 120 s product")
        if commit_s > part_a + 1e-9 or commit_s <= 0:
            raise ValueError(f"commit_s {commit_s} outside (0, {part_a}]")
        traj_s = commit_s if trajectory_s is None else float(trajectory_s)
        if traj_s > part_a + 1e-9 or traj_s < 0:
            raise ValueError(
                f"trajectory_s {traj_s} outside [0, {part_a}]")
        # the contract trajectory grid is 1 s (T = trajectory_s + 1)
        T = int(round(traj_s)) + 1

        samples = validate_state_samples(list(state_samples))
        for plan in plans:
            validate_plan(plan, controller_state)
        if not plans:
            raise ValueError("plans must be non-empty")

        tables = [parse_sample(s.hybrid, self.params) for s in samples]
        ids0 = set(tables[0].vehicles)
        for k, t in enumerate(tables[1:], start=1):
            if set(t.vehicles) != ids0:
                raise ValueError(
                    "tracked ids must be stable across the K samples "
                    f"(sample {k} differs)")
        K = len(samples)
        M = 1
        n_h = int(round(horizon_s / self.dt))
        n_d = int(round(commit_s / self.dt))
        dt = self.dt

        cav_ids = sorted(controller_state)
        native_plan, native_kept_walls = native_reference(
            controller_state, cav_ids)

        # native branch: computed ONCE per call, shared by every plan
        native_runs = [self._rollout(tables[k], native_plan, controller_state,
                                     continuation, n_h, commit_s)
                       for k in range(K)]
        out = []
        for plan in plans:
            plan_runs = [self._rollout(tables[k], plan, controller_state,
                                       continuation, n_h, commit_s)
                         for k in range(K)]
            out.append(self._prediction(
                samples=samples, tables=tables, plan=plan,
                plan_runs=plan_runs, native_runs=native_runs,
                native_plan=native_plan,
                native_kept_walls=native_kept_walls, horizon_s=horizon_s,
                commit_s=commit_s, continuation=continuation, T=T, n_h=n_h,
                n_d=n_d, wall_clock=time.perf_counter() - t_start))
        return out

    # ------------------------------------------------------------------

    def _rollout(self, table, plan: PlanCandidate, controller_state,
                 continuation: str, n_h: int, commit_s: float):
        """One deterministic engine rollout under one plan."""
        vehicles = {vid: dict(rec) for vid, rec in table.vehicles.items()}
        cav_set = set(controller_state) | set(table.cav_block)
        for vid in cav_set:
            if vid in vehicles:
                vehicles[vid]["is_cav"] = True
        hook = None
        wall_members = {m for w in plan.walls for m in w.members}
        has_control = (any(a != "native" for a in plan.per_cav.values())
                       or wall_members)
        if has_control and cav_set:
            from .cavctrl import CavControllerLoop
            switch = table.t_s + commit_s \
                if continuation == CONTINUATION_NATIVE else None
            loop = CavControllerLoop(t0=table.t_s, params=self.params,
                                     per_cav_actions=plan.per_cav,
                                     walls=plan.walls,
                                     cav_state_hint=table.cav_block,
                                     switch_native_at=switch)
            hook = loop
        eng = PartAV2GEngine(self.params, t0=table.t_s, vehicles=vehicles,
                             control_hook=hook)
        eng.run(n_h)
        snaps = MX.snaps_list(eng)
        return {"engine": eng, "snaps": snaps, "loop": hook}

    def _prediction(self, *, samples, tables, plan, plan_runs, native_runs,
                    native_plan, native_kept_walls, horizon_s, commit_s,
                    continuation, T, n_h, n_d, wall_clock):
        K = len(samples)
        M = 1
        ids = sorted(tables[0].vehicles)
        lane_lengths = self.params["geometry"]["lane_lengths_m"]

        def cell_stats(run, lo, hi):
            """Window/future stats over snapshot steps lo+1..hi."""
            snaps = run["snaps"]
            eng = run["engine"]
            base = lo
            expo, min_ttc, closing = MX.ttc_exposure(
                snaps[base:hi + 1], self.params, ids)
            n_coll = sum(eng.collisions_by_step.get(j, 0)
                         for j in range(base + 1, hi + 1))
            hb = MX.hard_brake_s(snaps[base:hi + 1], self.params, ids)
            return dict(collision=n_coll > 0, n_collisions=n_coll,
                        min_ttc=min_ttc, closing=closing,
                        expo3=expo, hard=hb)

        plan_win = [cell_stats(r, 0, n_d) for r in plan_runs]
        plan_fut = [cell_stats(r, n_d, n_h) for r in plan_runs]
        nat_win = [cell_stats(r, 0, n_d) for r in native_runs]
        nat_fut = [cell_stats(r, n_d, n_h) for r in native_runs]

        def col(seq, key):
            return np.array([[float(x[key]) for x in seq]], dtype=np.float64)

        def bcol(seq, key):
            return np.array([[bool(x[key]) for x in seq]], dtype=bool)

        # delays: part_a over the commit window, part_b = 0 (tracked
        # scope, no Part B head); identities hold by construction
        pa = np.array([[MX.window_delay(r["snaps"][:n_d + 1], self.params,
                                        ids)] for r in plan_runs]
                      ).T.reshape(1, K)
        na = np.array([[MX.window_delay(r["snaps"][:n_d + 1], self.params,
                                        ids)] for r in native_runs]
                      ).T.reshape(1, K)
        zeros = np.zeros((M, K), dtype=np.float64)
        plan_delay = pa + zeros
        native_delay = na + zeros

        # trajectories [M, K, T, 4]: NaN tail after exit (contract 4.3)
        traj = {vid: np.full((M, K, T, 4), np.nan, dtype=np.float64)
                for vid in ids}
        ntraj = {vid: np.full((M, K, T, 4), np.nan, dtype=np.float64)
                 for vid in ids}
        for k, run in enumerate(plan_runs):
            self._fill_traj(traj, k, run, T)
        for k, run in enumerate(native_runs):
            self._fill_traj(ntraj, k, run, T)

        # v2.1 per-cell tracked-scope fields
        n_tracked = np.full((M, K), float(len(ids)), dtype=np.float64)
        n_exited = np.array(
            [[float(r["engine"].n_exited_before(
                r["engine"].t0 + commit_s)) for r in plan_runs]],
            dtype=np.float64)
        n_free = np.array(
            [[float(len(free_road_flags(tables[k].vehicles,
                                        tables[k].zone_extent_s,
                                        lane_lengths)))]
             for k in range(K)]).T.reshape(1, K)

        # walls: in-window v1 fields + the v2 future heads (realised /
        # deterministic: p in {0, 1})
        walls = {}
        specs = {w.wall_id: w for w in plan.walls}
        for w in native_plan.walls:
            specs.setdefault(w.wall_id, w)
        for wall_id, w in specs.items():
            assembled = np.zeros((M, K), dtype=bool)
            at = np.full((M, K), np.nan, dtype=np.float64)
            abort = np.zeros((M, K), dtype=bool)
            for k, run in enumerate(plan_runs):
                loop = run["loop"]
                if loop is None:
                    continue
                wb = loop.walls
                if wall_id not in wb.specs:
                    continue
                if wb.assembled_at.get(wall_id) is not None:
                    assembled[0, k] = True
                    at[0, k] = float(wb.assembled_at[wall_id])
                abort[0, k] = bool(wb.safety_abort.get(wall_id, False))
            p_done = assembled.astype(np.float64)
            expected = np.where(assembled, at, np.nan)
            walls[wall_id] = WallOutcome(
                assembled=assembled, assembled_at_s=at,
                safety_abort=abort,
                abort_reason=np.where(abort, 1, 0).astype(np.int8),
                p_assembly_completes=p_done,
                expected_assembly_time_s=expected,
                p_assembly_abort=abort.astype(np.float64))

        meta = Meta(
            contract_version=CONTRACT_VERSION_V2,
            model_id=self.model_id,
            weights_sha256=(self.pid,),
            sample_ids=tuple(s.meta.sample_id for s in samples),
            member_ids=(MEMBER_ID,),
            part_a_horizon_s=float(self.part_a_horizon_s),
            safety_window_s=float(commit_s),
            horizon_s=float(horizon_s),
            native_kept_walls=tuple(native_kept_walls),
            native_unlocked_available=False,
            device="cpu",
            wall_clock_s=float(wall_clock),
            commit_s=float(commit_s),
            continuation=continuation,
            continuation_policy_id=(
                None if continuation == CONTINUATION_NATIVE
                else HOLD_ACTIONS_POLICY_ID),
            value_model_id=VALUE_MODEL_ID,
            value_model_sha256=(self.pid,),
            future_min_ttc_validated=False,
            future_min_ttc_validation_id=None,
            part_a_model="v2_general",
            param_set_id=self.pid,
            recon_schema_version=RECON_SCHEMA_VERSION,
            part_a_scope=PART_A_SCOPE_TRACKED,
        )

        return Prediction(
            delay_vs_native_s=plan_delay - native_delay,
            plan_delay_s=plan_delay,
            native_delay_s=native_delay,
            delay_part_a_s=pa,
            delay_part_b_s=zeros,
            collision=bcol(plan_win, "collision"),
            min_ttc_s=col(plan_win, "min_ttc"),
            ttc_lt3_exposure_s=col(plan_win, "expo3"),
            hard_brake_s=col(plan_win, "hard"),
            native_collision=bcol(nat_win, "collision"),
            native_min_ttc_s=col(nat_win, "min_ttc"),
            native_ttc_lt3_exposure_s=col(nat_win, "expo3"),
            native_hard_brake_s=col(nat_win, "hard"),
            trajectories_tracked=traj,
            native_trajectories_tracked=ntraj,
            walls=walls,
            meta=meta,
            cell_valid=np.ones((M, K), dtype=bool),
            native_cell_valid=np.ones((M, K), dtype=bool),
            has_closing_pair=bcol(plan_win, "closing"),
            native_has_closing_pair=bcol(nat_win, "closing"),
            future_cell_valid=np.ones((M, K), dtype=bool),
            native_future_cell_valid=np.ones((M, K), dtype=bool),
            future_has_closing_pair=bcol(plan_fut, "closing"),
            native_future_has_closing_pair=bcol(nat_fut, "closing"),
            future_ttc_lt3_exposure_s=col(plan_fut, "expo3"),
            future_hard_brake_s=col(plan_fut, "hard"),
            p_future_collision=bcol(plan_fut, "collision").astype(
                np.float64),
            future_min_ttc_q05_s=col(plan_fut, "min_ttc"),
            native_future_ttc_lt3_exposure_s=col(nat_fut, "expo3"),
            native_future_hard_brake_s=col(nat_fut, "hard"),
            native_p_future_collision=bcol(nat_fut, "collision").astype(
                np.float64),
            native_future_min_ttc_q05_s=col(nat_fut, "min_ttc"),
            n_tracked=n_tracked,
            n_exited_tracked=n_exited,
            n_free_road_assumed=n_free,
        )

    def _fill_traj(self, traj, k: int, run, T: int) -> None:
        """Fill the [.., T, 4] trajectory on the contract's 1 s grid
        (t_s, s_m, lane_id, speed_mps) from the 0.5 s engine snapshots;
        whole-row NaN from the exit step on (contract 4.3)."""
        table = run["engine"].trajectory_table()
        t0 = run["engine"].t0
        per_s = int(round(1.0 / self.dt))
        for vid, rows in table.items():
            arr = traj[vid]
            for i in range(T):
                j = i * per_s
                if j >= len(rows) or rows[j] is None:
                    break
                st = rows[j]
                arr[0, k, i, 0] = t0 + j * self.dt
                arr[0, k, i, 1] = st[1]
                arr[0, k, i, 2] = float(st[0])
                arr[0, k, i, 3] = st[2]
