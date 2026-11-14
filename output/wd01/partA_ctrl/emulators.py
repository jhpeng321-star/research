"""Batched, device-agnostic torch emulators of the chunk-1 CAV controllers.

WD-01 Part A action conditioning, ablation A2, step 1.  These emulators
reproduce the COMMANDED SPEED of the frozen traffic_shepherd stack

- CIRCLES damping  (traffic_shepherd.damping: ONNX policy, decide()
  fallbacks, direct_target bridge),
- PI external-U    (output.controller_verification_v4.pi_external_u
  recurrence through the cv7 PIOwnerAdapter decide bridge),
- wall-v2.1 member command law (traffic_shepherd.wall_group closure target + _pace through
  the ownership direct_target bridge),

as pure tensor kernels, batched over CAVs, with explicit controller state
for the PI recurrence.  Equivalence against the reference implementations
is pinned by tests (E1-E3) and validated against the logged chunk-1 TRAIN
commands (E4, validate_chunk1.py).  ZERO SIMULATION: no SUMO/TraCI code is
imported or executed.

Fallback-reason codes mirror the exact reference strings (the integer
codes are an emulator-internal encoding; REASON_STRINGS is the contract).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from onnx2torch import OnnxGraph, TorchOnnxGraph

# ---------------------------------------------------------------- contract

REASON_STRINGS = {
    0: None,  # ok: a command was issued
    1: "selected_native",
    2: "missing_observation",
    3: "noncurrent_observation",
    4: "invalid_observation",
    5: "coverage_exceeds_contract",
    6: "invalid_neighbor_measurement",
    7: "missing_or_uncovered_leader",
    8: "invalid_actor_input",
    9: "invalid_model_output",
    10: "infeasible_speed_bounds",
}
REASON_CODES = {v: k for k, v in REASON_STRINGS.items()}

ACTION_CODES = {"native": 0, "damping": 1, "pi-external-u": 2, "wall-v2": 3}

# Adopted direct-interface contract (baselines/circles/adapter.json, whose
# sha256-pinned values load_config refuses to see changed).
CONTROL_DT_S = 1.0
SENSOR_RANGE_M = 100.0
ACCEL_MIN_MPS2 = -3.0
ACCEL_MAX_MPS2 = 1.5
MODEL_SHA256 = "762fde0763c693c0e88f41b98b22fc0a11135a312b78c4c567e6211a3035ccee"

# Registered PI external-U parameters (cv2 PISaturationParameters defaults)
PI_LOWER_GAP_M = 7.0
PI_UPPER_GAP_M = 30.0
PI_CATCH_SPEED_MPS = 1.0
PI_GAMMA_M = 2.0
REGISTERED_U_MPS = 31.29

# wall-v2 pacing bounds (wall_group.py)
WALL_BUFFER_DECEL_MPS2 = 3.0
WALL_COMFORT_ACCEL_MPS2 = 1.5


def default_device() -> torch.device:
    """Device from WD_DEVICE ('cpu' | 'cuda' | 'auto'), else cpu."""
    choice = os.environ.get("WD_DEVICE", "cpu").strip().lower()
    if choice in ("", "cpu"):
        return torch.device("cpu")
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("WD_DEVICE=cuda but CUDA is unavailable")
        return torch.device("cuda")
    raise ValueError(f"unsupported WD_DEVICE {choice!r}")


# ---------------------------------------------------------------- damping

@dataclass
class FrameBatch:
    """Batched ideal-sensor frame content the damping/PI law consumes.

    All tensors are shape [B]; ``*_present`` flags are bool.  Non-present
    neighbours must carry gap=rel=0 (masked out internally).  A missing
    frame (missing_observation) is ``frame_present=False`` with all
    numeric fields 0.
    """

    now_s: torch.Tensor            # float64, the control tick
    ts_s: torch.Tensor             # float64, frame timestamp
    speed_mps: torch.Tensor        # float64 >= 0
    allowed_speed_mps: torch.Tensor
    front_coverage_m: torch.Tensor
    rear_coverage_m: torch.Tensor
    front_present: torch.Tensor    # bool
    front_gap_m: torch.Tensor      # float64 >= 0
    front_rel_mps: torch.Tensor    # float64, other - ego
    rear_present: torch.Tensor
    rear_gap_m: torch.Tensor
    rear_rel_mps: torch.Tensor
    frame_present: torch.Tensor    # bool

    @classmethod
    def stack(cls, frames: list[dict], device=None) -> "FrameBatch":
        """From per-frame dicts (None frame allowed) -> batched tensors."""
        def f(value, default=0.0):
            return torch.tensor(default if value is None else float(value),
                                dtype=torch.float64, device=device)

        def b(value):
            return torch.tensor(bool(value), device=device)

        rows = []
        for frame in frames:
            if frame is None:
                rows.append(dict(
                    now_s=0.0, ts_s=0.0, speed_mps=0.0, allowed_speed_mps=0.0,
                    front_coverage_m=0.0, rear_coverage_m=0.0,
                    front_present=False, front_gap_m=0.0, front_rel_mps=0.0,
                    rear_present=False, rear_gap_m=0.0, rear_rel_mps=0.0,
                    frame_present=False))
                continue
            front = frame.get("front")
            rear = frame.get("rear")
            # explicit-flag form (hand-built gate tests) is also accepted
            if front is None and frame.get("front_present"):
                front = dict(gap_m=frame.get("front_gap_m", 0.0),
                             relative_speed_mps=frame.get("front_rel_mps", 0.0))
            if rear is None and frame.get("rear_present"):
                rear = dict(gap_m=frame.get("rear_gap_m", 0.0),
                            relative_speed_mps=frame.get("rear_rel_mps", 0.0))
            rows.append(dict(
                now_s=frame["now_s"], ts_s=frame.get("ts_s", frame["now_s"]),
                speed_mps=frame["speed_mps"],
                allowed_speed_mps=frame["allowed_speed_mps"],
                front_coverage_m=frame["front_coverage_m"],
                rear_coverage_m=frame["rear_coverage_m"],
                front_present=front is not None,
                front_gap_m=0.0 if front is None else front["gap_m"],
                front_rel_mps=0.0 if front is None else front["relative_speed_mps"],
                rear_present=rear is not None,
                rear_gap_m=0.0 if rear is None else rear["gap_m"],
                rear_rel_mps=0.0 if rear is None else rear["relative_speed_mps"],
                frame_present=True))
        return cls(
            now_s=torch.tensor([r["now_s"] for r in rows], dtype=torch.float64, device=device),
            ts_s=torch.tensor([r["ts_s"] for r in rows], dtype=torch.float64, device=device),
            speed_mps=torch.tensor([r["speed_mps"] for r in rows], dtype=torch.float64, device=device),
            allowed_speed_mps=torch.tensor([r["allowed_speed_mps"] for r in rows], dtype=torch.float64, device=device),
            front_coverage_m=torch.tensor([r["front_coverage_m"] for r in rows], dtype=torch.float64, device=device),
            rear_coverage_m=torch.tensor([r["rear_coverage_m"] for r in rows], dtype=torch.float64, device=device),
            front_present=torch.tensor([r["front_present"] for r in rows], device=device),
            front_gap_m=torch.tensor([r["front_gap_m"] for r in rows], dtype=torch.float64, device=device),
            front_rel_mps=torch.tensor([r["front_rel_mps"] for r in rows], dtype=torch.float64, device=device),
            rear_present=torch.tensor([r["rear_present"] for r in rows], device=device),
            rear_gap_m=torch.tensor([r["rear_gap_m"] for r in rows], dtype=torch.float64, device=device),
            rear_rel_mps=torch.tensor([r["rear_rel_mps"] for r in rows], dtype=torch.float64, device=device),
            frame_present=torch.tensor([r["frame_present"] for r in rows], device=device),
        )


def frame_problem_codes(batch: FrameBatch) -> torch.Tensor:
    """Vectorized traffic_shepherd.damping.frame_problem, same order.

    Returns [B] int64 codes; 0 means the frame passed every check."""
    codes = torch.zeros(batch.now_s.shape[0], dtype=torch.int64,
                        device=batch.now_s.device)
    dt = torch.abs(batch.ts_s - batch.now_s)

    def set_where(mask: torch.Tensor, code: int) -> None:
        empty = codes == 0
        codes.masked_fill_(empty & mask, code)

    set_where(~batch.frame_present, REASON_CODES["missing_observation"])
    set_where(~torch.isfinite(dt) | (dt > 1e-6), REASON_CODES["noncurrent_observation"])
    core = torch.stack([batch.speed_mps, batch.allowed_speed_mps,
                        batch.front_coverage_m, batch.rear_coverage_m])
    bad_core = (~torch.isfinite(core)).any(0) | (core < 0).any(0)
    set_where(bad_core, REASON_CODES["invalid_observation"])
    set_where(torch.maximum(batch.front_coverage_m, batch.rear_coverage_m)
              > SENSOR_RANGE_M, REASON_CODES["coverage_exceeds_contract"])
    front_bad = (batch.front_present
                 & ((~torch.isfinite(batch.front_gap_m))
                    | (~torch.isfinite(batch.front_rel_mps))
                    | (batch.front_gap_m < 0)
                    | (batch.front_gap_m > batch.front_coverage_m)))
    rear_bad = (batch.rear_present
                & ((~torch.isfinite(batch.rear_gap_m))
                   | (~torch.isfinite(batch.rear_rel_mps))
                   | (batch.rear_gap_m < 0)
                   | (batch.rear_gap_m > batch.rear_coverage_m)))
    set_where(front_bad | rear_bad, REASON_CODES["invalid_neighbor_measurement"])
    return codes


def direct_target(speed_mps: torch.Tensor, allowed_speed_mps: torch.Tensor,
                  acceleration: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized traffic_shepherd.damping.direct_target (float64).

    Returns (command, target) where infeasible rows carry
    (nan, nan) for (None, None).  target = max(0, speed + dt*command)."""
    lo = torch.clamp(-speed_mps / CONTROL_DT_S, min=ACCEL_MIN_MPS2)
    hi = torch.clamp((allowed_speed_mps - speed_mps) / CONTROL_DT_S,
                     max=ACCEL_MAX_MPS2)
    infeasible = (~torch.isfinite(acceleration)) | (lo > hi)
    safe_lo = torch.where(infeasible, ACCEL_MIN_MPS2, lo)
    safe_hi = torch.where(infeasible, ACCEL_MIN_MPS2, hi)
    command = torch.clamp(acceleration, safe_lo, safe_hi)
    command = torch.where(infeasible,
                          torch.full_like(command, float("nan")), command)
    target = torch.clamp_min(speed_mps + CONTROL_DT_S * command, 0.0)
    target = torch.where(infeasible,
                         torch.full_like(target, float("nan")), target)
    return command, target


class DampingEmulator:
    """Batched CIRCLES damping commanded-speed emulator (E1).

    ``decide`` reproduces CirclesController.decide: frame_problem gates in
    reference order, then the ONNX policy on the raw SI triple
    (ego, leader, bumper gap), then the direct_target bridge.
    """

    def __init__(self, model_path: str | Path, *,
                 model_sha256: str = MODEL_SHA256,
                 device: torch.device | None = None):
        import hashlib
        payload = Path(model_path).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != model_sha256:
            raise ValueError("CIRCLES model checksum mismatch")
        self.model_sha256 = digest
        self.device = device or default_device()
        self.net = TorchOnnxGraph(OnnxGraph(model_path)).to(self.device).eval()

    @torch.no_grad()
    def infer(self, ego: torch.Tensor, leader: torch.Tensor,
              gap: torch.Tensor) -> torch.Tensor:
        """Raw policy acceleration [B] (float32 graph, per-row singleton
        semantics pinned against onnxruntime in tests)."""
        x = torch.stack([ego, leader, gap], dim=1).to(torch.float32)
        return self.net(x)

    @torch.no_grad()
    def decide(self, batch: FrameBatch):
        """-> (target_speed_mps [B] float64, reason_code [B] int64, accel [B]).

        target is nan where no command was issued (fallback rows)."""
        codes = frame_problem_codes(batch)
        leader_speed = batch.speed_mps + batch.front_rel_mps
        leader_missing = (~batch.front_present) | (
            torch.minimum(batch.front_coverage_m, batch.rear_coverage_m) <= 0)
        set_code = REASON_CODES["missing_or_uncovered_leader"]
        codes = torch.where(leader_missing & (codes == 0),
                            torch.full_like(codes, set_code), codes)
        actor_ok = (torch.isfinite(batch.speed_mps) & (batch.speed_mps >= 0)
                    & torch.isfinite(leader_speed) & (leader_speed >= 0)
                    & torch.isfinite(batch.front_gap_m)
                    & (batch.front_gap_m >= 0))
        codes = torch.where(~actor_ok & (codes == 0),
                            torch.full_like(codes, REASON_CODES["invalid_actor_input"]),
                            codes)
        accel = torch.full_like(batch.speed_mps, float("nan"))
        command = torch.full_like(batch.speed_mps, float("nan"))
        target = torch.full_like(batch.speed_mps, float("nan"))
        live = (codes == 0) & ~leader_missing & actor_ok
        live_idx = torch.nonzero(live).squeeze(1)
        if live_idx.numel():
            raw = self.infer(batch.speed_mps[live_idx], leader_speed[live_idx],
                             batch.front_gap_m[live_idx]).to(torch.float64)
            finite = torch.isfinite(raw)
            kept_idx = live_idx[finite]
            bad_idx = live_idx[~finite]
            if bad_idx.numel():
                codes[bad_idx] = REASON_CODES["invalid_model_output"]
            if kept_idx.numel():
                cmd, tgt = direct_target(batch.speed_mps[kept_idx],
                                         batch.allowed_speed_mps[kept_idx],
                                         raw[finite])
                command[kept_idx] = cmd
                target[kept_idx] = tgt
                accel[kept_idx] = raw[finite]
                infeasible = torch.isnan(cmd)
                codes[kept_idx] = torch.where(
                    infeasible,
                    torch.full_like(codes[kept_idx], REASON_CODES["infeasible_speed_bounds"]),
                    codes[kept_idx])
        return target, codes, accel


# --------------------------------------------------------------- PI external-U

@dataclass
class PIState:
    """Explicit per-CAV PI recurrence state (batched).

    raw_prev: last committed raw command mps (the 'running target');
    last_ts:  timestamp of that commit (nan when freshly reset);
    active:   False == invalidated (re-init from measured speed at the
              next engaged tick), mirroring the adapter's lifecycle.
    """

    raw_prev: torch.Tensor
    last_ts: torch.Tensor
    active: torch.Tensor

    @classmethod
    def zeros(cls, batch: int, device=None) -> "PIState":
        return cls(
            raw_prev=torch.zeros(batch, dtype=torch.float64, device=device),
            last_ts=torch.full((batch,), float("nan"), dtype=torch.float64, device=device),
            active=torch.zeros(batch, dtype=torch.bool, device=device))


class PIEmulator:
    """Batched PI external-U commanded-speed emulator with explicit state (E2).

    Reproduces cv7 PIOwnerAdapter._compute_decision on top of the cv4
    PIExternalU recurrence: lifecycle invalidation, measured-speed
    re-initialization, the one-sample-period clock, the registered U=31.29
    report, the raw command update, and the direct_target bridge with the
    infeasible reset path.
    """

    def __init__(self, *, u_mps: float = REGISTERED_U_MPS,
                 device: torch.device | None = None):
        if float(u_mps) != REGISTERED_U_MPS:
            raise ValueError("pi-external-u requires registered U=31.29 m/s")
        self.u_mps = float(u_mps)
        self.device = device or default_device()

    @torch.no_grad()
    def raw_command(self, state: PIState, engaged: torch.Tensor,
                    now_s: torch.Tensor, ego: torch.Tensor,
                    leader: torch.Tensor, gap: torch.Tensor) -> tuple[torch.Tensor, PIState]:
        """One control tick of the recurrence for the engaged slots.

        engaged: bool [B] (the adapter prepared a proposal this tick).
        Returns (raw_command [B] float64 (nan off-engagement), new_state).
        Slots that are off-engagement are invalidated (adapter lifecycle).
        """
        raw_prev, last_ts, active = state.raw_prev, state.last_ts, state.active
        fresh = engaged & ~active
        # a nan last_ts is a fresh reset awaiting its first proposal: the
        # reference skips the continuity check when last is not finite
        clock_bad = (engaged & active & ~fresh & torch.isfinite(last_ts)
                     & (torch.abs(now_s - last_ts - CONTROL_DT_S) > 1e-9))
        # The adapter invalidates upstream so this never fires on contract
        # inputs; defensively re-init (same recovery as the reference
        # lifecycle) and let the caller count occurrences.
        fresh = fresh | clock_bad
        raw_prev = torch.where(fresh, ego, raw_prev)
        engaged_ok = engaged & ~clock_bad

        relative = leader - ego
        frac = torch.clamp((gap - PI_LOWER_GAP_M) / (PI_UPPER_GAP_M - PI_LOWER_GAP_M),
                           0.0, 1.0)
        target = self.u_mps + PI_CATCH_SPEED_MPS * frac
        safety_gap = torch.clamp(2.0 * relative, min=4.0)
        alpha = torch.clamp((gap - safety_gap) / PI_GAMMA_M, 0.0, 1.0)
        beta = 1.0 - alpha / 2.0
        raw = beta * (alpha * target + (1.0 - alpha) * leader) \
            + (1.0 - beta) * raw_prev
        raw = torch.where(engaged_ok, raw, torch.full_like(raw, float("nan")))
        new_state = PIState(
            raw_prev=torch.where(engaged_ok, raw, raw_prev),
            last_ts=torch.where(engaged_ok, now_s, last_ts),
            active=engaged_ok)
        return raw, new_state

    @torch.no_grad()
    def decide(self, state: PIState, batch: FrameBatch, engaged: torch.Tensor):
        """Full adapter-decide equivalent for one tick.

        -> (target [B], reason_code [B], raw [B], new_state, reset_mask [B]).
        Mirrors: frame gates (missing leader -> invalidation + fallback),
        recurrence, direct_target bridge; infeasible bridges invalidate the
        recurrence (fresh re-init next engagement)."""
        codes = frame_problem_codes(batch)
        # PI gate (cv7 pi_owner_adapter): front present AND front coverage
        # > 0.  Unlike the damping decide() gate there is NO rear-coverage
        # clause - a PI CAV at a lane start (rear_coverage 0) still engages.
        leader_missing = (~batch.front_present) | (batch.front_coverage_m <= 0)
        gate_bad = (codes != 0) | leader_missing
        # no proposal this tick -> lifecycle invalidation (reference:
        # invalidate via native/missing masks + _initialized bookkeeping)
        eff_engaged = engaged & ~gate_bad
        raw, new_state = self.raw_command(
            state, eff_engaged, batch.now_s, batch.speed_mps,
            batch.speed_mps + batch.front_rel_mps, batch.front_gap_m)
        reset = (engaged & gate_bad & state.active) | (eff_engaged & ~state.active)
        target = torch.full_like(batch.speed_mps, float("nan"))
        code_out = codes.clone()
        code_out = torch.where(gate_bad & (code_out == 0),
                               torch.full_like(code_out, REASON_CODES["missing_or_uncovered_leader"]),
                               code_out)
        infeasible = torch.zeros_like(gate_bad)
        ok = eff_engaged
        if ok.any():
            requested = (raw[ok] - batch.speed_mps[ok]) / CONTROL_DT_S
            cmd, tgt = direct_target(batch.speed_mps[ok],
                                     batch.allowed_speed_mps[ok], requested)
            target[ok] = tgt
            infeasible[ok] = torch.isnan(cmd)
            code_out[ok] = torch.where(
                infeasible[ok],
                torch.full_like(code_out[ok], REASON_CODES["infeasible_speed_bounds"]),
                code_out[ok])
        # infeasible bridge discards the proposal and invalidates the state
        if infeasible.any():
            new_state = PIState(
                raw_prev=torch.where(infeasible, state.raw_prev, new_state.raw_prev),
                last_ts=torch.where(infeasible, state.last_ts, new_state.last_ts),
                active=new_state.active & ~infeasible)
            reset = reset | (eff_engaged & infeasible & state.active)
        return target, code_out, raw, new_state, reset


# --------------------------------------------------------------- wall-v2

# Assembly closure (owner decision 2026-09-30, Option B; mirrors
# traffic_shepherd.wall_group ASSEMBLY_CLOSE_* / ASSEMBLY_TIMEOUT_S; the
# parity tests in tests/test_e3_wall.py pin the values to the reference).
WALL_CLOSE_MAX_DELTA_MPS = 2.0
WALL_CLOSE_GAIN_PER_S = 0.1
WALL_ASSEMBLY_TIMEOUT_S = 60.0
WALL_ASSEMBLY_SPAN_TOL_M = 6.0
WALL_LAW_VERSION = "wall-v2.1-closure"


@torch.no_grad()
def wall_gap_to_front(positions_m: torch.Tensor, group: torch.Tensor,
                      closure: torch.Tensor | None = None) -> torch.Tensor:
    """Per-member distance behind its wall's FRONT (max downstream position
    within the same group id; group < 0 = not in a wall).  ``closure``
    (bool [B], default all True) marks members whose wall is still ASSEMBLING;
    other members get gap 0 (HOLD: everyone paced to v_wall)."""
    pos = positions_m.to(torch.float64)
    gid = group.to(torch.int64)
    n = int(gid.max().item()) + 1 if gid.numel() else 0
    gap = torch.zeros_like(pos)
    if n <= 0:
        return gap
    idx = gid.clamp(min=0)
    front = torch.full((n,), float("-inf"), dtype=torch.float64,
                       device=pos.device)
    front = front.scatter_reduce(0, idx[gid >= 0], pos[gid >= 0], "amax",
                                 include_self=True)
    gap = torch.where(gid >= 0, front[idx] - pos, gap)
    if closure is not None:
        gap = torch.where(closure, gap, torch.zeros_like(gap))
    return gap


@torch.no_grad()
def wall_pace(wall_speed_mps: torch.Tensor, speed_mps: torch.Tensor,
              dt_s: float = 1.0,
              gap_to_front_m: torch.Tensor | None = None) -> torch.Tensor:
    """Vectorized wall-v2.1 member pacing law (wall_group._pace with the
    assembly closure target): the bounded absolute acceleration
    a = clip((v_target - v)/dt, -3, +1.5), v_target = v_wall +
    min(2.0, 0.1 * gap_to_front) (gap 0 / None -> v_wall: front member and
    the HOLD phase)."""
    target = wall_speed_mps
    if gap_to_front_m is not None:
        target = wall_speed_mps + torch.clamp(
            WALL_CLOSE_GAIN_PER_S * torch.clamp(gap_to_front_m, min=0.0),
            max=WALL_CLOSE_MAX_DELTA_MPS)
    raw = (target - speed_mps) / dt_s
    return torch.clamp(raw, WALL_BUFFER_DECEL_MPS2 * -1.0, WALL_COMFORT_ACCEL_MPS2)


# --------------------------------------------------------------- thin API (E5)

@dataclass
class ControllerState:
    """Batched emulator state handed between ticks (E5 contract)."""
    pi: PIState
    damping_target_mps: torch.Tensor | None = None   # last issued targets
    pi_running_target_mps: torch.Tensor | None = None


class ControllerEmulators:
    """emulate(action_kind, state, inputs) -> commanded speed + new state.

    Batched over CAVs; device from WD_DEVICE (default cpu).  action_kind
    codes: 0 native, 1 damping, 2 pi-external-u, 3 wall-v2 (ACTION_CODES).
    inputs: FrameBatch plus optional wall_speed_mps [B] for wall slots.
    """

    def __init__(self, model_path: str | Path, *,
                 model_sha256: str = MODEL_SHA256,
                 device: torch.device | None = None):
        self.device = device or default_device()
        self.damping = DampingEmulator(model_path, model_sha256=model_sha256,
                                       device=self.device)
        self.pi = PIEmulator(device=self.device)

    def init_state(self, batch: int) -> ControllerState:
        return ControllerState(pi=PIState.zeros(batch, device=self.device))

    @torch.no_grad()
    def emulate(self, action_kind: torch.Tensor, state: ControllerState,
                batch: FrameBatch,
                wall_speed_mps: torch.Tensor | None = None,
                wall_gap_to_front_m: torch.Tensor | None = None):
        """One control tick for a batch of CAVs.

        -> dict(target_speed_mps [B] (nan = no command), reason_code [B],
        raw_command_mps [B] (pi slots; nan elsewhere), new_state,
        pi_reset [B]).  Damping targets and PI running targets are exposed
        in new_state for controller_state comparison (dispatcher rows)."""
        action_kind = action_kind.to(self.device)
        b = action_kind.shape[0]
        target = torch.full((b,), float("nan"), dtype=torch.float64,
                            device=self.device)
        codes = torch.full((b,), REASON_CODES["selected_native"], dtype=torch.int64,
                           device=self.device)
        raw = torch.full((b,), float("nan"), dtype=torch.float64,
                         device=self.device)
        pi_reset = torch.zeros(b, dtype=torch.bool, device=self.device)
        pi_state = state.pi

        is_damping = action_kind == ACTION_CODES["damping"]
        if is_damping.any():
            tgt, code, _ = self.damping.decide(batch)
            target = torch.where(is_damping, tgt, target)
            codes = torch.where(is_damping, code, codes)
        # PI state advances EVERY tick (engaged slots commit; routed-away
        # slots are lifecycle-invalidated, mirroring the adapter's
        # invalidate(live=pi_subset) even when the subset is empty)
        is_pi = action_kind == ACTION_CODES["pi-external-u"]
        tgt, code, raw_pi, pi_state, reset = self.pi.decide(state.pi, batch,
                                                            is_pi)
        target = torch.where(is_pi & ~torch.isnan(tgt), tgt, target)
        codes = torch.where(is_pi, code, codes)
        raw = torch.where(is_pi, raw_pi, raw)
        pi_reset = reset
        is_wall = action_kind == ACTION_CODES["wall-v2"]
        if is_wall.any():
            if wall_speed_mps is None:
                raise ValueError("wall-v2 slots require wall_speed_mps")
            accel = wall_pace(wall_speed_mps.to(torch.float64),
                              batch.speed_mps,
                              gap_to_front_m=(None if wall_gap_to_front_m is None
                                              else wall_gap_to_front_m.to(
                                                  self.device, torch.float64)))
            fp = frame_problem_codes(batch)
            _, tgt = direct_target(batch.speed_mps, batch.allowed_speed_mps,
                                   accel)
            wall_ok = is_wall & (fp == 0) & ~torch.isnan(tgt)
            target = torch.where(wall_ok, tgt, target)
            wall_fb = is_wall & ~wall_ok
            codes = torch.where(
                wall_fb & (fp != 0), fp, codes)
            codes = torch.where(
                wall_fb & (fp == 0),
                torch.full_like(codes, REASON_CODES["infeasible_speed_bounds"]),
                codes)
        # native slots keep selected_native / no command
        return dict(target_speed_mps=target, reason_code=codes,
                    raw_command_mps=raw, new_state=ControllerState(
                        pi=pi_state,
                        damping_target_mps=torch.where(
                            torch.isnan(target), torch.full_like(target, float("nan")),
                            target),
                        pi_running_target_mps=torch.where(
                            is_pi & ~torch.isnan(raw), raw,
                            torch.full_like(raw, float("nan")))),
                    pi_reset=pi_reset)
