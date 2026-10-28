"""Opt-in S-EXECPARITY executor; no simulator imports, learned driver parameters only.

SUMO speedMode 27 = safe speed + maximum acceleration + junction/red-light checks;
bit 2 (comfortable maximum deceleration) is OFF. Do not floor a safety cap at -3.
The actor's lower bound limits deviations, not SUMO emergency stopping authority.
"""
from __future__ import annotations

import os
from functools import lru_cache
import torch

PARITY_ENV = "C4_EXECUTION_PARITY"
ACCEL_MAX = 1.2
SPEED_MODE = 27
CAP_REJECT_FRACTION = .95  # task registration, not a fitted behavioural threshold


def mode():
    """0 flag-OFF (frozen legacy switch path), 1 combined native-relative law,
    2 the 00:20 EXACT ALWAYS-COMMANDED law (CTRL0 path, no native release)."""
    value = os.environ.get(PARITY_ENV, "0")
    if value not in ("0", "1", "2"):
        raise ValueError(f"{PARITY_ENV} must be 0, 1 or 2")
    return int(value)


def enabled():
    return mode() != 0


def contract():
    """Portable checkpoint binding; flag-off payloads retain their original shape."""
    import hashlib
    from pathlib import Path
    if not enabled():
        return None
    driver = os.environ.get("C4_EXECUTION_DRIVER")
    if not driver:
        raise ValueError("candidate-A parity requires a frozen C4_EXECUTION_DRIVER fit")
    load_learned_driver(driver)
    executor = ("native_relative_same_step/1" if mode() == 1
                else "always_commanded_ctrl0/1")
    return {"schema": "execparity_contract/2", "executor": executor,
            "driver_sha256": hashlib.sha256(Path(driver).read_bytes()).hexdigest(),
            "accel_max_mps2": ACCEL_MAX, "decel_mode": "speedMode27 bit2 disabled",
            "speed_mode": SPEED_MODE, "imagination_soft_safe_accel": False}


def check_contract(payload):
    stored = payload.get("execution_contract")
    if stored is not None and stored != contract():
        raise ValueError("checkpoint execution contract differs from current flags/learned driver")


def executed_residual(state, residual, native, lo=-3., hi=ACCEL_MAX):
    """Dispatch on the execution mode.

    Mode 1 (combined): native must be a feasible same-step prediction.
    Mode 2 (always-commanded): ``native`` is unused - the base law is the
    deterministic inferred mirror of ``sumo_native_accel``, never the WM head.
    """
    if mode() == 2:
        from .alwayscmd import executed as always_executed
        return always_executed(state, residual, lo, hi)
    from .continuous import native_relative
    return native_relative(state, residual, native, lo, hi)


def executed_switch_residual(state, residual, native, lo=-3., hi=ACCEL_MAX, *, force_zero=False):
    """Exact command/cap algebra given the prospective SUMO follow-speed ceiling.

    The ceiling is recomputed each physics step. In imagination it is inferred
    from observable geometry and WM driver parameters; it is NOT the native head.
    Exactly zero releases ownership, unless the registered executor-only control
    explicitly requests force_zero. Positive residuals have exactly zero gradient.
    """
    from wms.model.cav import EXEC_CAP_KEY
    if EXEC_CAP_KEY not in state:
        raise ValueError("execution parity needs a per-physics-step SUMO follow-speed cap")
    cap = state[EXEC_CAP_KEY].to(residual)
    upper = min(float(hi), ACCEL_MAX)
    command = cap + residual.clamp_max(0.)
    command = torch.maximum(command, torch.minimum(cap, torch.full_like(cap, float(lo))))
    command = torch.minimum(command, torch.full_like(cap, upper))
    executed = torch.minimum(command, cap)
    # speedMode 27 still enforces nonnegative speed, even when bit 2 is disabled.
    executed = torch.maximum(executed, -state["v_mps"] / .5)
    return executed if force_zero else torch.where(residual != 0., executed, native)


def idm_follow_speed(speed, leader_speed, bumper_gap, params, *, dt=.5,
                     respect_min_gap=True):
    """Differentiable SUMO IDM::_v integration (two .25 s updates at dt=.5).

    Algorithm: eclipse-sumo v1_24_0 MSCFModel_IDM.cpp::_v. All behavioural
    parameters are supplied by the learned WM encoder; no driver table is used.
    gap is the physical bumper-to-bumper gap from observable positions. SUMO's
    internal gap2pred/getLeader gap excludes minGap; _v adds it back. With physical
    geometry it is therefore already included and must not be added a second time.
    Only IDM is supported, not Krauss/IDMM; junction constraints are separate.
    """
    iterations = max(1, int(float(dt) / .25 + .5))
    a, b, tau, gap0, v0 = (params[k].to(speed) for k in
                            ("accel", "decel", "time_headway", "min_gap", "v0"))
    eps = torch.finfo(speed.dtype).eps
    v, gap = speed, bumper_gap
    for _ in range(iterations):
        desired = (v*tau + v*(v-leader_speed)/(2.*(a*b).clamp_min(eps).sqrt())).clamp_min(0.)
        if respect_min_gap:
            desired = desired + gap0
        accel = a*(1.-(v/v0.clamp_min(eps)).pow(4)-(desired/gap.clamp_min(eps)).square())
        v = (v+accel*float(dt)/iterations).clamp_min(0.)
        gap = gap - (v-leader_speed).clamp_min(0.)*float(dt)/iterations
    return v


def inferred_follow_cap(state, params, *, dt=.5):
    """Prospective commanded-CAV ceiling from learned settings and current geometry.

    Does not apply the WM's learned yielding patch or soft safe_accel: SUMO's
    externally commanded path omits native lane-change speed patching. A ramp
    lane-end stop is retained. Unobserved junction/stop state cannot be recovered
    from old episode tapes; report that residual model error in replay tests.
    """
    from .native_cav import leader_gap_dv
    from .contract import MERGE_S1_M
    v = state["v_mps"]
    fitted = os.environ.get("C4_EXECUTION_DRIVER")
    if fitted:
        values = load_learned_driver(fitted)
        params = {key: torch.full_like(v, value) for key,value in values.items()}
    gap, dv, has = leader_gap_dv(state)
    gap = torch.where(has, gap, torch.full_like(gap, float("inf")))
    target = idm_follow_speed(v, v-dv, gap, params, dt=dt)
    ramp_gap = (MERGE_S1_M-state["s_m"]).clamp_min(0.)
    stop = idm_follow_speed(v, torch.zeros_like(v), ramp_gap, params,
                           dt=dt, respect_min_gap=False)
    stop = torch.minimum(stop, ramp_gap/float(dt))
    target = torch.where(state["lane"] == 2, torch.minimum(target, stop), target)
    target = torch.minimum(target, params["v0"].to(v))
    cap = torch.minimum((target-v)/float(dt), params["accel"].to(v))
    return torch.minimum(cap, torch.full_like(v, ACCEL_MAX))


@lru_cache(maxsize=8)
def load_learned_driver(path):
    """Only a TRAIN-fitted, provenance-bearing IDM artifact is admissible."""
    import json
    from pathlib import Path
    import re
    p = Path(path)
    if re.search(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype",
                 str(p)+str(p.resolve()),re.I):
        raise ValueError("prohibited driver artifact")
    payload = json.loads(p.read_text(encoding="utf-8"))
    values = payload.get("parameters",{})
    if (payload.get("schema") != "execparity_learned_idm/1"
            or payload.get("fit_controller") != "direct"
            or payload.get("evaluation_data_used") is not False
            or not payload.get("fit_worlds") or not payload.get("score_worlds")
            or set(values) != {"accel","decel","time_headway","min_gap","v0"}
            or not all(isinstance(x,(float,int)) and 0. < x < float("inf") for x in values.values())):
        raise ValueError("invalid learned execution-driver artifact")
    return values


def authority_guard(residuals, *, checkpoint_sha256, recordings_sha256, executed_delta=None,
                    capped=None):
    """Fail-closed pre-SUMO receipt for the registered >95% clipping rule.

    Combined mode counts partial AND full envelope clipping, plus zero requests;
    clipping indicators come from the prospective feasible bounds. Full zero
    effect is reported separately. Without executed_delta this retains only the
    historical sign-based switch-path diagnostic, never a deployment receipt.
    Evaluate all acting rows, never select by the realised outcome.
    """
    import numpy as np
    r = np.asarray(residuals, float).reshape(-1)
    if not len(r) or not np.isfinite(r).all() or not checkpoint_sha256 or not recordings_sha256:
        raise ValueError("authority guard requires finite acting rows and bound inputs")
    if executed_delta is None:
        fraction = float((r >= 0.).mean())
        schema = "execparity_authority/1"  # retained switch-path diagnostic only
    else:
        d = np.asarray(executed_delta, float).reshape(-1)
        if d.shape != r.shape or not np.isfinite(d).all():
            raise ValueError("authority guard needs paired finite executed residual effects")
        cap = np.asarray(capped, bool).reshape(-1) if capped is not None else np.abs(d) < np.abs(r)
        if cap.shape != r.shape:
            raise ValueError("authority guard needs paired clipping indicators")
        fraction = float((cap | (r == 0.)).mean())
        schema = "execparity_authority/2"
    receipt = {"schema": schema, "checkpoint_sha256": checkpoint_sha256,
               "recordings_sha256": recordings_sha256, "acting_ticks": len(r),
               "capped_or_native_fraction": fraction, "maximum_fraction": CAP_REJECT_FRACTION,
                "passed": fraction <= CAP_REJECT_FRACTION}
    if executed_delta is not None:
        receipt["execution_contract"] = contract()
        receipt["cap_definition"] = "any_envelope_clipping_or_zero"
        receipt["fully_capped_fraction"] = float(((d == 0.) & (r != 0.)).mean())
        receipt["partially_capped_fraction"] = float((cap & (d != 0.)).mean())
    return receipt


def verify_authority(path, checkpoint):
    """Validate the replay receipt before constructing any deployment environment."""
    import hashlib
    import json
    from pathlib import Path
    if path is None:
        raise ValueError("execution parity deployment requires --execution-authority replay receipt")
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    fraction = receipt.get("capped_or_native_fraction", float("nan"))
    hashes = receipt.get("recordings", {})
    if (receipt.get("schema") != "execparity_authority/2" or not receipt.get("passed")
            or receipt.get("cap_definition") != "any_envelope_clipping_or_zero"
            or receipt.get("execution_contract") != contract()
            or receipt.get("acting_ticks", 0) <= 0 or not 0. <= fraction <= CAP_REJECT_FRACTION
            or receipt.get("maximum_fraction") != CAP_REJECT_FRACTION
            or receipt.get("checkpoint_sha256") != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
            or not hashes or receipt.get("recordings_sha256") != hashlib.sha256(
                json.dumps(hashes, sort_keys=True).encode()).hexdigest()):
        raise ValueError("missing/stale/failing execution-authority replay receipt")
    return receipt


def verify_parity(path):
    """The opt-in deployment cannot bypass a failed recorded-acceleration gate."""
    import hashlib
    import json
    from pathlib import Path
    if path is None:
        raise ValueError("execution parity requires --execution-parity-receipt")
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    driver = os.environ.get("C4_EXECUTION_DRIVER")
    native_rmse = receipt.get("native_arm_head_error", {}).get("rmse", float("nan"))
    errors = receipt.get("error", {})
    residual_rmse = receipt.get("c1_residual_execution_error", {}).get("rmse", float("nan"))
    if mode() == 2:
        # 00:20 law: real-SUMO measurement, CTRL0 zero identity and executed-accel
        # error inside the WM native-head band on the same states.
        if (receipt.get("schema") != "execparity_always_commanded_acceleration/1"
                or receipt.get("backend_contract") != "always_commanded_ctrl0/1"
                or receipt.get("measured_real_sumo") is not True
                or receipt.get("zero_ctrl0_identity") is not True
                or not receipt.get("passed") or not driver or not receipt.get("worlds")
                or receipt.get("driver_sha256") != hashlib.sha256(Path(driver).read_bytes()).hexdigest()
                or set(errors) != {"ours"} or not 0. <= residual_rmse <= native_rmse
                or not all(0. <= value.get("rmse", float("nan")) <= native_rmse
                           for value in errors.values())):
            raise ValueError("missing/stale/failing always-commanded parity receipt")
        load_learned_driver(driver)
        return receipt
    if (receipt.get("schema") != "execparity_continuous_acceleration/1"
            or receipt.get("backend_contract") != "native_relative_same_step/1"
            or receipt.get("measured_real_sumo") is not True
            or receipt.get("zero_native_identity") is not True
            or not receipt.get("passed") or not driver or not receipt.get("worlds")
            or receipt.get("driver_sha256") != hashlib.sha256(Path(driver).read_bytes()).hexdigest()
            or set(errors)!={"ours"} or not 0.<=residual_rmse<=native_rmse
            or not all(0.<=value.get("rmse",float("nan"))<=native_rmse for value in errors.values())):
        raise ValueError("missing/stale/failing recorded-acceleration parity receipt")
    load_learned_driver(driver)
    return receipt
