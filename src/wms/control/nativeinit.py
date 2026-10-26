"""S-NATIVEINIT-1 contracts. Pure gates never import or launch SUMO."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

PANEL_SHA256 = "3eaa69a170c17302660d12dd4d7fddb4657e43fa334eece17844dd0fa6c0411e"
SMOKE_WORLDS = 4


def pair_merge_streams(state, row_start, n_base, seed):
    """Hybrid first-passage stream keys are per START/slot, never per variant row.

    Saved carry wins. Keys cover padded slots too, so entrant births remain paired.
    An independent generator preserves all existing policy/lane/entry RNG streams.
    """
    import torch
    if "merge_pivot_rng_seed" not in state:
        generator = torch.Generator().manual_seed(int(seed))
        keys = torch.randint(0, 1 << 52, (n_base, state["s_m"].shape[-1]), generator=generator)
        keys = keys.to(state["s_m"].device)
        state["merge_pivot_rng_seed"] = keys if row_start is None else keys.index_select(0, row_start)
    return state


def paired_reward(reward_fn):
    """Same row-local reduction geometry, so identical CUDA rows have identical J.

    Batched float32 reductions can vary with row alignment even for identical
    trajectories. This does not add a tolerance or change the reward equation.
    """
    import torch
    def reward(state, next_state, accel, weights):
        rows = accel.shape[0]
        def row(mapping, i):
            return {k: (v[i:i + 1] if torch.is_tensor(v) and v.ndim and v.shape[0] == rows else v)
                    for k, v in mapping.items()}
        parts = [reward_fn(row(state, i), row(next_state, i), accel[i:i + 1], weights)
                 for i in range(rows)]
        return {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}
    return reward


def compose_native(native, residual, lo, hi):
    """Preserve the native command even outside the policy envelope; bound deviations."""
    import torch
    lower = torch.minimum(native, torch.full_like(native, float(lo)))
    upper = torch.maximum(native, torch.full_like(native, float(hi)))
    return torch.minimum(torch.maximum(native + residual, lower), upper)


@dataclass(frozen=True)
class NativeResidualCommand:
    residual: np.ndarray
    accel_min: float
    accel_max: float
    native_residual = True

    def __array__(self, *args, **kwargs):
        raise TypeError("residual is not absolute acceleration; use the native-aware SUMO executor")

    def subset(self, mask):
        residual = self.residual.copy()
        residual[0, ~np.asarray(mask, bool)] = 0.0
        return NativeResidualCommand(residual, self.accel_min, self.accel_max)


def sumo_native_accel(api, vehicle_id, dt=0.5):
    """Query SUMO's own car-following model, never the WM twin or a type file.

    This is a car-following COMMAND, not an oracle for the next realized speed:
    junction/stop/lane-change constraints remain SUMO's responsibility. No use of
    getSpeedWithoutTraCI (that is a retrospective value, not a next-step prediction).
    """
    vehicle = api.vehicle
    speed = float(vehicle.getSpeed(vehicle_id))
    leader = vehicle.getLeader(vehicle_id)
    if leader is None:
        # Infinite gap is the no-leader limit, not a hand-selected lookahead.
        target = float(vehicle.getFollowSpeed(vehicle_id, speed, float("inf"), speed,
                                               float(vehicle.getDecel(vehicle_id)), ""))
    else:
        lead_id, gap = leader
        target = float(vehicle.getFollowSpeed(
            vehicle_id, speed, float(gap) + float(vehicle.getMinGap(vehicle_id)),
            float(vehicle.getSpeed(lead_id)), float(vehicle.getDecel(lead_id)), lead_id))
    target = min(target, float(vehicle.getAllowedSpeed(vehicle_id)))
    return (max(0.0, target) - speed) / dt


def sumo_residual_commands(api, residuals, lo, hi, *, dt=0.5):
    """Prospective absolute commands for the observable collection action ledger."""
    composed = {}
    for vid, residual in residuals.items():
        if not np.isfinite(residual):
            raise ValueError("nonfinite residual for a designated CAV")
        if residual == 0.0:
            continue
        baseline = sumo_native_accel(api, vid, dt)
        upper = min(float(hi), float(api.vehicle.getAccel(vid)))
        composed[vid] = float(np.clip(baseline + residual, min(lo, baseline), max(upper, baseline)))
    return composed


def execute_sumo_residual(api, residuals, lo, hi, *, dt=0.5):
    """Every physics tick: release zeros; recompute native then add held residual.

    The caller owns only designated CAV ids. Native lanes are never taken over.
    SUMO speedMode27 remains the final safety executor.
    """
    composed = sumo_residual_commands(api, residuals, lo, hi, dt=dt)
    for vid, command in composed.items():
        api.vehicle.setSpeed(vid, -1)  # cancel last tick's external speed influence
        api.vehicle.setAcceleration(vid, command, dt)
    return composed


def panel_gate(policy_j, native_j, *, update, panel_sha256=PANEL_SHA256):
    p, n = np.asarray(policy_j, float), np.asarray(native_j, float)
    if p.shape != (16,) or n.shape != (16,) or not np.isfinite([p, n]).all():
        raise ValueError("native gate requires 16 finite paired costs")
    if panel_sha256 != PANEL_SHA256:
        raise ValueError("native gate panel SHA differs from registration")
    delta = p - n
    return {"schema": "nativeinit_panel/1", "update": int(update), "panel_sha256": panel_sha256,
            "policy_J": p.tolist(), "native_J": n.tolist(), "delta_per_start": delta.tolist(),
            "mean_delta": float(delta.mean()), "passed": bool((delta <= 0).all()),
            "critic_tail_included": False}


def select_update(curve):
    """Earliest measured minimum among trained, all-start noninferior checkpoints."""
    eligible = [r for r in curve if r["passed"] and r["update"] > 0]
    return min(eligible, key=lambda r: (r["mean_delta"], r["update"]))["update"] if eligible else None


def smoke_guard(world_deltas):
    """One registered look; each input sums BOTH paired starts of one world."""
    delta = np.asarray(world_deltas, float)
    if delta.shape != (SMOKE_WORLDS,) or not np.isfinite(delta).all():
        raise ValueError("smoke requires exactly four finite complete paired worlds")
    rng = np.random.default_rng(20261026)
    draws = delta[rng.integers(0, SMOKE_WORLDS, (5000, SMOKE_WORLDS))].mean(1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {"schema": "nativeinit_smoke/1", "world_deltas": delta.tolist(),
            "mean_delta": float(delta.mean()), "ci95": [float(lower), float(upper)],
            "margin": 0.0, "passed": bool(upper <= 0),
            "verdict": "harmful" if lower > 0 else "pass" if upper <= 0 else "inconclusive"}


def artifact_world_delta(artifact, expected_starts):
    """Read only hashed observable summary copies; never raw scenario/type data."""
    root = Path(artifact)
    manifest = json.loads((root / "pairs/C4_EPISODES_MANIFEST.json").read_text())
    hashes = json.loads((root / "HASHES.json").read_text())
    pairs = {}
    for entry in manifest["episodes"]:
        branch = entry.get("branch", {})
        arm, key = branch.get("kind"), entry.get("pair_key")
        if arm not in ("B", "POLICY") or not key:
            continue
        rel = entry["source"].removeprefix("@artifact/") + "/summary.json"
        path = root / rel
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise ValueError("smoke summary escaped artifact")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != hashes.get(rel):
            raise ValueError("smoke summary hash mismatch")
        value = float(json.loads(data)["ledger"]["J_vehicle_seconds"])
        group = pairs.setdefault(key, {})
        if arm in group:
            raise ValueError("duplicate smoke arm")
        group[arm] = value
    if len(pairs) != expected_starts or any(set(p) != {"B", "POLICY"} for p in pairs.values()):
        raise ValueError("incomplete paired smoke world")
    return sum(p["POLICY"] - p["B"] for p in pairs.values())


def verify_collection_gate(policy):
    """SHA-bind acceptance to the exact checkpoint before any seed charge/dispatch."""
    checkpoint = Path(policy["checkpoint"])
    receipt = json.loads(checkpoint.with_name("native_gate_result.json").read_text())
    if (receipt.get("checkpoint_sha256") != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            or receipt.get("panel_sha256") != PANEL_SHA256 or not receipt.get("passed")):
        raise ValueError("missing/stale/failing native acceptance gate")
    checked = panel_gate(receipt["policy_J"], receipt["native_J"], update=receipt["update"])
    if not checked["passed"] or checked["update"] <= 0:
        raise ValueError("no accepted trained checkpoint; keep native")
    if policy.get('breakdown_required') and 'breakdown' not in receipt:
        raise ValueError('missing required breakdown veto')
    if 'breakdown' in receipt:
        risk = receipt['breakdown']
        delta = np.asarray(risk.get('policy_minus_native'), float)
        if (delta.shape != (16, 3) or not np.isfinite(delta).all() or (delta > 0).any()
                or not risk.get('passed') or not risk.get('calibration_qualified')
                or risk.get('horizons_s') != [60, 120, 300]
                or not risk.get('rule_sha256')
                or risk.get('source_policy_sha256') != receipt['checkpoint_sha256']):
            raise ValueError('missing/failing breakdown native-relative veto')
    return receipt
