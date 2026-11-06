"""N7 sanity evaluation of the wz3to2 speed-only MAPPO (descriptive; NOT the paper comparison).

    python tools/nm_mappo/sanity_eval.py --config configs/c4_mappo_speed_wz3to2.toml \
        --policy u0=<policy_00169.pt> --policy wz=<run>/policy_latest.pt \
        --out <dir> --registered <SANITY_REGISTERED.json> [--workers 12]
    python tools/nm_mappo/sanity_eval.py ... --fake            # CPU fake env, no SUMO (tests)
    python tools/nm_mappo/sanity_eval.py ... --print-panel     # panel + quote only, no SUMO

Arms per world (identical seed / demand / penetration / 300 s native warmup / 600 s window):
``native`` (all CAVs released to SUMO: all-NaN command), ``u0`` (frozen MAPPO u0 transferred
as-is) and ``wz`` (the wz3to2-trained MAPPO), plus optional extra ``name=path`` arms.  Policy
arms run the DEPLOYMENT executor (``PolicyExecutor``, deterministic mean action, ``lanes=None``
asserted for speed-only actors).  Panel: the TRAINING cells of the config (penetrations x
demands after map resolution; canonical D3200_p60 for S-MAPPOWZ) x the LAST
``ceil(12 / n_cells)`` TRAIN seeds (989588-989599 for one cell); never a HELD-OUT seed (those
are reserved for the final paired evaluation).  ``--skip-native`` + ``--prior-rows`` run a
later arm (the trained actor) on the same panel and summarise it with the earlier rows.

Metrics per arm-world: window sums of the trainer's per-step team-reward components
(``delay``, ``source_wait``, ``total_wait`` = delay + source wait, ``hard_brake``,
``cost_ttc``; veh-s over the 600 s window), collisions, teleports, lane commands issued,
dropped vehicle steps (slot overflow), max present vehicles.  Paired differences
``arm - native`` and ``wz - u0`` (treatment - reference; NEGATIVE = improvement), world
percentile bootstrap (B = 2000, NumPy default_rng(0)), 95% CI.

Pre-registered sanity flags (MAPPO_WZ_PLAN.md section 4): S1 safety (0 collisions/teleports,
0 lane commands for speed-only arms, 0 dropped vehicle steps), S2 finite metrics, S3 the wz
arm is not significantly worse than native (CI lower bound of wz - native total_wait <= 0).
A failed S3 does NOT trigger checkpoint shopping: the comparator rule is fixed in the plan.
SUMO mode requires a registration receipt whose ``panel_sha256`` matches this panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

U0_SHA256 = "d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca"
WARMUP_S, HORIZON_S = 300.0, 600.0
METRICS = ("delay", "source_wait", "total_wait", "hard_brake", "cost_ttc", "reward")
BOOT_B, BOOT_SEED = 2000, 0


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_cfg(path):
    from wms.core.toml import loads
    return loads(Path(path).read_text(encoding="utf-8"))


def resolve(cfg: dict, *, fake: bool):
    """(env spec, binding, training cells) for the eval env; the stub is refused unless ``fake``."""
    from wms.control.mappo.mapbind import apply_map
    cfg = json.loads(json.dumps(cfg))
    if fake:
        cfg.setdefault("env", {})["name"] = "fake"
        cfg["env"]["kwargs"] = {}
    resolved, binding = apply_map(cfg)
    env = resolved["env"]
    spec = {"env_name": env["name"], "env_kwargs": dict(env.get("kwargs", {}))}
    roll = resolved["rollout"]
    cells = [(float(p), int(d)) for p in roll["penetrations"] for d in roll["demands_vph"]]
    return spec, binding, cells


def make_panel(binding, cells=None, *, min_worlds: int = 12, warmup_s: float = WARMUP_S,
               horizon_s: float = HORIZON_S) -> list[dict]:
    cells = cells or [(float(p), int(d)) for p in binding.penetrations for d in binding.demands_vph]
    per_cell = max(1, math.ceil(min_worlds / len(cells)))
    seeds = sorted(binding.train_seeds)[-per_cell:]
    held = set(binding.heldout_seeds)
    panel = [{"world": i, "seed": int(s), "demand_vph": d, "penetration": p, "map": binding.name,
              "warmup_s": float(warmup_s), "horizon_s": float(horizon_s)}
             for i, ((p, d), s) in enumerate((c, s) for c in cells for s in seeds)]
    assert not any(w["seed"] in held for w in panel)
    binding.check_seeds([w["seed"] for w in panel])
    return panel


def panel_sha256(panel) -> str:
    return hashlib.sha256(json.dumps(panel, sort_keys=True).encode()).hexdigest()


def load_actor(path):
    """``PolicyExecutor`` (deterministic) from a MAPPO export or trainer checkpoint."""
    from wms.control.lanes import LaneRules
    from wms.control.mappo.agent import build_networks
    from wms.control.policy import PolicyExecutor
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    lane_control = payload.get("lane_control") or (
        "native" if payload.get("lane_head") is False else "learned")
    policy, encoder, _ = build_networks(lane_control=lane_control)
    policy.load_state_dict(payload["policy"], strict=True)
    group = {k[len("encoder."):] if k.startswith("encoder.") else k: v
             for k, v in payload["group_encoder"].items()}
    encoder.load_state_dict(group, strict=True)
    policy.eval()
    encoder.eval()
    rules = payload.get("lane_rules") or (payload.get("config") or {}).get("lane_rules")
    return PolicyExecutor(policy, encoder, lane_rules=LaneRules.from_mapping(rules),
                          deterministic=True), lane_control


def run_arm(job: dict) -> dict:
    """One arm-world episode; never raises (failure rows are kept, no retry)."""
    from wms.control.contract import RewardWeights
    from wms.control.mappo.rollout import build_env, control_step_reward
    world, arm = job["world"], job["arm"]
    row = {"world": world["world"], "seed": world["seed"], "demand_vph": world["demand_vph"],
           "penetration": world["penetration"], "arm": arm, "policy": job.get("policy"),
           "policy_sha256": job.get("policy_sha256")}
    env = None
    try:
        env = build_env(job["spec"])
        exe, lane_control = (None, "native") if arm == "native" else load_actor(job["policy"])
        state = env.reset(seed=world["seed"], demand_vph=world["demand_vph"],
                          penetration=world["penetration"], warmup_s=world["warmup_s"],
                          horizon_s=world["horizon_s"])
        if exe is not None:
            exe.reset()
        weights = RewardWeights(**job.get("reward_weights", {}))
        sums = {k: 0.0 for k in METRICS}
        lane_cmd, steps, info = 0, 0, {}
        for _ in range(int(round(world["horizon_s"]))):
            if exe is None:
                accel, lanes = np.full(tuple(state["s_m"].shape), np.nan), None
            else:
                accel, lanes = exe(state, with_lanes=True)
                if lane_control == "native" and lanes is not None:
                    raise RuntimeError("speed-only actor produced lane commands")
            next_state, info = env.step(accel, lanes=lanes)
            comp = control_step_reward(env, state, next_state, weights, info)
            for k in METRICS:
                sums[k] += float(comp.get(k, 0.0))
            lane_cmd += int((info.get("lane_commands") or {}).get("issued", 0))
            steps += 1
            state = next_state
            if info.get("done") or info.get("terminated"):
                break
        summary = info.get("summary") or {}
        row.update(ok=True, steps=steps, lane_commands_issued=lane_cmd,
                   collisions=summary.get("collisions"), teleports=summary.get("teleports"),
                   dropped_vehicle_steps=summary.get("dropped_vehicle_steps", 0),
                   max_present_vehicles=summary.get("max_present_vehicles"),
                   sim_s=world["warmup_s"] + steps, **sums)
    except Exception as exc:  # noqa: BLE001 - recorded, never retried
        row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
    return row


def bootstrap_ci(diffs, *, b: int = BOOT_B, seed: int = BOOT_SEED):
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(b, diffs.size))
    means = diffs[idx].mean(1)
    return {"mean": float(diffs.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": int(diffs.size)}


def summarize(rows: list[dict], arms: list[str]) -> dict:
    ok = [r for r in rows if r.get("ok")]
    by = {(r["world"], r["arm"]): r for r in ok}
    worlds = sorted({r["world"] for r in rows})
    complete = [w for w in worlds if all((w, a) in by for a in arms)]
    paired = {}
    pairs = [(a, "native") for a in arms if a != "native" and "native" in arms]
    if "wz" in arms and "u0" in arms:
        pairs.append(("wz", "u0"))
    for treat, ref in pairs:
        paired[f"{treat}-{ref}"] = {m: bootstrap_ci([by[(w, treat)][m] - by[(w, ref)][m] for w in complete])
                                    for m in ("total_wait", "delay", "source_wait", "hard_brake")}
    speed_only = [r for r in ok if r["arm"] != "native"]
    s1 = (all((r.get("collisions") or 0) == 0 and (r.get("teleports") or 0) == 0
              and (r.get("dropped_vehicle_steps") or 0) == 0 for r in ok)
          and all(r.get("lane_commands_issued", 0) == 0 for r in speed_only))
    s2 = all(math.isfinite(r[m]) for r in ok for m in METRICS)
    wz_native = paired.get("wz-native", {}).get("total_wait")
    s3 = None if wz_native is None else bool(wz_native["lo"] <= 0.0)
    return {"arms": arms, "worlds": len(worlds), "complete_worlds": len(complete),
            "failed_rows": len(rows) - len(ok), "paired": paired,
            "flags": {"S1_safety": s1, "S2_finite": s2, "S3_wz_not_worse_than_native": s3},
            "sign": "treatment - reference; negative = improvement",
            "sim_s": float(sum(r.get("sim_s", 0.0) for r in ok))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="N7 wz3to2 MAPPO sanity evaluation")
    ap.add_argument("--config", default=str(ROOT / "configs/c4_mappo_speed_wz3to2.toml"))
    ap.add_argument("--policy", action="append", default=[], metavar="ARM=PATH")
    ap.add_argument("--out", required=False)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--registered", default=None, help="registration receipt (SUMO mode)")
    ap.add_argument("--fake", action="store_true", help="CPU fake env (no SUMO)")
    ap.add_argument("--horizon-s", type=float, default=HORIZON_S)
    ap.add_argument("--warmup-s", type=float, default=WARMUP_S)
    ap.add_argument("--print-panel", action="store_true")
    ap.add_argument("--allow-any-u0", action="store_true", help="tests only: skip the u0 sha check")
    ap.add_argument("--skip-native", action="store_true", help="native arm already run (--prior-rows)")
    ap.add_argument("--prior-rows", default=None, help="rows.jsonl of an earlier run on the same panel")
    ap.add_argument("--min-worlds", type=int, default=12)
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config)
    spec, binding, cells = resolve(cfg, fake=a.fake)
    panel = make_panel(binding, cells, min_worlds=a.min_worlds, warmup_s=a.warmup_s,
                       horizon_s=a.horizon_s)
    policies = dict(p.split("=", 1) for p in a.policy)
    arms = ([] if a.skip_native else ["native"]) + list(policies)
    plan = {"panel": panel, "panel_sha256": panel_sha256(panel), "arms": arms,
            "binding": binding.record(), "fake": a.fake,
            "planned_sim_s": len(panel) * len(arms) * (a.warmup_s + a.horizon_s)}
    if a.print_panel:
        print(json.dumps(plan, indent=1))
        return 0
    if not a.out:
        ap.error("--out is required")
    hashes = {arm: sha256_file(path) for arm, path in policies.items()}
    if "u0" in hashes and hashes["u0"] != U0_SHA256 and not a.allow_any_u0:
        raise SystemExit(f"u0 policy sha256 {hashes['u0']} != registered {U0_SHA256}")
    if not a.fake:
        if not a.registered:
            raise SystemExit("SUMO mode requires --registered (panel/budget registration receipt)")
        receipt = json.loads(Path(a.registered).read_text())
        if receipt.get("panel_sha256") != plan["panel_sha256"]:
            raise SystemExit("registration receipt panel_sha256 does not match this panel")
        if float(receipt.get("budget_sim_s", 0)) < plan["planned_sim_s"]:
            raise SystemExit("registered budget below the planned sim-s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "PLAN.json").write_text(json.dumps({**plan, "policy_sha256": hashes}, indent=1))
    weights = dict(cfg.get("reward_weights", {}))
    jobs = [{"world": w, "arm": arm, "spec": spec, "reward_weights": weights,
             "policy": policies.get(arm), "policy_sha256": hashes.get(arm)}
            for w in panel for arm in arms]
    if a.workers > 1:
        import multiprocessing
        with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            rows = list(pool.map(run_arm, jobs))
    else:
        rows = [run_arm(job) for job in jobs]
    with (out / "rows.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    all_rows, all_arms = list(rows), list(arms)
    if a.prior_rows:
        prior = [json.loads(x) for x in Path(a.prior_rows).read_text().splitlines() if x.strip()]
        keys = {(w["world"], w["seed"], w["demand_vph"], w["penetration"]) for w in panel}
        if {(r["world"], r["seed"], r["demand_vph"], r["penetration"]) for r in prior} - keys:
            raise SystemExit("prior rows are not on this panel")
        all_rows = prior + rows
        names = list(dict.fromkeys([r["arm"] for r in prior] + arms))
        all_arms = (["native"] if "native" in names else []) + [x for x in names if x != "native"]
    summary = summarize(all_rows, all_arms)
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"flags": summary["flags"], "complete_worlds": summary["complete_worlds"],
                      "failed_rows": summary["failed_rows"], "sim_s": summary["sim_s"]}))
    return 0 if summary["failed_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
