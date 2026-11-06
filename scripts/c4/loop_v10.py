#!/usr/bin/env python3
"""Closed Dreamer loop ``v10loop`` (prepared 2026-10-19; owner approval required before launch).

One closed iteration:

  1. collect      current policy (configs/c4_policy_v10entzone_v5m.toml,
                  trainer.entrance_zone_native = true) drives SUMO on gpuR CPUs + cpuN from a
                  fresh training seed block;
  2. convert      raw episodes -> WMS data dir (``wms.data.c4_episodes``);
  3. wm_finetune  continue the previous WM on gpuR with the lane_learnable (c4/v10unblock) and
                  horizon-curriculum + mixed-prioritised-replay (c4/v10horizon) keys passed
                  through VERBATIM; this script never parses or renames them;
  4. wm_check     finite training losses + validation sanity (same rule as dreamer_loop.py);
  5. policy_update  continue the previous policy inside the new WM
                  (``WMS_LOOP_REFRESH=1 train_cli --resume ... --wm-ckpt <new WM>``);
  6. sumo_check   seeds 969824-969835 vs the fixed native reference
                  ``runs/sumo_check_v8pol_u600/native`` with the v7ft pair/coop/WM-fidelity tools;
  7. delay_check  A6: 48 fixed TRAIN-val starts (WM vs SUMO delay level by region at 60/120/300 s)
                  + 48 paired ``genrand_branch`` pairs (WM vs SUMO arm-delta sign agreement,
                  Spearman, MAE, k-sample averaged) on the 48-core CPU box (no GPU);
  8. rollback     if SUMO total D (policy - native) regresses, the paired-delta sign agreement
                  drops > 0.05, or the 120 s delay-level |log ratio| worsens > 0.1 vs the previous
                  iteration, restore the previous WM/policy and halve the next policy-update count
                  (the near-CAV 10/30 s |ds| is reported, non-gating).
                  Every iteration is appended to ``<loop_root>/iterations.jsonl``.

Iteration 0 bootstraps the WM on native/random TRAIN data only
(``configs/c4_wm_v10loop_iter0.toml``) and starts the policy from scratch.  When the iteration-0
WM already exists (``wm.iter0_mode = "existing"``, the v10 launch case: the manual run wrote
``runs/c4_wm_v10_iter000``), the iteration-0 WM stages are skipped and the loop goes straight to
the from-scratch policy update inside that WM.  ``wm.iter0_wm_out`` names that existing run.

Modes (stdlib only):
  python scripts/c4/loop_v10.py plan     --config configs/c4_dreamer_loop_v10loop.json
  python scripts/c4/loop_v10.py dry-run  --config ... [--out-dir DIR]
  python scripts/c4/loop_v10.py status   --config ...
  python scripts/c4/loop_v10.py check-wm --run DIR [--ref-val-log LOG] [--tolerance 0.30]
  python scripts/c4/loop_v10.py parse-pair --native DIR --policy DIR [--out JSON]   (stdin lines)
  python scripts/c4/loop_v10.py run      --config ... --execute --owner-approved
                                         (needs execute_guard.allow AND owner_approved)

Safety: never uses evaluation seeds 964000-964199; the SUMO-check seeds 969824-969835 are never
allocated for training; refuses sealed/holdout/heldout/cal_split/chunk5/j4b/inc1 paths; dry-run
executes no external command.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

EVAL_SEEDS = (964000, 964199)
CHECK_SEEDS = (969824, 969835)
APPROVED_BLOCKS = ((965184, 965999), (966000, 969999))
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
ITER0_TRAIN_STAGES = ("wm_finetune_iter0", "wm_check_iter0", "lane_calib", "policy_update_iter0")
ITER0_EXISTING_STAGES = ("lane_calib", "policy_update_iter0")
ITER0_STAGES = ITER0_TRAIN_STAGES           # canonical (full) iteration-0 stage set
ITER_STAGES = ("collect", "convert", "wm_finetune", "wm_check", "lane_calib", "policy_update",
               "sumo_check", "delay_check", "rollback")
SHIP_STAGES = ("ship_cpuN",)
PAIR_ROWS = ("total", "source_wait", "on_road")
#: V10LANEOPT (2026-10-19): staged mainline lane-change modes of ``physics.mainline_lane_mode``.
LANE_MODES = ("free", "courtesy")
#: V10LANEOPT: own top-level config key ``lane_mode_curriculum`` (disabled = always 'free').
LANE_MODE_CURRICULUM_DEFAULTS = {
    "enabled": False,
    "courtesy_iterations": [1, 2],
    "tolerance": 0.2,
    "gating_heads": ["overtake", "return"],
    "calib_file": "lane_calib.json",
    "iter0_mode": "free",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def q(value) -> str:
    return shlex.quote(str(value))


def refuse_restricted(path) -> str:
    text = str(path)
    if any(token in text.lower() for token in FORBIDDEN):
        raise ValueError(f"restricted path refused: {text}")
    return text


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def _all_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _all_strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _all_strings(value)


def validate_config(cfg: dict) -> None:
    for key in ("loop_root", "data_root", "wm_runs_root", "seed", "hosts", "collect", "wm", "policy",
                "check", "delay_check", "rollback", "budget"):
        if key not in cfg:
            raise ValueError(f"config needs {key}")
    if cfg.get("mode") != "closed":
        raise ValueError("v10loop mode must be 'closed'")
    for name in ("wm", "gpuR", "cpuN"):
        if name not in cfg["hosts"]:
            raise ValueError(f"config needs hosts.{name}")
        for key in ("code_root", "python"):
            if not cfg["hosts"][name].get(key):
                raise ValueError(f"hosts.{name} needs {key}")
    block = cfg["seed"]["block"]
    if len(block) != 2 or int(block[0]) > int(block[1]):
        raise ValueError(f"bad seed block {block!r}")
    lo, hi = int(block[0]), int(block[1])
    if not any(a <= lo and hi <= b for a, b in APPROVED_BLOCKS):
        raise ValueError(f"seed block {lo}-{hi} is not inside an approved training block {APPROVED_BLOCKS}")
    if lo <= EVAL_SEEDS[1] and hi >= EVAL_SEEDS[0]:
        raise ValueError(f"seed block {lo}-{hi} overlaps evaluation seeds {EVAL_SEEDS}")
    if lo <= CHECK_SEEDS[1] and hi >= CHECK_SEEDS[0]:
        raise ValueError(f"seed block {lo}-{hi} overlaps SUMO-check seeds {CHECK_SEEDS}")
    mode = cfg["wm"].get("iter0_mode", "train")
    if mode not in ("existing", "train"):
        raise ValueError(f"wm.iter0_mode must be 'existing' or 'train', not {mode!r}")
    if mode == "existing" and not cfg["wm"].get("iter0_wm_out"):
        raise ValueError("wm.iter0_mode='existing' needs wm.iter0_wm_out")
    for key, value in (("extra_set", cfg["wm"].get("extra_set", [])),
                       ("extra_set_iter0", cfg["wm"].get("extra_set_iter0", [])),
                       ("extra_set_horizon", cfg["wm"].get("extra_set_horizon", [])),
                       ("extra_set_laneflow", cfg["wm"].get("extra_set_laneflow", [])),
                       ("extra_set_agg", cfg["wm"].get("extra_set_agg", []))):
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"wm.{key} must be a list of '--set' strings")
    if not isinstance(cfg["wm"].get("stage_steps"), list) or len(cfg["wm"]["stage_steps"]) < 1:
        raise ValueError("wm.stage_steps must be a non-empty list")
    lane_calib = cfg["wm"].get("lane_calib")
    if lane_calib is not None:
        if not isinstance(lane_calib, dict):
            raise ValueError("wm.lane_calib must be an object")
        for key in ("n_starts", "workers", "rounds"):
            if key in lane_calib and int(lane_calib[key]) < 1:
                raise ValueError(f"wm.lane_calib.{key} must be >= 1")
        for key in ("n_native", "n_genrand"):
            if key in lane_calib and int(lane_calib[key]) < 0:
                raise ValueError(f"wm.lane_calib.{key} must be >= 0")
    dc = cfg["delay_check"]
    dc_host = dc.get("host", "cpuN")
    if dc_host not in cfg["hosts"]:
        raise ValueError(f"delay_check.host {dc_host!r} is not a configured host")
    if not isinstance(dc.get("horizons", "60,120,300"), (str, list, tuple)):
        raise ValueError("delay_check.horizons must be a string or list")
    for key in ("sumo_total_D_max_worsen", "pair_sign_drop_max", "delay_log_ratio_worsen_max",
                "halve_factor", "min_policy_updates"):
        if key not in cfg["rollback"]:
            raise ValueError(f"rollback needs {key}")
    if not os.path.isdir(cfg["hosts"]["wm"]["code_root"]) and "REPO_SHA" not in cfg["hosts"]["wm"]["code_root"]:
        raise ValueError(f"hosts.wm.code_root does not exist: {cfg['hosts']['wm']['code_root']}")
    cur = cfg.get("lane_mode_curriculum")
    if cur is not None and not isinstance(cur, dict):
        raise ValueError("lane_mode_curriculum must be a table")
    if isinstance(cur, dict):
        if "enabled" in cur and not isinstance(cur["enabled"], bool):
            raise ValueError("lane_mode_curriculum.enabled must be a boolean")
        if "courtesy_iterations" in cur and not isinstance(cur["courtesy_iterations"], list):
            raise ValueError("lane_mode_curriculum.courtesy_iterations must be a list")
        if "gating_heads" in cur and not all(isinstance(h, str) for h in cur["gating_heads"]):
            raise ValueError("lane_mode_curriculum.gating_heads must be a list of head names")
        if "iter0_mode" in cur and cur["iter0_mode"] not in LANE_MODES:
            raise ValueError(f"lane_mode_curriculum.iter0_mode must be one of {LANE_MODES}")
        if "tolerance" in cur and not (isinstance(cur["tolerance"], (int, float))
                                       and math.isfinite(float(cur["tolerance"]))
                                       and float(cur["tolerance"]) >= 0.0):
            raise ValueError("lane_mode_curriculum.tolerance must be finite and >= 0")
    guard = cfg.get("execute_guard", {})
    if guard.get("require_owner_approval") and "owner_approved" not in guard:
        raise ValueError("execute_guard.require_owner_approval needs execute_guard.owner_approved")
    for text in _all_strings(cfg):
        refuse_restricted(text)


# --------------------------------------------------------------------------- lane-mode curriculum
def lane_mode_curriculum(cfg: dict) -> dict:
    """The ``lane_mode_curriculum`` key merged over :data:`LANE_MODE_CURRICULUM_DEFAULTS`."""
    cur = dict(LANE_MODE_CURRICULUM_DEFAULTS)
    cur.update(cfg.get("lane_mode_curriculum", {}) or {})
    return cur


def lane_calib_path(paths: dict, cfg: dict) -> Path:
    """Path of the lane_calib JSON (written by the c4/v10lanecal stage) of the WM in use.

    The calibration that gates the switch concerns the WM the next iteration fine-tunes from,
    so it sits next to that WM's checkpoint (``paths['wm_ckpt']``).
    """
    return Path(str(paths["wm_ckpt"])).parent / str(lane_mode_curriculum(cfg)["calib_file"])


def read_lane_calib(path) -> dict | None:
    """Read the lane_calib JSON, or ``None`` when absent/unreadable/malformed (robust)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def _head_rel_error(record) -> float | None:
    """Signed WM-vs-SUMO relative error of one lane_calib head (any of the documented fields)."""
    if not isinstance(record, dict):
        return None
    rel = record.get("rel_error")
    if isinstance(rel, (int, float)) and math.isfinite(float(rel)):
        return float(rel)
    ratio = record.get("ratio")
    if isinstance(ratio, (int, float)) and math.isfinite(float(ratio)):
        return float(ratio) - 1.0
    wm, sumo = record.get("wm"), record.get("sumo")
    if (isinstance(wm, (int, float)) and isinstance(sumo, (int, float))
            and math.isfinite(float(wm)) and math.isfinite(float(sumo)) and float(sumo) != 0.0):
        return float(wm) / float(sumo) - 1.0
    return None


def lane_calib_verdict(calib, curriculum: dict) -> dict:
    """Within tolerance on every gating head?  Missing/malformed calibration -> False."""
    tol = float(curriculum["tolerance"])
    heads = list(curriculum["gating_heads"])
    if not isinstance(calib, dict):
        return {"calibrated": False, "tolerance": tol, "heads": {}, "reason": "lane_calib missing"}
    table = calib.get("heads")
    if not isinstance(table, dict):
        flag = calib.get("within_tolerance")
        calibrated = flag is True
        return {"calibrated": calibrated, "tolerance": tol, "heads": {},
                "reason": ("top-level within_tolerance" if isinstance(flag, bool)
                           else "lane_calib has no heads")}
    per, ok = {}, True
    for head in heads:
        rel = _head_rel_error(table.get(head))
        within = rel is not None and abs(rel) <= tol + 1e-9
        ok = ok and within
        per[head] = {"rel_error": rel, "within": within}
    return {"calibrated": ok, "tolerance": tol, "heads": per,
            "reason": "within tolerance" if ok else "outside tolerance or missing heads"}


def lane_mode_for_iteration(cfg: dict, k: int, calib=None) -> dict:
    """``physics.mainline_lane_mode`` of iteration ``k`` (courtesy stage, then calibrated free).

    Iterations in ``courtesy_iterations`` are forced to 'courtesy'; later iterations switch to
    'free' only when ``calib`` (the lane_calib JSON of the WM being fine-tuned) is within
    tolerance on the gating heads.  A missing/malformed calibration keeps 'courtesy' (robust).
    """
    cur = lane_mode_curriculum(cfg)
    if not cur["enabled"]:
        return {"iteration": k, "mode": "free", "reason": "curriculum disabled", "calib": None,
                "curriculum": cur}
    if k == 0:
        mode = cur["iter0_mode"] if cur["iter0_mode"] in LANE_MODES else "free"
        return {"iteration": k, "mode": mode, "reason": "iteration 0", "calib": None,
                "curriculum": cur}
    if k in [int(x) for x in cur["courtesy_iterations"]]:
        return {"iteration": k, "mode": "courtesy", "reason": f"courtesy iteration {k}",
                "calib": None, "curriculum": cur}
    verdict = lane_calib_verdict(calib, cur)
    return {"iteration": k, "mode": "free" if verdict["calibrated"] else "courtesy",
            "reason": f"post-courtesy: {verdict['reason']}", "calib": verdict, "curriculum": cur}


# --------------------------------------------------------------------------- seeds
def ranges(seeds):
    out = []
    for seed in sorted(seeds):
        if out and seed == out[-1][1] + 1:
            out[-1][1] = seed
        else:
            out.append([seed, seed])
    return out


def seed_spec(seeds) -> str:
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges(seeds))


def expand(rs) -> set:
    return {s for a, b in rs for s in range(int(a), int(b) + 1)}


def check_seed(seed: int) -> None:
    if type(seed) is not int:
        raise ValueError(f"seed {seed!r} is not an integer")
    if EVAL_SEEDS[0] <= seed <= EVAL_SEEDS[1]:
        raise ValueError(f"seed {seed} is an evaluation seed")
    if CHECK_SEEDS[0] <= seed <= CHECK_SEEDS[1]:
        raise ValueError(f"seed {seed} is reserved for the SUMO check")
    if not any(a <= seed <= b for a, b in APPROVED_BLOCKS):
        raise ValueError(f"seed {seed} is outside the approved training blocks")


def allocate_seeds(cfg: dict, used=(), n: int = 1) -> list:
    lo, hi = (int(x) for x in cfg["seed"]["block"])
    blocked = expand(cfg["seed"].get("exclude", [])) | set(used)
    blocked |= set(range(EVAL_SEEDS[0], EVAL_SEEDS[1] + 1))
    blocked |= set(range(CHECK_SEEDS[0], CHECK_SEEDS[1] + 1))
    seeds = [s for s in range(lo, hi + 1) if s not in blocked][:n]
    if len(seeds) < n:
        raise ValueError(f"seed block {lo}-{hi} has only {len(seeds)} free seeds, need {n}")
    for seed in seeds:
        check_seed(seed)
    return seeds


# --------------------------------------------------------------------------- budgets / counters
def budget(cfg: dict, iteration: int) -> dict:
    collect = cfg["collect"]
    if iteration == 0:
        episodes = 0
        sim_s = 0.0
        policy_sim_s = 0.0
        check_episodes = 0
    else:
        episodes = sum(int(collect["batch_size"][h]) for h in collect["hosts"])
        sim_s = float(episodes * (int(collect["warmup"]) + int(collect["horizon"])))
        policy_sim_s = float(episodes * int(collect["horizon"]))
        check_episodes = CHECK_SEEDS[1] - CHECK_SEEDS[0] + 1
    ck = cfg.get("check", {})
    check_warmup = int(ck.get("warmup", collect["warmup"]))
    check_horizon = int(ck.get("horizon", collect["horizon"]))
    if iteration == 0:
        check_episodes = 0
    wm_steps = sum(int(x) for x in cfg["wm"]["stage_steps"])
    b = cfg["budget"]
    wm_gpu_h = wm_steps * float(b["gpu_hours_per_wm_step"])
    policy_gpu_h = int(cfg["policy"]["updates"]) * float(b["gpu_hours_per_policy_update"])
    check_gpu_h = check_episodes * float(b["gpu_hours_per_check_episode"])
    return {
        "collect_episodes": episodes,
        "check_episodes": check_episodes,
        "sim_s": sim_s,
        "policy_sim_s": policy_sim_s,
        "check_sim_s": float(check_episodes * (check_warmup + check_horizon)),
        "wm_steps": wm_steps,
        "policy_updates": int(cfg["policy"]["updates"]),
        "gpu_hours_wm": round(wm_gpu_h, 3),
        "gpu_hours_policy": round(policy_gpu_h, 3),
        "gpu_hours_check": round(check_gpu_h, 4),
        "gpu_hours_total": round(wm_gpu_h + policy_gpu_h + check_gpu_h, 3),
    }


def sample_counter(records) -> dict:
    episodes = sum(int(r.get("collect", {}).get("episodes", 0)) for r in records)
    check_episodes = sum(int(r.get("check", {}).get("episodes", 0)) for r in records)
    sim_s = sum(float(r.get("collect", {}).get("sim_s", 0.0)) for r in records)
    dt = 0.5
    for row in records:
        dt = float(row.get("budget", {}).get("mappo_control_dt_s", dt))
    return {
        "train_episodes": episodes,
        "check_episodes": check_episodes,
        "policy_sim_s": round(sim_s, 1),
        "control_samples": int(round(sim_s / dt)) if dt else 0,
        "mappo_control_dt_s": dt,
        "note": ("MAPPO fairness counter: control_samples is the number of policy control steps "
                 "seen in SUMO; train MAPPO on the same cumulative count."),
    }


def _rollback_metrics(doc: dict) -> dict:
    """The delay-check JSON carries its rollback metrics under ``rollback``; legacy dicts are flat."""
    block = doc.get("rollback") if isinstance(doc, dict) else None
    return block if isinstance(block, dict) else (doc or {})


def decide_rollback(prev: dict, cur: dict, rule: dict, policy_updates: int) -> dict:
    """Revert on a SUMO D regression, a paired-delta sign-agreement drop or a 120 s delay-level
    ``|log ratio|`` worsening (A6).  The near-CAV 10/30 s |ds| is carried in the decision for
    reporting only, never gating.
    """
    prev, cur = _rollback_metrics(prev), _rollback_metrics(cur)
    reasons = []
    worsen = float(cur.get("sumo_total_D_delta", 0.0) or 0.0)
    if worsen > float(rule["sumo_total_D_max_worsen"]):
        reasons.append(f"SUMO total D (policy - native) worsened by {worsen:+.4f}")
    prev_sign, cur_sign = prev.get("pair_sign_agreement"), cur.get("pair_sign_agreement")
    drop = float(rule["pair_sign_drop_max"])
    if _is_number(prev_sign) and _is_number(cur_sign) and float(cur_sign) < float(prev_sign) - drop:
        reasons.append(f"paired-delta sign agreement dropped to {float(cur_sign):.3f} "
                       f"(from {float(prev_sign):.3f}; > {drop:.2f})")
    prev_ratio, cur_ratio = prev.get("delay_level_log_ratio"), cur.get("delay_level_log_ratio")
    tol = float(rule["delay_log_ratio_worsen_max"])
    if (_is_number(prev_ratio) and _is_number(cur_ratio)
            and abs(float(cur_ratio)) > abs(float(prev_ratio)) + tol):
        reasons.append(f"120 s delay-level |log ratio| worsened to {abs(float(cur_ratio)):.3f} "
                       f"(from {abs(float(prev_ratio)):.3f}; > {tol:.2f})")
    action = "revert" if reasons else "keep"
    next_updates = policy_updates
    if action == "revert":
        next_updates = max(int(rule["min_policy_updates"]),
                           int(policy_updates * float(rule["halve_factor"])))
    return {"action": action, "reasons": reasons, "next_policy_updates": next_updates,
            "reported": {"near_cav_abs_ds_10": cur.get("near_cav_abs_ds_10"),
                         "near_cav_abs_ds_30": cur.get("near_cav_abs_ds_30"),
                         "note": "near-CAV per-vehicle |ds| is reported, non-gating (A6)"},
            "prev": prev, "cur": cur}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


# --------------------------------------------------------------------------- stage planning
def _wm_env(cfg: dict, converted: str | None) -> dict:
    env = {
        "PYTHONPATH": "src:.",
        "OMP_NUM_THREADS": "12",
        "WMS_JAM_WORKERS": "14",
        "WMS_LATEST_EVERY": "100",
    }
    for key in ("WMS_DATA_V5_NATIVE", "WMS_DATA_GENRAND_PLAIN", "WMS_DATA_GENRAND_BRANCH"):
        if key in cfg["policy"].get("env", {}):
            env[key] = cfg["policy"]["env"][key]
    if converted:
        env["WMS_DATA_V6_POLICY"] = converted
    return env


def wm_laneflow_keys(cfg: dict) -> list:
    """FIX 2 (LANEFLOW.md): ``loss.lane_flow`` keys merged into every iteration >= 1 WM run.

    Kept in its own JSON key and helper so the LANEFLOW merge stays separate from the
    c4/v10horizon ``extra_set_horizon`` list; iteration 0 never uses it.
    """
    return list(cfg["wm"].get("extra_set_laneflow", []))


def _wm_cmd(cfg: dict, out: str, init_from: str, config: str, extra: list) -> str:
    stages = "[" + ",".join(str(int(s)) for s in cfg["wm"]["stage_steps"]) + "]"
    argv = [cfg["hosts"]["wm"]["python"], "-m", "wms", "train", "--config", config, "--mode", "run",
            "--out", out, "--set", f"model.init_from={init_from}", "--set",
            f"curriculum.stage_steps={stages}"]
    for item in extra:
        argv += ["--set", item]
    return " ".join(q(a) for a in argv)


def _collect_cmd(cfg: dict, host: str, policy_ckpt: str, seeds: list, out: str) -> str:
    c = cfg["collect"]
    h = cfg["hosts"][host]
    argv = [h["python"], "-m", "traffic_shepherd.c4.collect", "--policy", policy_ckpt,
            "--policy-loader", c.get("policy_loader", "wms.control.policy:load_policy_fn"),
            "--population", c["population"], "--seeds", seed_spec(seeds),
            "--case-plan", c["case_plan"],
            "--penetrations", c["penetrations"], "--pen-weights", c["pen_weights"],
            "--demands", c["demands"], "--warmup", str(c["warmup"]), "--horizon", str(c["horizon"]),
            "--cav-vtype", c.get("cav_vtype", "normal_car"),
            "--workers", str(h.get("workers", 46)), "--no-physics-log", "--out", out]
    return " ".join(q(a) for a in argv)


def _collect_env(cfg: dict, host: str) -> dict:
    c = cfg["collect"]
    root = cfg["hosts"][host]["code_root"]
    loader_root = cfg["hosts"][host].get("entrance_mask_root")
    pythonpath = f"{loader_root}:{root}/src:{root}" if loader_root else f"{root}/src:{root}"
    env = {"PYTHONPATH": pythonpath, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1"}
    if loader_root:
        env["ENTRANCE_NATIVE_MAIN_M"] = str(int(c.get("entrance_native_main_m", 25)))
        env["ENTRANCE_NATIVE_RAMP_M"] = str(int(c.get("entrance_native_ramp_m", 25)))
    return env


def _repo_sha(cfg: dict) -> str:
    text = str(cfg["hosts"]["gpuR"]["code_root"])
    return text.rsplit("repo_", 1)[-1] if "repo_" in text else "REPO_SHA"


def _ship_cmd(cfg: dict, template: str) -> str:
    return template.format(gpuR_code_root=cfg["hosts"]["gpuR"]["code_root"],
                           gpuR_python=cfg["hosts"]["gpuR"]["python"],
                           gpuR_entrance_mask_root=cfg["hosts"]["gpuR"].get(
                               "entrance_mask_root", "<entrance_mask>"),
                           sha=_repo_sha(cfg))


def _convert_cmd(cfg: dict, list_file: str, dest: str) -> str:
    code = ("import sys; from wms.data.c4_episodes import main; "
            "sys.exit(main(['--episodes', *open(sys.argv[1]).read().split(), '--out', sys.argv[2]]))")
    return " ".join(q(a) for a in [cfg["hosts"]["wm"]["python"], "-B", "-c", code, list_file, dest])


def _lane_calib_settings(cfg: dict) -> dict:
    lc = dict(cfg["wm"].get("lane_calib") or {})
    n_starts = int(lc.get("n_starts", 48))
    n_native = int(lc.get("n_native", n_starts // 2))
    n_genrand = int(lc.get("n_genrand", n_starts - n_native))
    env = cfg["policy"].get("env", {})
    native = lc.get("native_dir") or env.get("WMS_DATA_V5_NATIVE") \
        or f"{cfg['data_root']}/v5_native"
    genrand = lc.get("genrand_dir") or env.get("WMS_DATA_GENRAND_PLAIN") \
        or f"{cfg['data_root']}/genrand_plain"
    return {"enabled": bool(lc.get("enabled", True)), "n_starts": n_starts,
            "n_native": n_native, "n_genrand": n_genrand,
            "workers": int(lc.get("workers", 48)), "rounds": int(lc.get("rounds", 6)),
            "horizon_s": float(lc.get("horizon_s", 120.0)),
            "tol": float(lc.get("tol", 0.10)), "damping": float(lc.get("damping", 0.7)),
            "max_step": float(lc.get("max_step", 1.0)),
            "max_offset": float(lc.get("max_offset", 6.0)),
            "heads": ",".join(lc.get("heads", ["overtake", "return"])),
            "offset": int(lc.get("offset", 0)), "t0": float(lc.get("t0", 420.0)),
            "native_dir": native, "genrand_dir": genrand,
            "lane_learnable": bool(lc.get("lane_learnable", True)),
            "out": lc.get("out", "lane_calib.json")}


def lane_calib_stage(cfg: dict, wm_out: str) -> dict | None:
    """LANERATECAL stage: fit the per-head rate offsets in the checkpoint's own directory.

    The policy stage then loads the SAME checkpoint path, so ``load_checkpoint`` picks the
    ``lane_calib.json`` up automatically.  Disabled by ``wm.lane_calib.enabled = false``.
    """
    lc = _lane_calib_settings(cfg)
    if not lc["enabled"]:
        return None
    argv = [cfg["hosts"]["gpuR"]["python"], "tools/lane_rate_calib.py", "fit",
            f"{wm_out}/wm_s_best.pt", "--native-dir", lc["native_dir"],
            "--genrand-dir", lc["genrand_dir"], "--n-native", str(lc["n_native"]),
            "--n-genrand", str(lc["n_genrand"]), "--n-starts", str(lc["n_starts"]),
            "--offset", str(lc["offset"]), "--t0", str(lc["t0"]),
            "--horizon-s", str(lc["horizon_s"]), "--workers", str(lc["workers"]),
            "--device", "cpu", "--rounds", str(lc["rounds"]), "--tol", str(lc["tol"]),
            "--damping", str(lc["damping"]), "--max-step", str(lc["max_step"]),
            "--max-offset", str(lc["max_offset"]), "--heads", lc["heads"],
            "--out-dir", wm_out]
    if lc["lane_learnable"]:
        argv.append("--lane-learnable")
    return {
        "name": "lane_calib",
        "host": "gpuR",
        "cwd": cfg["hosts"]["gpuR"]["code_root"],
        "cmd": " ".join(q(a) for a in argv),
        "env": {"PYTHONPATH": "src:.", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "out": f"{wm_out}/{lc['out']}",
        "note": ("closed-loop per-head lane-rate offsets on TRAIN-val starts; writes "
                 "lane_calib.json next to wm_s_best.pt, which the policy stage then loads"),
    }


def plan_iteration0(cfg: dict, paths: dict, updates: int) -> list:
    if iter0_existing(cfg):
        stages = [stage for stage in [lane_calib_stage(cfg, paths["wm_out"])] if stage]
        stages.append({
            "name": "policy_update_iter0",
            "host": "gpuR",
            "cwd": cfg["hosts"]["gpuR"]["code_root"],
            "cmd": _policy_cmd(cfg, paths["policy_out"], paths["wm_best"], None, updates,
                               iter0=True),
            "env": _policy_env(cfg, None, paths["policy_out"]),
            "out": paths["policy_out"],
            "note": ("policy from scratch inside the EXISTING iteration-0 WM at "
                     f"{paths['wm_out']} (wm.iter0_mode='existing'; WM stage skipped; loads "
                     "lane_calib.json when the calibration stage ran)"),
        })
        return stages
    wm_out = paths["wm_out"]
    mode0 = lane_mode_for_iteration(cfg, 0)["mode"]
    stages = [{
        "name": "wm_finetune_iter0",
        "host": "wm",
        "cwd": cfg["hosts"]["wm"]["code_root"],
        "cmd": _wm_cmd(cfg, wm_out, paths["wm_ckpt"], cfg["wm"]["config_iter0"],
                       [*cfg["wm"].get("extra_set_iter0", []),
                        f"physics.mainline_lane_mode={mode0}"]),
        "env": _wm_env(cfg, None),
        "out": wm_out,
        "note": ("WM v5m_ref fine-tuned with physics.lane_learnable on native/random TRAIN data "
                 f"only (mainline_lane_mode={mode0})"),
    }, {
        "name": "wm_check_iter0",
        "host": "wm",
        "cwd": cfg["hosts"]["wm"]["code_root"],
        "cmd": " ".join(q(a) for a in [cfg["hosts"]["wm"]["python"], "scripts/c4/loop_v10.py",
                                       "check-wm", "--run", wm_out, "--tolerance",
                                       str(cfg["wm"]["val_tolerance"])]),
        "env": {},
        "out": wm_out,
        "note": "finite losses; no reference val log for the first WM",
    }]
    calib = lane_calib_stage(cfg, wm_out)
    if calib:
        stages.append(calib)
    stages.append({
        "name": "policy_update_iter0",
        "host": "gpuR",
        "cwd": cfg["hosts"]["gpuR"]["code_root"],
        "cmd": _policy_cmd(cfg, paths["policy_out"], paths["wm_best"], None, updates,
                           iter0=True),
        "env": _policy_env(cfg, None, paths["policy_out"]),
        "out": paths["policy_out"],
        "note": ("policy from scratch inside the newly trained iteration-0 WM; the WM "
                 "checkpoint carries lane_calib.json from the calibration stage"),
    })
    return stages


def _policy_env(cfg: dict, converted: str | None, out: str) -> dict:
    env = {"PYTHONPATH": "src:.", "WMS_LOOP_REFRESH": "1", **cfg["policy"].get("env", {})}
    if converted:
        env["WMS_DATA_V6_POLICY"] = converted
    return env


def _policy_cmd(cfg: dict, out: str, wm_ckpt: str, resume: str | None, updates: int,
                iter0: bool = False) -> str:
    argv = [cfg["hosts"]["gpuR"]["python"], "-m", "wms.control.train_cli",
            "--config", cfg["policy"]["config_iter0" if iter0 else "config"],
            "--wm-ckpt", wm_ckpt, "--device", cfg["policy"]["device"],
            "--max-updates", str(updates), "--output-dir", out]
    if resume:
        argv += ["--resume", resume]
    return " ".join(q(a) for a in argv)


def plan_iteration(cfg: dict, k: int, paths: dict, seeds: list, updates: int) -> list:
    c = cfg["collect"]
    lane_mode = lane_mode_for_iteration(cfg, k, read_lane_calib(lane_calib_path(paths, cfg)))
    wm_extra = [*cfg["wm"].get("extra_set", []), *cfg["wm"].get("extra_set_horizon", []),
                f"physics.mainline_lane_mode={lane_mode['mode']}"]
    stages = []
    if c.get("ship_commands"):
        stages.append({
            "name": "ship_cpuN",
            "host": "gpuR",
            "cwd": cfg["hosts"]["gpuR"]["code_root"],
            "cmd": " && ".join(_ship_cmd(cfg, t) for t in c["ship_commands"]),
            "env": {},
            "out": cfg["hosts"]["cpuN"]["code_root"],
            "note": "ship repo_<sha>.tgz + entrance_mask to cpuN via gpuR scp; verify the md5 lines",
        })
    for host in c["hosts"]:
        n = int(c["batch_size"][host])
        part = seeds[:n]
        seeds = seeds[n:]
        out = f"{paths['iter_dir']}/collect/{host}"
        stages.append({
            "name": "collect",
            "host": host,
            "cwd": cfg["hosts"][host]["code_root"],
            "cmd": _collect_cmd(cfg, host, paths["policy_ckpt"], part, out),
            "env": _collect_env(cfg, host),
            "out": out,
            "note": (f"batch {len(part)} episodes, seeds {seed_spec(part)}, "
                     "entrance_zone_native + entrance_mask_loader policy"),
        })
    stages += [{
        "name": "convert",
        "host": "wm",
        "cwd": cfg["hosts"]["wm"]["code_root"],
        "cmd": _convert_cmd(cfg, f"{paths['iter_dir']}/convert_episodes.txt", paths["converted"]),
        "env": {"PYTHONPATH": "src:.", "OMP_NUM_THREADS": "4"},
        "out": paths["converted"],
        "note": "all complete raw episodes so far -> WMS data dir",
    }, {
        "name": "wm_finetune",
        "host": "wm",
        "cwd": cfg["hosts"]["wm"]["code_root"],
        "cmd": _wm_cmd(cfg, paths["wm_out"], paths["wm_ckpt"], cfg["wm"]["config"],
                       wm_extra + wm_laneflow_keys(cfg) + list(cfg["wm"].get("extra_set_agg", []))),
        "env": _wm_env(cfg, paths["converted"]),
        "out": paths["wm_out"],
        "note": ("physics.lane_learnable passed through verbatim; extra_set_horizon (c4/v10horizon); "
                 "FIX 2 loss.lane_flow keys in extra_set_laneflow (iterations >= 1); "
                 "A5 agg-delay keys in extra_set_agg; "
                 f"physics.mainline_lane_mode={lane_mode['mode']} ({lane_mode['reason']})"),
    }, {
        "name": "wm_check",
        "host": "wm",
        "cwd": cfg["hosts"]["wm"]["code_root"],
        "cmd": " ".join(q(a) for a in [cfg["hosts"]["wm"]["python"], "scripts/c4/loop_v10.py",
                                       "check-wm", "--run", paths["wm_out"],
                                       "--ref-val-log", paths["wm_val_log"], "--tolerance",
                                       str(cfg["wm"]["val_tolerance"])]),
        "env": {},
        "out": paths["wm_out"],
        "note": "finite losses; val mean_position within (1 + tolerance) x previous WM",
    }]
    calib = lane_calib_stage(cfg, paths["wm_out"])
    if calib:
        stages.append(calib)
    stages.append({
        "name": "policy_update",
        "host": "gpuR",
        "cwd": cfg["hosts"]["gpuR"]["code_root"],
        "cmd": _policy_cmd(cfg, paths["policy_out"], paths["wm_best"],
                           paths["policy_ckpt"], updates),
        "env": _policy_env(cfg, paths["converted"], paths["policy_out"]),
        "out": paths["policy_out"],
        "note": ("continue previous policy inside the new WM (physics.json lane_learnable=true); "
                 "load_checkpoint applies lane_calib.json from the new WM directory"),
    })
    for template in cfg["check"]["commands"]:
        stages.append({
            "name": "sumo_check",
            "host": "gpuR",
            "cwd": cfg["hosts"]["gpuR"]["code_root"],
            "cmd": _fill(template, cfg, paths),
            "env": _check_env(cfg),
            "out": paths["check_dir"],
            "note": "seeds 969824-969835 vs the native reference; pair_check + coop_metrics + WM fidelity",
        })
    stages.append(_delay_check_stage(cfg, paths))
    stages.append({
        "name": "rollback",
        "host": "local",
        "cwd": cfg["hosts"]["gpuR"]["code_root"],
        "cmd": ("python scripts/c4/loop_v10.py decide-rollback --config "
                f"configs/c4_dreamer_loop_v10loop.json --prev {paths['prev_check']} --cur "
                f"{paths['check_dir']}/delay_check.json --json-out {paths['check_dir']}/rollback.json"),
        "env": {},
        "out": paths["check_dir"],
        "note": ("A6 rollback: revert if the paired-delta sign agreement drops > 0.05 or the "
                 "120 s delay-level |log ratio| worsens > 0.1 (near-CAV |ds| reported only)"),
    })
    return stages


def _delay_check_stage(cfg: dict, paths: dict) -> dict:
    """A6 delay-accuracy stage: WM vs SUMO delay level + paired delta on the CPU box."""
    dc = cfg["delay_check"]
    host = dc.get("host", "cpuN")
    check = cfg["check"]
    selection = dc.get("selection", f"{paths['check_dir']}/delay_states_selection.json")
    pairs_selection = dc.get("pairs_selection",
                             f"{paths['check_dir']}/delay_pairs_selection.json")
    argv = [cfg["hosts"][host]["python"], dc.get("tool", "scripts/c4/wm_delay_check.py"),
            "--wm", dc.get("wm") or paths["wm_best"],
            "--states", dc.get("states", check["fixed_val_states"]),
            "--selection", selection, "--save-selection", selection,
            "--pairs-selection", pairs_selection, "--save-pairs-selection", pairs_selection,
            "--horizons", str(dc.get("horizons", "60,120,300")), "--k", str(int(dc.get("k", 8))),
            "--device", str(dc.get("device", "cpu")),
            "--workers", str(int(dc.get("workers", cfg["hosts"][host].get("workers", 44)))),
            "--entries", str(dc.get("entries", "queue")),
            "--sumo-pairs", f"{paths['check_dir']}/pairs.json",
            "--out", f"{paths['check_dir']}/delay_check.json",
            "--md", f"{paths['check_dir']}/delay_check.md"]
    if dc.get("pairs"):
        argv += ["--pairs", dc["pairs"]]
    return {
        "name": "delay_check",
        "host": host,
        "cwd": cfg["hosts"][host]["code_root"],
        "cmd": " ".join(q(a) for a in argv),
        "env": {"PYTHONPATH": "src:.", "OMP_NUM_THREADS": "1"},
        "out": f"{paths['check_dir']}/delay_check.json",
        "note": ("A6: 48 fixed TRAIN-val starts (delay level by region at 60/120/300 s) + 48 "
                 "paired genrand_branch pairs (WM vs SUMO arm delta: sign agreement, Spearman, "
                 "MAE, k-sample averaged); entries=queue uses the recorded entrance backlog. "
                 "The WM checkpoint must already be on this CPU box (delay_check.wm overrides "
                 "paths.wm_best)."),
    }


def _check_env(cfg: dict) -> dict:
    """Entrance-mask loader dir + native zone lengths for the SUMO check collect command."""
    check = cfg["check"]
    loader = check.get("entrance_mask_root", cfg["hosts"]["gpuR"].get("entrance_mask_root", ""))
    pythonpath = f"{loader}:src:." if loader else "src:."
    return {"PYTHONPATH": pythonpath,
            "ENTRANCE_NATIVE_MAIN_M": str(int(check.get("entrance_native_main_m", 25))),
            "ENTRANCE_NATIVE_RAMP_M": str(int(check.get("entrance_native_ramp_m", 25)))}


def _fill(template: str, cfg: dict, paths: dict) -> str:
    check = cfg["check"]
    values = {
        "gpuR_code_root": cfg["hosts"]["gpuR"]["code_root"],
        "gpuR_python": cfg["hosts"]["gpuR"]["python"],
        "gpuR_entrance_mask_root": cfg["hosts"]["gpuR"].get("entrance_mask_root", ""),
        "cpuN_python": cfg["hosts"]["cpuN"]["python"],
        "cpuN_code_root": cfg["hosts"]["cpuN"]["code_root"],
        "policy_ckpt": paths["policy_ckpt"],
        "new_wm": paths["wm_best"],
        "check_dir": paths["check_dir"],
        "native_ref": check["native_ref"],
        "v7ft_run": cfg["v7ft_run"],
        "check_pair_check": check.get("pair_check", f"{cfg['v7ft_run']}/pair_check.py"),
        "check_coop_metrics": check.get("coop_metrics", f"{cfg['v7ft_run']}/coop_metrics.py"),
        "fixed_val_states": check["fixed_val_states"],
    }
    return template.format(**values)


def parse_pair_lines(text: str, native: str = None, policy: str = None) -> dict:
    """Parse ``pair_check`` stdout into JSON.

    ``pair_check.py NATIVE_DIR POLICY_DIR`` prints one line per metric among
    :data:`PAIR_ROWS` (``total`` / ``source_wait`` / ``on_road``).  Each such line carries the
    native and policy values (two numbers) or a single signed value (already a difference);
    ``D = policy - native``.  Tolerant of formatting: any finite numbers on the line are read,
    the first is native and the last is policy.
    """
    import re
    rows = {}
    for line in str(text).splitlines():
        label = next((name for name in PAIR_ROWS if re.search(rf"(?<![a-z_]){name}(?![a-z_])",
                                                               line, re.IGNORECASE)), None)
        if label is None or label in rows:
            continue
        nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]
        if not nums:
            continue
        if len(nums) == 1:
            rows[label] = {"native": None, "policy": None, "policy_minus_native": nums[0]}
        else:
            rows[label] = {"native": nums[0], "policy": nums[-1],
                           "policy_minus_native": nums[-1] - nums[0]}
    total_d = rows.get("total", {}).get("policy_minus_native")
    return {"schema": "c4_pair_check/1", "native": native, "policy": policy, "rows": rows,
            "total_D": {"policy_minus_native": total_d}}


# --------------------------------------------------------------------------- WM sanity
def read_jsonl(path) -> list:
    rows = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def last_val_event(rows) -> dict:
    rows = [r for r in rows if not r.get("revalidated") and "step" in r and "horizon_s" in r]
    if not rows:
        return {}
    last = max(r["step"] for r in rows)
    return {float(r["horizon_s"]): r for r in rows if r["step"] == last}


def wm_sanity(run_dir, ref_val_log=None, metric="mean_position", tolerance=0.30) -> dict:
    out = {"ok": True, "reasons": [], "compare": {}}
    run = Path(run_dir)
    train = read_jsonl(run / "train_log.jsonl")
    if not train:
        out["reasons"].append("empty train_log")
    for row in train:
        value = row.get("loss")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            out["reasons"].append(f"nonfinite loss at step {row.get('step')}")
            break
    new = last_val_event(read_jsonl(run / "val_log.jsonl"))
    if not new:
        out["reasons"].append("no validation rows")
    for h, row in new.items():
        value = row.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            out["reasons"].append(f"nonfinite val {metric} at {h:g}s")
    ref = last_val_event(read_jsonl(ref_val_log)) if ref_val_log else {}
    if not ref:
        out["note"] = "no reference validation; finiteness checks only"
    for h in sorted(set(new) & set(ref)):
        a, b = new[h].get(metric), ref[h].get(metric)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (a, b)) or b <= 0:
            continue
        ratio = a / b
        out["compare"][f"{h:g}"] = {"new": a, "ref": b, "ratio": ratio}
        if ratio > 1.0 + tolerance:
            out["reasons"].append(f"val {metric} at {h:g}s worse by {ratio:.3f}x "
                                  f"(> {1.0 + tolerance:.2f})")
    out["ok"] = not out["reasons"]
    return out


# --------------------------------------------------------------------------- dry run
def _tiny(cfg: dict) -> dict:
    tiny = deepcopy(cfg)
    for host in tiny["collect"]["hosts"]:
        tiny["collect"]["batch_size"][host] = 1
    tiny["collect"]["warmup"] = 5
    tiny["collect"]["horizon"] = 10
    tiny["wm"]["stage_steps"] = [1, 1]
    tiny["policy"]["updates"] = 2
    tiny["iterations"] = 1
    return tiny


def iter0_existing(cfg: dict) -> bool:
    return cfg["wm"].get("iter0_mode", "train") == "existing" and bool(cfg["wm"].get("iter0_wm_out"))


def _prev_wm(k: int, out_dir: Path, cfg: dict) -> str:
    """WM checkpoint the k-th iteration fine-tunes from."""
    if k == 0:
        return cfg["wm"]["init_wm"]
    if k == 1 and iter0_existing(cfg):
        return cfg["wm"]["iter0_wm_out"] + "/wm_s_best.pt"
    return str(out_dir / f"iter_{k-1:03d}" / "wm" / "out" / "wm_s_best.pt")


def _paths(out_dir: Path, k: int, cfg: dict) -> dict:
    it = out_dir / f"iter_{k:03d}"
    wm_out = str(it / "wm" / "out")
    wm_val_log = str(it / "wm" / "out" / "val_log.jsonl")
    if k == 0 and iter0_existing(cfg):
        wm_out = cfg["wm"]["iter0_wm_out"]
        wm_val_log = wm_out + "/val_log.jsonl"
    return {
        "iter_dir": str(it),
        "wm_out": wm_out,
        "wm_best": wm_out + "/wm_s_best.pt",
        "wm_ckpt": _prev_wm(k, out_dir, cfg),
        "wm_val_log": wm_val_log,
        "converted": str(it / "convert" / "out"),
        "policy_out": str(it / "policy" / "out"),
        "policy_ckpt": str(out_dir / f"iter_{k-1:03d}" / "policy" / "out" / "policy_best.pt")
        if k > 0 else "<previous_policy_checkpoint>",
        "check_dir": str(it / "check"),
        # A6 rollback compares this iteration's delay_check.json with the previous iteration's.
        # Iteration 0 runs no delay_check (it is skipped in the existing-WM launch), so the
        # first comparison uses a placeholder the coordinator replaces with the baseline.
        "prev_check": (f"<iter0_delay_check.json>"
                       if k <= 1 else str(out_dir / f"iter_{k-1:03d}" / "check"
                                          / "delay_check.json")),
    }


def render_stage(stage: dict) -> str:
    env = " ".join(f"{k}={q(v)}" for k, v in sorted(stage.get("env", {}).items()))
    prefix = (env + " ") if env else ""
    where = "" if stage["host"] == "local" else f"[{stage['host']}] "
    return f"{where}cd {q(stage['cwd'])} && {prefix}{stage['cmd']}"


def dry_run(cfg: dict, out_dir) -> dict:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    tiny = _tiny(cfg)
    records = []
    for k in (0, 1):
        paths = _paths(root, k, tiny)
        if k == 0:
            stages = plan_iteration0(tiny, paths, tiny["policy"]["updates"])
        else:
            seeds = allocate_seeds(tiny, used=[], n=sum(int(tiny["collect"]["batch_size"][h])
                                                      for h in tiny["collect"]["hosts"]))
            stages = plan_iteration(tiny, k, paths, seeds, tiny["policy"]["updates"])
        for stage in stages:
            records.append({"iteration": k, "stage": stage["name"], "host": stage["host"],
                            "cmd": render_stage(stage), "out": stage["out"], "note": stage["note"]})
    prev = {"sumo_total_D_delta": 0.0, "pair_sign_agreement": 0.70,
            "delay_level_log_ratio": 0.10, "near_cav_abs_ds_10": 3.0}
    cur = {"sumo_total_D_delta": 0.05, "pair_sign_agreement": 0.50,
           "delay_level_log_ratio": 0.30, "near_cav_abs_ds_10": 3.5}
    decision = decide_rollback(prev, cur, tiny["rollback"], tiny["policy"]["updates"])
    samples = sample_counter([{
        "collect": {"episodes": 2, "sim_s": 2.0 * (tiny["collect"]["warmup"] + tiny["collect"]["horizon"]),
                    "horizon_s": tiny["collect"]["horizon"]},
        "budget": {"mappo_control_dt_s": tiny["budget"]["mappo_control_dt_s"]},
    }])
    summary = {
        "mode": "dry-run",
        "created": utcnow(),
        "wire_root": str(root),
        "seed_block": tiny["seed"]["block"],
        "seeds_iteration1": seed_spec(allocate_seeds(tiny, used=[], n=2)),
        "budget_iteration0": budget(tiny, 0),
        "budget_iteration1": budget(tiny, 1),
        "sample_counter": samples,
        "rollback_self_test": decision,
        "stages": [r["stage"] for r in records],
        "stage_count": len(records),
        "executed_external_commands": 0,
    }
    (root / "dry_run.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    (root / "dry_run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True),
                                               encoding="utf-8")
    return {"records": records, "summary": summary}


# --------------------------------------------------------------------------- plan / status
def plan_text(cfg: dict) -> str:
    out = []
    out.append("V10LOOP (closed Dreamer loop) - PREPARED, NOT LAUNCHED")
    out.append(f"  seed block      : {cfg['seed']['block'][0]}-{cfg['seed']['block'][1]} "
               f"(exclude {cfg['seed']['exclude']}; eval {cfg['seed']['eval']}; check {cfg['seed']['check']})")
    if iter0_existing(cfg):
        out.append(f"  iteration 0     : EXISTING WM {cfg['wm']['iter0_wm_out']} "
                   "(WM stage skipped), policy from scratch")
    else:
        out.append(f"  iteration 0     : WM {cfg['wm']['config_iter0']} on native/random TRAIN data, "
                   "then policy from scratch")
    out.append("  iteration k>=1  : ship -> collect -> convert -> wm_finetune -> wm_check -> "
               "policy_update -> sumo_check -> delay_check -> rollback")
    out.append(f"  WM pass-through : {cfg['wm'].get('extra_set', [])} "
               f"(horizon, disabled until c4/v10horizon lands: {cfg['wm'].get('extra_set_horizon', [])}) "
               f"(laneflow, FIX 2 iterations >= 1: {cfg['wm'].get('extra_set_laneflow', [])}) "
               f"(agg-delay: {cfg['wm'].get('extra_set_agg', [])})")
    cur = lane_mode_curriculum(cfg)
    out.append(f"  lane_mode       : enabled={cur['enabled']} courtesy_iterations={cur['courtesy_iterations']} "
               f"tolerance={cur['tolerance']} gating_heads={cur['gating_heads']} "
               f"calib_file={cur['calib_file']} (iteration k mode: "
               + ", ".join(f"{k}={lane_mode_for_iteration(cfg, k)['mode']}"
                           for k in range(int(cfg["iterations"]) + 1)) + ")")
    out.append(f"  policy config   : {cfg['policy']['config']} (entrance_zone_native=true)")
    out.append(f"  SUMO check      : seeds {CHECK_SEEDS[0]}-{CHECK_SEEDS[1]} vs {cfg['check']['native_ref']} "
               f"(demands {cfg['check'].get('demands')}, penetrations {cfg['check'].get('penetrations')})")
    dc = cfg["delay_check"]
    out.append(f"  delay check     : {dc.get('tool', 'scripts/c4/wm_delay_check.py')} on "
               f"{dc.get('host', 'cpuN')} (starts {dc.get('states', '<fixed_val_states>')}, "
               f"pairs {dc.get('pairs', '<none>')}, k={dc.get('k', 8)})")
    out.append(f"  rollback        : {cfg['rollback']}")
    for k in range(int(cfg["iterations"]) + 1):
        paths = _paths(Path("<wire_root>"), k, cfg)
        if k == 0:
            stages = plan_iteration0(cfg, paths, int(cfg["policy"]["updates"]))
        else:
            n = sum(int(cfg["collect"]["batch_size"][h]) for h in cfg["collect"]["hosts"])
            seeds = allocate_seeds(cfg, used=[], n=n)
            stages = plan_iteration(cfg, k, paths, seeds, int(cfg["policy"]["updates"]))
        out.append(f"iteration {k}: " + ", ".join(s["name"] + (":" + s["host"] if s["host"] != "local" else "")
                                                   for s in stages))
    return "\n".join(out)


def status(cfg: dict) -> int:
    root = Path(cfg["loop_root"])
    state = root / "state.json"
    log = root / "iterations.jsonl"
    if not state.exists() and not log.exists():
        print(f"no v10loop state at {root} (prepared, not launched)")
        return 1
    if state.exists():
        print(json.dumps(json.loads(state.read_text(encoding="utf-8")), indent=2, sort_keys=True))
    rows = read_jsonl(log)
    if rows:
        print(f"iterations logged: {len(rows)}")
        print(json.dumps(sample_counter(rows), indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- execution (guarded)
def execute(cfg: dict, runner=None) -> int:
    guard = cfg.get("execute_guard", {})
    if not guard.get("allow"):
        print("REFUSED: execute_guard.allow is false. This worktree is DRY-RUN ONLY; "
              "set it true on gpuR after owner approval and merge of c4/v10unblock + c4/v10horizon.")
        return 3
    if guard.get("require_owner_approval", True) and not guard.get("owner_approved"):
        print("REFUSED: execute_guard.require_owner_approval is true and owner_approved is false; "
              "set execute_guard.owner_approved=true only after the owner approves the launch.")
        return 3
    runner = runner or _run_stage
    root = Path(cfg["loop_root"])
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    step0 = _paths(root, 0, cfg)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "created": utcnow(), "iteration": 0, "wm_ckpt": step0["wm_best"],
        "wm_val_log": step0["wm_out"] + "/val_log.jsonl",
        "iter0_mode": cfg["wm"].get("iter0_mode", "train"),
        "policy_ckpt": None, "policy_updates": int(cfg["policy"]["updates"]),
        "seeds_used": [], "metrics": {},
    }
    for k in range(int(cfg["iterations"]) + 1):
        paths = _paths(root, k, cfg)
        if k == 0:
            stages = plan_iteration0(cfg, paths, int(state["policy_updates"]))
        else:
            n = sum(int(cfg["collect"]["batch_size"][h]) for h in cfg["collect"]["hosts"])
            seeds = allocate_seeds(cfg, used=state.get("seeds_used", []), n=n)
            state["seeds_used"] = sorted(set(state.get("seeds_used", [])) | set(seeds))
            stages = plan_iteration(cfg, k, paths, seeds, int(state["policy_updates"]))
        statuses = []
        for stage in stages:
            rc = runner(stage, cfg)
            statuses.append({"stage": stage["name"], "host": stage["host"], "rc": rc})
            if rc != 0:
                print(f"stage {stage['name']} failed rc={rc}; stopping")
                return 2
        state["iteration"] = k + 1
        state["wm_ckpt"] = paths["wm_best"]
        state["wm_val_log"] = paths["wm_val_log"]
        state["policy_ckpt"] = paths["policy_out"] + "/policy_best.pt"
        row = {"t": utcnow(), "iteration": k, "status": "done", "stages": statuses,
               "budget": budget(cfg, k), "seed_block": cfg["seed"]["block"]}
        with (root / "iterations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def _run_stage(stage: dict, cfg: dict) -> int:
    env = dict(os.environ, **stage.get("env", {}))
    cwd = stage["cwd"]
    if stage["host"] == "local":
        return subprocess.run(stage["cmd"], shell=True, cwd=cwd, env=env).returncode
    ssh = cfg["hosts"][stage["host"]].get("ssh", [stage["host"]])
    prefix = " ".join(f"{k}={q(v)}" for k, v in sorted(stage.get("env", {}).items()))
    remote = f"cd {q(cwd)} && {prefix + ' ' if prefix else ''}{stage['cmd']}"
    return subprocess.run(["ssh", "-o", "BatchMode=yes", *ssh, remote]).returncode


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("plan", "dry-run", "status", "run", "check-wm",
                                       "decide-rollback", "parse-pair"))
    ap.add_argument("--config")
    ap.add_argument("--out-dir")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--owner-approved", action="store_true",
                    help="run only: set execute_guard.owner_approved=true for this invocation")
    ap.add_argument("--iter0-wm", help="run/plan/dry-run: existing iteration-0 WM out dir "
                                       "(overrides wm.iter0_wm_out; sets wm.iter0_mode='existing')")
    ap.add_argument("--run")
    ap.add_argument("--ref-val-log")
    ap.add_argument("--tolerance", type=float, default=0.30)
    ap.add_argument("--prev")
    ap.add_argument("--cur")
    ap.add_argument("--json-out")
    ap.add_argument("--out")
    ap.add_argument("--native")
    ap.add_argument("--policy")
    args = ap.parse_args(argv)

    if args.action == "parse-pair":
        text = sys.stdin.read()
        parsed = parse_pair_lines(text, native=args.native, policy=args.policy)
        body = json.dumps(parsed, indent=2, sort_keys=True)
        if args.out or args.json_out:
            Path(args.out or args.json_out).write_text(body, encoding="utf-8")
        print(body)
        return 0
    if args.action == "check-wm":
        verdict = wm_sanity(args.run, args.ref_val_log, tolerance=args.tolerance)
        text = json.dumps(verdict, indent=2, sort_keys=True)
        print(text)
        if args.json_out:
            Path(args.json_out).write_text(text, encoding="utf-8")
        return 0 if verdict["ok"] else 2
    if args.action == "decide-rollback":
        if not (args.prev and args.cur):
            raise SystemExit("decide-rollback needs --prev and --cur JSON files")
        prev = json.loads(Path(args.prev).read_text(encoding="utf-8"))
        cur = json.loads(Path(args.cur).read_text(encoding="utf-8"))
        rule = {"sumo_total_D_max_worsen": 0.0, "pair_sign_drop_max": 0.05,
                "delay_log_ratio_worsen_max": 0.1, "halve_factor": 0.5,
                "min_policy_updates": 50}
        if args.config:
            rule = dict(load_config(args.config)["rollback"])
        updates = int(_rollback_metrics(cur).get("policy_updates", 200))
        decision = decide_rollback(prev, cur, rule, updates)
        text = json.dumps(decision, indent=2, sort_keys=True)
        print(text)
        if args.json_out:
            Path(args.json_out).write_text(text, encoding="utf-8")
        return 0
    if not args.config:
        raise SystemExit(f"{args.action} needs --config")
    cfg = load_config(args.config)
    if args.iter0_wm:
        cfg["wm"]["iter0_mode"] = "existing"
        cfg["wm"]["iter0_wm_out"] = args.iter0_wm
    if args.action == "plan":
        print(plan_text(cfg))
        return 0
    if args.action == "dry-run":
        out = args.out_dir or str(Path(cfg["loop_root"]) / "dry_run")
        result = dry_run(cfg, out)
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
        print("--- stage commands ---")
        for row in result["records"]:
            print(f"[iter {row['iteration']}] {row['stage']}: {row['cmd']}")
        return 0
    if args.action == "status":
        return status(cfg)
    if args.action == "run":
        if not args.execute:
            print("refused: pass --execute (and set execute_guard.allow=true after approval)")
            return 3
        if args.owner_approved:
            cfg["execute_guard"]["owner_approved"] = True
        return execute(cfg)
    return 1


if __name__ == "__main__":
    sys.exit(main())
