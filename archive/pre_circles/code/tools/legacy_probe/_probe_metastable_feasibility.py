"""Metastable-control feasibility probe driver (design spec 2026-08-12).

Stage 1 (loop gain):  no_op | meter_burst | pace | meter_aggressive
    - does the CAV's actuator move h at all on the 2700 scene?

Stage 2 (sufficiency): no_op | rule_full | rule_no_floor | rule_aggressive
    - can the h-table controller hold the upper branch without hurting TTC
      relative to champion B (iter_95)?

Reuses the _eval_hybrid_meta_v1 harness pattern: shared no-op cache, per-seed
spawn timeout, row/summary writing — so every number is directly comparable
to B's existing runs/eval summaries.

Usage:
  python -u _probe_metastable_feasibility.py --stage 1 --seeds 105 --workers 1 --steps 600
  python -u _probe_metastable_feasibility.py --stage 2 --scene 2700 --workers 1
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

SEED_TIMEOUT_S = int(os.environ.get("EVAL_SEED_TIMEOUT", "900"))

ROOT = Path(r"D:\research-dense")
SUMO = Path(r"D:\research\highway_merge_shepherd")
SUMO_LONG = Path(r"D:\research\highway_merge_shepherd_long")
PEN = 0.2
SUMO_DT = 0.5
SEEDS = [115, 112, 114, 105, 119, 110, 126, 149]
PRE = "1191554636"
RAMP = "11553048#1"
SLACK_L = 5.0
SLACK_G = 5.0

DENSITIES = {
    1400: "light_l1400",
    1700: "light_l1700",
    2000: "light_l2000",
    2400: "light_m2400r400",
    2500: "light_m2500r400",
    2600: "light_m2600r400",
    2700: "light_m2700r400",
    2800: "light_m2800r400",
    2900: "light_m2900r400",
}

# stage -> variant -> (policy_kind, params)
STAGE1_VARIANTS = {
    "no_op": ("no_op", {}),
    "meter_burst": ("scripted", {"mode": "fair_balance", "dial": 0.6,
                                 "on_steps": 10, "off_steps": 5, "trigger_h": 0.4}),
    "pace": ("scripted", {"mode": "protect", "dial": 0.2, "trigger_h": None}),
    "meter_aggressive": ("scripted", {"mode": "fair_balance", "dial": 1.0,
                                      "on_steps": 10, "off_steps": 5, "trigger_h": 0.4}),
}
STAGE2_VARIANTS = {
    "no_op": ("no_op", {}),
    "rule_full": ("rule", {"ablation": "full"}),
    "rule_no_floor": ("rule", {"ablation": "no_floor"}),
    "rule_aggressive": ("rule", {"ablation": "aggressive"}),
}
# Stage 3 (2026-08-13): PREVENTIVE injection control — metering acts while
# the merge is still on the upper branch (rho >= rho_set) and on the
# injection velocity delta (v_ramp >= v_c(rho) - margin), with the fairness
# gate (never meter a ramp already worse off) in every variant.
STAGE3_VARIANTS = {
    "no_op": ("no_op", {}),
    "prev_rho": ("preventive", {"levers": ("rho",), "fairness_gate": True}),
    "prev_dv": ("preventive", {"levers": ("dv",), "fairness_gate": True}),
    "prev_full": ("preventive", {"levers": ("rho", "dv"), "fairness_gate": True}),
}
# Stage 4 (2026-08-13): sweep for a mechanism that WORKS — earlier
# setpoints (act before the ramp queue forms), corridor-TTC suppression
# (never meter into an already near-missing corridor), and gentle MAINLINE
# damping (the stage-1 all-CAV burst — the only thing that moved h at 2700 —
# in a designed, floor-guarded form).
# Stage 4 (2026-08-13): sweep for a mechanism that WORKS — earlier
# setpoints (act before the ramp queue forms), corridor-TTC suppression
# (never meter into an already near-missing corridor), and gentle MAINLINE
# damping (the stage-1 all-CAV burst — the only thing that moved h at 2700 —
# in a designed, floor-guarded form).
STAGE4_VARIANTS = {
    "no_op": ("no_op", {}),
    "prev_full": ("preventive", {"levers": ("rho", "dv"), "fairness_gate": True}),
    "prev_rho35": ("preventive", {"levers": ("rho", "dv"), "fairness_gate": True,
                                  "cfg": {"rho_set": 3.5}}),
    "prev_rho30": ("preventive", {"levers": ("rho", "dv"), "fairness_gate": True,
                                  "cfg": {"rho_set": 3.0}}),
    "prev_cttc": ("preventive", {"levers": ("rho", "dv"), "fairness_gate": True,
                                 "cfg": {"corridor_ttc_block": True}}),
    "maindamp": ("preventive", {"levers": ("rho",), "fairness_gate": False,
                                "meter_side": "mainline",
                                "cfg": {"rho_set": 3.5, "meter_dial_lo": 0.15,
                                        "meter_dial_hi": 0.3}}),
    "maindamp_cttc": ("preventive", {"levers": ("rho",), "fairness_gate": False,
                                     "meter_side": "mainline",
                                     "cfg": {"rho_set": 3.5, "meter_dial_lo": 0.15,
                                             "meter_dial_hi": 0.3,
                                             "corridor_ttc_block": True}}),
}
# Stage 5 (2026-08-13): refine the stage-4 winner (gentle MAINLINE damping
# + corridor-TTC suppression): dial strengths, trigger setpoints, and
# combined mainline+ramp metering.
_MD = {"meter_side": "mainline", "levers": ("rho",), "fairness_gate": False}
STAGE5_VARIANTS = {
    "no_op": ("no_op", {}),
    "maindamp": ("preventive", {**_MD, "cfg": {"rho_set": 3.5, "meter_dial_lo": 0.15,
                                               "meter_dial_hi": 0.3}}),
    "maindamp2": ("preventive", {**_MD, "cfg": {"rho_set": 3.5, "meter_dial_lo": 0.2,
                                                "meter_dial_hi": 0.5}}),
    "maindamp2_cttc": ("preventive", {**_MD, "cfg": {"rho_set": 3.5, "meter_dial_lo": 0.2,
                                                     "meter_dial_hi": 0.5,
                                                     "corridor_ttc_block": True}}),
    "maindamp_lo30": ("preventive", {**_MD, "cfg": {"rho_set": 3.0, "meter_dial_lo": 0.15,
                                                    "meter_dial_hi": 0.3}}),
    "maindamp_cttc": ("preventive", {**_MD, "cfg": {"rho_set": 3.5, "meter_dial_lo": 0.15,
                                                    "meter_dial_hi": 0.3,
                                                    "corridor_ttc_block": True}}),
    "bothdamp": ("preventive", {"meter_side": "both", "levers": ("rho",),
                                "fairness_gate": False,
                                "cfg": {"rho_set": 3.5, "meter_dial_lo": 0.15,
                                        "meter_dial_hi": 0.3}}),
    "bothdamp_cttc": ("preventive", {"meter_side": "both", "levers": ("rho",),
                                     "fairness_gate": False,
                                     "cfg": {"rho_set": 3.5, "meter_dial_lo": 0.15,
                                             "meter_dial_hi": 0.3,
                                             "corridor_ttc_block": True}}),
}

# Stage 6 (2026-08-13): the "really upstream" hypothesis — platoon-breaking
# VSL: trigger on the UPSTREAM pre-edge mean speed (the platoon is detected
# where it forms, 20-60 s before it reaches the merge) with a meaningful
# fixed speed-cut dial, plus the always-on gentle harmonization.
STAGE6_VARIANTS = {
    "no_op": ("no_op", {}),
    "maindamp": ("preventive", {**_MD, "cfg": {"rho_set": 3.5, "meter_dial_lo": 0.15,
                                               "meter_dial_hi": 0.3}}),
    "vsl_18": ("preventive", {**_MD, "trigger_src": "pre_v",
                              "cfg": {"v_trig": 18.0, "meter_dial_fixed": 0.4,
                                      "rho_set": 3.5}}),
    "vsl_20": ("preventive", {**_MD, "trigger_src": "pre_v",
                              "cfg": {"v_trig": 20.0, "meter_dial_fixed": 0.4,
                                      "rho_set": 3.5}}),
    "vsl_18_cttc": ("preventive", {**_MD, "trigger_src": "pre_v",
                                   "cfg": {"v_trig": 18.0, "meter_dial_fixed": 0.4,
                                           "rho_set": 3.5,
                                           "corridor_ttc_block": True}}),
    "always": ("preventive", {**_MD, "cfg": {"rho_set": 0.0,
                                             "meter_dial_fixed": 0.15}}),
    "always_cttc": ("preventive", {**_MD, "cfg": {"rho_set": 0.0,
                                                  "meter_dial_fixed": 0.15,
                                                  "corridor_ttc_block": True}}),
}

# Stage 7 (2026-08-14): the inverted-dv probes.
#   merge_pace  — ramp merge-speed matching: pace the ramp approach down to
#                 the merge-zone mean speed (shrinks the inverted |dv|; the
#                 dv_mean diagnostic should drop).
#   hold_stable — heavy ramp metering at a LOW density setpoint: keep the
#                 merge zone well below critical with the queue upstream
#                 (capacity-drop logic). Measures the stability-vs-queue
#                 trade-off: h should rise, loss/G should grow, episodes may
#                 gridlock (queue spill) — steps reports it.
STAGE7_VARIANTS = {
    "no_op": ("no_op", {}),
    "merge_pace": ("preventive", {"levers": ("rho",), "meter_side": "ramp",
                                 "fairness_gate": False, "trigger_src": "match_v",
                                 "cfg": {"rho_set": 3.5, "meter_dial_fixed": 0.3,
                                         "match_v_margin": 3.0}}),
    "hold_stable": ("preventive", {"levers": ("rho",), "meter_side": "ramp",
                                   "fairness_gate": False,
                                   "cfg": {"rho_set": 3.0, "meter_dial_fixed": 0.8}}),
    "hold_stable_main": ("preventive", {"levers": ("rho",), "meter_side": "mainline",
                                        "fairness_gate": False,
                                        "cfg": {"rho_set": 3.0, "meter_dial_fixed": 0.8}}),
}


def _env() -> None:
    os.environ.setdefault("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
    os.environ["PATH"] = os.environ["SUMO_HOME"] + r"\bin;" + os.environ.get("PATH", "")
    os.environ["PYTHONPATH"] = str(ROOT) + ";" + os.environ.get("PATH", "")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _stream_loss(tripinfo: Path) -> dict:
    all_tt, all_loss, main_l, ramp_l = [], [], [], []
    for t in ET.parse(tripinfo).getroot().findall("tripinfo"):
        edge = t.get("departLane", "").rsplit("_", 1)[0]
        dur = float(t.get("duration", 0))
        loss = float(t.get("timeLoss", 0))
        all_tt.append(dur)
        all_loss.append(loss)
        if edge == RAMP:
            ramp_l.append(loss)
        elif edge == PRE:
            main_l.append(loss)
    return {
        "tt": mean(all_tt) if all_tt else None,
        "loss": mean(all_loss) if all_loss else None,
        "loss_main": mean(main_l) if main_l else None,
        "loss_ramp": mean(ramp_l) if ramp_l else None,
        "fair_gap": abs(mean(main_l) - mean(ramp_l)) if main_l and ramp_l else None,
    }


def _ttc_events(metrics: dict) -> int:
    ttc = metrics.get("ttc")
    if isinstance(ttc, dict):
        return int(ttc.get("total_events") or 0)
    return int(ttc or 0)


def _health_metrics(history: list) -> dict:
    """h-mean / upper-frac / collapse metrics from recorded (h, rho, v, n_veh)."""
    hs = [h for h, _, _, _ in history if h is not None]
    if not hs:
        return {"h_mean": None, "upper_frac": None, "first_collapse_step": None,
                "sustained_collapse": False, "density_at_onset": None}
    upper = sum(1 for h in hs if h >= 0.6) / len(hs)
    collapse = next((i for i, h in enumerate(hs) if h < 0.2), None)
    # Sustained collapse (Stage 8, recalibrated 2026-08-14): an ACTIVE
    # breakdown = h < 0.15 for >= 60 CONSECUTIVE steps AFTER the first 100
    # steps. Calibrated on the measured h series: the healthy-but-oscillating
    # 2400-mid cell's longest below-0.15 run is 24 steps (short recoverable
    # excursions — the metastable picture), while the collapsing 2700-short
    # cell locks at 145 steps. 60 cleanly separates locking from oscillation
    # (the first two versions — 10 and 30 steps — fired on everything).
    # density_at_onset = the on-road vehicle count at the FIRST step of that
    # run — the literal "how many cars the highway was carrying when it
    # collapsed" (the user's claim).
    run = 0
    onset_i = None
    for i, (h, _, _, _) in enumerate(history):
        if i < 100:
            continue
        run = run + 1 if (h is not None and h < 0.15) else 0
        if run >= 60:
            onset_i = i - 59
            break
    density_at_onset = None
    if onset_i is not None and onset_i < len(history):
        density_at_onset = history[onset_i][3]
    return {"h_mean": float(mean(hs)), "upper_frac": upper,
            "first_collapse_step": collapse,
            "sustained_collapse": onset_i is not None,
            "density_at_onset": density_at_onset}


def _build_policy(kind: str, params: dict):
    """Return (HealthRecordingWrapper(policy), policy_name)."""
    from policies.baselines import NoOpPolicy
    from policies.metastable_rule import (
        HealthRecordingWrapper,
        MetastableRulePolicy,
        PreventiveInjectionPolicy,
        ScriptedInterventionPolicy,
    )
    from config.constants import MODE_NAMES

    if kind == "no_op":
        inner = NoOpPolicy()
        name = "no_op"
    elif kind == "scripted":
        mode = MODE_NAMES.index(params["mode"])
        inner = ScriptedInterventionPolicy(
            mode, params["dial"],
            on_steps=params.get("on_steps", 10),
            off_steps=params.get("off_steps", 5),
            trigger_h=params.get("trigger_h", 0.4),
        )
        name = "shepherd"
    elif kind == "preventive":
        inner = PreventiveInjectionPolicy(
            levers=params.get("levers", ("rho", "dv")),
            fairness_gate=bool(params.get("fairness_gate", True)),
            meter_side=params.get("meter_side", "ramp"),
            trigger_src=params.get("trigger_src", "merge"),
            cfg_overrides=params.get("cfg"),
        )
        name = "shepherd"
    else:  # rule
        inner = MetastableRulePolicy(ablation=params.get("ablation", "full"))
        name = "shepherd"
    return HealthRecordingWrapper(inner), name


def _matches_window(row: dict, steps: int) -> bool:
    """Cached-row window check: rows must match the requested step cap.

    ``window`` records the requested cap (0 = full episode). Rows written by
    the old eval driver lack ``window`` — they ran full episodes, so they
    match full-episode requests only.
    """
    want = 0 if steps <= 0 else steps
    return row.get("window", 0) == want


def run_variant(variant: str, seed: int, out_dir: Path, noop_cache: Path,
                tag: str, steps: int, pen: float = 0.2,
                sroot: Path = SUMO, scope: str = "") -> dict:
    _env()
    from evaluate_policy import run_policy_evaluation

    # no_op is a pure-HDV baseline, independent of the variant: cache it in a
    # SHARED directory so every stage/variant eval reuses the same runs. The
    # no-op baseline may depend on penetration (CAV vtype dynamics), so
    # non-default penetrations get a scoped cache dir.
    if variant == "no_op":
        out_path = noop_cache / f"no_op.seed{seed}.json"
    else:
        out_path = out_dir / f"{variant}.seed{seed}.json"
    if out_path.exists():
        row = json.loads(out_path.read_text(encoding="utf-8"))
        if "status" in row or row.get("schema") != 2:
            # error/timeout rows and pre-schema-2 rows are never valid cache
            # — re-run (the print must not assume a status key exists)
            print(f"  stale cache {out_path.name} "
                  f"(status={row.get('status', 'pre-schema')}) — re-running", flush=True)
            out_path.unlink()
        elif _matches_window(row, steps):
            return row
        else:
            print(f"  stale cache {out_path.name} (window={row.get('window', 0)}, want {0 if steps <= 0 else steps}) "
                  f"— re-running", flush=True)
            out_path.unlink()

    t0 = time.time()
    ALL_VARIANTS = {**STAGE1_VARIANTS, **STAGE2_VARIANTS, **STAGE3_VARIANTS,
                    **STAGE4_VARIANTS, **STAGE5_VARIANTS, **STAGE6_VARIANTS,
                    **STAGE7_VARIANTS}
    spec = ALL_VARIANTS.get(variant)
    if spec is None:
        # Strict: an unknown variant must never silently fall back to no_op
        # (that bug made stage-3 rows bit-identical no-op copies, 2026-08-13).
        raise ValueError(f"Unknown variant: {variant}")
    policy, policy_name = _build_policy(*spec)
    env_overrides = {
        "simulation_step": 1.0,
        "sumo_step_length": SUMO_DT,
        "sumo_config": f"_light/osm.dense.{tag}.sumocfg",
        "shepherd_disabled_modes": ["vacate_drop"],
    }
    if policy_name == "shepherd":
        env_overrides["control_mode"] = "shepherd_hybrid"

    metrics = run_policy_evaluation(
        policy,
        density="dense",
        project_root=sroot,
        seed=seed,
        max_episode_steps=None if steps <= 0 else steps,
        sumo_binary="sumo",
        cav_penetration=pen,
        policy_name=policy_name,
        env_overrides=env_overrides,
        run_tag_prefix=f"metafeas_{scope}_{tag}_{variant}_{seed}",
    )
    stream = _stream_loss(Path(metrics["tripinfo_path"]))
    # Process evidence: mean injection velocity delta recorded by the
    # wrapper for EVERY variant (incl. no_op).
    dv = None
    if getattr(policy, "dv_history", None):
        dv = mean(policy.dv_history)
    row = {
        "schema": 2,
        "variant": variant, "seed": seed, "tag": tag,
        "window": 0 if steps <= 0 else steps,
        "pen": pen,
        "tt": stream["tt"], "loss": stream["loss"],
        "loss_main": stream["loss_main"], "loss_ramp": stream["loss_ramp"],
        "fair_gap": stream["fair_gap"],
        "collisions": int(metrics.get("collisions") or 0),
        "ttc_events": _ttc_events(metrics),
        "steps": int(metrics.get("steps", 0)),
        "dv_mean": dv,
        "elapsed_s": round(time.time() - t0, 1),
        **_health_metrics(policy.history),
    }
    out_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    hm = row["h_mean"] if row["h_mean"] is not None else float("nan")
    up = row["upper_frac"] if row["upper_frac"] is not None else float("nan")
    print(f"[{variant} seed{seed} @{tag}] loss={row['loss']:.2f} "
          f"G={row['fair_gap']:.2f} ttc={row['ttc_events']} "
          f"h_mean={hm:.3f} upper={up:.2f} coll={row['first_collapse_step']} "
          f"({row['elapsed_s']}s)", flush=True)
    return row


def _job_entry(q: mp.Queue, variant: str, seed: int, out_dir: Path,
               noop_cache: Path, tag: str, steps: int, pen: float,
               sroot: Path, scope: str) -> None:
    try:
        q.put(run_variant(variant, seed, out_dir, noop_cache, tag, steps, pen,
                          sroot, scope))
    except Exception as exc:  # noqa: BLE001 - report and fail the job, keep going
        print(f"FAIL {variant} seed{seed}: {exc}", flush=True)
        q.put({"variant": variant, "seed": seed, "status": "error", "error": str(exc)})


def _run_job_with_timeout(variant: str, seed: int, out_dir: Path,
                          noop_cache: Path, tag: str, steps: int,
                          pen: float = 0.2, sroot: Path = SUMO,
                          scope: str = "",
                          timeout: int = SEED_TIMEOUT_S) -> dict:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(
        target=_job_entry,
        args=(q, variant, seed, out_dir, noop_cache, tag, steps, pen, sroot,
              scope),
        daemon=True,
    )
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        print(f"TIMEOUT {variant} seed{seed} after {timeout}s — killed", flush=True)
        return {"variant": variant, "seed": seed, "status": "timeout"}
    try:
        row = q.get_nowait()
    except Exception:  # noqa: BLE001 - no result posted
        row = None
    if row is None:
        return {"variant": variant, "seed": seed, "status": "error"}
    return row


def _print_health_table(rows: dict, variants: list, seeds: list, noop: dict) -> None:
    print("\n--- health vs no_op ---", flush=True)
    for v in variants:
        if v == "no_op":
            continue
        d_upper, d_h, shifts = [], [], []
        for s in seeds:
            r = rows.get((v, s))
            n = noop.get(s)
            if not r or not n or "status" in r or "status" in n:
                continue
            if r.get("upper_frac") is not None and n.get("upper_frac") is not None:
                d_upper.append(r["upper_frac"] - n["upper_frac"])
            if r.get("h_mean") is not None and n.get("h_mean") is not None:
                d_h.append(r["h_mean"] - n["h_mean"])
            if (r.get("first_collapse_step") is not None
                    and n.get("first_collapse_step") is not None):
                shifts.append(r["first_collapse_step"] - n["first_collapse_step"])
        du = mean(d_upper) if d_upper else float("nan")
        dh = mean(d_h) if d_h else float("nan")
        print(f"{v:16s}: dUpper={du:+.3f} dH={dh:+.3f} collapse_shifts={shifts}", flush=True)


def _print_win_table(rows: dict, variants: list, seeds: list, noop: dict) -> None:
    print("\n--- win rule vs no_op (TTC<=noop, G<=noop+5, L<=noop+5) ---", flush=True)
    for v in variants:
        if v == "no_op":
            continue
        per = []
        for s in seeds:
            r, n = rows.get((v, s)), noop.get(s)
            if not r or not n or "status" in r or "status" in n:
                continue
            ttc_ok = r["ttc_events"] <= n["ttc_events"]
            g_ok = (r["fair_gap"] or 0) <= (n["fair_gap"] or 0) + SLACK_G
            l_ok = (r["loss"] or 0) <= (n["loss"] or 0) + SLACK_L
            per.append({"seed": s, "win": ttc_ok and g_ok and l_ok,
                        "d_ttc": r["ttc_events"] - n["ttc_events"]})
        if not per:
            continue
        wins = sum(1 for x in per if x["win"])
        dttc = mean(x["d_ttc"] for x in per)
        detail = " ".join(f"{x['seed']}:{x['d_ttc']:+d}" for x in per)
        print(f"{v:16s}: wins {wins}/{len(per)}  dTTC={dttc:+.1f}  {detail}", flush=True)


def _print_ttc_vs_b(rows: dict, seeds: list, tag: str,
                    variants: tuple = ("rule_full", "rule_no_floor", "rule_aggressive")) -> None:
    """PASS check: per-seed TTC(variant) <= TTC(B iter_95) for this scene."""
    # B evals used two dir-naming eras: "hybrid_meta_{exp}_{ckpt}_{tag}" (new)
    # and "hybrid_meta_{ckpt}_{tag}" (legacy). Prefer the new, fall back.
    cand = [
        ROOT / "runs" / "eval" / f"hybrid_meta_v1_iter_95_{tag}" / "summary.json",
        ROOT / "runs" / "eval" / f"hybrid_meta_iter_95_{tag}" / "summary.json",
    ]
    b_summary = next((p for p in cand if p.exists()), None)
    if b_summary is None:
        print(f"\n(B summary not present for {tag} — skipping TTC-vs-B check)", flush=True)
        return
    b = json.loads(b_summary.read_text(encoding="utf-8"))
    print(f"\n--- TTC vs B (iter_95 @{tag}, PASS if variant <= B per seed) ---", flush=True)
    for v in variants:
        line = []
        for s in seeds:
            r = rows.get((v, s))
            bk = b.get(f"always_active_{s}")
            if not r or "status" in r or not bk:
                continue
            b_ttc = int(bk.get("ttc_events") or 0)
            ok = r["ttc_events"] <= b_ttc
            line.append(f"{s}:{'OK' if ok else 'NO'} ({r['ttc_events']} vs {b_ttc})")
        print(f"{v:16s}: " + " ".join(line), flush=True)


def main() -> None:
    _env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--scene", type=int, default=2700, choices=sorted(DENSITIES))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("EVAL_WORKERS", "1")))
    ap.add_argument("--steps", type=int, default=0,
                    help="Episode length cap in controller steps (0 = full episode, "
                    "like the B evals; pass --steps 600 for Stage-1 runs)")
    ap.add_argument("--pen", type=float, default=PEN,
                    help="CAV penetration for this run (0.3 = 30%%)")
    ap.add_argument("--long", action="store_true",
                    help="Alias for --map-root highway_merge_shepherd_long.")
    ap.add_argument("--map-root", default=None,
                    help="Run on an arbitrary map root (e.g. "
                    "D:/research/highway_merge_shepherd_mid for the 1844 m "
                    "map); the no-op cache/output dirs get the map's tag.")
    ap.add_argument("--variants", default="")
    args = ap.parse_args()

    variants = (args.variants.split(",") if args.variants
                else list({1: STAGE1_VARIANTS, 2: STAGE2_VARIANTS,
                           3: STAGE3_VARIANTS, 4: STAGE4_VARIANTS,
                           5: STAGE5_VARIANTS,
                           6: STAGE6_VARIANTS,
                           7: STAGE7_VARIANTS}[args.stage]))
    if "no_op" not in variants:
        # the baseline is mandatory for the win/health tables
        variants = ["no_op"] + list(variants)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or list(SEEDS)
    tag = DENSITIES[args.scene]
    if args.map_root:
        sroot = Path(args.map_root)
    else:
        sroot = SUMO_LONG if args.long else SUMO
    scope = ""
    if abs(args.pen - 0.2) > 1e-9:
        scope = f"_pen{int(round(args.pen * 100))}"
    if sroot != SUMO:
        scope += sroot.name.replace("highway_merge_shepherd", "")
    out = ROOT / "runs" / "probe" / "metastable_feasibility" / f"stage{args.stage}_{tag}{scope}"
    out.mkdir(parents=True, exist_ok=True)
    noop_cache = ROOT / "runs" / "eval" / f"noop_cache_{tag}{scope}"
    noop_cache.mkdir(parents=True, exist_ok=True)

    jobs = [(v, s) for v in variants for s in seeds]
    rows: dict = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_run_job_with_timeout, v, s, out, noop_cache, tag, args.steps,
                      args.pen, sroot, scope): (v, s)
            for v, s in jobs
        }
        for fut in as_completed(futs):
            v, s = futs[fut]
            rows[(v, s)] = fut.result()

    noop = {s: rows[("no_op", s)] for s in seeds if ("no_op", s) in rows}
    _print_win_table(rows, variants, seeds, noop)
    _print_health_table(rows, variants, seeds, noop)
    if args.stage in (2, 3, 4, 5, 6, 7):
        if args.stage == 2:
            vnames = ("rule_full", "rule_no_floor", "rule_aggressive")
        elif args.stage == 3:
            vnames = ("prev_rho", "prev_dv", "prev_full")
        elif args.stage == 4:
            vnames = ("prev_full", "prev_rho35", "prev_rho30",
                      "prev_cttc", "maindamp", "maindamp_cttc")
        elif args.stage == 5:
            vnames = ("maindamp", "maindamp2", "maindamp2_cttc",
                      "maindamp_lo30", "maindamp_cttc", "bothdamp",
                      "bothdamp_cttc")
        elif args.stage == 6:
            vnames = ("maindamp", "vsl_18", "vsl_20", "vsl_18_cttc",
                      "always", "always_cttc")
        else:
            vnames = ("merge_pace", "hold_stable")
        _print_ttc_vs_b(rows, seeds, tag, variants=vnames)

    (out / "summary.json").write_text(
        json.dumps({f"{v}_{s}": rows[(v, s)] for v, s in rows}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
