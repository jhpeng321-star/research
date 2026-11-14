"""WM-N1-NOISEFLOOR-20261004 analysis (ANALYSIS ONLY, no simulation).

Reads the 90 existing branch runs (R0/N1/S1 x 30 TRAIN worlds), rebuilds the
v0.4a t==t0 window labels with the exact build_v04a_cov code path
(v0lib.build_run_trace + v0lib.window_label, base trace = the world's __base
run), reads the existing frozen native/pi t0 labels, and computes the
paired-difference noise floor and PI-native spread ratios with a world-cluster
bootstrap.

TRAIN only. Never opens CAL/HOLDOUT/chunk-5/J4B/INC1.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("E:/research-dense/runs/active/WM-N1-NOISEFLOOR-20261004")
HERE = ROOT / "analysis"
LABELCOV = Path("E:/rd-wt/labelcov/output/wd01/partB_v0data")
FROZEN = Path("E:/research-dense/output/wd01_partB_v0data_v04a")
COV = Path("E:/research-dense/output/wd01_partB_v0data_v04a_cov_20261004")
STAGING = Path("D:/rd-staging/wmn1_feat/labels_cov_20261004T072826Z")
TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)
ARMS = ("R0", "N1", "S1")
BOOT_N = 5000
BOOT_SEED = 0

sys.path.insert(0, str(LABELCOV))
sys.path.insert(0, str(LABELCOV.parent))
import v0lib as V  # noqa: E402
import actions_v04 as A  # noqa: E402

T0W = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0W:7.1f}s] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def physics_digest(path: Path) -> tuple[str, int]:
    """sha256 over per-tick sorted (id, lane, pos@6dp, speed@6dp) rows."""
    h = hashlib.sha256()
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            veh = r.get("vehicles", {})
            h.update((";".join(
                f"{vid}|{v.get('lane_id')}|"
                f"{round(float(v.get('lane_position_m', 0.0)), 6)}|"
                f"{round(float(v.get('speed_mps', 0.0)), 6)}"
                for vid, v in sorted(veh.items())) + "\n").encode())
            n += 1
    return h.hexdigest(), n


# ---------------------------------------------------------------- inventory --
def inventory(worlds: list[dict]) -> dict:
    events = [json.loads(ln) for ln in
              (ROOT / "RUNS.jsonl").read_text(encoding="utf-8").splitlines()
              if ln.strip()]
    started = {}
    done = {}
    for e in events:
        run = e.get("run") or f"{e.get('world')}__{e.get('kind')}"
        if e["event"] == "start":
            started[run] = e
        elif e["event"] in ("done", "error", "failed", "fail"):
            done[run] = e
    recs = []
    for w in worlds:
        for arm in ARMS:
            run = f"{w['world']}__{arm}"
            d = ROOT / "runs" / run
            receipt = d / "RUN_RECEIPT.json"
            physics = d / "run" / "physics_steps.jsonl"
            rec = dict(world=w["world"], chunk=w["chunk"], arm=arm, run=run,
                       started=run in started,
                       done=done.get(run, {}).get("event"),
                       exit_code=done.get(run, {}).get("exit_code"),
                       receipt_exists=receipt.is_file(),
                       physics_exists=physics.is_file(),
                       physics_bytes=physics.stat().st_size
                       if physics.is_file() else None)
            if receipt.is_file():
                rj = json.loads(receipt.read_text(encoding="utf-8"))
                rec["exit_status"] = rj.get("exit_status")
                rec["physics_rows_written"] = rj.get("physics_rows_written")
                rec["physics_rows_expected"] = rj.get(
                    "physics_rows_expected")
            recs.append(rec)
    missing = [r for r in recs
               if not (r["receipt_exists"] and r["physics_exists"]
                       and r.get("exit_status") == "complete")]
    return dict(n_runs=len(recs), n_started=len(started), n_done=len(done),
                n_exit0=sum(1 for r in recs if r["exit_code"] == 0),
                missing_or_failed=missing, runs=recs)


# ------------------------------------------------------------------- labels --
def read_existing(world: str, chunk: str, src: str, t0: float) -> dict | None:
    for root in (FROZEN, COV):
        p = root / "worlds" / chunk / f"{world}__{src}_labels.parquet"
        if p.is_file():
            d = pd.read_parquet(p)
            sub = d[(d["t"] == t0) & (d["tau"].isin(TAUS))]
            return {float(r["tau"]): dict(
                total_delay_ref_s=float(r["total_delay_ref_s"]),
                ttc_lt3_veh_s=float(r["ttc_lt3_veh_s"]),
                hard_brake_s=float(r["hard_brake_s"]),
                _root=str(root.name)) for _, r in sub.iterrows()}
    return None


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    wj = json.loads((ROOT / "WORLDS.json").read_text(encoding="utf-8"))
    worlds = wj["worlds"]
    assert all(str(w.get("split", "")).upper() == "TRAIN" for w in worlds)
    inv = inventory(worlds)
    log(f"inventory: {inv['n_runs']} runs, {inv['n_done']} done, "
        f"{len(inv['missing_or_failed'])} missing/failed")

    label_rows = []
    det_rows = []
    existing = {}
    for i, w in enumerate(worlds):
        world, chunk = w["world"], w["chunk"]
        base_dir = Path(w["base_run"])
        native_dir = Path(w["native_run"])
        r0 = ROOT / "runs" / f"{world}__R0"
        t0 = float(json.loads((r0 / "run" / "switch_schedule.json")
                              .read_text(encoding="utf-8"))["schedule"]["t0_s"])
        base_dir = base_dir if (base_dir / "run" / "physics_steps.jsonl").is_file() \
            else base_dir.parent / base_dir.name
        rt_base = V.build_run_trace(world, "base", base_dir, t0)
        t_pre = V.build_t_ins_pre(rt_base, t0)
        arm_labels = {}
        for arm in ARMS:
            arm_dir = ROOT / "runs" / f"{world}__{arm}"
            rt = V.build_run_trace(world, "native", arm_dir, t0,
                                   t_ins_pre=t_pre, base_arm_dir=base_dir)
            tl = A.ActionTimeline.from_run_dir(arm_dir,
                                               schedule_authoritative=True)
            ev = tl.verify_arm("native_fleet")
            for tau in TAUS:
                row = V.window_label(rt, t0, tau, skip_first_tick=False)
                arm_labels[(arm, tau)] = row
                label_rows.append(dict(world=world, chunk=chunk, arm=arm,
                                       tau=tau,
                                       total_delay_ref_s=row[
                                           "total_delay_ref_s"],
                                       ttc_lt3_veh_s=row["ttc_lt3_veh_s"],
                                       hard_brake_s=row["hard_brake_s"]))
        existing[world] = {}
        for src in ("native", "pi"):
            ex = read_existing(world, chunk, src, t0)
            existing[world][src] = ex
            for tau in TAUS:
                if ex and tau in ex:
                    label_rows.append(dict(
                        world=world, chunk=chunk, arm=src.upper(), tau=tau,
                        total_delay_ref_s=ex[tau]["total_delay_ref_s"],
                        ttc_lt3_veh_s=ex[tau]["ttc_lt3_veh_s"],
                        hard_brake_s=ex[tau]["hard_brake_s"]))
        # determinism: R0 vs original native (labels + normalized physics)
        d = dict(world=world, chunk=chunk, t0=t0,
                 arm_verify=bool(ev.get("ok")))
        maxlab = 0.0
        for tau in TAUS:
            r0v = arm_labels[("R0", tau)]
            natv = existing[world]["native"].get(tau) if existing[world]["native"] else None
            if natv:
                for k in ("total_delay_ref_s", "ttc_lt3_veh_s", "hard_brake_s"):
                    maxlab = max(maxlab, abs(r0v[k] - natv[k]))
        d["r0_vs_native_label_max_absdiff"] = maxlab
        d["r0_labels_exact"] = bool(maxlab == 0.0)
        rp = r0 / "run" / "physics_steps.jsonl"
        npth = native_dir / "run" / "physics_steps.jsonl"
        d["r0_physics_raw_sha"] = sha256_file(rp)
        d["native_physics_raw_sha"] = sha256_file(npth)
        d["s1_physics_raw_sha"] = sha256_file(
            ROOT / "runs" / f"{world}__S1" / "run" / "physics_steps.jsonl")
        d["r0_equals_s1_raw"] = d["r0_physics_raw_sha"] == d["s1_physics_raw_sha"]
        dig0, n0 = physics_digest(rp)
        dign, nn = physics_digest(npth)
        d["r0_physics_digest"] = dig0
        d["native_physics_digest"] = dign
        d["r0_physics_rows"] = n0
        d["native_physics_rows"] = nn
        d["r0_vs_native_physics_exact"] = bool(dig0 == dign)
        det_rows.append(d)
        log(f"[{i+1}/{len(worlds)}] {world} R0exact_lab="
            f"{d['r0_labels_exact']} R0exact_phys={d['r0_vs_native_physics_exact']}")

    pd.DataFrame(label_rows).to_csv(HERE / "labels.csv", index=False)
    (HERE / "inventory.json").write_text(
        json.dumps(inv, indent=2), encoding="utf-8")
    (HERE / "determinism.json").write_text(
        json.dumps(det_rows, indent=2), encoding="utf-8")

    lab = pd.DataFrame(label_rows)
    piv = {}
    for arm in ARMS + ("NATIVE", "PI"):
        sub = lab[lab.arm == arm]
        piv[arm] = {(r["world"], float(r["tau"])): r
                    for _, r in sub.iterrows()}
    worlds_ok = [w["world"] for w in worlds]

    def series(arm, key, tau):
        return np.array([piv[arm][(w, tau)][key] for w in worlds_ok],
                        dtype=float)

    def stats(dy):
        sd = float(np.std(dy, ddof=1))
        return dict(mean=float(np.mean(dy)), sd=sd,
                    median_abs=float(np.median(np.abs(dy))),
                    frac_pos=float(np.mean(dy > 0.0)))

    def boot_ratio(num, den, rng):
        n = len(num)
        out = np.empty(BOOT_N)
        for b in range(BOOT_N):
            idx = rng.integers(0, n, n)
            ds = float(np.std(den[idx], ddof=1))
            out[b] = np.inf if ds == 0.0 else float(np.std(num[idx], ddof=1)) / ds
        finite = out[np.isfinite(out)]
        return out, finite

    analysis = {"taus": list(TAUS), "n_worlds": len(worlds_ok),
                "primary": "delay_s (=total_delay_ref_s)",
                "secondary": "hard_brake_s", "per_tau": {}}
    rng = np.random.default_rng(BOOT_SEED)
    for tau in TAUS:
        taus = float(tau)
        pi = series("PI", "total_delay_ref_s", tau)
        nat = series("NATIVE", "total_delay_ref_s", tau)
        n1 = series("N1", "total_delay_ref_s", tau)
        s1 = series("S1", "total_delay_ref_s", tau)
        r0 = series("R0", "total_delay_ref_s", tau)
        contrast = {}
        for name, dy in (("N1-native", n1 - nat), ("S1-native", s1 - nat),
                         ("R0-native", r0 - nat), ("PI-native", pi - nat)):
            contrast[name] = stats(dy)
        ratio_n1 = None
        ratio_s1 = None
        if contrast["N1-native"]["sd"] > 0:
            _, fin = boot_ratio(pi - nat, n1 - nat, rng)
            ratio_n1 = dict(
                point=contrast["PI-native"]["sd"]
                / contrast["N1-native"]["sd"],
                ci90=[float(np.percentile(fin, 5)),
                      float(np.percentile(fin, 95))] if len(fin) else None,
                n_finite=int(len(fin)),
                clears_noise=bool(len(fin) and
                                  np.percentile(fin, 5) > 1.0))
        if contrast["S1-native"]["sd"] > 0:
            _, fin = boot_ratio(pi - nat, s1 - nat, rng)
            ratio_s1 = dict(
                point=contrast["PI-native"]["sd"]
                / contrast["S1-native"]["sd"],
                ci90=[float(np.percentile(fin, 5)),
                      float(np.percentile(fin, 95))] if len(fin) else None,
                n_finite=int(len(fin)),
                clears_noise=bool(len(fin) and
                                  np.percentile(fin, 5) > 1.0))
        # secondary hard brake
        pi_h = series("PI", "hard_brake_s", tau)
        nat_h = series("NATIVE", "hard_brake_s", tau)
        n1_h = series("N1", "hard_brake_s", tau)
        s1_h = series("S1", "hard_brake_s", tau)
        sec = {}
        for name, dy in (("N1-native", n1_h - nat_h),
                         ("S1-native", s1_h - nat_h),
                         ("PI-native", pi_h - nat_h)):
            sec[name] = stats(dy)
        analysis["per_tau"][str(tau)] = dict(
            delay_contrasts=contrast, sd_ratio_PI_over_N1=ratio_n1,
            sd_ratio_PI_over_S1=ratio_s1, hard_brake_contrasts=sec)
    (HERE / "analysis.json").write_text(json.dumps(analysis, indent=2),
                                        encoding="utf-8")
    log("wrote labels.csv / inventory.json / determinism.json / analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
