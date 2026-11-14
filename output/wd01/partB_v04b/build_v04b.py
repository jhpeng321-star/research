"""v0.4b t0-row build (registration s.2/s.8): features + labels for the
pair table, t0-row-only, ZERO simulation.

Per world (TRAIN for the fit/dry-run; CAL only inside the guarded
cal-read):
  * native + aux rows are taken from the v0.4a root
    (wd01_partB_v0data_v04a/worlds/chunk4/<world>__<source>_*) at
    t == t0: the deployable strict/detector blocks, the v0.4a per-CAV
    action token counts, the pi summary and the window labels (the
    v0.4a t0 convention already matches branch rows; build_v04a.py);
  * wall rows are built FROM THE RAW J4 RUNS: the label per registered
    tau from v0lib.window_label over the branch trace (restored at t0,
    pre-t0 insertion history merged from the shared base run), the
    request-time wall descriptor from wall_tokens_v04 (events <= t0
    only; FORBIDDEN_RECORD_FIELDS never read), the action counts from
    actions_v04 (schedule-authoritative) and the PLANNED-SCHEDULE
    descriptor from the request fields + plan wall parameters.

Contract asserts (loud, world-refusing):
  * the run receipt is complete and its saved_state_id equals the
    plan's saved_state_id for the world (the native sibling carries the
    same id -- registration s.2);
  * the arm verification (v0.4a machinery) passes for every aux source.

Never read: dispatcher_rows.jsonl, insertions.jsonl, any outcome block
of branch_record.json, CAL/HOLDOUT worlds in a TRAIN build, chunk-5.

The post-ack fit ALSO copies the needed v0.4a source files into the NEW
data root (COPY only, never hard-link) and writes MANIFEST + SHA256SUMS
(copy_v04a_sources / finalize_root); the dry-run uses the same row
builders read-only and writes under partB_v04b/results/.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import _paths  # noqa: F401
from . import j4plan, reg
from . import schedule_desc as SD

import actions_v04 as A          # noqa: E402  (partB_v0data, read-only)
import v0lib as V                # noqa: E402
import wall_tokens_v04 as WT     # noqa: E402
import wdpaths                   # noqa: E402


class BuildRefused(RuntimeError):
    """A world/source failed a loud build contract; it is refused."""


T0B = time.time()


def log(msg: str) -> None:
    print(f"[v04b {time.time() - T0B:7.1f}s] {msg}", flush=True)


def v04a_root(data_root: str | Path | None = None) -> Path:
    import wdpaths as W
    root = Path(data_root) if data_root else Path(W.RESEARCH_DENSE_ROOT)
    return Path(wdpaths.remap(root / reg.V04A_DATA_ROOT))


# --------------------------------------------------------------- aux rows --

def aux_t0_rows(world: str, rec: dict, root: Path, t0: float) -> list[dict]:
    """t0 rows of the native + aux sources from the v0.4a root (read-only)."""
    d = root / "worlds" / "chunk4"
    rows = []
    for source in (reg.NATIVE_SOURCE,) + reg.AUX_SOURCES:
        npz_p = d / f"{world}__{source}_feat_dep_strict.npz"
        lab_p = d / f"{world}__{source}_labels.parquet"
        if not npz_p.is_file() or not lab_p.is_file():
            raise BuildRefused(f"{world}/{source}: v0.4a files missing")
        npz = np.load(npz_p, allow_pickle=False)
        lab = pd.read_parquet(lab_p)
        m = np.isclose(lab["t"], t0)
        if int(m.sum()) != len(reg.TAUS):
            raise BuildRefused(f"{world}/{source}: {int(m.sum())} t0 label "
                               f"rows (need exactly {len(reg.TAUS)})")
        r0 = lab[m].iloc[0]
        # the v0.4a pi summary is per-horizon; v0.4b fixes it to the FULL
        # horizon (tau=570) for every source -- the per-tau structure of
        # the planned schedule lives in the s.12 segment descriptor
        r570 = lab[m][np.isclose(lab[m]["tau"], 570.0)].iloc[0]
        k = int(np.flatnonzero(np.isclose(npz["t"], t0))[0])
        tok = npz["cav_tokens"][k]
        msk = npz["cav_mask"][k] > 0
        act = tok[..., 5:10]
        counts = [float(((act[..., j] > 0.5) & msk).sum())
                  for j in range(4)] + [float(msk.sum() - sum(
                      ((act[..., j] > 0.5) & msk).sum() for j in range(4)))]
        arm_kind = str(r0["arm_kind"])
        with np.load(npz_p.parent /
                     f"{world}__{source}_feat_dep_det.npz",
                     allow_pickle=False) as nd:
            dem_det = nd["demand"][k].astype(np.float32)
            ctx_det = nd["context"][k].astype(np.float32)
        rows.append(dict(
            world=world, source=source, chunk_id="chunk4",
            split=str(rec["split"]), t0_s=t0,
            demand_veh_h=int(rec["loading_veh_h"]),
            penetration=float(rec["penetration"]),
            state_id=str(rec["base_state"]["saved_state_id"]),
            policy=str(r0["policy"]), arm_kind=arm_kind,
            dem_strict=npz["demand"][k].astype(np.float32),
            ctx_strict=npz["context"][k].astype(np.float32),
            dem_det=dem_det, ctx_det=ctx_det,
            cav_n_native=counts[0], cav_n_damping=counts[1],
            cav_n_pi=counts[2], cav_n_wall=counts[3],
            cav_n_unknown=counts[4],
            pi_native=float(r570["pi_native"]),
            pi_damping=float(r570["pi_damping"]),
            pi_pi=float(r570["pi_pi"]),
            **{f"y_{h}_{int(t)}": float(
                lab[m][np.isclose(lab[m]["tau"], t)].iloc[0][
                    V_TARGET_COL[h]])
               for t in reg.TAUS for h in reg.HEADS},
            **{f"tau_avail_{int(t)}": float(r0["tau_available"])
               for t in reg.TAUS},
        ))
    return rows


V_TARGET_COL = reg.HEAD_COL            # v0lib label dict key per head

# -------------------------------------------------------------- wall rows --


def _snapshot_positions(rt, t: float) -> tuple[dict, set]:
    """({vid: (s_down, lane)}, designated-CAV vid set) at t from a base
    trace snapshot (the member self-reports of the shared saved state)."""
    snap = rt.snapshot_at(t)
    if snap is None:
        raise BuildRefused(f"no base snapshot at t={t}")
    vids = [str(rt.vid_table[int(k)]) for k in snap["vid"]]
    pos = {v: (float(snap["s"][i]), int(snap["lane"][i]))
           for i, v in enumerate(vids)}
    cav = {vids[i] for i in range(len(vids))
           if int(snap["is_cav"][i]) == 1}
    return pos, cav


def record_request_fields(arm_dir: Path) -> dict:
    """branch_record.json request-time planner fields ONLY: the field
    set is wall_tokens_v04.ALLOWED_RECORD_FIELDS; the outcome block
    (outcome / assembled / released / aborted / abort_reason) is NEVER
    bound into the returned dict (registration s.3)."""
    p = None
    for cand in (Path(arm_dir) / "branch_record.json",
                 Path(arm_dir) / "run" / "branch_record.json"):
        if cand.is_file():
            p = cand
            break
    if p is None:
        raise BuildRefused(f"{arm_dir}: branch_record.json missing")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {k: raw[k] for k in WT.ALLOWED_RECORD_FIELDS if k in raw}


def wall_t0_rows(world: str, rec: dict, t0: float,
                 runs_root: Path, bases_root: Path,
                 rt_base=None, sched_cache: dict | None = None
                 ) -> tuple[list[dict], dict]:
    """The 4 wall arm t0 rows of one world, built from the raw runs."""
    sched_cache = {} if sched_cache is None else sched_cache
    base_dir = bases_root / f"{world}__base"
    ok, _r, reason = V.receipt_status(base_dir)
    if not ok:
        raise BuildRefused(f"{world}: base run incomplete: {reason}")
    if rt_base is None:
        rt_base = V.build_run_trace(world, "base", base_dir, t0,
                                    t_ins_pre={}, sched_cache=sched_cache)
    rt_base.demand_loading = int(rec["loading_veh_h"])
    t_pre = V.build_t_ins_pre(rt_base, t0)
    pos, cav = _snapshot_positions(rt_base, t0)
    want_state = str(rec["base_state"]["saved_state_id"])
    if str(rec["native_sibling"]["saved_state_id"]) != want_state:
        raise BuildRefused(f"{world}: native sibling saved_state_id != "
                           "base state (registration s.2)")
    rows, ev_refs = [], []
    state0 = rt_base_npz_state(rt_base, t0)
    for br in j4plan.wall_branches(rec):
        arm = br["arm"]
        arm_dir = runs_root / str(br["run"])
        ok, receipt, reason = V.receipt_status(arm_dir)
        if not ok:
            raise BuildRefused(f"{world}/{arm}: incomplete run: {reason}")
        if str(receipt.get("saved_state_id")) != want_state:
            raise BuildRefused(
                f"{world}/{arm}: receipt saved_state_id "
                f"{str(receipt.get('saved_state_id'))[:8]}... != plan "
                f"{want_state[:8]}... -- refusing the world")
        rt = V.build_run_trace(world, "native", arm_dir, t0,
                               t_ins_pre=t_pre, sched_cache=sched_cache,
                               base_arm_dir=base_dir)
        rt.demand_loading = int(rec["loading_veh_h"])
        tau_av = float(rt.data_end - t0)
        labels = {}
        for tau in reg.TAUS:
            complete = tau <= tau_av + 1e-9
            for h in reg.HEADS:
                labels[f"y_{h}_{int(tau)}"] = \
                    float(V.window_label(rt, t0, tau)[V_TARGET_COL[h]]) \
                    if complete else float("nan")
            labels[f"tau_avail_{int(tau)}"] = tau_av
        eps = WT.WallEpisodeSet.from_branch_dir(arm_dir)
        walls = eps.walls_at(t0, positions=pos, cav_roster=cav)
        if len(walls) > 1:
            raise BuildRefused(f"{world}/{arm}: {len(walls)} walls at t0 "
                               "(J4 arms plan exactly one)")
        w = walls[0] if walls else None
        tl = A.ActionTimeline.from_run_dir(
            arm_dir, schedule_authoritative=True)
        counts = {a: 0.0 for a in
                  ("native", "damping", "pi", "wall", "unknown")}
        for vid in cav:
            counts[tl.action_at(vid, t0)] += 1.0
        # ---- request-time wall parameters (planner-owned only) ----
        recf = record_request_fields(arm_dir)
        plan_wall = br["wall"]
        if str(recf.get("timing")) != str(plan_wall["timing"]):
            raise BuildRefused(
                f"{world}/{arm}: record timing={recf.get('timing')!r} != "
                f"plan {plan_wall['timing']!r} (drift; refusing)")
        for k in ("hold_s", "speed_factor"):
            if k in recf and float(recf[k]) != float(plan_wall[k]):
                raise BuildRefused(
                    f"{world}/{arm}: record {k}={recf[k]!r} != plan "
                    f"{plan_wall[k]!r} (drift; refusing)")
        hold_s = float(recf["hold_s"])
        speed_mps = float(recf["wall_speed_mps"])
        req_s = float(recf["request_s"]) - t0
        wall7 = SD.wall_block(
            hold_s=hold_s, wall_speed_mps=speed_mps,
            speed_factor=float(recf["speed_factor"]),
            timing=str(recf["timing"]),
            offset_s=float(reg.TIMING_OFFSET_S[str(recf["timing"])]),
            hold_cap_s=None)          # wall law v2.1 data: no hold cap
        members = tuple(recf.get("members") or
                        (w or {}).get("members", ()))
        sched = SD.wall_member_schedule(members, hold_s, t0=0.0,
                                        request_s=req_s, others="native")
        desc = SD.describe(sched, wall=wall7, t0=0.0)
        rows.append(dict(
            world=world, source=arm, chunk_id="chunk4",
            split=str(rec["split"]), t0_s=t0,
            demand_veh_h=int(rec["loading_veh_h"]),
            penetration=float(rec["penetration"]),
            state_id=want_state, policy="wall_v04b", arm_kind="wall",
            dem_strict=state0[0], ctx_strict=state0[1],
            dem_det=state0[2], ctx_det=state0[3],
            cav_n_native=counts["native"], cav_n_damping=counts["damping"],
            cav_n_pi=counts["pi"], cav_n_wall=counts["wall"],
            cav_n_unknown=counts["unknown"],
            pi_native=SD.pi_shares(sched)["native"],
            pi_damping=SD.pi_shares(sched)["damping"],
            pi_pi=SD.pi_shares(sched)["pi"],
            sched_desc=desc, sched_wall=wall7,
            **labels,
        ))
        ev_refs.append(dict(world=world, arm=arm, run=str(br["run"]),
                            receipt_sha=_sha_of(arm_dir),
                            phase_at_t0=(w or {}).get("phase"),
                            n_members=len(members),
                            request_s_rel=req_s,
                            hold_s=hold_s,
                            wall_speed_mps=speed_mps,
                            speed_factor=float(recf["speed_factor"]),
                            timing=str(recf["timing"])))
    return rows, dict(world=world, state_id=want_state, walls=ev_refs)


def rt_base_npz_state(rt_base, t0: float) -> tuple:
    """The strict/detector deployable blocks of the SHARED base trace at
    t0 (identical for every pair member of the world; v0.4a verified the
    detector/strict context against the oracle context at build)."""
    from deployable_features import SourceObs
    ins = dict(rt_base.t_ins_pre)
    ins.update(rt_base.t_ins)
    pairs = sorted(ins.items())
    obs = SourceObs(rt_base.sched_vid, rt_base.sched_t, rt_base.sched_ent,
                    rt_base.sched_is_cav,
                    [v for v, _ in pairs], [t for _, t in pairs])
    dem_s, ctx_s = obs.row(t0, t0, float(rt_base.demand_loading),
                           variant="strict")
    dem_d, ctx_d = obs.row(t0, t0, float(rt_base.demand_loading),
                           variant="detector")
    return dem_s, ctx_s, dem_d, ctx_d


def _pi_share(sched: dict, action: str) -> float:
    """Planner-known schedule summary over the horizon (the v0.4a
    action_shares semantics: known time only, renormalised)."""
    tot = 0.0
    acc = 0.0
    for ivs in sched.values():
        for lo, hi, a in ivs:
            tot += hi - lo
            if a == action:
                acc += hi - lo
    return float(acc / tot) if tot > 0 else 0.0


def _sha_of(arm_dir: Path) -> str:
    p = Path(arm_dir) / "run" / "physics_steps.jsonl"
    return V.sha256_file(p) if p.is_file() else ""


# ---------------------------------------------------- aux schedule descs --

def fleet_kind_of(policy: str) -> str:
    """(v0.4a policy class) -> the planned-schedule fleet action."""
    return {"damping_fleet": "damping", "native_fleet": "native",
            "pi_fleet": "pi"}.get(str(policy), "")


def sched_desc_for_aux(row: dict, runs_root: Path, t0: float) -> np.ndarray:
    """PLANNED-SCHEDULE descriptor of an aux arm from its raw run:
    fleet arms (native/damping/pi) are whole-window; fixed/switching
    arms come from switch_schedule.json events (planned actions only;
    no outcome field exists in a schedule)."""
    policy = str(row.get("policy"))
    fam = reg.ARM_FAMILY.get(str(row["source"]), "native")
    arm_dir = runs_root / f"{row['world']}__{row['source']}"
    if fam in ("native", "damping", "pi") and fleet_kind_of(policy):
        sched = SD.fleet_schedule(fleet_kind_of(policy), n_roster=1)
    else:
        sp = A._first_existing(arm_dir, A.SCHEDULE_NAME)
        if sp is None:
            raise BuildRefused(
                f"{row['world']}/{row['source']}: no switch_schedule.json")
        events = [(vid, ts - t0, a) for vid, ts, a in A._read_schedule(sp)]
        roster = None
        sched = SD.schedule_from_events(events, roster=roster, t0=0.0)
    return SD.describe(sched, wall=None, t0=0.0)


# ------------------------------------------------------------- assemble ----

def build_world_rows(world: str, rec: dict, root: Path,
                     runs_root: Path, bases_root: Path,
                     rt_base=None, sched_cache: dict | None = None
                     ) -> tuple[pd.DataFrame, dict]:
    """All t0 rows of one world (native + aux + 4 wall arms) with the
    PLANNED-SCHEDULE descriptor attached to every row."""
    t0 = float(rec["t0_s"])
    rows = aux_t0_rows(world, rec, root, t0)
    wall_rows, ref = wall_t0_rows(world, rec, t0, runs_root, bases_root,
                                  rt_base=rt_base,
                                  sched_cache=sched_cache)
    for r in rows:
        if r["source"] == reg.NATIVE_SOURCE:
            r["sched_desc"] = SD.describe(
                SD.fleet_schedule("native", n_roster=1), wall=None)
            r["sched_wall"] = np.zeros(7, np.float32)
        else:
            # the aux chunk-4 arms live in the chunk-4 campaign runs root
            r["sched_desc"] = sched_desc_for_aux(r, bases_root, t0)
            r["sched_wall"] = np.zeros(7, np.float32)
    rows.extend(wall_rows)
    df = pd.DataFrame(rows, columns=reg_row_cols())
    return df, ref


def reg_row_cols() -> list:
    from .pairs_v04b import ROW_COLS
    return list(ROW_COLS)


# ------------------------------------------------- post-ack data root ------

def copy_v04a_sources(out_root: Path, worlds: dict, v04a: Path) -> dict:
    """COPY (never hard-link) the needed v0.4a source files of the given
    worlds into the NEW data root; returns {rel: sha256} verification."""
    files: dict = {}
    for world in worlds:
        for source in (reg.NATIVE_SOURCE,) + reg.AUX_SOURCES:
            for suffix in ("_feat_dep_strict.npz", "_feat_dep_det.npz",
                           "_labels.parquet"):
                src = v04a / "worlds" / "chunk4" / \
                    f"{world}__{source}{suffix}"
                rel = f"worlds/chunk4/{src.name}"
                dst = out_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                files[rel] = V.sha256_file(dst)
    return files


def write_sha256sums(out_root: Path, extra: dict | None = None) -> int:
    rels = sorted(p.relative_to(out_root).as_posix()
                  for p in out_root.rglob("*") if p.is_file()
                  and p.name != "SHA256SUMS")
    lines = [f"{V.sha256_file(out_root / rel)}  {rel}" for rel in rels]
    (out_root / "SHA256SUMS").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")
    return len(lines)


def finalize_root(out_root: Path, refs: list, argv: list) -> dict:
    """MANIFEST.json of the NEW root (per-source verification evidence)."""
    man = dict(
        schema="wd01_partB_v0data/1", v04b_schema=reg.SCHEMA,
        zero_simulation=True,
        registration="output/wd01/REGISTRATION_PARTB_V04B.md",
        registration_lf_sha256=reg.reg_sha(),
        tau_s=list(reg.TAUS),
        note=("WD-01-PB-V04B root: J4 wall pairs + auxiliary chunk-4 "
              "contrasts; t0-row-only features/labels; v0.4a sources "
              "COPIED (never hard-linked); wall rows built from the raw "
              "J4 runs; dispatcher_rows/insertions/outcome blocks never "
              "read"),
        never_read=["dispatcher_rows.jsonl", "insertions.jsonl",
                    "branch_record.json outcome block", "chunk-5",
                    "HOLDOUT (none exists in J4)"],
        worlds={r["world"]: r for r in refs},
        build=dict(built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime()),
                   builder="partB_v04b/build_v04b.py",
                   command=" ".join(argv)))
    p = out_root / "MANIFEST.json"
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(man, indent=1), encoding="utf-8")
    tmp.replace(p)
    return man

