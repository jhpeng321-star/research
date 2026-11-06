"""WM-D qualification run: truth tapes + labels + backend -> metrics -> pre-registered gate verdict.

    python -m tools.wzqual.run_qual --out DIR --tapes GLOB [--tapes GLOB ...]
        --identity IDENTITY.json --labels-dir RUN_DIR --moments GLOB [...]
        --backend keep_speed | wm --checkpoint CKPT [--train-worlds MANIFEST.json]
        [--rule-tapes GLOB] [--samples 8] [--max-moments N] [--max-starts N] [--device cpu]

Order (integrity I3): load + check inputs -> fit and WRITE the breakdown rule -> roll out ->
score -> gate.  Everything is written under --out (never overwritten: the directory must be new).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import breakdown as BD
from . import distrib as D
from . import errors as E
from . import frame as F
from . import gate as G
from . import labels as L
from . import traffic as TR
from .backends import KeepSpeedBackend
from .moves import arms_for
from .traj import Traj, load_c4_episode, load_wzqual_tape, start_state

TRAIN = (989200, 989599)
RESTRICTED = ((989600, 989699), (964000, 964199), (962000, 962999), (977000, 977999))
HORIZON = 300.0


def restricted(seed: int) -> bool:
    return any(a <= seed <= b for a, b in RESTRICTED) or not (TRAIN[0] <= seed <= TRAIN[1])


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write_json(path: Path, doc) -> str:
    raw = json.dumps(_jsonable(doc), indent=1, sort_keys=True).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


# ------------------------------------------------------------------------ inputs
def load_tapes(patterns, identity: Optional[dict]) -> tuple[dict, list]:
    """``{world: Traj}`` from WM-D tapes (identity-gated) and WM-B WMQUAL episodes; receipts."""
    ok_sha = None
    if identity is not None:
        ok_sha = {r["tape_sha256"] for r in identity["rows"] if r.get("identical")}
    out, receipts = {}, []
    for pat in patterns:
        for p in sorted(glob.glob(pat, recursive=True)):
            raw = Path(p).read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            if p.endswith("TAPE.npz"):
                if ok_sha is not None and sha not in ok_sha:
                    receipts.append(dict(path=p, sha256=sha, used=False, reason="identity gate"))
                    continue
                tr = load_wzqual_tape(p)
            else:
                tr = load_c4_episode(p)
                if str(tr.meta.get("split", "")).upper() != "WMQUAL" and not tr.meta.get("wmqual"):
                    receipts.append(dict(path=p, sha256=sha, used=False, reason="not split=WMQUAL"))
                    continue
            seed = int(tr.meta.get("seed", -1))
            if restricted(seed):
                raise SystemExit(f"I5: restricted seed {seed} in {p}")
            dem = tr.meta.get("demand")
            dem = dem.get("main") if isinstance(dem, dict) else (dem or tr.meta.get("demand_vph"))
            cell = tr.meta.get("cell") or F.cell_key(dem, tr.meta.get("penetration"))
            tr.meta["cell"] = cell
            world = f"{seed}:{cell}"
            if world in out:
                receipts.append(dict(path=p, sha256=sha, used=False, reason="duplicate world"))
                continue
            out[world] = tr
            receipts.append(dict(path=p, sha256=sha, used=True, world=world))
    return out, receipts


def train_world_seeds(path: Optional[str]) -> Optional[set]:
    """Seeds in the checkpoint's train+val world list (WM-E manifest; any JSON with world ids)."""
    if not path:
        return None
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    seeds = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("seed", "world_seed") and isinstance(v, (int, str)) and str(v).isdigit():
                    seeds.add(int(v))
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            import re
            for m in re.finditer(r"(?<!\d)(98\d{4})(?!\d)", x):
                seeds.add(int(m.group(1)))
    walk(doc)
    return seeds


def fit_rule(tapes: dict, rule_patterns, out: Path) -> BD.BreakdownRule:
    if rule_patterns:
        fit, _ = load_tapes(rule_patterns, None)
        source = "rule tapes: " + ", ".join(rule_patterns)
    else:
        fit = tapes
        source = "qualification truth tapes (no separate fitting tapes supplied)"
    speeds = []
    for tr in fit.values():
        z = TR.zone_speed_bins(tr, float(tr.time_s[0]), float(tr.time_s[-1]))
        speeds += [x for x in z if math.isfinite(x)]
    rule = BD.fit_rule(speeds, source=source)
    write_json(out / "BREAKDOWN_RULE.json", rule.record())
    return rule


# ------------------------------------------------------------------------ backends
def make_backend(args):
    if args.backend == "keep_speed":
        return KeepSpeedBackend()
    if args.backend == "wm":
        from .wm_backend import WmBackend
        return WmBackend(args.checkpoint, device=args.device, map_spec=args.map_spec,
                         physics_toml=args.physics_toml)
    raise SystemExit(f"unknown backend {args.backend}")


def make_secondary(name, args):
    """Secondary action-response backends (amendment 1; never gating)."""
    if name in ("planapi_proposer", "planapi_full"):
        from .planapi_backend import PlanApiBackend
        return PlanApiBackend(args.checkpoint, device=args.device, map_spec=args.map_spec,
                              observability=name.split("_", 1)[1])
    raise SystemExit(f"unknown secondary backend {name}")


def check_contract(start, trajs: dict) -> list:
    issues = []
    fr = start.frame
    for arm, samples in trajs.items():
        for tr in samples:
            if abs(tr.time_s[0] - start.t0) > 1e-6 or abs(tr.dt - F.DT) > 1e-6:
                issues.append(f"{arm}: time grid")
                continue
            f0 = tr.frame(0)
            m = {n: (s, v) for n, s, v in zip(f0["ids"], f0["s"], f0["v"])}
            for n, s, v in zip(fr["ids"], fr["s"], fr["v"]):
                if n in m and (abs(m[n][0] - s) > 1e-3 or abs(m[n][1] - v) > 1e-3):
                    issues.append(f"{arm}: t0 frame differs for {n}")
                    break
    return issues


# ------------------------------------------------------------------------ parts
def _predict(backend, jobs, *, n_samples):
    """``jobs = [(start, arms, seed)]`` -> list of {arm: [Traj]} (batched when supported)."""
    if hasattr(backend, "rollout_batch"):
        return backend.rollout_batch(jobs, horizon_s=HORIZON, n_samples=n_samples)
    return [backend.rollout(st, arms, horizon_s=HORIZON, n_samples=n_samples, seed=seed)
            for st, arms, seed in jobs]


def distributional(tapes, backend, ks, rule, args, log) -> tuple[list, list, list]:
    records, car_wm, car_ks, traffic_wm = [], [], [], []
    issues = []
    worlds = sorted(tapes.items())
    if getattr(args, "start_shard", None):
        si, sn = (int(x) for x in args.start_shard.split("/"))
        worlds = worlds[si::sn]
    starts = []
    for world, tr in worlds:
        t_end = float(tr.time_s[-1])
        t0 = 300.0
        while t0 + HORIZON <= t_end + 1e-6:
            starts.append((world, tr, t0))
            t0 += 120.0
    if args.max_starts:
        starts = starts[: args.max_starts]
    chunk = max(1, int(getattr(args, "batch_starts", 1) or 1))
    for c0 in range(0, len(starts), chunk):
        part = starts[c0:c0 + chunk]
        descs, jobs = [], []
        for world, tr, t0 in part:
            st = start_state(tr, t0, history_s=120.0)
            seed = int(hashlib.sha256(f"wzqual:{world}:{t0}".encode()).hexdigest()[:8], 16)
            descs.append((world, tr, t0, st, seed))
            jobs.append((st, {"base": []}, seed))
        preds = _predict(backend, jobs, n_samples=args.samples)
        for (world, tr, t0, st, seed), pred in zip(descs, preds):
            pre = TR.zone_speed_bins(tr, t0 - 30.0, t0)
            broken = BD.start_class(pre, rule)
            cohort = np.asarray([tr.index_of(x) for x in st.frame["ids"]])
            truth = tr.window(t0, t0 + HORIZON)
            kpred = ks.rollout(st, {"base": []}, horizon_s=HORIZON, n_samples=1, seed=seed)
            issues += check_contract(st, pred)
            ids_t0 = st.frame["ids"]

            def summ(x, coh=None, t0=t0, broken=broken):
                return D.summarize_future(x, t0, HORIZON, rule, broken_at_start=broken, cohort=coh)

            def coh_of(p, ids_t0=ids_t0):
                return np.asarray([p.index_of(x) for x in ids_t0 if x in p._index])
            rec = dict(world=world, seed=st.seed, cell=st.cell, t0=t0, broken_at_start=broken,
                       truth=summ(truth, cohort), wm=[summ(p, coh_of(p)) for p in pred["base"]],
                       ks=[summ(p, coh_of(p)) for p in kpred["base"]],
                       lane_changes=dict(truth=TR.lane_changes(truth, t0, t0 + HORIZON),
                                         wm=[TR.lane_changes(p, t0, t0 + HORIZON) for p in pred["base"]]))
            records.append(rec)
            for p in pred["base"]:
                car_wm.append(E.car_errors(truth, p, t0))
                traffic_wm.append(E.traffic_errors(truth, p, t0, HORIZON))
            car_ks.append(E.car_errors(truth, kpred["base"][0], t0))
        log(f"distributional {len(records)}/{len(starts)} starts")
    return records, (car_wm, car_ks, traffic_wm), issues


def j_mode(backend, trajs) -> str:
    """'total' (road + source backlog, = label J), 'road' (all predicted vehicles incl. entrants,
    no source term) or 't0_cohort' (vehicles present at t0 only) - WM_QUAL_GATE.md section 4."""
    if all(p.backlog is not None and np.isfinite(p.backlog).all() for p in trajs):
        return "total"
    return "road" if getattr(backend, "models_entries", False) else "t0_cohort"


def pred_J(traj: Traj, t0: float, ids_t0, mode: str) -> float:
    if mode == "total":
        return TR.team_delay(traj, t0, t0 + HORIZON, include_backlog=True)
    if mode == "road":
        return TR.team_delay(traj, t0, t0 + HORIZON, include_backlog=False)
    coh = np.asarray([traj.index_of(x) for x in ids_t0 if x in traj._index])
    return TR.team_delay(traj, t0, t0 + HORIZON, cohort=coh, include_backlog=False)


def action_rows(cands, tapes, backend, args, log) -> tuple[list, dict]:
    supported = set(backend.supported_roles)
    rows, cov = [], dict(n_T=0, n_supported=0, n_no_tape=0, n_no_spec=0)
    by_moment = L.moments_of([c for c in cands if c.T is not None])
    moments = sorted(by_moment)
    if args.max_moments:
        moments = moments[: args.max_moments]
    if getattr(args, "moment_shard", None):
        i, n = (int(x) for x in args.moment_shard.split("/"))
        moments = moments[i::n]
    prepared = []
    for mom in moments:
        group = by_moment[mom]
        cov["n_T"] += len(group)
        c0 = group[0]
        tr = tapes.get(c0.world)
        if tr is None:
            cov["n_no_tape"] += len(group)
            continue
        ok = []
        for c in group:
            if not c.members:
                cov["n_no_spec"] += 1
            elif {m["role"] for m in c.members} <= supported:
                ok.append(c)
        cov["n_supported"] += len(ok)
        if not ok:
            continue
        st = start_state(tr, c0.t0, history_s=120.0)
        arms = {"base": []}
        for c in ok:
            a = arms_for(c.members, with_solos=c.S is not None)
            for name, members in a.items():
                if name == "base":
                    continue
                arms[f"{c.cid}::{name}"] = members
        seed = int(hashlib.sha256(f"wzqual:{mom}".encode()).hexdigest()[:8], 16)
        prepared.append((mom, ok, st, arms, seed))
    chunk = max(1, int(getattr(args, "batch_moments", 1) or 1))
    jds = []
    for c0_ in range(0, len(prepared), chunk):
        part = prepared[c0_:c0_ + chunk]
        if hasattr(backend, "evaluate_J"):
            for mom, ok, st, arms, seed in part:
                jds.append((backend.evaluate_J(st, arms, horizon_s=HORIZON, n_samples=args.samples,
                                               seed=seed), "backend_J"))
        else:
            preds = _predict(backend, [(st, arms, seed) for _m, _o, st, arms, seed in part],
                             n_samples=args.samples)
            for (mom, ok, st, arms, seed), pred in zip(part, preds):
                mode = j_mode(backend, pred["base"])
                jds.append(({k: np.asarray([pred_J(p, st.t0, st.frame["ids"], mode) for p in v])
                             for k, v in pred.items()}, mode))
        log(f"action-response {getattr(backend, 'name', '?')}: {len(jds)}/{len(prepared)} moments")
    for (mom, ok, st, arms, seed), (jd, mode) in zip(prepared, jds):
        live = set(st.frame["ids"])
        jb = jd["base"]
        for c in ok:
            jg = jd[f"{c.cid}::group"]
            row = dict(cid=c.cid, moment=mom, seed=c.seed, cell=c.cell, family=c.family, arm=c.arm,
                       T_true=c.T, S_true=c.S, S_lb_true=c.S_lb, admitted=c.admitted,
                       members_live_t0=all(x in live for x in c.member_ids),
                       pred_T=float(np.mean(jb - jg)), pred_T_sd=float(np.std(jb - jg)),
                       J_mode=mode)
            solo_names = [k for k in arms if k.startswith(f"{c.cid}::solo:")]
            if c.S is not None and solo_names:
                js = np.stack([jd[k] for k in solo_names])
                row["pred_S"] = float(np.mean(js.min(0) - jg))
            elif c.S is not None and len(c.members) == 1:
                row["pred_S"] = 0.0
            rows.append(row)
    cov["fraction_T_candidates_supported"] = cov["n_supported"] / max(cov["n_T"], 1)
    cov["supported_roles"] = sorted(supported)
    return rows, cov


def distributional_metrics(records, rule) -> dict:
    nb = [r for r in records if not r["broken_at_start"]]
    by_cell: dict = {}
    for r in nb:
        by_cell.setdefault(r["cell"], []).append(r)
    out = dict(
        n_starts=len(records), n_starts_not_broken=len(nb), n_broken_at_start=len(records) - len(nb),
        rule_degenerate=bool(rule.degenerate), rule_boundary_mps=rule.boundary_mps,
        n_truth_events=int(sum(r["truth"]["occ"]["300"] for r in nb)),
        breakdown_rate_truth={h: D.truth_rate(records, h) for h in ("60", "120", "300")},
        breakdown_rate_wm={h: D.breakdown_rate(records, h) for h in ("60", "120", "300")},
        breakdown_rate_keep_speed={h: D.breakdown_rate(records, h, "ks") for h in ("60", "120", "300")},
        rate_diff_pooled=D.rate_diff(records, "300"),
        rate_diff_pooled_ci=D.cluster_bootstrap(records, lambda rs: D.rate_diff(rs, "300")),
        rate_diff_by_cell={c: dict(n=len(rs), truth=D.truth_rate(rs), wm=D.breakdown_rate(rs),
                                   diff=D.rate_diff(rs)) for c, rs in sorted(by_cell.items())},
        onset_w1_s=D.onset_w1(records), onset_w1_keep_speed_s=D.onset_w1(records, "ks"),
        queue_rel_mean_err=D.rel_mean_err(records, "queue_mean"),
        queue_norm_w1=D.norm_w1(records, "queue_mean"),
        closing_queue_rel_mean_err=D.rel_mean_err(records, "closing_queue_mean"),
        extent_rel_mean_err=D.rel_mean_err(records, "extent_mean_m"),
        throughput_rel_err=D.rel_mean_err(records, "throughput_vph"),
        throughput_rel_err_keep_speed=D.rel_mean_err(records, "throughput_vph", "ks"),
        zone_crps={str(int(h)): dict(wm=D.zone_crps(records, h), keep_speed=D.zone_crps(records, h, "ks"))
                   for h in (60.0, 120.0, 300.0)},
        secondary=dict(brier_300=D.brier(records), brier_300_keep_speed=D.brier(records, "300", "ks"),
                       auc_300=D.auc([np.mean([s["occ"]["300"] for s in r["wm"]]) for r in nb],
                                     [r["truth"]["occ"]["300"] for r in nb]) if nb else None,
                       reliability_300=D.reliability(records),
                       recovery=D.recovery_diff(records),
                       lane_change_rates=_lane_rates(records)))
    return out


def _lane_rates(records) -> dict:
    """Pooled lane-change rates per WM-lane transition (per veh-h): truth vs WM sample mean."""
    def pool(items):
        cnt, vh = {}, 0.0
        for it in items:
            vh += it["veh_h"]
            for k, v in it["counts"].items():
                cnt[k] = cnt.get(k, 0) + v
        return {k: v / max(vh, 1e-9) for k, v in sorted(cnt.items())}
    recs = [r for r in records if r.get("lane_changes")]
    if not recs:
        return {}
    truth = pool([r["lane_changes"]["truth"] for r in recs])
    wm = pool([x for r in recs for x in r["lane_changes"]["wm"]])
    return dict(truth=truth, wm=wm)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tapes", action="append", required=True)
    ap.add_argument("--identity")
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--moments", action="append", default=[])
    ap.add_argument("--backend", default="keep_speed")
    ap.add_argument("--checkpoint")
    ap.add_argument("--train-worlds")
    ap.add_argument("--rule-tapes", action="append", default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--max-moments", type=int, default=0)
    ap.add_argument("--max-starts", type=int, default=0)
    ap.add_argument("--skip-distributional", action="store_true")
    ap.add_argument("--skip-action", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--map-spec", default=None, help="wz3to2 MAP_SPEC.json (binds the WM road)")
    ap.add_argument("--physics-toml", default=None,
                    help="training TOML whose build_physics() replaces a missing physics.json")
    ap.add_argument("--secondary", action="append", default=[],
                    help="secondary action-response backend(s): planapi_proposer, planapi_full")
    ap.add_argument("--moment-shard", default=None, help="i/n: score moments i::n only (parallel)")
    ap.add_argument("--batch-starts", type=int, default=16, help="starts per batched WM rollout")
    ap.add_argument("--batch-moments", type=int, default=2, help="moments per batched WM rollout")
    ap.add_argument("--start-shard", default=None, help="i/n: distributional worlds i::n only")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    logf = (out / "RUN.log").open("a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        logf.write(line + "\n")
        logf.flush()
        print(line, flush=True)

    spec, spec_sha = G.load_spec()
    log(f"gate spec sha256 {spec_sha}")
    identity = json.loads(Path(args.identity).read_text()) if args.identity else None
    tapes, receipts = load_tapes(args.tapes, identity)
    write_json(out / "INPUT_TAPES.json", receipts)
    log(f"{len(tapes)} truth worlds")
    trained = train_world_seeds(args.train_worlds)
    qual_seeds = {int(w.split(":")[0]) for w in tapes}
    integrity = dict(I1_world_disjoint=(trained is not None and not (qual_seeds & trained)),
                     I2_truth_identity=all(r.get("used") or r.get("reason") for r in receipts),
                     I5_no_restricted_seeds=True)
    if trained is not None and qual_seeds & trained:
        log(f"I1 FAIL overlap {sorted(qual_seeds & trained)[:20]}")
    rule = fit_rule(tapes, args.rule_tapes, out)
    integrity["I3_rule_frozen_first"] = True
    log(f"breakdown rule boundary {rule.boundary_mps:.3f} m/s degenerate={rule.degenerate} n={rule.n_fit}")
    backend = make_backend(args)
    ks = KeepSpeedBackend()
    if args.samples is None:
        args.samples = spec["samples_per_start"]["stochastic" if backend.stochastic else "deterministic"]
    metrics = dict(gate_spec_sha256=spec_sha, backend=backend.name, samples=args.samples,
                   checkpoint=args.checkpoint, breakdown_rule=rule.record(),
                   checkpoint_sha256=getattr(backend, "checkpoint_sha256", None),
                   physics_source=getattr(backend, "physics_source", None),
                   physics_sha256=getattr(backend, "physics_sha256", None),
                   road=getattr(getattr(backend, "road", None), "name", None),
                   shards=dict(moments=args.moment_shard, starts=args.start_shard))
    contract_issues = []
    if not args.skip_distributional:
        records, (car_wm, car_ks, traffic_wm), issues = distributional(tapes, backend, ks, rule, args, log)
        contract_issues += issues
        write_json(out / "DIST_RECORDS.json", records)
        metrics["distributional"] = distributional_metrics(records, rule)
        metrics["distributional"]["backend_models_entries"] = bool(getattr(backend, "models_entries", False))
        metrics["secondary_errors"] = dict(
            car_wm=E.summarize_car(car_wm), car_keep_speed=E.summarize_car(car_ks),
            traffic_wm={k: (float(np.nanmean([t[k] for t in traffic_wm if t.get(k) is not None]))
                            if any(t.get(k) is not None for t in traffic_wm) else None)
                        for k in (traffic_wm[0] if traffic_wm else {})})
    if not args.skip_action:
        cands = L.load_candidates(args.labels_dir, moments_paths=[p for g in args.moments for p in glob.glob(g)])
        for c in cands:
            if restricted(c.seed):
                raise SystemExit(f"I5: restricted label seed {c.seed}")
        rows, cov = action_rows(cands, tapes, backend, args, log)
        ks_rows, _ = action_rows(cands, tapes, ks, argparse.Namespace(**{**vars(args), "samples": 1}), log)
        sec = {}
        for name in args.secondary:
            sb = make_secondary(name, args)
            sr, scov = action_rows(cands, tapes, sb, args, log)
            sec[name] = dict(rows=sr, coverage=scov)
        write_json(out / "ACTION_ROWS.json", dict(wm=rows, keep_speed=ks_rows,
                                                   secondary={k: v["rows"] for k, v in sec.items()}))
        metrics["role_coverage"] = cov
        from .ranking import action_response
        metrics["action_response"] = action_response(rows, ks_rows=ks_rows)
        if sec:
            metrics["secondary_action_response"] = {
                k: dict(coverage=v["coverage"], metrics=action_response(v["rows"], ks_rows=ks_rows))
                for k, v in sec.items()}
    integrity["I4_backend_contract"] = not contract_issues and getattr(backend, "closed_loop", True)
    metrics["contract_issues"] = contract_issues[:50]
    metrics["integrity"] = integrity
    msha = write_json(out / "WM_QUAL_METRICS.json", metrics)
    verdict = G.evaluate(_jsonable(metrics), spec)
    verdict.update(metrics_sha256=msha, gate_spec_sha256=spec_sha, backend=backend.name,
                   checkpoint=args.checkpoint)
    write_json(out / "GATE_VERDICT.json", verdict)
    log(f"VERDICT {verdict['verdict']}")
    return 0




# ------------------------------------------------------------------------ shard merge
def merge_main(argv=None) -> int:
    """``python -m tools.wzqual.run_qual merge --out FINAL --shards DIR ...``: recompute every metric
    and the gate from the shards' raw records (rule, backend and checkpoint must agree)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec, spec_sha = G.load_spec()
    docs = [json.loads((Path(d) / "WM_QUAL_METRICS.json").read_text()) for d in args.shards]
    keys = ("gate_spec_sha256", "backend", "checkpoint_sha256", "samples")
    for k in keys:
        if len({json.dumps(d.get(k)) for d in docs}) != 1:
            raise SystemExit(f"shards disagree on {k}")
    if len({d["breakdown_rule"]["sha256"] for d in docs}) != 1:
        raise SystemExit("shards disagree on the breakdown rule")
    rule = BD.BreakdownRule(**{k: v for k, v in docs[0]["breakdown_rule"].items() if k != "sha256"})
    records, rows, ks_rows, sec = [], [], [], {}
    for d in args.shards:
        p = Path(d) / "DIST_RECORDS.json"
        if p.exists():
            records += json.loads(p.read_text())
        a = Path(d) / "ACTION_ROWS.json"
        if a.exists():
            doc = json.loads(a.read_text())
            rows += doc["wm"]
            ks_rows += doc["keep_speed"]
            for k, v in (doc.get("secondary") or {}).items():
                sec.setdefault(k, []).extend(v)
    metrics = {k: docs[0].get(k) for k in keys + ("checkpoint", "physics_source", "breakdown_rule")}
    metrics["merged_from"] = list(args.shards)
    metrics["integrity"] = {k: all(d.get("integrity", {}).get(k) is True for d in docs)
                            for k in docs[0].get("integrity", {})}
    metrics["contract_issues"] = [x for d in docs for x in d.get("contract_issues", [])][:50]
    if records:
        metrics["distributional"] = distributional_metrics(records, rule)
        metrics["distributional"]["backend_models_entries"] = all(
            d.get("distributional", {}).get("backend_models_entries", False) for d in docs if "distributional" in d)
        cars = [d["secondary_errors"] for d in docs if "secondary_errors" in d]
        metrics["secondary_errors"] = dict(per_shard=cars, note="per-shard pooled summaries (merge = list)")
    if rows:
        from .ranking import action_response
        covs = [d["role_coverage"] for d in docs if "role_coverage" in d]
        cov = {k: sum(c[k] for c in covs) for k in ("n_T", "n_supported", "n_no_tape", "n_no_spec")}
        cov["fraction_T_candidates_supported"] = cov["n_supported"] / max(cov["n_T"], 1)
        cov["supported_roles"] = covs[0]["supported_roles"]
        metrics["role_coverage"] = cov
        metrics["action_response"] = action_response(rows, ks_rows=ks_rows)
        if sec:
            metrics["secondary_action_response"] = {k: dict(metrics=action_response(v, ks_rows=ks_rows))
                                                    for k, v in sec.items()}
        write_json(out / "ACTION_ROWS.json", dict(wm=rows, keep_speed=ks_rows, secondary=sec))
    if records:
        write_json(out / "DIST_RECORDS.json", records)
    msha = write_json(out / "WM_QUAL_METRICS.json", metrics)
    verdict = G.evaluate(_jsonable(metrics), spec)
    verdict.update(metrics_sha256=msha, gate_spec_sha256=spec_sha, backend=metrics["backend"],
                   checkpoint=metrics.get("checkpoint"))
    write_json(out / "GATE_VERDICT.json", verdict)
    print("VERDICT", verdict["verdict"])
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        sys.exit(merge_main(sys.argv[2:]))
    sys.exit(main())
