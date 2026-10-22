"""Round-9 items 22--26: exact frozen heads, F1/F2, single-use TRAIN audit.

All data/artifacts are workspace-local. CPU only; zero support is never floored.
The old scalar fitter remains available explicitly as --legacy-approx.
"""
# The standalone entrypoint must bootstrap workspace packages before importing.
# ruff: noqa: E402
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from functools import lru_cache
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tools import merge_fit_v2 as old
from tools import r9_exact as exact
from tools import r9_fit_exact as heads

MODELS = (*heads.MODELS, "M0-original-runtime", "M0-approx", "F1", "P1", "P1+P2", "P1-P3")
FROZEN = ("M0", "M0-seqTF", "M0-corrected", "M0-original-runtime", "M0-approx")


def condition(dist, batch):
    """F2: divide full-path masses by S_(b-1), never reset the population prior."""
    start = torch.tensor([s.get("score_start", 0) for s in batch["sequences"]])
    unknown = torch.tensor([s.get("unknown_age", False) or s.get("unknown_history", False)
                            for s in batch["sequences"]])
    ix = (start - 1).clamp_min(0).unsqueeze(1)
    den = torch.where(start > 0, dist["log_survival"].gather(1, ix).squeeze(1), 0.)
    unsupported = unknown | ~torch.isfinite(den)
    safe_den = torch.where(unsupported, 0., den).unsqueeze(1)
    prefix = torch.arange(batch["valid"].shape[1]).unsqueeze(0) < start.unsqueeze(1)
    mass = torch.where(prefix, -torch.inf, dist["log_mass"] - safe_den)
    survival = torch.where(prefix, 0., dist["log_survival"] - safe_den)
    return {"log_mass": torch.where(unsupported.unsqueeze(1), torch.nan, mass),
            "log_survival": torch.where(unsupported.unsqueeze(1), torch.nan, survival),
            "cdf": torch.where(unsupported.unsqueeze(1), torch.nan, -torch.expm1(survival))}


def hybrid(c, tau, w, n_quad=32, eligible=None):
    """F1: ONE persistent Logistic z; independent conditional decision noise."""
    if not bool(torch.isfinite(w).all() and (w > 0).all()):
        raise ValueError("w must be positive finite")
    if not bool(torch.isfinite(tau).all() and (tau > 0).all()) or torch.isnan(c).any():
        raise ValueError("tau must be positive finite and margins observed")
    z, weights = old.logistic_quadrature(n_quad)
    finite = torch.isfinite(c)
    x = (torch.where(finite, c, 0.).unsqueeze(-2) + tau*z.unsqueeze(-1))/w
    logits = torch.where(finite.unsqueeze(-2), x, c.unsqueeze(-2))
    return old.hazard_distribution(logits, weights, eligible)


class Candidate(nn.Module):
    def __init__(self, kind, initial):
        super().__init__()
        self.kind = kind
        self.base = old.Candidate("M0" if kind == "M0-approx" else
                                  "P1-P3" if kind == "F1" else kind, initial)
        if kind == "F1":
            # A starting point only; validation chooses penalties including zero.
            self.base.raw["w"] = nn.Parameter(torch.tensor(
                math.log(math.expm1(initial["sigma"])), dtype=old.DTYPE))

    def params(self):
        return self.base.params()

    def median_margin(self, batch):
        return self.base.median_margin(batch)

    def distribution(self, batch, n_quad=32):
        if self.kind != "F1":
            return condition(self.base.distribution(batch, min(n_quad, 64)), batch)
        p, f = self.params(), batch["updated"]
        df = p["s_f"] + p["T_f"]*f["v_f"] + F.relu(f["v_f"]**2-f["v"]**2)/(2*p["b_f"])
        dl = p["s_l"] + p["T_l"]*f["v"] + F.relu(f["v"]**2-f["v_l"]**2)/(2*p["b_l"])
        sides = []
        for side, d in (("f", df), ("l", dl)):
            present, gap = f["has_"+side], f["gap_"+side]
            ratio = torch.where(gap > 0, gap, 1.).log() - d.log()
            sides.append(torch.where(present, torch.where(gap > 0, ratio, -torch.inf), torch.inf))
        c = torch.minimum(*sides)
        return condition(hybrid(c, p["tau"], p["w"], n_quad,
                                batch["physical"] & batch["valid"]), batch)

    def nll(self, batch, n_quad=32):
        dist = self.distribution(batch, n_quad)
        ix = batch["last"].unsqueeze(1)
        score = -torch.where(batch["event"], dist["log_mass"].gather(1, ix).squeeze(1),
                             dist["log_survival"].gather(1, ix).squeeze(1))
        competing = torch.tensor([s["terminal_kind"] not in
                                 ("event", "horizon_censor", "administrative_end")
                                 for s in batch["sequences"]])
        return torch.where(torch.isnan(score) | competing, torch.inf, score)


def pack(sequences):
    if all("exact" in s for s in sequences):
        return heads.exact_pack(sequences)
    batch = old.pack(sequences)
    batch["sequences"] = sequences
    return batch


def make_model(kind, initial, frozen):
    if kind == "M0-original-runtime":
        return heads.ExactCandidate("M0", frozen)
    if kind in heads.MODELS:
        return heads.ExactCandidate(kind, frozen, corrected_protocol=kind in ("M0", "M0-refit"))
    return Candidate(kind, initial)


def batches(sequences):
    # Length bucketing bounds quadrature padding without changing the objective.
    ordered = sorted(sequences, key=lambda s: len(s["times"]))
    return [pack(subset) for subset in old.sequence_batches(ordered, 64)]


def penalty(model, pair):
    p = model.params()
    return sum(pair[i] * p[key].square() for i, key in enumerate(("tau", "w")) if key in p)


def optimize(kind, initial, frozen, sequences, args, pair=(0., 0.)):
    model = make_model(kind, initial, frozen)
    if kind in FROZEN:
        return model, {"status": "untouched", "updates": 0}
    packed = batches(sequences)
    # Physical/support failures abort the WHOLE fit, never a supported-only subset.
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    best, best_loss, trace = copy.deepcopy(model.state_dict()), math.inf, []
    for step in range(args.iters):
        optimizer.zero_grad()
        total = 0.
        for batch in packed:
            losses = model.nll(batch, args.n_quad)
            if not torch.isfinite(losses).all():
                model.load_state_dict(best)
                return model, {"status": "exact_support_failure", "updates": step,
                               "unsupported_cars_in_batch": int((~torch.isfinite(losses)).sum()),
                               "cohort_cars": len(sequences), "trace": trace}
            # Advice objective is SUM NLL + lambda_tau*tau^2 + lambda_w*w^2.
            contribution = losses.sum()
            total += float(contribution.detach())
            contribution.backward()
        shrink = penalty(model, pair)
        if isinstance(shrink, torch.Tensor):
            total += float(shrink.detach())
            shrink.backward()
        grads = {k: p.grad for k, p in model.named_parameters() if p.requires_grad}
        bad = [k for k, v in grads.items() if v is None or not torch.isfinite(v).all()]
        if bad:
            model.load_state_dict(best)
            return model, {"status": "nonfinite_gradient", "parameters": bad, "updates": step}
        if total < best_loss:
            best_loss, best = total, copy.deepcopy(model.state_dict())
        if step in (0, args.iters-1):
            trace.append({"step": step, "sum_objective": total,
                          "gradient_norms": {k: float(v.norm()) for k, v in grads.items()}})
        optimizer.step()
    with torch.no_grad():
        final = sum(float(model.nll(b, args.n_quad).sum()) for b in packed)
        final += float(penalty(model, pair))
    if math.isfinite(final) and final < best_loss:
        best, best_loss = copy.deepcopy(model.state_dict()), final
    model.load_state_dict(best)
    return model, {"status": "fitted", "updates": args.iters, "sum_objective": best_loss, "trace": trace}


def select(kind, initial, frozen, sequences, args):
    if kind in FROZEN:
        model, report = optimize(kind, initial, frozen, sequences, args)
        return model, {"final_fit": report, "selection_status": "not_applicable"}
    worlds = sorted({s["world"] for s in sequences})
    folds = [worlds[i::args.inner_folds] for i in range(args.inner_folds)]
    pairs = (list(itertools.product(args.shrinkage, repeat=2)) if kind == "F1" else
             [(v, 0.) for v in args.shrinkage] if kind in ("P1+P2", "P1-P3") else [(0., 0.)])
    inner = []
    for pair in pairs:
        scores, reports = [], []
        for validation in folds:
            train = [s for s in sequences if s["world"] not in validation]
            val = [s for s in sequences if s["world"] in validation]
            if not train or not val:
                reports.append({"worlds": validation, "status": "insufficient_worlds"})
                continue
            model, info = optimize(kind, initial, frozen, train, args, pair)
            with torch.no_grad():
                score = (sum(float(model.nll(b, args.n_quad).sum()) for b in batches(val))/len(val)
                         if info["status"] == "fitted" else math.inf)
            scores.append(score)
            reports.append({"worlds": validation, "nll_per_car": score, "fit": info})
        score = float(np.mean(scores)) if len(scores) == args.inner_folds else math.inf
        inner.append({"lambda_tau": pair[0], "lambda_w": pair[1], "validation_nll": score, "folds": reports})
        # Invariant support failures cannot be repaired by a scale penalty.
        if reports and all(r.get("fit", {}).get("status") == "exact_support_failure" for r in reports):
            break
    chosen = min(inner, key=lambda r: (r["validation_nll"], r["lambda_tau"], r["lambda_w"]))
    ok = math.isfinite(chosen["validation_nll"])
    pair = (chosen["lambda_tau"], chosen["lambda_w"]) if ok else (0., 0.)
    model, info = optimize(kind, initial, frozen, sequences, args, pair)
    return model, {"final_fit": info, "selection_status": "ok" if ok else "unresolved",
                   "selected_penalties": pair if ok else None, "inner": inner}


def evaluate(model, sequences, args):
    # Legacy evaluate batches internally; aggregate once with cached laws.
    distributions, losses, margins = [], [], []
    with torch.no_grad():
        for subset in old.sequence_batches(sequences):
            b = pack(subset)
            dist = model.distribution(b, args.n_quad)
            loss = model.nll(b, args.n_quad)
            margin = model.median_margin(b)
            for i, s in enumerate(subset):
                n = len(s["times"])
                distributions.append({k: v[i, :n] for k, v in dist.items()})
                losses.append(float(loss[i]))
                margins.append(margin[i, :n])

    class Cached:
        offset = 0

        def distribution(self, batch, q):
            count, length = batch["valid"].shape
            self.range = range(self.offset, self.offset+count)
            result = {k: torch.zeros(count, length, dtype=old.DTYPE)
                      for k in ("log_mass", "log_survival", "cdf")}
            for j, i in enumerate(self.range):
                for k in result:
                    v = distributions[i][k]
                    result[k][j, :len(v)] = v
            return result

        def nll(self, batch, q):
            return torch.tensor([losses[i] for i in self.range])

        def median_margin(self, batch):
            result = torch.zeros_like(batch["valid"], dtype=old.DTYPE)
            for j, i in enumerate(self.range):
                result[j, :len(margins[i])] = margins[i]
            self.offset += len(self.range)
            return result

    result = old.evaluate(Cached(), sequences, args.n_quad)
    for row, seq, dist in zip(result["cars"], sequences, distributions):
        row["prefix_decisions"] = seq.get("score_start", 0)
        row["history_unresolved"] = bool(seq.get("unknown_age", False))
        row["model_support_failure"] = not math.isfinite(row["nll"])
        row["conditioning_denominator_failure"] = (not seq.get("unknown_age", False)
                                                     and not bool(torch.isfinite(dist["cdf"]).all()))
        row["competing_exit_unidentified"] = seq["terminal_kind"] not in ("event", "horizon_censor", "administrative_end")
    result["history"] = {"cars": len(sequences),
                         "complete_entry": sum(not s["unknown_age"] for s in sequences),
                         "prefix_conditioned": sum(s.get("score_start", 0) > 0 for s in sequences),
                         "unresolved": sum(s["unknown_age"] for s in sequences)}
    result["noncompletion"] = old.noncompletion(result["cars"], args)
    result["nll_world_bootstrap"] = old.world_bootstrap(
        [r["nll"] for r in result["cars"]], [r["world"] for r in result["cars"]],
        args.bootstrap, args.seed, weighting="car")
    return result


def refinement(model, sequences, args):
    """Report numerical sensitivity, not a newly invented convergence cutoff."""
    reports, previous = [], None
    for q in sorted({args.n_quad, min(64, args.n_quad*2), min(64, args.n_quad*4)}):
        nll, mass, gradients = [], [], []
        for batch in batches(sequences):
            model.zero_grad()
            losses = model.nll(batch, q)
            dist = model.distribution(batch, q)
            nll.extend(losses.detach().tolist())
            mass.extend(dist["log_mass"].detach().exp()[batch["valid"]].tolist())
            if any(p.requires_grad for p in model.parameters()) and torch.isfinite(losses).all():
                losses.sum().backward()
                gradients.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
        gradient = torch.stack(gradients).sum(0) if gradients else None
        entry = {"nodes": q, "mean_nll": float(np.mean(nll)) if nll else None,
                 "supported_cars": sum(math.isfinite(x) for x in nll), "cars": len(nll),
                 "gradient_norm": float(gradient.norm()) if gradient is not None else None}
        if previous is not None:
            pn, pm, pg = previous
            with np.errstate(invalid="ignore"):
                delta = np.asarray(nll)-np.asarray(pn)
            entry["max_abs_nll_change"] = float(np.max(np.abs(delta))) if np.isfinite(delta).all() else None
            delta = np.asarray(mass)-np.asarray(pm)
            entry["max_abs_mass_change"] = float(np.max(np.abs(delta))) if np.isfinite(delta).all() else None
            entry["gradient_change_norm"] = float((gradient-pg).norm()) if gradient is not None and pg is not None else None
        reports.append(entry)
        previous = nll, mass, gradient
    return reports


def reconcile(sequences, reports):
    excluded = [e for r in reports for e in r.get("exclusions", [])]
    def keys(s):
        return (s["world"], s["source"], s["episode"], s["vehicle"], s["slot"], s["entry_row"])
    ids = [keys(s) for s in sequences] + [keys(s) for s in excluded]
    if len(ids) != len(set(ids)):
        raise ValueError("overlapping windows/duplicate risk identities would overweight cohort")
    reasons = {reason: sum(e["reason"] == reason for e in excluded)
               for reason in sorted({e["reason"] for e in excluded})}
    return {"original_observed_risk_identities": len(ids), "retained": len(sequences),
            "retained_events": sum(s["event"] for s in sequences), "excluded": len(excluded),
            "exclusion_reasons": reasons, "exclusions": excluded,
            "identity_sha256": hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest(),
            "equation": "original_observed_risk_identities = retained + excluded",
            "LCSTATE_2915_mapping": "unresolved; a different source cohort, never substituted as denominator"}


def gates(kind, results):
    result = results[kind]
    ev = result["audit"]
    if not ev["cars"]:
        return {"passed": False, "status": "audit_not_opened"}
    checks = {"full_support": all(math.isfinite(r["nll"]) for r in ev["cars"]),
              "physical_events": ev["physical_support"]["share"] == 1.,
              "complete_history": ev["history"]["unresolved"] == 0,
              "fit_and_inner_selection": result["fit"]["selection_status"] == "ok"}
    # Item 26: fallback treatment is tested against untouched exact M0 only.
    # The matched refit is descriptive here, never an impossible self-superiority gate.
    baselines = ("M0",) if kind == "Soft-hazard" else ("M0", "M0-refit")
    for base in baselines:
        pair = ev["paired_vs"][base]["nll"]
        checks["nll_UCB_below_zero_vs_"+base] = pair.get("status") == "ok" and pair["ci95"][1] < 0
    early = ev["paired_vs"]["M0"]["early_mass"]
    checks["early_mass_UCB_below_zero_vs_M0"] = early.get("status") == "ok" and early["ci95"][1] < 0
    timing = ev["timing"]
    sb, ab = timing["signed_median_bounds_s"], timing["absolute_median_bounds_s"]
    checks["signed_median_within_2s"] = sb[0] is not None and sb[0] >= -2 and sb[1] <= 2
    checks["median_absolute_within_12s"] = ab[1] is not None and ab[1] <= 12
    for q in ("25", "50", "90"):
        checks["population_CDF_p"+q] = ev["population_cdf"]["quantiles"].get(q, {}).get("gate") is True
    for horizon in ("120", "300"):
        checks["noncompletion_in_band_"+horizon] = ev["noncompletion"]["all"]["tails"][horizon].get("inside_empirical_band_certified") is True
    # Keep the original absolute/relative tail and timing gates with the same
    # amended baseline scope, in addition to the paired uncertainty gates.
    for base in baselines:
        ref = results[base]["audit"]
        rs, ra = ref["timing"]["signed_median_bounds_s"], ref["timing"]["absolute_median_bounds_s"]
        far = max(abs(sb[0]), abs(sb[1])) if sb[0] is not None else math.inf
        near = 0. if rs[0] is None or rs[0] <= 0 <= rs[1] else min(abs(rs[0]), abs(rs[1]))
        checks["signed_median_strictly_better_vs_"+base] = far < near
        checks["median_absolute_no_worse_vs_"+base] = ab[1] is not None and ra[0] is not None and ab[1] <= ra[0]
        for h in ("120", "300"):
            tail, rt = ev["noncompletion"]["all"]["tails"][h], ref["noncompletion"]["all"]["tails"][h]
            checks[f"noncompletion_no_worse_{h}_vs_{base}"] = (
                "absolute_error_bounds" in tail and "absolute_error_bounds" in rt
                and tail["absolute_error_bounds"][1] <= rt["absolute_error_bounds"][0])
    return {"passed": all(checks.values()), "status": "passed" if all(checks.values()) else "CHANGES",
            "gates": checks, "required_NLL_baselines": baselines,
            "refit_role": "descriptive_comparator" if kind == "Soft-hazard" else "required_comparator"}


@lru_cache(maxsize=1)
def _worker_frozen(checkpoint, physics_json):
    return exact.load_frozen(checkpoint, physics=json.loads(physics_json) if physics_json else None)


def prepare_cached(job):
    """Independent CPU episode replay; stable input-order aggregation in parent."""
    ep, checkpoint, physics_json, n_quad, allow_unverified, path = job
    path = exact.guard(path)
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        print(f"cache {ep['episode']}: {len(cached['sequences'])} cars", flush=True)
        return cached["sequences"], cached["report"]
    frozen = _worker_frozen(checkpoint, physics_json)
    subset, report = heads.build_dataset([ep], frozen, n_quad=n_quad,
                                        allow_unverified_physics=allow_unverified)
    temporary = exact.guard(path.with_suffix(f".{os.getpid()}.tmp"))
    torch.save({"sequences": subset, "report": report}, temporary)
    temporary.replace(path)
    return subset, report


def paired(result, baseline, args):
    fields = ("world", "source", "episode", "vehicle", "entry_row", "event_row")
    def ids(ev):
        return [tuple(row[k] for k in fields) for row in ev["cars"]]
    if ids(result) != ids(baseline):
        raise ValueError("paired comparison identity/order mismatch")
    return old.paired_comparisons(result, baseline, args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--round8-params", default=str(exact.DEFAULT_CHECKPOINT))
    parser.add_argument("--models", default="Soft-hazard,F1,P1,P1+P2,P1-P3")
    parser.add_argument("--physics", help="workspace-local authoritative PhysicsConfig JSON")
    parser.add_argument("--allow-unverified-physics", action="store_true")
    parser.add_argument("--max-worlds", type=int)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--lr", type=float, default=.03)
    parser.add_argument("--n-quad", type=int, default=16)
    parser.add_argument("--shrinkage", nargs="+", type=float, default=[0., .01, 1.])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--audit-once", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--md", required=True)
    parser.add_argument("--cache-dir", default="results/r9_stage2a_cache")
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1),
                        help="independent single-thread CPU episode workers; results ordered deterministically")
    args = parser.parse_args(argv)
    kinds = list(dict.fromkeys(["M0", "M0-original-runtime", "M0-seqTF", "M0-refit", "M0-approx"]+args.models.split(",")))
    if any(k not in MODELS for k in kinds):
        parser.error("unknown model")
    if (args.inner_folds < 2 or args.iters < 1 or not 1 <= args.n_quad <= 64
            or args.bootstrap < 0 or args.workers < 1 or not 0 < args.lr < math.inf
            or args.max_worlds is not None and args.max_worlds < 1
            or 0. not in args.shrinkage or any(not math.isfinite(v) or v < 0 for v in args.shrinkage)):
        parser.error("invalid fit/grid/quadrature configuration")
    if args.audit_once and args.max_worlds is not None:
        parser.error("smokes may not consume the fixed audit")
    args.seed = old.check_seed(args.seed)
    out, md, cache_dir = (exact.guard(v) for v in (args.out, args.md, args.cache_dir))
    if out == md:
        parser.error("out and md must differ")
    ledger = exact.guard(ROOT / "results/r9_stage2a_audit_use.json")
    if args.audit_once and ledger.exists():
        raise ValueError("fixed TRAIN audit already consumed: results/r9_stage2a_audit_use.json")
    started = perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    physics = None if args.physics is None else old.read_json(args.physics)
    frozen = exact.load_frozen(args.round8_params, physics=physics)
    if not frozen.metadata["physics_confirmed"] and not args.allow_unverified_physics:
        raise ValueError("authoritative physics unavailable; --allow-unverified-physics enables diagnostic replay only")
    frozen_snapshot = {k: v.clone() for k, v in frozen.model.state_dict().items()}
    initial = {k: float(v.detach()) for k, v in frozen.core.params().items()}
    initial["sigma"] = float(frozen.core.alpha_scale())
    # Discover and partition ALL worlds BEFORE selecting a smoke prefix.
    episodes, sources = old.discover(args.source)
    worlds = sorted({e["world"] for e in episodes})
    audit_worlds = worlds[4::5]
    fit_worlds = [w for w in worlds if w not in audit_worlds]
    if args.max_worlds is not None:
        fit_worlds = fit_worlds[:args.max_worlds]
    fit_eps = [e for e in episodes if e["world"] in fit_worlds]
    audit_eps = [e for e in episodes if e["world"] in audit_worlds]
    # Input hashes and implementation hashes prevent stale geometry/head caches.
    code_paths = [ROOT / "tools" / name for name in
                  ("r9_exact.py", "r9_runtime.py", "merge_fit_v2.py")]
    code_paths += sorted((ROOT / "src/wms").rglob("*.py"))
    preparation_code = "".join(inspect.getsource(getattr(heads, name)) for name in
                               ("_SparsePairAttention", "_replay_model", "_arrays", "_batchable",
                                "prepare_episode", "attach_episode", "build_dataset"))
    code_hash = hashlib.sha256(("".join(old.sha256(p) for p in code_paths)+preparation_code).encode()).hexdigest()

    def prepare(eps):
        sequences, reports, jobs = [], [], []
        cache_dir.mkdir(parents=True, exist_ok=True)
        for ep in eps:
            stamp = {"checkpoint": frozen.metadata["sha256"], "physics": frozen.metadata["physics"],
                     "physics_confirmed": frozen.metadata["physics_confirmed"],
                     "code": code_hash, "n_quad": args.n_quad, "source": ep["source_name"],
                     "inputs": {name: old.sha256(ep["path"]/(name+".npy")) for name in
                                ("veh_kin", "veh_mask", "vid", "is_cav", "time_s", "episode")
                                if name != "episode"}}
            stamp["episode"] = old.sha256(ep["path"]/"episode.json")
            key = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
            path = exact.guard(cache_dir/(key+".pt"))
            jobs.append((ep, str(exact.guard(args.round8_params)), json.dumps(physics) if physics else None,
                         args.n_quad, args.allow_unverified_physics, str(path)))
        if args.workers == 1:
            prepared = map(prepare_cached, jobs)
            for subset, report in prepared:
                sequences.extend(subset)
                reports.append(report)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for subset, report in pool.map(prepare_cached, jobs):
                    sequences.extend(subset)
                    reports.append(report)
        return sequences, reports

    fit, reports = prepare(fit_eps)
    prepared = perf_counter()
    results, models = {}, {}
    for kind in kinds:
        tick = perf_counter()
        print(f"fitting {kind}: {len(fit)} TRAIN-fit cars; audit unopened", flush=True)
        model, info = select(kind, initial, frozen, fit, args)
        models[kind] = model
        results[kind] = {"parameters": {k: float(v.detach()) for k, v in model.params().items()},
                         "fit": info, "fit_diagnostics_not_audit": evaluate(model, fit, args),
                         "quadrature_refinement_fit_only": refinement(model, fit, args) if kind in ("F1", "Soft-hazard") else None,
                         "fit_seconds": perf_counter()-tick}
        # Persist complete heads, including alpha location/residual and both scales.
        weights = exact.guard(out.with_name(out.stem+"_"+kind.replace("+", "_")+".pt"))
        weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": kind, "state_dict": model.state_dict(), "fit": info,
                    "checkpoint_sha256": frozen.metadata["sha256"], "protocol": vars(args)}, weights)
        results[kind]["weights"] = {"path": str(weights.relative_to(ROOT)), "sha256": old.sha256(weights)}
    fitted = perf_counter()
    preserved = all(torch.equal(v, frozen.model.state_dict()[k]) for k, v in frozen_snapshot.items())
    if not preserved:
        raise RuntimeError("frozen round-8 tensors changed during replay/fitting")
    audit, audit_reports = [], []
    primary = [k for k in kinds if k in ("Soft-hazard", "F1")]
    audit_ready = bool(primary) and frozen.metadata["physics_confirmed"] and all(
        results[k]["fit"]["selection_status"] == "ok" and
        results[k]["fit"]["final_fit"]["status"] == "fitted" for k in primary)
    audit_used = args.audit_once and audit_ready
    if audit_used:
        # Exclusive create is done BEFORE opening audit observations, after ALL
        # choices/weights are frozen. Failures still consume the audit attempt.
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("x", encoding="utf-8") as stream:
            json.dump({"config": vars(args), "checkpoint": frozen.metadata["sha256"],
                       "audit_worlds": audit_worlds,
                       "fits": {k: v["fit"] for k, v in results.items()}}, stream)
        audit, audit_reports = prepare(audit_eps)
    for kind, result in results.items():
        result["audit"] = evaluate(models[kind], audit, args)
    for kind, result in results.items():
        for partition in ("audit", "fit_diagnostics_not_audit"):
            result[partition]["paired_vs"] = {base: paired(
                result[partition], results[base][partition], args) for base in ("M0", "M0-refit")}
        result["s3b"] = gates(kind, results)
    blockers = ["SUMO substep/length/projection and joint conflict scoring remain unverified",
                "Legacy seqTF omits learned location/residual: deployment and seqTF are distinct laws",
                "Original LCSTATE cohort mapping is unresolved; observed risk cohort is reconciled locally"]
    if not frozen.metadata["physics_confirmed"]:
        blockers.append("Checkpoint has no authoritative physics metadata; replay is explicitly DIAGNOSTIC")
    if not audit_used:
        blockers.append("Fixed one-in-five TRAIN audit remains unopened")
    if args.audit_once and not audit_ready:
        blockers.append("Audit request refused before observation read: fit/inner support or authoritative physics unresolved")
    unsupported = sum(not math.isfinite(r["nll"]) for r in results["M0"]["fit_diagnostics_not_audit"]["cars"])
    if unsupported:
        blockers.append(f"Exact M0: {unsupported} fit-cohort histories/outcomes/support unresolved")
    finished = perf_counter()
    report = {"schema": "r9_offline_fit/3-stage2a", "verdict": "CHANGES", "config": vars(args),
              "frozen_tensors_preserved": preserved,
              "implementation_sha256": {str(p.relative_to(ROOT)): old.sha256(p) for p in
                                         (*code_paths, ROOT / "tools/r9_fit_exact.py", Path(__file__))},
              "checkpoint": frozen.metadata, "sources": sources,
              "split": {"fit_worlds": fit_worlds, "audit_worlds": audit_worlds,
                        "rule": "sorted complete TRAIN worlds 4::5, grouped across sources",
                        "audit_requested": args.audit_once, "audit_used": audit_used},
              "cohort": reconcile(fit+audit, reports+audit_reports),
              "event_cross_tab_fit_only": [{
                  **{k: s[k] for k in ("world", "source", "episode", "vehicle", "slot", "entry_row", "event_row")},
                  "physical_veto": not s["physical"][-1], "veto_reason": s["physical_reasons"][-1],
                  "history_unresolved": bool(s["unknown_age"]), "prefix_decisions": s.get("score_start", 0),
                  "models": {kind: {k: result["fit_diagnostics_not_audit"]["cars"][i][k] for k in
                                    ("zero_mass_event", "model_support_failure", "conditioning_denominator_failure")}
                             for kind, result in results.items()}}
                  for i, s in enumerate(fit) if s["event"]],
              "data": {"fit": reports, "audit": audit_reports}, "models": results,
              "timing": {"prepare_seconds": prepared-started, "fit_and_diagnostics_seconds": fitted-prepared,
                         "audit_seconds": finished-fitted, "total_seconds": finished-started,
                         "fit_worlds": len(fit_worlds), "episodes": len(fit_eps),
                         "projected_full_prepare_seconds": (prepared-started)*len(episodes)/len(fit_eps)},
              "semantics": {"M0": "untouched full learned deployed head on common corrected causal path and F2",
                            "M0-original-runtime": "same frozen head, original runtime geometry/envelope; bookkeeping effect diagnostic",
                            "M0-seqTF": "untouched actual seq_tf_chunk zero-location/no-residual law",
                            "M0-approx": "old scalar-only approximation; never deployed round 8",
                            "M0-refit": "complete learned head refit on same corrected fixed geometry/F2 as M0",
                            "Soft-hazard": "fallback full learned head refit, corrected geometry and F2; item26 vs untouched M0",
                            "F1": "sigmoid((c+tau*z)/w); persistent logistic z; six learned gap parameters",
                            "F2": "recorded risk-entry prefixes condition survival once; unknown histories unresolved",
                            "P1-P3": "strict persistent first passage diagnostic; zero masses retained",
                            "exits": "competing outcomes counted separately; no administrative-censor assumption"},
              "qualification_blockers": blockers}
    out.parent.mkdir(parents=True, exist_ok=True)
    md.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(old.clean_json(report), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    text = ["# Round-9 stage-2A offline fit", "", "Verdict: CHANGES", "",
            f"Fit worlds: {len(fit_worlds)}. Audit opened: {audit_used}.",
            f"CPU seconds: {finished-started:.3f}.", "", "## Blockers", *["- "+b for b in blockers]]
    for kind, result in results.items():
        ev = result["audit"] if audit_used else result["fit_diagnostics_not_audit"]
        text.extend(["", "## "+kind, "- Fit: "+result["fit"]["final_fit"]["status"],
                     "- Parameters: `"+json.dumps(result["parameters"])+"`",
                     "- NLL: `"+json.dumps(old.clean_json(ev["nll"]))+"`",
                     "- History: `"+json.dumps(ev["history"])+"`"])
    md.write_text("\n".join(text)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(out.relative_to(ROOT)), "timing": report["timing"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
