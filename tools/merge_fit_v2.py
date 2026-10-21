"""CPU-only Round-9 A-only recorded-path qualification; see R9_FIT.md.

No simulation, future-state features, likelihood floors, or behavioural cutoffs.
The default causal proposal is explicitly a diagnostic, NOT simulator parity.
"""
from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from wms.model.merge_core import core_features
from wms.model.merge_frailty import gauss_hermite
from wms.train.losses.merge_core import _core_margin

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
MODELS = ("M0", "M0-refit", "P1", "P1+P2", "P1-P3")
FEATURES = ("gap_l", "gap_f", "v", "v_l", "v_f", "a", "a_l", "a_f", "has_l", "has_f")
PARAM_KEYS = {"s0": "s0_m", "T": "T_s", "b": "b_mps2", "tau_a": "tau_a_s",
              "temperature": "temperature_m", "sigma": "alpha_log_sigma"}
DTYPE = torch.float64


def guard(path) -> Path:
    """Resolve traversal/symlinks before *any* data open or output creation."""
    raw = str(path)
    resolved = Path(path).resolve()
    if any(t in (raw + "/" + str(resolved)).lower() for t in FORBIDDEN):
        raise ValueError(f"restricted path refused: {path}")
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"path outside workspace refused: {path}")
    for match in re.finditer(r"(?<!\d)(964\d{3})(?!\d)", raw):
        check_seed(int(match[1]))
    return resolved


def check_seed(seed):
    seed = int(seed)
    if 964000 <= seed <= 964199:
        raise ValueError(f"evaluation seed {seed} refused")
    return seed


def aligned_row(previous_row: int) -> int:
    return int(previous_row) + 1


def first_passage(c, tau):
    """Exact logistic running-maximum law, last axis = time.

    log(sigmoid(y)-sigmoid(x)) = logsigmoid(y)+logsigmoid(-x)
    + log(1-exp(x-y)). Branch non-records/infinities before arithmetic.
    """
    if not bool(torch.isfinite(tau).all() and (tau > 0).all()):
        raise ValueError("tau must be positive finite")
    if torch.isnan(c).any():
        raise ValueError("missing observations cannot be free-road evidence")
    finite = torch.isfinite(c)
    scaled = torch.where(finite, c, torch.zeros_like(c)) / tau
    x = torch.where(finite, scaled, c)
    current = torch.cummax(x, dim=-1).values
    previous = torch.cat((torch.full_like(current[..., :1], -math.inf), current[..., :-1]), -1)
    log_mass = torch.full_like(current, -math.inf)
    records = current > previous
    first = records & torch.isneginf(previous)
    log_mass[first] = F.logsigmoid(current[first])
    atom = records & torch.isposinf(current) & ~first
    log_mass[atom] = F.logsigmoid(-previous[atom])
    regular = records & torch.isfinite(previous) & torch.isfinite(current)
    old, new = previous[regular], current[regular]
    log_mass[regular] = F.logsigmoid(new) + F.logsigmoid(-old) + torch.log(-torch.expm1(old - new))
    return {"log_mass": log_mass, "log_survival": F.logsigmoid(-current),
            "cdf": torch.sigmoid(current), "running_max": current}


def soft_hazard(features, params, n_quad=8, eligible=None):
    """Round-8 seq_tf_chunk math, integrating one persistent normal draw/car."""
    sample = features["v"]
    nodes, weights = gauss_hermite(n_quad, dtype=sample.dtype, device=sample.device)
    alpha = torch.exp((params["sigma"] * nodes).clamp(-8, 8)).unsqueeze(-1)
    margin = _core_margin(alpha, *(features[k].unsqueeze(-2) for k in FEATURES),
                          params["s0"], params["T"], params["b"], params["tau_a"], None, None)
    return hazard_distribution(margin / params["temperature"].clamp_min(1e-3), weights, eligible)


def hazard_distribution(logit, weights, eligible=None):
    finite = torch.isfinite(logit)
    safe = torch.where(finite, logit, torch.zeros_like(logit))
    log_h = torch.where(finite, F.logsigmoid(safe),
                        torch.where(logit > 0, torch.zeros_like(logit), torch.full_like(logit, -math.inf)))
    log_wait = torch.where(finite, F.logsigmoid(-safe),
                           torch.where(logit > 0, torch.full_like(logit, -math.inf), torch.zeros_like(logit)))
    if eligible is not None:
        eligible = eligible.unsqueeze(-2)
        log_h = torch.where(eligible, log_h, torch.full_like(log_h, -math.inf))
        log_wait = torch.where(eligible, log_wait, torch.zeros_like(log_wait))
    cumulative = torch.cumsum(log_wait, -1)
    previous = torch.cat((torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]), -1)
    logw = weights.log().unsqueeze(-1)
    log_mass = exact_logsumexp(previous + log_h + logw, -2)
    log_survival = exact_logsumexp(cumulative + logw, -2)
    return {"log_mass": log_mass, "log_survival": log_survival,
            "cdf": -torch.expm1(log_survival)}


def exact_logsumexp(x, dim):
    """Avoid undefined all-zero-mixture backward without adding event support."""
    supported = torch.isfinite(x).any(dim, keepdim=True)
    value = torch.logsumexp(torch.where(supported, x, torch.zeros_like(x)), dim)
    return torch.where(supported.squeeze(dim), value, torch.full_like(value, -math.inf))


@lru_cache(maxsize=64)
def logistic_quadrature(n_quad):
    """Gauss-Legendre integration over population quantiles, z=logit(U).

    P2 and P3 share the proposed Logistic latent. P3 integrates it analytically;
    this soft-hazard comparator integrates its sequence likelihood, not each tick.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    u = torch.tensor((nodes+1)/2, dtype=DTYPE)
    return u.log()-torch.log1p(-u), torch.tensor(weights/2, dtype=DTYPE)


def world_bootstrap(values, worlds, n_boot, seed, *, weighting="world"):
    """Whole-world resampling; pass paired car deltas for paired comparisons.

    Public default is equal-world means. Audit NLL explicitly uses car weighting
    (resampled sums/counts) to match its mean-per-car estimand.
    """
    check_seed(seed)
    if len(values) != len(worlds) or weighting not in ("world", "car"):
        raise ValueError("bootstrap needs matching values/worlds and world or car weighting")
    if not len(values):
        return {"estimate": None, "ci95": None, "status": "empty", "n_worlds": 0}
    groups = sorted(set(worlds))
    arrays = [np.asarray([v for v, w in zip(values, worlds) if w == g], float) for g in groups]
    if any(not np.isfinite(a).all() for a in arrays):
        return {"estimate": float(np.mean(values)), "ci95": None,
                "status": "nonfinite_no_CI", "n_worlds": len(groups)}
    sums = np.array([a.sum() for a in arrays])
    counts = np.array([len(a) for a in arrays])
    means = sums / counts
    estimate = float(means.mean() if weighting == "world" else sums.sum() / counts.sum())
    if n_boot < 1:
        return {"estimate": estimate, "ci95": None, "status": "disabled", "n_worlds": len(groups)}
    rng = np.random.default_rng(check_seed(seed))
    samples = rng.integers(len(groups), size=(n_boot, len(groups)))
    draws = (means[samples].mean(1) if weighting == "world" else
             sums[samples].sum(1) / counts[samples].sum(1))
    return {"estimate": estimate, "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "status": "ok" if len(groups) > 1 else "one_world_degenerate",
            "n_worlds": len(groups), "weighting": weighting}


def read_json(path):
    return json.loads(guard(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    h = hashlib.sha256()
    with guard(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_params(path):
    path = guard(ROOT / "src/wms/model/merge_core_init.json" if path == "default" else path)
    if path.suffix.lower() in (".pt", ".pth", ".ckpt"):
        raw = torch.load(path, map_location="cpu", weights_only=True)
        state = raw.get("model_state_dict", raw.get("state_dict", raw.get("model", raw)))
        aliases = {"s0": "s0_raw", "T": "t_raw", "b": "b_raw", "tau_a": "tau_raw",
                   "temperature": "temperature_raw", "sigma": "alpha_log_sigma_raw"}
        out = {}
        for name, suffix in aliases.items():
            matches = [v for k, v in state.items() if k.endswith("merge_core." + suffix) or k == suffix]
            if len(matches) != 1:
                raise ValueError(f"checkpoint needs one merge_core.{suffix}")
            value = matches[0].detach().cpu().reshape(())
            out[name] = float(value.clamp(-4, 2).exp() if name == "sigma" else F.softplus(value) + 1e-6)
        for key, value in state.items():
            if "merge_core." in key and (key.endswith("alpha_bias") or
                    re.search(r"(?:residual_head|alpha_net)\.2\.(?:weight|bias)$", key)):
                if torch.count_nonzero(value):
                    raise ValueError("checkpoint has learned residual/location: need a full frozen comparator adapter")
    else:
        raw = read_json(path)
        raw = raw.get("fitted", raw)
        out = {k: float(raw[v]) for k, v in PARAM_KEYS.items()}
    if any(not math.isfinite(v) or v <= 0 for v in out.values()):
        raise ValueError("all round-8 parameters must be positive finite")
    return out, {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                 "semantics": "context-free round-8 seq-TF; legacy residual/location off"}


def causal_proposal(kin, dt):
    """Diagnostic constant-current-acceleration proposal; never future motion."""
    out = kin.copy()
    out[..., 4] = np.maximum(kin[..., 4] + kin[..., 5] * dt, 0)
    out[..., 0] = kin[..., 0] + .5 * (kin[..., 4] + out[..., 4]) * dt
    return out


def load_adapter(path):
    if path is None:
        return None, {"name": "constant_current_acceleration_DIAGNOSTIC", "verified": False}
    path = guard(path)
    spec = importlib.util.spec_from_file_location("r9_local_causal_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "reset_episode"):
        module.decision_state.reset_episode = module.reset_episode
    return module.decision_state, {"name": str(path.relative_to(ROOT)), "sha256": sha256(path), "verified": False}


def current_actions(ep, times, vid, mask):
    """Issued controls available at input time, identity-bound across held ticks."""
    root = ep["path"]
    names = ("controller", "issued_acceleration_mps2", "issued_target_speed_mps")
    paths = [guard(root/(name+".npy")) for name in ("control_time_s", *names)]
    if not all(p.is_file() for p in paths):
        return None
    control_time = np.load(paths[0], allow_pickle=False)
    controls = {name: np.load(path, allow_pickle=False) for name, path in zip(names, paths[1:])}
    current = np.searchsorted(control_time, times[:-1], side="right")-1
    safe = current.clip(0, max(len(control_time)-1, 0))
    if not len(control_time):
        return None
    start_rows = np.searchsorted(times, control_time[safe], side="left")
    valid = (current >= 0)[:, None] & mask[:-1] & mask[start_rows] & (vid[:-1] == vid[start_rows])
    return {**{name: np.where(valid, value[safe], 0) for name, value in controls.items()}, "valid": valid}


def discover(sources, max_worlds=None):
    episodes, notes, seen = [], [], set()
    for source in sources:
        name, sep, directory = source.partition("=")
        if not sep or not name or not directory:
            raise ValueError("--source expects NAME=DIR")
        guard(ROOT / name)
        root = guard(directory)
        manifest = read_json(root / "C4_EPISODES_MANIFEST.json")
        if manifest.get("split") != "TRAIN":
            raise ValueError(f"{name}: only TRAIN source manifests accepted")
        entries = manifest.get("episodes", [])
        if isinstance(entries, dict):
            entries = list(entries.values())
        missing = []
        for item in entries:
            check_seed(item.get("seed", -1))
            path = guard(root / item["dir"])
            if not path.is_dir():
                missing.append(item.get("episode", item["dir"]))
                continue
            meta = read_json(path / "episode.json")
            check_seed(meta["seed"])
            world = str(meta["world"])
            guard(ROOT / world)
            if path in seen:
                raise ValueError(f"duplicate source episode: {path}")
            seen.add(path)
            episodes.append({**meta, "source_name": name, "path": path,
                             "manifest_role": item.get("split_role", "unspecified")})
        notes.append({"source": name, "manifest_sha256": sha256(root / "C4_EPISODES_MANIFEST.json"),
                      "unmaterialized_episodes": missing})
    worlds = sorted({ep["world"] for ep in episodes})
    if max_worlds is not None:
        notes.append({"max_worlds_excluded_worlds": worlds[max_worlds:]})
        worlds = worlds[:max_worlds]
    episodes = [e for e in episodes if e["world"] in worlds]
    if not episodes:
        raise ValueError("no materialized TRAIN episodes")
    return sorted(episodes, key=lambda e: (e["world"], e["source_name"], e["episode"])), notes


def geometry(kin, mask, slot):
    """All-target bumper intervals, front positions, no search radius/padding."""
    ego = kin[slot]
    target = mask & (kin[:, 2] == 1)
    target[slot] = False
    rows = kin[target]
    result = {k: 0.0 for k in FEATURES}
    result.update(v=float(ego[4]), a=float(ego[5]), has_l=False, has_f=False)
    if not np.isfinite(kin[mask]).all() or np.any(kin[mask, 6] <= 0):
        return result, False, "missing_geometry"
    if ego[0] < 0 or ego[0] > 75.44:
        return result, False, "outside_merge_zone"
    overlap = np.minimum(ego[0], rows[:, 0]) - np.maximum(ego[0] - ego[6], rows[:, 0] - rows[:, 6])
    physical = not np.any(overlap > 0)
    ds = rows[:, 0] - ego[0]
    for side, side_mask in (("l", ds >= 0), ("f", ds < 0)):
        if np.any(side_mask):
            idx = np.argmin(np.where(side_mask, abs(ds), np.inf))
            row = rows[idx]
            gap = row[0] - row[6] - ego[0] if side == "l" else ego[0] - ego[6] - row[0]
            result.update({"gap_" + side: float(gap), "v_" + side: float(row[4]),
                           "a_" + side: float(row[5]), "has_" + side: True})
    return result, physical, "ok" if physical else "target_interval_intersection"


def build_dataset(episodes, adapter=None):
    sequences, exclusions, diagnostics = [], [], []
    for ep in episodes:
        def load(name):
            return np.load(guard(ep["path"] / (name + ".npy")), allow_pickle=False)
        kin, mask, vid, cav, times = (load(k) for k in ("veh_kin", "veh_mask", "vid", "is_cav", "time_s"))
        mask, cav = mask.astype(bool), cav.astype(bool)
        if kin.shape != (*mask.shape, 7) or vid.shape != mask.shape or cav.shape != mask.shape:
            raise ValueError(f"{ep['episode']}: incompatible arrays")
        delta = np.diff(times)
        if len(delta) == 0 or not np.allclose(delta, delta[0], rtol=0, atol=1e-9) or delta[0] != .5:
            raise ValueError("P1 requires uniform dt=0.5 seconds")
        dt = float(delta[0])
        # Bound O(K^2) temporary geometry memory independently of episode length.
        old_chunks = {k: [] for k in FEATURES if k not in ("v", "a")}
        with torch.no_grad():
            for lo in range(0, len(times)-1, 32):
                hi = min(lo+32, len(times)-1)
                feats = core_features(torch.tensor(kin[lo:hi]).unsqueeze(0),
                                      torch.tensor(mask[lo:hi]).unsqueeze(0),
                                      SimpleNamespace(merge_lane=2, merge_target_lane=1))
                for key in old_chunks:
                    old_chunks[key].append(feats[key][0].numpy())
        old = {key: np.concatenate(values) for key, values in old_chunks.items()}
        if adapter is None:
            proposed = causal_proposal(kin[:-1], dt)
        else:
            if hasattr(adapter, "reset_episode"):
                adapter.reset_episode()
            actions = current_actions(ep, times, vid, mask)
            # The adapter sees ONLY the current snapshot/actions, never a future
            # state, lane label, event timestamp, world id, or driver truth.
            proposed = np.stack([np.asarray(adapter(
                kin=kin[row].copy(), present=mask[row].copy(), is_cav=cav[row].copy(), dt_s=dt,
                actions=None if actions is None else {k: v[row].copy() for k, v in actions.items()}))
                for row in range(len(times)-1)])
        if proposed.shape != kin[:-1].shape or not np.array_equal(proposed[..., 1:4], kin[:-1, :, 1:4]):
            raise ValueError("causal adapter must preserve pre-change lanes and shape")
        if not np.isfinite(proposed[mask[:-1]]).all():
            raise ValueError("nonfinite causal proposal")
        lane = np.where(kin[..., 1:4].sum(-1) > 0, kin[..., 1:4].argmax(-1), -1)
        warmup = float(ep.get("warmup_s", 0))
        counts = {"episode": ep["episode"], "candidate_cars": 0, "events": 0, "warmup_events": 0}
        for slot in range(mask.shape[1]):
            starts = np.flatnonzero(mask[:, slot] & np.r_[True, (~mask[:-1, slot]) | (vid[1:, slot] != vid[:-1, slot])])
            for start in starts:
                end = start + 1
                while end < len(times) and mask[end, slot] and vid[end, slot] == vid[start, slot]:
                    end += 1
                ramp = np.flatnonzero((lane[start:end, slot] == 2) & (kin[start:end, slot, 0] >= 0) &
                                     (kin[start:end, slot, 0] <= 75.44)) + start
                if not len(ramp):
                    continue
                entry = int(ramp[0])
                transitions = np.flatnonzero((lane[entry:end-1, slot] == 2) & (lane[entry+1:end, slot] == 1)) + entry + 1
                event_row = int(transitions[0]) if len(transitions) else None
                identity = {"world": ep["world"], "source": ep["source_name"], "episode": ep["episode"],
                            "vehicle": int(vid[start, slot]), "slot": slot, "entry_row": entry, "event_row": event_row}
                if event_row is not None and times[event_row] < warmup:
                    counts["warmup_events"] += 1
                    exclusions.append({**identity, "reason": "event_before_analysis_window"})
                    continue
                if event_row is None and times[end-1] < warmup:
                    exclusions.append({**identity, "reason": "risk_ended_before_analysis_window"})
                    continue
                if cav[entry, slot]:
                    exclusions.append({**identity, "reason": "CAV_not_supervised_neighbor_retained"})
                    continue
                terminal = event_row - 1 if event_row is not None else min(int(ramp[-1]), end-2, len(times)-2)
                if terminal < entry:
                    exclusions.append({**identity, "reason": "no_observed_transition_exposure"})
                    continue
                stop = next((r for r in range(entry, terminal+1) if lane[r, slot] != 2 or
                             not (0 <= kin[r, slot, 0] <= 75.44)), None)
                if stop is not None and event_row is None:
                    terminal = stop - 1
                if terminal < entry:
                    exclusions.append({**identity, "reason": "zero_risk_exposure"})
                    continue
                legacy = {k: [] for k in FEATURES}
                updated = {k: [] for k in FEATURES}
                physical, reasons, legacy_support = [], [], []
                for row in range(entry, terminal+1):
                    for k in FEATURES:
                        legacy[k].append(float(kin[row, slot, 4 if k == "v" else 5]) if k in ("v", "a") else old[k][row, slot])
                    feat, ok, reason = geometry(proposed[row], mask[row], slot)
                    if stop is not None and row >= stop:
                        ok, reason = False, "structural_exit_no_regain"
                    for k in FEATURES:
                        updated[k].append(feat[k])
                    physical.append(ok)
                    reasons.append(reason)
                    legacy_support.append(bool(lane[row, slot] == 2 and 0 <= kin[row, slot, 0] <= 75.44))
                event = event_row is not None
                unknown_age = entry == 0 or (entry == start and kin[entry, slot, 0] > 0)
                next_row = terminal + 1
                kind = ("event" if event else "horizon_censor" if next_row == len(times)-1 and
                        lane[next_row, slot] == 2 else "junction_exit" if lane[next_row, slot] == -1 and
                        mask[next_row, slot] else "teleport_or_identity_exit" if
                        terminal+2 >= end and end < len(times) else "route_exit")
                q = int(np.sum(mask[entry] & (lane[entry] == 2) & (kin[entry, :, 0] > kin[entry, slot, 0])))
                sequences.append({**identity, "legacy": legacy, "updated": updated, "physical": physical,
                    "physical_reasons": reasons, "event": event, "terminal_kind": kind,
                    "legacy_support": legacy_support,
                    "unknown_age": unknown_age, "queue": "front" if q == 0 else "behind",
                    "demand": ep.get("demand_vph"), "dt": dt,
                    "times": [float(times[aligned_row(row)] - times[entry]) for row in range(entry, terminal+1)],
                    "observed_s": float(times[event_row] - times[entry]) if event else None,
                    "horizon_s": float(ep.get("horizon_s", times[-1]-times[0]))})
                counts["candidate_cars"] += 1
                counts["events"] += int(event)
        diagnostics.append({**counts, "world": ep["world"], "manifest_role": ep["manifest_role"],
                            "input_sha256": {k: sha256(ep["path"]/(k+".npy")) for k in
                                             ("veh_kin", "veh_mask", "vid", "is_cav", "time_s")}})
        print(f"built {ep['episode']}: {counts['candidate_cars']} cars", flush=True)
    if not sequences:
        raise ValueError("no recorded ramp sequences")
    return sequences, {"exclusions": exclusions, "episodes": diagnostics,
                       "original_LCSTATE_event_denominator": 2915,
                       "LCSTATE_caveats": {"no_left_want_rows": 16, "teleports": 2,
                                          "cohort_identity_not_assumed": True}}


def pack(sequences):
    n, length = len(sequences), max(len(s["times"]) for s in sequences)
    batch = {"legacy": {}, "updated": {}}
    for mode in ("legacy", "updated"):
        for key in FEATURES:
            array = torch.zeros(n, length, dtype=torch.bool if key.startswith("has_") else DTYPE)
            for i, s in enumerate(sequences):
                array[i, :len(s["times"])] = torch.tensor(s[mode][key], dtype=array.dtype)
            batch[mode][key] = array
    batch["valid"] = torch.zeros(n, length, dtype=torch.bool)
    batch["physical"] = torch.zeros(n, length, dtype=torch.bool)
    batch["legacy_support"] = torch.zeros(n, length, dtype=torch.bool)
    batch["last"] = torch.tensor([len(s["times"])-1 for s in sequences])
    batch["event"] = torch.tensor([s["event"] for s in sequences])
    for i, s in enumerate(sequences):
        batch["valid"][i, :len(s["times"])] = True
        batch["physical"][i, :len(s["times"])] = torch.tensor(s["physical"])
        batch["legacy_support"][i, :len(s["times"])] = torch.tensor(s.get("legacy_support", [True]*len(s["times"])))
    return batch


def sequence_batches(sequences, size=128):
    for lo in range(0, len(sequences), size):
        yield sequences[lo:lo+size]


def score_losses(model, sequences, n_quad):
    with torch.no_grad():
        return [value for subset in sequence_batches(sequences)
                for value in model.nll(pack(subset), n_quad).tolist()]


class Candidate(nn.Module):
    def __init__(self, kind, initial):
        super().__init__()
        self.kind = kind
        values = dict(initial)
        if kind == "M0":
            for key, value in initial.items():
                self.register_buffer("frozen_"+key, torch.tensor(value, dtype=DTYPE))
        if kind in ("P1+P2", "P1-P3"):
            values = {"s_f": initial["s0"], "T_f": initial["T"], "b_f": initial["b"],
                      "s_l": initial["s0"], "T_l": initial["T"], "b_l": initial["b"]}
            values["tau"] = initial["sigma"]
            if kind == "P1+P2":
                values["temperature"] = initial["temperature"]
        self.raw = nn.ParameterDict({k: nn.Parameter(torch.tensor(math.log(math.expm1(v)) if v < 20 else v,
                                                               dtype=DTYPE), requires_grad=kind != "M0")
                                     for k, v in values.items()})

    def params(self):
        if self.kind == "M0":
            return {key: getattr(self, "frozen_"+key) for key in self.raw}
        return {k: F.softplus(v) for k, v in self.raw.items()}

    def distribution(self, batch, n_quad=8):
        p = self.params()
        f = batch["legacy" if self.kind in ("M0", "M0-refit") else "updated"]
        eligible = batch["legacy_support"] if self.kind in ("M0", "M0-refit") else batch["physical"]
        if self.kind not in ("P1+P2", "P1-P3"):
            return soft_hazard(f, p, n_quad, eligible)
        d_f = p["s_f"] + p["T_f"] * f["v_f"] + torch.relu(f["v_f"].square()-f["v"].square())/(2*p["b_f"])
        d_l = p["s_l"] + p["T_l"] * f["v"] + torch.relu(f["v"].square()-f["v_l"].square())/(2*p["b_l"])
        if self.kind == "P1-P3":
            sides = []
            for side, d in (("f", d_f), ("l", d_l)):
                gap = f["gap_" + side]
                positive = f["has_"+side] & (gap > 0)
                ratio = torch.where(positive, gap, torch.ones_like(gap)).log() - d.log()
                sides.append(torch.where(f["has_"+side], torch.where(positive, ratio,
                             torch.full_like(ratio, -math.inf)), torch.full_like(ratio, math.inf)))
            c = torch.where(eligible, torch.minimum(*sides), torch.full_like(d_f, -math.inf))
            return first_passage(c, p["tau"])
        nodes, weights = logistic_quadrature(n_quad)
        alpha = torch.exp(p["tau"] * nodes).unsqueeze(-1)
        ml = f["gap_l"].unsqueeze(1) - d_l.unsqueeze(1)/alpha
        mf = f["gap_f"].unsqueeze(1) - d_f.unsqueeze(1)/alpha
        ml = torch.where(f["has_l"].unsqueeze(1), ml, torch.full_like(ml, math.inf))
        mf = torch.where(f["has_f"].unsqueeze(1), mf, torch.full_like(mf, math.inf))
        margin = torch.minimum(ml, mf)
        finite = torch.isfinite(margin)
        logits = torch.where(finite, margin, torch.zeros_like(margin))/p["temperature"]
        logits = torch.where(finite, logits, margin)
        return hazard_distribution(logits, weights, eligible)

    def median_margin(self, batch):
        """Current eligible boundary at population median assertiveness (not F)."""
        p = self.params()
        f = batch["legacy" if self.kind in ("M0", "M0-refit") else "updated"]
        if self.kind in ("P1+P2", "P1-P3"):
            df = p["s_f"] + p["T_f"]*f["v_f"] + torch.relu(f["v_f"]**2-f["v"]**2)/(2*p["b_f"])
            dl = p["s_l"] + p["T_l"]*f["v"] + torch.relu(f["v"]**2-f["v_l"]**2)/(2*p["b_l"])
            ml = torch.where(f["has_l"], f["gap_l"]-dl, torch.full_like(dl, math.inf))
            mf = torch.where(f["has_f"], f["gap_f"]-df, torch.full_like(df, math.inf))
            return torch.minimum(ml, mf)
        return _core_margin(torch.ones((), dtype=DTYPE), *(f[k] for k in FEATURES),
                            p["s0"], p["T"], p["b"], p["tau_a"], None, None)

    def nll(self, batch, n_quad=8):
        dist = self.distribution(batch, n_quad)
        ix = batch["last"].unsqueeze(1)
        event = dist["log_mass"].gather(1, ix).squeeze(1)
        censor = dist["log_survival"].gather(1, ix).squeeze(1)
        return -torch.where(batch["event"], event, censor)


def optimize(kind, initial, sequences, *, iters, lr, shrinkage, n_quad):
    model = Candidate(kind, initial)
    if kind == "M0":
        return model, {"status": "untouched", "updates": 0}
    if not sequences:
        return model, {"status": "no_fit_cars", "updates": 0}
    if kind in ("P1", "P1+P2", "P1-P3"):
        failures = [s for s in sequences if s["event"] and not s["physical"][-1]]
        if failures:
            return model, {"status": "exact_support_failure", "updates": 0,
                           "physical_event_failures": len(failures),
                           "event_denominator": sum(s["event"] for s in sequences),
                           "trace": [], "reason": "100% physical event support required before optimization"}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trace, best, best_loss = [], copy.deepcopy(model.state_dict()), math.inf
    for step in range(iters):
        optimizer.zero_grad()
        total, unsupported = 0.0, 0
        for subset in sequence_batches(sequences):
            nll = model.nll(pack(subset), n_quad)
            unsupported += int((~torch.isfinite(nll)).sum())
            if unsupported:
                # No optimization of a supported-only subset, epsilon rescue, or
                # hidden deletion. Failed exact fits remain failed comparisons.
                model.load_state_dict(best)
                return model, {"status": "exact_support_failure", "updates": step,
                               "zero_likelihood_cars_at_least": unsupported, "trace": trace}
            contribution = nll.sum()/len(sequences)
            total += float(contribution.detach())
            contribution.backward()
        penalty = shrinkage * model.params()["tau"].square() if kind in ("P1+P2", "P1-P3") else None
        if penalty is not None:
            total += float(penalty.detach())
            penalty.backward()
        if total < best_loss:
            best_loss, best = total, copy.deepcopy(model.state_dict())
        gradients = {k: None if v.grad is None else float(v.grad) for k, v in model.raw.items()}
        if any(v is None or not math.isfinite(v) for v in gradients.values()):
            model.load_state_dict(best)
            return model, {"status": "nonfinite_gradient", "updates": step, "gradients": gradients, "trace": trace}
        if step == 0 or step == iters-1:
            trace.append({"step": step, "loss": total, "raw_gradients": gradients,
                          "finite_nonzero_gradient_count": sum(v != 0 for v in gradients.values())})
        optimizer.step()
    with torch.no_grad():
        value = float(np.mean(score_losses(model, sequences, n_quad)))
        if kind in ("P1+P2", "P1-P3"):
            value += float(shrinkage * model.params()["tau"].square())
        if math.isfinite(value) and value < best_loss:
            best_loss = value
            best = copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model, {"status": "fitted", "updates": iters, "objective": best_loss, "trace": trace}


def select_fit(kind, initial, sequences, args):
    worlds = sorted({s["world"] for s in sequences})
    folds = [worlds[i::args.inner_folds] for i in range(args.inner_folds)]
    lambdas = args.shrinkage if kind in ("P1+P2", "P1-P3") else [0.0]
    inner = []
    for lam in lambdas:
        scores, fold_reports = [], []
        for validation in folds:
            train = [s for s in sequences if s["world"] not in validation]
            val = [s for s in sequences if s["world"] in validation]
            if not train or not val:
                fold_reports.append({"worlds": validation, "status": "insufficient_worlds"})
                continue
            model, info = optimize(kind, initial, train, iters=args.iters, lr=args.lr,
                                   shrinkage=lam, n_quad=args.n_quad)
            with torch.no_grad():
                score = (float(np.mean(score_losses(model, val, args.n_quad)))
                         if info["status"] == "fitted" else math.inf)
            scores.append(score)
            fold_reports.append({"worlds": validation, "nll_per_car": score, "fit": info})
        mean = float(np.mean(scores)) if len(scores) == args.inner_folds else math.inf
        inner.append({"lambda": lam, "mean_validation_nll": mean, "folds": fold_reports})
    chosen = min(inner, key=lambda r: (r["mean_validation_nll"], r["lambda"]))
    selected = chosen["lambda"] if math.isfinite(chosen["mean_validation_nll"]) else 0.0
    model, info = optimize(kind, initial, sequences, iters=args.iters, lr=args.lr,
                           shrinkage=selected, n_quad=args.n_quad)
    return model, {"selected_lambda": selected, "inner": inner, "final_fit": info,
                   "selection_status": "ok" if math.isfinite(chosen["mean_validation_nll"]) else "unresolved"}


def median(values):
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    return values[n//2] if n % 2 else (values[n//2-1]+values[n//2])/2


def summary_nll(rows):
    if not rows:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p99": None, "max": None}
    x = sorted(r["nll"] for r in rows)
    return {"count": len(x), "mean": float(np.mean(x)), "max": x[-1],
            **{f"p{q}": x[max(0, math.ceil(q/100*len(x))-1)] for q in (50, 90, 99)}}


def population_cdf(sequences, distributions):
    known = [(s, d) for s, d in zip(sequences, distributions) if not s["unknown_age"]]
    if not known:
        return {"status": "unknown_entry", "quantiles": {}}
    grid = sorted({t for s, _ in known for t in s["times"]})
    model_lo, model_hi, obs_lo, obs_hi = [], [], [], []
    for t in grid:
        ml, mh, ol, oh = [], [], [], []
        for s, d in known:
            idx = int(np.searchsorted(s["times"], t, side="right"))-1
            value = 0.0 if idx < 0 else float(d[min(idx, len(d)-1)])
            ml.append(value)
            mh.append(1.0 if t > s["times"][-1] else value)
            occurred = s["event"] and s["observed_s"] <= t
            ol.append(float(occurred))
            oh.append(float(occurred or (not s["event"] and t > s["times"][-1])))
        model_lo.append(float(np.mean(ml))); model_hi.append(float(np.mean(mh)))
        obs_lo.append(float(np.mean(ol))); obs_hi.append(float(np.mean(oh)))
    def bound(low, high, q):
        early = next((t for t, f in zip(grid, high) if f >= q), math.inf)
        late = next((t for t, f in zip(grid, low) if f >= q), math.inf)
        return [early, late]
    quantiles = {}
    for q, limit in ((.25, 5), (.5, 5), (.9, 10)):
        mb, ob = bound(model_lo, model_hi, q), bound(obs_lo, obs_hi, q)
        identifiable = mb[0] == mb[1] and ob[0] == ob[1] and math.isfinite(mb[0]+ob[0])
        error = abs(mb[0]-ob[0]) if identifiable else None
        quantiles[str(int(q*100))] = {"model_bounds_s": mb, "empirical_bounds_s": ob,
                                    "identifiable": identifiable, "absolute_error_s": error,
                                    "gate": error <= limit if error is not None else None}
    return {"status": "recorded_path_bounds_not_free_running", "cars": len(known),
            "quantiles": quantiles,
            "curve": [{"t_s": t, "model": [ml, mh], "empirical": [ol, oh]} for t, ml, mh, ol, oh in
                      zip(grid, model_lo, model_hi, obs_lo, obs_hi)]}


def evaluate(model, sequences, n_quad):
    if not sequences:
        return {"cars": [], "nll": summary_nll([]), "status": "no_audit_worlds"}
    losses, distributions, margins = [], [], []
    with torch.no_grad():
        for subset in sequence_batches(sequences):
            batch = pack(subset)
            dist = model.distribution(batch, n_quad)
            losses.extend(model.nll(batch, n_quad).tolist())
            margin = model.median_margin(batch)
            for j, seq in enumerate(subset):
                length = len(seq["times"])
                distributions.append({k: v[j, :length].clone() for k, v in dist.items()})
                margins.append(float(margin[j, length-1]))
    rows, cdfs = [], []
    for i, s in enumerate(sequences):
        n = len(s["times"])
        dist = distributions[i]
        cdf = dist["cdf"].numpy()
        cdfs.append(cdf)
        hits = np.flatnonzero(cdf >= .5)
        reached = bool(len(hits))
        pred = s["times"][hits[0]] if reached else s["times"][-1]
        error_low = pred - s["observed_s"] if s["event"] and not s["unknown_age"] else None
        error_high = (error_low if reached else math.inf) if error_low is not None else None
        early = float(cdf[-2]) if n > 1 else 0.0
        zero = bool(s["event"] and torch.isneginf(dist["log_mass"][n-1]))
        rows.append({k: s[k] for k in ("world", "source", "episode", "vehicle", "entry_row", "event_row",
                                     "queue", "demand", "terminal_kind", "unknown_age", "horizon_s")})
        rows[-1].update(nll=losses[i], event=s["event"], physical_supported=bool(s["physical"][-1]),
                        physical_reason=s["physical_reasons"][-1], zero_mass_event=zero,
                        early_mass=early if s["event"] else None, median_reached=reached,
                        error_bounds_s=[error_low, error_high] if error_low is not None else None,
                        survival_at_stop=float(dist["log_survival"][n-1].exp()), stop_s=s["times"][-1],
                        median_driver_margin=margins[i], positive_margin=margins[i] > 0)
        tails = {}
        for horizon in (120, 300):
            idx = int(np.searchsorted(s["times"], horizon, side="right"))-1
            survival = float(dist["log_survival"][min(idx, n-1)].exp()) if idx >= 0 else 1.0
            if s["unknown_age"]:
                tails[str(horizon)] = {"identifiable_empirical": False, "model_bounds": [0.0, 1.0]}
            elif s["times"][-1] >= horizon:
                tails[str(horizon)] = {"identifiable_empirical": True,
                                       "empirical": float(not (s["event"] and s["observed_s"] <= horizon)),
                                       "model_bounds": [survival, survival]}
            elif s["event"]:
                tails[str(horizon)] = {"identifiable_empirical": True, "empirical": 0.0,
                                       "model_bounds": [0.0, survival]}
            elif s["terminal_kind"] != "horizon_censor":
                # Exit consumes all remaining non-merge mass; no future road path.
                tails[str(horizon)] = {"identifiable_empirical": True, "empirical": 1.0,
                                       "model_bounds": [survival, survival]}
            else:
                tails[str(horizon)] = {"identifiable_empirical": False, "model_bounds": [0.0, survival]}
        rows[-1]["tail_noncompletion"] = tails
    events = [r for r in rows if r["event"]]
    known_timing = [r["error_bounds_s"] for r in events if r["error_bounds_s"] is not None]
    timing = [r["error_bounds_s"] if r["error_bounds_s"] is not None else [-math.inf, math.inf] for r in events]
    abs_low = [0.0 if a <= 0 <= b else min(abs(a), abs(b)) for a, b in timing]
    abs_high = [max(abs(a), abs(b)) for a, b in timing]
    signed_bounds = [median([x[0] for x in timing]), median([x[1] for x in timing])]
    absolute_bounds = [median(abs_low), median(abs_high)]
    per_world = {w: summary_nll([r for r in rows if r["world"] == w]) for w in sorted({r["world"] for r in rows})}
    return {"status": "scored", "cars": rows, "nll": summary_nll(rows),
            "strata": {k: summary_nll([r for r in rows if r["terminal_kind"] == k]) for k in sorted({r["terminal_kind"] for r in rows})},
            "physical_support": {"event_denominator": len(events),
                "share": float(np.mean([r["physical_supported"] for r in events])) if events else None,
                "failures": [r for r in events if not r["physical_supported"]]},
            "zero_mass_event_count": sum(r["zero_mass_event"] for r in events),
            "positive_margin_event_share": float(np.mean([r["positive_margin"] for r in events])) if events else None,
            "early_mass_mean": float(np.mean([r["early_mass"] for r in events])) if events else None,
            "timing": {"event_denominator": len(events), "known_entry_events": len(known_timing),
                "unknown_age_events": len(events)-len(known_timing),
                "unreached_medians": sum(not r["median_reached"] for r in events),
                "signed_median_bounds_s": signed_bounds,
                "absolute_median_bounds_s": absolute_bounds,
                "signed_median_s": signed_bounds[0] if signed_bounds[0] == signed_bounds[1] else None,
                "median_absolute_s": absolute_bounds[0] if absolute_bounds[0] == absolute_bounds[1] else None},
            "per_world_nll": per_world,
            "worst_world": max(per_world, key=lambda w: per_world[w]["mean"]),
            "population_cdf": population_cdf(sequences, cdfs)}


def paired_comparisons(result, baseline, args):
    rows, refs = result["cars"], baseline["cars"]
    if len(rows) != len(refs):
        raise ValueError("paired audit cohort mismatch")
    out = {}
    for metric, event_only in (("nll", False), ("early_mass", True)):
        selected = [(r, b) for r, b in zip(rows, refs) if not event_only or r["event"]]
        if any(not math.isfinite(r[metric]) or not math.isfinite(b[metric]) for r, b in selected):
            out[metric] = {"status": "nonfinite_paired_losses", "estimate": None, "ci95": None}
        else:
            out[metric] = world_bootstrap([r[metric]-b[metric] for r, b in selected],
                                         [r["world"] for r, b in selected], args.bootstrap, args.seed, weighting="car")
    return out


def noncompletion(rows, args):
    """Fully observed 120/300s entry cohorts; post-event model tails stay bounds."""
    groups = {"all": rows}
    for field in ("source", "queue", "demand"):
        for value in sorted({str(r[field]) for r in rows}):
            groups[f"{field}={value}"] = [r for r in rows if str(r[field]) == value]
    result = {}
    for name, group in groups.items():
        result[name] = {"cars": len(group), "tails": {},
                        "competing_exits": sum(r["terminal_kind"] not in ("event", "horizon_censor") for r in group)}
        for horizon in (120, 300):
            key = str(horizon)
            eligible = [r for r in group if r["tail_noncompletion"][key]["identifiable_empirical"]]
            if not eligible:
                result[name]["tails"][key] = {"status": "unidentifiable", "cars": 0}
                continue
            empirical = world_bootstrap([r["tail_noncompletion"][key]["empirical"] for r in eligible],
                                        [r["world"] for r in eligible], args.bootstrap, args.seed, weighting="car")
            low = float(np.mean([r["tail_noncompletion"][key]["model_bounds"][0] for r in eligible]))
            high = float(np.mean([r["tail_noncompletion"][key]["model_bounds"][1] for r in eligible]))
            observed = empirical["estimate"]
            error = [0.0 if low <= observed <= high else min(abs(low-observed), abs(high-observed)),
                     max(abs(low-observed), abs(high-observed))]
            band = empirical["ci95"]
            result[name]["tails"][key] = {
                "status": "identified" if low == high else "bounded_no_post_event_continuation",
                "cars": len(eligible), "unidentifiable_cars": len(group)-len(eligible),
                "empirical_world_bootstrap": empirical, "model_noncompletion_bounds": [low, high],
                "absolute_error_bounds": error,
                "inside_empirical_band_certified": (low >= band[0] and high <= band[1]) if band else None}
    return result


def s3b_gates(kind, results):
    current = results[kind]["audit"]
    if not current["cars"]:
        return {"status": "blocked_no_audit", "passed": False}
    timing = current["timing"]
    signed = timing["signed_median_bounds_s"]
    absolute = timing["absolute_median_bounds_s"]
    def abs_range(bounds):
        a, b = bounds
        return [0.0 if a <= 0 <= b else min(abs(a), abs(b)), max(abs(a), abs(b))]
    signed_abs = abs_range(signed) if signed[0] is not None else [0, math.inf]
    gates = {"physical_support_100pct": current["physical_support"]["share"] == 1,
             "zero_mass_events": current["zero_mass_event_count"] == 0,
             "signed_median_within_2s": signed_abs[1] <= 2,
             "median_absolute_within_12s": absolute[1] is not None and absolute[1] <= 12,
             "full_event_timing_identifiable": timing["unknown_age_events"] == 0 and bool(timing["event_denominator"])}
    fitting = results[kind]["fit"]
    gates["fit_and_inner_selection_completed"] = bool(kind == "M0" or (
        fitting.get("final_fit", {}).get("status") == "fitted" and fitting.get("selection_status") == "ok"))
    for baseline in ("M0", "M0-refit"):
        pair = current["paired_vs"][baseline]["nll"]
        gates["nll_CI_below_0_vs_"+baseline] = bool(pair.get("status") == "ok" and pair["ci95"][1] < 0)
        ref = results[baseline]["audit"]["timing"]
        sb, ab = ref["signed_median_bounds_s"], ref["absolute_median_bounds_s"]
        gates["signed_median_strictly_better_vs_"+baseline] = bool(sb[0] is not None and signed_abs[1] < abs_range(sb)[0])
        gates["median_absolute_no_worse_vs_"+baseline] = bool(ab[0] is not None and absolute[1] is not None and absolute[1] <= ab[0])
        for horizon in ("120", "300"):
            tail = current["noncompletion"]["all"]["tails"][horizon]
            other = results[baseline]["audit"]["noncompletion"]["all"]["tails"][horizon]
            gates[f"noncompletion_no_worse_{horizon}_vs_{baseline}"] = bool(
                "absolute_error_bounds" in tail and "absolute_error_bounds" in other and
                tail["absolute_error_bounds"][1] <= other["absolute_error_bounds"][0])
    early = current["paired_vs"]["M0"]["early_mass"]
    gates["early_mass_CI_below_0_vs_M0"] = bool(early.get("status") == "ok" and early["ci95"][1] < 0)
    for q in ("25", "50", "90"):
        metric = current["population_cdf"]["quantiles"].get(q, {})
        gates["population_CDF_p"+q] = metric.get("gate") is True
    for horizon in ("120", "300"):
        gates["noncompletion_in_band_"+horizon] = current["noncompletion"]["all"]["tails"][horizon].get("inside_empirical_band_certified") is True
    return {"status": "passed" if all(gates.values()) else "CHANGES", "passed": all(gates.values()), "gates": gates}


def clean_json(value):
    """Strict RFC JSON: infinities are explicit strings, never discarded."""
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return "+Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def legacy_main(argv=None):
    """Superseded scalar diagnostics only; all exported M0 labels are approximate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="NAME=DIR converted TRAIN episode source")
    parser.add_argument("--models", nargs="+", default=list(MODELS), help=", or space separated: " + ", ".join(MODELS))
    parser.add_argument("--audit-frac", type=float, default=.2)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--round8-params", default="default")
    parser.add_argument("--out", required=True)
    parser.add_argument("--md", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-worlds", type=int)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=.03)
    parser.add_argument("--n-quad", type=int, default=8)
    parser.add_argument("--shrinkage", nargs="+", type=float, default=[0, .001, .01, .1, 1.0])
    parser.add_argument("--causal-adapter", help="workspace file exposing decision_state; see R9_FIT.md")
    args = parser.parse_args(argv)
    args.models = [v for item in args.models for v in item.split(",")]
    if len(set(args.models)) != len(args.models) or any(k not in MODELS for k in args.models):
        parser.error("unknown or duplicate model name")
    if args.audit_frac != .2 or args.inner_folds < 2 or args.bootstrap < 0 or args.iters < 1 or not (0 < args.lr < math.inf):
        parser.error("requires audit-frac 0.2, inner-folds >=2, bootstrap >=0, iters >=1, positive finite lr")
    if not 1 <= args.n_quad <= 64 or (args.max_worlds is not None and args.max_worlds < 1):
        parser.error("n-quad must be 1..64 and max-worlds positive")
    if 0 not in args.shrinkage or any(not math.isfinite(x) or x < 0 for x in args.shrinkage):
        parser.error("shrinkage candidates must be finite nonnegative and include zero")
    args.seed = check_seed(args.seed)
    out, md = guard(args.out), guard(args.md)
    if out == md:
        parser.error("--out and --md must differ")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    initial, identity = load_params(args.round8_params)
    adapter, adapter_info = load_adapter(args.causal_adapter)
    episodes, source_notes = discover(args.source, args.max_worlds)
    sequences, data = build_dataset(episodes, adapter)
    worlds = sorted({e["world"] for e in episodes})
    audit = worlds[4::5]
    fit = [s for s in sequences if s["world"] not in audit]
    audit_sequences = [s for s in sequences if s["world"] in audit]
    results = {}
    kinds = list(dict.fromkeys(["M0", "M0-refit"] + args.models))
    for kind in kinds:
        print(f"fitting {kind}; {len(fit)} fit / {len(audit_sequences)} audit cars", flush=True)
        if kind == "M0":
            model, info = Candidate(kind, initial), {"status": "untouched", "updates": 0}
        else:
            model, info = select_fit(kind, initial, fit, args)
        results[kind] = {"parameters": {k: float(v.detach()) for k, v in model.params().items()},
                         "fit": info, "fit_diagnostics_not_audit": evaluate(model, fit, args.n_quad),
                         "audit": evaluate(model, audit_sequences, args.n_quad)}
    blockers = ["P1 simulator decision-substep parity is unresolved; causal adapter is not qualification evidence",
                "Converted lengths are nominal; exact bumper/projection geometry needs runtime parity",
                "S1/S4-S6 runtime, full-cohort and closed-loop evidence is outside this offline packet"]
    if adapter is None:
        blockers.append("Default constant-current-acceleration proposal is diagnostic; supply frozen longitudinal adapter")
    if not audit:
        blockers.append("Fewer than five worlds: fixed one-in-five audit is empty (sample smoke only)")
    if any(n.get("unmaterialized_episodes") for n in source_notes):
        blockers.append("Source manifests contain unmaterialized episodes; exclusions enumerated")
    for result in results.values():
        for partition in ("audit", "fit_diagnostics_not_audit"):
            ev = result[partition]
            ev["noncompletion"] = noncompletion(ev["cars"], args)
            ev["nll_world_bootstrap"] = world_bootstrap([r["nll"] for r in ev["cars"]],
                [r["world"] for r in ev["cars"]], args.bootstrap, args.seed, weighting="car")
    for kind, result in results.items():
        ev = result["audit"]
        ev["paired_vs"] = {base: paired_comparisons(ev, results[base]["audit"], args) for base in ("M0", "M0-refit")}
        if ev.get("zero_mass_event_count", 0):
            blockers.append(f"{kind}: {ev['zero_mass_event_count']} exact zero-mass audit events")
        fit_info = result["fit"].get("final_fit", result["fit"])
        if fit_info.get("status") not in ("untouched", "fitted"):
            blockers.append(f"{kind}: fitting status {fit_info.get('status')}")
        if kind != "M0" and result["fit"].get("selection_status") != "ok":
            blockers.append(f"{kind}: grouped inner validation unresolved")
    for kind, result in results.items():
        result["s3b"] = s3b_gates(kind, results)
    physical_events = [s for s in sequences if s["event"]]
    data["physical_support_all_recorded_events"] = {
        "denominator": len(physical_events),
        "share": float(np.mean([s["physical"][-1] for s in physical_events])) if physical_events else None,
        "failures": [{k: s[k] for k in ("world", "source", "episode", "vehicle", "event_row")} |
                     {"reason": s["physical_reasons"][-1]} for s in physical_events if not s["physical"][-1]]}
    data["unknown_age_sequences"] = [{k: s[k] for k in ("world", "source", "episode", "vehicle", "entry_row")}
                                     for s in sequences if s["unknown_age"]]
    data["alignment"] = {"event_timestamp": "first post-merge/authorizing output row",
                         "input_timestamp": "preceding recorded row, causal proposal once; pre-change membership",
                         "post_event_rows_used": 0, "recorded_future_motion_used": False}
    report = {"schema": "r9_offline_fit/2", "scope": "A-only; P4/B off", "verdict": "CHANGES",
              "seed": args.seed, "round8": identity, "initial_parameters": initial,
              "causal_proposal": adapter_info, "sources": source_notes,
              "config": vars(args),
              "split": {"worlds": worlds, "audit_worlds": audit,
                        "fit_worlds": [w for w in worlds if w not in audit],
                        "rule": "sorted world ids, index 4::5; all arms/windows grouped"},
              "data": {**data, "cars": len(sequences), "events": sum(s["event"] for s in sequences)},
              "models": results, "qualification_blockers": blockers,
              "semantics": {"M0": "untouched context-free round-8 soft hazard on pre-update geometry",
                "M0-refit": "refitted context-free round-8 soft hazard on pre-update geometry",
                "P1": "causal proposal + all-neighbor physical mask; legacy soft hazard boundary",
                "P1+P2": "causal proposal + separate learned gap sides; persistent logistic latent soft hazard",
                "P1-P3": "P2 boundary; persistent logistic latent; exact eligible running maximum",
                "sequence_latent_integration": "M0/M0-refit/P1: persistent Gauss-Hermite normal; P1+P2: persistent Gauss-Legendre logistic quantiles; P1-P3: analytic logistic law",
                "physical_mask_repair": "P1 exact interval intersection reported separately; M0 not retrofitted",
                "unknown_age": "enumerated and masked for entry timing; likelihood uses observed-start prior",
                "competing_exits": "separate exit strata, survival to observed exit; no merge invented"}}
    def approx_labels(value):
        if isinstance(value, dict):
            return {approx_labels(k): approx_labels(v) for k, v in value.items()}
        if isinstance(value, list):
            return [approx_labels(v) for v in value]
        if isinstance(value, str):
            return value.replace("M0", "M0-approx")
        return value
    report = approx_labels(clean_json(report))
    text = ["# Round-9 offline fit", "", "Verdict: **CHANGES** (offline evidence only)", "",
            f"Cars: {len(sequences)}; fit worlds: {len(worlds)-len(audit)}; audit worlds: {len(audit)}.", "",
            "## Qualification blockers", *["- " + b for b in blockers], ""]
    for kind, result in report["models"].items():
        text.extend(["## " + kind, "", "- Parameters: `" + json.dumps(result["parameters"]) + "`",
                     "- Audit NLL: `" + json.dumps(result["audit"]["nll"]) + "`",
                     "- Timing bounds: `" + json.dumps(result["audit"].get("timing")) + "`",
                     "- Paired world bootstrap: `" + json.dumps(result["audit"].get("paired_vs")) + "`",
                     "- Physical support: `" + json.dumps({k: v for k, v in result["audit"].get("physical_support", {}).items() if k != "failures"}) + "`",
                     "- Zero-mass audit events: `" + str(result["audit"].get("zero_mass_event_count")) + "`",
                     "- Positive-margin event share: `" + str(result["audit"].get("positive_margin_event_share")) + "`",
                     "- Worst world: `" + str(result["audit"].get("worst_world")) + "`",
                     "- Population CDF quantiles: `" + json.dumps(result["audit"].get("population_cdf", {}).get("quantiles")) + "`",
                     "- Noncompletion: `" + json.dumps(result["audit"]["noncompletion"].get("all")) + "`",
                     "- S3b gates: `" + json.dumps(result["s3b"]) + "`", ""])
        diagnostic = result["fit_diagnostics_not_audit"]
        text.extend(["### Fit diagnostics (not audit evidence)", "",
                     "- NLL: `" + json.dumps(diagnostic["nll"]) + "`",
                     "- Zero-mass events: `" + str(diagnostic.get("zero_mass_event_count")) + "`", ""])
    for path, content in ((out, json.dumps(report, indent=2, allow_nan=False)+"\n"), (md, "\n".join(text))):
        guard(path.parent).mkdir(parents=True, exist_ok=True)
        guard(path).write_text(content, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} and {md.relative_to(ROOT)}; verdict CHANGES", flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--legacy-approx" in argv:
        argv.remove("--legacy-approx")
        return legacy_main(argv)
    from tools.merge_fit_stage2a import main as stage2a_main
    return stage2a_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
