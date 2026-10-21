"""CPU, TRAIN-only exact-runtime first-passage support audit (no fitting).

Replay each episode once; reuse its frozen causal geometry for nonoverlapping
windows and their strictly pre-window survival prefixes. CSV includes every
identity-continuous recorded 2->1 transition, including unmodelled CAV events.
Recorded successor positions/lanes are diagnostics/labels only, never decisions.
"""
# The executable adds this worktree's src before importing the application.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from wms.data.c4_episodes import EpisodeArrays, read_source_manifest
from wms.data.recorded_prefix import causal_scene
from wms.model import ModelSpec, WmSModel
from wms.model.checkpoint import load_checkpoint
from wms.model.merge_pivot import log_event_mass, log_survival
from wms.physics.config import PhysicsConfig
from wms.train.losses.merge_pivot import (
    MergePivotConfig, build_recorded_geometry, certify_pre_window, pre_window_maximum, score_chunk,
)

FORBIDDEN = ("holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")


def local_path(path) -> Path:
    """Check lexical AND resolved paths before any access (including symlinks)."""
    path = Path(path)
    for candidate in (path, path.resolve()):
        low = str(candidate).lower()
        if any(word in low for word in FORBIDDEN) or "sealed" in (
                part.lower() for part in candidate.parts):
            raise ValueError("restricted audit path")
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("audit inputs and outputs must be inside this worktree")
    return path


def check_seed(entry):
    seeds = [int(entry["seed"])] if "seed" in entry else []
    for field in ("world", "episode", "dir"):
        seeds += [int(s) for s in re.findall(r"(?:^|[/_w])(?:s)?(\d{6})(?=[_m/]|$)",
                                           str(entry.get(field, "")))]
    if not seeds:
        raise ValueError("episode has no verifiable TRAIN seed")
    if any(964000 <= seed <= 964199 for seed in seeds):
        raise ValueError("evaluation seeds 964000-964199 are refused")


def number(value):
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, float) and not math.isfinite(value):
        return "-inf" if value == -math.inf else "inf" if value == math.inf else "nan"
    return value


def clean(value):
    if isinstance(value, dict):
        return {key: clean(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(val) for val in value]
    return number(value)


def runtime_replay(model, rec, present, cav, cont, timeline):
    clock = time.perf_counter()
    inp = SimpleNamespace(states=rec, present=present, is_cav=cav, continuity=cont,
                          hist=None, hist_mask=None, observed_hist=None,
                          observed_hist_mask=None, controller_id=None,
                          action_timeline=timeline,
                          replay_progress=lambda i, n: print(
                              f"replay {i}/{n} ({time.perf_counter() - clock:.1f}s)",
                              file=sys.stderr, flush=True))
    if model.long_history:
        from wms.data.history import long_history_features
        hist, mask = long_history_features(
            rec[0, :1], present[0, :1], cont[0, :0], present[0, 0],
            history_s=model.history_s, history_stride_s=model.history_stride_s,
            dt_s=model.physics.dt_s)
        inp.observed_hist, inp.observed_hist_mask = hist[None], mask[None]
    return build_recorded_geometry(model, inp, SimpleNamespace(n_steps=rec.shape[1] - 1))


def event_details(geo, raw, vid, cav, t, k, params):
    def vehicle(j):
        if j < 0:
            return None
        s, length = float(geo["s"][0, t, j]), float(geo["length"][0, t, j])
        return dict(slot=j, vehicle_id=int(vid[t, j]), is_cav=bool(cav[0, t, j]),
                    pre_lane=int(geo["lane"][0, t, j]), front=s, rear=s - length,
                    length=length, v=float(geo["v"][0, t, j]),
                    recorded_pre_front=float(raw[t, j, 0]),
                    recorded_post_front=float(raw[t + 1, j, 0]),
                    recorded_post_lane=(int(raw[t + 1, j, 1:4].argmax())
                                        if raw[t + 1, j, 1:4].sum() else -1),
                    predicted_minus_recorded_front=s - float(raw[t + 1, j, 0]))
    d = {key: number(val[0, t, k]) for key, val in geo.items()}
    d.update(ego=vehicle(k), follower=vehicle(int(geo["index_f"][0, t, k])),
             leader=vehicle(int(geo["index_l"][0, t, k])),
             blocker=vehicle(int(geo["overlap_index"][0, t, k])),
             fixed_blocker=vehicle(int(geo["fixed_overlap_index"][0, t, k])))
    ego = d["ego"]
    d["all_pre_target_intersections"] = [vehicle(j) for j in range(raw.shape[1])
        if j != k and bool(geo["pre_target"][0, t, j])
        and max(ego["rear"], float(geo["s"][0, t, j] - geo["length"][0, t, j]))
        < min(ego["front"], float(geo["s"][0, t, j]))]
    v = float(geo["v"][0, t, k])
    for side in ("f", "l"):
        other = float(geo["v_" + side][0, t, k])
        fast, slow = (other, v) if side == "f" else (v, other)
        d["required_" + side] = (params["s_" + side] + params["T_" + side] * fast
                                    + max(fast * fast - slow * slow, 0.) / (2 * params["b_" + side]))
    return clean(d)


def diagnose(rows):
    """Diagnostic successor projections never alter support or enter replay."""
    out = Counter()
    for row in rows:
        if row["is_cav"] or row["support_verdict"] != "vetoed":
            continue
        d = json.loads(row["geometry_json"])
        ego, blocker = d.get("ego"), d.get("blocker")
        if ego is None or blocker is None:
            continue
        if blocker["recorded_post_lane"] == -1:
            out["veto_blocker_successor_is_junction"] += 1
        elif blocker["recorded_post_lane"] != blocker["pre_lane"]:
            out["veto_blocker_successor_leaves_target_lane"] += 1
        es, bs = ego["recorded_post_front"], blocker["recorded_post_front"]
        if max(es - ego["length"], bs - blocker["length"]) < min(es, bs):
            out["veto_also_intersects_in_recorded_successor_projection"] += 1
        else:
            out["veto_only_in_predicted_motion_projection"] += 1
    return dict(out)


@torch.no_grad()
def audit_episode(model, ep, entry, source, window_s=120., start_s=None, cache=None):
    raw = np.asarray(ep.array("veh_kin"), np.float32)
    present_np = np.asarray(ep.array("veh_mask"), bool)
    cont_np = np.asarray(ep.array("target_mask")[:len(raw) - 1], bool)
    rec = torch.from_numpy(causal_scene(raw, present_np, cont_np))[None]
    present = torch.tensor(present_np)[None]
    cav = torch.tensor(np.asarray(ep.array("is_cav"), bool))[None]
    cont = torch.tensor(cont_np)[None]
    times = np.asarray(ep.array("time_s"))
    dt = float(model.physics.dt_s)
    if not np.allclose(np.diff(times), dt, rtol=0, atol=1e-6):
        raise ValueError("episode/model native-tick mismatch")
    controls = np.asarray(ep.array("control_time_s"))
    if not np.allclose(controls, times[0] + np.arange(len(controls)), rtol=0, atol=1e-6):
        raise ValueError("audit requires complete recorded 1 s control clock")
    timeline = {key: torch.tensor(np.asarray(ep.array(name)))[None] for key, name in (
        ("controller", "controller"), ("acceleration_mps2", "issued_acceleration_mps2"),
        ("target_speed_mps", "issued_target_speed_mps"))}
    timeline["control_dt_s"] = 1.
    geo = None
    if cache is not None:
        digest = hashlib.sha256()
        for tensor in (*model.state_dict().values(), rec, present, cav, cont,
                       *(val for val in timeline.values() if torch.is_tensor(val))):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        for path in (ROOT / "src/wms/model/merge_pivot.py", ROOT / "src/wms/model/wms.py",
                     ROOT / "src/wms/train/cav_drive.py", ROOT / "src/wms/train/losses/merge_pivot.py"):
            digest.update(path.read_bytes())
        digest.update(str(model.physics).encode())
        cache = local_path(cache / (digest.hexdigest() + ".pt"))
        if cache.is_file():
            geo = torch.load(cache, map_location="cpu", weights_only=True)
    if geo is None:
        geo = runtime_replay(model, rec, present, cav, cont, timeline)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(geo, cache)
    c = model.merge_pivot(geo)
    tau = model.merge_pivot.params()["tau"]
    params = {key: float(val) for key, val in model.merge_pivot.params().items()}
    cfg = MergePivotConfig(merge_lane=model.physics.merge_lane,
                           target_lane=model.physics.merge_target_lane,
                           zone_s0_m=model.physics.merge_zone_s0_m,
                           zone_s1_m=model.physics.merge_zone_s1_m)
    lane = rec[..., 1:4].argmax(-1)
    pre_risk = (present & (rec[..., 1:4].sum(-1) > 0) & (lane == cfg.merge_lane)
                & (rec[..., 0] >= cfg.zone_s0_m) & (rec[..., 0] <= cfg.zone_s1_m))
    events = (present[:, :-1] & present[:, 1:] & cont
              & (lane[:, :-1] == cfg.merge_lane) & (lane[:, 1:] == cfg.target_lane))
    events &= torch.tensor((raw[:-1, :, 1:4].sum(-1) > 0)
                           & (raw[1:, :, 1:4].sum(-1) > 0))[None]
    own = ~cav[:, :-1]
    risk = (geo["ramp"] | pre_risk[:, :-1] | events) & own
    n = c.shape[1]
    start = int(np.searchsorted(times, entry.get("warmup_s", 0.) if start_s is None else start_s))
    size = int(round(window_s / dt))
    if size < 1:
        raise ValueError("window length must be positive")
    vid = ep.array("vid")
    rows, counts = [], Counter()
    for lo in range(start, n, size):
        hi = min(lo + size, n)
        reset = torch.cat((torch.zeros_like(present[:, :1]),
                           ~cont[:, lo:hi - 1] | ~present[:, lo:hi - 1]), 1)
        prior_outside = torch.cat((~pre_risk[:, lo:lo + 1],
                                   present[:, lo:hi - 1] & ~pre_risk[:, lo:hi - 1]), 1)
        known_before = ~pre_risk[:, lo:hi] | prior_outside
        initial = torch.full_like(c[:, 0], torch.nan)
        if lo:
            prefix_geo = {key: val[:, :lo] for key, val in geo.items()}
            history = certify_pre_window(rec[:, :lo + 1], present[:, :lo + 1],
                                         cont[:, :lo], prefix_geo, cfg)
            initial = pre_window_maximum(model, [history], c[:, 0])
        kw = dict(risk=risk[:, lo:hi], event=events[:, lo:hi] & own[:, lo:hi],
                  reset=reset, eligible=geo["eligible"][:, lo:hi])
        for label, prior in (("before", None), ("after", initial)):
            _, _, stats = score_chunk(c[:, lo:hi], tau, known_entry=known_before,
                                       initial_maximum=prior, **kw)
            for key in ("unknown_age", "unknown_age_events", "events", "zero_mass_events",
                        "positive_mass_events", "zero_survival", "scored_intervals"):
                counts[label + "_" + key] += stats[key]
        # Per-event exact mass, including unsupported and unknown-age observations.
        maximum = torch.where(torch.isnan(initial), -torch.inf, initial)
        max_tick = torch.full_like(maximum, -1, dtype=torch.long)
        if lo:
            matching = history["history_mask"] & (c[:, :lo] == initial.unsqueeze(1))
            hit = matching.any(1) & (initial > -torch.inf)
            max_tick = torch.where(hit, matching.long().argmax(1), max_tick)
        known = torch.zeros_like(present[:, 0])
        old_known = known.clone()
        started, terminal = known.clone(), known.clone()
        for t in range(lo, hi):
            i = t - lo
            fresh = reset[:, i]
            maximum = torch.where(fresh, -torch.inf, maximum)
            max_tick = torch.where(fresh, -1, max_tick)
            started, terminal, known, old_known = (v & ~fresh for v in
                                                   (started, terminal, known, old_known))
            born = risk[:, t] & ~started & ~terminal
            old_known = torch.where(born, known_before[:, i], old_known)
            known = torch.where(born, known_before[:, i] | ((~torch.isnan(initial)) & (i == 0)), known)
            terminal |= started & ~risk[:, t]
            started |= born
            active = risk[:, t] & ~terminal
            updated = torch.where(active, torch.maximum(maximum, c[:, t]), maximum)
            logmass = log_event_mass(maximum, updated, tau)
            for k in torch.nonzero(events[0, t]).flatten().tolist():
                is_cav = bool(cav[0, t, k])
                physical = bool(geo["physical_eligible"][0, t, k])
                eligible = bool(geo["eligible"][0, t, k])
                reason = []
                for key, negate, text in (("scene_valid", True, "unobserved_scene"),
                    ("ramp", True, "outside_decision_zone"), ("overlap", False, "body_intersection"),
                    ("fixed_overlap", False, "fixed_cav_intersection"),
                    ("reservation_veto", False, "simultaneous_reservation")):
                    if bool(geo[key][0, t, k]) != negate:
                        reason.append(text)
                if is_cav:
                    reason.append("cav_not_learned")
                zero = bool(torch.isneginf(logmass[0, k]))
                details = event_details(geo, raw, vid, cav, t, k, params) if (not eligible or zero) else {}
                details.update(c=number(c[0, t, k]), previous_maximum=number(maximum[0, k]),
                               log_event_mass=number(logmass[0, k]),
                               entry_maximum=number(initial[0, k]),
                               log_survival_to_entry=(number(log_survival(initial[0, k], tau))
                                                      if not torch.isnan(initial[0, k]) else None))
                if zero:
                    details["zero_mass_reason"] = ("absorbed_before_event" if terminal[0, k] else
                                                  "physical_veto" if not eligible else
                                                  "event_does_not_raise_running_maximum")
                    earlier = int(max_tick[0, k])
                    if earlier >= 0:
                        details["record_setting_prestate_time_s"] = float(times[earlier])
                        details["record_setting_geometry"] = event_details(
                            geo, raw, vid, cav, earlier, k, params)
                rows.append(dict(source=source, episode=entry["episode"],
                    event_id=f'{source}/{entry["episode"]}/{int(vid[t, k])}/{t + 1}',
                    vehicle_id=int(vid[t, k]), slot=k, event_time_s=float(times[t + 1]),
                    window_start_s=float(times[lo]), is_cav=is_cav,
                    support_verdict="cav_not_modelled" if is_cav else "supported" if eligible else "vetoed",
                    physical_supported=physical, veto_reason=";".join(reason),
                    unknown_age_before=not bool(old_known[0, k]), unknown_age_after=not bool(known[0, k]),
                    scored_event_after=bool(active[0, k] & known[0, k]) and not is_cav,
                    absorbed_before_event=bool(terminal[0, k]) and not is_cav,
                    zero_mass=zero, gap_f=number(geo["gap_f"][0, t, k]),
                    gap_l=number(geo["gap_l"][0, t, k]),
                    geometry_json=json.dumps(clean(details), sort_keys=True, allow_nan=False)))
            max_tick = torch.where(active & (c[:, t] > maximum), t, max_tick)
            maximum = updated
            terminal |= events[:, t] & own[:, t]
    hdv = [row for row in rows if not row["is_cav"]]
    counts.update(recorded_merges=len(rows), hdv_events=len(hdv), cav_merges=len(rows) - len(hdv),
                  physically_supported=sum(row["physical_supported"] for row in hdv),
                  runtime_supported=sum(row["support_verdict"] == "supported" for row in hdv),
                  all_hdv_zero_mass=sum(row["zero_mass"] for row in hdv),
                  absorbed_before_event=sum(row["absorbed_before_event"] for row in hdv), episodes=1)
    return rows, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--checkpoint", type=local_path)
    parser.add_argument("--output", type=local_path, default=ROOT / "_out/r9_support_audit")
    parser.add_argument("--window-s", type=float, default=120.)
    parser.add_argument("--start-s", type=float)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    if args.checkpoint:
        # Avoid implicit sidecar access: explicit physics is used unless embedded.
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        physics = payload.get("config", {}).get("physics")
        if not physics:
            sidecar = local_path(args.checkpoint.with_name("physics.json"))
            if not sidecar.is_file():
                parser.error("checkpoint needs embedded physics or workspace-local physics.json")
            physics = json.loads(sidecar.read_text(encoding="utf-8"))
        local_path(args.checkpoint.with_name("lane_calib.json"))
        loaded = load_checkpoint(args.checkpoint, physics=physics,
                                 model_config={"merge_decision": "first_passage"})
        model, identity = loaded.model, dict(checkpoint_sha256=loaded.sha256,
                                            initialization="checkpoint")
    else:
        model = WmSModel(ModelSpec(rollout_mode="hybrid", lane_model="hazard_v5",
                                  merge_decision="first_passage"), physics=PhysicsConfig()).eval()
        identity = dict(initialization="untrained_WmSModel_seed0_defaults",
                        qualification=False,
                        note="No round-8 checkpoint supplied; not a reproduction of gpuR.")
    if model.physics.dt_s != .5:
        parser.error("audit requires the registered 0.5 s native clock")
    rows, counts, sources = [], Counter(), []
    try:
        for source in args.source:
            name, path = source.split("=", 1)
            root = local_path(path)
            doc = read_source_manifest(root)
            sources.append(dict(name=name, path=str(root)))
            for entry in doc["episodes"]:
                check_seed(entry)
                directory = local_path(root / entry["dir"])
                if not directory.is_dir():
                    counts["manifest_episodes_not_present"] += 1
                    continue
                if args.max_episodes is not None and counts["episodes"] >= args.max_episodes:
                    break
                # EpisodeArrays opens a fixed observation/action whitelist, no truth tables.
                from wms.data.c4_episodes import ARRAYS, QUEUE_ARRAYS
                for array in (*ARRAYS, *QUEUE_ARRAYS):
                    local_path(directory / (array + ".npy"))
                found, summary = audit_episode(model, EpisodeArrays(directory), entry, name,
                                               args.window_s, args.start_s,
                                               cache=args.output / ".replay_cache")
                rows.extend(found)
                counts.update(summary)
        if not counts["episodes"]:
            parser.error("no workspace-local TRAIN episodes are present")
        args.output = local_path(args.output)
        args.output.mkdir(parents=True, exist_ok=True)
        if rows:
            with local_path(args.output / "events.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        summary = dict(schema="r9_support_audit/1", **identity, sources=sources,
                       parameters={key: float(val.detach()) for key, val in model.merge_pivot.params().items()},
                       window_s=args.window_s, counts=dict(counts),
                       physical_support_share=counts["physically_supported"] / counts["hdv_events"]
                       if counts["hdv_events"] else None,
                       zero_mass_count_at_init=counts["all_hdv_zero_mass"],
                       diagnostic_counts=diagnose(rows),
                       veto_reasons=dict(Counter(row["veto_reason"] for row in rows
                                                if not row["is_cav"] and row["veto_reason"])))
        local_path(args.output / "summary.json").write_text(
            json.dumps(clean(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(clean(summary), allow_nan=False))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
