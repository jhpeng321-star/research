"""Causal observations and independently reconciled labels from explicit tapes.

No SUMO imports. Future request schedules belong exclusively to the label path.
The projection is a NumPy port of belief.detection_mask + RsuZones, using nominal
length (never the recorded length_true_m). test_reconstruct checks helper parity.
"""
import gzip
import json
import random

import numpy as np

from .audit import safe
from .core import EDGES, action_summary, boolean, finite

SENSOR_RANGE = 100.
LENGTH = 5.
ZONES = ((-800., 225.44), (-800., 225.44), (-378.66, 75.44))
BOUNDARIES = (-2645.67, -2645.67, -378.66)
SPEED_LIMIT = 31.29
OBS_KEYS = ("time_s", "s_m", "lane", "v_mps", "a_mps2", "present", "is_cav", "vid_index")


def project(kin, present, cav, flow_counts, time_s):
    """The same current-frame observation operator for real and imagined states."""
    kin = np.asarray(kin, dtype=float)
    n = len(kin)
    if kin.shape != (n, 4):
        raise ValueError("expected [N,4] position/lane/speed/acceleration")
    present = boolean(present, (n,))
    cav = boolean(cav, (n,)) & present
    s, lane = kin[:, 0], kin[:, 1]
    rsu = np.zeros(n, bool)
    coverage = np.zeros((3, len(EDGES)-1), bool)
    for ln, (lo, hi) in enumerate(ZONES):
        rsu |= present & (lane == ln) & (s >= lo) & (s < hi)
        coverage[ln] = (EDGES[:-1] < hi) & (EDGES[1:] > lo)
    detected = np.zeros(n, bool)
    for i in np.flatnonzero(cav):
        same = present & (lane == lane[i])
        for front in (True, False):
            candidates = np.flatnonzero(same & ((s > s[i]) if front else (s < s[i])))
            if len(candidates):
                gaps = (s[candidates]-s[i]-LENGTH) if front else (s[i]-s[candidates]-LENGTH)
                j = int(np.argmin(gaps))
                if gaps[j] <= SENSOR_RANGE:
                    detected[candidates[j]] = True
    sensed = present & (rsu | cav | detected)
    visible = np.flatnonzero(sensed)
    finite(kin[visible])
    # CAV detections report relative speed + gap, not neighbours' acceleration.
    clean = np.full_like(kin, np.nan)
    clean[visible] = kin[visible]
    clean[detected & ~rsu & ~cav, 3] = 0.
    for j in visible:
        ln, pos = int(lane[j]), s[j]
        if ln == -1:  # Known self in an internal junction: no valid segment coordinate.
            continue
        if ln not in (0, 1, 2):
            raise ValueError("visible invalid lane")
        b = int(np.searchsorted(EDGES, pos, side="right")-1)
        if 0 <= b < len(EDGES)-1:
            coverage[ln, b] = True
    return dict(time_s=float(time_s), kin=clean, present=present, sensor_mask=sensed,
                known_cav=cav, coverage=coverage, flow_counts=finite(flow_counts, (2, 2)))


def boundary_events(prefix):
    """Deduplicate entry detector events using only prefix IDs and entry positions.

    First appearance inside the network is an audit error, not an entry detector.
    No future registry length, ordinal, identity name, or demand is consulted.
    """
    times = finite(prefix["time_s"])
    seen, events = set(), []
    if times[0] != 0 or not np.allclose(np.diff(times), .5):
        raise ValueError("entry observations require full prefix from t0")
    for tick, time in enumerate(times):
        mask = np.asarray(prefix["present"][tick], dtype=bool)
        ids = np.asarray(prefix["vid_index"][tick])[mask]
        if len(set(ids.tolist())) != len(ids) or (ids < 0).any():
            raise ValueError("duplicate/unknown present identity")
        for j in np.flatnonzero(mask):
            identity = int(prefix["vid_index"][tick, j])
            if identity in seen:
                continue
            lane = int(prefix["lane"][tick, j])
            pos = float(prefix["s_m"][tick, j])
            if lane not in (0, 1, 2) or not BOUNDARIES[lane]-1 <= pos <= BOUNDARIES[lane]+SPEED_LIMIT*.5+1:
                raise ValueError(f"first appearance not at entry detector: t{time}, lane{lane}, s{pos}")
            seen.add(identity)
            events.append((float(time), int(lane == 2)))
    return np.asarray(events, dtype=float).reshape(-1, 2)


def flow_at(events, time_s):
    out = np.zeros((2, 2))
    for entrance in range(2):
        t = events[events[:, 1] == entrance, 0]
        out[entrance] = [np.sum(t <= time_s), np.sum((t > time_s-60) & (t <= time_s))]
    return out


def observe_prefix(tape, time_s):
    """Slice BEFORE accessing kinematics/events; poisoned futures cannot affect X."""
    tick = int(round(time_s*2))
    prefix = {k: np.asarray(tape[k])[:tick+1] for k in OBS_KEYS}
    if len(prefix["time_s"]) != tick+1 or prefix["time_s"][-1] != time_s:
        raise ValueError("missing decision boundary")
    events = boundary_events(prefix)
    kin = np.column_stack([prefix[k][-1] for k in ("s_m", "lane", "v_mps", "a_mps2")])
    return project(kin, prefix["present"][-1], prefix["is_cav"][-1], flow_at(events, time_s), time_s)


def observations(tape, decision_times):
    """One forward pass; yield each feature state before visiting its future."""
    wanted = set(decision_times)
    seen, events = set(), []
    for tick, time in enumerate(tape["time_s"]):
        if time > max(wanted):
            break
        if time != tick*.5:
            raise ValueError("nonuniform observation prefix")
        mask = np.asarray(tape["present"][tick], dtype=bool)
        ids = tape["vid_index"][tick, mask]
        if len(set(ids.tolist())) != len(ids) or (ids < 0).any():
            raise ValueError("invalid entry detector identity")
        for j in np.flatnonzero(mask):
            identity = int(tape["vid_index"][tick, j])
            if identity not in seen:
                lane, pos = int(tape["lane"][tick, j]), float(tape["s_m"][tick, j])
                if lane not in (0, 1, 2) or not BOUNDARIES[lane]-1 <= pos <= BOUNDARIES[lane]+SPEED_LIMIT*.5+1:
                    raise ValueError(f"first appearance not at entry detector: t{time}, lane{lane}, s{pos}")
                events.append((float(time), int(lane == 2)))
                seen.add(identity)
        if time in wanted:
            kin = np.column_stack([tape[k][tick] for k in ("s_m", "lane", "v_mps", "a_mps2")])
            yield project(kin, mask, tape["is_cav"][tick],
                          flow_at(np.asarray(events).reshape(-1, 2), time), time)


def departures(seed, rate, stream):
    """Label-only independent replay of request times; no assignment files."""
    rng = random.Random(f"damping_requests_v1:{seed}:{stream}")
    t, out = 0., []
    while True:
        t += rng.expovariate(rate/3600.)
        if t >= 1800:
            return np.asarray(out)
        out.append(round(t, 6))


def label_clock(tape, seed, demand, objectives):
    """Observer parity: union cohort, end-speed fleet, post-step source rectangles.

    Names/IDs are used only for real label reconciliation. No label-derived
    quantity is passed to observe_prefix/project or to the learned feature path.
    """
    times = finite(tape["time_s"])
    if not np.array_equal(times, np.arange(0., 900.5, .5)):
        raise ValueError("need full real t0..900 clock")
    counts, seen, speed_maps = [], set(), []
    names = tape["vehicle_ids"]
    for tick in range(len(times)):
        slots = np.flatnonzero(tape["present"][tick])
        row = {int(tape["vid_index"][tick, j]): float(tape["v_mps"][tick, j]) for j in slots}
        if len(row) != len(slots) or any(i < 0 for i in row):
            raise ValueError("duplicate/unknown identities in real clock")
        seen.update(row)
        counts.append(len(seen))
        speed_maps.append(row)
    # Independent recorded ordinal reconciliation: no silently dropped births.
    if seen != set(range(len(names))):
        raise ValueError("recorded identities missing from the full clock")
    if any(not (str(n).startswith("mainline.") or str(n).startswith("ramp.")) for n in names):
        raise ValueError("unknown entrance identity")
    due = sum(np.searchsorted(departures(seed, rate, stream), times, side="right")
              for rate, stream in ((demand, "mainline"), (400, "ramp")))
    queue = due-np.asarray(counts)
    if (queue < 0).any():
        raise ValueError("negative label-only source backlog")
    road, source = [], []
    for tick in range(600, 1800):
        cohort = {**speed_maps[tick], **speed_maps[tick+1]}
        road.append(.5*sum(max(0., 1.-v/SPEED_LIMIT) for v in cohort.values()))
        source.append(.5*float(queue[tick+1]))
    road, source = np.asarray(road), np.asarray(source)
    reconciled = {}
    for h in (120, 300, 600):
        actual = dict(road=float(road[:h*2].sum()), source=float(source[:h*2].sum()))
        actual["J"] = actual["road"]+actual["source"]
        delta = {k: actual[k]-float(objectives[str(h)][k]) for k in actual}
        # Float32 recorded speed vs live double speed; source must be exact.
        if abs(delta["source"]) > 1e-6 or max(abs(delta[k]) for k in ("road", "J")) > .01:
            raise ValueError(f"real objective_{h} disagreement: {delta}")
        reconciled[str(h)] = delta
    cumulative = np.r_[0., np.cumsum(road+source)]
    remaining = float(objectives["600"]["J"])-cumulative
    remaining[-1] = 0.
    if (remaining < -.01).any():
        raise ValueError("negative real remaining cost")
    return dict(time_s=times[600:], remaining=np.maximum(remaining, 0.),
                reconciled=reconciled, label_only_source=True)


def command_summaries(path, decision_times, known_cavs):
    """Read explicit issued command null masks at t0, never measured motion."""
    wanted = set(decision_times)-{900}
    commands = {t: [] for t in wanted}
    masks = {t: [] for t in wanted}
    with gzip.open(safe(path), "rt", encoding="utf-8") as stream:
        for line in stream:
            d = json.loads(line)
            t = float(d["timestamp_s"])
            if t > max(wanted):
                break
            if t in wanted and d["vehicle_id"] in known_cavs[t]:
                a, v = d["issued_command"], d["issued_target_speed_mps"]
                commands[t].append([0. if a is None else a, 0. if v is None else v])
                masks[t].append([a is not None, v is not None])
    out = {900: np.zeros(6)}
    for t in wanted:
        out[t] = action_summary(np.asarray(commands[t]).reshape(-1, 2),
                                np.asarray(masks[t]).reshape(-1, 2))
    return out
