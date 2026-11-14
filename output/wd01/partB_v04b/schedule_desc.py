"""The PLANNED-SCHEDULE action descriptor (registration s.12, VERBATIM
owner requirement).

The action input is the PLANNED SCHEDULE, not the arm name: the per-CAV
(or per-CAV-group) controller for each 30 s segment over the 570 s
horizon (19 segments) PLUS the wall hold, speed and cap (hold_cap under
wall law v2.2; the v2.1 J4 data carries no cap and says so with
has_cap=0).  "Damp 60 s then native" and "damp 570 s" MUST be
distinguishable inputs -- they are: their segment share tensors differ
(unit-tested).

Descriptor layout (one arm at one decision time t0):
  seg_shares  [19, 4]  float32 -- per segment, the roster fraction under
               native / damping / pi / wall (SCHED_ACTIONS order),
               time-weighted over the segment; computed over the CAVs
               WITH schedule evidence and renormalised (unknown time is
               excluded, mirroring actions_v04.ActionTimeline.action_shares).
  wall        [7]      float32 -- hold/600, speed_mps/40, speed_factor,
               timing_late (0/1), offset/600, has_cap, cap/600 (0 when
               has_cap=0).
  flat layout: 19*4 + 7 = 83 columns; a canonical sha256 over the
  rounded array (desc_hash) identifies equal plans.

Disclosure (s.12.1 item 2, repeated in every report): the training data
is dominated by whole-window actions; the switching arms and the J4
wall arms supply the start/stop variation, and short commitments are
barely represented until J-PULSE returns (s.12.2).

Stratified reporting (s.12.1 item 3): duration/onset of a PAIR = the
span of segments in which the ARM's planned schedule differs from its
native sibling's -- duration bins le60 (<= 60 s), mid (60-240 s),
whole (whole window / > 240 s); onset bins 0-30, 30-60, 60-240,
after-240 s.  A pair differing ONLY in the first 60 s ("60 s then
revert") is counted by first60_only() (s.12.1 item 4).

Pure functions over interval lists; zero simulation, zero file IO.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import _paths  # noqa: F401
from . import reg

N_COLS = reg.N_SEG * len(reg.SCHED_ACTIONS) + 7
WALL_IDX = reg.SCHED_ACTIONS.index("wall")

DURATION_STRATA = ("le60", "mid", "whole")
ONSET_BINS = ("on0_30", "on30_60", "on60_240", "on_after240")


# ------------------------------------------------------------- intervals --

def schedule_from_events(events, roster=None, horizon_s: float = 570.0,
                         t0: float = 0.0) -> dict[str, list[tuple]]:
    """Per-CAV planned action intervals from switch-schedule style events
    [(vid, time_s, canonical action)] restricted to (t0, t0+horizon].

    A CAV's action at t is its latest event at or before t; before its
    first event the CAV carries no evidence (excluded from the shares,
    as in actions_v04).  Events at t0 itself start at t0."""
    per: dict[str, list[tuple[float, str]]] = {}
    for vid, ts, a in events:
        if ts > t0 + horizon_s + 1e-9:
            continue
        per.setdefault(str(vid), []).append((float(ts), str(a)))
    out: dict[str, list[tuple]] = {}
    vids = ({str(v) for v in roster} & set(per)) if roster is not None \
        else set(per)
    for vid in vids:
        evs = sorted(per[vid])
        ivs = []
        for k, (ts, a) in enumerate(evs):
            lo = max(ts, t0)
            hi = evs[k + 1][0] if k + 1 < len(evs) else t0 + horizon_s
            hi = min(hi, t0 + horizon_s)
            if hi > lo + 1e-9:
                ivs.append((lo, hi, a))
        if ivs:
            out[vid] = ivs
    return out


def fleet_schedule(action: str, n_roster: int = 1,
                   horizon_s: float = 570.0, t0: float = 0.0
                   ) -> dict[str, list[tuple]]:
    """A fleet-uniform whole-window plan (native/damping/pi)."""
    return {"fleet": [(float(t0), float(t0) + horizon_s, str(action))]
            } if n_roster == 1 else {
            f"cav{i}": [(float(t0), float(t0) + horizon_s, str(action))]
            for i in range(n_roster)}


def wall_member_schedule(members, hold_s: float, t0: float = 0.0,
                         request_s: float | None = None,
                         horizon_s: float = 570.0,
                         others: str = "native") -> dict[str, list[tuple]]:
    """The PLANNED wall arm schedule: members hold the wall from the
    request tick for hold_s, then return to fleet native; every other
    roster CAV runs `others` for the whole window (planner-owned
    request-time state only -- no realised/outcome field enters)."""
    req = float(t0 if request_s is None else request_s)
    out: dict[str, list[tuple]] = {}
    for m in members:
        out[str(m)] = [(req, min(req + float(hold_s), t0 + horizon_s),
                        "wall")]
    if others != "none":
        out["__others__"] = [(float(t0), float(t0) + horizon_s,
                              str(others))]
    return out


# ------------------------------------------------------------- descriptor --

def seg_shares(schedule: dict[str, list[tuple]],
               weights: dict[str, float] | None = None,
               t0: float = 0.0) -> np.ndarray:
    """[19, 4] per-segment roster action shares (SCHED_ACTIONS order).

    weights: {vid: weight} (default: uniform over the groups).  Within
    a segment, shares are renormalised over the KNOWN time of that
    segment (the v0.4a action_shares semantics: unknown time -- a CAV
    without schedule evidence -- is excluded); a segment with no
    evidence at all stays all-zero, never silently "native"."""
    W = np.ones(len(schedule)) if weights is None else \
        np.array([float(weights.get(k, 1.0)) for k in schedule])
    out = np.zeros((reg.N_SEG, len(reg.SCHED_ACTIONS)), np.float64)
    known = np.zeros(reg.N_SEG, np.float64)
    for (vid, ivs), w in zip(schedule.items(), W):
        if w <= 0:
            continue
        for lo, hi, a in ivs:
            if a not in reg.SCHED_ACTIONS:
                raise ValueError(f"unknown planned action {a!r} "
                                 f"({reg.SCHED_ACTIONS})")
            j = reg.SCHED_ACTIONS.index(a)
            for s in range(reg.N_SEG):
                s0, s1 = t0 + s * reg.SEG_S, t0 + (s + 1) * reg.SEG_S
                ov = min(hi, s1) - max(lo, s0)
                if ov > 1e-9:
                    out[s, j] += w * ov / reg.SEG_S
                    known[s] += w * ov / reg.SEG_S
    nz = known > 1e-9
    out[nz] /= known[nz, None]
    return out.astype(np.float32)


def wall_block(hold_s: float, wall_speed_mps: float, speed_factor: float,
               timing: str, offset_s: float, hold_cap_s: float | None
               ) -> np.ndarray:
    """[7] wall hold/speed/cap block (0 where absent; has_cap flags)."""
    if timing not in reg.TIMING_OFFSET_S:
        raise ValueError(f"unknown wall timing {timing!r}")
    cap = 0.0 if hold_cap_s is None else float(hold_cap_s)
    return np.array([
        float(hold_s) / 600.0, float(wall_speed_mps) / 40.0,
        float(speed_factor), 1.0 if timing == "late" else 0.0,
        float(offset_s) / 600.0, 0.0 if hold_cap_s is None else 1.0,
        cap / 600.0,
    ], np.float32)


def describe(schedule: dict[str, list[tuple]],
             wall: np.ndarray | None = None, t0: float = 0.0,
             weights: dict[str, float] | None = None) -> np.ndarray:
    """The flat [83] float32 descriptor (seg_shares ++ wall block)."""
    seg = seg_shares(schedule, weights=weights, t0=t0)
    wb = np.zeros(7, np.float32) if wall is None else \
        np.asarray(wall, np.float32)
    if wb.shape != (7,):
        raise ValueError(f"wall block must be [7], got {wb.shape}")
    return np.concatenate([seg.ravel(), wb]).astype(np.float32)


def pi_shares(schedule: dict[str, list[tuple]]) -> dict[str, float]:
    """Planner-known schedule summary over the horizon (the v0.4a
    ActionTimeline.action_shares semantics): the time fraction of
    native / damping / pi over the schedule's KNOWN time, renormalised
    (wall time is excluded from the v0.4a share slots exactly as in
    actions_v04)."""
    tot = 0.0
    acc = {a: 0.0 for a in ("native", "damping", "pi")}
    for ivs in schedule.values():
        for lo, hi, a in ivs:
            tot += hi - lo
            if a in acc:
                acc[a] += hi - lo
    return {a: (acc[a] / tot if tot > 0 else 0.0)
            for a in ("native", "damping", "pi")}


def desc_hash(d: np.ndarray) -> str:
    """Canonical sha256 of the descriptor (rounded to 1e-6)."""
    a = np.round(np.asarray(d, np.float64), 6).astype(np.float64)
    return hashlib.sha256(a.tobytes()).hexdigest()


# ------------------------------------------------------------ pair strata --

def _diff_span(arm: np.ndarray, ref: np.ndarray
               ) -> tuple[int, int, float] | None:
    """(first differing segment, last differing segment, duration_s) of
    two descriptors, or None when the plans are equal.  A no-evidence
    (all-zero) segment on either side is NOT a difference -- unknown is
    not a committed action (the strata compare commitments)."""
    a = np.asarray(arm, np.float32).reshape(-1)[:reg.N_SEG * 4] \
        .reshape(reg.N_SEG, -1)
    r = np.asarray(ref, np.float32).reshape(-1)[:reg.N_SEG * 4] \
        .reshape(reg.N_SEG, -1)
    known = (np.abs(a).sum(axis=1) > 1e-6) & \
            (np.abs(r).sum(axis=1) > 1e-6)
    dif = known & (np.abs(a - r).max(axis=1) > 1e-6)
    if not dif.any():
        return None
    first = int(np.argmax(dif))
    last = int(reg.N_SEG - 1 - np.argmax(dif[::-1]))
    return first, last, (last + 1 - first) * reg.SEG_S


def duration_stratum(arm: np.ndarray, ref: np.ndarray) -> str:
    """'le60' | 'mid' | 'whole' from the differing span of the pair's
    plans (registration s.12.1 item 3: <= 60 s, 60-240 s, whole window).
    A whole-window difference -- or any span over 240 s -- is 'whole'."""
    sp = _diff_span(arm, ref)
    if sp is None:
        return "whole"             # identical plans: no short commitment
    _f, _l, dur = sp
    if dur <= 60.0 + 1e-9:
        return "le60"
    if dur <= 240.0 + 1e-9 and not (sp[1] == reg.N_SEG - 1
                                    and sp[0] == 0):
        return "mid"
    return "whole"


def onset_bin(arm: np.ndarray, ref: np.ndarray) -> str:
    """Onset bin of the differing span (s.12.2: 0-30, 30-60, 60-240,
    after-240 s)."""
    sp = _diff_span(arm, ref)
    if sp is None:
        return ONSET_BINS[0]
    s0 = sp[0] * reg.SEG_S
    if s0 < 30.0 - 1e-9:
        return ONSET_BINS[0]
    if s0 < 60.0 - 1e-9:
        return ONSET_BINS[1]
    if s0 < 240.0 - 1e-9:
        return ONSET_BINS[2]
    return ONSET_BINS[3]


def first60_only(arm: np.ndarray, ref: np.ndarray) -> bool:
    """True iff the plans differ ONLY inside the first 60 s ('60 s then
    revert'; the s.12.1 item 4 a-priori representation count)."""
    sp = _diff_span(arm, ref)
    return sp is not None and sp[1] <= 1


def count_by_stratum(pairs: list[tuple[np.ndarray, np.ndarray]]
                     ) -> dict[str, int]:
    """Duration-stratum counts over (arm, ref) descriptor pairs."""
    out = {k: 0 for k in DURATION_STRATA}
    for a, r in pairs:
        out[duration_stratum(a, r)] += 1
    out["n"] = len(pairs)
    out["n_first60_only"] = sum(1 for a, r in pairs
                                if first60_only(a, r))
    return out
