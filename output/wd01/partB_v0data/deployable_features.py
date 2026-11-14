"""Deployable (RSU-observable) demand/context blocks for v0.3.2.

Replaces the leaking oracle blocks (PARTB_LEAK_AUDIT.md) with inputs a
real RSU planner could know at decision time t.  TWO sensing variants
(owner decision, output/wd01/OWNER_DECISIONS_WD01.md: CAV-STRICT is the
HEADLINE deployable variant; the all-vehicle entrance-detector variant is
reported only as a labelled SENSITIVITY):

  detector demand [12]  recent OBSERVED entrance activity, as per-minute
               rates /50 (the same /50 scale as the oracle grid):
               realised insertions per entrance over (t-60, t],
               (t-180, t], (t-300, t] -- entrances mainline/ramp -- and
               the designated-CAV subset of the same counts.  ALL-vehicle
               entrance counts require entrance detectors: sensitivity
               variant only, never a deployable claim.
  strict  demand [6]   designated-CAV entrance counts only (CAV
               self-reports at the entrance; the RSU hears CAVs): the
               same windows/entrances, CAV subset only.  No all-vehicle
               entrance counts and no CAV/all-vehicle ratio, so true
               penetration is not recoverable from the block.
  context [8] (both)  [t/1000, 0, 0, loading/2700, 0, 0,
               q_main/50, q_ramp/50]
               identical to the oracle layout EXCEPT context[1] and
               context[2] are identically 0: context[1] was (t-t0)/600,
               which together with context[0] = t/1000 made the campaign
               branch time t0 exactly recoverable (v0.3.2 fix #1 -- t0 is
               a campaign design event, not a deployable input), and
               context[2] (time-to-branch, another future campaign design
               event) was always 0 in deployable blocks.  q_main/q_ramp =
               scheduled-departed but not yet inserted at t (entrance
               queues; same definition as v0lib.queue_lengths) -- carried
               by the detector variant only; the strict variant zeroes
               them (a stopped non-CAV at an entrance is invisible without
               an entrance detector).

NOTHING with a timestamp > t is read: every schedule/insertion entry
entering either block has time <= t (the searchsorted windows end at t;
the queue count is a difference of two counts at t), so perturbing all
entries after any query time leaves the output bit-identical -- asserted
in partB_v0data/tests/test_build_v031.py.

True/configured penetration is NOT an input (AGENTS.md) in either
variant: the strict CAV counts are absolute, never normalised by an
all-vehicle count.  The demand LEVEL (configured loading, an
announced/known rate) is, per the audit's "allowed" column.

Pure numpy: schedule + realised-insertion arrays in, blocks out.  The
file-side plumbing (run-log light pass, sidecar writer, MANIFEST) lives
in build_v031.py / build_v032.py.
"""
from __future__ import annotations

import numpy as np

OBS_WINDOWS_S = (60.0, 180.0, 300.0)     # (t-w, t]
DEMAND_DEP_RAW = 12                      # 2 entrances x 3 windows x (all, cav)
DEMAND_STRICT_RAW = 6                    # 2 entrances x 3 windows, CAV only
CTX_RAW = 8
DEMAND_SCALE = 50.0                      # per-minute rate /50 (oracle scale)

# sensing variants (v0.3.2 #5): "detector" = all-vehicle entrance counts
# (sensitivity), "strict" = CAV self-reports only (headline)
VARIANTS = ("detector", "strict")

# column order of the deployable demand block (mirrors
# partB_v0.gbm.DEMAND_DEP_NAMES)
DEMAND_DEP_NAMES = (
    "obs_ent_main_60", "obs_ent_main_180", "obs_ent_main_300",
    "obs_ent_ramp_60", "obs_ent_ramp_180", "obs_ent_ramp_300",
    "obs_cav_main_60", "obs_cav_main_180", "obs_cav_main_300",
    "obs_cav_ramp_60", "obs_cav_ramp_180", "obs_cav_ramp_300")

# strict variant: the designated-CAV subset of the same counts (columns
# 6..11 of DEMAND_DEP_NAMES); no all-vehicle entrance counts
STRICT_DEMAND_NAMES = DEMAND_DEP_NAMES[6:]


def _between(times: np.ndarray, t: float, w: float) -> float:
    """#{x in sorted(times) : t-w < x <= t} (left-open window)."""
    lo = np.searchsorted(times, t - w, side="right")
    hi = np.searchsorted(times, t, side="right")
    return float(hi - lo)


def _at_most(times: np.ndarray, t: float) -> float:
    """#{x in sorted(times) : x <= t}."""
    return float(np.searchsorted(times, t, side="right"))


class SourceObs:
    """Per-source observation tables built once, queried per decision time.

    sched_*: the shared per-world schedule (vid order arbitrary);
    ins_vids/ins_time: realised insertion events of the source's run
    (first insertion per vehicle wins; a native branch's caller must have
    merged the shared pre-t0 history from its base run).  Entrance and
    designated-CAV flags of an inserted vehicle are joined from the
    schedule by vid (a vid inserted but missing from the schedule cannot
    happen -- every insertion is a scheduled departure -- and counts as
    mainline/non-CAV if it ever did).
    """

    def __init__(self, sched_vids, sched_t, sched_ent, sched_is_cav,
                 ins_vids, ins_time):
        sched_vids = [str(v) for v in sched_vids]
        sched_t = np.asarray(sched_t, np.float64)
        sched_ent = np.asarray(sched_ent, np.int64)
        sched_is_cav = np.asarray(sched_is_cav, np.int64)
        ent_of = dict(zip(sched_vids, sched_ent.tolist()))
        cav_of = dict(zip(sched_vids, sched_is_cav.tolist()))

        # first insertion per vid
        ins_of: dict = {}
        for vid, ti in zip(ins_vids, np.asarray(ins_time, np.float64)):
            ti = float(ti)
            prev = ins_of.get(vid)
            if prev is None or ti < prev:
                ins_of[vid] = ti

        # insertion times grouped by (entrance, designated-CAV), sorted
        grouped: dict = {(e, c): [] for e in (0, 1) for c in (0, 1)}
        for vid, ti in ins_of.items():
            grouped[(ent_of.get(vid, 0), cav_of.get(vid, 0))].append(ti)
        self._grouped = {k: np.sort(np.asarray(v, np.float64))
                         for k, v in grouped.items()}

        # per entrance: sorted schedule departures / sorted insertions
        # (queue = sched-departed <= t minus inserted <= t, per entrance)
        self._sched_sorted = {}
        self._ins_sorted = {}
        for ent in (0, 1):
            m = sched_ent == ent
            self._sched_sorted[ent] = np.sort(sched_t[m])
            self._ins_sorted[ent] = np.sort(
                np.asarray([ti for vid, ti in ins_of.items()
                            if ent_of.get(vid, 0) == ent], np.float64))

    def queue_lengths(self, t: float) -> tuple[float, float]:
        """(q_main, q_ramp): scheduled-departed but not inserted at t."""
        q = []
        for ent in (0, 1):
            n_sched = _at_most(self._sched_sorted[ent], t)
            n_ins = _at_most(self._ins_sorted[ent], t)
            q.append(max(0.0, n_sched - n_ins))
        return q[0], q[1]

    def row(self, t: float, t0: float, loading: float,
            variant: str = "detector") -> tuple[np.ndarray, np.ndarray]:
        """Deployable (demand, context) at decision time t.

        variant="detector": demand[12] (all-vehicle + CAV entrance
        counts) and the entrance-queue context cols 6/7 -- the labelled
        SENSITIVITY variant.  variant="strict": demand[6] (CAV
        self-reports only) and q identically 0 -- the HEADLINE variant.
        Reads only entries with timestamp <= t: the observed-count
        windows end at t and the queue count is evaluated at t."""
        if variant not in VARIANTS:
            raise ValueError(f"unknown sensing variant {variant!r} "
                             f"({' | '.join(VARIANTS)})")
        demand = np.zeros(DEMAND_DEP_RAW, np.float32)
        col = 0
        for ent in (0, 1):                 # entrance-major, like the names
            for w in OBS_WINDOWS_S:
                both = _between(self._grouped[(ent, 0)], t, w) \
                    + _between(self._grouped[(ent, 1)], t, w)
                demand[col] = both / (w / 60.0) / DEMAND_SCALE
                col += 1
        for ent in (0, 1):                 # designated-CAV subset
            for w in OBS_WINDOWS_S:
                demand[col] = _between(self._grouped[(ent, 1)], t, w) \
                    / (w / 60.0) / DEMAND_SCALE
                col += 1
        q_main, q_ramp = self.queue_lengths(t)
        ctx = np.array([
            t / 1000.0,
            0.0,        # (t-t0)/600 removed (v0.3.2 #1): with ctx[0]=t/1000
                        # it made the campaign branch time t0 recoverable
            0.0,                             # time-to-branch: NOT deployable
            float(loading) / 2700.0, 0.0, 0.0,
            q_main / DEMAND_SCALE, q_ramp / DEMAND_SCALE,
        ], np.float32)
        if variant == "strict":
            demand = demand[6:].copy()       # CAV self-reports only
            ctx[6] = 0.0                     # all-vehicle entrance queues
            ctx[7] = 0.0                     # are invisible without detectors
        return demand, ctx

    def blocks(self, ts, t0: float, loading: float,
               variant: str = "detector") -> tuple[np.ndarray, np.ndarray]:
        """(demand [n,D], context [n,8]) over the decision-time grid;
        D = 12 (detector) or 6 (strict).  t0 is accepted for interface
        compatibility only: no deployable column may depend on it
        (v0.3.2 #1 -- asserted in tests)."""
        ts = np.asarray(ts, np.float64)
        n_dem = DEMAND_STRICT_RAW if variant == "strict" else DEMAND_DEP_RAW
        demand = np.zeros((len(ts), n_dem), np.float32)
        ctx = np.zeros((len(ts), CTX_RAW), np.float32)
        for i, t in enumerate(ts):
            demand[i], ctx[i] = self.row(float(t), t0, loading,
                                         variant=variant)
        return demand, ctx
