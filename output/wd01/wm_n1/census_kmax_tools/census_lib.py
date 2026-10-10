"""Read-only K_max drop-location census (REGISTRATION_WM_N1 AMENDMENT 5, §1-§3).

Replays the FROZEN ``features.featurizer.parse_run`` slot-assignment loop by
importing it unmodified (no copy of the parser).  A scoped, read-only
observation hook rebinds the featurizer module's ``json`` name to a recording
shim for the duration of the ``parse_run`` call: every physics row the frozen
parser consumes is handed to the census recorder first, then returned
unchanged.  The recorder mirrors the parser's row filter (rows at or before
``t_origin`` are skipped) and, for every rostered-or-dropped vehicle-tick,
records ``(lane_phys, s_down)`` via ``features.geometry.s_down_of``.

A vehicle-tick is *dropped* exactly when it is present in a row the parser
processed but never reached the parser's slot records (pool full); the census
drop count therefore must equal the parser's ``n_dropped_concurrent`` and the
shard meta ``n_dropped_slot_steps`` -- this equality is the per-run integrity
check.  The K = 192 residual replay calls the same frozen ``parse_run`` with
``features.constants.K_MAX`` temporarily set to 192 (restored in ``finally``).

Counts and positions only.  No outcome, label or summary file is read.
"""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WM_N1 = _HERE.parent                       # output/wd01/wm_n1
for _p in (str(_WM_N1), str(_WM_N1.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from features import constants as C         # noqa: E402
from features import geometry as G          # noqa: E402
from features import featurizer as F        # noqa: E402

ZONE_S_LO = -800.0                          # A5 §2 upstream mainline bound
ZONE_S_HI = 225.44                          # A5 §2 downstream mainline bound
K_CENSUS = 128                              # A5 §1 census K
K_RESIDUAL = 192                            # A5 §1 descriptive residual K


def in_zone(lane_phys: int, s_down: float, *, junction: bool = False) -> bool:
    """A5 §2 near-merge zone.

    ``lane_phys == 2`` anywhere; lanes 0-1 with s_down in [-800, 225.44);
    junction-internal vehicle-ticks (no s_down) count as IN-zone.
    """
    if junction or int(lane_phys) < 0:
        return True
    lp = int(lane_phys)
    if lp == 2:
        return True
    if lp in (0, 1):
        return ZONE_S_LO <= float(s_down) < ZONE_S_HI
    raise G.GeometryError(f"unexpected lane_phys {lane_phys!r}")


class DropRecorder:
    """Read-only per-vehicle-tick observer fed by the recording json shim.

    Mirrors the frozen parser's row filter: rows with ``t <= t_origin`` are
    ignored exactly like the parser's ``continue``.  For every vehicle entry
    of every processed row the zone class is computed once, via
    ``features.geometry.s_down_of`` for physical lanes (junction-internal and
    missing-lane entries are IN-zone).
    """

    def __init__(self, t_origin: float):
        self.t_origin = float(t_origin)
        self.keys = []                      # packed tick*2**40 + vid_code
        self.zone_in = []                   # 1 IN-zone, 0 OUT-of-zone
        self._vid_code = {}
        self._zone_cache = {}
        self.n_rows = 0

    def _zone_of(self, vv) -> bool:
        lane_id = vv.get("lane_id")
        hit = self._zone_cache.get(lane_id)
        if hit is None:
            if lane_id is None or G.is_junction_lane(lane_id):
                hit = (True, -1)
            else:
                hit = (False, G.lane_phys_of(lane_id))
            self._zone_cache[lane_id] = hit
        junction, lane_phys = hit
        if junction:
            return True
        _, s_down = G.s_down_of(lane_id, float(vv["lane_position_m"]))
        return in_zone(lane_phys, s_down)

    def __call__(self, row) -> None:
        t = row.get("time_s")
        if t is None or float(t) <= self.t_origin + 1e-9:
            return
        self.n_rows += 1
        k = int(round((float(t) - self.t_origin) / C.DT_S))
        k40 = k << 40
        for vid, vv in (row.get("vehicles") or {}).items():
            vid = str(vid)
            code = self._vid_code.get(vid)
            if code is None:
                code = self._vid_code[vid] = len(self._vid_code)
            self.keys.append(k40 + code)
            self.zone_in.append(1 if self._zone_of(vv) else 0)


class _RecordingJson:
    """``json.loads`` shim: real parse, then feed the recorder, return row."""

    def __init__(self, recorder: DropRecorder):
        self._recorder = recorder
        self._loads = json.loads

    def loads(self, s):
        row = self._loads(s)
        self._recorder(row)
        return row


def census_run(run_dir, t_origin: float) -> dict:
    """One-run census: K=128 drop locations + roster, plus K=192 residual.

    Returns counts only.  Raises :class:`features.featurizer.FeaturizeError`
    for contract-violating runs and :class:`AssertionError` when the
    recorder/parser drop counts disagree (integrity failure).
    """
    rec = DropRecorder(t_origin)
    saved_json = F.json
    F.json = _RecordingJson(rec)
    try:
        parsed = F.parse_run(run_dir, t_origin)
    finally:
        F.json = saved_json

    # rostered (tick, vehicle) pairs from the parser's own records
    codes = [rec._vid_code.get(v, -1) for v in parsed["slot_vid"]]
    rostered = set()
    add = rostered.add
    for k, vi in zip(parsed["rec_tick"].tolist(),
                     parsed["rec_vididx"].tolist()):
        c = codes[vi]
        if c >= 0:
            add((int(k) << 40) + c)

    # dropped = recorded presence minus rostered
    dropped_in = dropped_out = 0
    for key, zin in zip(rec.keys, rec.zone_in):
        if key in rostered:
            continue
        if zin:
            dropped_in += 1
        else:
            dropped_out += 1
    dropped_total = dropped_in + dropped_out
    if dropped_total != int(parsed["n_dropped_concurrent"]):
        raise AssertionError(
            f"census integrity: recorder dropped {dropped_total} != parser "
            f"n_dropped_concurrent={parsed['n_dropped_concurrent']}")

    # rostered zone split from the parser's own records (vectorised)
    import numpy as np

    lane = parsed["rec_lane"]
    sdown = parsed["rec_sdown"]
    junc = parsed["rec_junc"] > 0
    rz = junc | (lane == 2) | ((lane >= 0) & (lane <= 1)
                               & (sdown >= ZONE_S_LO) & (sdown < ZONE_S_HI))
    rostered_total = int(lane.shape[0])
    rostered_in = int(rz.sum())
    rostered_out = rostered_total - rostered_in

    # per-tick rostered (concurrent) counts for the descriptive distribution
    tick_hist = np.bincount(parsed["rec_tick"],
                            minlength=int(parsed["n_ticks"]))

    # descriptive: what would still be dropped at K = 192 (same frozen loop)
    saved_k = C.K_MAX
    C.K_MAX = K_RESIDUAL
    try:
        p192 = F.parse_run(run_dir, t_origin)
    finally:
        C.K_MAX = saved_k

    return dict(
        n_ticks=int(parsed["n_ticks"]),
        rostered=rostered_total,
        rostered_in=rostered_in,
        rostered_out=rostered_out,
        dropped_total=dropped_total,
        dropped_in=dropped_in,
        dropped_out=dropped_out,
        n_dropped_192=int(p192["n_dropped_concurrent"]),
        tick_hist=[int(x) for x in np.bincount(tick_hist)],
    )


def _sha256_file(path, block: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def census_job(job: dict) -> dict:
    """Worker entry: census one sampled run under the open-path audit.

    ``job`` carries ``key``, ``run_dir``, ``t_origin``, ``expected_dropped``
    (the pack_v2c meta ``n_dropped_slot_steps`` for the run, from
    SLOT_POOL_CENSUS.json) and optionally ``input_sha256`` (the pack INDEX
    hash of the raw physics file the featurizer read; verified before
    parsing so the census replays the identical bytes).
    """
    from corpus import _util as U

    U.lower_priority_below_normal()
    key = job["key"]
    row = dict(key=key, ok=False, error=None, duration_s=None,
               loading=job.get("loading"), split=job.get("split"))
    audit = U.OpenedAudit
    audit.reset()
    audit.start()
    t0 = _time.monotonic()
    try:
        want = job.get("input_sha256")
        if want:
            got = _sha256_file(job["run_dir"] + "/run/physics_steps.jsonl")
            if got != want:
                raise AssertionError(
                    f"raw physics sha mismatch: {got[:16]}… != pack INDEX "
                    f"{want[:16]}… (not the featurizer's source bytes)")
        out = census_run(job["run_dir"], float(job["t_origin"]))
        audit.stop()
        audit.assert_clean(
            forbidden=("dispatcher_rows", "insertions", "evaluate_summary",
                       "suffix_summary", "outcomes"))
        row.update(ok=True, audit_clean=True, **out)
        row["reconcile_ok"] = (out["dropped_total"]
                               == int(job["expected_dropped"]))
        row["expected_dropped"] = int(job["expected_dropped"])
    except BaseException as exc:               # noqa: BLE001 (recorded, raised)
        audit.stop()
        row["ok"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        row["duration_s"] = round(_time.monotonic() - t0, 2)
    return row
