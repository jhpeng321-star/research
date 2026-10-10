"""The ONE-CAL-read guard (registration s.7).

Two independent locks, both bound to the registration file:

1. OWNER_ACK gate -- the CAL read refuses to start unless
   OWNER_ACK_PB_V04B.json exists next to this package (or an explicit
   path) AND names the registration by its FULL LF-sha256
   (e00cd98e...).  A stale ack for a different registration is loud.

2. Read-once marker -- CAL_READ_ONCE_V04B.json is written in the
   results dir BEFORE the first CAL row (start timestamp, registration
   sha) and stamped with the finish timestamp afterwards.  A second
   begin() on the same or a different marker refuses: "no refit after
   the CAL read"; a crash BEFORE the read may be continued (no marker),
   a crash AFTER it is reported as is (marker present, unfinished).

The guard also provides assert_train_only(): a TRAIN build hands it the
split column of every row/dir it touched and it refuses on anything
that is not TRAIN -- the CAL/HOLDOUT/chunk-5 never-open rule enforced
at the pipeline level, not by discipline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import _paths  # noqa: F401
from . import reg


class CalReadNotAllowed(RuntimeError):
    """The CAL read is not authorized (no/stale OWNER_ACK) or already
    spent (read-once marker present)."""


class MarkerMismatch(RuntimeError):
    """The read-once marker names a different registration."""


def owner_ack_path(explicit: str | Path | None = None) -> Path:
    return Path(explicit) if explicit else \
        reg.RESULTS_DIR / reg.OWNER_ACK_NAME


def require_owner_ack(ack_path: str | Path | None = None,
                      reg_sha: str | None = None) -> dict:
    """Refuse unless the OWNER_ACK file binds THIS registration's full
    LF-sha256.  Returns the parsed ack (for the report)."""
    sha = reg_sha or reg.reg_sha()
    p = owner_ack_path(ack_path)
    if not p.is_file():
        raise CalReadNotAllowed(
            f"CAL read refused: no OWNER_ACK file at {p} -- the fit may "
            "run on TRAIN, but the ONE CAL read needs the owner's ack "
            "naming this registration")
    try:
        ack = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                                 # noqa: BLE001
        raise CalReadNotAllowed(f"CAL read refused: OWNER_ACK at {p} is "
                                f"not valid JSON: {e}") from e
    got = str(ack.get("registration_lf_sha256", ""))
    if got != sha:
        raise CalReadNotAllowed(
            f"CAL read refused: OWNER_ACK at {p} binds registration "
            f"LF-sha {got[:8] or '<missing>'}... != this registration "
            f"{sha[:8]}... (stale or foreign ack)")
    if not str(ack.get("ack", "")).strip():
        raise CalReadNotAllowed(
            f"CAL read refused: OWNER_ACK at {p} carries no ack statement")
    return ack


class CalReadOnce:
    """Read-once marker context manager (CAL_READ_ONCE_V04B.json)."""

    def __init__(self, results_dir: str | Path | None = None,
                 reg_sha: str | None = None):
        self.results_dir = Path(results_dir) if results_dir \
            else reg.RESULTS_DIR
        self.reg_sha = reg_sha or reg.reg_sha()
        self.path = self.results_dir / reg.CAL_MARKER_NAME

    def _existing(self) -> dict | None:
        if not self.path.is_file():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:                             # noqa: BLE001
            raise MarkerMismatch(
                f"CAL marker at {self.path} is unreadable ({e}) -- do "
                "not delete it; report as is") from e

    def begin(self) -> dict:
        """Write the marker BEFORE the first CAL row; refuse a second
        read (same registration: spent; different: loud mismatch)."""
        old = self._existing()
        if old is not None:
            if old.get("registration_lf_sha256") != self.reg_sha:
                raise MarkerMismatch(
                    f"CAL marker at {self.path} binds a DIFFERENT "
                    f"registration ({str(old.get('registration_lf_sha256'))[:8]}..."
                    f" != {self.reg_sha[:8]}...) -- refusing")
            raise CalReadNotAllowed(
                "CAL read refused: the ONE CAL read of this registration "
                f"was already started at {old.get('started_utc')} "
                f"(marker {self.path}) -- no second read, no refit after "
                "the CAL read (registration s.7)")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        marker = dict(
            schema="wd01_partB_v04b_cal_read_once/1",
            registration_lf_sha256=self.reg_sha,
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            finished_utc=None)
        tmp = Path(str(self.path) + ".tmp")
        tmp.write_text(json.dumps(marker, indent=1), encoding="utf-8")
        tmp.replace(self.path)
        return marker

    def finish(self) -> dict:
        m = self._existing()
        if m is None:
            raise CalReadNotAllowed(
                "finish() before begin(): no CAL marker present")
        if m.get("finished_utc"):
            raise CalReadNotAllowed(
                "the CAL read is already finished (marker stamped "
                f"{m['finished_utc']})")
        m["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime())
        tmp = Path(str(self.path) + ".tmp")
        tmp.write_text(json.dumps(m, indent=1), encoding="utf-8")
        tmp.replace(self.path)
        return m

    def spent(self) -> bool:
        m = self._existing()
        return bool(m and m.get("finished_utc"))

    # used as: with cal_read: <read CAL rows>
    def __enter__(self) -> "CalReadOnce":
        self.begin()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.finish()
        return False                    # propagate: a crash after the
        # CAL read is reported as is (registration s.7)


def assert_train_only(splits, context: str = "") -> None:
    """Refuse unless every split value is TRAIN (the TRAIN build's
    hard guard; CAL/HOLDOUT rows can never pass by accident)."""
    vals = np.asarray(splits, dtype=object)
    bad = sorted({str(s) for s in vals.tolist() if s != "TRAIN"})
    if bad:
        raise CalReadNotAllowed(
            f"{context}: non-TRAIN split value(s) {bad} reached a TRAIN "
            "build -- splits are taken by name from the J4 plan first; "
            "CAL is read once, inside the guarded cal-read only")
