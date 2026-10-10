"""Pre-allocation budget guard with persistent, crash-safe accounting.

Enforces campaign ceilings across versioned directories: a new registration
or output version never resets consumed budget. Reservations are written
atomically BEFORE launch and survive crashes (a crashed launch leaves the
reservation charged). A STOP receipt blocks further reservation for that
campaign path unless an explicitly authorized resolution record exists.
Concurrent launch attempts serialize on an exclusive lock file, so parallel
starts cannot overspend.

Pure accounting + file machinery: no SUMO, no launcher invocation. Callers
pass a ``launcher`` callable; the guard invokes it only after a successful
reservation and records the outcome afterwards.
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

LEDGER_SCHEMA = "campaign_budget_ledger/1"
LOCK_SUFFIX = ".reserve.lock"
_REPLACE_ATTEMPTS = 5
_REPLACE_RETRY_DELAY_S = 0.05
_REPLACE_WINERRORS = frozenset((5, 32, 33))


class BudgetExceeded(Exception):
    pass


class BlockedByStop(Exception):
    pass


class PrerequisiteMissing(Exception):
    pass


class _FileLock:
    """Exclusive lock via O_CREAT|O_EXCL spin; stale locks expire."""

    def __init__(self, path, timeout_s=10.0, stale_s=30.0):
        self.path = Path(str(path) + LOCK_SUFFIX)
        self.timeout_s = timeout_s
        self.stale_s = stale_s

    def __enter__(self):
        deadline = time.time() + self.timeout_s
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except PermissionError as exc:
                # Access/share denial on the create itself (antivirus/indexer
                # class; observed with and without a winerror attribute):
                # bounded retry inside the normal deadline.  The lock file is
                # created and deleted by our own processes constantly, so any
                # denial on it is contention by construction; a persistent
                # denial surfaces as the deadline TimeoutError.
                if time.time() >= deadline:
                    raise TimeoutError("budget lock busy: " + str(self.path)) from exc
                time.sleep(0.05)
            except FileExistsError:
                # The holder may release between exists() and stat(): a
                # vanished lock is not stale, just retry the atomic acquire.
                try:
                    stale = (self.path.exists()
                             and time.time() - self.path.stat().st_mtime > self.stale_s)
                except FileNotFoundError:
                    stale = False
                if stale:
                    self.path.unlink(missing_ok=True)  # stale lock from a crash
                    continue
                if time.time() >= deadline:
                    raise TimeoutError("budget lock busy: " + str(self.path))
                time.sleep(0.05)

    def __exit__(self, *exc):
        os.close(self.fd)
        self.path.unlink(missing_ok=True)
        return False


class BudgetGuard:
    def __init__(self, ledger_path):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if not self.ledger_path.exists():
            return dict(schema=LEDGER_SCHEMA,
                        campaigns={}, reservations=[])
        return json.loads(self.ledger_path.read_text(encoding="utf-8"))

    def _save(self, data):
        tmp = self.ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        # Keep the prepared payload and lock unchanged while accommodating a
        # transient Windows access/share denial.  This is strictly
        # an atomic replace retry: it never retries a launcher or regenerates
        # the ledger between attempts.
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, self.ledger_path)  # atomic: no half-write
                return
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in _REPLACE_WINERRORS:
                    raise
                if attempt + 1 == _REPLACE_ATTEMPTS:
                    raise
                time.sleep(_REPLACE_RETRY_DELAY_S)

    @staticmethod
    def _used(data, campaign, category):
        """Charged seconds: live reservations PLUS reclassified/excluded
        charges (a moved or reclassified record never restores capacity)."""
        live = sum(r["amount_s"] for r in data["reservations"]
                   if r["campaign"] == campaign and r["category"] == category)
        excluded = sum(x["amount_s"] for x in data.get("excluded_charges", [])
                       if x["campaign"] == campaign and x["category"] == category)
        return live + excluded

    @staticmethod
    def _used_total(data, campaign):
        live = sum(r["amount_s"] for r in data["reservations"]
                   if r["campaign"] == campaign)
        excluded = sum(x["amount_s"] for x in data.get("excluded_charges", [])
                       if x["campaign"] == campaign)
        return live + excluded

    def stop_campaign(self, campaign, reason):
        """Sticky campaign-level stop (e.g. first failed mechanics
        allocation). Never cleared by this API: a new authorization must use
        a new campaign identity, and moved/reclassified records cannot
        restore capacity."""
        with _FileLock(self.ledger_path):
            data = self._load()
            campaigns = data.setdefault("campaigns", {})
            if campaigns.get(campaign, {}).get("stopped"):
                return
            campaigns[campaign] = dict(stopped=True, stop_reason=reason,
                                       stopped_at=datetime.now(timezone.utc).isoformat())
            self._save(data)

    def mark_progress(self, reservation_id, completed_simulated_s):
        """Record ACTUAL completed simulated time for one reservation,
        separately from the charged amount."""
        with _FileLock(self.ledger_path):
            data = self._load()
            for r in data["reservations"]:
                if r["id"] == reservation_id:
                    r["completed_simulated_s"] = float(completed_simulated_s)
                    break
            self._save(data)

    def mark_latest_progress(self, campaign, category, completed_simulated_s):
        """Convenience wrapper: mark actual progress on the most recent
        reservation of a campaign/category.  Advisory telemetry only:
        terminal receipts are authoritative for actuals, so a transient
        lock/IO failure here is counted and swallowed, never propagated
        into the simulation loop."""
        try:
            with _FileLock(self.ledger_path):
                data = self._load()
                matches = [r for r in data["reservations"]
                           if r["campaign"] == campaign and r["category"] == category]
                if matches:
                    matches[-1]["completed_simulated_s"] = float(completed_simulated_s)
                self._save(data)
        except OSError as exc:
            self.progress_mark_failures = getattr(self, "progress_mark_failures", 0) + 1
            self.last_progress_mark_error = repr(exc)

    def exclude_charge(self, campaign, category, amount_s, reservation_id, reason):
        """Reclassify a charge out of the live reservation list while KEEPING
        it counted against the ceilings (append-only tombstone)."""
        with _FileLock(self.ledger_path):
            data = self._load()
            before = len(data["reservations"])
            data["reservations"] = [r for r in data["reservations"]
                                    if r["id"] != reservation_id]
            if len(data["reservations"]) == before:
                raise ValueError("reservation not found: " + reservation_id)
            data.setdefault("excluded_charges", []).append(dict(
                reservation_id=reservation_id, campaign=campaign,
                category=category, amount_s=float(amount_s), reason=reason,
                excluded_at=datetime.now(timezone.utc).isoformat()))
            self._save(data)

    def campaign_state(self, campaign):
        with _FileLock(self.ledger_path):
            data = self._load()
        return data.get("campaigns", {}).get(campaign, {})

    def stop_block_reason(self, campaign, target_dir):
        """Public variant acquiring the lock (external inspection)."""
        with _FileLock(self.ledger_path):
            data = self._load()
        return self._stop_block_reason(data, campaign, target_dir)

    def _stop_block_reason(self, data, campaign, target_dir):
        """Lock-free variant used inside reserve() (lock already held)."""
        state = data.get("campaigns", {}).get(campaign, {})
        if state.get("stopped"):
            return "campaign " + campaign + " is stopped: " + \
                str(state.get("stop_reason"))
        stop = Path(target_dir) / "STOP.json"
        if not stop.exists():
            return None
        resolution = Path(target_dir) / "verification_resolution.json"
        if resolution.exists():
            try:
                record = json.loads(resolution.read_text(encoding="utf-8"))
            except Exception:
                record = {}
            if record.get("author") == "owner":
                return None
            return ("STOP receipt present at " + str(stop) +
                    "; the resolution record beside it is agent-generated "
                    "(author != owner) and cannot clear the block")
        return "STOP receipt present at " + str(stop) + \
            " without an authorized resolution record"

    def reserve(self, *, campaign, category, amount_s, target_dir,
                ceilings, launcher, launch_args=(), prerequisite=None,
                campaign_stop_on_failure=False):
        """Atomically reserve amount_s under category, then run launcher.

        ceilings: dict(categories={name: {"max_allocations": int,
                   "max_seconds": float}}, total_seconds=float)
        prerequisite: (kind, path) of a required completed evidence record,
                   e.g. ("mechanics_registration", path) - must exist and be
                   marked passed in the ledger's prerequisite section.
        Raises BudgetExceeded / BlockedByStop / PrerequisiteMissing without
        touching the ledger when the request is not admissible.
        """
        with _FileLock(self.ledger_path):
            data = self._load()
            stop_reason = self._stop_block_reason(data, campaign, target_dir)
            if stop_reason:
                raise BlockedByStop(stop_reason)
            # Ceilings are cumulative across versioned directories: keyed by
            # campaign name only, never by output version.
            cat = ceilings["categories"][category]
            used_cat = self._used(data, campaign, category)
            used_tot = self._used_total(data, campaign)
            if used_cat + amount_s > cat["max_seconds"]:
                raise BudgetExceeded(
                    f"{campaign}/{category}: {used_cat}+{amount_s} exceeds "
                    f"{cat['max_seconds']} s ceiling")
            n_cat = len([r for r in data["reservations"]
                         if r["campaign"] == campaign and r["category"] == category])
            if cat.get("max_allocations") is not None \
                    and n_cat + 1 > cat["max_allocations"]:
                raise BudgetExceeded(
                    f"{campaign}/{category}: allocation count {n_cat + 1} exceeds "
                    f"{cat['max_allocations']}")
            if used_tot + amount_s > ceilings["total_seconds"]:
                raise BudgetExceeded(
                    f"{campaign}: total {used_tot}+{amount_s} exceeds "
                    f"{ceilings['total_seconds']} s ceiling")
            stop_reason = self._stop_block_reason(data, campaign, target_dir)
            if stop_reason:
                raise BlockedByStop(stop_reason)
            if prerequisite is not None:
                kind, path = prerequisite
                record = (data.setdefault("prerequisites", {})
                          .setdefault(campaign, {}).get(kind))
                if record is None or record.get("path") != str(path) \
                        or record.get("passed") is not True:
                    raise PrerequisiteMissing(
                        f"{campaign}: {kind} prerequisite not recorded as passed: {path}")
            reservation = dict(
                id=f"{campaign}:{category}:{len(data['reservations']) + 1}",
                campaign=campaign, category=category, amount_s=amount_s,
                target_dir=str(target_dir),
                reserved_at=datetime.now(timezone.utc).isoformat(),
                status="reserved")  # crash here leaves it charged
            data["reservations"].append(reservation)
            self._save(data)
        # Lock released; launcher runs after the reservation is durable.
        try:
            outcome = launcher(*launch_args)
        except Exception as exc:
            self._mark(reservation["id"], "failed", detail=repr(exc))
            if campaign_stop_on_failure:
                self.stop_campaign(campaign, "launcher failure on "
                                   + reservation["id"] + ": " + repr(exc))
            raise
        self._mark(reservation["id"], "complete")
        return reservation, outcome

    def _mark(self, reservation_id, status, detail=None):
        with _FileLock(self.ledger_path):
            data = self._load()
            for r in data["reservations"]:
                if r["id"] == reservation_id:
                    r["status"] = status
                    if detail:
                        r["detail"] = detail
            self._save(data)

    def charged_seconds(self, campaign, category=None):
        """Total charged seconds for a campaign (one category or all)."""
        with _FileLock(self.ledger_path):
            data = self._load()
        return (self._used(data, campaign, category) if category is not None
                else self._used_total(data, campaign))

    def record_prerequisite(self, campaign, kind, path, *, passed=True, note=None):
        """Record a completed prerequisite (e.g. a passing mechanics fixture).

        Bookkeeping only: it never creates budget or authorization. A
        reservation that declares this prerequisite checks exactly this
        record (same path, passed is True).
        """
        with _FileLock(self.ledger_path):
            data = self._load()
            data.setdefault("prerequisites", {}).setdefault(campaign, {})[kind] = dict(
                path=str(path), passed=bool(passed),
                recorded_at=datetime.now(timezone.utc).isoformat(), note=note)
            self._save(data)


def reserve_and_launch(budget, *, target_dir, launcher, campaign_stop_on_failure=False):
    """The registered runners' launch boundary: reserve budget, then launch.

    ``budget`` is the registration-declared enforcement block (campaign,
    ledger_path, category, amount_s, ceilings, optional prerequisite
    [kind, path]). The launcher is invoked only after the reservation is
    durably recorded; any rejection (BudgetExceeded, BlockedByStop,
    PrerequisiteMissing) raises before the launcher runs, so a rejected
    runner never starts the simulator and never writes the ledger. A
    launcher that fails or crashes leaves its reservation charged, which
    blocks a silent retry.
    """
    prerequisite = budget.get("prerequisite")
    guard = BudgetGuard(budget["ledger_path"])
    return guard.reserve(
        campaign=budget["campaign"], category=budget["category"],
        amount_s=budget["amount_s"], target_dir=target_dir,
        ceilings=budget["ceilings"],
        prerequisite=tuple(prerequisite) if prerequisite else None,
        launcher=launcher, campaign_stop_on_failure=campaign_stop_on_failure)
