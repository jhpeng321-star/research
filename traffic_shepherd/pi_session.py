"""Roster-bound session for the opt-in fixed-U PI adapter."""
from dataclasses import asdict

from output.controller_verification_v7.pi_owner_adapter import PIOwnerAdapter, REGISTERED_U_MPS, VERSION as ADAPTER_VERSION


VERSION = "pi_session/1"


class PISession:
    def __init__(self, *, config, cav_ids, desired_speed_mps):
        if float(desired_speed_mps) != REGISTERED_U_MPS:
            raise ValueError("pi-external-u requires registered U=31.29 m/s")
        self.adapter = PIOwnerAdapter(config=config, cav_ids=set(cav_ids),
                                      u_mps=REGISTERED_U_MPS,
                                      u_evidence_id="cv11-registered-u31.29")
        self.cav_ids = frozenset(cav_ids)
        self.identity = {"session_version": VERSION, "adapter_version": ADAPTER_VERSION,
                         "u_mps": REGISTERED_U_MPS,
                         "u_provenance": "fixed_registered_setpoint",
                         "u_evidence_id": "cv11-registered-u31.29",
                         "cav_ids": sorted(self.cav_ids)}
        self._evidence_cursor = 0
        self._receipt_cursor = 0

    def step(self, owner, frames, *, now_s, live_ids, native_ids=(), wall_commands=None,
             departed_ids=(), id_reuse_ids=()):
        return self.adapter.step(owner, frames, now_s=now_s, live_ids=live_ids,
                                 native_ids=native_ids, wall_commands=wall_commands,
                                 departed_ids=departed_ids, id_reuse_ids=id_reuse_ids)

    def observe_lifecycle(self, *, live_ids, disappeared_ids=(), id_reuse_ids=()):
        """Invalidate state observed between control ticks before re-entry."""
        self.adapter.invalidate(live_ids=set(live_ids), departed_ids=set(disappeared_ids),
                                id_reuse_ids=set(id_reuse_ids))

    def pending_evidence(self):
        evidence = self.adapter.evidence[self._evidence_cursor:]
        receipts = self.adapter.receipts[self._receipt_cursor:]
        return ([dict(kind="pi_source_evidence", **row) for row in evidence] +
                [dict(kind="pi_owner_receipt", **asdict(row)) for row in receipts])

    def flush_evidence(self, sink, *, finalize=True):
        """Persist new rows transactionally, then release persisted prefixes."""
        rows = self.pending_evidence()
        try:
            for row in rows:
                sink(row)
        except BaseException:
            raise
        if not finalize:
            return len(rows)
        self.commit_evidence()
        return len(rows)

    def commit_evidence(self):
        self._evidence_cursor = len(self.adapter.evidence)
        self._receipt_cursor = len(self.adapter.receipts)
        del self.adapter.evidence[:self._evidence_cursor]
        del self.adapter.receipts[:self._receipt_cursor]
        self._evidence_cursor = self._receipt_cursor = 0

    def drain_evidence(self):
        """Compatibility accessor; callers needing durability should flush_evidence."""
        return self.pending_evidence()
