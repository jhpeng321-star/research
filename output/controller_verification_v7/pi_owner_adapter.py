"""Isolated PI controller for the existing CommandOwnership callback seam."""
from dataclasses import asdict, dataclass
from math import isfinite

from output.controller_verification_v2.pi_saturation_v1 import PISaturationParameters
from output.controller_verification_v4.pi_external_u import PIExternalU, UReport
from traffic_shepherd.config import load_config
from traffic_shepherd.damping import DampingDecision, direct_target, frame_problem

VERSION = "pi_owner_adapter/2"
REGISTERED_U_MPS = 31.29


class PIOwnerSessionError(RuntimeError):
    """A callback, lifecycle, or transaction failure stops this instance."""


@dataclass(frozen=True)
class ProposalReceipt:
    vehicle_id: str
    timestamp_s: float
    status: str
    raw_command_mps: float | None = None
    issued_target_speed_mps: float | None = None
    reason: str | None = None


class PIOwnerAdapter:
    """Stateful PI controller consumed by real CommandOwnership.step.

    ``decide`` is called by the owner for analytical vehicles. The owner writes
    the returned DampingDecision, then calls ``record_callback`` with its real
    immutable CommandRecord. Native and wall rows are valid receipts without a
    PI proposal; they do not mutate PI recurrence state.
    """

    def __init__(self, *, config=None, cav_ids, sample_period_s=1.0,
                 u_mps=REGISTERED_U_MPS, u_evidence_id="cv07-registered-u31.29",
                 parameters=None):
        if float(u_mps) != REGISTERED_U_MPS or not isfinite(float(u_mps)):
            raise ValueError("CV07 requires registered U=31.29 m/s")
        if not cav_ids:
            raise ValueError("cav_ids must be nonempty")
        self.config = config or load_config()
        if (self.config.control_dt_s, self.config.physics_dt_s, self.config.sensor_range_m,
            self.config.speed_mode, self.config.acceleration_min_mps2,
            self.config.acceleration_max_mps2) != (1.0, .5, 100., 31, -3., 1.5):
            raise ValueError("PI owner requires the adopted interface contract")
        if float(sample_period_s) != self.config.control_dt_s:
            raise ValueError("PI sample period must equal the control period")
        if parameters is not None and parameters != PISaturationParameters():
            raise ValueError("CV07 uses the registered published PI parameters")
        self.cav_ids = frozenset(cav_ids)
        if not self.cav_ids or any(not isinstance(v, str) or not v for v in self.cav_ids):
            raise ValueError("cav_ids must contain nonempty vehicle-id strings")
        self.pi = PIExternalU(sample_period_s=sample_period_s, parameters=parameters)
        self.u_report = UReport(float(u_mps), "fixed_registered_setpoint", u_evidence_id)
        self._initialized = set()
        self._pending = {}
        self._prepared = {}
        self._prepared_fallback = {}
        self._expected_active = set()
        self._tick = None
        self._in_owner_call = False
        self._batch_expected = set()
        self._callback_seen = set()
        self._bound_owner = None
        self.receipts = []
        self.evidence = []
        self.stopped = False

    def invalidate(self, *, live_ids, native_ids=(), wall_ids=(), missing_ids=(),
                   stale_ids=(), departed_ids=(), id_reuse_ids=()):
        if self.stopped:
            raise PIOwnerSessionError("PI owner session is stopped")
        live = set(live_ids) & self.cav_ids
        removed = (set(native_ids) | set(wall_ids) | set(missing_ids) |
                   set(stale_ids) | set(departed_ids) | set(id_reuse_ids))
        self.pi.invalidate(vehicle_ids=live, native_ids=set(native_ids), wall_ids=set(wall_ids),
                           missing_ids=set(missing_ids), stale_ids=set(stale_ids),
                           departed_ids=set(departed_ids), id_reuse_ids=set(id_reuse_ids))
        self._initialized.intersection_update(live - removed)

    def _compute_decision(self, frame, *, now_s):
        """Prepare raw PI proposal and return the existing direct bridge decision."""
        if self.stopped:
            raise PIOwnerSessionError("PI owner session is stopped")
        reason = frame_problem(frame, now_s, self.config)
        if reason or frame.front is None or frame.front_coverage_m <= 0:
            return DampingDecision(fallback_reason=reason or "missing_or_uncovered_leader")
        vehicle = frame.vehicle_id
        if vehicle not in self.cav_ids:
            raise PIOwnerSessionError("PI frame is not a registered CAV")
        if vehicle in self._pending:
            raise PIOwnerSessionError("duplicate PI proposal in one owner tick")
        try:
            if vehicle not in self._initialized:
                self.pi.reset(vehicle, measured_ego_speed_mps=frame.speed_mps)
                self._initialized.add(vehicle)
            previous_raw = self.pi._state[vehicle][0]
            proposal = self.pi.propose(vehicle, timestamp_s=now_s,
                ego_speed_mps=frame.speed_mps,
                leader_speed_mps=frame.speed_mps + frame.front.relative_speed_mps,
                gap_m=frame.front.gap_m, u_report=self.u_report)
            requested = (proposal.command_mps - frame.speed_mps) / self.config.control_dt_s
            issued, target = direct_target(frame, requested, self.config)
            if issued is None:
                self.pi.discard(proposal)
                self.pi.invalidate(vehicle_ids=self.cav_ids, native_ids={vehicle})
                self._initialized.discard(vehicle)
                return DampingDecision(fallback_reason="infeasible_speed_bounds")
            self._pending[vehicle] = (proposal, issued, target, now_s)
            self.evidence.append(dict(vehicle_id=vehicle, timestamp_s=now_s,
                frame=asdict(frame), u_report=asdict(self.u_report),
                previous_raw_command_mps=previous_raw,
                generation=proposal.generation, raw_proposal_mps=proposal.command_mps,
                requested_acceleration_mps2=requested,
                issued_acceleration_mps2=issued, issued_target_speed_mps=target,
                owner_record=None))
            return DampingDecision(command_acceleration_mps2=issued, target_speed_mps=target)
        except BaseException as exc:
            self.stopped = True
            raise PIOwnerSessionError("PI proposal preparation failed") from exc

    def decide(self, frame, *, now_s):
        if not self._in_owner_call:
            raise PIOwnerSessionError("PI decide is only valid inside an owner tick")
        if frame is None:
            return DampingDecision(fallback_reason="missing_observation")
        fallback = self._prepared_fallback.get(frame.vehicle_id)
        if fallback is not None:
            expected_frame, expected_time, decision = fallback
            if frame != expected_frame or now_s != expected_time:
                raise PIOwnerSessionError("owner supplied a changed fallback frame or tick")
            return decision
        prepared = self._prepared.get(frame.vehicle_id) if frame is not None else None
        if prepared is not None:
            expected_frame, expected_time, decision = prepared
            if frame != expected_frame or now_s != expected_time:
                raise PIOwnerSessionError("owner supplied a changed PI frame or tick")
            return decision
        raise PIOwnerSessionError("owner supplied an unprepared PI frame")

    def record_callback(self, record):
        """Commit only the proposal matching the successful owner record."""
        if self.stopped:
            raise PIOwnerSessionError("PI owner session is stopped")
        if not self._in_owner_call:
            raise PIOwnerSessionError("owner callback is outside an owner tick")
        vehicle = record.vehicle_id
        if self._bound_owner is None or getattr(self._bound_owner, "_last", {}).get(vehicle) is not record:
            raise PIOwnerSessionError("owner callback is not the persisted owner record")
        if vehicle not in self._batch_expected or vehicle in self._callback_seen:
            raise PIOwnerSessionError("owner callback vehicle is outside or duplicated in this tick")
        self._callback_seen.add(vehicle)
        pending = self._pending.pop(vehicle, None)
        if pending is None:
            # Native, wall, or source-fallback owner rows are valid lifecycle
            # receipts; they have no PI proposal to commit.
            if record.owner == "damping" and record.fallback_reason is None and vehicle in self._expected_active:
                self.stopped = True
                raise PIOwnerSessionError("successful damping row has no staged PI proposal")
            self.receipts.append(ProposalReceipt(vehicle, record.timestamp_s, "owner_row_without_pi"))
            return
        proposal, issued, target, timestamp = pending
        try:
            if record.timestamp_s != timestamp:
                raise PIOwnerSessionError("owner tick does not match PI proposal")
            self.receipts.append(ProposalReceipt(vehicle, timestamp, "written_owner_receipt",
                                                 proposal.command_mps, target, record.fallback_reason))
            for evidence in reversed(self.evidence):
                if evidence["vehicle_id"] == vehicle and evidence["timestamp_s"] == timestamp:
                    evidence["owner_record"] = record.as_dict()
                    break
            if record.owner != "damping" or record.fallback_reason is not None:
                self.pi.discard(proposal)
                self.stopped = True
                raise PIOwnerSessionError("prepared PI proposal received a fallback owner row")
            if record.issued_acceleration_mps2 != issued or record.issued_target_speed_mps != target:
                raise PIOwnerSessionError("owner issued values do not match prepared PI bridge")
            self.pi.commit(proposal, writer_ack=True)
            self.receipts.append(ProposalReceipt(vehicle, timestamp, "committed_raw_proposal",
                                                 proposal.command_mps, target))
        except BaseException:
            self.pi.discard(proposal)
            self.stopped = True
            raise

    def abort(self, reason="owner batch failed"):
        """Discard unacknowledged proposals after an owner write exception."""
        for proposal, *_ in self._pending.values():
            self.pi.discard(proposal)
        self._pending.clear()
        self.stopped = True
        raise PIOwnerSessionError(reason)

    def finish_tick(self):
        if self._pending:
            self.abort("owner did not acknowledge every PI proposal")

    def step(self, owner, frames, *, now_s, live_ids, native_ids=(), wall_commands=None,
             departed_ids=(), id_reuse_ids=()):
        """Prepare a full tick, call real CommandOwnership exactly once, and close it."""
        try:
            now_s = float(now_s)
            if (not isfinite(now_s) or now_s < 0 or
                    abs(now_s / self.pi.sample_period_s - round(now_s / self.pi.sample_period_s)) > 1e-9):
                raise ValueError("PI owner tick must be finite, nonnegative and on its control grid")
        except (TypeError, ValueError, OverflowError):
            self.stopped = True
            raise
        if owner.controller is not self:
            raise ValueError("owner must use this PI adapter as its controller")
        if owner.cav_ids != self.cav_ids:
            raise ValueError("owner CAV identity does not match PI adapter")
        # WM-48: the owner may command in legacy mode 31 or the adopted
        # emergency-braking candidate mode 27 (WM-46 / campaign_stage1_v2:
        # EVERY CAV runs 27; the owner re-writes 31->27 on every live CAV at
        # every tick). Mode 27 is legacy 31 minus the regard-maximum-
        # deceleration bit, so SUMO may brake beyond the vType decel for
        # safety; chunk-1 evidence
        # (output/campaign_stage1_chunk1_launch/COLLISION_TALLY_FINAL.txt)
        # shows every tallied follower braking at exactly -3.0 m/s2 (the
        # vType decel) into a stopped queue - in native arms too - which is
        # why every campaign arm now commands in mode 27. Only the OWNER mode
        # identity widens to {31, 27}: the adapter's own commands and the
        # controller-config contract checked above (speed_mode 31, bounds
        # -3..1.5) are unchanged, and any other owner mode still stops the
        # session.
        if owner.cav_speed_mode not in (31, 27):
            self.stopped = True
            raise ValueError("PI prototype requires the mode31 or mode27 command owner")
        try:
            wall_commands = {v: float(a) for v, a in (wall_commands or {}).items()}
            if any(not isfinite(a) for a in wall_commands.values()):
                raise ValueError("wall requests must be finite")
        except (TypeError, ValueError, OverflowError):
            self.stopped = True
            raise
        if self._tick is not None and abs(now_s - self._tick - self.pi.sample_period_s) > 1e-9:
            self.stopped = True
            raise ValueError("PI owner tick must advance by exactly one sample period")
        live = set(live_ids) & self.cav_ids
        native, wall = set(native_ids), set((wall_commands or {}))
        if native & wall or not (native | wall) <= live:
            self.stopped = True
            raise ValueError("live/native/wall masks must contain only disjoint registered CAVs")
        if not set(frames) <= live or any(frame is not None and frame.vehicle_id != vehicle
                                          for vehicle, frame in frames.items()):
            self.stopped = True
            raise ValueError("frame identity does not match live CAV mask")
        missing = live - set(frames) - native - wall
        missing |= {v for v in live & set(frames) if frames[v] is None}
        stale = {v for v in live & set(frames) if frames[v] is not None and frames[v].timestamp_s != now_s}
        missing |= {v for v in live & set(frames) if frames[v] is not None and
                    (frames[v].front is None or frames[v].front_coverage_m <= 0)}
        invalid = {v for v in live - native - wall - missing - stale
                   if frame_problem(frames[v], now_s, self.config) is not None}
        missing |= invalid
        self.invalidate(live_ids=live, native_ids=native, wall_ids=wall,
                        missing_ids=missing, stale_ids=stale, departed_ids=departed_ids,
                        id_reuse_ids=id_reuse_ids)
        self._prepared.clear()
        self._prepared_fallback.clear()
        self._expected_active = set()
        self._batch_expected = set(live)
        self._callback_seen = set()
        self._in_owner_call = True
        self._bound_owner = owner
        try:
            for vehicle in sorted(live - native - wall - missing - stale):
                decision = self._compute_decision(frames[vehicle], now_s=now_s)
                self._prepared[vehicle] = (frames[vehicle], now_s, decision)
                self._expected_active.add(vehicle)
            for vehicle in sorted(live - native - wall - self._expected_active):
                candidate = frames.get(vehicle)
                reason = (frame_problem(candidate, now_s, self.config)
                          if candidate is not None else "missing_observation")
                self._prepared_fallback[vehicle] = (candidate, now_s,
                    DampingDecision(fallback_reason=reason or "missing_or_uncovered_leader"))
            records = owner.step(frames, now_s=now_s, live_ids=live,
                native_ids=native, wall_commands=wall_commands or {}, record_callback=self.record_callback)
            self.finish_tick()
            if self._callback_seen != self._batch_expected:
                raise PIOwnerSessionError("owner did not callback every live CAV")
            self._tick = now_s
            self._prepared.clear()
            self._prepared_fallback.clear()
            self._expected_active.clear()
            self._batch_expected.clear()
            self._callback_seen.clear()
            self._in_owner_call = False
            self._bound_owner = None
            return records
        except BaseException:
            for proposal, *_ in self._pending.values():
                self.pi.discard(proposal)
            self._pending.clear()
            self._prepared.clear()
            self._prepared_fallback.clear()
            self._expected_active.clear()
            self._batch_expected.clear()
            self._callback_seen.clear()
            self._in_owner_call = False
            self._bound_owner = None
            self.stopped = True
            raise
