"""Part B v0.4 per-CAV action inputs (BRIEF_PARTB_WALL_ACTION_INPUTS_DRAFT.md
item 1; owner decision (d) 2026-09-30: this is the v0.4 input layer).

Per-CAV action at decision time t, reconstructed ONLY from

  * ``switch_schedule.json``  -- the per-CAV switching schedule
     (wm44_branch_types/1: schedule.events = [{vehicle_id, time_s, action}]);
  * ``suffix_commands.jsonl`` / ``telemetry.jsonl`` command rows
     (kind == "command"; the ``owner`` field is the realised longitudinal
     writer per 1 s control tick).

The per-tick dispatcher action log is NEVER read -- it re-stamps old rows
(a known slice bug); asserted by static and functional open-guard tests
and by the builder attestation.

Canonical token vocabulary (DESIGN_PARTB_V2.md §3): native / damping / pi /
wall, plus ``unknown``.  A CAV with no command at t and no schedule entry is
``unknown`` and carries the unknown mask channel -- it is NEVER silently
stamped "native".

Fleet runs (the chunk-4 ``__native_base`` arms, the chunk-2/3 ``__base``
damping arms) carry their run-level policy from ``manifest.json``
(controller: native | damping; the fleet's own policy, not a per-CAV
schedule).  Where that run's commands are logged they VERIFY the fleet
policy (owners must be the fleet policy or the native fallback; at least one
owner must equal the fleet policy).  Base segments therefore use the base
run's own policy, as the brief requires.

Pure offline reconstruction over frozen logs; zero simulation.
"""
from __future__ import annotations

import json
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import numpy as np

# canonical action token vocabulary (token slot order -- frozen with
# CAV_RAW; see part4/partB/featurize.py ACTIONS)
ACTIONS_V04 = ("native", "damping", "pi", "wall", "unknown")
ACTION_UNKNOWN = "unknown"

# raw log spellings -> canonical token (wall spellings from
# campaign_walls_v1.wall_driver WALL_ACTION / PER_CAV_ACTIONS)
ACTION_ALIASES = {
    "native": "native",
    "damping": "damping",
    "pi": "pi",
    "pi-external-u": "pi",
    "wall": "wall",
    "wall-v2": "wall",
}

NATIVE_FALLBACK = "native"          # damping/wall controllers may yield to it

# v0.4a (WD-01-PB-V04A section 2): per-CAV arms are verified from the
# logged schedule plus the kind == "command" owner rows.  The command
# owner is the realised WRITER: a PI-external-U CAV is written through the
# damping writer (record owner "damping"; action_labels.py documents the
# asymmetry), and any damping/PI CAV may yield to the native fallback
# ("missing_or_uncovered_leader").  A native-scheduled CAV can only ever
# show a native owner.
SCHEDULE_ALLOWED_OWNERS = {
    "native": frozenset({"native"}),
    "damping": frozenset({"damping", "native"}),
    "pi": frozenset({"damping", "native"}),
}
ARM_KINDS = ("damping_fleet", "native_fleet", "pi_fleet", "fixed",
             "switching", "wall")
# Owners a wall branch run may show: the wall writer plus the declared
# native fallback (a wall driver may yield uncovered CAVs to native).
WALL_ALLOWED_OWNERS = frozenset({"wall", NATIVE_FALLBACK})
SHARE_ORDER = ("native", "damping", "pi")   # pi_* share columns

SCHEDULE_NAME = "switch_schedule.json"
COMMAND_FILES = ("suffix_commands.jsonl", "telemetry.jsonl")
MANIFEST_NAME = "manifest.json"


class ActionReconstructionError(ValueError):
    """A command/schedule contract violation; never silently relabelled."""


def canonical_action(raw: str) -> str:
    """Raw schedule/command spelling -> canonical token (loud on drift)."""
    a = ACTION_ALIASES.get(str(raw))
    if a is None:
        raise ActionReconstructionError(
            f"unknown action/owner spelling {raw!r} (known: "
            f"{sorted(ACTION_ALIASES)}); the token vocabulary is "
            f"{ACTIONS_V04} -- extend ACTION_ALIASES explicitly, never "
            "silently relabel")
    return a


def _first_existing(root: Path, name: str) -> Path | None:
    for cand in (root / name, root / "run" / name):
        if cand.is_file():
            return cand
    return None


def _read_schedule(path: Path) -> list[tuple[str, float, str]]:
    """[(vid, time_s, canonical action)] from wm44_branch_types/1."""
    d = json.loads(path.read_text(encoding="utf-8"))
    sched = d.get("schedule", d)     # arm-root file wraps it in 'schedule'
    out = []
    for ev in sched.get("events", ()):
        out.append((str(ev["vehicle_id"]),
                    float(ev["time_s"]),
                    canonical_action(ev["action"])))
    return out


def _read_commands(path: Path) -> list[tuple[str, float, str]]:
    """[(vid, time_s, canonical owner)] from kind == 'command' rows only."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("kind") != "command":
                continue          # pi receipts / evidence are NOT owners
            owner = r.get("owner")
            if owner is None:
                raise ActionReconstructionError(
                    f"{path}: command row without owner at "
                    f"t={r.get('timestamp_s')} vid={r.get('vehicle_id')!r}")
            out.append((str(r["vehicle_id"]), float(r["timestamp_s"]),
                        canonical_action(owner)))
    return out


class ActionTimeline:
    """Per-CAV piecewise-constant action reconstruction.

    per-CAV evidence: the merged (time, action) entries of the switching
    schedule (planned) and the command owners (realised; wins at equal
    times).  The action at t is the latest entry at or before t; before the
    CAV's first entry the action is ``unknown``.

    Fleet runs: ``fleet_policy`` is set (from the run manifest) and the
    whole roster carries that policy at every t; the command log is used to
    VERIFY it (verify_fleet_policy), never to relabel per tick.
    """

    def __init__(self, entries: dict[str, list[tuple[float, int]]],
                 *, fleet_policy: str | None,
                 command_owner_counts: dict[str, int],
                 n_schedule_events: int, n_command_rows: int,
                 sources: dict, sched_rows=(), cmd_rows=(),
                 schedule_authoritative: bool = False):
        # entries: vid -> time-sorted [(t, action_slot)] (realised command
        # evidence already merged over planned schedule entries at ties)
        self._times = {v: [e[0] for e in evs] for v, evs in entries.items()}
        self._slots = {v: [e[1] for e in evs] for v, evs in entries.items()}
        self.fleet_policy = fleet_policy
        self._owner_counts = dict(command_owner_counts)
        self.n_schedule_events = n_schedule_events
        self.n_command_rows = n_command_rows
        self.sources = sources
        self.schedule_authoritative = bool(schedule_authoritative)
        # raw (vid, time, canonical action) tuples, kept for the v0.4a
        # arm verification / schedule-share queries (small: <= ~1e5 rows)
        self._sched_rows = list(sched_rows)
        self._cmd_rows = list(cmd_rows)

    # ------------------------------------------------------------ build --
    @classmethod
    def from_run_dir(cls, run_dir, *, fleet_policy="auto",
                     command_files=COMMAND_FILES,
                     schedule_authoritative: bool = False
                     ) -> "ActionTimeline":
        """Reconstruct from one run directory (arm root or arm root/run).

        fleet_policy: "auto" resolves the fleet policy from the run
        manifest's ``controller`` field (native/damping); None disables the
        fleet attribution (pure per-CAV evidence); a string pins it.

        schedule_authoritative (v0.4a): for per-CAV branch arms the
        token is the PLANNED action (latest schedule entry at or before
        t); the kind == "command" owners only VERIFY it (verify_arm) and
        supply the token of a CAV that has no schedule entry.  Needed
        because a PI-external-U CAV is written with owner "damping":
        with realised commands winning, every PI arm would be mislabelled
        damping.  Default False = the unchanged v0.4 realised-wins
        behaviour.
        """
        root = Path(run_dir)
        sources: dict = {}
        sched_rows: list[tuple[str, float, str]] = []
        sp = _first_existing(root, SCHEDULE_NAME)
        if sp is not None:
            sched_rows = _read_schedule(sp)
            sources[SCHEDULE_NAME] = str(sp)
        cmd_rows: list[tuple[str, float, str]] = []
        for name in command_files:
            cp = _first_existing(root, name)
            if cp is not None:
                cmd_rows.extend(_read_commands(cp))
                sources[name] = str(cp)
        if fleet_policy == "auto":
            mp = _first_existing(root, MANIFEST_NAME)
            fleet_policy = None
            if mp is not None:
                ctrl = json.loads(mp.read_text(encoding="utf-8")) \
                    .get("controller")
                fleet_policy = (canonical_action(ctrl)
                                if ctrl in ACTION_ALIASES else None)
                if mp is not None:
                    sources[MANIFEST_NAME] = str(mp)
        elif fleet_policy is not None:
            fleet_policy = canonical_action(fleet_policy)

        merged: dict[str, list[tuple[float, int, int]]] = {}
        for vid, t, a in sched_rows:      # priority 0 (planned)
            merged.setdefault(vid, []).append((t, 0,
                                               ACTIONS_V04.index(a)))
        sched_vids = {v for v, _t, _a in sched_rows}
        for vid, t, a in cmd_rows:        # priority 1 (realised; wins ties)
            if schedule_authoritative and vid in sched_vids:
                continue                  # verification only, not a token
            merged.setdefault(vid, []).append((t, 1, ACTIONS_V04.index(a)))
        entries = {v: [(e[0], e[2]) for e in
                       sorted(evs, key=lambda e: (e[0], e[1]))]
                   for v, evs in merged.items()}
        owners = Counter(a for _v, _t, a in cmd_rows)
        return cls(
            entries, fleet_policy=fleet_policy,
            command_owner_counts={k: int(c) for k, c in owners.items()},
            n_schedule_events=len(sched_rows), n_command_rows=len(cmd_rows),
            sources=sources, sched_rows=sched_rows, cmd_rows=cmd_rows,
            schedule_authoritative=schedule_authoritative)

    # ------------------------------------------------------------ query --
    def action_at(self, vid: str, t: float) -> str:
        """Canonical action token of ``vid`` at time t (never silent)."""
        if self.fleet_policy is not None:
            return self.fleet_policy      # fleet run: uniform policy
        evs = self._times.get(str(vid))
        if not evs:
            return ACTION_UNKNOWN
        i = bisect_right(evs, float(t) + 1e-9)
        if i == 0:
            return ACTION_UNKNOWN         # before the CAV's first evidence
        return ACTIONS_V04[self._slots[str(vid)][i - 1]]

    def action_ids(self, vids, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(int8 slot ids, float32 unknown mask) for ``vids`` at t."""
        ids = np.zeros(len(vids), np.int8)
        unk = np.ones(len(vids), np.float32)
        for i, vid in enumerate(vids):
            a = self.action_at(vid, t)
            ids[i] = ACTIONS_V04.index(a)
            unk[i] = 1.0 if a == ACTION_UNKNOWN else 0.0
        return ids, unk

    def command_owner_counts(self) -> dict[str, int]:
        """Realised owner counts over kind=='command' rows (the acceptance
        accounting: e.g. chunk2 w8481 damping 556 / native 487)."""
        return dict(self._owner_counts)

    # -------------------------------------------------------- verify ----
    def verify_fleet_policy(self) -> dict:
        """Loud check of a fleet run's policy against its command log.

        Every logged command owner must be the fleet policy or the
        declared native fallback, and the fleet policy must actually
        appear when any commands are logged.  Returns the check record;
        raises ActionReconstructionError on violation."""
        if self.fleet_policy is None:
            raise ActionReconstructionError(
                "verify_fleet_policy needs a resolved fleet policy")
        allowed = {self.fleet_policy, NATIVE_FALLBACK}
        bad = {o: c for o, c in self._owner_counts.items() if o not in allowed}
        if bad:
            raise ActionReconstructionError(
                f"fleet policy {self.fleet_policy!r}: command owners "
                f"outside {{{sorted(allowed)}}}: {bad}")
        if self._owner_counts and \
                self._owner_counts.get(self.fleet_policy, 0) == 0:
            raise ActionReconstructionError(
                f"fleet policy {self.fleet_policy!r}: no command owner "
                f"matches it (owners {self._owner_counts}) -- the run "
                "manifest and its command log disagree")
        return dict(fleet_policy=self.fleet_policy,
                    owner_counts=dict(self._owner_counts),
                    n_command_rows=self.n_command_rows,
                    n_schedule_events=self.n_schedule_events,
                    ok=True)

    # ------------------------------------------- v0.4a: arm verification --
    def _per_cav_events(self) -> dict:
        ev: dict = {}
        for vid, t, a in self._sched_rows:
            ev.setdefault(vid, []).append((t, a))
        for v in ev:
            ev[v].sort()
        return ev

    @staticmethod
    def _sched_action(ev, times, vid, t):
        evs = ev.get(vid)
        if not evs:
            return None
        i = bisect_right(times[vid], float(t) + 1e-9)
        return evs[i - 1][1] if i else None

    def verify_arm(self, kind: str) -> dict:
        """WD-01-PB-V04A section 2: verify the arm's policy from its
        switch_schedule.json plus the kind == "command" owners; raise
        ArmPolicyMismatch on any disagreement (the source is refused).

        kind in ARM_KINDS.  damping_fleet / native_fleet on a run WITHOUT
        a switch schedule (the chunk-4 ``__base`` / ``__native_base``
        runs) use the run manifest's fleet policy verified against the
        command owners; the per-CAV branch arms (native / pi fleets,
        fixed, switching) use the schedule.  Every logged command row must
        carry an owner allowed by the scheduled action at its tick
        (SCHEDULE_ALLOWED_OWNERS); a damping/PI schedule must show the
        damping writer at least once (else the CAV never ran it).
        Returns the evidence record."""
        if kind not in ARM_KINDS:
            raise ValueError(f"unknown arm kind {kind!r} ({ARM_KINDS})")
        if kind == "wall":
            return self._verify_wall_arm()
        if kind in ("damping_fleet", "native_fleet") and \
                not self._sched_rows:
            want = kind.split("_")[0]
            if self.fleet_policy != want:
                raise ArmPolicyMismatch(
                    f"{kind}: run manifest fleet policy "
                    f"{self.fleet_policy!r} != {want!r}")
            rec = self.verify_fleet_policy()
            rec.update(kind=kind, mismatches=0)
            return rec
        ev = self._per_cav_events()
        if not ev:
            raise ArmPolicyMismatch(f"{kind}: no switch_schedule events")
        if not self._cmd_rows:
            raise ArmPolicyMismatch(f"{kind}: no kind=='command' rows")
        acts = {a for evs in ev.values() for _t, a in evs}
        first_acts = {evs[0][1] for evs in ev.values()}
        multi = sum(1 for evs in ev.values() if len(evs) > 1)
        if kind in ("damping_fleet", "native_fleet", "pi_fleet"):
            want = kind.split("_")[0]
            if acts != {want}:
                raise ArmPolicyMismatch(
                    f"{kind}: schedule actions {sorted(acts)} != "
                    f"{{{want!r}}}")
        elif kind == "fixed":
            if multi:
                raise ArmPolicyMismatch(
                    f"fixed: {multi} CAV(s) with more than one schedule "
                    "event (not a constant assignment)")
        elif kind == "switching":
            if not multi:
                raise ArmPolicyMismatch(
                    "switching: no CAV has more than one schedule event")
        bad: dict = {}
        used: dict = {}
        times = {v: [t for t, _a in evs] for v, evs in ev.items()}
        n_noev = n_before = 0
        n_writer = {"damping": 0, "pi": 0}
        for vid, t, owner in self._cmd_rows:
            evs = ev.get(vid)
            if not evs:
                n_noev += 1
                continue
            i = bisect_right(times[vid], float(t) + 1e-9)
            if i == 0:
                n_before += 1
                continue
            act = evs[i - 1][1]
            used[act] = used.get(act, 0) + 1
            if owner not in SCHEDULE_ALLOWED_OWNERS[act]:
                k = f"{act}->{owner}"
                bad[k] = bad.get(k, 0) + 1
            elif owner == "damping" and act in n_writer:
                n_writer[act] += 1
        if bad:
            raise ArmPolicyMismatch(
                f"{kind}: command owners disagree with the schedule: "
                f"{bad}")
        if n_noev or n_before:
            raise ArmPolicyMismatch(
                f"{kind}: {n_noev} command rows for CAVs without a "
                f"schedule entry, {n_before} before a CAV's first entry")
        if ({"damping", "pi"} & acts) and sum(n_writer.values()) == 0:
            raise ArmPolicyMismatch(
                f"{kind}: damping/pi scheduled but the damping writer "
                "never owned a tick of any CAV")
        return dict(kind=kind, ok=True, mismatches=0,
                    n_cavs=len(ev), n_schedule_events=len(self._sched_rows),
                    n_command_rows=len(self._cmd_rows),
                    schedule_actions=sorted(acts),
                    first_actions=sorted(first_acts),
                    n_cavs_multi_event=multi,
                    owner_counts=dict(self._owner_counts),
                    scheduled_command_ticks=used,
                    damping_writer_ticks_by_schedule=n_writer)

    def _verify_wall_arm(self) -> dict:
        """kind == "wall" (WM-N1 label coverage 2026-10-04): the walls-pilot
        branch runs carry NO switch_schedule.json -- the wall driver writes
        kind == "command" rows owned ``wall`` (or the declared native
        fallback).  Verification is owner-based, fail-closed:

        * a switch schedule must NOT exist (a scheduled wall arm would be a
          different, unregistered arm shape);
        * every command owner must be ``wall`` or the native fallback;
        * the wall writer must own at least one tick (else the run never
          assembled a wall and is not a wall arm).
        """
        if self._sched_rows:
            raise ArmPolicyMismatch(
                "wall: a switch schedule exists; wall arms are verified "
                "from their command owners only")
        if not self._cmd_rows:
            raise ArmPolicyMismatch("wall: no kind=='command' rows")
        bad = {o: c for o, c in self._owner_counts.items()
               if o not in WALL_ALLOWED_OWNERS}
        if bad:
            raise ArmPolicyMismatch(
                f"wall: command owners outside {sorted(WALL_ALLOWED_OWNERS)}"
                f": {bad}")
        if self._owner_counts.get("wall", 0) == 0:
            raise ArmPolicyMismatch(
                "wall: the wall writer never owned a command tick "
                f"(owners {self._owner_counts})")
        return dict(kind="wall", ok=True, mismatches=0,
                    n_command_rows=len(self._cmd_rows),
                    owner_counts=dict(self._owner_counts),
                    allowed_owners=sorted(WALL_ALLOWED_OWNERS))

    # ------------------------------------- v0.4a: schedule summary (pi) --
    def action_shares(self, vids, t: float, horizon: float) -> np.ndarray:
        """Planner-known continuation summary pi: the time-fraction of
        (t, t + horizon] the given CAVs spend under native / damping / pi
        (SHARE_ORDER), from the PLANNED per-CAV schedule (fleet runs: the
        manifest fleet policy, exact one-hot).  Time in `unknown` (no
        evidence) is excluded and the shares renormalised over known
        time; no known time -> all zeros."""
        out = np.zeros(len(SHARE_ORDER), np.float64)
        if not len(vids) or horizon <= 0:
            return out
        if self.fleet_policy is not None:
            if self.fleet_policy in SHARE_ORDER:
                out[SHARE_ORDER.index(self.fleet_policy)] = 1.0
            return out
        t0, t1 = float(t), float(t) + float(horizon)
        for vid in vids:
            times = self._times.get(str(vid))
            if not times:
                continue
            slots = self._slots[str(vid)]
            for k, ts in enumerate(times):
                lo = max(ts, t0)
                hi = min(times[k + 1] if k + 1 < len(times) else t1, t1)
                if hi <= lo:
                    continue
                a = ACTIONS_V04[slots[k]]
                if a in SHARE_ORDER:
                    out[SHARE_ORDER.index(a)] += hi - lo
                elif a != ACTION_UNKNOWN:
                    raise ActionReconstructionError(
                        f"action {a!r} has no share slot (wall arms are "
                        "out of scope for v0.4a)")
        tot = out.sum()
        return out / tot if tot > 0 else out


class ArmPolicyMismatch(ActionReconstructionError):
    """An arm's schedule/command evidence contradicts its declared
    policy; the source is refused (WD-01-PB-V04A section 2)."""
