"""One consenting WALL group over the peer's disjoint per-CAV command owners.

Pass ``make_mixed_fleet(shared_runtime)`` as the shared Backend's
``mixed_factory``. The Backend must instantiate that class as ``factory(self)``.
The normal shared loader must expose the frozen switchable_wall module first;
this adapter neither installs extensions nor patches peer modules.

The only WALL-specific sensor translation is into WallLaw's public frame API.
Self reports determine candidate CAVs; observable RSU detections and nonconsenting
self reports remain available for its unchanged rear-risk check. No env.frame,
env.builder, history tape, or global vehicle telemetry is read. Slot capacity is
the environment's configured k, not a second observation/environment pipeline.

Information gap: standard C1SWITCH receipts expose carried lane, not junction.
Without an explicit public ``junction`` field, that lane is used with junction
false. Vehicles outside observable coverage are absent from rear surveillance.
Thus the frozen law/thresholds are reused, but full-truth baseline parity is not
claimed. A peer receipt extension is needed to recover junction information;
unobserved traffic must not be filled from simulator truth.
"""
import importlib
import math
from types import SimpleNamespace

import numpy as np


CONTROLLERS = ("native", "ctrl0", "mappo", "pi", "circles", "wall")


def _new_wall():
    return importlib.import_module(
        "traffic_shepherd.c4.switchable_wall").SwitchableWallController()


class _PublicWallView:
    """Stable routing slots for existing public receipts, never numerical ID features."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.builder = SimpleNamespace(vehicle_ids=[])
        self.indices, self.slots = {}, {}

    def frame(self, row, *, eligible, now_s):
        # No age cutoff is invented: only this decision's receipts enter the law.
        if row["now_s"] != now_s:
            raise ValueError("WALL requires the current observable decision")
        own, sensed = {}, {}
        for source, target in ((row.get("detections", ()), sensed),
                               (row["self_reports"], own)):
            for receipt in source:
                key = receipt["vehicle_id"]
                if key in target:
                    raise ValueError("duplicate public vehicle receipt")
                if receipt["timestamp_s"] != now_s:
                    continue
                if (not all(math.isfinite(receipt[f]) for f in ("s_m", "v_mps"))
                        or receipt["v_mps"] < 0 or receipt["lane"] not in (0, 1, 2)):
                    raise ValueError("invalid public WALL telemetry")
                target[key] = receipt
        sensed.update(own)  # self telemetry takes precedence over RSU sightings
        if len(sensed) > self.capacity:
            raise ValueError("public WALL slot capacity exceeded")
        self.slots = {key: slot for key, slot in self.slots.items() if key in sensed}
        free = iter(sorted(set(range(self.capacity)) - set(self.slots.values())))
        for key in sorted(sensed):
            if key not in self.indices:
                self.indices[key] = len(self.builder.vehicle_ids)
                self.builder.vehicle_ids.append(key)
            if key not in self.slots:
                self.slots[key] = next(free)
        k = self.capacity
        frame = SimpleNamespace(s=np.zeros(k), v=np.zeros(k), lane=np.full(k, -1),
                                lane_carry=np.full(k, -1), present=np.zeros(k, bool),
                                is_cav=np.zeros(k, bool), junction=np.zeros(k, bool),
                                vid_index=np.full(k, -1, dtype=np.int64))
        for key, receipt in sensed.items():
            slot = self.slots[key]
            frame.s[slot], frame.v[slot] = receipt["s_m"], receipt["v_mps"]
            frame.lane[slot] = frame.lane_carry[slot] = receipt["lane"]
            frame.junction[slot] = receipt.get("junction", False)
            frame.present[slot] = True
            frame.is_cav[slot] = key in own and key in eligible
            frame.vid_index[slot] = self.indices[key]
        return frame, set(own)


def make_mixed_fleet(shared, *, wall_factory=None):
    """Return a shared.MixedFleet subclass; optional factory supports CPU fakes.

    WALL is a sentinel in each context, never lib.Controller('wall', ...).
    Non-WALL stepping, PI evidence, and CTRL0 execute_residual are inherited.
    """
    new_wall = _new_wall if wall_factory is None else wall_factory

    class CavContext(shared.CavContext):
        def __init__(self, key, backend, fleet):
            super().__init__(key, backend)
            self.fleet = fleet

        def select(self, name):
            # Handover lifecycle clocks must follow the public controller clock.
            self.view.now = self.backend.env.now
            if name != "wall":
                return super().select(name)
            if self.name == name:
                return
            # Minimal sentinel handover, using this context's sole existing owner.
            previous = self.handover.controller
            if previous is not None and hasattr(previous, "stop"):
                previous.stop(now_s=self.backend.env.now)
            self.owner.release_all(live_ids={self.key})
            self.owner.controller = self.backend.env._circles
            self.view.pi_session, self.view._direct_pi = None, False
            self.pi = None
            self.handover.name, self.handover.controller = name, None
            self.handover.switches += 1

        def step(self, frames, *, now_s, live_ids):
            if self.name == "ctrl0":
                # The frozen env, under shared Backend.tick's zero_scope, owns
                # CTRL0 recomposition before BOTH half-steps. Native handback
                # here avoids a spurious one-second CIRCLES/slowDown writer.
                self.owner.wall_accel_max_mps2 = None
                return self.owner.step(
                    {self.key: frames[self.key]} if self.key in frames else {},
                    now_s=now_s, live_ids={self.key}, native_ids={self.key})
            if self.name != "wall":
                return super().step(frames, now_s=now_s, live_ids=live_ids)
            request = {self.key: self.fleet.requests[self.key]} if self.key in self.fleet.requests else {}
            # WALL uses the frozen baseline cap, not learned DIRECT's raised cap.
            self.owner.wall_accel_max_mps2 = None
            return self.owner.step({self.key: frames[self.key]} if self.key in frames else {},
                                   now_s=now_s, live_ids={self.key}, wall_commands=request)

        def close(self, *, live):
            if self.key in self.fleet.members:
                active = (self.fleet.live - (set() if live else {self.key})
                          if getattr(self.backend.env, "api", None) is not None else set())
                self.fleet.stop_group(live=active)
            return super().close(live=live)

    class GroupHandover(shared.PerCavHandover):
        def __init__(self, fleet):
            super().__init__(lambda key: CavContext(key, fleet.backend, fleet))
            self.fleet = fleet

        def sync(self, live, *, retired=()):
            live, retired = set(live), set(retired)
            self.fleet.live = live
            if self.fleet.members & ((set(self.contexts) - live) | retired):
                self.fleet.stop_group(live=live)
            super().sync(live, retired=retired)
            if not any(c.name == "wall" for c in self.contexts.values()):
                self.fleet.stop_group(live=live)

        def select(self, names):
            # Validate the whole batch before stopping a group or releasing peers.
            if (not set(names) <= set(self.contexts)
                    or any(name not in CONTROLLERS for name in names.values())):
                raise ValueError("invalid per-CAV hand-over")
            if any(names.get(key, "wall") != "wall" for key in self.fleet.members):
                self.fleet.stop_group(live=self.fleet.live)
            for key, name in names.items():
                old = self.contexts[key].name
                self.contexts[key].select(name)
                self.switches += int(old != name)
            if not any(c.name == "wall" for c in self.contexts.values()):
                self.fleet.stop_group(live=self.fleet.live)

    class MixedFleet(shared.MixedFleet):
        def __init__(self, backend):
            super().__init__(backend)
            self.wall, self.members, self.requests, self.live = None, set(), {}, set()
            self.public = _PublicWallView(backend.env.k)
            self.handover = GroupHandover(self)

        def stop_group(self, *, live):
            if self.wall is not None:
                # Frozen step closes TraCI before returning the final receipt.
                # Backend caches that receipt's terminal clock.
                now = (self.backend.now if hasattr(type(self.backend), "now")
                       else self.backend.env.now)
                self.wall.stop(now_s=now)
            # All active owners hand back before any new controller can step.
            for key in sorted(self.members):
                context = self.handover.contexts.get(key)
                if context is not None:
                    context.owner.release_all(live_ids={key} & set(live))
            self.wall, self.members, self.requests = None, set(), {}

        def step(self, owner, frames, *, now_s, live_ids,
                 disappeared_ids=(), reappeared_ids=()):
            live = set(live_ids) & set(self.backend.env.cav_ids)
            self.handover.sync(live, retired=set(disappeared_ids) | set(reappeared_ids))
            eligible = {key for key, c in self.handover.contexts.items() if c.name == "wall"}
            self.requests = {}
            if eligible:
                row = self.backend.observable()["history"][-1]
                frame, reported = self.public.frame(row, eligible=eligible, now_s=now_s)
                if not self.members <= (eligible & reported & live):
                    self.stop_group(live=live)
                    # A missing public member ends this tick's entire group.
                else:
                    if self.wall is None:
                        self.wall = new_wall()
                        self.wall.start(now_s=now_s)
                    commands = self.wall.commands(frame, self.public.builder,
                                                  now_s=now_s, live_cavs=eligible & reported & live)
                    if not set(commands) <= eligible & reported & live:
                        raise ValueError("WALL requested a nonconsenting or absent CAV")
                    self.requests = commands
                    law = self.wall.law
                    self.members = (set() if law is None else
                                    {self.public.builder.vehicle_ids[idx]
                                     for idx in law.members.values()})
            # Retirement was consumed above: avoid recreating re-entrants twice.
            return super().step(owner, frames, now_s=now_s, live_ids=live)

    return MixedFleet
