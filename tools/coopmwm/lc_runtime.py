"""LCEXEC env-side CAV action interface ("knobs") for clean5 on a lane-drop map (DEFAULT OFF).

Opt-in per process with ``COOPMWM_LC_RUNTIME=1`` (like ``COOPMWM_G5_RUNTIME``); without the flag
:func:`install` returns None and imports/patches nothing, so the frozen runtime is untouched.
``accel_plant.bind_runtime`` calls it right after ``map_runtime.install``. With the flag the
process map must be a registered lane-drop map (wz3to2) whose ``map_runtime`` binding is already
installed; the merge map is never patched (fail closed). The CURRENT ``traffic_shepherd.c4.env.
CavSumoEnv`` (the map subclass on top of the frozen env / G5 overlay) is subclassed and rebound
process-wide; the frozen env bytes (59aecd0b) and the G5 overlay (5aac1769) are not modified.

Per control second the LCEXEC writer stashes ONE request for the NEXT ``env.step``::

    env.lc_tick({'accel': {vid: a}, 'native': [vid], 'lanes': {vid: [wm_target, duration_s]},
                 'lane_mode': {vid: 513 | 512}, 'tau': {vid: tau_s}})       # every key optional

Every vid must be a live CAV of ``env.cav_ids`` present in the current frame (HDVs are never
touched; ValueError otherwise); numbers finite, durations > 0, lane modes 513/512, tau inside the
registered ``lc_program.REGISTRY['tau']`` grid. The stash is consumed (and cleared) by the next
``step``. Semantics of that step (``direct`` mode only):

* ``accel``: the CAV's ``accel_by_slot`` entry is replaced by ``a`` -> the frozen
  CommandOwnership wall path -> ``damping.direct_target`` (clip to [max(-3, -v),
  min(direct_accel_max, allowed - v)]) -> ``vehicle.slowDown(vid, max(0, v + a'), 1.0)``; for
  |a| <= 3 and v <= allowed + 3 this is the probe ``slowDown(min(max(0, v + a), allowed), 1.0)``
  (to float rounding). ``native``: the entry is set to NaN -> SUMO car-following (``setSpeed(vid, -1)`` when
  the owner held it; such members are excluded from a damping/PI fallback via ``fallback_ids``).
  NOTE: replacing the entries BYPASSES the base executor's pre-safety MAPPO filter for program
  members (exactly as the SUMO probes did); SUMO speedMode 27 and the direct_target bounds
  remain. Members not in the request keep the base executor's command.
* lane control: the policy lane-control CAVs of the tick are the frozen ``policy_cavs`` (finite
  direct command + G5 lanes tensor [+ lane_actors]) UNION the live ``lane_mode`` keys (also
  without a finite command: native longitudinal + lane control, like the probes' lane_to
  members). An LCEXEC actor's laneChangeMode is its requested value (original saved once in the
  frozen ``_lc_original``; switched when the value changes; restored when it leaves, as the
  frozen ``_lane_control`` does); every other policy CAV gets LC_MODE_POLICY exactly as frozen.
* ``lanes``: issued by an extended copy of the frozen ``_issue_lane_commands`` (junction refusal,
  adjacent-only, LANE_TABLE index on the CURRENT edge, per-vid duration instead of
  LANE_REQUEST_S, edge-end deferral with ``v*duration + LANE_EDGE_GUARD_M``). On a lane-drop map
  the frozen "never into WM lane 2" refusal is lifted for the CLOSING lane iff that lane exists
  on the CAV's current edge and its remaining length ahead of the CAV exceeds
  ``v*duration + LANE_EDGE_GUARD_M`` (else ``deferred_edge_end``). A request for the CAV's
  CURRENT lane is a persistent lane HOLD (LCMOVES: changeLane(target, 2 s) every second, also after
  reaching it; lane changes into the closing lane only worked with it): issued as
  ``changeLane(vid, current SUMO index, duration)`` with status ``held`` under the same junction /
  edge-end / closing-lane rules (a hold in a merge-map ramp lane is refused). G5/N2/N9 ``lanes``
  tensor targets still go through the frozen method unchanged (target == current stays ``noop``);
  a vid with both an LCEXEC lateral request and a G5 target raises (single lateral writer).
  Per-vehicle telemetry for EVERY requested vid: ``info['lc_lane_vehicles'] = {vid: {target,
  status, completed}}`` with status ``issued`` / ``held`` / ``deferred_edge_end`` / ``refused_*``
  (executor receipts, e.g. merge-request timeouts), ``completed`` = post-physics WM lane == target
  (completion evidence of the frozen G5 telemetry; for ``held``: still in the lane), and the
  counts in ``info['lc_lane_commands']`` (never mixed into the frozen ``info['lane_commands']``).
* ``tau``: ``setTau(vid, x)`` when first requested (``getTau`` saved once), re-set when the value
  changes, ``setTau(saved)`` when the vid is no longer requested; a vid that left the network
  drops its saved value without a call. CAVs only.

A ``step`` without a stash is ``super().step`` EXACTLY (byte-identical) unless LCEXEC still holds
a vehicle setting (an LCEXEC lane mode or a saved tau); that step runs as an EMPTY request so the
saved originals are restored, after which the env is byte-identical again.

Known differences from the probe ``Arm`` (test_lc_runtime_cpu documents each):
* accel -> native WITHOUT a release: the owner issues ``setSpeed(vid, -1)`` (owned -> native); the
  probe issued nothing, so its last ``slowDown(target, 1 s)`` still pinned one more 0.5 s step;
* the probe's release ``setSpeed(vid, -1)`` for a member it never slowed is not issued (SUMO
  no-op: empty speed timeline); slowDown targets agree to float rounding for |a| <= 3;
* very fast vehicle (v > allowed + 3 m/s): direct_target's bounds are infeasible
  (max(-3, -v) > allowed - v), so CommandOwnership releases the CAV to native (``setSpeed(vid, -1)``
  if owned) where the probe issued ``slowDown(vid, allowed, 1 s)``; |a| > 3 is clipped to [-3, 3];
* the edge-end deferral (status ``deferred_edge_end``) also defers changes OUT of the closing lane
  and holds within ``v*duration + 10 m`` of the drop, where the probe kept requesting until its
  fallback / give-up position.
"""
from __future__ import annotations

from collections import Counter
import math
import os

FLAG = 'COOPMWM_LC_RUNTIME'
SCHEMA = 'lcexec-runtime-v1'
KEYS = ('accel', 'native', 'lanes', 'lane_mode', 'tau')
LANE_MODES = (513, 512)              # lc_program.LC_MODE_POLICY / LC_MODE_TRACI_ONLY

_INSTALLED = None
_RECORD: dict = {}
_CLASSES: dict = {}


def enabled():
    return os.environ.get(FLAG) == '1'


def _world(job):
    return job.get('world', job) if isinstance(job, dict) else job


# ---------------------------------------------------------------- request validation
def _number(x, name):
    if isinstance(x, bool):
        raise ValueError(f'lc request {name} must be a finite number')
    try:
        value = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'lc request {name} must be a finite number') from exc
    if not math.isfinite(value):
        raise ValueError(f'lc request {name} must be a finite number')
    return value


def _integer(x, name):
    if isinstance(x, bool) or not hasattr(x, '__index__'):
        raise ValueError(f'lc request {name} must be an integer')
    return int(x.__index__())


def _tau_registered(x):
    from .lc_program import REGISTRY
    rng = REGISTRY['tau']
    if rng.values:
        return any(abs(x - float(v)) <= 1e-9 for v in rng.values)
    return rng.ok(x)


def empty_request():
    return dict(accel={}, native=(), lanes={}, lane_mode={}, tau={})


def validate(requests, live_cavs):
    """Normalised copy of one tick request; ``live_cavs`` = live CAV ids (frame-present, cav_ids)."""
    if not isinstance(requests, dict):
        raise ValueError('lc request must be a dict')
    unknown = set(requests) - set(KEYS)
    if unknown:
        raise ValueError(f'unknown lc request keys {sorted(unknown)}')

    def vid_of(v):
        if not isinstance(v, str) or v not in live_cavs:
            raise ValueError(f'lc request vehicle {v!r} is not a live CAV (HDVs are never commanded)')
        return v

    out = empty_request()
    out['accel'] = {vid_of(v): _number(a, f'accel[{v}]') for v, a in dict(requests.get('accel') or {}).items()}
    native = [vid_of(v) for v in (requests.get('native') or ())]
    if len(set(native)) != len(native) or set(native) & set(out['accel']):
        raise ValueError('lc native list must be distinct and disjoint from accel')
    out['native'] = tuple(sorted(native))
    for v, spec in dict(requests.get('lanes') or {}).items():
        vid_of(v)
        if isinstance(spec, (str, bytes)) or len(spec) != 2:
            raise ValueError(f'lc lane request {v} must be [target_wm, duration_s]')
        target, duration = _integer(spec[0], f'lanes[{v}].target'), _number(spec[1], f'lanes[{v}].duration')
        if target < 0 or not duration > 0:
            raise ValueError(f'lc lane request {v}: target >= 0 and duration > 0 required')
        out['lanes'][v] = (target, duration)
    for v, mode in dict(requests.get('lane_mode') or {}).items():
        vid_of(v)
        mode = _integer(mode, f'lane_mode[{v}]')
        if mode not in LANE_MODES:
            raise ValueError(f'lc lane_mode must be one of {LANE_MODES}')
        out['lane_mode'][v] = mode
    for v, x in dict(requests.get('tau') or {}).items():
        vid_of(v)
        x = _number(x, f'tau[{v}]')
        if not _tau_registered(x):
            raise ValueError(f'lc tau {x} outside the registered lc_program tau values')
        out['tau'][v] = x
    return out


# ---------------------------------------------------------------- env subclass
def lc_env_class(base, desc):
    """LCEXEC subclass of ``base`` (the process CavSumoEnv) for map descriptor ``desc``.

    A non-lane-drop ``desc`` keeps the frozen "never into WM lane 2" rule (used by the CPU tests
    only; :func:`install` refuses such maps). Cached per (base, descriptor)."""
    if getattr(base, 'LC_RUNTIME', None) is not None:
        if base.LC_DESCRIPTOR_SHA256 != desc.digest:
            raise ValueError(f'{base.__qualname__} is already the LCEXEC env of map {base.LC_MAP}')
        return base
    key = (base, desc.digest)
    if key in _CLASSES:
        return _CLASSES[key]
    G = base._issue_lane_commands.__globals__       # the env module namespace (live rebinding)
    closing = int(desc.bottleneck['closing_lane']) if desc.kind == 'lane_drop' else None

    class CavSumoEnv(base):
        LC_RUNTIME = SCHEMA
        LC_MAP = desc.name
        LC_DESCRIPTOR_SHA256 = desc.digest
        LC_CLOSING_LANE = closing

        # ---- state
        def _lc_state(self):
            d = self.__dict__
            return (d.setdefault('_lc_mode_applied', {}), d.setdefault('_lc_tau_saved', {}),
                    d.setdefault('_lc_tau_applied', {}))

        def _lc_clear(self):
            for k in ('_lc_stash', '_lc_now', '_lc_mode_applied', '_lc_tau_saved', '_lc_tau_applied',
                      '_lc_lane_status', '_lc_lane_issued'):
                self.__dict__.pop(k, None)

        def _lc_outstanding(self):
            d = self.__dict__
            return bool(d.get('_lc_mode_applied') or d.get('_lc_tau_saved'))

        def _lc_live_cavs(self):
            import numpy as np
            f = self.frame
            ids = self.builder.vehicle_ids
            return frozenset(ids[int(f.vid_index[s])] for s in np.flatnonzero(f.present & f.is_cav).tolist()) \
                & frozenset(self.cav_ids)

        def reset(self, *args, **kwargs):
            self._lc_clear()
            return super().reset(*args, **kwargs)

        # ---- request
        def lc_tick(self, requests):
            """Stash one validated LCEXEC request for the NEXT ``step`` (see the module docstring)."""
            if self.__dict__.get('_lc_stash') is not None:
                raise ValueError('one lc_tick per env.step')
            if self.done or self.mode != 'direct':
                raise ValueError("lc requests need a running episode in 'direct' mode")
            stash = validate(requests, self._lc_live_cavs())
            self._lc_stash = stash
            return stash

        def _lc_slot(self, vid):
            f = self.frame
            slot = self.builder.pool.slot_of(vid)
            if (slot is None or not bool(f.present[slot]) or not bool(f.is_cav[slot])
                    or self.builder.vehicle_ids[int(f.vid_index[slot])] != vid):
                raise ValueError(f'lc request vehicle {vid!r} has no present CAV slot')
            return int(slot)

        def _lc_accel(self, accel_by_slot, stash):
            if not stash['accel'] and not stash['native']:
                return accel_by_slot                         # identical object: base path unchanged
            import numpy as np
            if accel_by_slot is None:
                arr = np.full(self.k, np.nan)
            else:
                try:
                    import torch
                    if torch.is_tensor(accel_by_slot):
                        accel_by_slot = accel_by_slot.detach().cpu().numpy()
                except ImportError:  # pragma: no cover
                    pass
                arr = np.array(np.asarray(accel_by_slot, dtype=np.float64).reshape(-1))
                if arr.shape[0] != self.k:
                    raise ValueError(f'accel_by_slot must have K={self.k} entries (shape [1,K])')
            for vid, a in stash['accel'].items():
                arr[self._lc_slot(vid)] = a
            for vid in stash['native']:
                arr[self._lc_slot(vid)] = np.nan
            return arr.reshape(1, -1)

        # ---- step
        def step(self, accel_by_slot=None, **kwargs):
            stash = self.__dict__.pop('_lc_stash', None)
            if stash is None:
                if not self._lc_outstanding():
                    return super().step(accel_by_slot, **kwargs)
                stash = empty_request()                      # restore LCEXEC-held settings
            if self.mode != 'direct':
                raise ValueError("lc requests need 'direct' mode")
            if getattr(accel_by_slot, 'native_residual', False):
                raise ValueError('lc requests cannot be combined with a native-residual command')
            lanes = kwargs.get('lanes') if kwargs.get('lanes') is not None else kwargs.get('lane')
            lateral = set(stash['lanes']) | set(stash['lane_mode'])
            if lanes is not None and lateral:
                clash = set(self._lanes_by_vehicle(lanes)) & lateral
                if clash:
                    raise ValueError(f'lane target from both LCEXEC and the lanes tensor: {sorted(clash)}')
            accel = self._lc_accel(accel_by_slot, stash)
            if stash['native'] and kwargs.get('fallback', 'native') != 'native':
                ids = kwargs.get('fallback_ids')
                kwargs['fallback_ids'] = frozenset(self.cav_ids if ids is None else ids) - set(stash['native'])
            self._lc_state()
            self._lc_lane_status, self._lc_lane_issued = {}, {}
            self._lc_now = stash
            try:
                state, info = super().step(accel, **kwargs)
            finally:
                self._lc_now = None
            completed = getattr(self, '_lane_completed', None) or self._lc_lane_completed
            vehicles, counts = {}, Counter()
            for vid, status in sorted(self._lc_lane_status.items()):
                target = int(stash['lanes'][vid][0])
                done = bool(status in ('issued', 'held') and vid in self._lc_lane_issued
                            and completed(vid, target))
                vehicles[vid] = dict(target=target, status=status, completed=done)
                counts[status] += 1
                if status == 'issued':
                    counts['completed' if done else 'not_completed'] += 1
            info['lc_lane_vehicles'] = vehicles
            info['lc_lane_commands'] = dict(counts)
            self.__dict__.setdefault('lc_lane_command_counts', Counter()).update(counts)
            return state, info

        def _lc_lane_completed(self, vid, target):
            # identical evidence to the G5 env's _lane_completed (pre-G5 frozen env lacks it)
            slot = self.builder.pool.slot_of(vid)
            return (slot is not None and bool(self.frame.present[slot])
                    and int(self.frame.lane[slot]) == target)

        # ---- the single lateral writer of the tick (called once per direct tick, before physics)
        def _lane_control(self, policy_cavs, live):
            lc = self.__dict__.get('_lc_now')
            if lc is None:
                return super()._lane_control(policy_cavs, live)
            api = self.api
            policy_mode = G['LC_MODE_POLICY']
            applied, _, _ = self._lc_state()
            want = {v: m for v, m in lc['lane_mode'].items() if v in live}
            policy = set(policy_cavs) | set(want)
            held = policy & set(self._lc_original)
            for vid in sorted(policy - set(self._lc_original)):
                self._lc_original[vid] = int(api.vehicle.getLaneChangeMode(vid))
                mode = want.get(vid, policy_mode)
                api.vehicle.setLaneChangeMode(vid, mode)
                if vid in want:
                    applied[vid] = mode
            for vid in sorted(held):
                mode = want.get(vid, policy_mode)
                if applied.get(vid, policy_mode) != mode:
                    api.vehicle.setLaneChangeMode(vid, mode)
                if vid in want:
                    applied[vid] = mode
                else:
                    applied.pop(vid, None)
            for vid in sorted(set(self._lc_original) - policy):
                original = self._lc_original.pop(vid)
                applied.pop(vid, None)
                if vid in live:
                    api.vehicle.setLaneChangeMode(vid, original)
            for vid in [v for v in applied if v not in self._lc_original]:
                del applied[vid]
            self._lc_tau(lc['tau'], live)
            self._lc_issue_lane_commands(lc['lanes'], policy)
            return None

        def _lc_tau(self, requested, live):
            _, saved, applied = self._lc_state()
            api = self.api
            for vid in sorted(set(saved) - set(requested)):
                original = saved.pop(vid)
                applied.pop(vid, None)
                if vid in live:
                    api.vehicle.setTau(vid, original)
            for vid, x in sorted(requested.items()):
                if vid not in live:
                    continue
                if vid not in saved:
                    saved[vid] = float(api.vehicle.getTau(vid))
                if applied.get(vid) != x:
                    api.vehicle.setTau(vid, x)
                    applied[vid] = x

        def _lc_lane_length(self, lane_id):
            length = self._lane_lengths.get(lane_id)
            if length is None:
                length = self._lane_lengths[lane_id] = float(self.api.lane.getLength(lane_id))
            return length

        def _lc_issue_lane_commands(self, targets, policy_cavs):
            """Extended copy of the frozen ``_issue_lane_commands`` for LCEXEC requests (docstring)."""
            api = self.api
            note, issued = self._lc_lane_status, self._lc_lane_issued
            for vid, (target, duration) in sorted(targets.items()):
                if vid not in policy_cavs:
                    note[vid] = 'refused_not_policy'
                    continue
                row = self._table.get(vid)
                if row is None or G['is_junction_lane'](row['lane_id']):
                    note[vid] = 'refused_junction'
                    continue
                current, s_now = G['lane_and_s'](row['lane_id'], row['lane_position_m'])
                hold = target == current                  # LCEXEC: persistent lane hold, not a noop
                closing_target = closing is not None and target == closing
                if (target not in (0, 1, 2) or (not hold and abs(target - current) != 1)
                        or (target == 2 and not closing_target)):
                    note[vid] = 'refused_not_allowed'     # adjacent only; never into / held in a ramp lane
                    continue
                edge = row['road_id']
                index = next((i for (e, i), (lane, _) in G['LANE_TABLE'].items()
                              if e == edge and lane == target), None)
                if index is None:
                    note[vid] = 'refused_no_lane'
                    continue
                margin = row['speed_mps'] * duration + G['LANE_EDGE_GUARD_M']
                if self._lc_lane_length(row['lane_id']) - row['lane_position_m'] <= margin:
                    note[vid] = 'deferred_edge_end'
                    continue
                if closing_target:
                    target_id = f'{edge}_{int(index)}'
                    _, s_end = G['lane_and_s'](target_id, self._lc_lane_length(target_id))
                    if s_end - s_now <= margin:
                        note[vid] = 'deferred_edge_end'
                        continue
                api.vehicle.changeLane(vid, int(index), float(duration))
                issued[vid] = (int(target), int(index), float(duration))
                note[vid] = 'held' if hold else 'issued'

    CavSumoEnv.__module__ = base.__module__
    CavSumoEnv.__qualname__ = base.__qualname__
    _CLASSES[key] = CavSumoEnv
    return CavSumoEnv


# ---------------------------------------------------------------- process binding
def install(job):
    """Bind the LCEXEC env subclass in this process (flag + lane-drop map); None without the flag."""
    global _INSTALLED
    if not enabled():
        return None
    from . import map_descriptor as md
    desc = md.for_world(_world(job))
    if desc.kind != 'lane_drop':
        raise ValueError(f'{FLAG}=1 needs a registered lane-drop map; process map is {desc.name!r}')
    import importlib
    env = importlib.import_module('traffic_shepherd.c4.env')
    base = env.CavSumoEnv
    if _INSTALLED is not None:
        if base is _INSTALLED and _INSTALLED.LC_DESCRIPTOR_SHA256 == desc.digest:
            return _INSTALLED
        raise ValueError('LCEXEC runtime already installed for another env class or map')
    from . import map_runtime
    bound = map_runtime._INSTALLED
    if getattr(base, 'MAP_NAME', None) != desc.name or bound is None or bound.digest != desc.digest:
        raise ValueError('map_runtime.install must bind the lane-drop map before the LCEXEC runtime')
    cls = lc_env_class(base, desc)
    record = dict(schema=SCHEMA, flag=FLAG, map=desc.name, descriptor_sha256=desc.digest,
                  env_module=getattr(env, '__file__', None),
                  g5_overlay=getattr(env, 'G5_RUNTIME_OVERLAY_SHA256', None),
                  base_class=f'{base.__module__}.{base.__qualname__}', closing_lane=cls.LC_CLOSING_LANE)
    record['env_class_rebound'] = map_runtime._rebind(base, cls, 'CavSumoEnv')
    env.LC_RUNTIME = SCHEMA
    _RECORD.clear()
    _RECORD.update(record)
    _INSTALLED = cls
    return cls


def installed_record():
    return dict(_RECORD)


# ---------------------------------------------------------------- N7 lane MAPPO exports (S-MAPPOWZL)
def _process_desc(desc=None):
    from . import map_descriptor as md
    desc = desc if desc is not None else md.current()
    if desc.kind != 'lane_drop':
        raise ValueError(f'LCEXEC lane rules/writer are lane-drop only; process map is {desc.name!r}')
    return desc


def lane_rule_table(desc=None):
    """JSON lane rule table of the clean5 action set (``lc_families.lane_rule_table``) for the
    process lane-drop map (or ``desc``)."""
    from . import lc_program as lp
    from .lc_families import lane_rule_table as table
    desc = _process_desc(desc)
    return table(lp.frame_from_descriptor(desc), int(desc.frame.n_lanes))


def mappo_lane_rules(desc=None):
    """``tools.nm_mappo.lane_actions.GateRules`` of :func:`lane_rule_table` (the N7 lane-MAPPO
    config entry ``lane_actions = "tools.nm_mappo.lane_actions:lcexec"`` resolves this); the plain
    table dict when that module is not in the tree."""
    table = lane_rule_table(desc)
    try:
        from tools.nm_mappo.lane_actions import GateRules, Transition
    except ImportError:
        return table
    return GateRules(transitions=tuple(Transition(t['source'], t['target'], t['s_lo'], t['s_hi'], t['guard'])
                                       for t in table['transitions']),
                     guard_points_s_m=tuple(table['guard_points_s_m']), n_lanes=table['n_lanes'],
                     sensor_range_m=table['sensor_range_m'], source=table['source'])


def _frozen_globals(env):
    """The env module namespace of the frozen ``_issue_lane_commands`` in ``env``'s MRO."""
    need = ('LANE_TABLE', 'LANE_REQUEST_S', 'LANE_EDGE_GUARD_M', 'is_junction_lane', 'lane_and_s')
    for klass in type(env).__mro__:
        fn = klass.__dict__.get('_issue_lane_commands')
        if fn is not None and all(k in fn.__globals__ for k in need):
            return fn.__globals__
    raise ValueError('no frozen _issue_lane_commands in the env class hierarchy')


def mappo_lane_writer(env, targets, policy_cavs, statuses=None):
    """N7 lane-MAPPO lane writer (signature of ``CavSumoEnv._issue_lane_commands``).

    The frozen method EXACTLY (target == current -> noop, adjacent only, junction refusal, LANE_TABLE
    index on the current edge, edge-end deferral, ``changeLane(vid, idx, LANE_REQUEST_S)``), except
    that on a lane-drop process map the CLOSING lane (WM 2) may be targeted where it exists on the
    CAV's current edge and its remaining length ahead exceeds ``v*LANE_REQUEST_S + LANE_EDGE_GUARD_M``
    (else ``deferred_edge_end``). On the original merge map it is the frozen rule (never into lane 2).
    Returns ``(issued {vid: (target, sumo index)}, Counter)``; ``statuses`` filled per vid."""
    from . import map_descriptor as md
    G = _frozen_globals(env)
    desc = md.current()
    closing = int(desc.bottleneck['closing_lane']) if desc.kind == 'lane_drop' else None
    api = env.api
    issued, counts = {}, Counter()
    note = statuses if statuses is not None else {}

    def length_of(lane_id):
        length = env._lane_lengths.get(lane_id)
        if length is None:
            length = env._lane_lengths[lane_id] = float(api.lane.getLength(lane_id))
        return length

    for vid, target in sorted(targets.items()):
        if vid not in policy_cavs:
            counts['refused_not_policy'] += 1
            note[vid] = 'refused_not_policy'
            continue
        row = env._table.get(vid)
        if row is None or G['is_junction_lane'](row['lane_id']):
            counts['refused_junction'] += 1
            note[vid] = 'refused_junction'
            continue
        current, s_now = G['lane_and_s'](row['lane_id'], row['lane_position_m'])
        if target == current:
            counts['noop'] += 1
            note[vid] = 'noop'
            continue
        closing_target = closing is not None and target == closing
        if target not in (0, 1, 2) or abs(target - current) != 1 or (target == 2 and not closing_target):
            counts['refused_not_allowed'] += 1
            note[vid] = 'refused_not_allowed'
            continue
        edge = row['road_id']
        index = next((i for (e, i), (lane, _) in G['LANE_TABLE'].items() if e == edge and lane == target), None)
        if index is None:
            counts['refused_no_lane'] += 1
            note[vid] = 'refused_no_lane'
            continue
        margin = row['speed_mps'] * G['LANE_REQUEST_S'] + G['LANE_EDGE_GUARD_M']
        deferred = length_of(row['lane_id']) - row['lane_position_m'] <= margin
        if not deferred and closing_target:
            target_id = f'{edge}_{int(index)}'
            _, s_end = G['lane_and_s'](target_id, length_of(target_id))
            deferred = s_end - s_now <= margin
        if deferred:
            counts['deferred_edge_end'] += 1
            note[vid] = 'deferred_edge_end'
            continue
        api.vehicle.changeLane(vid, int(index), G['LANE_REQUEST_S'])
        issued[vid] = (target, int(index))
        counts['issued'] += 1
        note[vid] = 'issued'
    return issued, counts


__all__ = ['FLAG', 'SCHEMA', 'KEYS', 'enabled', 'validate', 'empty_request', 'lc_env_class', 'install',
           'installed_record', 'lane_rule_table', 'mappo_lane_rules', 'mappo_lane_writer']
