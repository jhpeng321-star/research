"""S-WMVERIFY executor: WM-C strategy rules on the S-REGIMEU0 u0 plant with a pluggable member EXECUTOR.

Trigger, member selection, roles and activation timing are always WM-C's ``strategy_hook.StrategyCore`` UNCHANGED
(the same ``strategy_search.select_members`` the WM imagination uses; cooldown, slots, observable gate). The members of
an activation are then executed by one of:

* ``plant``  - WM-C's hook path as is: hold law a = clip(v_w - v, -dmax, 1.0) requested as the absolute pre-safety
               acceleration through the plant's single writer (MAPPO safety downstream) and the role's lane target
               through ``controller.g5_lane_targets`` (frozen G5 path: own lane / adjacent change for 1 s, never into
               the closing lane). This is the S-REGIMEU0 solo2 path: DEFAULT for hold-only strategies.
* ``n2``     - the plant path WITHOUT the G5 own-lane target: hold law through the plant writer only, members keep
               their native laneChangeMode = what the policy's N2 hold execution (FULLPOL group_hold / SoloHold,
               lane_drop.member_command) issues at TraCI (wmverify_n2.parity); selectable for hold-only strategies.
* ``lcexec`` - the LCEXEC POLICY executor (c4/lcexec 0fe9574e; tools/coopmwm/lc_families.Protocol +
               lc_program.MemberExecutor + the own-sensing lane gate + env.lc_tick, process flag
               COOPMWM_LC_RUNTIME=1), exactly the production path of S-LCEXEC-SMOKE (lc_smoke.run_episode):
               activation members -> ``strategy_search.member_specs`` (the WM's own probe specs: LCMOVES 'prog'
               when any role changes lanes, nm2 'hold' otherwise) -> ``lc_roles.member_program`` ->
               ``lc_families.template_from_program`` -> ``Protocol.start`` (initiator + joins at the activation
               second) -> each second ``Protocol.receive`` (own rows / own 100 m sensing / teammates' t-1 reports)
               -> ``Protocol.commands`` -> ``env.lc_tick``. Program members bypass the MAPPO pre-safety filter
               exactly as in S-LCEXEC-SMOKE (speedMode 27, direct_target bounds and the env edge-end guard stay).
               DEFAULT for strategies with lane-change roles (MAIN 2026-11-07 ~00:0x: 'verified' = executable by
               the policy executor; the probe executor reads references beyond own sensing).

CAV rows only: every requested vid must be an is_cav row of this second (RuntimeError otherwise; HDVs are never
commanded). ``register_executor(name, cls)`` plugs another executor (``step(rows, commands, t) -> lc request``).
"""
from __future__ import annotations

from collections import Counter

from . import geometry as G
from . import strategy_hook as SH
from . import strategy_search as SS

MAP = G.MAP
END_S = 1200                               # S-REGIMEU0 episode stop
LC_CAP = 6                                 # lc_program.REGISTRY['cap'] max (<= 6 roles per strategy)
PROVENANCE = 'S-WMVERIFY wm_verify: WM-C strategy activations through the LCEXEC policy executor'
LCEXEC_PARITY = dict(id='S-LCEXEC-SMOKE', phase='lcexec_smoke', code='c4/lcexec 9685dc2f / dcb8a792',
                     evidence='c4/lcexec 0fe9574e evidence/lcexec_smoke_20261106/SMOKE_SUMMARY.json',
                     verdict='PASS (mechanics + safety); TraCI parity partial, differences classified '
                             '(gate, ref_unobservable, release_timing, precision, other)')


def has_lane_roles(spec):
    return any(r.get('lc_target') is not None for r in spec['roles'])


def resolve_executor(spec, *, lane_executor='lcexec', hold_executor='plant'):
    """Executor name of one strategy ('none' for u0)."""
    if spec is None:
        return 'none'
    name = lane_executor if has_lane_roles(spec) else hold_executor
    if name not in EXECUTORS:
        raise ValueError(f'unknown executor {name!r} (registered: {sorted(EXECUTORS)})')
    if name == 'n2' and has_lane_roles(spec):
        raise ValueError('the n2 executor has no lane requests (hold-only strategies)')
    return name


def _reports(table, cavs, now):
    """lc_smoke._reports (c4/lcexec 0fe9574e, verbatim): own SensorReport per CAV for the next second's receive."""
    from tools.coopmwm.merge_features import ObservedVehicle, SensorReport, SENSOR_RANGE_M
    out = {}
    vehicle = lambda r: ObservedVehicle(r['vid'], r['s_m'], int(r['lane']), r['v_mps'], r['length_m'])  # noqa: E731
    for vid in cavs:
        ego = table[vid]
        if int(ego['lane']) not in (0, 1, 2):
            continue
        sensed = [r for r in table.values() if r['vid'] != vid and abs(r['s_m']-ego['s_m']) <= SENSOR_RANGE_M
                  and int(r['lane']) in (0, 1, 2)]
        out[vid] = SensorReport(float(now), vehicle(ego), (0, 1, 2), tuple(vehicle(r) for r in sensed))
    return out


class _Commit:
    """MoveExecutor stand-in (lc_smoke._Commit): every member starts at the activation second."""

    def start(self, move, *, ego, observable_cav_ids, now, until):
        return True


def build_free_program(spec, member_rows, frame, t, end_s=END_S):
    """(ids, FreeProgram) of one activation: WM member specs -> LCEXEC member programs (lc_smoke order)."""
    from tools.coopmwm import lc_program as lp, lc_roles
    members = SS.member_specs(spec, member_rows)
    ids = tuple(m['id'] for m in members)
    programs = tuple(lc_roles.member_program(m['role'], m['params'], frame) for m in members)
    dur = max(int(p.start_el) + int(p.hold_s if p.hold_s is not None else end_s - t) for p in programs)
    fp = lp.FreeProgram(ids, programs, frame.name, (0,) * len(ids), max(1, min(dur, end_s - t)),
                        dict(study='S-WMVERIFY', strategy=spec['name'], strategy_key=SS.strategy_key(spec), t0=int(t)))
    return ids, fp


class LcexecExecutor:
    """Activation members through the LCEXEC production Protocol (see the module docstring)."""
    name = 'lcexec'
    needs_lc_runtime = True

    def __init__(self, spec, core, end_s=END_S):
        self.spec, self.core, self.end_s = spec, core, int(end_s)
        self.proto = self.frame = self.lf = None
        self.active = {}                         # vid -> ProgramOption
        self.reports_prev = {}
        self.sessions, self.refusals = [], []
        self.gate = Counter()
        self.released = Counter()

    def _init(self):
        from tools.coopmwm import lc_families as lf, lc_program as lp, map_descriptor as md
        desc = md.current()
        if desc.kind != 'lane_drop' or desc.name != MAP:
            raise RuntimeError(f'lcexec executor needs the bound {MAP} lane-drop map (process map {desc.name!r})')
        self.lf, self.frame = lf, lp.frame_from_descriptor(desc)
        cfg = dict(group_lane_moves=True, **lf.config_keys(provenance=PROVENANCE, cap=LC_CAP))
        self.proto = lf.Protocol(lf.settings(cfg))

    def _start(self, act, table, cavs, sensing, t):
        lf, proto = self.lf, self.proto
        ids = list(act['members'])
        sid = f'{lf.NAME}:{ids[0]}@{t}'
        try:
            ids_, fp = build_free_program(self.spec, [SH.to_sumo_row(table[v]) for v in ids], self.frame, t, self.end_s)
            template = lf.template_from_program(fp, self.frame, table)
            proto.receive(t, {v: table[v] for v in cavs}, sensing, self.reports_prev)
            options = []
            for slot, vid in enumerate(ids_):
                kind = 'initiate' if slot == 0 else 'join'
                body = template.to_dict() if slot == 0 else proto.sessions[sid]['template_dict']
                option = lf.ProgramOption(f'{lf.NAME}_{kind}', (vid,), sid, ids_[0], slot, body, 0,
                                          int(template.duration_s), 0., max(1, slot), proto.cap, 0., proto.range_m,
                                          int(table[vid]['lane']))
                if not proto.start(option, ego=vid, executor=_Commit(), observable_cav_ids=[vid], now=t):
                    raise ValueError(f'protocol refused slot {slot} ({vid})')
                options.append((vid, option))
        except ValueError as exc:
            for v in ids:                        # never leave a half-started session behind (lc_smoke)
                if proto.membership.get(v) == sid:
                    del proto.membership[v]
            proto.sessions.pop(sid, None)
            act['executor_refused'] = repr(exc)
            self.refusals.append(dict(t=int(t), members=ids, error=repr(exc)))
            return
        for v, o in options:
            self.active[v] = o
        act['session_id'] = sid
        act['template_sha256'] = lf.template_sha256(template.to_dict())
        self.sessions.append(dict(t=int(t), session_id=sid, members=ids, duration_s=int(template.duration_s)))

    def step(self, rows, commands, t):
        if self.proto is None:
            self._init()
        core = self.core
        # members of a running program are not selectable (StrategyCore keeps its own release bookkeeping)
        avail = {v: c for v, c in commands.items() if v in core.active or v not in self.active}
        n0 = len(core.acts)
        core.step(rows, avail, t)                # trigger / selection / timing only: its commands are not used
        table = {r['vid']: r for r in rows}
        cavs = sorted(v for v, r in table.items() if r.get('is_cav'))
        from tools.coopmwm.merge_features import SENSOR_RANGE_M
        sensing = {v: [r for r in rows if abs(float(r['s_m']) - float(table[v]['s_m'])) <= SENSOR_RANGE_M]
                   for v in cavs} if (self.active or len(core.acts) > n0) else {}
        if len(core.acts) > n0:
            self._start(core.acts[-1], table, cavs, sensing, t)
        tick = {}
        if self.active:
            self.proto.receive(t, {v: table[v] for v in cavs}, sensing, self.reports_prev)
            live_active = [(v, o) for v, o in self.active.items() if v in table]
            tick = self.proto.commands(t, live_active)
            for g in (self.proto.last_gate or {}).values():
                self.gate[str(g.get('reason'))] += 1
            for v in self.proto.just_released:
                self.active.pop(v, None)
                self.released['released'] += 1
            for v in [v for v in self.active if v not in table]:
                self.active.pop(v, None)
                self.released['left'] += 1
        self.reports_prev = _reports(table, cavs, t)
        bad = (set(tick.get('accel') or ()) | set(tick.get('native') or ()) | set(tick.get('lanes') or ())
               | set(tick.get('lane_mode') or ()) | set(tick.get('tau') or ())) - set(cavs)
        if bad:
            raise RuntimeError(f'S-WMVERIFY: LCEXEC request for non-CAV rows {sorted(bad)}')
        return tick or None

    def receipts(self):
        return dict(sessions=len(self.sessions), refusals=self.refusals[:20], n_refusals=len(self.refusals),
                    gate_withheld=dict(self.gate), released=dict(self.released), session_log=self.sessions[:50])


EXECUTORS = {'plant': None, 'n2': None, 'lcexec': LcexecExecutor}


def register_executor(name, cls):
    """Plug another member executor (cls(spec, core) with .step(rows, commands, t) -> env.lc_tick request)."""
    if name in EXECUTORS and EXECUTORS[name] is not cls:
        raise ValueError(f'executor {name!r} already registered')
    EXECUTORS[name] = cls


def make_hook(strategy, warmup_s=300, executor=None):
    """Harness hook instance (drop-in for regimeu0.RuleHook / strategy_hook.make_hook). executor None = default
    resolution (lane roles -> lcexec, hold-only -> plant)."""
    from tools.coopmwm import imitreplay as ir
    name = resolve_executor(strategy) if executor is None else executor
    if strategy is not None and name not in EXECUTORS:
        raise ValueError(f'unknown executor {name!r}')
    if strategy is not None and name == 'n2' and has_lane_roles(strategy):
        raise ValueError('the n2 executor has no lane requests (hold-only strategies)')

    class VerifyHook(ir.Recorder):
        def __init__(self):
            super().__init__()
            self.core = SH.StrategyCore(strategy, warmup_s) if strategy is not None else None
            self.strategy = None if strategy is None else strategy['name']
            self.executor = 'none' if strategy is None else name
            cls = EXECUTORS.get(self.executor)
            self.exe = cls(strategy, self.core) if cls is not None else None
            self.pending = self.sent = None
            self.now = None
            self.lc_ticks = 0
            self.lc_counts = Counter()           # env info['lc_lane_commands'] summed over ticks
            self.lc_vehicles = Counter()         # status counts of info['lc_lane_vehicles']
            self.lc_completed = 0
            self.cav_guard = 0

        @property
        def needs_lc_runtime(self):
            return bool(self.exe is not None and self.exe.needs_lc_runtime)

        @property
        def acts(self):
            return [] if self.core is None else self.core.acts

        @property
        def issued(self):
            # plant path: the hold commands requested through the single writer; lcexec: commands go through the
            # Protocol (its receipts), StrategyCore's own law outputs are discarded, so none are reported here
            return [] if self.core is None or self.exe is not None else self.core.issued

        @property
        def lane_held(self):
            return 0 if self.core is None else self.core.lane_held

        def receipts(self):
            out = dict(executor=self.executor, lc_ticks=self.lc_ticks, lc_counts=dict(self.lc_counts),
                       lc_vehicle_status=dict(self.lc_vehicles), lc_completed=self.lc_completed,
                       activations=len(self.acts), cav_guard_checks=self.cav_guard,
                       refused_activations=sum(1 for a in self.acts if a.get('executor_refused')))
            if self.exe is not None:
                out['protocol'] = self.exe.receipts()
            return out

        def __call__(self, controller, job, prefixes):
            self.controller = controller
            if self.core is None:
                return
            previous = controller.apply

            def apply(rows, commands, now):
                desired = previous(rows, commands, now)
                t = self.now = int(now)
                cav = {r['vid'] for r in rows if r.get('is_cav')}
                self.cav_guard += 1
                if self.exe is not None:
                    self.pending = self.exe.step(rows, commands, t)
                    return desired
                accel, lanes = self.core.step(rows, commands, t)          # WM-C strategy_hook path, unchanged
                bad = (set(accel) | set(lanes)) - cav
                if bad:
                    raise RuntimeError(f'S-WMVERIFY: request for non-CAV rows {sorted(bad)} (HDVs never commanded)')
                for vid, a in accel.items():
                    desired[vid] = dict(requested_raw=a, acceleration_residual=a - commands[vid],
                                        wmsearch_rule=self.strategy)
                if lanes and self.executor != 'n2':
                    targets = dict(getattr(controller, 'g5_lane_targets', {}) or {})
                    targets.update(lanes)
                    controller.g5_lane_targets = targets
                return desired

            controller.apply = apply
            if not self.needs_lc_runtime:
                return
            orig_request = controller.lane_request
            orig_actors = controller.lane_actors
            orig_observe = getattr(controller, 'observe_lane', None)

            def lane_request(env):
                req, self.pending = self.pending, None
                if req:
                    tick = getattr(env, 'lc_tick', None)
                    if tick is None:
                        raise RuntimeError('S-WMVERIFY lcexec executor needs the LCEXEC env '
                                           '(COOPMWM_LC_RUNTIME=1, tools.coopmwm.lc_runtime.install)')
                    tick(req)
                    self.sent = req
                    self.lc_ticks += 1
                return orig_request(env)

            def lane_actors():
                actors = frozenset(orig_actors())
                if self.sent:
                    actors |= frozenset(self.sent.get('lane_mode') or ())
                return actors

            def observe_lane(info):
                if orig_observe is not None:
                    orig_observe(info)
                if self.sent is not None:
                    self.lc_counts.update({k: v for k, v in (info.get('lc_lane_commands') or {}).items()
                                           if isinstance(v, (int, float))})
                    for _vid, r in (info.get('lc_lane_vehicles') or {}).items():
                        self.lc_vehicles[str(r.get('status'))] += 1
                        if r.get('completed') and r.get('status') == 'issued':
                            self.lc_completed += 1
                    self.sent = None

            controller.lane_request = lane_request
            controller.lane_actors = lane_actors
            controller.observe_lane = observe_lane

    return VerifyHook()


# ------------------------------------------------------------------------------------------ FULLPOL family metadata
def wz_descriptor():
    """The wz3to2 descriptor WITHOUT binding the process map (md.current() would bind as a side effect)."""
    import os
    from tools.coopmwm import map_descriptor as md
    from tools.coopmwm.lc_writer_harness import MAP_DIR
    desc = md._BOUND if md._BOUND is not None else (md.load(os.environ[md.ENV]) if os.environ.get(md.ENV) else None)
    return desc if desc is not None and desc.kind == 'lane_drop' else md.load(MAP_DIR)


def family_template(spec, frame=None):
    """lcexec-free-program-v1 SLOT template of a strategy (FULLPOL 'wm_verified' generator): slot k = role k's
    member program (the same member_specs -> lc_roles path the lcexec executor runs; for hold-only strategies the
    nm2 'hold' program), join rule of slot k = role k's WM lane at activation and ds to the initiator within the
    trigger spread. Validated by lc_families.template_of (registered grids)."""
    from tools.coopmwm import lc_families as lf, lc_program as lp
    if frame is None:
        frame = lp.frame_from_descriptor(wz_descriptor())
    lo, hi = SS.ZONES[spec['zone']]
    rows = [dict(id=f'm{k}', edge='up', lane=int(r['lane']), x=(lo + hi) / 2.0, v=20.0)
            for k, r in enumerate(spec['roles'])]
    _ids, fp = build_free_program(spec, rows, frame, 0, END_S)
    spread = float(G.WALL_SPREAD if spec['trigger'] == 'wall' else spec['spread_m'])
    rules = (None,) + tuple(dict(lane=SH.wm_lane_of(r['lane']), ds=(-spread, spread)) for r in spec['roles'][1:])
    programs = tuple(fp.programs)
    tpl = lp.FreeProgram(tuple(f'@{k}' for k in range(len(programs))), programs, frame.name, (0,) * len(programs),
                         int(fp.duration_s), dict(fp.source, t0=None, template_of='strategy'),
                         rules if len(programs) > 1 else ())
    lf.template_of(tpl, cap=LC_CAP, frame=frame, n_lanes=3)
    d = tpl.to_dict()
    return d, lf.template_sha256(d)


def offer_condition(spec):
    """Trigger geometry the verification used (strategy_search.select_members), in SUMO x, WM s and distance
    upstream of the drop; the observable gate; cooldown/slots."""
    lo, hi = SS.ZONES[spec['zone']]
    roles = [dict(slot=k, sumo_lane=int(r['lane']), wm_lane=SH.wm_lane_of(r['lane']), rank=int(r['rank']),
                  onset_s=int(r['onset_s']), hold_s=int(r['hold_s']), v_w=float(r['v_w']), dmax=float(r['dmax']),
                  lc_target_sumo=r['lc_target'],
                  lc_target_wm=None if r['lc_target'] is None else SH.wm_lane_of(r['lc_target']))
             for k, r in enumerate(spec['roles'])]
    rule = ('wall: one CAV per SUMO up lane 0/1/2 in the zone, spread <= 200 m, key (spread, -max x); roles take the '
            'member of their lane' if spec['trigger'] == 'wall' else
            'lanes: per role the rank-th most downstream zone CAV of its SUMO lane (ties by id), members distinct, '
            f'spread <= {float(spec["spread_m"]):g} m')
    gate = None if spec.get('gate') is None else dict(
        max_mean_speed_mps=float(spec['gate']), field_x=list(SS.FIELD_X),
        field_upstream_m=[0.0, round(G.X_B - SS.FIELD_X[0], 2)], lane='SUMO up_0 (WM 2, closing)',
        observed='CAV self + CAV-detected rows (|ds| <= 100 m, |dWM lane| <= 1)', closed_if_no_observation=True)
    return dict(trigger=spec['trigger'], zone=spec['zone'], zone_x=[lo, hi],
                zone_s=[round(lo - SH.S_OFF, 2), round(hi - SH.S_OFF, 2)],
                zone_upstream_m=[round(G.X_B - hi, 2), round(G.X_B - lo, 2)], edge='up',
                spread_m=float(G.WALL_SPREAD if spec['trigger'] == 'wall' else spec['spread_m']), member_rule=rule,
                roles=roles, own_wm_lanes=sorted({x['wm_lane'] for x in roles}), gate=gate,
                cooldown_s=int(spec['cooldown_s']), slots=int(spec.get('slots', 1)), decision='every whole second '
                't >= 300 s once t >= next_ok (= last activation + cooldown) and fewer than `slots` activations run')
