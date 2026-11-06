"""N2 executor parity of HOLD-ONLY strategies (MAIN 2026-11-07: the registered substitute for lcexec_parity on holds).

Question: does the POLICY's N2 execution of a hold strategy (FULLPOL clean5a: ``lane_drop.LaneDropMove`` /
``fullpol_ev.SoloHold`` driven by ``lane_drop.member_command`` on the single-writer committed-offset column, NO lane
request for holds) issue the identical TraCI command sequence as the verified plant path (S-REGIMEU0 RuleHook /
StrategyCore: hold law through the plant writer + G5 own-lane target, i.e. laneChangeMode 513 on the members)?

Method (CPU, open loop on FIXTURE states, zero SUMO):
1. fixture = the zero-physics fake plant (wmverify_fake) driven by the plant-path hook of the strategy; every second's
   observation rows, the plant hook's commands (requested_raw) and G5 lane targets, and its activations are kept;
2. N2 path on the SAME states: per activation the strategy maps to N2 moves (one ``LaneDropMove`` of N2's
   resolve_family when every role shares v_w/hold/dmax/onset and 2-3 members, else one ``SoloHold`` per role; onset
   must lie in N2's registered window, else not representable); per second ``member_command(move, vid, el, view,
   speed=v, positions={vid: s})`` (own observable position);
3. both command streams go through the REAL env writer on a recording fake TraCI
   (lc_writer_harness.Harness: CavSumoEnv.step -> CommandOwnership -> slowDown/setSpeed, frozen _lane_control /
   _issue_lane_commands); the per-second TraCI write multisets are compared, split into speed writes
   (slowDown/setSpeed) and lane writes (setLaneChangeMode/changeLane).
MAPPO's pre-safety filter sits downstream of BOTH paths identically (requested accel in both), so it is not modelled.

The N2 executor source is FULLPOL's c4/fullpol-5a e36ece92 ``tools/coopmwm/lane_drop.py`` (vendored byte-identical
as fixtures/wmverify/n2_lane_drop_fullpol_e36ece92.pysrc, loaded as an isolated module; the runtime lane_drop of
this branch is untouched); SoloHold's arity rule is copied from fullpol_ev.py e36ece92.
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import sys
from collections import Counter
from pathlib import Path

from . import strategy_hook as SH

N2_FILE = Path(__file__).with_name('fixtures') / 'wmverify' / 'n2_lane_drop_fullpol_e36ece92.pysrc'
N2_SHA256 = '40515bc01f7f39933dc3e24fd999c45d9057d048bbb2c408a2534fcc324a4404'
N2_SOURCE = 'c4/fullpol-5a e36ece92 tools/coopmwm/lane_drop.py (+ fullpol_ev.SoloHold arity rule)'
PIN_SOURCE = ('c4/fullpol-5a 5cbd616e fullpol_ev.hold_pins (hold_lane_pin, MAIN 2026-11-07 registered, default OFF; '
              'lane_drop.py unchanged since e36ece92)')
SPEED_WRITES = ('slowDown', 'setSpeed', 'setSpeedMode')
LANE_WRITES = ('setLaneChangeMode', 'changeLane')
_MODULE = None


def load_n2():
    global _MODULE
    if _MODULE is None:
        raw = N2_FILE.read_bytes()
        if hashlib.sha256(raw).hexdigest() != N2_SHA256:
            raise ValueError('N2 executor fixture sha256 differs from FULLPOL e36ece92')
        import tools.coopmwm  # noqa: F401  (package context for the relative imports)
        name = 'tools.coopmwm._wmverify_n2_fullpol_e36ece92'
        spec = importlib.util.spec_from_file_location(name, N2_FILE, loader=importlib.machinery.SourceFileLoader(
            name, str(N2_FILE)))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _MODULE = mod
    return _MODULE


def solo_hold_cls(ld):
    class SoloHold(ld.LaneDropMove):
        """fullpol_ev.SoloHold (e36ece92): group_hold with ONE member, N2 law/release/descriptor."""

        def __post_init__(self):
            object.__setattr__(self, 'members', tuple(self.members))
            object.__setattr__(self, 'lanes', tuple(self.lanes))
            if self.family != 'group_hold' or len(self.members) != 1 or len(self.lanes) != 1:
                raise ValueError('a solo hold is group_hold with exactly one member and its lane')
            if not isinstance(self.members[0], str) or not self.members[0]:
                raise ValueError('solo hold member must be a vehicle id')
            if type(self.lanes[0]) is not int or self.lanes[0] not in range(ld.N_LANES):
                raise ValueError('solo hold needs its observed lane')
            ld._check_schedule(self.onset, self.hold, self.depth)
            ld._check_speeds(self.target_mps, self.gap_dv_mps)
            if self.zip or float(self.gap_dv_mps) != 0. or self.anchor != -1:
                raise ValueError('a solo hold holds ONE common speed, no zip/gap/anchor')
    return SoloHold


def hold_pins(active, own_lanes, zip_members=()):
    """fullpol_ev.hold_pins (c4/fullpol-5a 5cbd616e, verbatim): {vid: own current WM lane} for the active N2 hold
    members; zip-request members and vehicles without an own lane observation are excluded."""
    zips = set(zip_members)
    return {v: int(own_lanes[v]) for v in sorted(set(active)) if v not in zips and v in own_lanes}


def n2_moves(spec, members, wm_lanes, view):
    """N2 moves of one activation (members in role order, their WM lanes at activation) or raise ValueError."""
    ld = load_n2()
    roles = spec['roles']
    if any(r.get('lc_target') is not None for r in roles):
        raise ValueError('lane-change roles have no N2 hold representation')
    keys = {(float(r['v_w']), int(r['hold_s']), float(r['dmax']), int(r['onset_s'])) for r in roles}
    if len(keys) == 1 and 2 <= len(roles) <= ld.GROUP_SIZE[1]:
        v, h, d, o = keys.pop()
        lanes = tuple(int(x) for x in wm_lanes)
        family = ld.resolve_family(lanes, view)
        order = sorted(range(len(members)), key=lambda i: lanes[i]) if family == 'abreast_wall' else (
            sorted(range(len(members)), key=lambda i: lanes[i] != view.dropped_lane) if family == 'closing_pair'
            else list(range(len(members))))
        return [ld.LaneDropMove(family, tuple(members[i] for i in order), o, h, d, v, 0., False,
                                tuple(lanes[i] for i in order))]
    solo = solo_hold_cls(ld)
    return [solo('group_hold', (vid,), int(r['onset_s']), int(r['hold_s']), float(r['dmax']), float(r['v_w']), 0.,
                 False, (int(lane),)) for r, vid, lane in zip(roles, members, wm_lanes)]


def _state(row):
    s = float(row['s_m'])
    x = s + SH.S_OFF
    lane = int(row['lane'])
    if s < SH.RELEASE_S:
        edge, sumo = 'up', 2 - lane
    else:
        edge, sumo = (':n1_0' if x < 2704.0 else 'wz'), (1 if lane == 0 else 0)
    return dict(edge=edge, sumo_lane=sumo, x=x, v=float(row['v_mps']), allowed=31.29, is_cav=True)


def _writes(desc, roster, ticks):
    """ticks: [(el, states, base{vid: a}, lanes{vid: wm} | None)] -> [(el, kind, vid, value)] via the real writer."""
    from tools.coopmwm.lc_writer_harness import Harness
    h = Harness(vehicles={v: True for v in roster}, desc=desc)
    out = []
    for el, states, base, lanes in ticks:
        calls = h.tick(el, states, None, base=base or None, lanes=lanes or None,
                       lane_actors=frozenset(lanes) if lanes else None)
        out += [(c[0], c[2], c[1], c[3]) for c in calls]
    return out


def parity(spec, *, executor='plant', seed=989950, share=0.6, seconds=900, max_acts=3, n2_pin=True, both=True):
    """n2_parity record of one hold-only strategy executed by ``executor`` ('plant' or 'n2'; module docstring).
    ``n2_pin``: the N2 side with FULLPOL hold_lane_pin ON (5cbd616e: every active N2 law member gets its own current
    lane through the single lanes writer); ``both`` adds the other setting under 'pin_off' / 'pin_on'."""
    gated = spec.get('gate') is not None
    if gated:   # the observable gate only decides WHEN the rule activates; execution is that of the ungated twin
        spec = dict(spec, gate=None)
    rec = _parity(spec, executor=executor, seed=seed, share=share, seconds=seconds, max_acts=max_acts, n2_pin=n2_pin)
    if gated:
        rec['evaluated_on'] = 'ungated twin (the gate is a trigger condition; member execution is identical)'
    if both:
        other = _parity(spec, executor=executor, seed=seed, share=share, seconds=seconds, max_acts=max_acts,
                        n2_pin=not n2_pin)
        rec['pin_off' if n2_pin else 'pin_on'] = {k: other.get(k) for k in (
            'verdict', 'identical', 'speed_writes_identical', 'lane_writes_identical', 'write_counts', 'differences',
            'reason', 'note')}
    return rec


def _parity(spec, *, executor, seed, share, seconds, max_acts, n2_pin):
    from . import wmverify_fake as F
    from . import wmverify_hook as VH
    ld = load_n2()
    desc = VH.wz_descriptor()
    view = ld.MapView.from_descriptor(desc)
    rec = dict(executor=executor, n2_hold_lane_pin=bool(n2_pin), source=N2_SOURCE,
               pin_source=PIN_SOURCE if n2_pin else None, sha256=N2_SHA256, method='open-loop fixture states (fake plant), real env writer '
               'on a recording fake TraCI (lc_writer_harness.Harness)', fixture=dict(seed=seed, share=share,
                                                                                    seconds=seconds))
    hook = VH.make_hook(spec, executor=executor)
    res = F.run_episode(hook, seed=seed, share=share, seconds=seconds, record=True, keep_rows=True)
    by_t = {x['t']: x for x in res['trace']}
    acts = [a for a in hook.acts][:max_acts]
    if not acts:
        return dict(rec, verdict='no_activation_on_fixture', identical=None)
    moves_out, speed_ok, lane_ok, diffs, n_ticks = [], True, True, [], 0
    counts = Counter()
    for act in acts:
        t0, members = int(act['t']), list(act['members'])
        rows0 = {r['vid']: r for r in by_t[t0]['rows']}
        try:
            moves = n2_moves(spec, members, [int(rows0[v]['lane']) for v in members], view)
        except ValueError as exc:
            return dict(rec, verdict='not_representable', reason=str(exc), identical=False)
        moves_out.append([m.to_dict() for m in moves])
        end = max(t0 + int(r['onset_s']) + int(r['hold_s']) for r in spec['roles']) + 2
        plant_ticks, n2_ticks = [], []
        for t in range(t0, min(end, max(by_t) + 1)):
            rows = {r['vid']: r for r in by_t[t]['rows']}
            live = [v for v in members if v in rows]
            if not live:
                break
            states = {v: _state(rows[v]) for v in live}
            el = t - t0
            pa = {v: by_t[t]['desired'][v][0] for v in live if v in by_t[t]['desired']}
            pl = {v: lane for v, lane in by_t[t]['g5'].items() if v in live}
            na = {}
            for m in moves:
                for v in m.members:
                    if v in rows:
                        a = ld.member_command(m, v, el, view, speed=float(rows[v]['v_mps']),
                                              positions={v: float(rows[v]['s_m'])})
                        if a is not None:
                            na[v] = a
            plant_ticks.append((el, states, pa, pl))
            pins = hold_pins(set(na), {v: int(rows[v]['lane']) for v in live}) if n2_pin else None
            n2_ticks.append((el, states, na, pins or None))
            n_ticks += 1
        wp, wn = _writes(desc, members, plant_ticks), _writes(desc, members, n2_ticks)
        for kinds, flag in ((SPEED_WRITES, 'speed'), (LANE_WRITES, 'lane')):
            a = Counter((e, k, v, round(x, 9) if isinstance(x, float) else x) for e, k, v, x in wp if k in kinds)
            b = Counter((e, k, v, round(x, 9) if isinstance(x, float) else x) for e, k, v, x in wn if k in kinds)
            counts[f'{flag}_plant'] += sum(a.values())
            counts[f'{flag}_n2'] += sum(b.values())
            if a != b:
                if flag == 'speed':
                    speed_ok = False
                else:
                    lane_ok = False
                for item in sorted((a - b).elements())[:3]:
                    diffs.append(dict(act_t=t0, side='plant_only', write=list(item)))
                for item in sorted((b - a).elements())[:3]:
                    diffs.append(dict(act_t=t0, side='n2_only', write=list(item)))
    identical = speed_ok and lane_ok
    verdict = ('identical' if identical else 'speed_identical_lane_mode_differs' if speed_ok else
               'speed_differs')
    note = None
    if speed_ok and not lane_ok and not n2_pin:
        note = ('the plant path pins laneChangeMode 513 on hold members through the G5 own-lane target '
                '(S-REGIMEU0 solo2 path); N2 holds without hold_lane_pin issue no lane request (native lcMode)')
    elif speed_ok and not lane_ok:
        note = ('with hold_lane_pin the remaining lane-write differences are the seconds where the plant path pins a '
                'member that N2 does not command (e.g. before a role onset > 0)')
    return dict(rec, verdict=verdict, identical=identical, speed_writes_identical=speed_ok,
                lane_writes_identical=lane_ok, activations=len(acts), ticks=n_ticks, write_counts=dict(counts),
                n2_moves=moves_out, differences=diffs[:12], note=note)
