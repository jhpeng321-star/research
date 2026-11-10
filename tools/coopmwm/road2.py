"""A38 (MAIN 2026-11-10 amendment 38): second-road (wz2to1, 2->1 lane drop) bindings for the plant-side executor.

What this module does (and nothing else):
* ``HOLD_R2`` / ``HOLD_R2_DUO01``: the hand-written hold ported mechanically from duo12_v20, fixed in
  E:/agent-work/a38/HANDWRITTEN_HOLD_ROAD2.md (sha256 4a77c461...) before any wz2to1 outcome. HOLD_R2 is the
  PRIMARY port and it is DEGENERATE: one continuing lane means a single-CAV hold.
* ``install_oracle(O)``: oracle_plan._pick_wall hard-codes SUMO lanes (0, 1, 2), so on a 2-lane approach the wall
  trigger could never fire. For a wz2to1 process this rebinds it to the identical rule over lanes (0, 1): one CAV
  per approach lane, spread <= WALL_SPREAD, key (spread, -max x). The wz3to2 path is never touched (no-op unless
  the process map is wz2to1); oracle_plan source and digest 7409a7b4 are unchanged.
* ``hold_bank(spec, cell_key)``: a one-row unconditional bank doc in the a32 ``solo_bank`` form that
  oracle_plan.bank_plans reads.
* ``wm_feasibility(desc)``: documents why the WM-S model cannot take this map (A38 feasibility (b)).

Geometry: X_B = 2696, the zone x in [896, 2296] and release s = 75.44 are numerically identical to wz3to2.
"""
from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import sys
import types

MAP = 'wz2to1'
WALL_LANES = (0, 1)
HOLD_FILE_SHA256 = '4a77c461932ac8edd881877dc4093a63729e2f2584055bfab068b9d5b504f68e'


def _role(lane):
    return dict(lane=lane, rank=0, v_w=20.0, dmax=1.0, hold_s=120, onset_s=0, lc_target=None)


def _spec(name, lanes):
    return dict(name=name, trigger='wall', zone='full', spread_m=200.0, cooldown_s=120, slots=1, gate=None,
                start_gate=None, phases=None, roles=[_role(ln) for ln in lanes])


#: PRIMARY (degenerate single-CAV) port: one role per continuing lane = SUMO lane 1 only.
HOLD_R2 = _spec('duo12_v20_r2port', (1,))
#: documented two-CAV alternative (arm only if MAIN registers it before any outcome): all approach lanes abreast.
HOLD_R2_DUO01 = _spec('duo12_v20_r2duo01', (1, 0))
HOLDS = {'HOLD_R2': HOLD_R2, 'HOLD_R2_DUO01': HOLD_R2_DUO01}


def pick_wall_lanes(z, lanes=WALL_LANES, spread=200.0):
    """oracle_plan._pick_wall with the lane set as a parameter (identical body otherwise)."""
    per = [[c for c in z if int(c['lane']) == ln] for ln in lanes]
    if any(not x for x in per):
        return None
    best = None
    for combo in itertools.product(*per):
        xs = [float(c['x']) for c in combo]
        sp = max(xs) - min(xs)
        if sp <= spread:
            key = (sp, -max(xs))
            if best is None or key < best[0]:
                best = (key, list(combo))
    return None if best is None else {int(c['lane']): c for c in best[1]}


def install_oracle(O, map_name=None):
    """Rebind O._pick_wall for a wz2to1 process; returns True when installed, False (no-op) otherwise."""
    if map_name is None:
        from . import map_descriptor as md
        map_name = md.current().name
    if map_name != MAP:
        return False
    if getattr(O._pick_wall, '_a38', False):
        return True

    def _pick_wall(z):
        return pick_wall_lanes(z, WALL_LANES, O.G.WALL_SPREAD)
    _pick_wall._a38 = True
    _pick_wall._a38_orig = O._pick_wall
    O._pick_wall = _pick_wall
    return True


def hold_bank(spec, cell_key):
    """One unconditional rule row (a32_arms.solo_bank form) carrying ``spec``."""
    s = spec
    cond = dict(trigger=s['trigger'], zone=s['zone'], spread_m=s['spread_m'], cooldown_s=s['cooldown_s'],
                slots=s['slots'], gate=s['gate'], start_gate=s['start_gate'], phases=s['phases'],
                situation=dict(unconditional=True),
                roles=[dict(sumo_lane=r['lane'], rank=r['rank'], v_w=r['v_w'], dmax=r['dmax'], hold_s=r['hold_s'],
                            onset_s=r['onset_s'], lc_target_sumo=r['lc_target']) for r in s['roles']])
    from . import oracle_plan as O
    key = O.strategy_key(s)
    return dict(rows=[dict(regime=cell_key, family=s['name'], lb95_veh_s=0.0, plan_source='rule',
                           situational=dict(unconditional=True), template_sha256=None)],
                clean5c=dict(templates=[dict(cell=cell_key, family=s['name'], offer_condition=cond,
                                             template=dict(source=dict(strategy_key=key)))]))


def make_core(arm, cell_key, warmup_s=300):
    from . import oracle_plan as O
    if arm not in HOLDS:
        raise ValueError(f'unknown A38 hold arm {arm!r}')
    install_oracle(O)
    spec = HOLDS[arm]
    core = O.OracleCore(hold_bank(spec, cell_key), cell_key, warmup_s)
    if [p['family'] for p in core.plans] != [spec['name']] or core.plans[0]['row_class'] != 'unconditional':
        raise ValueError(f'A38: expected the single unconditional plan {spec["name"]}, got {core.plans} '
                         f'refused {core.refused}')
    return core


def spec_digest(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def wm_feasibility(desc):
    """(ok, reason) for the WM-S road binding of a descriptor (no torch import; mirrors wms.core.road rules)."""
    wm_lanes = {wm for (_e, _i), (wm, _x) in desc.lane_table.items()}
    missing = sorted(set(range(desc.frame.n_lanes)) - wm_lanes)
    if missing:
        return False, (f'WM lanes {missing} have no road lane on {desc.name}: wms.core.road.road_from_descriptor '
                       'cannot build lane extents, the WM entry process inserts into main lanes (0, 1), and the '
                       'learned lane head / PHYSICAL_TARGET_OF_LANE move lane-1 vehicles into WM lane 0; fixing that '
                       'is a model-side change (forbidden in A38; lane heads were trained with lane 0 present)')
    return True, 'all WM lanes exist'


def make_hook(arm, cell_key, warmup_s=300):
    """controller_hook (RULE-arm protocol), identical to a32_arms.make_hook except the core (make_core): the U0
    controller.apply runs first, then the hold core overrides the accel (and keeps the lane) of its CAV members only;
    a request for a non-CAV row raises (HDVs never commanded). ``job`` and ``prefixes`` are never read."""

    class A38Hook:
        def __init__(self):
            self.arm = arm
            self.core = make_core(arm, cell_key, warmup_s)
            self.controller = None

        def __call__(self, controller, job, prefixes):
            self.controller = controller
            previous = controller.apply

            def apply(rows, commands, now):
                desired = previous(rows, commands, now)
                accel, lanes = self.core.step(rows, commands, int(now))
                cav = {r['vid'] for r in rows if r.get('is_cav')}
                bad = (set(accel) | set(lanes)) - cav
                if bad:
                    raise RuntimeError(f'{arm}: request for non-CAV rows {sorted(bad)} (HDVs never commanded)')
                owner = self.core.owners()
                for vid, a in accel.items():
                    desired[vid] = dict(requested_raw=a, acceleration_residual=a - commands[vid],
                                        oracle_plan=owner.get(vid))
                if lanes:
                    targets = dict(getattr(controller, 'g5_lane_targets', {}) or {})
                    targets.update(lanes)
                    controller.g5_lane_targets = targets
                return desired

            controller.apply = apply

        def receipts(self):
            from . import oracle_plan as O
            return dict(self.core.receipts(), a38_arm=arm, hold=HOLDS[arm]['name'], hold_spec_sha256=spec_digest(HOLDS[arm]),
                        wall_lanes=list(WALL_LANES), pick_wall_a38=bool(getattr(O._pick_wall, '_a38', False)))

    return A38Hook()


def _oracle_unpatched(mod):
    """oracle_plan as registered (digest 7409a7b4 on wz3to2): the module's executed names with the ORIGINAL
    _pick_wall even after install_oracle (the wz2to1 rebinding is covered by this module's own source)."""
    names = tuple(mod.EXECUTED) + tuple(mod.CONSTANTS)
    proxy = types.SimpleNamespace(**{n: getattr(mod, n) for n in names}, G=mod.G)
    proxy._pick_wall = getattr(mod._pick_wall, '_a38_orig', mod._pick_wall)
    return proxy


def a38_digest():
    """sha256 of the A38 code as executed: oracle_plan.oracle_digest() of the registered (unpatched) executor + the
    full source of this module (the hold specs, the lane-(0, 1) wall rebinding, the bank row, make_core/make_hook)."""
    from . import oracle_plan as O
    parts = [O.oracle_digest(_oracle_unpatched(O)), inspect.getsource(sys.modules[__name__])]
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()
