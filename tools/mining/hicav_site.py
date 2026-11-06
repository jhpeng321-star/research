# High-CAV-share probe (owner 2026-11-06 12:46, phase hicav_probe) sitecustomize.
# Deployed as <HICAV_ROOT>/source/sitecustomize.py. Identical to the coop_probe sitecustomize (mining G5 env
# overlay, sha-checked) with the lowcoop variant removed and ONE addition: when HICAV_WORLDS points to the
# registered HICAV world list (fresh seeds 995000-995199, 2700 vph, CAV share 0.6 / 0.8), the frozen
# protocol accepts exactly those worlds under the role 'hicav' and the seed scope admits exactly those
# (seed, 2700) cases. Every other world/role/seed check is unchanged. Scenario generation is untouched:
# the same CavSumoEnv textbook population reset(seed, demand_vph, penetration) as the training cells.
import os


def _hicav_worlds():
    import json
    path = os.environ.get('HICAV_WORLDS')
    if not path:
        return None
    return json.loads(open(path).read())['worlds']


def install_scope(worlds):
    """Admit exactly the registered HICAV (seed, demand) cases; sealed/blocked seeds still refuse."""
    import sys
    from traffic_shepherd import seeds as s
    from traffic_shepherd.c4 import seeds as c
    allowed = {(int(w['seed']), int(w['demand'])) for w in worlds}
    for seed, _ in allowed:
        if not 995000 <= seed <= 995199:
            raise ValueError('HICAV world outside the registered range')
        if seed in getattr(s, 'SEALED_SEEDS', ()) or seed in getattr(c, 'BLOCKED_SEEDS', ()):
            raise ValueError('HICAV seed is sealed/blocked')
    if getattr(s.require_development_case, '_hicav', False) and getattr(c.check_case, '_hicav', False):
        return
    old_s, old_c = s.require_development_case, c.check_case

    def base(seed, loading, role):
        if type(seed) is int and type(loading) is int and role == 'exposed' and (seed, loading) in allowed:
            return
        return old_s(seed, loading, role)

    def case(seed, demand_vph):
        if type(seed) is int and type(demand_vph) is int and (seed, demand_vph) in allowed:
            return
        return old_c(seed, demand_vph)
    base._hicav = case._hicav = True
    s.require_development_case, c.check_case = base, case
    for name, module in list(sys.modules.items()):
        if name.startswith('traffic_shepherd.') and module is not None:
            if getattr(module, 'check_case', None) is old_c:
                module.check_case = case
            if getattr(module, 'require_development_case', None) is old_s:
                module.require_development_case = base


def install_protocol(worlds):
    """tools.coopmwm.protocol.worlds('hicav') -> the registered HICAV list; all other roles unchanged."""
    import tools.coopmwm.protocol as P
    original = P.worlds

    def worlds_(role='train', training_seed=0, _o=original, _w=worlds):
        if role == 'hicav':
            return [dict(w) for w in _w]
        return _o(role, training_seed)
    worlds_._hicav = True
    if not getattr(P.worlds, '_hicav', False):
        P.worlds = worlds_
    # tools.coopmwm.accel_protocol (the guard accel_plant actually uses) captured the original
    # protocol.worlds at import as ORIGINAL_WORLDS; route only the 'hicav' role through it too.
    import tools.coopmwm.accel_protocol as AP
    if not getattr(AP.ORIGINAL_WORLDS, '_hicav', False):
        prior = AP.ORIGINAL_WORLDS

        def accel_worlds(role='train', training_seed=0, _o=prior, _w=worlds):
            if role == 'hicav':
                return [dict(w) for w in _w]
            return _o(role, training_seed)
        accel_worlds._hicav = True
        AP.ORIGINAL_WORLDS = accel_worlds


if os.environ.get('MINING_G5_ENV') or os.environ.get('HICAV_WORLDS'):
    import sys, importlib.util, hashlib
    import tools.leverprobe.backend as _b
    _orig = _b.setup

    def _setup(runtime, checkpoint):
        _orig(runtime, checkpoint)
        path = os.environ.get('MINING_G5_ENV')
        if path:
            want = os.environ.get('MINING_G5_ENV_SHA')
            if want and hashlib.sha256(open(path, 'rb').read()).hexdigest() != want:
                raise ValueError('mining G5 env overlay hash mismatch')
            import traffic_shepherd.c4 as pkg
            if 'traffic_shepherd.c4.env' in sys.modules and getattr(sys.modules['traffic_shepherd.c4.env'], '__file__', '') != path:
                raise ValueError('frozen env imported before the mining G5 overlay')
            spec = importlib.util.spec_from_file_location('traffic_shepherd.c4.env', path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules['traffic_shepherd.c4.env'] = mod
            spec.loader.exec_module(mod)
            pkg.env = mod
        hw = _hicav_worlds()
        if hw:
            install_scope(hw)
            install_protocol(hw)
    _b.setup = _setup
