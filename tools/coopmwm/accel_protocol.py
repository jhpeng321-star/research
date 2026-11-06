"""Prospective acceleration restart; superseded protocol remains reproducible."""
from . import protocol as old
from .protocol import CAP, UPDATES, HORIZONS, digest, checked, sha, REGISTRY, FORBIDDEN

EXTRA_START=990600
SUPERSEDED_CHARGE=811640
SEED=20261101
ORIGINAL_WORLDS=old.worlds
PHASES=dict(superseded=SUPERSEDED_CHARGE, train=2073600, checks=259200,
            dev=259200, mechanics=3600, calibration=57600, discovery=240000,
            final_test=460800, random_guide_test=115200)
# Later bank/cap amendments are additive runtime bindings, never mutations of
# this historical six-million quote (also used by the historical resume quote).


def worlds(role='train', training_seed=0):
    if role!='final_test':return ORIGINAL_WORLDS(role,training_seed)
    prior=ORIGINAL_WORLDS('final_test')
    from tools.coopbandit.protocol import CELLS
    extra=[dict(seed=EXTRA_START+i,cell=i//5,repeat=3+i%5,
        penetration=CELLS[i//5][0],demand=CELLS[i//5][1]) for i in range(60)]
    return prior+extra


def budget():
    return dict(total=CAP,phases=PHASES,allocated=sum(PHASES.values()),
        unallocated=CAP-sum(PHASES.values()),updates=24,episodes_per_update=24,
        training_seeds=[SEED],arms=['A','B','R'],checkpoint_rule='fixed final update24',
        final_panel_sha256=digest(worlds('final_test')),
        seed2_dropped='owner19:38 first-drop rule: two replications add1958400; total exceeds6M',
        with_seed2_total=sum(PHASES.values())+1958400)


def guard(world,role):
    if isinstance(world,dict) and 'map' in world:
        # N1: map worlds (wz3to2) have their own registry; merge-map worlds carry no 'map' key.
        from .map_runtime import guard_world
        guard_world(world,role);return
    if world not in worlds(role):raise ValueError('unregistered acceleration world/role')
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    require_development_case(world['seed'],world['demand'],'exposed')
    check_case(world['seed'],world['demand'])


def install_seed_scope():
    # Existing dual guard installation is extended only to registered metadata.
    original=old.worlds
    try:
        import traffic_shepherd.seeds as s
        import traffic_shepherd.c4.seeds as c
        import sys
        allowed={(w['seed'],w['demand']) for role in ('train','check','dev','final_test') for w in worlds(role)}
        os_,oc=s.require_development_case,c.check_case
        def base(seed,loading,role):
            if type(seed) is int and type(loading) is int and role=='exposed' and (seed,loading) in allowed:return
            return os_(seed,loading,role)
        def case(seed,demand_vph):
            if type(seed) is int and type(demand_vph) is int and (seed,demand_vph) in allowed:return base(seed,demand_vph,'exposed')
            return oc(seed,demand_vph)
        s.require_development_case,c.check_case=base,case
        for name,module in list(sys.modules.items()):
            if name.startswith('traffic_shepherd.') and module is not None:
                if getattr(module,'check_case',None) is oc:module.check_case=case
                if getattr(module,'require_development_case',None) is os_:module.require_development_case=base
    finally:old.worlds=original
