# Cooperation probe (owner 2026-11-06 05:10) sitecustomize: the mining G5 env overlay (unchanged
# operator-2 logic) plus the OWNER-APPROVED probe-only scenario VARIANT for setting 3.
# Variant (PROBE_HDV_VARIANT=lowcoop_v1): every HDV vType written by the textbook population gets
# lcCooperative=0, lcCooperativeSpeed=0 and lcSpeedGain=0.2 (scenario setting, set in the route-file
# vType attributes; no driver-parameter or truth file is read). CAV vType (cav_typical) unchanged.
import os

VARIANTS = {'lowcoop_v1': {'lcCooperative': '0.0', 'lcCooperativeSpeed': '0.0', 'lcSpeedGain': '0.2'}}

if os.environ.get('MINING_G5_ENV'):
    import sys, importlib.util, hashlib
    import tools.leverprobe.backend as _b
    _orig = _b.setup

    def _setup(runtime, checkpoint):
        _orig(runtime, checkpoint)
        path = os.environ['MINING_G5_ENV']
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
        variant = os.environ.get('PROBE_HDV_VARIANT')
        if variant:
            over = VARIANTS[variant]
            from traffic_shepherd.c4 import population as pop
            if getattr(pop.vtype_attrs, '_probe_variant', None) != variant:
                base = pop.vtype_attrs

                def vtype_attrs(vtype_id, params, _base=base, _over=over):
                    attrs = _base(vtype_id, params)
                    if vtype_id != pop.TEXTBOOK_CAV_VTYPE:
                        attrs.update(_over)
                        vtype_attrs.count += 1
                    return attrs
                vtype_attrs.count = 0
                vtype_attrs._probe_variant = variant
                pop.vtype_attrs = vtype_attrs
                pkg.population = pop
    _b.setup = _setup
