"""Identity-pinned ablation of OUR pre-signed-cap LC implementation.

This executes project code from committed0a1e7e93, never downloaded SUMO code.
Geometry and teacher-forced observations are shared with the new kernel.
"""
import hashlib
import sys
import types

PRE_SIGNED_COMMIT = '0a1e7e9396bac5bc23ce1def030332ea6a79a8a3'
PRE_SIGNED_SHA = 'c1aa89fafb69ee18d34ba6b7c75445a6985a10096d41237439e8dd06189e1921'


def load_prior_kernel(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PRE_SIGNED_SHA:
        raise ValueError('unbound pre-signed LC source; refusing execution')
    module = types.ModuleType('wms.physics._rblane_pre_signed_ablation')
    module.__package__ = 'wms.physics'
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module.lane_change
