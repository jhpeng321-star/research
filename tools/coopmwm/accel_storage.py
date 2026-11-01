"""New namespace, unchanged append-only accounting, all historical charges protected."""
import types
from tools.coopnative import storage as old
from .accel_protocol import CAP,PHASES,SUPERSEDED_CHARGE
from .storage import ledger
read,write=old.read,old.write
_reserve=types.FunctionType(old.reserve.__code__,dict(old.reserve.__globals__,CAP=CAP-SUPERSEDED_CHARGE,CAPS=PHASES))


def reserve(root,jobs,phase):
    import fcntl
    with (root/'ACCOUNTING.claim').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _reserve(root,jobs,phase)


def charged(root):return SUPERSEDED_CHARGE+sum(x['sim_s'] for x in ledger(root))
