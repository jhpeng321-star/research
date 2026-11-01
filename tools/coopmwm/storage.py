"""Reuse append-only no-refund accounting with COOPMWM phase limits."""
import types
from tools.coopnative import storage as old
from .protocol import CAP,CAPS
read,write,ledger,reconciled=old.read,old.write,old.ledger,old.reconciled
_reserve=types.FunctionType(old.reserve.__code__,dict(old.reserve.__globals__,CAP=CAP,CAPS=CAPS))


def ledger(root):
    import fcntl
    with (old.checked(root)/'ACCOUNTING.claim').open('a') as claim:
        fcntl.flock(claim,fcntl.LOCK_SH)
        return old.ledger(root)


def reserve(root,jobs,phase):
    # One supervisor, plus its diagnostic thread: serialize complete phase/cap checks.
    import fcntl
    with (old.checked(root)/'ACCOUNTING.claim').open('a') as claim:
        fcntl.flock(claim,fcntl.LOCK_EX)
        return _reserve(root,jobs,phase)
