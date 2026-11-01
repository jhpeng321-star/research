"""Reuse append-only no-refund accounting with COOPMWM phase limits."""
import types
from tools.coopnative import storage as old
from .protocol import CAP,CAPS
read,write,ledger,reconciled=old.read,old.write,old.ledger,old.reconciled
reserve=types.FunctionType(old.reserve.__code__,dict(old.reserve.__globals__,CAP=CAP,CAPS=CAPS))
