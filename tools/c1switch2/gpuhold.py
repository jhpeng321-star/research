"""Durable owner GPU hold; read-only collection and CPU arms remain available."""
from .operations import EVIDENCE

HOLD = EVIDENCE / 'OWNER2014_GPU_HOLD.json'


def active():
    # Clearing this operational flag requires a separately recorded reconciliation
    # of MAIN's subsequent explicit GPU RESUME line. It never resets science claims.
    return HOLD.exists()


def require_available():
    if active():
        raise ValueError('OWNER20:14 GPU HOLD: await and reconcile MAIN GPU RESUME')
