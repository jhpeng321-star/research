"""Keep persistent read-only telemetry closures out of disposable dense replay.

The replay engine clears every callback closure cell after a future. Pilot
telemetry wrappers close over a persistent run state, so they must not enter
that graph. Only unperturbed FORWARD workers supply executed training telemetry;
dense replays have always been excluded from that denominator.
"""
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def vjp_guard(core_factory=None):
    from wms.control import parallel_futures, policy14
    core = policy14.make_cav_controller if core_factory is None else core_factory
    if getattr(core, '__closure__', None):
        raise ValueError('dense replay requires a module-level core controller factory')
    original = parallel_futures.backward_futures

    def backward(trainer, *args, **kwargs):
        # Covers serial _vjp AND the trainer snapshot sent to parallel VJP workers.
        with patch.object(trainer, 'controller_factory', core):
            return original(trainer, *args, **kwargs)

    with patch.object(parallel_futures, 'backward_futures', backward):
        yield
