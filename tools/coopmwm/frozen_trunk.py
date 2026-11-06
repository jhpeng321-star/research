"""Bound frozen-MAPPO-trunk binding for TRAINING (and native-reference) workers (clean5a, MAIN 2026-11-06).

``tools.leverprobe.backend.setup(runtime, checkpoint)`` hash-pins the frozen MAPPO executor to
d9042cc5 (policy_00169). clean5a may use an S-MAPPOWZ snapshot as its frozen base. Mirroring N6's
eval-only ``nm_eval_runner.comparator_setup`` (c4/nm-finaleval 9f116b6c), this module calls the SAME
setup code with only its ``MAPPO_SHA`` global replaced by the BOUND sha, after verifying the checkpoint
path and file bytes against the binding at every load. Every other frozen check (runtime manifest,
SOURCE_SHA, nice/CPU, SUMO version, deterministic MAPPO) is unchanged.

Binding: environment ``COOPMWM_FROZEN_TRUNK=<absolute path>|<sha256>`` set by the run chain for every
child (like COOPMWM_G5_RUNTIME / COOPMWM_MAP_SPEC) and recorded in the run CONFIG (``frozen_trunk``).
Absent = the original setup call, byte-identical (d9042cc5 stays the default pin).
"""
import hashlib
import os
from pathlib import Path
import re
import types

ENV = 'COOPMWM_FROZEN_TRUNK'
DEFAULT_SHA256 = 'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'


def binding(environ=None):
    """(path, sha256) of the bound trunk, or None when unbound. Malformed values raise."""
    raw = (os.environ if environ is None else environ).get(ENV)
    if not raw:
        return None
    if raw.count('|') != 1:
        raise ValueError(ENV+' must be "<absolute path>|<sha256>"')
    path, sha = raw.split('|')
    if not Path(path).is_absolute() or not re.fullmatch('[0-9a-f]{64}', sha):
        raise ValueError(ENV+' needs an absolute checkpoint path and a lowercase sha256')
    return path, sha


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(runtime, checkpoint, *, backend=None, environ=None):
    """leverprobe setup honouring the bound trunk; unbound -> the original call (unchanged path)."""
    if backend is None:
        from tools.leverprobe import backend
    bound = binding(environ)
    if bound is None:
        return backend.setup(runtime, checkpoint)
    path, sha = bound
    if os.path.normpath(str(checkpoint)) != os.path.normpath(path):
        raise ValueError('frozen trunk checkpoint path differs from the bound base: '+str(checkpoint))
    if file_sha256(checkpoint) != sha:
        raise ValueError('frozen trunk checkpoint bytes differ from the bound sha256')
    original = backend.setup
    if sha == backend.MAPPO_SHA:
        return original(runtime, checkpoint)
    shim = types.FunctionType(original.__code__, dict(original.__globals__, MAPPO_SHA=sha),
                              original.__name__, original.__defaults__, original.__closure__)
    return shim(runtime, checkpoint)


def config_record(path, sha256, provenance):
    """CONFIG ``frozen_trunk`` block (the config sha binds the chosen base)."""
    if file_sha256(path) != sha256:
        raise ValueError('bound base file differs from its sha256')
    return dict(path=str(path), sha256=sha256, env=ENV, default_sha256=DEFAULT_SHA256,
                provenance=provenance, rule='only the frozen MAPPO trunk pin swaps; verified at every load')


def check_process(cfg):
    """Chain/release check: the process binding equals the CONFIG binding (both absent, or identical)."""
    record = cfg.get('frozen_trunk')
    bound = binding()
    if record is None:
        if bound is not None:
            raise ValueError(ENV+' is set but the config binds no frozen trunk')
        return None
    if bound != (record['path'], record['sha256']) or cfg.get('checkpoint') != record['path']:
        raise ValueError('process frozen-trunk binding differs from the config')
    return record
