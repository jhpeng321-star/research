"""AMENDMENT 75 spec helper (zero SUMO): turn registered LEARNED_TRIGGER plant specs (same plant / U0 binding, as
ops/a44_lcmech.a44_specs builds them) into A75 executor specs, and check an A75 record. Pure functions; the driver
that builds the base specs and the bank entry is the caller's."""
from __future__ import annotations

from copy import deepcopy


def a75_specs(base, arm, entry, *, family, tag, digest):
    """base: LEARNED_TRIGGER episode specs; entry: dict(path=<host bank path>, file_sha256=...); digest: the
    a75_holdforms.a75_digest() of the shipped executor. Returns new specs (inputs untouched)."""
    if arm not in ('A75_PLAN', 'A75_BANK') or (arm == 'A75_PLAN') != bool(family):
        raise ValueError('A75_PLAN needs a family; A75_BANK takes none')
    out = []
    for i, s in enumerate(base):
        ck = deepcopy(s['checkpoint'])
        ck.pop('learned_trigger_sha256', None)
        ck.pop('learned_trigger_model', None)
        ck.update(kind='a75', a75_arm=arm, a75_sha256=digest, a75_bank=dict(entry), a75_family=family)
        out.append(dict(s, arm=arm, checkpoint=ck, job_id=f'{tag}_{i:04}'))
    return out


def check_a75(record, spec):
    """Raises ValueError unless the record ran the registered A75 executor with receipts."""
    what = f'{spec["arm"]} {spec.get("world_id")}'
    if record.get('a75_sha256_loaded') != spec['checkpoint']['a75_sha256']:
        raise ValueError(f'{what}: a75_holdforms actually loaded differs from its registration')
    r = record.get('a75')
    if not isinstance(r, dict) or r.get('a75_arm') != spec['arm'] or 'holdform_summary' not in r:
        raise ValueError(f'{what}: a75 receipts missing')
    return r
