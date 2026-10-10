"""E2 arm contrasts: fixed/switching mix, wall and trajectory arms vs native.

Registration ``WM-N1-EVALSLICE-20261004`` (E2) adds arm-level contrasts to the
CAL-tune scorer alongside ``pi-native``.  Unlike the historical policy
contrasts (``pi`` / ``damping`` / ``wall``), each new contrast names the exact
v0.4a arm whose outcome is compared with the *same physical world and t0*
``native`` reference:

* ``fixed_mix-native``   pooled over its members (separately: ``fixed1``,
  ``fixed2``);
* ``switching_mix-native`` pooled over its members (separately:
  ``switching1``..``switching3``);
* ``wall-native``        pooled over its members (separately: ``w-lo-early``,
  ``w-lo-late``, ``w-mid-early``, ``w-mid-late``);
* ``trajectory-native``  pooled over the direct-speed arms (separately:
  ``arm0``..``arm3``).

This module is pure and label-schema agnostic: :func:`build_arm_cells` consumes
plain records (mappings or objects with ``world``/``source``/``t0_s``/``t``/
``tau``/``tau_available``/``total_delay_ref_s``) and returns per-contrast
``(tau)`` empirical paired-dV quantile cells.  The consumer is
:mod:`evalh.scorer.build_climatology_v2`, which writes them under the existing
``wm_n1_train_climatology/1`` schema so
:class:`evalh.scorer.baselines.TrainClimatologyBaseline` reads them unchanged.

Fail-closed rules (mirroring the V1 TRAIN climatology):

* only rows with ``t == t0_s`` (the t0 mapper), ``tau_available > 0`` and a
  tau in the registered set enter a cell;
* the reference is exactly the ``native`` arm on the same ``(world, t0, tau)``;
* a duplicate ``(world, t0, tau, source)`` treatment row or ``(world, t0, tau)``
  reference row raises;
* a member with no reference sibling is OMITTED (never zero-filled).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from evalh.protocol import N_QUANTILES, QUANTILE_LEVELS

#: Reuse the V1 schema: the frozen baseline reader needs no change.
SCHEMA = "wm_n1_train_climatology/1"

NATIVE = "native"

#: Registered outcome taus (seconds) of the scorer metric loop.
TAUS_S = (30.0, 60.0, 120.0, 300.0, 570.0)

#: ``(group, (member arms...))`` in registration order.  The pooled contrast is
#: ``<group>-native``; each member also gets its own ``<arm>-native`` contrast.
ARM_GROUPS = (
    ("fixed_mix", ("fixed1", "fixed2")),
    ("switching_mix", ("switching1", "switching2", "switching3")),
    ("wall", ("w-lo-early", "w-lo-late", "w-mid-early", "w-mid-late")),
    ("trajectory", ("arm0", "arm1", "arm2", "arm3")),
)
GROUP_MEMBERS = dict(ARM_GROUPS)
MEMBER_TO_GROUP = {arm: group for group, arms in ARM_GROUPS for arm in arms}
ALL_ARMS = tuple(MEMBER_TO_GROUP)
E2_ALLOWED_ARMS = ("native", "pi") + ALL_ARMS

#: Decision-quality *arm types* (the pooled controller families plus the two
#: historical baselines and the damping arm).  These are the arms the TRAIN-only
#: fixed-best V2 (``build_fixed_best_v2``) ranks; the arm-contrast cells keep the
#: per-member names.
E2_ARM_TYPES = ("native", "pi", "fixed_mix", "switching_mix", "wall",
                "trajectory", "damping")

#: The direct-speed trajectory members.  Fix 3 (2026-10-05): under
#: ``--dq-arms e2`` each of these is its OWN decision-quality candidate (its own
#: realized cost and its own head prediction from its own plan); ``trajectory``
#: is only a *fixed-best arm type*, never a scored candidate.
TRAJECTORY_ARMS = ("arm0", "arm1", "arm2", "arm3")

#: Candidate labels whose rows are still POOLED by mean under ``--dq-arms e2``
#: because their members are random mixes, not distinct plans.
E2_DQ_POOL_ARMS = ("fixed_mix", "switching_mix", "wall")

#: The ``--dq-arms e2`` decision-quality candidate labels (Fix 3).  The pooled
#: mixes, the ``native``/``pi`` baselines (de-duplicated, not averaged), the
#: excluded-in-training ``damping`` arm, and every trajectory member separately.
E2_DQ_CANDIDATE_ARMS = (("native", "pi", "fixed_mix", "switching_mix", "wall",
                         "damping") + TRAJECTORY_ARMS)

#: Any row token (raw label ``source``, packed ``policy``/``arm_kind`` or a
#: member/group arm name) -> its E2 decision-quality arm type.  Unknown tokens
#: pass through unchanged and are dropped by the caller's allowed-arm filter.
_ARM_TYPE_ALIASES = {
    "native": "native", "native_fleet": "native", "native_base": "native",
    "pi": "pi", "pi_fleet": "pi",
    "fixed": "fixed_mix", "fixed_mix": "fixed_mix", "fixed1": "fixed_mix",
    "fixed2": "fixed_mix",
    "switching": "switching_mix", "switching_mix": "switching_mix",
    "switching1": "switching_mix", "switching2": "switching_mix",
    "switching3": "switching_mix",
    "wall": "wall", "w-lo-early": "wall", "w-lo-late": "wall",
    "w-mid-early": "wall", "w-mid-late": "wall",
    "trajectory": "trajectory", "arm0": "trajectory", "arm1": "trajectory",
    "arm2": "trajectory", "arm3": "trajectory",
    "damping": "damping", "damping_fleet": "damping", "base": "damping",
}


def arm_type_of(value) -> str:
    """Canonical E2 decision-quality arm *type* for a raw row token."""
    token = str(value or "").strip().lower()
    return _ARM_TYPE_ALIASES.get(token, token)


class ArmContrastRefusal(RuntimeError):
    """An arm-contrast input is malformed or conflicts with a sibling."""


def group_contrast(group: str) -> str:
    return f"{group}-native"


def member_contrast(arm: str) -> str:
    return f"{arm}-native"


#: Every pooled contrast name produced by this module.
GROUP_CONTRASTS = tuple(group_contrast(g) for g, _ in ARM_GROUPS)
#: Every per-member contrast name produced by this module.
MEMBER_CONTRASTS = tuple(member_contrast(a) for a in ALL_ARMS)


def member_contrast_specs() -> dict:
    """``{member contrast: (treatment arm, "native")}`` for the metric loop."""
    return {member_contrast(arm): (arm, NATIVE) for arm in ALL_ARMS}


def contrast_members(name: str) -> tuple:
    """The member arms a contrast pools over (``("native", ...)`` if unknown)."""
    for group, arms in ARM_GROUPS:
        if name == group_contrast(group):
            return tuple(arms)
    for group, arms in ARM_GROUPS:
        for arm in arms:
            if name == member_contrast(arm):
                return (arm,)
    raise ArmContrastRefusal(f"unknown arm contrast {name!r}")


def _field(record, name, default=None):
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _hashable(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    return str(value)


def _index_rows(rows, *, sources, taus, require, label):
    """``{(world, t0, tau, source): delay}`` for the requested sources."""
    tau_set = {round(float(t), 6) for t in taus}
    out: dict = {}
    for rec in rows:
        source = _hashable(_field(rec, "source", ""))
        if source not in sources:
            continue
        tau = _field(rec, "tau", None)
        if tau is None:
            continue
        tau = round(float(tau), 6)
        if tau not in tau_set:
            continue
        if float(_field(rec, "tau_available", 0.0) or 0.0) <= 0.0:
            continue
        t0 = _field(rec, "t0_s", None)
        if t0 is None:
            continue
        t0 = round(float(t0), 6)
        if round(float(_field(rec, "t", t0) or t0), 6) != t0:
            continue
        world = _hashable(_field(rec, "world", ""))
        if not world:
            continue
        key = (world, t0, tau, source) if require == "source" else (world, t0, tau)
        if key in out:
            raise ArmContrastRefusal(
                f"duplicate {label} TRAIN label row for {key} (fail closed)")
        out[key] = float(_field(rec, "total_delay_ref_s", float("nan")))
    return out


def build_arm_cells(treatment_rows, reference_rows, *,
                    taus=TAUS_S, levels=QUANTILE_LEVELS,
                    groups=ARM_GROUPS) -> tuple:
    """Per-contrast empirical paired-dV cells and the items used.

    ``treatment_rows`` carry the member arms; ``reference_rows`` carry the
    ``native`` reference.  Returns ``(cells, items_by_contrast)`` where
    ``cells[name][f"{tau:g}"]`` is ``{n_pairs, n_worlds, quantiles}``.
    """
    lv = np.asarray(levels, dtype=float)
    if lv.shape != (N_QUANTILES,):
        raise ArmContrastRefusal(
            f"levels must be the {N_QUANTILES} registered quantiles")
    treatments = _index_rows(
        treatment_rows, sources={a for _g, arms in groups for a in arms},
        taus=taus, require="source", label="treatment")
    references = _index_rows(
        reference_rows, sources={NATIVE}, taus=taus, require="reference",
        label="reference")

    cells: dict = {}
    items_by_contrast: dict = {}
    for group, arms in groups:
        pooled: dict = {}
        used: dict = {}
        for arm in arms:
            member_name = member_contrast(arm)
            member_pairs: dict = {}
            for (world, t0, tau, source), y_a in treatments.items():
                if source != arm:
                    continue
                key_ref = (world, t0, tau)
                y_b = references.get(key_ref)
                if y_b is None:
                    continue
                member_pairs.setdefault(tau, []).append(
                    (world, float(y_a) - float(y_b)))
                used[(world, source, t0)] = "treatment"
                used[(world, NATIVE, t0)] = "reference"
            if not member_pairs:
                continue
            cells[member_name] = {
                f"{tau:g}": _summarise(member_pairs[tau])
                for tau in sorted(member_pairs)}
            for tau, pairs in member_pairs.items():
                pooled.setdefault(tau, []).extend(pairs)
        if not pooled:
            continue
        cells[group_contrast(group)] = {
            f"{tau:g}": _summarise(pooled[tau]) for tau in sorted(pooled)}
        items_by_contrast[group_contrast(group)] = [
            dict(world=w, source=s, t0_s=float(t0), role=role)
            for (w, s, t0), role in sorted(used.items())]
    return cells, items_by_contrast


def _summarise(pairs, *, levels=QUANTILE_LEVELS) -> dict:
    dv = np.asarray([d for _w, d in pairs], dtype=float)
    lv = np.asarray(levels, dtype=float)
    return dict(
        n_pairs=int(dv.size),
        n_worlds=int(len({w for w, _d in pairs})),
        quantiles=[float(x) for x in np.quantile(dv, lv)])


def e2_metadata(*, input_sha256s: dict | None = None,
                contrasts: tuple | None = None) -> dict:
    """The additive E2 block appended to a climatology document."""
    return dict(
        schema_version="wm_n1_train_climatology_v2/1",
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        reference_policy=NATIVE,
        arm_groups={g: list(a) for g, a in ARM_GROUPS},
        group_contrasts=list(GROUP_CONTRASTS),
        member_contrasts=list(MEMBER_CONTRASTS),
        input_sha256s=dict(input_sha256s or {}),
        contrasts=sorted(contrasts or tuple(
            list(GROUP_CONTRASTS) + list(MEMBER_CONTRASTS))))


__all__ = [
    "SCHEMA", "NATIVE", "TAUS_S", "ARM_GROUPS", "GROUP_MEMBERS",
    "MEMBER_TO_GROUP", "ALL_ARMS", "E2_ALLOWED_ARMS", "E2_ARM_TYPES",
    "TRAJECTORY_ARMS", "E2_DQ_POOL_ARMS", "E2_DQ_CANDIDATE_ARMS",
    "arm_type_of", "GROUP_CONTRASTS",
    "MEMBER_CONTRASTS", "ArmContrastRefusal", "group_contrast",
    "member_contrast", "member_contrast_specs", "contrast_members",
    "build_arm_cells", "e2_metadata",
]
