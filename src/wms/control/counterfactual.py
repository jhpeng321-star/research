"""Counterfactual difference rewards: cooperation as the main training signal.

Per selected start state, every variant is replayed with common random numbers (same start,
same WM members, same policy noise per (start, control step, slot)):

* ``base``        B      every CAV native (``native_cav``: simple IDM CAV or hold speed), no lane changes;
* ``individual``  D_i  = J(only CAV i acts) - B;
* ``group``       D_G  = J(close group G acts) - B (G = a contiguous window of <= ``group_max_size``
                         members of a connected component of CAVs chained within ``group_radius_m``,
                         size >= ``group_min_size``; groups of one start are disjoint);
* ``flip-one``    D_G\\i = J(G without i acts) - B for ``n_flip`` sampled members i of each group
                         (for |G| = 2 this IS the other member's individual row and is reused);
* ``all``         D_all = J(all CAVs act) - B.

Derived per group: cooperation gain C_G = D_G - sum_{i in G} D_i; marginal-in-team
M_i = D_G - D_G\\i; interaction I_i = M_i - D_i (> 0: CAV i is worth more together than alone).

Actor objective (maximised; every term a differentiable lambda-return difference through the WM,
the lane head trained by REINFORCE with the same per-row coefficients):

    w_all * D_all + w_group * sum_G D_G + w_ind * sum_i D_i + w_coop * sum_G C_G + w_team * sum_G sum_i M_i

averaged over starts.  As a linear combination of row returns, a row's coefficient is
``all: w_all``, ``group: w_group + w_coop + n_flip(G) * w_team``, ``individual i: w_ind - w_coop *
[i in a group]``, ``flip G\\i: -w_team``.  ANTI-GAMING GUARD: C_G rewards making solo behaviour
worse, so ``w_coop <= w_ind`` is enforced at every curriculum knot (hence everywhere, the schedule is
piecewise linear) and groups of one start are disjoint, so every individual benefit keeps a net
non-negative weight ``w_ind - w_coop``.  For |G| = 2, G\\i is the other member alone (its
individual row is reused, no extra rollout) and M_i = D_G - D_j would put ``-w_team`` on a solo
benefit, so the team term enters the objective for groups of >= 3 only (M_i / I_i are logged for
all groups); :func:`row_coefficients` re-checks at run time that no individual row has a negative
coefficient.  The flip-one rows of larger groups carry ``-w_team`` (the marginal-contribution form
of difference rewards): a residual incentive to make the TEAM G\\i worse, bounded by ``w_team``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional

import torch
from torch import Tensor

from wms.core.constants import DEFAULT_LENGTH_M

from .contract import ACCEL_MIN_MPS2, MERGE_S1_M, NATIVE_CAV_ACCEL_MAX_MPS2, SPEED_LIMIT_MPS

KIND_ALL, KIND_BASE, KIND_INDIVIDUAL, KIND_GROUP, KIND_FLIP = 0, 1, 2, 3, 4
KIND_NAMES = ("all", "base", "individual", "group", "flip")
WEIGHT_NAMES = ("individual", "group", "coop", "team", "all")


@dataclass
class CounterfactualConfig:
    enabled: bool = False
    n_individual: int = 2            # individual rows per start (at least; group members are added)
    n_group: int = 2                 # disjoint group rows per start
    n_flip: int = 2                  # flip-one rows (G without i) per group
    # IDM a_max of the 'native' CAV rows = SUMO-native CAV accel (ACCELCAP_NOTES.md; was the
    # policy cap 1.5 before 2026-10-13 - set 1.5 to reproduce older runs).  Not the policy cap.
    native_accel_max_mps2: float = NATIVE_CAV_ACCEL_MAX_MPS2
    group_radius_m: float = 150.0    # |ds| chaining radius (lanes ignored: walls span lanes)
    group_min_size: int = 2
    group_max_size: int = 6
    cover_group_members: bool = True  # individual rows for every member of a sampled group
    max_starts: int = 8              # counterfactual rows for at most this many starts per batch
    native_cav: str = "idm"          # "idm" (simple IDM CAV) or "hold" (a = 0); both safety-filtered
    # Group-rich start sampling (fraction of each batch whose start satisfies `group_rich`).
    group_state_fraction: float = 0.6
    group_zone_s_m: list = field(default_factory=lambda: [-800.0, MERGE_S1_M])
    congestion_speed_mps: float = 15.0
    max_group_tries: int = 20        # sampler draws per group-rich slot before falling back
    curriculum_updates: list = field(default_factory=lambda: [0, 1500])
    w_individual: list = field(default_factory=lambda: [1.0, 0.5])
    w_group: list = field(default_factory=lambda: [0.5, 1.0])
    w_coop: list = field(default_factory=lambda: [0.0, 0.5])
    w_team: list = field(default_factory=lambda: [0.0, 0.5])
    w_all: list = field(default_factory=lambda: [0.25, 1.0])

    def __post_init__(self):
        for name in ("n_individual", "n_group", "n_flip", "group_min_size", "group_max_size",
                     "max_starts", "max_group_tries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.group_min_size < 2 or self.group_max_size < self.group_min_size:
            raise ValueError("need 2 <= group_min_size <= group_max_size")
        for name in ("group_radius_m", "congestion_speed_mps"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= float(self.group_state_fraction) <= 1.0:
            raise ValueError("group_state_fraction must lie in [0, 1]")
        zone = [float(x) for x in self.group_zone_s_m]
        if len(zone) != 2 or not zone[0] < zone[1]:
            raise ValueError("group_zone_s_m must be [lo, hi] with lo < hi")
        self.group_zone_s_m = zone
        if self.native_cav not in ("idm", "hold"):
            raise ValueError("native_cav must be 'idm' or 'hold'")
        names = ("curriculum_updates", "w_individual", "w_group", "w_coop", "w_team", "w_all")
        knots = [list(map(float, getattr(self, n))) for n in names]
        if len({len(k) for k in knots}) != 1 or not knots[0]:
            raise ValueError("curriculum lists must be nonempty and of equal length")
        if any(b <= a for a, b in zip(knots[0], knots[0][1:])):
            raise ValueError("curriculum_updates must be strictly increasing")
        if any(not math.isfinite(x) or x < 0 for k in knots[1:] for x in k):
            raise ValueError("curriculum weights must be finite and nonnegative")
        for name, value in zip(names, knots):
            setattr(self, name, value)
        # Anti-gaming guard: C_G = D_G - sum D_i rewards worse solo behaviour; keep every
        # individual benefit's net weight w_ind - w_coop >= 0 (linear schedule: knots suffice).
        if any(c > i + 1e-12 for c, i in zip(self.w_coop, self.w_individual)):
            raise ValueError("anti-gaming guard: w_coop must not exceed w_individual at any knot")
        if any(self.w_coop) and not self.cover_group_members:
            raise ValueError("w_coop > 0 needs cover_group_members = true (C_G needs every D_i)")
        if any(self.w_team) and self.n_flip < 1:
            raise ValueError("w_team > 0 needs n_flip >= 1 (M_i needs the flip-one rows)")

    def weights(self, update: int) -> dict:
        """Piecewise-linear curriculum at ``update`` (held beyond the end knots).
        Disabled: only ``all = 1`` (the plain all-CAV Dreamer objective)."""
        if not self.enabled:
            return {"individual": 0.0, "group": 0.0, "coop": 0.0, "team": 0.0, "all": 1.0}
        xs = self.curriculum_updates

        def interp(ys):
            if update <= xs[0]:
                return ys[0]
            for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
                if update <= x1:
                    return y0 + (y1 - y0) * (update - x0) / (x1 - x0)
            return ys[-1]

        return {"individual": interp(self.w_individual), "group": interp(self.w_group),
                "coop": interp(self.w_coop), "team": interp(self.w_team), "all": interp(self.w_all)}

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ native CAV
def idm_native_accel(state: Mapping[str, Tensor], *, v0: float = SPEED_LIMIT_MPS,
                     a_max: float = NATIVE_CAV_ACCEL_MAX_MPS2, b: float = 2.0, s0: float = 2.0,
                     time_headway_s: float = 1.2, delta: float = 4.0) -> Tensor:
    """Simple IDM CAV acceleration [B,K] (same-lane nearest leader), clamped to [-3, a_max].
    A designed baseline used ONLY as the counterfactual 'native' reference, never as an action.
    ``a_max`` models SUMO-native CAV behaviour (CAVs keep their driver vType; default the
    normal_car accel 2.6), NOT the policy's action cap."""
    present = state["present"].bool()
    lane = state["lane"].long()
    valid = present & (lane >= 0)
    s = torch.where(valid, state["s_m"], torch.zeros_like(state["s_m"]))
    v = torch.where(valid, state["v_mps"], torch.zeros_like(s)).clamp_min(0.0)
    length = state.get("length_m", torch.full_like(s, DEFAULT_LENGTH_M))
    length = torch.where(valid, length, torch.zeros_like(length))
    k = s.shape[-1]
    ds = s[:, None, :] - s[:, :, None]
    other = ~torch.eye(k, dtype=torch.bool, device=s.device)[None]
    ahead = valid[:, :, None] & valid[:, None, :] & (lane[:, :, None] == lane[:, None, :]) \
        & (ds >= 0) & other
    distance, index = torch.where(ahead, ds, torch.full_like(ds, math.inf)).min(-1)
    has = torch.isfinite(distance)
    gap = (torch.where(has, distance, torch.zeros_like(distance)) - length.gather(-1, index)).clamp_min(0.1)
    dv = v - v.gather(-1, index)
    s_star = s0 + (v * time_headway_s + v * dv / (2.0 * math.sqrt(a_max * b))).clamp_min(0.0)
    free = 1.0 - (v / v0).pow(delta)
    accel = a_max * (free - torch.where(has, (s_star / gap).square(), torch.zeros_like(gap)))
    accel = accel.clamp(ACCEL_MIN_MPS2, a_max)
    return torch.where(valid, accel, torch.zeros_like(accel))


def hold_native_accel(state: Mapping[str, Tensor]) -> Tensor:
    return torch.zeros_like(state["s_m"])


def native_fn(name: str, a_max: float = None):
    if name != "idm":
        return hold_native_accel
    if a_max is None:
        return idm_native_accel
    a_max = float(a_max)

    def idm(state):
        return idm_native_accel(state, a_max=a_max)
    return idm


# ------------------------------------------------------------------ groups / start selection
def acting_cavs(state: Mapping[str, Tensor]) -> Tensor:
    return state["present"].bool() & state["is_cav"].bool() & (state["lane"] >= 0)


def close_groups(s: Tensor, cav: Tensor, radius_m: float, min_size: int = 2) -> list[list[int]]:
    """Connected components of CAV slots chained along s (|ds| <= radius), size >= min_size.
    ``s``/``cav`` are [K] for one start state; returns slot lists sorted by position."""
    slots = torch.nonzero(cav, as_tuple=False).flatten().tolist()
    if not slots:
        return []
    order = sorted(slots, key=lambda i: (float(s[i]), i))
    groups, current = [], [order[0]]
    for prev, nxt in zip(order, order[1:]):
        if float(s[nxt]) - float(s[prev]) <= radius_m:
            current.append(nxt)
        else:
            groups.append(current)
            current = [nxt]
    groups.append(current)
    return [g for g in groups if len(g) >= min_size]


def group_rich(state: Mapping[str, Tensor], cfg: CounterfactualConfig) -> bool:
    """Start state worth a cooperation check: >= 2 CAVs chained within the radius inside
    ``group_zone_s_m`` (near/upstream of the merge) AND congestion seen by the RSU field (a queue
    cell or an occupied cell with mean speed below ``congestion_speed_mps``).  ``state`` [1,K]."""
    from .rsu_field import build_rsu_field

    s = state["s_m"].reshape(-1)
    cav = acting_cavs({k: v.reshape(1, -1) for k, v in state.items()
                       if torch.is_tensor(v) and v.numel() == s.numel()}).reshape(-1)
    lo, hi = cfg.group_zone_s_m
    zone = (s >= lo) & (s <= hi)
    if not close_groups(s, cav & zone, cfg.group_radius_m, 2):
        return False
    field = build_rsu_field({k: v.reshape(1, -1) for k, v in state.items()
                             if torch.is_tensor(v) and v.numel() == s.numel()})[0]
    occupied = field[:, 4] > 0
    congested = (field[:, 3] > 0) | (occupied & (field[:, 1] < cfg.congestion_speed_mps))
    return bool(congested.any())


def _randint(n: int, generator=None) -> int:
    return int(torch.randint(n, (), generator=generator))


@dataclass
class Variants:
    """Expanded imagination batch: row r replays start ``row_start[r]`` with ``act_mask[r]``."""
    row_start: Tensor            # [N] long
    act_mask: Tensor             # [N,K] bool (all CAVs act on KIND_ALL rows)
    row_kind: Tensor             # [N] long (KIND_*)
    n_starts: int                # B (rows 0..B-1 are the all-CAV rows of starts 0..B-1)
    base_row: dict               # start -> row index of its base row
    individual_rows: dict        # (start, slot) -> row index
    groups: list                 # [{"row", "start", "members", "flips": {slot: row}}]

    @property
    def n_rows(self) -> int:
        return int(self.row_start.numel())


def build_variants(state: Mapping[str, Tensor], cfg: CounterfactualConfig, generator=None) -> Variants:
    """All-CAV rows for every start, then (for up to ``max_starts`` starts that have CAVs) base,
    individual, group and flip-one rows.  Groups of one start are disjoint (distinct components).
    Uses ``generator`` (or the global torch RNG)."""
    cav = acting_cavs(state)
    b, k = cav.shape
    device = cav.device
    starts, masks, kinds = list(range(b)), [cav[i].clone() for i in range(b)], [KIND_ALL] * b
    base_row, individual_rows, groups = {}, {}, []

    def add(start, slots, kind):
        mask = torch.zeros(k, dtype=torch.bool, device=device)
        if slots:
            mask[list(slots)] = True
        starts.append(start)
        masks.append(mask)
        kinds.append(kind)
        return len(starts) - 1

    used = 0
    for start in range(b):
        if not cfg.enabled or used >= cfg.max_starts or not bool(cav[start].any()):
            continue
        used += 1
        base_row[start] = add(start, [], KIND_BASE)
        components = close_groups(state["s_m"][start], cav[start], cfg.group_radius_m,
                                  cfg.group_min_size)
        chosen = []
        while components and len(chosen) < cfg.n_group:
            component = components.pop(_randint(len(components), generator))
            size = min(len(component), cfg.group_max_size)
            offset = _randint(len(component) - size + 1, generator)
            chosen.append(component[offset:offset + size])
        singles = []
        if cfg.cover_group_members:
            for members in chosen:
                singles.extend(members)
        others = [i for i in torch.nonzero(cav[start]).flatten().tolist() if i not in singles]
        while len(singles) < cfg.n_individual and others:
            singles.append(others.pop(_randint(len(others), generator)))
        for slot in singles:
            individual_rows[(start, slot)] = add(start, [slot], KIND_INDIVIDUAL)
        for members in chosen:
            row = add(start, members, KIND_GROUP)
            pool = list(members)
            flips = {}
            for _ in range(min(cfg.n_flip, len(pool))):
                slot = pool.pop(_randint(len(pool), generator))
                rest = [m for m in members if m != slot]
                if len(rest) == 1 and (start, rest[0]) in individual_rows:
                    flips[slot] = individual_rows[(start, rest[0])]     # G\i = the other member
                else:
                    flips[slot] = add(start, rest, KIND_FLIP)
            groups.append({"row": row, "start": start, "members": list(members), "flips": flips})
    return Variants(torch.tensor(starts, dtype=torch.long, device=device), torch.stack(masks),
                    torch.tensor(kinds, dtype=torch.long, device=device), b, base_row,
                    individual_rows, groups)


def expand_rows(value, index: Tensor, batch_size: int):
    """Index every tensor whose leading dim is the start batch (recursively in dicts)."""
    if torch.is_tensor(value) and value.ndim and value.shape[0] == batch_size:
        return value.index_select(0, index.to(value.device))
    if isinstance(value, dict):
        return {key: expand_rows(item, index, batch_size) for key, item in value.items()}
    return value


# ------------------------------------------------------------------ objective / decomposition
def row_coefficients(variants: Optional[Variants], weights: Mapping[str, float], n_rows: int,
                     n_starts: int, *, device=None, dtype=torch.float32) -> Tensor:
    """Coefficient of each row's return in the (maximised) actor objective, averaged over starts."""
    c = torch.zeros(n_rows, device=device, dtype=dtype)
    c[:n_starts] = weights["all"] / max(n_starts, 1)
    if variants is None or not variants.base_row:
        return c
    per_start = 1.0 / len(variants.base_row)
    for row in variants.individual_rows.values():
        c[row] += weights["individual"] * per_start
    for group in variants.groups:
        c[group["row"]] += (weights["group"] + weights["coop"]) * per_start
        for member in group["members"]:
            c[variants.individual_rows[(group["start"], member)]] -= weights["coop"] * per_start
        if len(group["members"]) < 3:
            # |G| = 2: G\i is the other member ALONE, so M_i would put -w_team on a solo benefit;
            # the team term is applied to groups of >= 3 only (M_i is still logged for pairs).
            continue
        for flip_row in group["flips"].values():
            c[group["row"]] += weights["team"] * per_start
            c[flip_row] -= weights["team"] * per_start
    solo = list(variants.individual_rows.values())
    if solo and float(c[solo].min()) < -1e-9:
        raise RuntimeError("anti-gaming guard violated: an individual benefit has negative weight")
    return c


def baseline_index(variants: Optional[Variants], n_rows: int, device=None) -> Tensor:
    """Row index of the base row of each row's start (itself for starts without a base row)."""
    if variants is None:
        return torch.arange(n_rows, device=device)
    return torch.tensor([variants.base_row.get(int(start), row) for row, start in
                         enumerate(variants.row_start.tolist())], device=device)


def decompose(j: Tensor, variants: Variants) -> dict:
    """B, D_i, D_G, D_all, C_G, M_i, I_i from per-row returns ``j`` [N] (one value per row)."""
    j = [float(x) for x in j.detach().reshape(-1).tolist()]
    base = {s: j[row] for s, row in variants.base_row.items()}
    d_ind = {key: j[row] - base[key[0]] for key, row in variants.individual_rows.items()}
    out = {"B": list(base.values()), "D_all": [j[s] - base[s] for s in base],
           "D_i": list(d_ind.values()), "groups": [], "M_i": [], "I_i": []}
    for group in variants.groups:
        s = group["start"]
        d_group = j[group["row"]] - base[s]
        members = group["members"]
        covered = all((s, m) in d_ind for m in members)
        record = {"start": s, "size": len(members), "D_G": d_group,
                  "C_G": d_group - sum(d_ind[(s, m)] for m in members) if covered else None}
        for member, flip_row in group["flips"].items():
            marginal = d_group - (j[flip_row] - base[s])
            out["M_i"].append(marginal)
            if (s, member) in d_ind:
                out["I_i"].append(marginal - d_ind[(s, member)])
        out["groups"].append(record)
    out["D_G"] = [g["D_G"] for g in out["groups"]]
    out["C_G"] = [g["C_G"] for g in out["groups"] if g["C_G"] is not None]
    return out


def mean_or_none(values) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None
