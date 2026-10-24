"""R13T tail matching: per-cell quantile score of the rollout's HDV acceleration and |jerk| on ALL rows.

BRAKE_DIAG (2026-10-24): the learned temporal emission adds zero-mean, heavy-tailed acceleration noise in
closed loop (lane-1 HDV rows with a < -3: 4.0 % WM vs 0.18 % recorded).  The persistence terms of
:mod:`.brake_persist` supervise recorded onsets only, so spurious onsets are free, and the position/speed
terms barely see a zero-mean pulse.  This term scores the rollout's own (closed-loop) distribution.

For every chunk, every rollout row ``(b, r, k)`` with the slot present at step start and end, an HDV, and
its step-start lane in ``lanes`` (default lane 1) gets a cell id from its PREDICTED step-start state with
the brake-persistence convention (:func:`.brake_persist.leader_cells`: gap x closing x leader decel, 4 x 4 x
4).  Its predicted acceleration is the speed difference ``(v_{r+1} - v_r) / dt`` (the definition of the
braking terms and of the BRAKE_DIAG metric); |jerk| is the difference of consecutive accelerations of a
slot present over three samples.  Per cell ``c`` with ``n_c`` rows the predicted empirical quantiles
``q_hat(u)`` (``torch.quantile``, linear interpolation, differentiable in the rows at those ranks) at the
table's levels ``u`` are scored with the pinball loss against the recorded distribution of the same cell,
represented by its quantiles ``y_m`` on the same grid (equally weighted pseudo-samples):

    S_c = mean_u mean_m rho_u(y_m - q_hat(u)) - mean_u mean_m rho_u(y_m - y(u)),
    rho_u(d) = max(u d, (u - 1) d).

The pinball loss is a proper score of a quantile: ``d E_y rho_u(y - q) / dq = F_rec(q) - u``, zero exactly
at the recorded quantile, bounded (robust to the heavy tails it targets), and the mean over a uniform level
grid is the quantile decomposition of the CRPS, i.e. it matches the whole cell distribution, tails
included.  The subtracted constant makes ``S_c = 0`` at a perfect match (gradient unchanged).  A level is
scored only when ``n_c * min(u, 1 - u) >= 1`` (at least one predicted row beyond it; otherwise the
"quantile" is a single extreme row); the term is the row-weighted mean of ``S_c`` over cells.

Cells, levels and recorded quantiles come from a table written by ``tools/r13t_tail_table.py`` from TRAIN
episodes only (``split_roles == ["train"]`` is required unless ``require_train_split`` is false, which only
the offline tools/tests use).  A table cell with fewer than ``1 / min(u, 1 - u)`` recorded rows (the same
support rule) falls back to the pooled lane distribution.  Targets and cell ids are detached; only the
predicted speeds carry gradient (pathwise adjoint through the rollout).  Default OFF (weights 0).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from .base import LossContext, LossTerm, TermValue
from .brake_persist import _N_CELLS, leader_cells

TABLE_SCHEMA = "r13t_tail_table/1"
VARIABLES = ("accel", "abs_jerk")


@dataclass(frozen=True)
class AccelTailConfig:
    """``weight`` scores the acceleration, ``jerk_weight`` the |jerk| (both 0 = off)."""

    weight: float = 0.0
    jerk_weight: float = 0.0
    table: Optional[str] = None
    lanes: tuple[int, ...] = (1,)
    require_train_split: bool = True

    def __post_init__(self) -> None:
        for name in ("weight", "jerk_weight"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"accel_tail.{name} must be finite and >= 0 (0 = off)")
        if self.enabled and not self.table:
            raise ValueError("accel_tail needs a recorded quantile table (loss.accel_tail.table)")
        if not self.lanes or any(int(lane) not in (0, 1, 2) for lane in self.lanes):
            raise ValueError("accel_tail.lanes must be a non-empty subset of {0, 1, 2}")

    @property
    def enabled(self) -> bool:
        return float(self.weight) > 0.0 or float(self.jerk_weight) > 0.0


def resolve_table(path: Optional[str]) -> Optional[str]:
    """Absolute path of the table; a relative path resolves against the repository root."""
    if not path:
        return None
    p = Path(str(path))
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[4] / p
    return str(p)


def support_ok(n: int, level: float) -> bool:
    """At least one row beyond the level: ``n * min(u, 1 - u) >= 1``."""
    return float(n) * min(float(level), 1.0 - float(level)) >= 1.0 - 1e-9


@lru_cache(maxsize=8)
def _load_table(path: str, require_train_split: bool) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != TABLE_SCHEMA:
        raise ValueError(f"accel_tail table {path}: schema {data.get('schema')!r} != {TABLE_SCHEMA!r}")
    if require_train_split and list(data.get("split_roles") or []) != ["train"]:
        raise ValueError(f"accel_tail table {path}: built from split roles {data.get('split_roles')!r}; "
                         "the training term needs TRAIN-split episodes only")
    levels = [float(u) for u in data["levels"]]
    if not levels or sorted(levels) != levels or not all(0.0 < u < 1.0 for u in levels):
        raise ValueError("accel_tail table: levels must be increasing and inside (0, 1)")
    out = {"levels": torch.tensor(levels, dtype=torch.float64), "max_gap_m": float(data["max_gap_m"]),
           "dt_s": float(data["dt_s"]), "lanes": tuple(int(x) for x in data["lanes"])}
    for var in VARIABLES:
        block = data[var]
        pooled = block["pooled"]
        if not all(support_ok(int(pooled["n"]), u) for u in levels):
            raise ValueError(f"accel_tail table: pooled {var} has too few rows for its levels")
        q = torch.tensor(pooled["q"], dtype=torch.float64).expand(_N_CELLS, -1).clone()
        used = torch.zeros(_N_CELLS, dtype=torch.bool)
        for key, cell in block["cells"].items():
            c = int(key)
            if all(support_ok(int(cell["n"]), u) for u in levels):
                q[c] = torch.tensor(cell["q"], dtype=torch.float64)
                used[c] = True
        if q.shape[1] != len(levels):
            raise ValueError(f"accel_tail table: {var} quantile width != number of levels")
        out[var] = q
        out[var + "_own_cell"] = used
    return out


def _pinball_grid(pred_q: Tensor, rec_y: Tensor, levels: Tensor) -> Tensor:
    """``mean_u mean_m rho_u(y_m - q(u))`` for one cell (``pred_q`` ``[L]``, ``rec_y`` ``[M]``)."""
    d = rec_y.view(1, -1) - pred_q.view(-1, 1)                  # [L, M]
    u = levels.view(-1, 1)
    return torch.maximum(u * d, (u - 1.0) * d).mean()


def cell_scores(values: Tensor, cells: Tensor, table_q: Tensor, levels: Tensor) -> tuple[Tensor, int, int]:
    """Row-weighted mean quantile score over cells, number of scored rows and cells."""
    total = values.new_zeros(())
    rows = 0
    n_cells = 0
    if values.numel() == 0:
        return total, 0, 0
    for c in torch.unique(cells).tolist():
        v = values[cells == c]
        n = int(v.numel())
        keep = torch.tensor([support_ok(n, float(u)) for u in levels], dtype=torch.bool)
        if not bool(keep.any()):
            continue
        u = levels[keep].to(v.dtype)
        y = table_q[c].to(v.dtype)
        q_hat = torch.quantile(v, u)
        score = _pinball_grid(q_hat, y, u) - _pinball_grid(y[keep], y, u)
        total = total + n * score
        rows += n
        n_cells += 1
    return (total / max(rows, 1)), rows, n_cells


def _step_states(ctx: LossContext) -> dict[str, Tensor]:
    """Predicted state at every step START of the chunk (``[B, L, K]`` each)."""
    cur = ctx.current
    pred = ctx.pred
    s = torch.cat((cur["s_m"].unsqueeze(1), pred["positions"][:, :-1]), 1)
    v = torch.cat((cur["v_mps"].unsqueeze(1), pred["speeds"][:, :-1]), 1)
    lane_idx = torch.cat((cur["lane"].unsqueeze(1), pred["lanes"][:, :-1]), 1)
    present = torch.cat((cur["present"].bool().unsqueeze(1), pred["masks"][:, :-1].bool()), 1)
    a0 = cur.get("a_mps2")
    if a0 is None:
        a0 = torch.zeros_like(cur["v_mps"])
    a = torch.cat((a0.unsqueeze(1), pred["accel"][:, :-1]), 1)
    return {"s": s.detach(), "v": v.detach(), "lane": lane_idx, "present": present, "a": a.detach()}


@torch.no_grad()
def chunk_cells(ctx: LossContext, max_gap_m: float) -> tuple[Tensor, Tensor]:
    """Cell id and lane index of every chunk row from the predicted step-start state."""
    st = _step_states(ctx)
    b, length, k = st["s"].shape
    cells = torch.full((b, length, k), -1, dtype=torch.long, device=st["s"].device)
    for r in range(length):
        lane = st["lane"][:, r]
        on_road = (lane >= 0) & st["present"][:, r]
        onehot = torch.nn.functional.one_hot(lane.clamp(0, 2), 3).to(st["s"].dtype)
        cells[:, r] = leader_cells(st["s"][:, r], onehot, st["v"][:, r], st["a"][:, r], on_road,
                                   max_gap_m=max_gap_m)
    return cells, st["lane"]


def _chunk_rows(ctx: LossContext, lanes: tuple[int, ...], max_gap_m: float) -> dict:
    """Shared per-chunk inputs of the two tail terms (cached on the context)."""
    key = ("accel_tail", tuple(int(x) for x in lanes), float(max_gap_m))
    cached = ctx.cache.get(key)
    if cached is not None:
        return cached
    c0, c1 = int(ctx.c0), int(ctx.c1)
    dt_s = float(ctx.targets.dt_s)
    speeds = torch.cat((ctx.current["v_mps"].unsqueeze(1), ctx.pred["speeds"]), 1)
    accel = (speeds[:, 1:] - speeds[:, :-1]) / dt_s                       # [B, L, K]
    cells, lane = chunk_cells(ctx, max_gap_m)
    present_start = torch.cat((ctx.current["present"].bool().unsqueeze(1),
                               ctx.pred["masks"][:, :-1].bool()), 1)
    present_end = ctx.pred["masks"].bool()
    hdv = ~ctx.targets.is_cav[:, c0:c1].bool()
    in_lane = torch.zeros_like(present_end)
    for target_lane in lanes:
        in_lane |= lane == int(target_lane)
    rows = present_start & present_end & hdv & in_lane
    out = {"accel": accel, "cells": cells, "rows": rows, "dt_s": dt_s}
    ctx.cache[key] = out
    return out


class AccelTailLoss(LossTerm):
    """Per-cell pinball quantile score of the rollout's HDV acceleration on all rows (``weight``)."""

    name = "accel_tail"
    variable = "accel"

    def __init__(self, config: AccelTailConfig) -> None:
        self.config = config
        self.last_stats: dict[str, float] = {}

    @property
    def weight(self) -> float:
        return float(self.config.weight)

    def _table(self) -> dict:
        return _load_table(str(self.config.table), bool(self.config.require_train_split))

    def _values(self, inputs: dict) -> tuple[Tensor, Tensor]:
        rows = inputs["rows"]
        return inputs["accel"][rows], inputs["cells"][rows]

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        table = self._table()
        if abs(float(ctx.targets.dt_s) - table["dt_s"]) > 1e-9:
            raise ValueError(f"accel_tail table dt {table['dt_s']} != rollout dt {ctx.targets.dt_s}")
        inputs = _chunk_rows(ctx, tuple(self.config.lanes), table["max_gap_m"])
        values, cells = self._values(inputs)
        score, rows, n_cells = cell_scores(values, cells, table[self.variable], table["levels"])
        self.last_stats = {"rows": rows, "cells": n_cells, "score": float(score.detach())}
        return TermValue(self.weight * score, log=score.detach(), valid_count=rows)


class JerkTailLoss(AccelTailLoss):
    """The same score on |jerk| = |a_r - a_{r-1}| / dt of slots present over three samples (``jerk_weight``)."""

    name = "jerk_tail"
    variable = "abs_jerk"

    @property
    def weight(self) -> float:
        return float(self.config.jerk_weight)

    def _values(self, inputs: dict) -> tuple[Tensor, Tensor]:
        accel, rows = inputs["accel"], inputs["rows"]
        jerk = (accel[:, 1:] - accel[:, :-1]).abs() / inputs["dt_s"]      # at r = 1..L-1
        keep = rows[:, 1:] & rows[:, :-1]
        return jerk[keep], inputs["cells"][:, 1:][keep]


def accel_tail_terms(config: AccelTailConfig) -> list[LossTerm]:
    """The enabled tail terms in a fixed order (empty when both weights are 0)."""
    terms: list[LossTerm] = []
    if float(config.weight) > 0.0:
        terms.append(AccelTailLoss(config))
    if float(config.jerk_weight) > 0.0:
        terms.append(JerkTailLoss(config))
    return terms


__all__ = ["AccelTailConfig", "AccelTailLoss", "JerkTailLoss", "TABLE_SCHEMA", "resolve_table", "accel_tail_terms", "cell_scores",
           "chunk_cells", "support_ok"]
