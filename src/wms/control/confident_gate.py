"""S-SUMOCREDIT (1): calibrated confident-deployment gate (owner rule 2026-10-18).

A CAV unit (the whole fleet, a close CAV group, or one CAV) executes the learned policy's
action only when the CALIBRATED probability that the plan beats the reference,
``P(J_plan < J_reference | RSU belief) >= 0.8``, otherwise it falls back to the reference
(``hold``: safety-filtered direct a = 0, stay in lane - the executable direct fallback of the
deployment check; or ``native``: released to SUMO's own driver model).  0.8 is the owner's
threshold and is the only number of the rule; it is not a tunable parameter here.

Pieces:

* :class:`ArmImagination` - paired WM-S rollouts of ``REF`` (every CAV on the reference) and
  ``PLAN_u`` (the units' members on the deterministic policy, everyone else on the reference)
  from ONE (belief or oracle) state.  Common random numbers: the same standard-normal driver
  draws for every arm and ensemble member, the same lane uniforms per sample index, and a
  deterministic policy.  ``J`` = imagined road delay (``team_reward`` ``delay``: speed-deficit
  vehicle-seconds, the SUMO ledger's on-road term) summed over the horizon.
* raw score of unit u = mean over (WM member, driver sample) of ``J_REF - J_PLAN_u``
  (positive = the plan is predicted better).  The raw score is NOT a probability.
* :class:`ReliabilityTable` - the calibration map raw -> ``P(SUMO J_plan < SUMO J_ref)``,
  fitted on TRAIN paired SUMO branches (``policy_pairs`` gate-calibration records: the same
  saved state, same SUMO RNG, the plan arm and the reference arm over the gate horizon; SUMO
  J = authoritative ledger speed-deficit + source-waiting vehicle-seconds).  Fit = isotonic
  (pool-adjacent-violators) blocks of the Beta(1,1)-posterior win rate; a raw score between
  two blocks maps to the LOWER block, above the data to the top block.  No probability is
  hand-set: the uniform prior is the only modelling choice and it shrinks small blocks toward
  0.5 (so a 1/1 block reads 0.667 < 0.8 and keeps the gate closed).
* :class:`ConfidentGate` - one decision per unit per decision interval; logs the open rate.
* :class:`GatedSumoPolicy` - SUMO executor ``(env, state) -> (accel, lanes)``: the plan for the
  open units' CAVs, the reference for every other CAV.  The ungated policy stays available as
  its own arm/controller (paper comparison).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .contract import PHYSICS_DT_S, RewardWeights

#: owner rule 2026-10-18: switch only with calibrated probability >= 0.8
GATE_PROBABILITY = 0.8
UNITS = ("fleet", "group", "cav")
REFERENCES = ("hold", "native")
TABLE_SCHEMA = "c4_gate_reliability/1"
RECORD_SCHEMA = "c4_gate_calibration_record/1"
EVAL_SEEDS = (964000, 964199)


def _seed_of(*parts) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:15], 16)


# ------------------------------------------------------------------ units
def acting_slots(state: Mapping[str, Tensor]) -> list[int]:
    """Present CAV slots with a WM lane (a batch-1 state ``[1,K]``)."""
    mask = state["present"][0].bool() & state["is_cav"][0].bool() & (state["lane"][0] >= 0)
    return torch.nonzero(mask).flatten().tolist()


def gate_units(state: Mapping[str, Tensor], unit: str) -> list[list[int]]:
    """Decision units as slot lists: ``fleet`` (all acting CAVs), ``group`` (single-linkage chains
    along s within ``traffic_shepherd.c4.branching.GROUP_LINK_M``, the SUMO branching group rule)
    or ``cav`` (one unit per CAV).  Ordered downstream first."""
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}")
    slots = acting_slots(state)
    if not slots:
        return []
    s = state["s_m"][0]
    if unit == "fleet":
        return [sorted(slots, key=lambda i: (-float(s[i]), i))]
    if unit == "cav":
        return [[i] for i in sorted(slots, key=lambda i: (-float(s[i]), i))]
    from traffic_shepherd.c4.branching import GROUP_LINK_M, chain_groups
    chains = chain_groups({i: float(s[i]) for i in slots}, link_m=GROUP_LINK_M)
    return [sorted(c, key=lambda i: (-float(s[i]), i))
            for c in sorted(chains, key=lambda c: -max(float(s[i]) for i in c))]


def reference_native(reference: str, head_path: Optional[str] = None) -> Callable:
    """Imagined behaviour of reference CAVs: ``hold`` = a 0 (the hold fallback) or ``native`` = the
    learned native-CAV head (``configs/native_cav_head_v5.json``; never the IDM twin)."""
    from .counterfactual import native_fn
    if reference not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}")
    return native_fn("hold" if reference == "hold" else "head", head_path=head_path)


# ------------------------------------------------------------------ imagination
class ArmImagination:
    """Paired arm rollouts ``J [members, arms, samples]`` (imagined road vehicle-seconds)."""

    def __init__(self, world_models: Sequence, policy, group_encoder, *, lane_rules=None,
                 reference: str = "hold", horizon_s: float = 20.0, n_driver_samples: int = 1,
                 driver_noise_scale: float = 0.0, embedding_noise_std: float = 0.0,
                 seed: int = 0, head_path: Optional[str] = None, controller_kwargs=None):
        if not world_models:
            raise ValueError("world_models must not be empty")
        if not (math.isfinite(horizon_s) and horizon_s > 0):
            raise ValueError("horizon_s must be positive")
        if isinstance(n_driver_samples, bool) or int(n_driver_samples) < 1:
            raise ValueError("n_driver_samples must be a positive integer")
        if int(n_driver_samples) > 1 and driver_noise_scale == 0 and embedding_noise_std == 0:
            raise ValueError("identical driver samples: use n_driver_samples=1 without driver noise")
        self.models = list(world_models)
        self.policy, self.group_encoder, self.lane_rules = policy, group_encoder, lane_rules
        self.reference, self.horizon_s = reference, float(horizon_s)
        self.native = reference_native(reference, head_path)
        self.n_samples = int(n_driver_samples)
        self.driver_noise_scale, self.embedding_noise_std = driver_noise_scale, embedding_noise_std
        self.seed = int(seed)
        self.controller_kwargs = dict(controller_kwargs or {"symmetric_exec_cap": True,
                                                            "mask_forced_merge": True})
        self.calls = 0

    def _actor(self):
        from .policy import policy_accel_bounds
        policy = self.policy

        def actor(obs, signal, **kwargs):
            return policy(obs, signal, **{**kwargs, "deterministic": True})
        actor.accel_max = policy_accel_bounds(policy)[1]
        return actor

    def _factory(self, member: Optional[Tensor], rows: int):
        from .bridge import make_cav_controller
        from .safety import safe_accel
        lane = {} if self.lane_rules is None else {"lane_rules": self.lane_rules}
        if member is None:
            return lambda: make_cav_controller(self._actor(), self.group_encoder, safe_accel,
                                               **lane, **self.controller_kwargs)
        act = member.bool()[None].expand(rows, -1).clone()
        return lambda: make_cav_controller(self._actor(), self.group_encoder, safe_accel, **lane,
                                           act_mask=act, native=self.native,
                                           **self.controller_kwargs)

    @torch.no_grad()
    def arms_J(self, state: Mapping[str, Tensor], members: Sequence[Optional[Tensor]], *,
               history=None, history_mask=None, driver_history=None, driver_history_mask=None,
               key="") -> Tensor:
        """``members[a]``: None = every CAV on the policy; bool ``[K]`` = only these CAVs on the
        policy (all False = the reference arm).  Returns float64 ``[M, A, S]``."""
        from .futures import arm_major_lane_uniform, sample_driver_encodings
        from .reward import team_reward
        from .rsu_check import _repeat, _slice_state

        if state["s_m"].ndim != 2 or state["s_m"].shape[0] != 1:
            raise ValueError("arms_J needs a single state [1,K]")
        self.calls += 1
        n_arm, n = len(members), self.n_samples
        batch = n_arm * n
        k = state["s_m"].shape[1]
        bstate = {name: (_repeat(v, batch) if torch.is_tensor(v) and v.ndim else v)
                  for name, v in state.items()}
        hist = dict(history=history, history_mask=history_mask, driver_history=driver_history,
                    driver_history_mask=driver_history_mask)
        for h, m in (("history", "history_mask"), ("driver_history", "driver_history_mask")):
            if hist[h] is not None and hist[m] is None:
                hist[m] = torch.ones_like(hist[h][..., 0], dtype=torch.bool)
        bhist = {name: _repeat(v, batch) for name, v in hist.items()}
        call_seed = _seed_of(self.seed, key, self.calls)
        normal_gen = torch.Generator().manual_seed(call_seed)
        cache: dict = {}

        def normal(shape, reference, cache_key):
            if cache_key not in cache:
                cache[cache_key] = torch.randn(tuple(shape), generator=normal_gen)
            return cache[cache_key].to(reference)

        weights = RewardWeights()
        out = []
        for model in self.models:
            dt = float(getattr(getattr(model, "physics", None), "dt_s", PHYSICS_DT_S))
            steps = int(round(self.horizon_s / dt))
            if steps < 1 or not math.isclose(steps * dt, self.horizon_s, abs_tol=1e-6):
                raise ValueError("horizon_s must be an integer number of WM physics steps")
            sampled = sample_driver_encodings(model, state, hist, n, normal,
                                              driver_noise_scale=self.driver_noise_scale,
                                              embedding_noise_std=self.embedding_noise_std)
            drivers = {}
            if "driver_params" in sampled:
                drivers["driver_params"] = {p: v[:, 0].repeat(n_arm, 1)
                                            for p, v in sampled["driver_params"].items()}
            if "driver_embedding" in sampled:
                drivers["driver_embedding"] = sampled["driver_embedding"][:, 0].repeat(n_arm, 1, 1)
            controllers = [self._factory(m, n)() for m in members]

            def dispatch(predicted, controllers=controllers):
                motions = []
                for index, controller in enumerate(controllers):
                    subset = _slice_state(predicted, index * n, (index + 1) * n, batch)
                    motions.append(controller(subset))
                return torch.cat(motions, dim=0)

            lanes = arm_major_lane_uniform([model], n_arm, n, steps, k,
                                           torch.Generator().manual_seed(call_seed ^ 0x5A5A))
            rollout = model.rollout(dict(bstate), n_steps=steps, cav_teacher=None,
                                    cav_controller=dispatch, **bhist, **drivers, **lanes)
            current = dict(bstate)
            total = torch.zeros(batch, dtype=torch.float64)
            for step in range(steps):
                accel = rollout["accel"][:, step]
                nxt = {**current, "s_m": rollout["positions"][:, step],
                       "v_mps": rollout["speeds"][:, step], "lane": rollout["lanes"][:, step],
                       "present": rollout["masks"][:, step], "a_mps2": accel}
                nxt.pop("veh_kin", None)
                total = total + team_reward(current, nxt, accel, weights)["delay"].double().cpu()
                current = nxt
            out.append(total.reshape(n_arm, n))
        return torch.stack(out)


# ------------------------------------------------------------------ calibration
@dataclass
class ReliabilityTable:
    """Monotone calibration map raw paired score -> P(SUMO J_plan < SUMO J_ref)."""

    upper: list            # block upper raw edges (ascending)
    lower: list            # block lower raw edges
    wins: list
    counts: list
    meta: dict = field(default_factory=dict)

    @staticmethod
    def posterior(wins: float, count: float) -> float:
        return (float(wins) + 1.0) / (float(count) + 2.0)

    @classmethod
    def fit(cls, raw: Sequence[float], outcome: Sequence[int], meta: Optional[dict] = None
            ) -> "ReliabilityTable":
        x = np.asarray(raw, np.float64)
        y = np.asarray(outcome, np.int64)
        if x.shape != y.shape or x.ndim != 1 or x.size == 0:
            raise ValueError("raw and outcome must be non-empty 1-D sequences of equal length")
        if not np.isfinite(x).all() or not np.isin(y, (0, 1)).all():
            raise ValueError("raw must be finite and outcomes 0/1")
        blocks = []                       # [lo, hi, wins, count]
        for value in np.unique(x):
            sel = x == value
            blocks.append([float(value), float(value), int(y[sel].sum()), int(sel.sum())])
        def pool(blocks, violates):
            merged = []
            for block in blocks:
                merged.append(block)
                while len(merged) > 1 and violates(merged[-2], merged[-1]):
                    b = merged.pop()
                    a = merged.pop()
                    merged.append([a[0], b[1], a[2] + b[2], a[3] + b[3]])
            return merged
        # 1) maximal isotonic blocks of the empirical win rate (ties pooled), then 2) pool any
        # remaining violator of the Beta(1,1) posterior mean (blocks of different size).
        merged = pool(blocks, lambda a, b: a[2] / a[3] >= b[2] / b[3])
        merged = pool(merged, lambda a, b: cls.posterior(a[2], a[3]) > cls.posterior(b[2], b[3]))
        table = cls([b[1] for b in merged], [b[0] for b in merged], [b[2] for b in merged],
                    [b[3] for b in merged], dict(meta or {}))
        probs = np.array([table.probability(v) for v in x])
        table.meta.update(n=int(x.size), wins=int(y.sum()), brier_in_sample=float(np.mean((probs - y) ** 2)),
                          prior="Beta(1,1)", method="isotonic_pav_posterior_mean",
                          threshold=GATE_PROBABILITY,
                          max_probability=max(table.posterior(w, c) for w, c in zip(table.wins, table.counts)),
                          blocks_at_or_above_threshold=sum(
                              table.posterior(w, c) >= GATE_PROBABILITY for w, c in zip(table.wins, table.counts)))
        return table

    def block(self, raw: float) -> int:
        raw = float(raw)
        if not math.isfinite(raw):
            return 0
        for j, hi in enumerate(self.upper):
            if raw <= hi:
                # between block j-1 and block j -> the LOWER block (conservative)
                return j if (j == 0 or raw >= self.lower[j]) else j - 1
        return len(self.upper) - 1

    def probability(self, raw: float) -> float:
        if not math.isfinite(float(raw)):
            return 0.0
        j = self.block(raw)
        return self.posterior(self.wins[j], self.counts[j])

    def beta_lower(self, raw: float, level: float = 0.95) -> float:
        """One-sided lower credible bound of the block's Beta posterior (reporting only)."""
        from scipy.stats import beta  # noqa: PLC0415 - optional, reporting only
        j = self.block(raw)
        return float(beta.ppf(1.0 - level, self.wins[j] + 1, self.counts[j] - self.wins[j] + 1))

    def to_json(self) -> dict:
        return {"schema": TABLE_SCHEMA, "upper": self.upper, "lower": self.lower,
                "wins": self.wins, "counts": self.counts, "meta": self.meta}

    @classmethod
    def from_json(cls, doc: Mapping) -> "ReliabilityTable":
        if doc.get("schema") != TABLE_SCHEMA:
            raise ValueError("not a c4 gate reliability table")
        if doc.get("meta", {}).get("split") != "TRAIN":
            raise ValueError("the gate calibration must come from TRAIN paired SUMO data")
        return cls(list(doc["upper"]), list(doc["lower"]), list(doc["wins"]), list(doc["counts"]),
                   dict(doc.get("meta", {})))

    @classmethod
    def load(cls, path) -> "ReliabilityTable":
        from .start_states import checked_path
        return cls.from_json(json.loads(checked_path(path).read_text(encoding="utf-8")))


def calibration_rows(records: Sequence[Mapping], *, unit: str, reference: str, horizon_s: float,
                     state_source: str) -> tuple[list, list, list]:
    """``(raw, outcome, seeds)`` of matching TRAIN calibration records (fail closed on eval seeds)."""
    raw, y, seeds = [], [], []
    for rec in records:
        if rec.get("schema") != RECORD_SCHEMA:
            raise ValueError("unknown gate calibration record")
        seed = int(rec["seed"])
        if EVAL_SEEDS[0] <= seed <= EVAL_SEEDS[1]:
            raise ValueError(f"evaluation seed {seed} in gate calibration data")
        if rec.get("split") != "TRAIN":
            raise ValueError("gate calibration records must be TRAIN")
        if (rec["unit"], rec["reference"], float(rec["horizon_s"]), rec["state_source"]) != (
                unit, reference, float(horizon_s), state_source):
            continue
        for u in rec["units"]:
            if u.get("sumo_J_plan") is None or u.get("sumo_J_ref") is None or not u.get("valid", True):
                continue
            raw.append(float(u["raw"]))
            y.append(int(float(u["sumo_J_plan"]) < float(u["sumo_J_ref"])))
            seeds.append(seed)
    return raw, y, seeds


def fit_table(records, *, unit, reference, horizon_s, state_source, sources=()) -> ReliabilityTable:
    raw, y, seeds = calibration_rows(records, unit=unit, reference=reference, horizon_s=horizon_s,
                                     state_source=state_source)
    if not raw:
        raise ValueError("no matching TRAIN calibration units")
    return ReliabilityTable.fit(raw, y, {"split": "TRAIN", "unit": unit, "reference": reference,
                                         "horizon_s": float(horizon_s), "state_source": state_source,
                                         "outcome": "sumo_J_plan < sumo_J_ref (speed deficit + source wait, veh-s)",
                                         "raw": "mean imagined J_ref - J_plan (road delay veh-s)",
                                         "seeds": sorted(set(seeds)), "sources": list(sources)})


# ------------------------------------------------------------------ gate
@dataclass
class GateDecision:
    time_s: Optional[float]
    units: list                   # compact slot lists
    raw: list
    probability: list
    open: list
    J: Optional[list] = None      # [M, A, S] imagined, for the log

    @property
    def open_slots(self) -> list[int]:
        return sorted(i for u, ok in zip(self.units, self.open) if ok for i in u)

    def record(self) -> dict:
        return {"time_s": self.time_s, "units": self.units, "raw": self.raw,
                "probability": self.probability, "open": self.open,
                "n_units": len(self.units), "n_open": int(sum(self.open))}


class ConfidentGate:
    """Per-unit calibrated gate on one (belief/oracle) state; logs the open rate."""

    def __init__(self, imagination: ArmImagination, table: ReliabilityTable, *, unit: str = "fleet",
                 state_source: str = "belief"):
        if unit not in UNITS:
            raise ValueError(f"unit must be one of {UNITS}")
        meta = table.meta
        if meta.get("split") != "TRAIN":
            raise ValueError("gate calibration must be TRAIN")
        expected = (unit, imagination.reference, float(imagination.horizon_s), state_source)
        found = (meta.get("unit"), meta.get("reference"), float(meta.get("horizon_s", -1)),
                 meta.get("state_source"))
        if found != expected:
            raise ValueError(f"calibration table {found} does not match the gate {expected}")
        self.imagination, self.table, self.unit = imagination, table, unit
        self.state_source = state_source
        self.log: list[dict] = []
        self.stats = {"decisions": 0, "units": 0, "units_open": 0, "decisions_any_open": 0}

    def decide(self, state, *, now_s=None, **history) -> GateDecision:
        units = gate_units(state, self.unit)
        if not units:
            decision = GateDecision(now_s, [], [], [], [])
        else:
            k = state["s_m"].shape[1]
            members: list = [torch.zeros(k, dtype=torch.bool)]
            for u in units:
                if self.unit == "fleet":
                    members.append(None)
                else:
                    m = torch.zeros(k, dtype=torch.bool)
                    m[u] = True
                    members.append(m)
            J = self.imagination.arms_J(state, members, key=f"gate{now_s}", **history)
            diff = J[:, :1] - J[:, 1:]                       # [M, U, S], + = plan better
            raw = [float(diff[:, j].mean()) for j in range(len(units))]
            prob = [self.table.probability(r) for r in raw]
            decision = GateDecision(now_s, units, raw, prob, [p >= GATE_PROBABILITY for p in prob],
                                    J.tolist())
        self.stats["decisions"] += 1
        self.stats["units"] += len(decision.units)
        self.stats["units_open"] += int(sum(decision.open))
        self.stats["decisions_any_open"] += int(any(decision.open))
        self.log.append(decision.record())
        return decision

    def summary(self) -> dict:
        s = dict(self.stats)
        s["unit_open_rate"] = s["units_open"] / s["units"] if s["units"] else None
        s.update(unit=self.unit, reference=self.imagination.reference,
                 horizon_s=self.imagination.horizon_s, state_source=self.state_source,
                 threshold=GATE_PROBABILITY, table_n=self.table.meta.get("n"))
        return s


# ------------------------------------------------------------------ SUMO execution
def env_vehicle_id(env, slot: int) -> Optional[str]:
    vi = int(env.frame.vid_index[int(slot)])
    return env.builder.vehicle_ids[vi] if vi >= 0 else None


def compact_slot_ids(source_name: str, env, tracker, keep: Tensor, slots: Sequence[int]) -> set:
    """Vehicle ids of compact imagination slots (oracle: env slots; belief: CAV track ids)."""
    out = set()
    full = keep.tolist() if keep is not None else list(slots)
    if source_name == "oracle":
        for j in slots:
            vid = env_vehicle_id(env, full[j])
            if vid is not None:
                out.add(vid)
        return out
    by_slot = {tr.slot: tr for tr in tracker.tracks.values()}
    for j in slots:
        tr = by_slot.get(full[j])
        if tr is not None and tr.is_cav and tr.cav_id is not None:
            out.add(str(tr.cav_id))
    return out


class GatedSumoPolicy:
    """``(env, state) -> (accel [1,K], lanes [1,K] | None)``: plan for open units' CAVs, reference
    (``hold`` safety-filtered a = 0 / ``native`` release) for the rest.  ``observe(env)`` after every
    SUMO step keeps the belief source current (no-op for the oracle)."""

    def __init__(self, plan_fn: Callable, gate: ConfidentGate, source, *, decision_period_s: float,
                 hold_fn: Optional[Callable] = None, memory=None):
        if gate.imagination.reference == "hold" and hold_fn is None:
            raise ValueError("the hold reference needs a hold executor")
        if not (math.isfinite(decision_period_s) and decision_period_s > 0):
            raise ValueError("decision_period_s must be positive")
        self.plan_fn, self.gate, self.source, self.hold_fn = plan_fn, gate, source, hold_fn
        self.memory = memory
        self.period = float(decision_period_s)
        self.next_decision = -math.inf
        self.open_ids: set = set()
        self.ticks = {"cav_ticks": 0, "cav_ticks_open": 0}
        self.decisions: list[dict] = []

    def observe(self, env) -> None:
        self.source.observe(env)

    def _decide(self, env) -> None:
        from .deploy import compact_inputs
        raw, info = self.source.inputs(env)
        inputs, keep = compact_inputs(raw)
        st = inputs.pop("state")
        decision = self.gate.decide(st, now_s=float(env.now), **inputs)
        tracker = getattr(self.source, "tracker", None)
        self.open_ids = compact_slot_ids(self.source.name, env, tracker, keep, decision.open_slots)
        rec = decision.record()
        rec.update(open_ids=sorted(self.open_ids), source={k: info.get(k) for k in ("n_present", "n_cav", "coverage")})
        self.decisions.append(rec)

    def __call__(self, env, state):
        if env.now >= self.next_decision - 1e-9:
            self._decide(env)
            self.next_decision = float(env.now) + self.period
        accel, lanes = self.plan_fn(env, state)
        accel = np.array(accel, dtype=np.float64, copy=True)
        cav = np.isfinite(accel[0])
        open_mask = np.array([cav[j] and env_vehicle_id(env, j) in self.open_ids
                              for j in range(accel.shape[1])], bool)
        closed = cav & ~open_mask
        if self.gate.imagination.reference == "hold":
            hold = np.asarray(self.hold_fn(state), np.float64)
            accel[0, closed] = hold[0, closed]
        else:
            accel[0, closed] = np.nan
        if lanes is not None:
            lanes = np.array(lanes, copy=True)
            lanes[0, closed] = -1
        if self.memory is not None and self.memory.accel is not None and closed.any():
            idx = torch.as_tensor(closed)[None]
            executed = torch.as_tensor(np.nan_to_num(accel, nan=0.0), dtype=self.memory.accel.dtype)
            self.memory.accel = torch.where(idx, executed, self.memory.accel)
            self.memory.intent = torch.where(idx, torch.zeros_like(self.memory.intent), self.memory.intent)
        self.ticks["cav_ticks"] += int(cav.sum())
        self.ticks["cav_ticks_open"] += int(open_mask.sum())
        return accel, lanes

    def summary(self) -> dict:
        out = self.gate.summary()
        out.update(self.ticks)
        out["cav_tick_open_rate"] = (self.ticks["cav_ticks_open"] / self.ticks["cav_ticks"]
                                     if self.ticks["cav_ticks"] else None)
        return out


def clone_belief_source(source):
    """Copy of a :class:`wms.control.deploy.BeliefSource` (tracker deep-copied, model shared)."""
    import copy

    from .deploy import BeliefSource
    if not isinstance(source, BeliefSource):
        return source
    new = BeliefSource.__new__(BeliefSource)
    new.model = source.model
    new.tracker = copy.deepcopy(source.tracker)
    new.update_seconds = []
    return new


# ------------------------------------------------------------------ CLI: fit
def main(argv=None) -> int:
    import argparse

    from .start_states import checked_path
    ap = argparse.ArgumentParser(description="Fit the gate reliability table from TRAIN calibration records")
    ap.add_argument("--records", nargs="+", required=True, help="GATE_CALIBRATION.jsonl files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--unit", choices=UNITS, default="fleet")
    ap.add_argument("--reference", choices=REFERENCES, default="hold")
    ap.add_argument("--horizon", type=float, default=20.0)
    ap.add_argument("--state-source", choices=("belief", "oracle"), default="belief")
    args = ap.parse_args(argv)
    records = []
    for path in args.records:
        for line in checked_path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    table = fit_table(records, unit=args.unit, reference=args.reference, horizon_s=args.horizon,
                      state_source=args.state_source, sources=[str(p) for p in args.records])
    out = checked_path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists (never overwritten)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table.to_json(), indent=1), encoding="utf-8")
    print(json.dumps({k: table.meta[k] for k in ("n", "wins", "brier_in_sample", "max_probability",
                                                 "blocks_at_or_above_threshold")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
