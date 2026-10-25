"""S-SUMOCREDIT (1): calibrated confident deployment + test-time plan check (owner rules 2026-10-18/25).

A CAV unit (the whole fleet, a close CAV group, or one CAV) executes a learned plan only when
the CALIBRATED probability that it beats the reference, ``P(J_plan < J_reference | RSU belief)
>= 0.8``; otherwise it falls back to the reference (``native``: released to SUMO's own driver
model, the owner's comparator; or ``hold``: safety-filtered direct a = 0, stay in lane).  0.8
is the owner's threshold and the only number of the rule; it is not a tunable parameter.

Test-time plan check (owner 2026-10-25, ``n_candidates = K > 1``): at each decision the RSU
imagines K candidate plans of the LEARNED POLICY ITSELF - the deterministic mean plan (always
candidate 0) and K-1 samples of its own stochastic head (odd k: independent per-CAV noise; even
k: one acceleration draw shared by every CAV = correlated group noise) - never a designed
controller (no PI / wall / MAPPO).  A candidate is ELIGIBLE when calibrated ``P(J_cand < J_ref)
>= 0.8`` and, for a sample (it would replace the actor's mean plan), also calibrated ``P(J_cand <
J_mean) >= 0.8`` (a second reliability table fitted on sample-vs-mean SUMO pairs; without it no
sample can replace the mean).  The eligible candidate with the lowest imagined J is executed;
no eligible candidate = abstain (fallback).  K = 1 is the gate-only arm (mean plan).

Pieces:

* :class:`IdNoiseStream` - candidate noise keyed by (seed, candidate, decision, control step,
  VEHICLE ID), so the draw of a CAV is identical in imagination (belief track id / oracle slot
  id) and in SUMO execution (:class:`SumoNoise` on env slots): common random numbers.
* :class:`ArmImagination` - paired WM-S rollouts of ``REF`` (every CAV on the reference) and the
  arms (members on a candidate, everyone else on the reference) from ONE state.  The same
  standard-normal driver draws for every arm and ensemble member, the same lane uniforms per
  sample index.  ``J`` = imagined road delay (``team_reward`` ``delay``) over the horizon.
* raw score = mean over (WM member, driver sample) of ``J_A - J_B`` (positive = B predicted
  better).  The raw score is NOT a probability.
* :class:`ReliabilityTable` - raw -> ``P(SUMO J_B < SUMO J_A)``, fitted on TRAIN paired SUMO
  branches (``policy_pairs`` gate-calibration records: the same saved state and SUMO RNG;
  SUMO J = authoritative speed-deficit + source-waiting vehicle-seconds).  Fit = maximal
  isotonic blocks of the win rate, then pooling on the Beta(1,1) posterior mean; a raw score
  between two blocks maps to the LOWER block.  No probability is hand-set: the uniform prior
  shrinks small blocks toward 0.5 (a 1/1 block reads 0.667, a 3/3 block 0.8).
  ``meta.comparison`` = ``reference`` (plan vs reference) or ``mean`` (sample vs mean plan).
* :class:`ConfidentGate` - one decision per unit per interval; logs open / abstain rates, the
  chosen candidate, chosen-vs-mean imagined improvement and decision compute time.
* :class:`GatedSumoPolicy` - SUMO executor ``(env, state) -> (accel, lanes)``.  The ungated
  policy remains its own arm/controller (paper comparison).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
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
COMPARISONS = ("reference", "mean")
TABLE_SCHEMA = "c4_gate_reliability/1"
RECORD_SCHEMA = "c4_gate_calibration_record/1"
EVAL_SEEDS = (964000, 964199)


def _seed_of(*parts) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:15], 16)


# ------------------------------------------------------------------ candidate noise
class IdNoiseStream:
    """Policy noise of candidate ``index`` keyed by vehicle id (module docstring)."""

    def __init__(self, seed: int, index: int, *, correlated: bool = False):
        if int(index) < 1:
            raise ValueError("candidate 0 is the deterministic mean plan (no noise)")
        self.seed, self.index, self.correlated = int(seed), int(index), bool(correlated)
        self.decision = 0
        self._cache: dict = {}

    def begin(self, decision) -> None:
        self.decision = decision
        self._cache = {}

    def draw(self, step: int, ids: Sequence[Optional[str]]):
        """``PolicyNoise`` ``[1, K]`` (zeros on slots without an id)."""
        from .policy import PolicyNoise, draw_policy_noise
        key = (int(step), tuple(ids))
        if key in self._cache:
            return self._cache[key]
        k = len(ids)
        accel = torch.zeros(1, k)
        gumbel = torch.zeros(1, k, 3)
        common = None
        if self.correlated:
            g = torch.Generator().manual_seed(_seed_of(self.seed, self.index, self.decision, step, "common"))
            common = torch.randn(1, generator=g)
        for j, vid in enumerate(ids):
            if vid is None:
                continue
            g = torch.Generator().manual_seed(_seed_of(self.seed, self.index, self.decision, step, vid))
            n = draw_policy_noise((1, 1), generator=g)
            accel[0, j] = common[0] if common is not None else n.accel[0, 0]
            gumbel[0, j] = n.lane_gumbel[0, 0]
        out = PolicyNoise(accel, gumbel)
        self._cache[key] = out
        return out


def candidate_streams(seed: int, n_candidates: int) -> list:
    """``[None (mean plan), stream_1, ..., stream_{K-1}]``: odd k independent, even k correlated."""
    if isinstance(n_candidates, bool) or int(n_candidates) < 1:
        raise ValueError("n_candidates must be a positive integer")
    return [None] + [IdNoiseStream(seed, k, correlated=(k % 2 == 0)) for k in range(1, int(n_candidates))]


class SumoNoise:
    """``PolicyExecutor`` noise_fn replaying a candidate stream on env slots (vehicle ids)."""

    def __init__(self, stream: IdNoiseStream, env):
        self.stream, self.env, self.step = stream, env, 0

    def begin(self) -> None:
        self.step = 0

    def __call__(self, shape):
        from .policy import PolicyNoise
        f = self.env.frame
        ids = [env_vehicle_id(self.env, j) if (bool(f.present[j]) and bool(f.is_cav[j])) else None
               for j in range(int(shape[1]))]
        out = self.stream.draw(self.step, ids)
        self.step += 1
        return PolicyNoise(out.accel.expand(int(shape[0]), -1), out.lane_gumbel.expand(int(shape[0]), -1, -1))


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
    """Imagined behaviour of reference CAVs: ``native`` = the learned native-CAV head
    (``configs/native_cav_head_v5.json``; never the IDM twin) or ``hold`` = a 0."""
    from .counterfactual import native_fn
    if reference not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}")
    return native_fn("hold" if reference == "hold" else "head", head_path=head_path)


# ------------------------------------------------------------------ imagination
class ArmImagination:
    """Paired arm rollouts ``J [members, arms, samples]`` (imagined road vehicle-seconds)."""

    def __init__(self, world_models: Sequence, policy, group_encoder, *, lane_rules=None,
                 reference: str = "native", horizon_s: float = 20.0, n_driver_samples: int = 1,
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

    def _actor(self, stream=None, ids=None):
        from .policy import policy_accel_bounds
        policy = self.policy
        if stream is None:
            def actor(obs, signal, **kwargs):
                return policy(obs, signal, **{**kwargs, "deterministic": True})
        else:
            from .policy import PolicyNoise
            counter = [0]

            def actor(obs, signal, **kwargs):
                rows = int(obs.cav_mask.shape[0])
                draw = stream.draw(counter[0], ids)
                counter[0] += 1
                noise = PolicyNoise(draw.accel.expand(rows, -1).to(obs.own.device),
                                    draw.lane_gumbel.expand(rows, -1, -1).to(obs.own.device))
                return policy(obs, signal, **{**kwargs, "deterministic": False, "noise": noise})
        actor.accel_max = policy_accel_bounds(policy)[1]
        return actor

    def _controller(self, member: Optional[Tensor], rows: int, stream=None, ids=None):
        from .bridge import make_cav_controller
        from .safety import safe_accel
        lane = {} if self.lane_rules is None else {"lane_rules": self.lane_rules}
        actor = self._actor(stream, ids)
        if member is None:
            return make_cav_controller(actor, self.group_encoder, safe_accel, **lane, **self.controller_kwargs)
        act = member.bool()[None].expand(rows, -1).clone()
        return make_cav_controller(actor, self.group_encoder, safe_accel, **lane, act_mask=act,
                                   native=self.native, **self.controller_kwargs)

    @torch.no_grad()
    def arms_J(self, state: Mapping[str, Tensor], arms: Sequence, *, ids=None, history=None,
               history_mask=None, driver_history=None, driver_history_mask=None, key="") -> Tensor:
        """``arms[a]``: ``member`` or ``(member, stream)``; member None = every CAV on the
        candidate, bool ``[K]`` = only these CAVs (all False = the reference arm); stream None =
        the deterministic mean plan.  ``ids`` (vehicle id per slot) is required for sampled
        candidates.  Returns float64 ``[M, A, S]``."""
        from .futures import arm_major_lane_uniform, sample_driver_encodings
        from .reward import team_reward
        from .rsu_check import _repeat, _slice_state

        if state["s_m"].ndim != 2 or state["s_m"].shape[0] != 1:
            raise ValueError("arms_J needs a single state [1,K]")
        specs = [a if isinstance(a, tuple) else (a, None) for a in arms]
        if any(s is not None for _, s in specs) and ids is None:
            raise ValueError("sampled candidates need the vehicle ids of the imagined slots")
        self.calls += 1
        n_arm, n = len(specs), self.n_samples
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
            controllers = [self._controller(m, n, s, ids) for m, s in specs]

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
    """Monotone calibration map raw paired score -> P(SUMO J_B < SUMO J_A)."""

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
        table.meta.setdefault("comparison", "reference")
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
                     state_source: str, comparison: str = "reference") -> tuple[list, list, list]:
    """``(raw, outcome, seeds)`` of matching TRAIN calibration units (fail closed on eval seeds).

    ``reference``: every candidate vs the reference arm; ``mean``: every SAMPLED candidate vs the
    mean plan of the same unit."""
    if comparison not in COMPARISONS:
        raise ValueError(f"comparison must be one of {COMPARISONS}")
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
            if not u.get("valid", True):
                continue
            if comparison == "reference":
                if u.get("sumo_J_plan") is None or u.get("sumo_J_ref") is None:
                    continue
                raw.append(float(u["raw"]))
                y.append(int(float(u["sumo_J_plan"]) < float(u["sumo_J_ref"])))
            else:
                if int(u.get("candidate", 0)) == 0 or u.get("sumo_J_mean") is None:
                    continue
                raw.append(float(u["raw_mean"]))
                y.append(int(float(u["sumo_J_plan"]) < float(u["sumo_J_mean"])))
            seeds.append(seed)
    return raw, y, seeds


def fit_table(records, *, unit, reference, horizon_s, state_source, comparison="reference",
              sources=()) -> ReliabilityTable:
    raw, y, seeds = calibration_rows(records, unit=unit, reference=reference, horizon_s=horizon_s,
                                     state_source=state_source, comparison=comparison)
    if not raw:
        raise ValueError(f"no matching TRAIN calibration units ({comparison})")
    other = "sumo_J_ref" if comparison == "reference" else "sumo_J_mean (same unit)"
    return ReliabilityTable.fit(raw, y, {"split": "TRAIN", "unit": unit, "reference": reference,
                                         "horizon_s": float(horizon_s), "state_source": state_source,
                                         "comparison": comparison,
                                         "outcome": f"sumo_J_plan < {other} (speed deficit + source wait, veh-s)",
                                         "raw": "mean imagined J_other - J_plan (road delay veh-s)",
                                         "seeds": sorted(set(seeds)), "sources": list(sources)})


# ------------------------------------------------------------------ gate
@dataclass
class GateDecision:
    time_s: Optional[float]
    units: list                   # compact slot lists
    raw: list                     # per unit: raw score of the MEAN plan vs the reference
    probability: list             # per unit: its calibrated probability
    open: list                    # per unit: a candidate is executed
    choice: list = field(default_factory=list)       # per unit: chosen candidate (None = abstain)
    candidates: list = field(default_factory=list)   # per unit: per-candidate evidence
    wall_ms: float = 0.0

    @property
    def open_slots(self) -> list[int]:
        return sorted(i for u, ok in zip(self.units, self.open) if ok for i in u)

    def record(self) -> dict:
        return {"time_s": self.time_s, "units": self.units, "raw": self.raw,
                "probability": self.probability, "open": self.open, "choice": self.choice,
                "candidates": self.candidates, "n_units": len(self.units),
                "n_open": int(sum(self.open)), "wall_ms": self.wall_ms}


class ConfidentGate:
    """Per-unit calibrated gate (+ K-candidate plan check) on one belief/oracle state."""

    def __init__(self, imagination: ArmImagination, table: ReliabilityTable, *, unit: str = "fleet",
                 state_source: str = "belief", n_candidates: int = 1,
                 mean_table: Optional[ReliabilityTable] = None, seed: int = 0):
        if unit not in UNITS:
            raise ValueError(f"unit must be one of {UNITS}")
        expected = (unit, imagination.reference, float(imagination.horizon_s), state_source)
        for name, tab, comparison in (("table", table, "reference"), ("mean_table", mean_table, "mean")):
            if tab is None:
                continue
            meta = tab.meta
            if meta.get("split") != "TRAIN":
                raise ValueError("gate calibration must be TRAIN")
            found = (meta.get("unit"), meta.get("reference"), float(meta.get("horizon_s", -1)),
                     meta.get("state_source"))
            if found != expected:
                raise ValueError(f"{name} {found} does not match the gate {expected}")
            if meta.get("comparison", "reference") != comparison:
                raise ValueError(f"{name} must be a '{comparison}' comparison table")
        self.imagination, self.table, self.mean_table = imagination, table, mean_table
        self.unit, self.state_source = unit, state_source
        self.n_candidates = int(n_candidates)
        self.streams = candidate_streams(seed, self.n_candidates)
        self.log: list[dict] = []
        self.stats = {"decisions": 0, "units": 0, "units_open": 0, "decisions_any_open": 0,
                      "units_abstain": 0, "units_chose_sample": 0}
        self.chosen_vs_mean: list = []
        self.wall_ms: list = []

    def arms_for(self, units, k_slots: int) -> list:
        arms: list = [torch.zeros(k_slots, dtype=torch.bool)]
        for u in units:
            member = None
            if self.unit != "fleet":
                member = torch.zeros(k_slots, dtype=torch.bool)
                member[u] = True
            arms += [(member, stream) for stream in self.streams]
        return arms

    def decide(self, state, *, now_s=None, ids=None, **history) -> GateDecision:
        t0 = time.perf_counter()
        units = gate_units(state, self.unit)
        index = self.stats["decisions"]
        for stream in self.streams[1:]:
            stream.begin(index)
        K = self.n_candidates
        raw, prob, open_, choice, detail = [], [], [], [], []
        if units:
            J = self.imagination.arms_J(state, self.arms_for(units, state["s_m"].shape[1]),
                                        ids=ids if K > 1 else None, key=f"gate{index}", **history)
            for u in range(len(units)):
                col = [1 + u * K + k for k in range(K)]
                raw_ref = [float((J[:, 0] - J[:, c]).mean()) for c in col]
                p_ref = [self.table.probability(r) for r in raw_ref]
                raw_mean = [None] + [float((J[:, col[0]] - J[:, c]).mean()) for c in col[1:]]
                p_mean = [None] + [None if self.mean_table is None else self.mean_table.probability(r)
                                   for r in raw_mean[1:]]
                eligible = [p_ref[0] >= GATE_PROBABILITY] + [
                    p_ref[k] >= GATE_PROBABILITY and p_mean[k] is not None and p_mean[k] >= GATE_PROBABILITY
                    for k in range(1, K)]
                pick = None
                for k in range(K):
                    if eligible[k] and (pick is None or raw_ref[k] > raw_ref[pick]):
                        pick = k
                raw.append(raw_ref[0])
                prob.append(p_ref[0])
                open_.append(pick is not None)
                choice.append(pick)
                detail.append({"raw_ref": raw_ref, "p_ref": p_ref, "raw_mean": raw_mean,
                               "p_mean": p_mean, "eligible": eligible,
                               "chosen_vs_mean": (raw_mean[pick] if pick else (0.0 if pick == 0 else None))})
                if pick is not None:
                    self.chosen_vs_mean.append(raw_mean[pick] if pick else 0.0)
        decision = GateDecision(now_s, units, raw, prob, open_, choice, detail,
                                (time.perf_counter() - t0) * 1000.0)
        self.wall_ms.append(decision.wall_ms)
        self.stats["decisions"] += 1
        self.stats["units"] += len(units)
        self.stats["units_open"] += int(sum(open_))
        self.stats["units_abstain"] += int(len(units) - sum(open_))
        self.stats["units_chose_sample"] += int(sum(1 for c in choice if c))
        self.stats["decisions_any_open"] += int(any(open_))
        self.log.append(decision.record())
        return decision

    def summary(self) -> dict:
        s = dict(self.stats)
        s["unit_open_rate"] = s["units_open"] / s["units"] if s["units"] else None
        s["abstain_rate"] = s["units_abstain"] / s["units"] if s["units"] else None
        s["chosen_vs_mean_mean"] = (float(np.mean(self.chosen_vs_mean)) if self.chosen_vs_mean else None)
        s["decision_ms"] = ({"mean": float(np.mean(self.wall_ms)), "max": float(np.max(self.wall_ms)),
                             "n": len(self.wall_ms)} if self.wall_ms else None)
        s.update(unit=self.unit, reference=self.imagination.reference, n_candidates=self.n_candidates,
                 horizon_s=self.imagination.horizon_s, state_source=self.state_source,
                 threshold=GATE_PROBABILITY, table_n=self.table.meta.get("n"),
                 mean_table_n=None if self.mean_table is None else self.mean_table.meta.get("n"))
        return s


# ------------------------------------------------------------------ SUMO execution
def env_vehicle_id(env, slot: int) -> Optional[str]:
    vi = int(env.frame.vid_index[int(slot)])
    return env.builder.vehicle_ids[vi] if vi >= 0 else None


def compact_ids(source_name: str, env, tracker, keep: Optional[Tensor], k: int) -> list:
    """Vehicle id per compact imagination slot (oracle: env slots; belief: CAV track ids; None for
    HDVs / unknown)."""
    full = keep.tolist() if keep is not None else list(range(k))
    if source_name == "oracle":
        f = env.frame
        return [env_vehicle_id(env, s) if (bool(f.present[s]) and bool(f.is_cav[s])) else None for s in full]
    by_slot = {tr.slot: tr for tr in tracker.tracks.values()}
    out = []
    for s in full:
        tr = by_slot.get(s)
        out.append(str(tr.cav_id) if (tr is not None and tr.is_cav and tr.cav_id is not None) else None)
    return out


def compact_slot_ids(source_name: str, env, tracker, keep: Tensor, slots: Sequence[int]) -> set:
    """Vehicle ids of the given compact slots."""
    ids = compact_ids(source_name, env, tracker, keep, len(keep))
    return {ids[j] for j in slots if ids[j] is not None}


class GatedSumoPolicy:
    """``(env, state) -> (accel [1,K], lanes [1,K] | None)``: each open unit's CAVs execute their
    chosen candidate (mean plan or a replayed policy sample), every other CAV the reference
    (``native`` release / ``hold`` safety-filtered a = 0, stay).  ``observe(env)`` after every
    SUMO step keeps the belief source current (no-op for the oracle).

    ``sample_fn(stream, env) -> callable(env, state) -> (accel, lanes)`` builds a stochastic
    executor that replays a candidate stream (:class:`SumoNoise`); required when K > 1."""

    def __init__(self, plan_fn: Callable, gate: ConfidentGate, source, *, decision_period_s: float,
                 hold_fn: Optional[Callable] = None, memory=None, sample_fn: Optional[Callable] = None):
        if gate.imagination.reference == "hold" and hold_fn is None:
            raise ValueError("the hold reference needs a hold executor")
        if gate.n_candidates > 1 and sample_fn is None:
            raise ValueError("the K-candidate check needs a sample executor factory")
        if not (math.isfinite(decision_period_s) and decision_period_s > 0):
            raise ValueError("decision_period_s must be positive")
        self.plan_fn, self.gate, self.source, self.hold_fn = plan_fn, gate, source, hold_fn
        self.sample_fn, self.memory = sample_fn, memory
        self.period = float(decision_period_s)
        self.next_decision = -math.inf
        self.choice_by_id: dict = {}
        self.samplers: dict = {}
        self.ticks = {"cav_ticks": 0, "cav_ticks_open": 0, "cav_ticks_sample": 0}
        self.decisions: list[dict] = []

    @property
    def open_ids(self) -> set:
        return set(self.choice_by_id)

    def observe(self, env) -> None:
        self.source.observe(env)

    def _decide(self, env) -> None:
        from .deploy import compact_inputs
        raw, info = self.source.inputs(env)
        inputs, keep = compact_inputs(raw)
        st = inputs.pop("state")
        tracker = getattr(self.source, "tracker", None)
        ids = compact_ids(self.source.name, env, tracker, keep, int(st["s_m"].shape[1]))
        decision = self.gate.decide(st, now_s=float(env.now), ids=ids, **inputs)
        self.choice_by_id = {}
        for u, k in zip(decision.units, decision.choice):
            if k is None:
                continue
            for j in u:
                if ids[j] is not None:
                    self.choice_by_id[ids[j]] = int(k)
        self.samplers = {k: self.sample_fn(self.gate.streams[k], env)
                         for k in sorted(set(self.choice_by_id.values())) if k > 0}
        rec = decision.record()
        rec.update(open_ids=sorted(self.choice_by_id),
                   source={key: info.get(key) for key in ("n_present", "n_cav", "coverage")})
        self.decisions.append(rec)

    def __call__(self, env, state):
        if env.now >= self.next_decision - 1e-9:
            self._decide(env)
            self.next_decision = float(env.now) + self.period
        sampled = {k: fn(env, state) for k, fn in self.samplers.items()}
        accel, lanes = self.plan_fn(env, state)          # the mean plan, called last (memory)
        accel = np.array(accel, dtype=np.float64, copy=True)
        lanes = None if lanes is None else np.array(lanes, copy=True)
        cav = np.isfinite(accel[0])
        k_of = np.full(accel.shape[1], -1)
        for j in np.flatnonzero(cav):
            k_of[j] = self.choice_by_id.get(env_vehicle_id(env, j), -1)
        closed = cav & (k_of < 0)
        for k, (acc_k, lanes_k) in sampled.items():
            sel = cav & (k_of == k)
            accel[0, sel] = np.asarray(acc_k, np.float64)[0, sel]
            if lanes is not None:
                lanes[0, sel] = -1 if lanes_k is None else np.asarray(lanes_k)[0, sel]
        if self.gate.imagination.reference == "hold":
            hold = np.asarray(self.hold_fn(state), np.float64)
            accel[0, closed] = hold[0, closed]
        else:
            accel[0, closed] = np.nan
        if lanes is not None:
            lanes[0, closed] = -1
        changed = closed | (cav & (k_of > 0))
        if self.memory is not None and self.memory.accel is not None and changed.any():
            idx = torch.as_tensor(changed)[None]
            executed = torch.as_tensor(np.nan_to_num(accel, nan=0.0), dtype=self.memory.accel.dtype)
            lane_now = state["lane"].long().cpu()
            intent = torch.zeros_like(self.memory.intent)
            if lanes is not None:
                tgt = torch.as_tensor(lanes).long()
                intent = torch.where(tgt >= 0, (tgt - lane_now).sign().to(intent.dtype), intent)
            self.memory.accel = torch.where(idx, executed, self.memory.accel)
            self.memory.intent = torch.where(idx, intent, self.memory.intent)
        self.ticks["cav_ticks"] += int(cav.sum())
        self.ticks["cav_ticks_open"] += int((cav & (k_of >= 0)).sum())
        self.ticks["cav_ticks_sample"] += int((cav & (k_of > 0)).sum())
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
    ap = argparse.ArgumentParser(description="Fit a gate reliability table from TRAIN calibration records")
    ap.add_argument("--records", nargs="+", required=True, help="GATE_CALIBRATION.jsonl files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--unit", choices=UNITS, default="fleet")
    ap.add_argument("--reference", choices=REFERENCES, default="native")
    ap.add_argument("--horizon", type=float, default=20.0)
    ap.add_argument("--state-source", choices=("belief", "oracle"), default="belief")
    ap.add_argument("--comparison", choices=COMPARISONS, default="reference",
                    help="reference: candidate vs reference arm; mean: sampled candidate vs the mean plan")
    args = ap.parse_args(argv)
    records = []
    for path in args.records:
        for line in checked_path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    table = fit_table(records, unit=args.unit, reference=args.reference, horizon_s=args.horizon,
                      state_source=args.state_source, comparison=args.comparison,
                      sources=[str(p) for p in args.records])
    out = checked_path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists (never overwritten)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table.to_json(), indent=1), encoding="utf-8")
    print(json.dumps({k: table.meta[k] for k in ("comparison", "n", "wins", "brier_in_sample",
                                                 "max_probability", "blocks_at_or_above_threshold")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
