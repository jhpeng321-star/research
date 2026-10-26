"""S-SUMOCREDIT extras of the LOOP14 policy-pair collection (``collect --policy-pairs``).

All are opt-in (``PairExtras`` fields default off) and run AFTER the unchanged POLICY / B arms
of every saved decision state, each arm in a fresh SUMO process loaded from the same state file
(same SUMO RNG = common random numbers).  Nothing here writes branch episodes: the WM fine-tune
data and the critic returns of the loop are untouched; results go to ``GATE_CALIBRATION.jsonl``,
``GATED_ARMS.jsonl`` and ``SUMOCREDIT_POINTS.jsonl`` in the job's branch directory (gathered by
``wms.control.real_credit14 gather``).

1. **Gate calibration** (``gate_horizon > 0``): at the decision state the RSU belief (a
   ``BeliefSource`` fed during the native base episode, copied at the save) or the oracle state is
   imagined (``wms.control.confident_gate``): REF + for every unit u the K candidates of the
   learned policy (k = 0 mean plan, k >= 1 replayed policy samples).  SUMO then runs the same
   arms for ``gate_horizon`` s: REF (every CAV on the reference) and PLAN_{u,k} (unit u on
   candidate k, everyone else on the reference).  One record per (unit, candidate): imagined
   raw scores vs the reference (and vs the unit's mean plan for samples) and the SUMO J of the
   arms.  The fleet mean arm and the native REF arm reuse the POLICY / B prefixes when
   ``gate_horizon <= branch_horizon``.
2. **Gated arms** (``gate_table``): ``GATED`` = the mean plan behind the calibrated 0.8 gate
   (gate-only) and, with ``gate_candidates = K > 1`` and a ``gate_mean_table``, ``GATED_K`` =
   gate + K-sample plan check, each for ``branch_horizon`` s; compared with the ungated POLICY
   arm and native B of the same state (three evaluation arms + native).
3. **Real credit** (``credit_horizon > 0``): the close CAV group G (``branching.select_group``,
   at most ``credit_group_max``) and arms ONLY:i, GROUP and ``credit_flips`` FLIP:i (B and ALL =
   the B / POLICY prefixes); optionally imagined from the oracle state (reference = learned
   native head) for the trainer's credit correction (``wms.control.real_credit14``).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def prefix_record(ledger: dict, totals: dict) -> dict:
    return {"J": float(ledger["speed_deficit_vehicle_seconds"] + ledger["source_wait_vehicle_seconds"]),
            "road": float(ledger["speed_deficit_vehicle_seconds"]),
            "source": float(ledger["source_wait_vehicle_seconds"]),
            "collisions": int(totals["collisions"]), "teleports": int(totals["teleports"]),
            "dropped_vehicle_steps": int(totals["dropped_vehicle_steps"])}


def _valid(rec: Optional[dict]) -> bool:
    return bool(rec) and not (rec["collisions"] or rec["teleports"] or rec["dropped_vehicle_steps"])


def sim_s_per_point(x: dict, branch_h: int) -> int:
    """Upper bound of the extra SUMO seconds per decision point (budget charge; mirrored by
    ``scripts/c4/loop14.py``).  ``x``: PairExtras fields; ``gated_arm`` (loop config) counts like a
    table being present."""
    h = int(branch_h)
    total = 0
    g, c = int(x.get("gate_horizon", 0)), int(x.get("credit_horizon", 0))
    k = int(x.get("gate_candidates", 1))
    if g:
        unit = x.get("gate_unit", "fleet")
        total += 0 if (x.get("gate_reference", "native") == "native" and g <= h) else g
        n_units = 1 if unit == "fleet" else int(x.get("gate_max_units", 4))
        total += n_units * k * g - (g if (unit == "fleet" and g <= h) else 0)
    if x.get("gate_table") or x.get("gated_arm"):
        total += h * (2 if k > 1 else 1)
    if c:
        total += (int(x.get("credit_group_max", 3)) + 1 + int(x.get("credit_flips", 0))) * c
        total += 0 if c <= h else 2 * c
    total += int(x.get("long_group_horizon", 0))  # worst-case selected point; B/ALL must exist
    total += int(x.get('long_single_horizon', 0))
    return total


@dataclass
class PairExtras:
    credit_horizon: int = 0
    credit_group_max: int = 3
    credit_flips: int = 0
    credit_imagine: bool = True
    gate_horizon: int = 0
    gate_unit: str = "fleet"
    gate_reference: str = "native"
    gate_state: str = "belief"
    gate_max_units: int = 4
    gate_candidates: int = 1
    gate_table: Optional[str] = None
    gate_mean_table: Optional[str] = None
    gate_decision_period: float = 5.0
    n_driver_samples: int = 1
    seed: int = 0
    long_group_horizon: int = 0
    long_group_time: int = 0
    long_single_horizon: int = 0
    long_single_time: int = 0

    def __post_init__(self):
        from wms.control.confident_gate import REFERENCES, UNITS
        if self.gate_unit not in UNITS or self.gate_reference not in REFERENCES:
            raise ValueError("unknown gate unit / reference")
        if self.gate_state not in ("belief", "oracle"):
            raise ValueError("gate_state must be belief or oracle")
        for name in ("credit_horizon", "gate_horizon", "credit_group_max", "credit_flips",
                     "gate_max_units"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")
        if int(self.gate_candidates) < 1:
            raise ValueError("gate_candidates must be >= 1 (1 = gate-only, mean plan)")
        if self.credit_horizon and self.credit_group_max < 2:
            raise ValueError("credit_group_max must be >= 2 (a group has two or more CAVs)")
        if self.gate_mean_table and not self.gate_table:
            raise ValueError("gate_mean_table needs the reference gate_table")
        if self.long_group_horizon not in (0, 300) or self.long_group_time < 0:
            raise ValueError("long GROUP horizon must be 0/OFF or 300 s, time >= 0")
        if self.long_single_horizon not in (0, 300) or self.long_single_time < 0:
            raise ValueError('long single horizon must be 0/OFF or 300 s, time >= 0')

    @property
    def active(self) -> bool:
        return bool(self.credit_horizon or self.gate_horizon or self.gate_table
                    or self.long_group_horizon or self.long_single_horizon)

    @property
    def needs_wm(self) -> bool:
        return bool(self.gate_horizon or self.gate_table
                    or ((self.credit_horizon or self.long_group_horizon) and self.credit_imagine))

    def marks(self, branch_h: int) -> set:
        hs = [self.credit_horizon, self.gate_horizon, branch_h if self.gate_table else 0,
              self.long_group_horizon, self.long_single_horizon]
        return {int(h) for h in hs if 0 < int(h) <= int(branch_h)}

    def sim_s_per_point(self, branch_h: int) -> int:
        return sim_s_per_point(asdict(self), branch_h)

    # ------------------------------------------------------------------ helpers
    def base_belief(self, policy_factory):
        if not ((self.gate_horizon or self.gate_table) and self.gate_state == "belief"):
            return None
        from wms.control.deploy import BeliefSource
        return BeliefSource(policy_factory.models[0])

    def _source(self, policy_factory, point):
        from wms.control.confident_gate import clone_belief_source
        from wms.control.deploy import OracleSource
        if self.gate_state == "oracle":
            return OracleSource(policy_factory.models[0])
        return clone_belief_source(point["belief"])

    def _imagination(self, policy_factory, reference, horizon):
        from wms.control.confident_gate import ArmImagination
        policy, encoder, rules = policy_factory.components
        return ArmImagination(policy_factory.models, policy, encoder, lane_rules=rules,
                              reference=reference, horizon_s=float(horizon),
                              n_driver_samples=int(self.n_driver_samples), seed=int(self.seed))

    @staticmethod
    def _hold(policy_factory):
        from wms.control.obs import CommandMemory
        from wms.control.policy import PolicyExecutor
        policy, _, rules = policy_factory.components
        return PolicyExecutor(policy, None, lane_rules=rules, hold=True, memory=CommandMemory())

    @staticmethod
    def _sampler(policy_factory, memory=None):
        """``(stream, env) -> PairPolicy`` replaying a candidate stream on env slots (SumoNoise)."""
        from wms.control.confident_gate import SumoNoise
        from wms.control.policy import PolicyExecutor

        from .policy_pairs import PairPolicy
        policy, encoder, rules = policy_factory.components

        def make(stream, env):
            exe = PolicyExecutor(policy, encoder, lane_rules=rules, deterministic=False,
                                 noise_fn=SumoNoise(stream, env), memory=memory,
                                 device=getattr(policy_factory, "device", "cpu"))
            return PairPolicy(exe, getattr(policy_factory, "settings", None))
        return make

    @staticmethod
    def _load(env, point, horizon):
        from .branching import load_state, prepare_branch_mode
        load_state(env, point["path"], point["snap"])
        prepare_branch_mode(env)
        env.end_s = max(env.end_s, point["t_s"] + int(horizon) + 1)
        env.recorder = None

    def _run_arm(self, env, point, horizon, step_fn=None, observe=None) -> dict:
        self._load(env, point, horizon)
        st = env.state()
        totals = {"collisions": 0, "teleports": 0, "dropped_vehicle_steps": 0}
        ledger = {"speed_deficit_vehicle_seconds": 0.0, "source_wait_vehicle_seconds": 0.0}
        for _ in range(int(horizon)):
            if step_fn is None:
                st, info = env.step(None)
            else:
                accel, kw = step_fn(env, st)
                st, info = env.step(accel, **kw)
            iv = info["interval"]
            totals["collisions"] += int(iv.get("collision_involvements", 0))
            totals["teleports"] += int(iv.get("starting_teleports", 0))
            totals["dropped_vehicle_steps"] += int(iv.get("dropped_vehicle_steps", 0))
            for key in ledger:
                ledger[key] += iv.get(key, 0)
            if observe is not None:
                observe(env)
        out = prefix_record(ledger, totals)
        out["sim_s"] = int(horizon)
        return out

    @staticmethod
    def _members_step(fn, members, hold=None, make=None):
        """Members on the policy candidate ``fn`` (or ``make(env)``, built at the first tick after
        the load); the other CAVs released (``hold`` None) or held (a = 0, stay)."""
        from .branching import member_mask
        box = {"fn": fn}

        def step(env, st):
            if box["fn"] is None and make is not None:
                box["fn"] = make(env)
            f = box["fn"]
            accel, lanes = f(env, st) if f is not None else (None, None)
            mask = member_mask(env, members)
            cav = env.frame.present & env.frame.is_cav
            out = np.full((1, env.k), np.nan)
            if accel is not None:
                out[0, mask] = np.asarray(accel)[0, mask]
            kw = {}
            lane_out = None
            if lanes is not None:
                lane_out = np.full((1, env.k), -1, dtype=np.int64)
                lane_out[0, mask] = np.asarray(lanes)[0, mask]
            if hold is not None:
                closed = cav & ~mask
                h_acc, _ = hold(st, with_lanes=True)
                out[0, closed] = np.asarray(h_acc)[0, closed]
                if lane_out is None:
                    lane_out = np.full((1, env.k), -1, dtype=np.int64)
            if lane_out is not None:
                kw["lanes"] = lane_out
            return out, kw
        return step

    # ------------------------------------------------------------------ point
    def run_point(self, env, point, prefix, *, policy_factory, seed, world, branch_h) -> dict:
        out = {"t_s": point["t_s"], "seed": int(seed), "world": world, "sim_s": 0}
        t0 = time.monotonic()
        if self.gate_horizon:
            out["gate"] = self._gate_calibration(env, point, prefix, policy_factory, seed, branch_h)
            out["sim_s"] += out["gate"]["sim_s"]
        if self.gate_table:
            out["gated"] = self._gated_arms(env, point, prefix, policy_factory, seed, branch_h)
            out["sim_s"] += out["gated"]["sim_s"]
        if self.credit_horizon:
            out["credit"] = self._credit(env, point, prefix, policy_factory, seed, world, branch_h)
            out["sim_s"] += out["credit"]["extra_sim_s"]
        if self.long_group_horizon and int(point["t_s"]) == self.long_group_time:
            if (branch_h < self.long_group_horizon or any(
                    (arm, self.long_group_horizon) not in prefix for arm in ("B", "POLICY"))):
                raise ValueError("long GROUP requires existing 300 s B/ALL prefixes")
            out["credit_long"] = self._credit(env, point, prefix, policy_factory, seed, world, branch_h,
                                               horizon=self.long_group_horizon, group_only=True)
            out["sim_s"] += out["credit_long"]["extra_sim_s"]
        if self.long_single_horizon and int(point['t_s']) == self.long_single_time:
            if branch_h < 300 or ('B', 300) not in prefix:
                raise ValueError('long single requires the existing 300 s native prefix')
            out['single_long'] = self._single_hold(env, point, prefix, seed, world)
            out['sim_s'] += out['single_long']['extra_sim_s']
        out["wall_s"] = time.monotonic() - t0
        return out

    def _gate_calibration(self, env, point, prefix, policy_factory, seed, branch_h) -> dict:
        import torch

        from wms.control.confident_gate import RECORD_SCHEMA, _seed_of, candidate_streams, compact_ids, gate_units
        from wms.control.deploy import compact_inputs
        g = int(self.gate_horizon)
        K = int(self.gate_candidates)
        self._load(env, point, g)
        source = self._source(policy_factory, point)
        raw_inputs, info = source.inputs(env)
        inputs, keep = compact_inputs(raw_inputs)
        st = inputs.pop("state")
        k_slots = int(st["s_m"].shape[1])
        tracker = getattr(source, "tracker", None)
        ids = compact_ids(source.name, env, tracker, keep, k_slots)
        units = gate_units(st, self.gate_unit)[: (1 if self.gate_unit == "fleet" else int(self.gate_max_units))]
        rec = {"schema": RECORD_SCHEMA, "split": "TRAIN", "seed": int(seed), "t_s": point["t_s"],
               "unit": self.gate_unit, "reference": self.gate_reference, "horizon_s": float(g),
               "state_source": self.gate_state, "n_candidates": K, "units": [], "sim_s": 0,
               "source": {k: info.get(k) for k in ("n_present", "n_cav", "coverage")}}
        if not units:
            return rec
        streams = candidate_streams(_seed_of(self.seed, seed, point["t_s"], "cal"), K)
        for s in streams[1:]:
            s.begin(0)
        arms: list = [torch.zeros(k_slots, dtype=torch.bool)]
        for u in units:
            member = None
            if self.gate_unit != "fleet":
                member = torch.zeros(k_slots, dtype=torch.bool)
                member[u] = True
            arms += [(member, s) for s in streams]
        imag = self._imagination(policy_factory, self.gate_reference, g)
        J = imag.arms_J(st, arms, ids=ids if K > 1 else None, key=f"cal:{seed}:{point['t_s']}", **inputs)
        hold = self._hold(policy_factory) if self.gate_reference == "hold" else None
        if self.gate_reference == "native" and ("B", g) in prefix:
            ref = prefix[("B", g)]
        else:
            ref = self._run_arm(env, point, g, self._members_step(None, frozenset(), hold=hold))
            rec["sim_s"] += g
        sampler = self._sampler(policy_factory)
        for ui, u in enumerate(units):
            members = None if self.gate_unit == "fleet" else frozenset(
                ids[j] for j in u if ids[j] is not None)
            mean_res = None
            for k, stream in enumerate(streams):
                col = 1 + ui * K + k
                if k == 0 and self.gate_unit == "fleet" and ("POLICY", g) in prefix:
                    res = prefix[("POLICY", g)]
                else:
                    if k == 0:
                        step = self._members_step(policy_factory(), members, hold=hold)
                    else:
                        step = self._members_step(None, members, hold=hold,
                                                  make=lambda env, s=stream: sampler(s, env))
                    res = self._run_arm(env, point, g, step)
                    rec["sim_s"] += g
                if k == 0:
                    mean_res = res
                entry = {"members": sorted(ids[j] for j in u if ids[j] is not None), "candidate": k,
                         "correlated": bool(stream is not None and stream.correlated),
                         "raw": float((J[:, 0] - J[:, col]).mean()),
                         "imagined_J_ref": float(J[:, 0].mean()), "imagined_J_plan": float(J[:, col].mean()),
                         "sumo_J_ref": ref["J"], "sumo_J_plan": res["J"],
                         "sumo_road_ref": ref["road"], "sumo_road_plan": res["road"],
                         "valid": _valid(ref) and _valid(res)}
                if k > 0:
                    entry.update(raw_mean=float((J[:, col - k] - J[:, col]).mean()),
                                 sumo_J_mean=mean_res["J"], valid=entry["valid"] and _valid(mean_res))
                rec["units"].append(entry)
        return rec

    def _gate(self, policy_factory, n_candidates):
        from wms.control.confident_gate import ConfidentGate, ReliabilityTable
        table = ReliabilityTable.load(self.gate_table)
        meta = table.meta
        if meta["state_source"] != self.gate_state:
            raise ValueError("the gate table's state source differs from --gate-state")
        mean_table = (ReliabilityTable.load(self.gate_mean_table)
                      if (n_candidates > 1 and self.gate_mean_table) else None)
        imag = self._imagination(policy_factory, meta["reference"], float(meta["horizon_s"]))
        return ConfidentGate(imag, table, unit=meta["unit"], state_source=meta["state_source"],
                             n_candidates=n_candidates, mean_table=mean_table, seed=int(self.seed))

    def _gated_arms(self, env, point, prefix, policy_factory, seed, branch_h) -> dict:
        from wms.control.confident_gate import GatedSumoPolicy
        h = int(branch_h)
        pol, nat = prefix.get(("POLICY", h)), prefix.get(("B", h))
        out = {"schema": "c4_gated_arm/2", "split": "TRAIN", "seed": int(seed), "t_s": point["t_s"],
               "horizon_s": h, "policy_J": None if pol is None else pol["J"],
               "native_J": None if nat is None else nat["J"], "table": str(self.gate_table),
               "mean_table": self.gate_mean_table, "arms": {}, "sim_s": 0}
        variants = [("GATED", 1)] + ([("GATED_K", int(self.gate_candidates))] if self.gate_candidates > 1 else [])
        for name, K in variants:
            gate = self._gate(policy_factory, K)
            plan = policy_factory()
            hold = self._hold(policy_factory) if gate.imagination.reference == "hold" else None
            source = self._source(policy_factory, point)
            gp = GatedSumoPolicy(plan, gate, source, decision_period_s=float(self.gate_decision_period),
                                 hold_fn=hold, memory=plan.executor.memory,
                                 sample_fn=self._sampler(policy_factory, memory=plan.executor.memory) if K > 1 else None)

            def step(env, st, gp=gp):
                accel, lanes = gp(env, st)
                return accel, ({} if lanes is None else {"lanes": lanes})
            res = self._run_arm(env, point, h, step, observe=gp.observe)
            out["arms"][name] = {"result": res, "gate": gp.summary(), "decisions": gp.decisions}
            out["sim_s"] += h
        return out

    def _credit(self, env, point, prefix, policy_factory, seed, world, branch_h, *,
                horizon=None, group_only=False) -> dict:
        import torch

        from wms.control.real_credit14 import POINT_SCHEMA

        from .branching import select_group
        c = int(self.credit_horizon if horizon is None else horizon)
        self._load(env, point, c)
        f = env.frame
        ids = env.builder.vehicle_ids
        s_by_vid = {}
        for slot in np.flatnonzero(f.present & f.is_cav).tolist():
            vi = int(f.vid_index[slot])
            if vi >= 0 and int(f.lane[slot]) >= 0:
                s_by_vid[ids[vi]] = float(f.s[slot])
        group = select_group(s_by_vid, max_size=int(self.credit_group_max))
        rec = {"schema": POINT_SCHEMA, "split": "TRAIN", "seed": int(seed), "world": world,
               "t_s": point["t_s"], "horizon_s": c, "group": group, "flips": [], "sumo": None,
               "sumo_road": None, "imagined": None, "collisions": 0, "teleports": 0, "extra_sim_s": 0}
        if not group:
            return rec
        rng = np.random.default_rng([int(seed), int(point["t_s"]), 0x5C]) if self.credit_flips else None
        flips = (sorted(rng.choice(group, size=min(int(self.credit_flips), len(group)), replace=False).tolist())
                 if rng is not None else [])
        rec["flips"] = flips
        arms = {"B": frozenset(), "ALL": None, **{f"ONLY:{v}": frozenset([v]) for v in group},
                "GROUP": frozenset(group), **{f"FLIP:{v}": frozenset(group) - {v} for v in flips}}
        if group_only:
            arms = {k: arms[k] for k in ("B", "ALL", "GROUP")}
            rec["flips"] = []
            rec["long_group"] = True
        if self.credit_imagine:
            from wms.control.deploy import OracleSource, compact_inputs
            raw, _ = OracleSource(policy_factory.models[0]).inputs(env)
            inputs, keep = compact_inputs(raw)
            st = inputs.pop("state")
            slot_vid = {}
            for j, slot in enumerate(keep.tolist()):
                vi = int(f.vid_index[slot])
                slot_vid[j] = ids[vi] if vi >= 0 else None
            k = st["s_m"].shape[1]
            masks = []
            for name, members in arms.items():
                if members is None:
                    masks.append(None)
                else:
                    masks.append(torch.tensor([slot_vid[j] in members for j in range(k)], dtype=torch.bool))
            imag = self._imagination(policy_factory, "native", c)
            J = imag.arms_J(st, masks, key=f"credit:{seed}:{point['t_s']}", **inputs)
            rec["imagined"] = {name: float(J[:, a].mean()) for a, name in enumerate(arms)}
        sumo, road = {}, {}
        for name, members in arms.items():
            key = {"B": ("B", c), "ALL": ("POLICY", c)}.get(name)
            if key is not None and key in prefix:
                res = prefix[key]
            else:
                step = None if name == "B" else self._members_step(policy_factory(), members)
                res = self._run_arm(env, point, c, step)
                rec["extra_sim_s"] += c
                if group_only:
                    rec["extra_collisions"] = rec.get("extra_collisions", 0) + res["collisions"]
                    rec["extra_teleports"] = rec.get("extra_teleports", 0) + res["teleports"]
            sumo[name], road[name] = res["J"], res["road"]
            rec["collisions"] += res["collisions"]
            rec["teleports"] += res["teleports"]
            if res.get("dropped_vehicle_steps", 0):
                # Existing reader excludes incident records; an incomplete fleet
                # also invalidates long differences, without hiding its charge.
                rec["dropped_vehicle_steps"] = rec.get("dropped_vehicle_steps", 0) + res["dropped_vehicle_steps"]
        rec["sumo"], rec["sumo_road"] = sumo, road
        return rec

    # ------------------------------------------------------------------ output
    def _single_hold(self, env, point, prefix, seed, world):
        from wms.control.contract import MERGE_S0_M
        from wms.control.real_credit14 import POINT_SCHEMA
        self._load(env, point, 300)
        frame = env.frame
        ids = env.builder.vehicle_ids
        candidates = []
        for slot in np.flatnonzero(frame.present & frame.is_cav).tolist():
            vi = int(frame.vid_index[slot])
            if vi >= 0 and int(frame.lane[slot]) in (0, 1) and float(frame.s[slot]) < MERGE_S0_M:
                candidates.append((-float(frame.s[slot]), str(ids[vi])))
        rec = {'schema': POINT_SCHEMA, 'split': 'TRAIN', 'seed': int(seed), 'world': world,
               't_s': point['t_s'], 'horizon_s': 300, 'single_hold': True, 'group': [], 'flips': [],
               'selected_vehicle': None, 'sumo': None, 'imagined': None,
               'collisions': 0, 'teleports': 0, 'extra_sim_s': 0}
        if not candidates:
            rec['skipped'] = 'no upstream mainline CAV'
            return rec
        _, vehicle = min(candidates)
        rec['selected_vehicle'] = vehicle
        # Constant zero acceleration on one stable vehicle ID, everyone else
        # released to native SUMO. This is an offline credit probe, not an actor action.
        def hold(env, state):
            return np.zeros((1, env.k)), None
        step = self._members_step(hold, frozenset([vehicle]))
        result = self._run_arm(env, point, 300, step)
        base = prefix[('B', 300)]
        rec.update(sumo={'B': base['J'], 'SINGLE_HOLD': result['J']},
                   sumo_road={'B': base['road'], 'SINGLE_HOLD': result['road']},
                   source_wait={'B': base['source'], 'SINGLE_HOLD': result['source']},
                   D_hold=base['J'] - result['J'], extra_sim_s=result['sim_s'],
                   collisions=base['collisions'] + result['collisions'],
                   teleports=base['teleports'] + result['teleports'],
                   extra_collisions=result['collisions'], extra_teleports=result['teleports'],
                   dropped_vehicle_steps=base.get('dropped_vehicle_steps', 0) + result.get('dropped_vehicle_steps', 0))
        return rec

    def write(self, branch_root: Path, points: list) -> dict:
        from .genrand import _jsonable
        branch_root = Path(branch_root)
        files = {"gate": "GATE_CALIBRATION.jsonl", "gated": "GATED_ARMS.jsonl",
                 "credit": "SUMOCREDIT_POINTS.jsonl"}
        counts = {}
        for key, name in files.items():
            rows = [p[key] for p in points if key in p]
            if key == "credit":
                rows += [p["credit_long"] for p in points if "credit_long" in p]
                rows += [p['single_long'] for p in points if 'single_long' in p]
            if not rows:
                continue
            with (branch_root / name).open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(_jsonable(row), allow_nan=False) + "\n")
            counts[key] = len(rows)
        return {"schema": "c4_sumocredit_extras/1", "options": asdict(self), "files": counts,
                "sim_s": int(sum(p["sim_s"] for p in points)),
                "wall_s": float(sum(p["wall_s"] for p in points))}
