# Metastable-Control Feasibility Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the two-stage feasibility probe (design: `docs/superpowers/specs/2026-08-12-metastable-feasibility-probe-design.md`) that answers whether ANY controller can hold the merge on the metastable upper branch without harming TTC vs champion B.

**Architecture:** Stage 1 (`ScriptedInterventionPolicy`) measures loop gain — whether CAV actions move h at all. Stage 2 (`MetastableRulePolicy`) is an h-table controller (hysteresis + TTC floor + smoothness budget) with ablations, inside the existing `Policy` ABC so the unmodified `run_policy_evaluation` harness drives it. A `HealthRecordingWrapper` records per-step (h, ρ, v) for any policy. A new driver `_probe_metastable_feasibility.py` clones the `_eval_hybrid_meta_v1.py` harness (spawn-timeout, no-op cache, row/summary writing) so every number is directly comparable to B.

**Tech Stack:** Python 3.10, SUMO/TraCI (Windows, Git Bash), pytest. No RLlib, no training.

**Key facts an implementer needs:**
- Policy interface (`policies/__init__.py`): `reset(env, seed=None)`, `act(obs_dict) -> {cav_id: action}`, `close()`; `direct_control` class attr.
- Hybrid actions are `(mode_index: int, dial: float)` tuples; `MODE_NAMES` in `config/constants.py:42` (`coast=0, fair_balance=1, recover_speed=2, vacate_drop=3, protect=4, ...`). Use `MODE_NAMES.index(...)` — never hardcode.
- `RAMP_EDGE = "11553048#1"` (`config/constants.py:87`); `env.snapshot.get_active_snapshot()` gives `snap.road(cav_id)`; `env.traci_query.leader_ttc(veh_id)` returns float or `+inf`.
- The env's `_merge_metastable_health` (env/core.py:1214-1314) computes h with EMA state on the env (`_met_health_rho_ema` etc.). It is called by the engine ONLY when `metastable_health_mask` config is on — the probe keeps that OFF so the EMA advances exactly once per step (via the new accessor from the policy).
- Health calibration: rho_free 2.0, rho_crit 5.5 veh/km/lane, v_free 25, v_break 12 m/s, var_ratio 3.0/6.0, baseline window 120.
- `run_policy_evaluation` (`evaluate_policy.py:181`) drives the episode and returns metrics with `metrics["ttc"]` (dict with `total_events`, or int — use `_ttc_events` pattern from `_eval_hybrid_meta_v1.py:87`).
- No-op cache shared at `runs/eval/noop_cache_{tag}/`. Do NOT run two evals of the same scene concurrently (shared SUMO tripinfo filenames). Per-seed timeout: spawn + `proc.join(timeout)` + `terminate()`.

---

### Task 1: Env accessor `metastable_health_detail` (refactor, behavior-preserving)

**Files:**
- Modify: `env/core.py:1214-1314`
- Test: `tests/test_metastable_rule.py` (new file, part 1)

- [ ] **Step 1: Write the failing accessor test**

Create `tests/test_metastable_rule.py`:

```python
"""Tests for the metastable-control feasibility probe (design spec
docs/superpowers/specs/2026-08-12-metastable-feasibility-probe-design.md).

Covers: env.metastable_health_detail accessor, the pure rule decision
function, the scripted-intervention policy, and the health-recording wrapper.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from config.constants import MODE_NAMES, RAMP_EDGE
from env.core import TrafficShepherdMultiAgentEnv

COAST = MODE_NAMES.index("coast")
FAIR = MODE_NAMES.index("fair_balance")
PROTECT = MODE_NAMES.index("protect")


def test_metastable_health_detail_accessor():
    env = TrafficShepherdMultiAgentEnv(
        {
            "sumo_binary": "sumo",
            "sumo_config": "osm.light.sumocfg",
            "cav_penetration": 0.20,
            "max_episode_steps": 30,
            "simulation_step": 1.0,
        }
    )
    obs, _ = env.reset(seed=123)
    detail = None
    for _ in range(5):
        obs, _, _, _, _ = env.step({cav_id: 0 for cav_id in obs.keys()})
        detail = env.metastable_health_detail()
        if detail is not None:
            break
    env.close()
    if detail is not None:
        h, rho, v_mean = detail
        assert 0.0 <= h <= 1.0
        assert rho > 0.0
        assert v_mean >= 0.0
        # consistent with the Experiment-C private wrapper (same snapshot)
        assert env._merge_metastable_health() == h
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metastable_rule.py::test_metastable_health_detail_accessor -v`
Expected: FAIL with `AttributeError: ... object has no attribute 'metastable_health_detail'`

- [ ] **Step 3: Implement the refactor**

In `env/core.py`, replace the body of `_merge_metastable_health` (lines 1214-1314) with a delegate, and add the new public method before it. Exact replacement:

```python
def metastable_health_detail(self) -> Optional[tuple]:
    """RSU merge-zone metastable health detail (feasibility probe).

    Public accessor: returns ``(h, rho_e, v_mean)`` — h in [0,1] (see
    ``_merge_metastable_health`` for the semantics), the merge-zone
    density EMA in veh/km/lane, and the merge-zone mean speed in m/s.
    Snapshot-only (zero extra TraCI calls); None when the snapshot is
    unavailable or the merge edge has no vehicles. The EMA state advances
    once per call — the probe calls this once per step and keeps the
    Experiment-C mask disabled, so the EMA advances exactly once per step.
    """
    snap = get_active_snapshot()
    if snap is None:
        return None
    try:
        merge_ids = [v for v in snap.ordered_ids() if snap.road(v) == MERGE_EDGE]
    except Exception:
        return None
    speeds = [snap.speed(v) for v in merge_ids if snap.speed(v) is not None]
    if not speeds:
        return None
    # Merge-edge total length is STATIC (network geometry never changes) —
    # cache it after the first call instead of re-querying every lane ID via
    # TraCI on every CAV step (this was the training slowdown: 20+ CAVs x
    # per-step getIDList+getLength). Snapshot-only stays true afterwards.
    if self._met_health_merge_len is None:
        try:
            self._met_health_merge_len = sum(
                traci.lane.getLength(l) for l in traci.lane.getIDList() if l.startswith(MERGE_EDGE)
            )
        except Exception:
            self._met_health_merge_len = 300.0
    mlen = self._met_health_merge_len
    rho = len(merge_ids) / max(mlen, 1e-6) * 1000.0 / MERGE_NLANES
    v_mean = float(mean(speeds))
    var = float(pvariance(speeds)) if len(speeds) > 1 else 0.0

    a = float(self.config.get("metastable_health_ema_alpha", 0.3))
    if self._met_health_rho_ema is None:
        self._met_health_rho_ema = rho
    else:
        self._met_health_rho_ema = (1 - a) * self._met_health_rho_ema + a * rho
    if self._met_health_var_ema is None:
        self._met_health_var_ema = var
    else:
        self._met_health_var_ema = (1 - a) * self._met_health_var_ema + a * var
    rho_e = self._met_health_rho_ema
    var_e = self._met_health_var_ema

    # Free-flow variance baseline: sample the var_ema over the first
    # ``var_baseline_window`` steps (steady window) and hold it — NOT a
    # running minimum. On the critical 2700 scene the merge-zone variance
    # is intrinsically high (mean ~66 even when healthy), so a soft-min
    # baseline captures a transient dip and makes every later step look
    # "about to collapse" (h stuck at ~0). A fixed early-window baseline
    # reflects the scene's normal operating variance instead.
    if self._met_health_var_base is None:
        self._met_health_var_base = var_e
    window = int(self.config.get("metastable_health_var_baseline_window", 120))
    if self._met_health_var_steps < window:
        self._met_health_var_base = 0.95 * self._met_health_var_base + 0.05 * var_e
    self._met_health_var_steps += 1

    rho_free = float(self.config.get("metastable_health_rho_free", 2.0))
    rho_crit = float(self.config.get("metastable_health_rho_crit", 5.5))
    v_free = float(self.config.get("metastable_health_v_free", 25.0))
    v_break = float(self.config.get("metastable_health_v_break", 12.0))
    var_lo = float(self.config.get("metastable_health_var_ratio_lo", 3.0))
    var_hi = float(self.config.get("metastable_health_var_ratio_hi", 6.0))

    rho_h = 1.0 - min(max((rho_e - rho_free) / max(rho_crit - rho_free, 1e-6), 0.0), 1.0)
    v_h = min(max((v_mean - v_break) / max(v_free - v_break, 1e-6), 0.0), 1.0)
    # Krauss 1997 §VI: speed variance rises in FREE FLOW shortly before
    # breakdown. The warning is only informative while the density is
    # still healthy (rho_h > 0.5) — once rho is already climbing toward
    # crit, the density term carries the signal and variance (intrinsically
    # high at critical load) adds nothing but noise.
    if rho_h > 0.5:
        var_ratio = var_e / max(self._met_health_var_base, 1e-3)
        var_h = 1.0 - min(max((var_ratio - var_lo) / max(var_hi - var_lo, 1e-6), 0.0), 1.0)
        return float(min(rho_h, v_h, var_h)), rho_e, v_mean
    return float(min(rho_h, v_h)), rho_e, v_mean


def _merge_metastable_health(self) -> Optional[float]:
    """RSU merge-zone metastable health h in [0,1] (Experiment C).

    Snapshot-only (zero extra TraCI calls): aggregate merge-edge vehicles
    into density (veh/km/lane), mean speed and speed VARIANCE, then map to
    health via three linear ramps AND-combined, following Krauß-Wagner-
    Gawron 1997 (Phys. Rev. E 55, 5597 — the metastable-state paper behind
    the SUMO Krauss model):

      rho ramp (§V hysteresis):  rho_e <= rho_free -> 1.0 (upper branch)
                                 rho_e >= rho_crit -> 0.0 (collapse)
      v   ramp (§V):             v_mean >= v_free  -> 1.0 (free flow)
                                 v_mean <= v_break -> 0.0 (crawling jam)
      var ramp (§VI pre-breakdown signal): speed variance rises ~160x
                                 (1.1 -> 177.8) in the ~30 s BEFORE the
                                 merge collapses, even while rho/v still
                                 look healthy — so a rising var_ratio
                                 (var_ema / free-flow baseline) drops h
                                 EARLY, opening metering before the crash.

    h = min(rho_h, v_h, var_h). 1 = deep metastable (free coast); falling
    h = merge sickening (density rising, speed dropping, or variance
    amplifying); 0 = collapse imminent (meter to restore). Calibration
    follows the measured window: rho_free 2.0, rho_crit 5.5 veh/km/lane,
    v_free 25, v_break 12 m/s (docs/metastable_merge_control.md §1.3).
    """
    detail = self.metastable_health_detail()
    return None if detail is None else detail[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metastable_rule.py::test_metastable_health_detail_accessor -v`
Expected: PASS

- [ ] **Step 5: Run the full suite (regression guard on the refactor)**

Run: `python -m pytest tests/ -q`
Expected: all pass (the mask tests exercise the same code path via `eng.met_health_fn`)

- [ ] **Step 6: Commit**

```bash
git add env/core.py tests/test_metastable_rule.py
git commit -m "probe: expose env.metastable_health_detail (h, rho, v_mean) — refactor of the exp-C health provider"
```

---

### Task 2: `policies/metastable_rule.py` — decision fn + two policies + wrapper

**Files:**
- Create: `policies/metastable_rule.py`
- Test: `tests/test_metastable_rule.py` (append part 2)

- [ ] **Step 1: Write the failing tests (append to tests/test_metastable_rule.py)**

```python
from policies.metastable_rule import (
    DEFAULT_RULE_CFG,
    HealthRecordingWrapper,
    MetastableRulePolicy,
    ScriptedInterventionPolicy,
    _new_state,
    metastable_rule_decision,
)


def _cfg(**over):
    c = dict(DEFAULT_RULE_CFG)
    c.update(over)
    return c


def _fresh():
    return _new_state()


# ---- pure decision function (smoothness off for exact dial assertions) ----


def test_decision_healthy_coasts():
    mode, dial, _ = metastable_rule_decision(
        _fresh(), 0.8, 3.0, "ramp", None, _cfg(smoothness_on=False)
    )
    assert (mode, dial) == (COAST, 0.0)


def test_decision_transition_band_paces_mainline_only():
    s = _fresh()
    m, d, _ = metastable_rule_decision(s, 0.5, 3.0, "mainline", None, _cfg(smoothness_on=False))
    assert (m, d) == (PROTECT, 0.2)
    m, d, _ = metastable_rule_decision(_fresh(), 0.5, 3.0, "ramp", None, _cfg(smoothness_on=False))
    assert (m, d) == (COAST, 0.0)


def test_decision_sick_meters_ramp_by_rho_excess():
    s = _fresh()
    m, d, _ = metastable_rule_decision(s, 0.2, 5.5, "ramp", None, _cfg(smoothness_on=False))
    assert m == FAIR
    assert abs(d - 0.3) < 1e-9  # dial at rho == rho_crit
    m2, d2, _ = metastable_rule_decision(
        _fresh(), 0.2, 6.5, "ramp", None, _cfg(smoothness_on=False)
    )
    assert m2 == FAIR
    assert abs(d2 - 0.7) < 1e-9  # saturates at rho_dial_hi


def test_decision_sick_low_rho_no_meter():
    m, _, _ = metastable_rule_decision(_fresh(), 0.2, 3.0, "ramp", None, _cfg(smoothness_on=False))
    assert m == COAST


def test_decision_sick_paces_mainline():
    m, d, _ = metastable_rule_decision(
        _fresh(), 0.2, 6.0, "mainline", None, _cfg(smoothness_on=False)
    )
    assert (m, d) == (PROTECT, 0.2)


def test_decision_hysteresis_release_needs_sustained_recovery():
    cfg = _cfg(smoothness_on=False)
    s = _fresh()
    # start metering
    m, _, s = metastable_rule_decision(s, 0.2, 6.0, "ramp", None, cfg)
    assert m == FAIR and s["metering"] is True
    # h recovered but only briefly: still metering
    for _ in range(4):
        m, _, s = metastable_rule_decision(s, 0.7, 6.0, "ramp", None, cfg)
        assert m == FAIR, "must hold metering during the release window"
    # 5th consecutive recovered step -> release
    m, _, s = metastable_rule_decision(s, 0.7, 6.0, "ramp", None, cfg)
    assert s["metering"] is False
    assert m == COAST


def test_decision_ttc_floor_downgrades_meter_to_protect():
    cfg = _cfg(smoothness_on=False)
    m, d, _ = metastable_rule_decision(_fresh(), 0.2, 6.0, "ramp", 1.5, cfg)
    assert (m, d) == (PROTECT, 0.2)
    # floor disabled -> meter again
    cfg2 = _cfg(smoothness_on=False, ttc_floor_on=False)
    m, _, _ = metastable_rule_decision(_fresh(), 0.2, 6.0, "ramp", 1.5, cfg2)
    assert m == FAIR


def test_decision_smoothness_caps_dial_rate():
    s = _fresh()
    s["prev_dial"] = 0.7
    s["prev_mode"] = FAIR
    cfg = _cfg()  # smoothness on
    m, d, _ = metastable_rule_decision(s, 0.5, 3.0, "mainline", None, cfg)  # target 0.2
    assert 0.65 <= d <= 0.7  # capped at 0.05/step


def test_decision_smoothness_min_dwell():
    s = _fresh()
    cfg = _cfg()  # smoothness on
    m, d, s = metastable_rule_decision(s, 0.8, 3.0, "ramp", None, cfg)  # coast
    assert (m, d) == (COAST, 0.0)
    m, _, s = metastable_rule_decision(s, 0.2, 6.0, "ramp", None, cfg)  # wants meter
    assert m == COAST, "dwell: mode must be held >= min_dwell steps"
    m, _, s = metastable_rule_decision(s, 0.2, 6.0, "ramp", None, cfg)
    assert m == COAST
    m, _, s = metastable_rule_decision(s, 0.2, 6.0, "ramp", None, cfg)
    assert m == FAIR, "dwell satisfied after 3 steps"


# ---- policy-level wiring (mock env, no SUMO) ----


class _FakeEnv:
    def __init__(self, detail=(0.2, 6.0, 8.0)):
        self._detail = detail

    def metastable_health_detail(self):
        return self._detail


def test_rule_policy_meters_ramp_paces_mainline():
    env = _FakeEnv((0.2, 6.0, 8.0))
    snap = MagicMock()
    snap.road.side_effect = lambda v: RAMP_EDGE if v == "cav_r" else "some_edge"
    with (
        patch("policies.metastable_rule.get_active_snapshot", return_value=snap),
        patch("policies.metastable_rule.leader_ttc", return_value=float("inf")),
    ):
        pol = MetastableRulePolicy()
        pol.reset(env)
        acts = pol.act({"cav_r": np.zeros(10), "cav_m": np.zeros(10)})
    assert acts["cav_r"][0] == FAIR
    assert acts["cav_m"][0] == PROTECT


def test_scripted_burst_triggers_on_low_h_and_cycles():
    pol = ScriptedInterventionPolicy(FAIR, 0.6, on_steps=2, off_steps=2, trigger_h=0.4)
    pol.reset(_FakeEnv((0.2, 6.0, 8.0)))
    a1 = pol.act({"cav0": np.zeros(4)})
    assert a1["cav0"] == (FAIR, 0.6), "trigger fires immediately"
    a2 = pol.act({"cav0": np.zeros(4)})
    assert a2["cav0"] == (FAIR, 0.6), "second on-step"
    a3 = pol.act({"cav0": np.zeros(4)})
    assert a3["cav0"] == (COAST, 0.0), "burst exhausted -> coast"
    a6 = pol.act({"cav0": np.zeros(4)})
    a6 = pol.act({"cav0": np.zeros(4)})
    a6 = pol.act({"cav0": np.zeros(4)})
    assert a6["cav0"] == (FAIR, 0.6), "re-triggers after off window"


def test_scripted_continuous_always_on():
    pol = ScriptedInterventionPolicy(PROTECT, 0.2, trigger_h=None)
    pol.reset(_FakeEnv((0.8, 2.0, 25.0)))
    a = pol.act({"cav0": np.zeros(4)})
    assert a["cav0"] == (PROTECT, 0.2)


def test_recording_wrapper_records_health_per_step():
    inner = MagicMock()
    inner.direct_control = False
    inner.act.return_value = {"cav0": 0}
    env = _FakeEnv((0.6, 3.0, 20.0))
    w = HealthRecordingWrapper(inner)
    w.reset(env)
    w.act({"cav0": np.zeros(4)})
    w.act({"cav0": np.zeros(4)})
    assert len(w.history) == 2
    assert w.history[0] == (0.6, 3.0, 20.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metastable_rule.py -q -k "decision or rule_policy or scripted or recording"`
Expected: FAIL with `ModuleNotFoundError: No module named 'policies.metastable_rule'`

- [ ] **Step 3: Implement `policies/metastable_rule.py`**

```python
"""Metastable-control feasibility probe policies (design spec 2026-08-12).

  - `metastable_rule_decision`   — pure h-table decision (unit-testable)
  - `MetastableRulePolicy`       — Stage-2 h-table controller (+ ablations)
  - `ScriptedInterventionPolicy` — Stage-1 loop-gain scripted interventions
  - `HealthRecordingWrapper`     — records per-step (h, rho, v_mean) for ANY
                                   policy (incl. no_op) via the env accessor

Stage-2 rule table (per CAV, role-aware):

  h >= h_coast              -> coast (upper branch, free flow)
  h in [h_pace_lo, h_coast) -> mainline paces (protect, gentle dial), ramp flows
  h <  h_pace_lo            -> ramp meters (fair_balance, dial by rho excess)
                               only if rho >= rho_crit; mainline paces

Constraints: metering releases only after h >= h_release for
h_release_steps consecutive steps (KWG upper/lower branch hysteresis); leader
TTC < ttc_floor downgrades meter -> protect (never meter into a near-miss);
dial rate capped at dial_rate_cap/step with min_dwell mode dwell.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from config.constants import MODE_NAMES, RAMP_EDGE
from env.snapshot import get_active_snapshot
from env.traci_query import leader_ttc
from policies import Policy

COAST = MODE_NAMES.index("coast")
FAIR = MODE_NAMES.index("fair_balance")
PROTECT = MODE_NAMES.index("protect")

DEFAULT_RULE_CFG = {
    "h_coast": 0.6,  # h >= this -> coast
    "h_pace_lo": 0.35,  # h in [h_pace_lo, h_coast) -> pace mainline
    "h_release": 0.55,  # metering releases only after sustained h >= this
    "h_release_steps": 5,
    "rho_crit": 5.5,  # metering density gate (veh/km/lane, h calibration)
    "rho_dial_hi": 6.5,  # rho where the metering dial saturates at its max
    "pace_dial": 0.2,  # protect dial used for pacing (gentle)
    "meter_dial_lo": 0.3,  # dial at rho == rho_crit
    "meter_dial_hi": 0.7,  # dial at rho == rho_dial_hi
    "meter_dial_fixed": None,  # ablation: force a fixed metering dial
    "ttc_floor": 1.8,  # leader TTC below this -> meter downgraded
    "ttc_floor_on": True,
    "smoothness_on": True,  # ablation: cap dial rate + enforce mode dwell
    "dial_rate_cap": 0.05,
    "min_dwell": 3,
}


def _new_state() -> dict:
    return {
        "metering": False,
        "release_count": 0,
        "prev_mode": COAST,
        "prev_dial": 0.0,
        "dwell": 0,
    }


def metastable_rule_decision(
    state: dict,
    h: Optional[float],
    rho: Optional[float],
    role: str,
    ttc: Optional[float],
    cfg: dict,
) -> tuple:
    """h-table decision: returns (mode_index, dial, state').

    ``role`` is "ramp" (on the ramp approach edge) or "mainline". ``state``
    carries the hysteresis / dwell / dial-rate memory and is mutated in place.
    """
    if h is None:
        return COAST, 0.0, state

    # --- TTC floor: never meter into a near-miss (per-CAV downgrade) ---
    floor = cfg.get("ttc_floor")
    if cfg.get("ttc_floor_on", True) and floor is not None and ttc is not None and ttc < floor:
        return PROTECT, cfg.get("pace_dial", 0.2), state

    # --- hysteresis: release only after sustained recovery ---
    if state["metering"]:
        if h >= cfg.get("h_release", 0.55):
            state["release_count"] += 1
            if state["release_count"] >= cfg.get("h_release_steps", 5):
                state["metering"] = False
                state["release_count"] = 0
        else:
            state["release_count"] = 0

    h_coast = cfg.get("h_coast", 0.6)
    h_pace_lo = cfg.get("h_pace_lo", 0.35)
    sick = h < h_pace_lo
    meter_ok = sick and role == "ramp" and rho is not None and rho >= cfg.get("rho_crit", 5.5)
    fixed = cfg.get("meter_dial_fixed")
    if fixed is not None:
        meter_target = float(fixed)
    else:
        span = max(cfg.get("rho_dial_hi", 6.5) - cfg.get("rho_crit", 5.5), 1e-6)
        frac = (
            min(max((rho - cfg.get("rho_crit", 5.5)) / span, 0.0), 1.0) if rho is not None else 0.0
        )
        meter_target = cfg.get("meter_dial_lo", 0.3) + frac * (
            cfg.get("meter_dial_hi", 0.7) - cfg.get("meter_dial_lo", 0.3)
        )

    if state["metering"]:
        # release hold: keep the regime until the sustained-recovery counter
        if role == "mainline":
            mode, target = PROTECT, cfg.get("pace_dial", 0.2)
        elif rho is not None and rho >= cfg.get("rho_crit", 5.5):
            mode, target = FAIR, meter_target
        else:
            mode, target = COAST, 0.0
    elif meter_ok:
        mode, target = FAIR, meter_target
        state["metering"] = True
        state["release_count"] = 0
    elif h >= h_coast:
        mode, target = COAST, 0.0
    elif role == "mainline":
        mode, target = PROTECT, cfg.get("pace_dial", 0.2)
    else:  # ramp in the transition band: let it flow
        mode, target = COAST, 0.0

    # --- smoothness: dial rate cap + min mode dwell ---
    if cfg.get("smoothness_on", True):
        cap = cfg.get("dial_rate_cap", 0.05)
        prev = state.get("prev_dial", 0.0)
        dial = min(max(target, prev - cap), prev + cap)
        if mode != state.get("prev_mode"):
            # a mode switch is allowed only after the current mode has been
            # held for min_dwell steps (count grows every step regardless)
            if state.get("dwell", 0) < cfg.get("min_dwell", 3):
                mode = state["prev_mode"]
                dial = prev
            else:
                state["dwell"] = 0
        state["dwell"] = state.get("dwell", 0) + 1
    else:
        dial = target
    state["prev_mode"] = mode
    state["prev_dial"] = dial
    return mode, dial, state


class MetastableRulePolicy(Policy):
    """Stage-2 h-table controller over the (mode, dial) action space.

    Reads h/rho/v_mean from the env's public health accessor once per step
    and applies `metastable_rule_decision` per CAV (ramp approach edge
    meters, mainline paces). Ablations:
      "full"       — hysteresis + TTC floor + smoothness budget
      "no_floor"   — TTC floor disabled
      "aggressive" — no TTC floor, no smoothness, fixed full metering dial
    """

    ABLATIONS = {
        "full": {},
        "no_floor": {"ttc_floor_on": False},
        "aggressive": {
            "ttc_floor_on": False,
            "smoothness_on": False,
            "meter_dial_fixed": 1.0,
        },
    }

    def __init__(self, ablation: str = "full"):
        cfg = dict(DEFAULT_RULE_CFG)
        cfg.update(self.ABLATIONS.get(ablation, {}))
        self.cfg = cfg
        self._env = None
        self._state = _new_state()

    def reset(self, env, seed: int = None) -> None:
        self._env = env
        self._state = _new_state()

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, tuple]:
        detail = self._env.metastable_health_detail() if self._env is not None else None
        h = detail[0] if detail else None
        rho = detail[1] if detail else None
        snap = get_active_snapshot()
        actions: Dict[str, tuple] = {}
        for cav_id in obs_dict.keys():
            role = "ramp" if snap is not None and snap.road(cav_id) == RAMP_EDGE else "mainline"
            ttc = None
            if self.cfg.get("ttc_floor_on"):
                try:
                    ttc = leader_ttc(cav_id)
                except Exception:  # noqa: BLE001 - transient TraCI gap, act blind
                    ttc = None
            mode, dial, self._state = metastable_rule_decision(
                self._state, h, rho, role, ttc, self.cfg
            )
            actions[cav_id] = (mode, dial)
        return actions


class ScriptedInterventionPolicy(Policy):
    """Stage-1 loop-gain policy: scripted h-triggered interventions, all CAVs.

    With ``trigger_h=None`` the mode is forced every step (continuous variant,
    e.g. pace). Otherwise the mode is forced for ``on_steps`` whenever
    h < ``trigger_h`` (and the off-window elapsed), then coast for
    ``off_steps``, then re-arms.
    """

    def __init__(
        self,
        mode: int,
        dial: float,
        on_steps: int = 10,
        off_steps: int = 5,
        trigger_h: Optional[float] = 0.4,
    ):
        self.mode = int(mode)
        self.dial = float(dial)
        self.on_steps = int(on_steps)
        self.off_steps = int(off_steps)
        self.trigger_h = trigger_h
        self._env = None
        self._on = False
        self._timer = 0
        self._off_timer = 0

    def reset(self, env, seed: int = None) -> None:
        self._env = env
        self._on = False
        self._timer = 0
        self._off_timer = 0

    def _health(self) -> Optional[float]:
        if self._env is None:
            return None
        d = self._env.metastable_health_detail()
        return None if d is None else d[0]

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, tuple]:
        h = self._health()
        if self.trigger_h is not None:
            if self._on:
                self._timer -= 1
                if self._timer <= 0:
                    self._on = False
                    self._off_timer = self.off_steps
            elif h is not None and h < self.trigger_h and self._off_timer <= 0:
                self._on = True
                self._timer = self.on_steps
            elif self._off_timer > 0:
                self._off_timer -= 1
        take = self.trigger_h is None or self._on
        action = (self.mode, self.dial) if take else (COAST, 0.0)
        return {cav_id: action for cav_id in obs_dict.keys()}


class HealthRecordingWrapper(Policy):
    """Wrap any policy; record per-step (h, rho, v_mean) for probe rows.

    Uses the env's public `metastable_health_detail()` accessor once per step.
    The probe never enables the Experiment-C mask, so the EMA advances exactly
    once per step and the recorded series is comparable across variants.
    """

    def __init__(self, inner: Policy):
        self.inner = inner
        self.direct_control = bool(getattr(inner, "direct_control", False))
        self.history: list = []
        self._env = None

    def reset(self, env, seed: int = None) -> None:
        self._env = env
        self.history = []
        self.inner.reset(env, seed=seed)

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        detail = self._env.metastable_health_detail() if self._env is not None else None
        if detail is not None:
            self.history.append((float(detail[0]), float(detail[1]), float(detail[2])))
        else:
            self.history.append((None, None, None))
        return self.inner.act(obs_dict)

    def close(self) -> None:
        self.inner.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metastable_rule.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add policies/metastable_rule.py tests/test_metastable_rule.py
git commit -m "probe: h-table rule policy + scripted intervention + health-recording wrapper (feasibility probe stages 1-2)"
```

---

### Task 3: `_probe_metastable_feasibility.py` — probe driver

**Files:**
- Create: `_probe_metastable_feasibility.py`
- Smoke: run stage 1 on 1 seed, short steps

- [ ] **Step 1: Implement the driver**

```python
"""Metastable-control feasibility probe driver (design spec 2026-08-12).

Stage 1 (loop gain):  no_op | meter_burst | pace | meter_aggressive
    - does the CAV's actuator move h at all on the 2700 scene?

Stage 2 (sufficiency): no_op | rule_full | rule_no_floor | rule_aggressive
    - can the h-table controller hold the upper branch without hurting TTC
      relative to champion B (iter_95)?

Reuses the _eval_hybrid_meta_v1 harness pattern: shared no-op cache, per-seed
spawn timeout, row/summary writing — so every number is directly comparable
to B's existing runs/eval summaries.

Usage:
  python -u _probe_metastable_feasibility.py --stage 1 --seeds 105 --workers 1
  python -u _probe_metastable_feasibility.py --stage 2 --scene 2700 --workers 1
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

SEED_TIMEOUT_S = int(os.environ.get("EVAL_SEED_TIMEOUT", "900"))

ROOT = Path(r"D:\research-dense")
SUMO = Path(r"D:\research\highway_merge_shepherd")
PEN = 0.2
SUMO_DT = 0.5
SEEDS = [115, 112, 114, 105, 119, 110, 126, 149]
PRE = "1191554636"
RAMP = "11553048#1"
SLACK_L = 5.0
SLACK_G = 5.0
B_SUMMARY = ROOT / "runs" / "eval" / "hybrid_meta_iter_95_light_m2700r400" / "summary.json"

DENSITIES = {
    2000: "light_l2000",
    2400: "light_m2400r400",
    2700: "light_m2700r400",
}

# stage -> variant -> (policy_kind, params)
STAGE1_VARIANTS = {
    "no_op": ("no_op", {}),
    "meter_burst": ("scripted", {"mode": "fair_balance", "dial": 0.6,
                                 "on_steps": 10, "off_steps": 5, "trigger_h": 0.4}),
    "pace": ("scripted", {"mode": "protect", "dial": 0.2, "trigger_h": None}),
    "meter_aggressive": ("scripted", {"mode": "fair_balance", "dial": 1.0,
                                      "on_steps": 10, "off_steps": 5, "trigger_h": 0.4}),
}
STAGE2_VARIANTS = {
    "no_op": ("no_op", {}),
    "rule_full": ("rule", {"ablation": "full"}),
    "rule_no_floor": ("rule", {"ablation": "no_floor"}),
    "rule_aggressive": ("rule", {"ablation": "aggressive"}),
}


def _env() -> None:
    os.environ.setdefault("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
    os.environ["PATH"] = os.environ["SUMO_HOME"] + r"\bin;" + os.environ.get("PATH", "")
    os.environ["PYTHONPATH"] = str(ROOT) + ";" + os.environ.get("PATH", "")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _stream_loss(tripinfo: Path) -> dict:
    all_tt, all_loss, main_l, ramp_l = [], [], [], []
    for t in ET.parse(tripinfo).getroot().findall("tripinfo"):
        edge = t.get("departLane", "").rsplit("_", 1)[0]
        dur = float(t.get("duration", 0))
        loss = float(t.get("timeLoss", 0))
        all_tt.append(dur)
        all_loss.append(loss)
        if edge == RAMP:
            ramp_l.append(loss)
        elif edge == PRE:
            main_l.append(loss)
    return {
        "tt": mean(all_tt) if all_tt else None,
        "loss": mean(all_loss) if all_loss else None,
        "loss_main": mean(main_l) if main_l else None,
        "loss_ramp": mean(ramp_l) if ramp_l else None,
        "fair_gap": abs(mean(main_l) - mean(ramp_l)) if main_l and ramp_l else None,
    }


def _ttc_events(metrics: dict) -> int:
    ttc = metrics.get("ttc")
    if isinstance(ttc, dict):
        return int(ttc.get("total_events") or 0)
    return int(ttc or 0)


def _health_metrics(history: list) -> dict:
    """h-mean / upper-branch fraction / first-collapse step from recorded (h,rho,v)."""
    hs = [h for h, _, _ in history if h is not None]
    if not hs:
        return {"h_mean": None, "upper_frac": None, "first_collapse_step": None}
    upper = sum(1 for h in hs if h >= 0.6) / len(hs)
    collapse = next((i for i, h in enumerate(hs) if h < 0.2), None)
    return {"h_mean": float(mean(hs)), "upper_frac": upper,
            "first_collapse_step": collapse}


def _build_policy(kind: str, params: dict):
    """Return (HealthRecordingWrapper(policy), policy_name)."""
    from policies.baselines import NoOpPolicy
    from policies.metastable_rule import (
        HealthRecordingWrapper,
        MetastableRulePolicy,
        ScriptedInterventionPolicy,
    )
    from config.constants import MODE_NAMES

    if kind == "no_op":
        inner = NoOpPolicy()
        name = "no_op"
    elif kind == "scripted":
        mode = MODE_NAMES.index(params["mode"])
        inner = ScriptedInterventionPolicy(
            mode, params["dial"],
            on_steps=params.get("on_steps", 10),
            off_steps=params.get("off_steps", 5),
            trigger_h=params.get("trigger_h", 0.4),
        )
        name = "shepherd"
    else:  # rule
        inner = MetastableRulePolicy(ablation=params.get("ablation", "full"))
        name = "shepherd"
    return HealthRecordingWrapper(inner), name


def run_variant(variant: str, seed: int, out_dir: Path, noop_cache: Path,
                tag: str, steps: int) -> dict:
    _env()
    from evaluate_policy import run_policy_evaluation

    # no_op is a pure-HDV baseline, independent of the variant: cache it in a
    # SHARED directory so every stage/variant eval reuses the same runs.
    if variant == "no_op":
        out_path = noop_cache / f"no_op.seed{seed}.json"
    else:
        out_path = out_dir / f"{variant}.seed{seed}.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))

    t0 = time.time()
    policy, policy_name = _build_policy(*STAGE1_VARIANTS.get(
        variant, STAGE2_VARIANTS.get(variant, ("no_op", {}))))
    env_overrides = {
        "simulation_step": 1.0,
        "sumo_step_length": SUMO_DT,
        "sumo_config": f"_light/osm.dense.{tag}.sumocfg",
        "shepherd_disabled_modes": ["vacate_drop"],
    }
    if policy_name == "shepherd":
        env_overrides["control_mode"] = "shepherd_hybrid"

    metrics = run_policy_evaluation(
        policy,
        density="dense",
        project_root=SUMO,
        seed=seed,
        max_episode_steps=None if steps <= 0 else steps,
        sumo_binary="sumo",
        cav_penetration=PEN,
        policy_name=policy_name,
        env_overrides=env_overrides,
        run_tag_prefix=f"metafeas_{tag}_{variant}_{seed}",
    )
    stream = _stream_loss(Path(metrics["tripinfo_path"]))
    row = {
        "variant": variant, "seed": seed, "tag": tag,
        "tt": stream["tt"], "loss": stream["loss"],
        "loss_main": stream["loss_main"], "loss_ramp": stream["loss_ramp"],
        "fair_gap": stream["fair_gap"],
        "collisions": int(metrics.get("collisions") or 0),
        "ttc_events": _ttc_events(metrics),
        "steps": int(metrics.get("steps", 0)),
        "elapsed_s": round(time.time() - t0, 1),
        **_health_metrics(policy.history),
    }
    out_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"[{variant} seed{seed} @{tag}] loss={row['loss']:.2f} "
          f"G={row['fair_gap']:.2f} ttc={row['ttc_events']} "
          f"h_mean={row['h_mean'] if row['h_mean'] is not None else float('nan'):.3f} "
          f"upper={row['upper_frac'] if row['upper_frac'] is not None else float('nan'):.2f} "
          f"coll={row['first_collapse_step']} ({row['elapsed_s']}s)", flush=True)
    return row


def _job_entry(q: mp.Queue, variant: str, seed: int, out_dir: Path,
               noop_cache: Path, tag: str, steps: int) -> None:
    try:
        q.put(run_variant(variant, seed, out_dir, noop_cache, tag, steps))
    except Exception as exc:  # noqa: BLE001 - report and fail the job, keep going
        print(f"FAIL {variant} seed{seed}: {exc}", flush=True)
        q.put({"variant": variant, "seed": seed, "status": "error", "error": str(exc)})


def _run_job_with_timeout(variant: str, seed: int, out_dir: Path,
                          noop_cache: Path, tag: str, steps: int,
                          timeout: int = SEED_TIMEOUT_S) -> dict:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(
        target=_job_entry,
        args=(q, variant, seed, out_dir, noop_cache, tag, steps),
        daemon=True,
    )
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        print(f"TIMEOUT {variant} seed{seed} after {timeout}s — killed", flush=True)
        return {"variant": variant, "seed": seed, "status": "timeout"}
    try:
        row = q.get_nowait()
    except Exception:  # noqa: BLE001
        row = None
    if row is None:
        return {"variant": variant, "seed": seed, "status": "error"}
    return row


def _print_health_table(rows: dict, variants: list, seeds: list, noop: dict) -> None:
    print("\n--- health vs no_op ---", flush=True)
    for v in variants:
        if v == "no_op":
            continue
        d_upper, d_h, shifts = [], [], []
        for s in seeds:
            r = rows.get((v, s))
            n = noop.get(s)
            if not r or not n or "status" in r or "status" in n:
                continue
            if r["upper_frac"] is not None and n["upper_frac"] is not None:
                d_upper.append(r["upper_frac"] - n["upper_frac"])
            if r["h_mean"] is not None and n["h_mean"] is not None:
                d_h.append(r["h_mean"] - n["h_mean"])
            if r["first_collapse_step"] is not None and n["first_collapse_step"] is not None:
                shifts.append(r["first_collapse_step"] - n["first_collapse_step"])
        du = mean(d_upper) if d_upper else float("nan")
        dh = mean(d_h) if d_h else float("nan")
        print(f"{v:16s}: dUpper={du:+.3f} dH={dh:+.3f} collapse_shifts={shifts}", flush=True)


def _print_win_table(rows: dict, variants: list, seeds: list, noop: dict) -> None:
    print("\n--- win rule vs no_op (TTC<=noop, G<=noop+5, L<=noop+5) ---", flush=True)
    for v in variants:
        if v == "no_op":
            continue
        per = []
        for s in seeds:
            r, n = rows.get((v, s)), noop.get(s)
            if not r or not n or "status" in r or "status" in n:
                continue
            ttc_ok = r["ttc_events"] <= n["ttc_events"]
            g_ok = (r["fair_gap"] or 0) <= (n["fair_gap"] or 0) + SLACK_G
            l_ok = (r["loss"] or 0) <= (n["loss"] or 0) + SLACK_L
            per.append({"seed": s, "win": ttc_ok and g_ok and l_ok,
                        "d_ttc": r["ttc_events"] - n["ttc_events"]})
        if not per:
            continue
        wins = sum(1 for x in per if x["win"])
        dttc = mean(x["d_ttc"] for x in per)
        print(f"{v:16s}: wins {wins}/{len(per)}  dTTC={dttc:+.1f}  "
              f"{' '.join(f'{x[\"seed\"]}:{x[\"d_ttc\"]:+d}' for x in per)}", flush=True)


def _print_ttc_vs_b(rows: dict, seeds: list, tag: str) -> None:
    """Stage-2 PASS check: per-seed TTC(rule) <= TTC(B iter_95)."""
    if tag != "light_m2700r400" or not B_SUMMARY.exists():
        print("\n(B summary not present for this scene — skipping TTC-vs-B check)", flush=True)
        return
    b = json.loads(B_SUMMARY.read_text(encoding="utf-8"))
    print("\n--- TTC vs B (iter_95, PASS if rule <= B per seed) ---", flush=True)
    for v in ("rule_full", "rule_no_floor", "rule_aggressive"):
        line = []
        for s in seeds:
            r = rows.get((v, s))
            bk = b.get(f"always_active_{s}")
            if not r or "status" in r or not bk:
                continue
            b_ttc = int(bk.get("ttc_events") or 0)
            ok = r["ttc_events"] <= b_ttc
            line.append(f"{s}:{'OK' if ok else 'NO'} ({r['ttc_events']} vs {b_ttc})")
        print(f"{v:16s}: " + " ".join(line), flush=True)


def main() -> None:
    _env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    ap.add_argument("--scene", type=int, default=2700, choices=sorted(DENSITIES))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("EVAL_WORKERS", "1")))
    ap.add_argument("--steps", type=int, default=0,
                    help="Episode length cap in controller steps (0 = full episode, "
                    "like the B evals; pass --steps 600 for Stage-1 runs)")
    ap.add_argument("--variants", default="")
    args = ap.parse_args()

    variants = (args.variants.split(",") if args.variants
                else list(STAGE1_VARIANTS if args.stage == 1 else STAGE2_VARIANTS))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tag = DENSITIES[args.scene]
    out = ROOT / "runs" / "probe" / "metastable_feasibility" / f"stage{args.stage}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    noop_cache = ROOT / "runs" / "eval" / f"noop_cache_{tag}"
    noop_cache.mkdir(parents=True, exist_ok=True)

    jobs = [(v, s) for v in variants for s in seeds]
    rows: dict = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_run_job_with_timeout, v, s, out, noop_cache, tag, args.steps): (v, s)
            for v, s in jobs
        }
        for fut in as_completed(futs):
            v, s = futs[fut]
            rows[(v, s)] = fut.result()

    noop = {s: rows[("no_op", s)] for s in seeds if ("no_op", s) in rows}
    _print_win_table(rows, variants, seeds, noop)
    _print_health_table(rows, variants, seeds, noop)
    if args.stage == 2:
        _print_ttc_vs_b(rows, seeds, tag)

    (out / "summary.json").write_text(
        json.dumps({f"{v}_{s}": rows[(v, s)] for v, s in rows}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run stage 1 on one seed, short episode**

Run: `python -u _probe_metastable_feasibility.py --stage 1 --seeds 105 --workers 1 --steps 120`
Expected: `no_op` + `meter_burst` rows print with h_mean/upper/coll fields, `Wrote ... stage1_light_m2700r400/summary.json`. (no_op comes from the shared cache if present, else runs.)

- [ ] **Step 3: Commit**

```bash
git add _probe_metastable_feasibility.py
git commit -m "probe: metastable feasibility driver (stage1 scripted loop-gain, stage2 h-table + ablations, no-op cache, per-seed timeout)"
```

---

### Task 4: Verification

- [ ] **Step 1: Full test suite**

Run: `python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 2: Stage-1 full smoke (3 seeds, 200 steps, workers 1)**

Run: `python -u _probe_metastable_feasibility.py --stage 1 --seeds 105,112 --workers 1 --steps 200`
Expected: all 4 variants × 2 seeds complete; win table + health table print with no exceptions.

- [ ] **Step 3: Stage-2 single-seed sanity (workers 1)**

Run: `python -u _probe_metastable_feasibility.py --stage 2 --seeds 105 --workers 1 --steps 200`
Expected: `rule_full` etc. rows print; TTC-vs-B check line prints (B summary exists for 2700).

- [ ] **Step 4: Report to user with the go/no-go summary of what to run next**

Full Stage 1 (8 seeds × 600 steps) and, if the lever is real, Stage 2 (8 seeds × {2700, 2400} × 3 rule variants, full episodes, workers 1, sequential — never two evals of the same scene concurrently).
