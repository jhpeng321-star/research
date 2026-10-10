# WD-01 part 4 — total-delay accounting + Part B skeleton (REPORT)

Date 2026-09-26. Branch `zcode/wd01-delay`, worktree `D:\rd-wt\wd01c`.
**ZERO SIMULATION**: only read-only parsing of the Part A v0 NATIVE
dataset (`partA_data` npz), the regenerated runs' `requests.rou.xml`, and
the frozen network XML. No SUMO/TraCI anywhere.

Scope: (A) contract §4.1 total-delay library + validation on all 26
native roots; (B) Part B model code skeleton per DESIGN §3, synthetic
tests only (no real branch data exists; none used for training).

Inputs (authoritative copies, hash-checked):

| input | sha256 (first 16) |
| --- | --- |
| `D:/rd-wt/wd01/output/wd01/PREDICT_CONTRACT_v1.md` | `74a1de8edb52437b` |
| `D:/rd-wt/wd01/output/wd01/DESIGN_PARTS_A_B.md` | `35f2ab5f7b455ca8` |
| `partA_data/MANIFEST.json` | `2439004d528a860c` (matches the brief; also re-checked at runtime) |

## 1. Delay definitions (implemented conventions)

Section = the WM-30 physical corridor: main_left 3199.00 m, main_right
3197.55 m (31.29 m/s), accel 454.10 m (22.22 m/s ramp segment + 31.29 m/s
merge segment). Geometry is built from the network XML only
(`delay/geometry.py`); no SUMO imports.

Total delay (vehicle-seconds), per PREDICT_CONTRACT_v1.md §4.1:

```
total_delay = Σ_vehicles ( time_on_section − free_flow_time(traversed) )
            + wait_mainline + wait_ramp
```

over vehicles present at t0 or inserted in `[t0, t0+H]`; every term
truncated at `t0+H`. Components are computed per vehicle
(`delay/core.py`); the contract ships totals only, the breakdown is
internal/validation output.

* **Time on section** — interval-ending tick convention: a 0.5 s record
  certifies occupancy of the step ending at that tick. A present-at-t0
  vehicle contributes its records in `(t0, t0+H]`; an in-window insertee
  additionally drops its insertion record (a point event, not an
  interval). Exact for present-at-t0 vehicles and for insertees that
  survive to the horizon; residual boundary error ≤ 0.5 s elsewhere.
* **Free-flow time** — integrated per step as
  `advance / (lane speed limit × speedFactor)`:
  * `free_flow_ref` (contract default, vType-agnostic): speedFactor = 1.
  * `free_flow_oracle` (diagnostic only): the vType speedFactor from
    `requests.rou.xml` (aggressive 1.08 / normal 1.0 / cautious 0.92).
  * Advance is the longitudinal advance along the vehicle's path: raw
    `ds` within a lane; within-edge position difference across a lane
    change (chain-offset jumps removed); junction records span zero
    physical length (topology convention) so they bridge with ~0 advance
    but count their time; a lane change + chain-boundary crossing inside
    one step uses the chain-continuity decomposition (residual bounded by
    the 1.45 m chain-offset difference; counted in
    `n_boundary_change_steps`).
* **Entry waiting** — scheduled departure (`requests.rou.xml`,
  entrance from route) matched by id to the logged insertion record
  (`departed` births): `wait = clip(min(t_ins, t0+H) − max(t_dep, t0),
  0, H)`. Vehicles inserted ≤ t0 contribute 0 (already in); vehicles
  scheduled but un-inserted at t0+H contribute the full truncated wait
  (queued past the horizon). Insertees absent from the schedule are
  counted (`n_unmatched_sched_id`, expected 0) and contribute 0.
* **Queue membership**: vehicles still queued at the entrance at t0+H
  and vehicles queued at t0 (scheduled before, inserted after) are
  included in the waiting sum — see spec questions (S2).

## 2. Validation on the 26 native roots (truth delay labels)

Grid: `t0 ∈ {120, 150, …, 480}` (13 points), `H ∈ {60, 120, 300, 480}`
with `t0+H ≤ 600` → 34 windows per root (13+13+7+1), **884 windows**.

Command and wall-clock:

```
python delay/validate.py --out results          # 2 m 21 s wall-clock
```

Outputs: `results/delay_labels.parquet` (884 × 23 columns, one row per
window with the full component breakdown + diagnostics),
`results/delay_labels_summary.csv` (26 rows, per-root mean over its 34
windows), `results/delay_labels_by_loading.csv`,
`results/delay_validation_receipt.json`.

Consistency checks (per window, all 884): component-sum identity exact
(max |sum − total| = 0.0); every `wait ≥ 0` (min 0.0); waiting ==
min(t_ins, t0+H) − max(t_dep, t0) re-derived per matched id (0
mismatches); **n_unmatched_sched_id total = 0**; vehicles queued at the
horizon across all windows = 8163 (0.9/window average, up to 384 on the
densest root/window); MANIFEST npz sha256 verified per root.

Per-loading means (vehicle-seconds per window):

| loading | total (ref) | total (oracle) | travel (ref) | wait mainline | wait ramp | on-section | free-flow (ref) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2000 veh/h | 1979.5 | 1889.8 | 1566.4 | 67.0 | 346.0 | 10300.1 | 8733.6 |
| 2400 veh/h | 4596.8 | 4516.2 | 3712.2 | 512.6 | 372.0 | 14339.0 | 10626.8 |
| 2700 veh/h | 7701.5 | 7591.8 | 5403.3 | 2047.7 | 250.5 | 16150.0 | 10746.6 |
| (unspecified: cv17, cv19_*) | 2740.7 | 2664.3 | 2275.7 | 164.3 | 300.7 | 11397.1 | 9121.4 |

Delay grows steeply with loading, and the split moves from ramp give-way
waiting (2000) to mainline entrance queuing (2700). The oracle variant
reads 3–5 % lower than the contract default because the realized vType
mix drives slightly above the prior-mean speedFactor. Per-root numbers
are in `delay_labels_summary.csv` (train/holdout means: 4333 / 6187
veh·s per window).

Adapter cross-check (contract-layout vs truth adapter, root cv17,
t0=120, H=60, 1 s sampling): waiting components match exactly; on-section
5109.5 → 5085.0 s; travel delay 720.8 → 646.1 s; total 814.9 → 740.2 s
(−9.2 %, i.e. −0.6 s per vehicle, within the 1 s-per-vehicle sampling
bound asserted in `test_contract_adapter_cross_check`).

## 3. Part B skeleton (`partB/`)

Per DESIGN §3.2/§3.3 — synthetic-data tests only; NO training on real
branch data (the WM-36 campaign does not exist).

* `featurize.py` — end-of-Part-A state (contract trajectory layout,
  NaN-prefix/suffix rules respected) + plan + arrival draw → fixed-size
  numpy features: lane bin field `[3, 320, 3]` (count/mean/std of speed
  per 10 m bin), CAV tokens `[64, 17]` (s, lane, v, action, wall
  phase/speed/time-in-phase, member flag, remaining hold), wall tokens
  `[4, 6]`, demand `[21]` (per-minute counts per entrance over the
  remaining window + CAV share), context `[8]` (t0, A, branch context,
  regime, Part A delay/TTC stats, queues). Snapshots exclude vehicles
  not on the section at `t_end`. No vType anywhere (supervision-only).
* `model.py` — `DelayEstimator`: per-lane 1D conv encoder (shared
  weights) → 3 lane tokens, + CAV/wall/demand/context tokens, + token-type
  embeddings → 3-layer TransformerEncoder (d=192, ff=512, norm_first) →
  masked-mean pool → two heteroscedastic heads `(mean, log σ²)` for
  `R_native` and `ΔR`. **1,100,836 parameters** at the default width
  (within the 1–3 M design range). Device-agnostic: `device=` everywhere,
  no `.cuda()`; numpy or torch batches accepted.
* `losses.py` — Gaussian NLL on both targets, pairwise ranking hinge
  within a branch point (`relu(margin − sign(Δyᵢ−Δyⱼ)·(mᵢ−mⱼ))`,
  ties skipped), near-zero `ΔR` up-weighting `1 + exp(−(y/w)²)`,
  combined `partb_loss`.
* `ensemble.py` — `DelayEnsemble` (M=5 members, distinct seeds;
  forward `[M,B,4]`), `save_pretrained`/`from_pretrained` via plain
  `state_dict` files loaded with `torch.load(..., weights_only=True)`;
  `resolve_device()` = `WD_DEVICE` env, else cuda-if-available, else cpu.

## 4. Tests

```
python -m pytest tests/ -q      # 45 passed in ~8–10 s
```

* `test_delay_core.py` (17): hand-computed synthetic cases on a
  round-number geometry — present-at-t0, born mid-window, exit
  mid-window, still-present at horizon, queued speed, per-lane speed
  limits, lane change within edge (offset correction), junction steps,
  boundary-change steps, truncation beyond horizon, oracle speedFactor,
  gapped records, and the full waiting/totals bookkeeping (insertion
  after schedule, queued across t0, queued past horizon, already
  entered, unmatched insertee, beyond-horizon schedule).
* `test_delay_contract_io.py` (6): contract layout == direct traces
  (exact on a common grid), NaN prefix/suffix handling, non-trailing NaN
  rejected, tracked-must-start-at-t0, bad entrance label rejected, speed
  attribution.
* `test_delay_truth.py` (4, real data, read-only): schedule ⊇ logged ids,
  speedFactor table, per-window identities on cv17 across the grid,
  contract-adapter cross-check with the sampling bound.
* `test_partB.py` (16): shapes/dtypes, bin content, featurizer
  determinism, collate, forward shapes, default param count in 1–3 M,
  seed determinism (same seed → identical outputs, different seed →
  different), NLL decreases to the target, ranking-hinge sign/group
  semantics, combined loss differentiability, near-zero weighting,
  ensemble output/device, member diversity + reproducibility, save/load
  via `weights_only=True`, and a 40-step CPU overfit loop (toy synthetic
  task only).

## 5. Spec questions / documented decisions

* **S1**: the `wd01c` worktree's local `output/wd01/PREDICT_CONTRACT_v1.md`
  (9,861 B) differs from the authoritative copy in `wd01` (14,097 B,
  sha256 `74a1de8e…`); per the brief the authoritative copy was used
  (it contains the §8 v1 clarifications). The local copy was left
  untouched.
* **S2** §4.1 ambiguity: entrance queues. "Vehicles present at t0 or
  arriving in [t0, t0+H]" does not state whether (a) vehicles still
  waiting at an entrance at t0+H, or (b) vehicles queued at t0 whose
  scheduled entry predates t0, belong in the sum. We include both, with
  the wait truncated to the window (this is the only reading under which
  "time waiting to enter" can be nonzero for a demand draw that the
  sim cannot insert). Flag: the delay labels are monotone under this
  choice; excluding queued vehicles is a one-line change in
  `core.waiting_component`'s candidate rule.
* **S3** §4.1 gives no discretisation convention. We use the
  interval-ending tick convention above (max 0.5 s error per vehicle,
  zero for present-at-t0 and surviving insertees). Labels are produced
  at the 0.5 s physics step.
* **S4** "free-flow time on the traversed part of the section": we
  integrate per-step `advance / speed limit` over the physical path.
  Junction internal length is zero in the physical coordinate (the
  topology's documented convention), contributing zero advance and zero
  free-flow time while time-on-section still counts it.
* **S5** The arrivals draw rows carry no vehicle ids (§3.1), so the
  contract-layout adapter cannot know which queued draw row is which
  vehicle; queued waiting therefore requires the schedule
  (`{vid: (t_dep, entrance)}`) as an explicit input, and the
  already-entered set (`entered_before_t0`) for vehicles without
  trajectories in the window. Both are plain function inputs.
* **S6** Part B featurizer: the per-entrance queue split is not
  derivable from id-less draw rows; the skeleton puts the total queue on
  the mainline slot and expects the predictor's queue state via
  `ctx.queue_mainline/queue_ramp` (documented approximation).
* **S7** Heteroscedastic NLL has no variance floor in the loss; the
  ensemble clamps σ ≥ 1e-3 at prediction time. A variance prior can be
  added when real data exists.

## 6. Reproduction

```
cd D:
d-wt\wd01c\output\wd01\part4
C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe -m pytest tests/ -q
C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe delay/validate.py --out results
```

Env: torch 2.12.1+cpu, numpy 2.2.6, pandas 2.3.3, pyarrow 24.0.0
(`rl_driving` conda env). Inputs are read from
`D:/rd-wt/wd01/output/wd01/partA_data` (override: `WD01_PARTA_DATA`),
the frozen network at `D:/research-dense/assets/research/`
`highway_merge_shepherd_long/osm.net.xml.gz`, and the regenerated
runs' `requests.rou.xml` via `part1/results/tag_run_map.json`.
`results/` data files are git-ignored; code/tests/report are
force-added.

Code file hashes (sha256, first 12):

```
17c4f500e19f  delay/__init__.py
2b04a0bdc937  delay/contract_io.py
81b8356ce374  delay/core.py
331063e3720d  delay/geometry.py
157c1c9c481e  delay/truth.py
ca5d590ec8b8  delay/validate.py
91dd1d833bba  partB/__init__.py
a460024bc519  partB/ensemble.py
a37931a77c06  partB/featurize.py
41de73242d6a  partB/losses.py
6b12611b6057  partB/model.py
865595cbc31d  tests/conftest.py
cfb33fcf5cff  tests/test_delay_contract_io.py
1a9f4b654809  tests/test_delay_core.py
78f3a8cde555  tests/test_delay_truth.py
699adbdc024c  tests/test_partB.py
```
