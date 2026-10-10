# REGISTRATION_WM_N1 — AMENDMENT 5 (RULE NOTE, PRE-CENSUS): choose the vehicle-slot fix by a fixed rule

**Status:** RULE FIXED, 2026-10-01, world model session. Owner-approved
approach, relayed by main discussion. Committed BEFORE any drop-location
census output exists or is read. A5 is frozen only after the rule below has
been applied to the census, with the result appended in §5; §1–§4 do not
change.

- Amends `REGISTRATION_WM_N1.md` (`666233aa…`) and amendments 1–4
  (`decb9656…`, `56fa99a0…`, `a597d986…`, `d392b300…`).
- **UNCHANGED:**
  - §4 targets and decision rule;
  - chunk-5 TEST sealed (`f8c7d760…`);
  - HOLDOUT sealed;
  - CAL split `bc4e32f2…`;
  - variants and tie order.
- **Background:** at K_max = 128 the featurizer drops 5.9998 % of rostered
  vehicle-ticks (`SLOT_POOL_CENSUS.md`, pack_v2c). By loading: 0 % at 2000,
  1.89 % at 2400, 5.25 % at 2600, 7.75 % at 2700. Option C (keep 128 and
  disclose) is therefore rejected, and only B or K = 192 remain
  (`KMAX_OPTIONS.md`).

## 1. Census definition (read-only)

- **Population:** featurizable TRAIN + CAL runs of pack_v2c.
- **Sample:** to keep the census to about 30 min, a fixed stratified random
  sample of **300 runs**:
  - strata are batch × loading × penetration;
  - allocation is proportional, with at least 1 run per non-empty stratum;
  - the draw uses seed **20261001**;
  - the sampled run list is written before any run is parsed.
- **Method:** replay the frozen featurizer's slot-assignment loop
  (`features/featurizer.py`, K = 128, persistent pool with reuse) on each
  sampled run's raw source.
  - Each vehicle-tick that the loop drops ("> 128 concurrent") is recorded
    with its `(lane_phys, s_down)` from `features/geometry.s_down_of`.
  - Every rostered vehicle-tick is classified the same way.
  - Counts and positions only. No outcome, label or summary file is read.
  - HOLDOUT, excluded and chunk-5 runs are refused.
- **Descriptive only (not rule inputs):**
  - the concurrent-roster count distribution (p50, p99, max) per loading;
  - the share that would still be dropped at K = 192;
  - a run-cluster bootstrap 95 % CI for both rule quantities.

## 2. Near-merge zone (the registered RSU merge + ramp zone)

This is the zone used by recon_v6 / the hybrid state and RSU emulation, in
featurizer `s_down` coordinates (merge start s_road 2644.2 ≈ s_down 0):

- **ramp:** every vehicle-tick with `lane_phys == 2` (the accel/ramp chain),
  i.e. the whole on-ramp lane (378.66 m) and the accel lane;
- **merge edge:** mainline lanes (`lane_phys` 0, 1) with s_down ∈ [0, 75.44) m;
- **upstream mainline:** s_down ∈ [−800, 0) m;
- **downstream mainline:** s_down ∈ [75.44, 225.44) m.

So the mainline zone is s_down ∈ [−800, 225.44) on lanes 0–1, plus all of
lane 2. Junction-internal vehicle-ticks (no `s_down`) count as **IN-zone**.
This is the conservative choice: it pushes toward K = 192.

## 3. Rule (applied mechanically to the point estimates)

- `out_share` = dropped vehicle-ticks outside the zone ÷ all dropped
  vehicle-ticks, pooled over the sample.
- `in_zone_drop_share(L)` = dropped in-zone vehicle-ticks ÷ rostered in-zone
  vehicle-ticks, at loading L ∈ {2000, 2400, 2600, 2700}.

**Choose B** if `out_share ≥ 0.95` AND `in_zone_drop_share(L) < 0.005` for
every L. **Otherwise choose K = 192.**

## 4. What each option means (fixed now)

- **B (spatial cap at K = 128):** the slot pool stays 128. When the pool is
  full, an arriving IN-zone vehicle takes the slot of the occupied OUT-of-zone
  vehicle farthest from the merge (largest |s_down|); that vehicle is dropped
  from that tick on.
  - Out-of-zone arrivals are dropped as now.
  - If every slot is held by in-zone vehicles, the in-zone arrival is dropped
    and counted.
  - Tensors, model, memory and timing are unchanged.
- **K = 192:** K_max = 192 everywhere (features, pack, model, rollout2); the
  pool rule is otherwise unchanged.
- **Either way:**
  - a `FORMAT.md` revision;
  - a full re-featurization of the TRAIN + CAL inventory with `--idea-fast`
    into NEW D:/rd-staging/wmn1_feat/{shards_v3, pack_v3, upload_v3}, never
    touching v2 or v2c;
  - corpus verify ×2, a new manifest and SHA256SUMS/PARTS;
  - a post-fix slot census reporting the residual drop share.
- Member 1 and held-out test (i) train on pack_v3. The timing gate runs only
  on pack_v3 with the fixed real-batch path (`zcode/m1realbatch`).

## 5. Result (FROZEN 2026-10-01; §1–§4 unchanged from the pre-census note, LF-sha `4d10a7cd…`)

- **Census:** `KCENSUS_RESULT.json`, LF-sha `2b21c484…`, sample sha
  `affce2e2…`. The rule note was committed at 148439f, before the census ran.
  - 290 of the 300 sampled runs were censused.
  - All 290 match the shard drop counters (290/290).
  - 10 sampled runs (all chunk1, p40) have no recoverable raw source. They
    were not substituted. This is disclosed.
- **Rule quantities:**

  | quantity | value |
  | --- | --- |
  | `out_share` | **0.729** (CI 0.711–0.745) |
  | `in_zone_drop_share`, m2000 | 0.0000 |
  | `in_zone_drop_share`, m2400 | **0.0149** |
  | `in_zone_drop_share`, m2600 | **0.0346** |
  | `in_zone_drop_share`, m2700 | **0.0429** |

  - Both B conditions fail, so **the chosen option is K = 192**.
  - The 10 missing runs cannot change the outcome: B would need
    `out_share ≥ 0.95` and every in-zone share below 0.5 %.
  - Descriptive only: the residual pooled drop share at K = 192 is 0.14 %.
- **Disclosure of relocated raw sources** (corrected before the sha was
  distributed):
  - 73 inventory runs, all `chunk1/*p40*` branch arms, are no longer at their
    recorded raw path; `MISSING_RAW_ALL.json` lists them.
  - With owner approval on 2026-10-01 at 15:34, after the FEAT1 v2
    featurization, another session archived them losslessly. The archive is
    `runs/campaign_stage1_chunk1/_archives/chunk1_p40_branches_superseded.tar.zst`
    (sha256 `4166f207…`), with a per-file manifest.
  - For pack_v3, their physics sources are extracted read-only from that
    archive into a staging directory. Each is checked against the pack v2c
    INDEX `input_sha256` before featurization; the runs directory is not
    restored.
  - The census did not use the archive for its 10 affected sample runs.
    Those runs stay disclosed as not censused; they cannot change the outcome
    (see above).
