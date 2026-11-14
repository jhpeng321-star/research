# REVIEW: WD-01-PA2G-CNT1 — FIRST-PASS

**Reviewed:** 2026-09-30, branch `zcode/revcnt1` at `77073d0` (= `zcode/wd01cnt1b`
`77073d0`; merged into `zcode/wd01` as `b9200ce`).
**Against:** `output/wd01/REGISTRATION_PA2G_CNT1.md` — LF-sha256 recomputed at
review time: `1bfbce27c7ada42a680fc807344c21ab8303fa8b03e469c8d8705c4e1c213116`
(matches the hash quoted in the report and used in `eval_cnt1.py` metadata).
**Method:** read-only audit of code/report/tables vs registration + a tables and
bootstrap RE-RUN from the saved raw rows only (`results/eval_cnt1/raw_rows.jsonl`,
8960 rows) into a temp directory, diffed against the committed
`tables.json`. ZERO SUMO; no world evaluation re-run; no chunk-4 CALIBRATION or
HOLDOUT file opened (`WD_FORBID_D=1` set for every command).

**Verdict: PASS** — all acceptance items pass; three small documentation defects
found and fixed in a `review fix` commit (`5f4a1f3`). No numbers, CIs or the
verdict changed.

---

## 1. Acceptance items

### 1.1 Injector matches registration §3 — PASS

Registration §3 requires: rate λ_e = count_e(w − W, w] / W with **W = 60 s**
carried forward; deterministic evenly spaced schedule "the k-th injected vehicle
is due at w + (k − 0.5)/λ_e while its due time ≤ w + H"; placement at s = 0,
mainline alternating lanes 0/1 "starting with lane 1 (right)", ramp single lane;
v_ins = min(lane limit, nearest-leader speed); IDM safe gap s0 + v_ins·T with
s0 = 2 m, T = 1.5 s; FIFO virtual queue retried each step; injected vehicles are
generic HDVs with the nominal parameters; counts from `departed_ids` classified
by first physical row edge, mainline edge `1191554636`, ramp `11553048#1`;
"no vType, no CAV flag, no penetration, no future schedule".

`partA_v2g/arrivals.py` implements exactly this. Counter window (half-open,
future filtered BEFORE counting):

```python
def counts_from_departures(departures, w, W=W_COUNTER_S):
    out = {e: 0 for e in ENTRANCES}
    for t, ent in departures:
        if t > w + 1e-9:            # after w: never read
            continue
        if w - W + 1e-9 < t <= w + 1e-9:
            out[ent] += 1
    return out
```

Schedule, lanes, and safe-gap rule:

```python
due = w + (k - 0.5) / lam
if due > w + H + 1e-9: break
...
def lane_for(entrance, k):
    if entrance == "main":
        return 1 if k % 2 == 1 else 0        # starts lane 1 (right)
    if entrance == "ramp":
        return 2
...
v_ins = min(limit, float(eng.v[leader]))
gap = float(eng.s[leader]) - float(eng.length[leader])
if gap < s0 + v_ins * T:                     # s0=2 m, T=1.5 s from params
    self.blocked_steps += 1
    return False                             # FIFO retry each step
```

**Test proving no count after w** — `tests/test_cnt1_injector.py::
test_no_count_after_w_is_ever_read` (passing): feeds sentinel future departures
`(w + 1e-6, "main"), (w + 0.5, "ramp"), (150.0, "main"), (1e9, "ramp")` and
asserts the counts are identical to the unpolluted departures for W ∈
{60, 30, 1}. The half-open window boundary is separately asserted
(`t = w` counted; the lower edge excluded).

**No type/CAV/penetration input** — `test_injector_takes_no_type_input`
(passing) asserts the `PartAV2GEngine.insert_vehicle` signature is exactly
`{self, vid, lane, s, v, length_m}` and greps the injector source for
`vtype / penetration / is_cav / type_id` (none). The registry even forbids the
strings inside `ArrivalInjector`.

The ramp entry edge is recorded (`arrivals.py` header: "both verified against
the preserved network: two-lane motorway start / single-lane 378.66 m on-ramp,
neither has a feeder edge") as §2 requires.

### 1.2 `params_nominal.json` unchanged — PASS

LF-sha256 recomputed: `079a55768f803b23c92af188680c1df91dc63f04acbbbac948870b46e75935f7`
(raw-bytes sha256 `ba1fd93f...3455ce` = the recorded `param_set_id`), matching
the registered value. `git log -- params_nominal.json`: last touched in
`615ac1f` (pre-CNT1); untouched by all eight CNT1 commits.

### 1.3 Injector-off byte-identity test present and passing — PASS

`tests/test_cnt1_injector.py::test_nominal_byte_identity_with_injector_off`
(passing): the same engine run with and without an `ArrivalInjector` attached
produces identical `snapshots`, `collisions`, `collisions_by_step`,
`lc_counts`, `exited`. Additionally the registered preflight gate
(`eval_cnt1.py --preflight`, results in `results/eval_cnt1/preflight.json`,
commit `c03979d`) verified the runner's injector-off v2g/const_speed rows equal
the UNTOUCHED `eval_a2.window_rows` output on 2 worlds × 2 windows, field by
field, and that an empty-schedule injector reproduces the v2g rollout exactly.

### 1.4 Tables reproduce from raw rows — PASS

Re-ran `cnt1_tables.py` (module import with `OUT_DIR` redirected to a temp
directory; raw rows `results/eval_cnt1/raw_rows.jsonl` only) — bootstrap
B = 2000, seed 20260930, deterministic. Diff vs committed
`results/eval_cnt1/tables.json`: byte-identical EXCEPT one line, the embedded
absolute path `"raw_rows"`, which records the worktree of the generating
session (`E:\\rd-wt\\wd01cnt1b\\...` committed vs `E:\\rd-wt\\revcnt1\\...`
re-run). Every number — all 16 per-model×H aggregates, all four bootstrap
blocks, the decision rule, the arrival counts, the post-LC1 cross-check
(max |float diff| 5.684341886080802e-14, all n match) — is identical. stdout
excerpt from the re-run (real output):

```
| H60 | -183.3 [-196.5, -169.6] | -19.0 % [-19.9, -18.3] | +183.3 [+169.6, +196.5] | -0.8 [-1.0, -0.6] | -0.0 [-0.0, -0.0] |
| H120 | -722.9 [-791.3, -654.9] | -23.7 % [-24.9, -22.8] | +724.9 [+657.8, +791.8] | -0.4 [-0.7, -0.2] | -0.0 [-0.0, -0.0] |
- (1) H60: rel -19.0 % CI [-19.9, -18.3] -> False; H120: -23.7 % CI [-24.9, -22.8] -> False
- **VERDICT: NOT WORTH IT**
DONE cnt1_tables
```

Report §4/§5/§6 tables were checked number-by-number against `tables.json`:
all match (including n = 576/512 per H, the 342.7/1047.1/2145.5/4275.0 s mean
true all-vehicle delays, and the p3o collision column 0.234→17.021/1000).

### 1.5 Section-5 decision rule applied verbatim — PASS

Registration §5 vs `cnt1_tables.decision_rule`:

| registration | code | match |
|---|---|---|
| "(1) ... H60 AND at H120 is each ≥ 25 % lower than v2g ..., with the world-bootstrap 90 % CI of the relative change excluding 0" | `rel[H]["point"] <= -0.25 and rel[H]["hi"] < 0.0` for H ∈ {60, 120} | yes |
| "(2) ... H10 and at H30 is not worse than v2g by more than 5 %" | `rel[H]["point"] <= 0.05` for H ∈ {10, 30} | yes |
| "(3) tracked-vehicle s_RMSE at H10 and at H30 not worse ... by more than 5 %" | `(sc - sv)/sv <= 0.05` | yes |
| "(4) collisions ≤ v2g + 0.1 per 1000 veh-steps" | `collisions_per_1000 <= v2g + 0.1` at every H (conservative reading, disclosed in the module docstring and printed per H) | yes |
| "PARTIAL if (1) holds for only one of H60/H120 and (2)–(4) hold; NOT WORTH IT otherwise" | `sum(c1.values()) == 1` for PARTIAL | yes |

### 1.6 Verdict follows from the numbers — PASS

H60 −19.0 % (CI [−19.9, −18.3]) and H120 −23.7 % (CI [−24.9, −22.8]): both CIs
exclude 0 but both magnitudes are below the 25 % bar → clause (1) False at both
horizons → WORTH IT unreachable and PARTIAL unreachable → **NOT WORTH IT**.
Clauses (2) −4.2 %/−11.6 %, (3) 0.0 %/0.0 %, (4) margins
−0.006/−0.006/−0.007/−0.012 all hold, as the report states.

### 1.7 % of mean all-vehicle true delay reported — PASS

Report §4 has the "MAE % of true" column (e.g. v2g H120 71.2 % → v2g-CNT
54.3 %) plus the mean true all-vehicle window delay per H (342.7 / 1047.1 /
2145.5 / 4275.0 s) — matches `delay_all_mae_pct` and `delay_true_all_mean` in
`tables.json`.

### 1.8 p3o labelled oracle reference — PASS

Report table rows and §4 text: "v1_p3o_replica (oracle-arrival reference, not
deployable)" and "bought with LOGGED true arrivals ... not deployable and not
a candidate" — matches registration §4 item (iv).

### 1.9 Attempts 1–5 + resume disclosed — PASS

Report §3 discloses: attempt 1 (tool timeout + two code bugs, fixed,
`results/eval_cnt1_aborted_tooltimeout/`), attempts 2–4 hard-killed externally
(`RUN2/3/4_KILLED_NOTE.txt`, code exonerated by two clean smoke runs), attempt
5 (8 salvaged worlds w8623–w8634), the crash-isolated per-world runner
(`5cfcbf1`, `67556d7`), and the 240-min-limit resume that ran the last 5 worlds
fresh (wall seconds per world recorded, ≤ 2 workers). The ledger draft repeats
the same disclosure. `git log` confirms the commit trail.

### 1.10 `LEDGER_DRAFT_PA2G_CNT1.json` in ENTRY schema — PASS

Exactly the 11 required top-level keys: `id, name, controller, question,
scenario, budget_registered, budget_actual, results, criteria_verdict,
limitations, artifacts` — no extra keys. Content is consistent with the report
and `tables.json` (verdict, clause values, disclosure).

### 1.11 Guard / label / provenance — PASS

Split is checked from the chunk-4 `WORLD_PLAN.json` before any run-file open
(`load_panel_worlds` asserts `split == "TRAIN"`); close-out
`cnt1_merge_worlds.out`: "panel-only + plan assertions: PASSED" (24/32 worlds
have `opened.json`; the 8 salvaged worlds' gap is disclosed with reason).
Mandatory label present in report §1 and §8. Truth/plan provenance recorded in
report §10 (frozen evidence paths of the generating session).

## 2. Defects found → fixed in `review fix` commit `5f4a1f3`

1. **Report §1 queue-delay wording.** Said "FIFO virtual entrance queue with
   free-flow-reference queue delay", which reads as the registration's DEFAULT
   convention, while the implemented and tested convention (registration's
   alternative branch, correctly applied because the truth physics rows contain
   a vehicle only from insertion onward) accrues NO delay while queued —
   identical on both sides. Registration required the worker to "state which";
   the wording now states it and cites
   `test_queue_delay_accrues_only_after_insertion`.
2. **Report §2 "16/17 model-rows per window".** Recounted from the raw rows:
   rows-per-window histogram is {16: 512, 12: 64} — 16 model-rows per full
   window (4 models × 4 H) and 12 at the 64 edge windows that drop an H
   exceeding the run end. Corrected.
3. **Report §8 clause-4 margin "−0.013".** `tables.json` gives −0.01249 →
   −0.012 at the report's 3-decimal rounding. Corrected (the clause verdict is
   unaffected by either rounding).

None of the three touches any table number, CI, clause outcome or the verdict.

## 3. Larger observations (NOT defects, no action required here)

- The committed `tables.json` embeds the absolute raw-rows path of the
  generating worktree (`E:\rd-wt\wd01cnt1b\...`). Cosmetic; frozen evidence is
  not edited. Noted for anyone diffing tables in a moved worktree.
- Clause (4) is evaluated at every H (conservative); the registration text
  could be read per-H-only-at-(2)/(3). Both readings pass with large margin,
  and the conservative reading is disclosed in the code docstring and the
  report.
- Report §10 truth/plan provenance strings are absolute paths of the original
  session; per the migration rules these stay as frozen references.

## 4. Verification commands run during this review

```
$env:WD_FORBID_D='1'; python -m pytest -q output/wd01/partA_v2g/tests   # 93 passed
$env:WD_FORBID_D='1'; python -m pytest -q output/wd01/predict/tests     # 260 passed
$env:WD_FORBID_D='1'; python -m pytest -q output/wd01/tests             # 26 passed
# cnt1_tables.py re-run into temp dir, diff vs committed tables.json     # numbers identical
```
