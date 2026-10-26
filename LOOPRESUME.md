# S-LOOPRESUME — amendment installed; paired-critic data blocker

## Status

**Cycle 1 is NOT launched.** The attempted detached continuation stopped
`OPERATIONAL_PAUSED` during paired export: all four cycle-0 worlds are assigned to
S-PAIRCRIT's existing development partition, leaving **zero fitting worlds**.
The original attempt and three operational retries are exhausted and preserved.
No partition was changed and no extra world was collected.

The separately authorized **cycle-0 WM fine-tune is running detached on gpuA**,
followed automatically by the **unchanged acceptance check**. Its coordinator
cannot launch an actor or collectors, advance the loop, or promote a candidate
into shared state. Completion/acceptance is pending.

## Installed amendment and checks

- Appended the five prospective owner decisions and exact pre-launch bindings to
  `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md`; appended the same
  amendment to the loop JSON before execution. Twin-readable artifact:
  `gpuA:/root/autodl-tmp/c4/restart20261026/AMENDMENT_C1.json`, SHA256
  `049abecbf6063ad5ecef2dcc4d95eaa60d91c11e51b41c5771ae3a9f2ca43403`.
- Cycle 0 remains historically **INCONCLUSIVE**, policy `1b9949cc...` retained.
  Its collection is closed at worlds968144–968147; nine reservations968148–968156
  stay unconsumed. Original stop/state/config/registration/journal/PID/log retained.
- Cycles>=1 use the same four-world bootstrap and zero margin; only lower CI>0
  stops collection. Inconclusive continues without being labelled PASS. Both-arm
  applicability is explicit; the twin was not contacted.
- Integrated `c4/paircrit@f8374e1f`; paired critic ON in the boundary config,
 100 critic-only warmup updates; longcritic/absolute-share selection replaced.
  Breakdown/conformal OFF. Fit/development RMSE monitoring was wired but could
  not run on an eligible cohort. Any next-boundary reversion remains unexercised.
- First cycle-1 phase is configured to measure150 new updates, select the best
  eligible frozen16-panel native-relative J at the unchanged10s scoring clock,
  and retain SHA/WM/config vetoes. Added phase-relative update10 progress gating
  and subsequent measured-count wiring. None of those new actor updates ran.
- Final offline checks: **422 passed, 2 skipped** (410 integrated18.53s plus
  12 paired9.66s). Skips are optional historical exports absent on gpuA;
  current-policy exports were independently checked against real common starts.
  Ruff passes. Final staged39-file delta-manifest SHA:
  `90cc238df1445ae37bb7f4e4fdc46c52ba52d394dfe558ce45c57f6b67c84852`,
  identical on gpuA/cpuP/cpuQ. The measured-count follow-up was staged after the
  pause; future continuation requires fresh bindings, not the initial launch receipt.

## Why the actor cannot proceed

The registered rule is `sha256("965140:" + world_seed) % 5 == 0` for development.
Remainders for968144/145/146/147 are **0/0/0/0**. All16 converted episodes and
eight matched300s POLICY/B pairs are present; both observable initial arrays and
raw common-start proofs match. The exporter then refuses `no paired fitting worlds`.
The resulting partial proof is evidence of common starts, **not fitting eligibility**.

Driver **62093/start841844830** exited. `OPERATIONAL_PAUSED.json` and
`LOOPRESUME_PAIRING_BLOCKED.json` record the blocker. There is no new scientific
WAIT_OWNER verdict and no fourth exporter retry is armed. A new owner-authorized
fitting-data/partition amendment is required to enable the paired cycle-1 actor;
development labels and the nine reserved worlds remain untouched.

## Independent WM and gates

External coordinator **62645/start841871931**, trainer **62910/start841872654**,
started2026-10-26T18:15:05Z. Script SHA
`32fd991c8659c691e8f935610498565c6f9678dfd23cb6a65820a5a425c11d66`.
It reuses verified attempt1 arrays, without retrying the exporter or repairing the
failed shared phase. Original R16 config resolves40 steps,4.5h;72 tensors/core
parameters frozen, no unmatched freeze prefixes,171 checkpoint tensors loaded
with no new/mismatched/unused tensors. Loop pairs supply304 windows at the
original0.15 mix; original sources remain included. Same acceptance runs afterward.
At2026-10-26T18:23:01Z, step1 is finite: loss181.57535, grad norm81.53729,
242.623s, peak allocated1888.71MiB; coordinator incarnation is live. Shared
loop state is byte-identical to its independent-WM starting snapshot. This is
progress evidence, not a completed40-step fine-tune or acceptance verdict.

Existing confidence-table recalibration completed: reference64 samples,
in-sample Brier0.001510,1 open block; mean comparison56 samples,
Brier0.0002973,0 open blocks. These are in-sample diagnostics, not efficacy or
deployment qualification. No different confidence rule was enabled.

Evidence stays under `gpuA:/root/autodl-tmp/c4/restart20261026/`:
`LOOPRESUME_{BINDINGS,SOURCE,TESTS_PASS,LAUNCH,PAIRING_BLOCKED,WM_PID,WM_ONLY}.json`,
`loopresume_wm_only.log`, `arm/loop/wm/c00{,.log}` and subsequent checks.
Small snapshots are in `tools/loopresume/receipts/`.

**Charged simulation remains102,607/210,000; remaining107,393.** No new simulation,
actor updates, evaluation seeds, MAPPO initialization, GPU-R writes, restricted
data access, peer-process signals or push. cpuP/cpuQ source/configuration is staged;
cycle-1 collectors have not started.

Reproduce the local handoff checks with:
`python -B tools/loopresume/verify_receipts.py`.
Read current main progress with `python -B tools/loopresume/operate.py status`;
the independent WM's authoritative progress is `LOOPRESUME_WM_ONLY.json` on gpuA.

## S-C1FIX — policy recovered; collection deployment/allowance pause (2026-10-27 05:04 UTC+8)

- Driver64792 was healthy and had automatically replayed from cycle0 in `c01.a2`.
  It was never signalled/restarted. Cooperatively stopped only that owned replay,
  retaining its files; existing retry logic launched `c01.a3` from original best
  full checkpoint170/SHA`2463e550...`. Config-only policy cap is now0.65 (~30.8GiB),
  leaving7.1GiB even at the WM's full0.20cap. Original WM62645/62910 untouched.
- **150 logical updates completed through180; native-relative gate PASS,16/16
  improved, mean deltaJ=-0.3837442398**, unchanged10s frozen panel/WM/config vetoes.
  Selected checkpoint `arm/loop/policy/c01.a3/policy_latest.pt`, SHA256
  `64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c`.
  Retained original140updates through170 and reran171–180; no repeated warmup/share
  selection, no bit-identical uninterrupted-run claim. **72 pre-retry tests PASS;
  final73 PASS** including an explicit pre-training corpus-SHA veto; Ruff PASS.
- **Collection did launch on both CPUs, but all12 dispatched world jobs failed
  before SUMO**: missing successor-runtime `baselines/circles/adapter.json`.
  First failed worker PIDs cpuP589756/589471, cpuQ582909/583052; **none active**.
  Identical original adapter now installed on both (SHA`8a925e9e...`); missing
  network/template metadata links repaired without opening scenario assignments.
  This is deployment preflight only, not a successful collector run.
- Automatic attempts1–3 retained34,092charged sim-s/7,200long-credit sim-s.
  Driver exited naturally `OPERATIONAL_PAUSED` at04:59:13; cycle caps36,933/8,000
  cannot fund another four-world dispatch. **136,699/210,000 charged**,0completed
  cycle1worlds; convert/WM1 pending, independent WM0 remains autonomous.
  Owner amendment is required before further dispatch: same13worlds, no refunds,
  cycle attempt allowance>=71,025 and long-credit allowance>=15,000 would fund
  one complete remaining collection at173,632total; these increases are **not installed**.
  **ETA blocked pending that amendment.**
- Registration/events append-only; authoritative GPU receipts
  `C1FIX_{SOURCE,TESTS,RECOVERY,VERIFIED,COLLECTION_BLOCKED}.json`, repaired current
  `C1_STATUS.json`; CPU `C1FIX_COLLECTOR_REPAIR.json`. Compact local result:
  `tools/loopresume/receipts_c1fix/RESULT.json`. No push.
