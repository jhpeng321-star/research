# S-C1FLOOR — prospective RNG-only divergence design (2026-10-31)

Authority: owner-approved S-C1FLOOR registration 2026-10-31, line 5567.
This protocol is fixed before any SUMO. Paper B reference only; never tuning.

## Fixed panel and estimand

Use the first 24 worlds by index of C1CLOSED's primary WMDATA panel:
969400–969423, t0=300 s, native controller, textbook population, native lanes.
They comprise two repetitions of all 12 penetration/demand cells. Preserve the
recorded native continuation as reference. These are explicitly reused TRAIN/dev
worlds, not unseen-world data. Native is the sole arm; do not generalize to
PI/CIRCLES/MAPPO or pool arm-specific C1CLOSED comparisons without a label.
For world index i=0..23, alternative SUMO RNG seeds are 969050+2*i and 969051+2*i.
They are RNG initializers only: world seed, requests, future vehicle attributes,
identities, routes, demand and penetration remain those of the original world.

## Exact reproduction and RNG intervention

Rerun the original 300 s native trunk using its original seed and the byte-frozen
WMDATA E3 runtime/observer. At t300 save SUMO state with precision 17 and RNG,
plus the COOPMEAS2 Python snapshot (recorder disabled only for snapshot copying).
Continue the live trunk for 300 s and require bitwise equality of every recorded
observable tape array through t600 against the original WMDATA tape, on EVERY
world. No outcomes used for selection or seed replacement.

Canary world 969400 additionally restores the complete saved state/RNG into a
fresh SUMO and continues 300 s; all observable frames must equal the live trunk
bitwise. Only after this passes may the two alternative continuations run.
For alternatives, create a byte-preserving copy of the saved state with ONLY
`rngState` XML elements omitted, launch fresh SUMO with the alternative seed,
and use the inherited `branching.load_state` + Python snapshot/subscription and
speed-mode restoration. The launch seed initializes fresh random streams; the
unchanged snapshot retains all physical/driver/route-loader state and RNG-index
assignments. Record removed-element count and state hashes; fail on unknown
RNG encoding. Never rewrite scenario or assignment files. Verify restored t300
observable state, request/identity digest and native ownership before stepping.
No prefix is simulated for a saved-state load. Keep original world seed in the
Python environment after restore, so no non-SUMO stochastic source changes.

## Budget and mechanics gate

24 x (300 prefix + 300 original continuation + 2 x 300 alternative) = 28,800 sim-s.
One same-seed saved-RNG restore canary = 300 sim-s; exact plan 29,100 sim-s.
Hard cap 30,000 ACTUAL including every prefix, partial failure and canary.
900 sim-s remain unallocated, not an automatic retry allowance. No retry on
scientific failure. A failure stops dispatch; preserve all bytes and charges.
Before every world, append/fsync its maximum reservation to `accounting.jsonl`
under an exclusive task lock (1,500 canary; 1,200 each remaining world). Workers
refuse absent/mismatched reservations, report actual physics ticks, and enforce
their reservation. Actual advances include warmup; conservative charges are
never refunded. No further job admitted if cumulative reservations exceed cap.

## Guards, exposure, resources

Run BOTH frozen `require_development_case(seed,demand,'exposed')` and
`check_case(seed,demand)` for original worlds and alternative RNG initializers.
Also explicitly reject 964000–964199, 962xxx, 977xxx and all denied paths.
Audit exact registration tokens/ranges and safe filenames before allocation;
original-world WMDATA reuse is intended, alternative RNG exposure must be empty.
Never open denied directories, truth data or simulator assignment files.
Hash permitted frozen source closure, observer and original reference tapes.
One nice19, single-thread gpuR CPU worker (<=8 permitted), CUDA hidden. Check
one-minute load <=20 before EVERY world/branch; wait in 60 s intervals otherwise.
SUMOCF/COOPMEASURE have priority. Use task-only STOP/locks, own PIDs and fresh
owned SUMO children only. Existing environments only; no deletions or push.

## Metric and output contract

Reuse upstream C1CLOSED metric implementation, import or byte-identical copy;
do not implement alternative endpoint/delay/aggregate definitions. Horizons
10/30/60/90/120/180/300 s. Original t300 HDVs, stable identities, intersection
of survivors in the original and alternative continuation, median absolute
position/speed error within each start/draw, median over two draws per world,
then median across worlds. Report survivors and missing late horizons, no
zero imputation. Bootstrap WHOLE worlds, both draws together, 2,000 resamples,
seed 0, percentile 95% CI. Keep per-world/per-draw exports.
Reuse C1CLOSED's left-boundary .5 s road+source J components, crossing counts
and its aggregate-field scorer once available. Preserve full native traces so
an upstream scorer arriving later requires no new SUMO. Same CSV columns as
C1CLOSED error-vs-horizon export; mark native arm and SUMO RNG reference clearly.
Deliver `docs/paper_results/C1FLOOR.md` <=40 lines, CSV and one completion
registration line. Commit source review, design, tested canary, run, report.

## Zero-step mechanics amendment (before any trajectory outcome)

The first canary attempt stopped at actual 0: the inherited zero wrapper indexes
`inputs['policy']` even for native (which never loads it). Retain the failed
attempt directory and its entire 1,500 sim-s precharge; supply the original frozen
policy path on the repaired native call. This is a manual mechanics repair,
not an automatic or outcome-dependent retry. Repair canary ID gains `_attempt2`.
To keep even conservative CHARGES within 30,000, freeze the panel to the first
23 worlds, 969400–969422, omitting the final index before any scientific result.
Revised plan: 23 x 1,200 + 300 canary = 27,900 actual; plus retained 1,500
zero-step failure reservation = 29,400 charged. 600 charged headroom stays
unallocated. The remaining design/seed mapping is unchanged. No additional
repair is authorized automatically; all failures remain preserved and reported.

Resource-only clarification, still actual 0 before repaired dispatch: gpuR has
224 host CPUs but a 25-CPU cgroup quota. Host load 21.74 coincided with only
7.92 actually busy cgroup CPUs over five seconds. Thus admission is host load
<=20 OR measured five-second cgroup CPU use <18, preserving >=7 quota CPUs
before adding this ONE nice19 worker. Otherwise yield 60 s. Check each branch.
Own queued coordinator 507673/start4206262749 was terminated only after proving
no repaired precharge or worker existed. No peer process was changed.

Final pre-outcome admission refinement: observed cgroup load stabilized near
20–21 busy CPUs, with 4–5 genuinely idle quota CPUs despite host load >25.
Use <23 busy CPUs (rather than <18) for ONE worker, leaving >1 CPU headroom
even after admission; nice19 and before-every-branch checks remain mandatory.
Own queued 508011/start4206272443 was likewise stopped before any new precharge.
No simulator has yet advanced; no scientific selection or budget change.

Second zero-step repair, still before SUMO advances: the path audit refused
the frozen builder's WRITE of its own generated vehicle definitions. The owner
prohibits READING those files; permit write-only creation inside this task's
scenario output, while refusing every Python read and all other denied paths.
A dedicated CPU regression covers read/write/update modes. Both actual-0 failed
attempts and both 1,500 reservations stay preserved. Manual repaired canary
uses `_attempt3`; final fixed first-22-world panel 969400–969421, same original
index RNG mapping (969050–969093). Plan 26,700 actual + 3,000 retained failed
precharges = 29,700 conservatively charged; 300 unallocated. No outcome exists
at this amendment. No assignment file was opened by the refused write.

Third retained actual-0 setup failure was the same write-only guard, specifically
the builder's generated truth OUTPUT (source `scenarios.py` lines161–168 confirms
the write path). Permit creation of that own output too, NEVER its read. An
end-to-end native setup probe intercepts the launch function before any SUMO
process/step to verify the complete setup before the next charged attempt.
Final amendment, before any scientific outcome: first21 worlds969400–969420,
RNG initializers969050–969091; repaired canary `_attempt4`; planned actual25,500
and conservative charges30,000 including all three retained1,500 zero-step
reservations. No charged headroom; no further automatic attempt authorized.

Fourth manual mechanics repair: attempt4 used600actual sim-s; its original
continuation PASSED bitwise t/ID/s/v/a/lane reproduction, then the inherited
snapshot deep-copy refused a module because run_e3.close had cleared env.api.
Reattach the snapshot's original API module before calling inherited restore,
so its existing shared-object memo works. No simulation dynamics change.
Retain that attempt's1,500charge/600actual. Revised fixed first19worlds
969400–969418/RNG969050–969087, canary `_attempt5`: successful plan23,100actual
+600retained=23,700actual; precharge23,100+6,000retained=29,100. No RNG-divergence
outcome has been observed or used for this index-only mechanical budget cut.
The published C1CLOSED analyzer is now available: import its forecast, endpoint,
errors and CSV functions verbatim; copy its bootstrap with ONLY panel dimension
40→19. Its endpoint intersection is over truth and BOTH alternative draws;
per-draw survivors also exported. Aggregate draw medians use its exact common
J-order component weights. This resolves the earlier unpublished-CSV dependency.

Fifth manual mechanics repair: attempt5 reached900actual (600original+300saved
continuation) but recording its last frame queried env.now after the native
episode closed its API. Use the unchanged step receipt's time_s to record that
already-observed frame; no extra physics or horizon extension. Retain1,500charge/
900actual. Fixed first18worlds969400–969417, RNG969050–969085, canary `_attempt6`:
21,900successful actual +1,500retained=23,400actual;21,900+7,500=29,400charged.
Bootstrap dimension is correspondingly40→18. No RNG-divergence outcome observed;
index-only panel cuts remain mechanics/budget driven, never error based.

## Terminal gate result — no floor inferred

Attempt6 reproduced the original recorded continuation bitwise, but the
original-seed, complete-RNG saved-state continuation FAILED bitwise identity:
at300.5s two HDVs already have different position/speed/acceleration. This is
a mechanics/determinism failure, not evidence of an RNG-only divergence floor.
STOP is durable; no alternative-seed branch or production world was run.
Final budget:6retained reservations×1,500=9,000charged; actual0+0+0+600+900+900
=2,400sim-s, including all prefixes and partial canaries. All task PIDs exited.
Any future run needs a genuinely bitwise-compatible restoration/reseeding
mechanism, a new prospective binding and fresh precharges; no retry is armed.
