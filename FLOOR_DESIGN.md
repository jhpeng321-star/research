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
