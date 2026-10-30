# WMFIX design / registration — 2026-10-31
Base: c4/wmfixmerge, WM2 b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436.
CPU engineering only here; MAIN owns any later GPU allocation. Budget superseded to <=6 h, gpuR24GB.
## Evidence and retarget rule
WMREQUAL upstream-exact diagnostic cuts J MAE49/54/60% at120/180/300s; discharge error associates with48–68% variance.
Source-only exact correction worsens total MAE: correcting a source offset is insufficient and hides cancellation.
ERRSRC +68/−297veh/h is counter-confounded; no capacity-drop scalar is justified.
WMREQUAL2 identifies shared lane/native ownership mismatch. WMREQUAL3 report is absent at registration.
Before training, read the corrected report; use matched native-lane arms only, MAPPO lane-policy arm diagnostic separately.
If corrected WM2 no longer shows the proposed residual, do not launch this correction; record a retarget amendment first.
## Mechanism: causal spatial discharge field plus exact entrance ledger
New zero-initialized `merge_response` field: per-lane spatial occupancy, mean speed and CAV flux proxies over the whole upstream road,
plus each ego's position/speed/acceleration/lane and nearest-front bumper clearance/relative speed.
Spatial basis derives from observed network entrance/merge-end geometry, not speed/jam/hold behavioural thresholds.
Shared pooled field exposes mainline/ramp competition and upstream backpressure; smooth positional weights preserve state derivatives.
Output is an HDV acceleration correction, applied before the EXISTING emergency envelope and plant limits; CAV callbacks retain command ownership.
No new lane veto, merge hazard offset, driver-setting override, hand hold-duration or unconditional capacity multiplier.
WM2 all old tensors frozen. Zero output at initialization; flag OFF constructs no parameters, draws no RNG and performs no new arithmetic.
This differs from RAMPFIX's local residual: pooled full-upstream state plus explicit crossing/whole-fleet response targets, new-head-only fitting.
`EntryConfig.conserve_queue` makes per-entrance cumulative due−successful insertions the sole queue state, no max_pending truncation/drop.
Due increments are counted before service, each successful insertion once; finite slot exhaustion retains backlog, never removes demand.
Q0 must be recorded per entrance; no guessed split. Source rectangle timing matches corrected scorer exactly.
Conservation alone does NOT force real intervention invariance; the discharge/admission response must pass the paired tests below.
Physical admission remains the incumbent insertion law; field-induced motion feeds it through actual entrance gaps (no disconnected cost offset).
## Supervision / exact source scope
Pilot source: existing S-E3DATA FIT pairs only, seed allowlist
973002,973053,973060,973083,973093,973123,973126,973133,973151,973160,973190,973224.
Source root: `/root/autodl-tmp/c4/e3data`; use manifest-bound native/ours_c1/coopoff_c1/mappo tapes with role=fit only.
Preparation must list exact selected episode paths + SHA256 + seed/arm/start, and pair common starts/history/due schedules.
No pairs300 (including its nominal TRAIN portion), validation/calibration/eval source, 964000–964199,962xxx,977xxx or SPEEDHEAD worlds for fitting.
No driver truth/XML/assignment files. Inputs are observable positions/speeds/presence/IDs/CAV flags and request-counter ledgers only.
A missing source identity/queue/due ledger stops preparation; old service counts are not request counts.
Fit scales from selected TRAIN pairs only (RMS contrasts; positive numerical floor), equal-world weighting.
Loss = normalized paired all-vehicle J contrasts + paired upstream road/source/crossing contrasts + absolute crossing/road anchors.
J uses reference31.29m/s and exact source waiting with0.5s scorer convention; no vehicle divisor or t0-only roster.
Crossings: same identity bracket across unlabelled junction gaps; reject identity changes, reverse motion and absent-endpoint ambiguity.
No disappearance exits. Crossing surface75.44m; smooth cumulative crossing surrogate for optimization, exact counts for verdict.
Hard insertion source gradients are zero away from service events: source term MUST be reported separately, never called differentiable by fiat.
Do not use straight-through admission gradients without a separate measured finite-difference gate and amendment.
## Cheap feasibility / hard launch gates
CPU: verify actual queue conservation incl. pool exhaustion, source rectangles, crossing identity bracketing and frozen/OFF parity.
TRAIN-only teacher diagnostic: replace predicted upstream component/crossings by recorded targets; report paired J ceiling by horizon.
Then real short WM rollout: nonzero finite paired-J/crossing gradients to this head; central finite differences for head outputs/weights.
Show an actual optimizer step reduces TRAIN paired objective (native/arm together), with source error separately retained.
A synthetic fixture only proves plumbing; no GPU launch based solely on it or an algebraic teacher ceiling.
No real-data/contract/gradient PASS => NOT LAUNCH READY, with explicit missing artifacts; no silent weaker gate.
## Frozen training schedule (MAIN launch only after gates + independent review)
New head only, Adam lr3e-4; FP32, one paired group/microbatch, full600tick graph, existing activation checkpointing.
Start with60s CPU smoke; production60/120/300s stages, up to30/30/60 updates, global wall cap21600s incl. assembly/selection.
Save step0, every completed update, immutable hourly checkpoints and final; stop at cap with latest completed checkpoint.
Selection: TRAIN paired-J normalized MAE only; no calibration/validation feedback; predeclare final checkpoint as primary verdict.
Old parameters/buffers compare bitwise after each update and on reload; optimizer parameter names exactly merge_response.*.
Estimate (not measured): 1group/2rows full300s checkpointed graph ~8–20GB,120–300s/update;120steps~4–10h, cap governs.
Thus6h yields ~60–120steps; latest usable every update, hourly snapshots meet1–2h request; no promise of120steps.
Pre-launch gpuR microbenchmark required for24GB/time feasibility; CPU engineering cannot certify CUDA memory.
## Preregistered independent verdict (same candidates/WM2/runtime/CRN)
A: corrected WMREQUAL3 validation, K8, mix ON, native lanes,60/90/120/180/300s; score WM2 and final candidate identically.
Report paired sign/r/MAE/bias and q10-confident gain correctness/count/coverage, source/upstream/downstream/crossings.
Require pooled-horizon MAE decrease with95% paired-world bootstrap CI excluding0; sign/r/confident-gain accuracy improve pointwise pooled,
and no horizon has significant regression in any of those metrics. Empty confident-gain set is unavailable, never a pass.
Perturbation sigma remains unknown; q10 diagnostic gives no new H_G or policy authority.
B: SPEEDHEAD963880–963887 (evaluation-only here), all4 arms, exactt300 histories/Q0, K8 common futures;300/600s prefixes.
Report each world's arm−CTRL0 J direction AND magnitude for MAPPO/PI/CIRCLES, both WM2 and candidate, plus source/road.
Require lower pooled per-world contrast MAE, no controller's pooled MAE worse, no fewer correct directions (zero effects excluded).
C: unchanged C1 unseen-PI/unseen-wall paired zero-shot scorer and canonical accept.py; no significant state-error regression,
no increased blowups and accept.py ACCEPT. Missing cohort/runtime/safety result blocks adoption.
Bootstrap2000 world draws, seed0, common draws for candidates; no re-tuning on any verdict panel.
One verdict driver must run A+B+C and emit incomplete/blocker status on missing inputs; no assumed pass from process exit alone.
Independent Astra must check the mechanism, source-gradient limitation, exact input roles and actual free-running feasibility before launch.
