# LC2013 FIT lane diagnostic

`validate.py` reads **only** `episode.json`, `time_s.npy`, `veh_kin.npy`,
`veh_mask.npy`, `is_cav.npy`, `vid.npy` from WMDATA observable views. Exact
allowlist: worlds 969400–969431, native/ctrl0/mappo/pi/circles, 160 episodes.
No global manifest/split is opened. Each input file has a SHA256 receipt.

Frozen diagnostic: causal 240–300s wish warmup; score 300–360s. Both models see
the same measured post-motion positions/speeds/accelerations with pre-change
lanes. Same-ID HDVs on normal lanes in both adjacent frames are scored;
connectors, births, exits and CAV decisions are excluded. All visible normal
road bodies, including CAVs, constrain safety. Parameters: existing SUMO-fidelity
population defaults, no hidden driver files, no fitting or selection.

Reports exact directed-event recall/precision, ±2s one-to-one event matches,
matched timing/position MAE, per-transition and per-arm results. One-to-one
matching maximizes cardinality then minimizes total timing difference. Exact
position error is zero up to coordinate-rounding under longitudinal teacher
forcing and is not a trajectory prediction score. The ±2s tolerance is evaluation-only.

Positive *steps* can repeat while a model wants to change but the teacher still
stays. Thus both positive-step counts/precision and compressed positive-run
counts/window matches are reported. Wishes reset on **observed** lane changes,
never on a hypothetical predicted success. This diagnostic does not measure
free-running lane-change rate, discharge or breakdown fidelity.

Historical MOBIL 1.4% is from another panel/contract. The meaningful paired
comparison reruns current MOBIL on exactly these snapshots and settings.

CPU execution: `operate.py smoke` stages one immutable source bundle on cpuQ;
`operate.py panel` stages one full panel. Existing dispatch receipts block
duplicates. One nice19 worker, torch/OMP/MKL/OpenBLAS single-thread, CUDA hidden,
load admission <20. No signals or process-name killing. `operate.py status
--kind panel` checks identity-bound PID/startticks. `operate.py collect --kind
panel` requires completed output and exited worker. Old results retained.

The completed 969400/native smoke uses the original greedy window matcher;
the full panel uses the CPU-tested maximum-cardinality scorer. Model physics
and settings are unchanged after smoke. No tape outcomes were used to tune
parameters. A junction-zero-onehot bug was corrected before the smoke.

The first full panel is retained unchanged. A source-completeness amendment
adds SUMO's right-overtaking prohibition and its next-step speed advice.
`operate.py panel2` evaluates that revised LC2013 candidate and **reuses** the
already-completed MOBIL baseline. Input SHA256s, physics, horizon and cohort
must match exactly; no completed MOBIL inference is repeated. No tuning.

## Precision continuation / corrected coordinate contractv2

Panels3–8 preserve successive source fixes and parameter sensitivities under
the originalv1 scoring transform. Details and per-predicted-motivation
precision/recall contributions: `docs/LC2013_PRECISION_ITERATIONS.md`.
Tapes do not contain motivation labels: these are contributions to ALL-event
recall, not true class-conditional recalls. TP/FP profiles split physical
upstream/merge/downstream edges, plus bounded-window nearest-event timing.

Starting with9, reconstruct source-lane raw position before deciding. SUMO
preserves raw position on a lane change, whereas upstream normalized WM s
jumps±1.45m because lane origins differ. `snapshot.py` undoes the recorded
jump, and maps hypothetical proposal positions to their own destination
origin. Current `snapshot` string is the scoring version identity. Old v1
MOBIL cannot be reused underv2; one new matched score per setting is needed.
The operator auto-reuses subsequent completed MOBIL scores only when settings,
snapshot, physics and all MOBIL/dependency source-byte hashes match.

`--idm-parameters <published idm_pooled.json>` consumes the existing S-RB-FIT
pre300 FIT point without fitting. `--encoder-checkpoint <wm_s.pt>` strictly
consumes registered WM2 b8797242… encoder weights, authentic240..300 long
history, static observed-VID parameters and zero-history centre fallback for
entrants. No post300 motion is used to infer settings. CAV defaults and SUMO
LC style defaults retained. Both algorithms are scored under matching settings.
Frozen real-weight encoder/full-source parity is BIT-EXACT for all9 outputs.

Operate successive unique `panelN` names; every dispatch is immutable. Verify
each collection using `python -m tools.rblane.verify_receipts --kind panelN`.
Verification uses immutable bundle bytes, not a newer model in this worktree.

`operate.py panel10 --encoder-checkpoint <wm_s.pt> --compare-pre-signed` freezes
a same-input ablation of the signed-current-leader-cap fix: both our prior
0a1e7e93 kernel and current kernel share one causal encoder inference andv2
snapshot, with independent persistent wish state. The prior OWN project source
is SHA-pinned before execution; downloaded SUMO code is never executed. Current
and prior motivation scores are separate. Persistent states must remain finite.
Encoder source/featurizer fingerprints join checkpoint SHA in the MOBIL-reuse gate.

`--motion-points <MOTION_POINT_TABLE.json> --native-only` is a separate
sensitivity consuming S-RB-FIT's already published MotionFit point reader.
It performs no fitting. The export checks the exact32 native input SHA bindings
against completed lane receipts, retains the reader's deterministic point rule,
and uses its population point for unseen IDs. CAV and LC defaults stay unchanged.
Native-only avoids assuming numeric-ID alignment between counterfactual arms;
these32 episodes must be compared with native-only A/B results. Compatible
MOBIL cache reuse accepts a selected-arm subset of an already completed panel.

Final runtime message fields add `lc2013_has_advice` and
`lc2013_advice_weight` to STATE_KEYS. They preserve explicit zero/positive
messages and source wanted-speed blending; old states keep negative-only
fallback behavior. Urgent speed-gain follower help and IDM minNextSpeed slack
are CPU-tested. `check_message_equivalence.py` proves conditional choices/wish
state unchanged relative to scored source52d2df5c while outward advice changes.
`finalize_diagnostics.py` assembles only already-verified score receipts; it
does not repeat inference or establish autonomous distributional fidelity.

## Fresh05:00 source continuation / publication-only followup

Source80541278 completes optional lcSpeedGainLookahead (default0) and nonlegacy
acceptance-time follower response (default-1).71CPU cases pass. Default LC-style
and ALL209published five-parameter profile equivalence receipts pin every old
output BIT-EXACT to aefc4bd9. Current panels14–16 are retained; no repeated scoring.

`consume_fit_precision.py --fit PUBLISHED_DIRECTORY --receipt NEW_NAME.json`
checks publication/VERIFIED/artifact hashes and32native-world count sums, then
derives exact-direction precision/recall from raw-physics expected counts.
It does not fit, select settings, run a model or read original tapes. It reports
the upstream source separately and does not invent missing motivation counts.

`watch_fit_precision.py launch` claims a single local publication-only observer
through2026-11-01T09:00+08. It follows the EXISTING FIT producer's latest verified
`lane_raw_<digest>` publications, generates count summaries once, updates only
the marked HANDOFF block, and commits only its own receipts/HANDOFF. Any unrelated
work blocks publication; no git reset/clean/stash, signals or peer edits. A pinned
consumer-source hash prevents silently changing the running observer's arithmetic.
The create-once dispatch/claim must be inspected on resume, never re-launched.
Runtime status/logs are in `E:/Temp/opencode/rblane-fit-precision-20261101-0520`;
dispatch/completion receipts live under `tools/rblane/receipts`. It performs no
fitting, WM/SUMO/GPU inference or remote dispatch and closes at the owner deadline.
