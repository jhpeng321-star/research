# S-R13E offline empirical study — 2026-10-25

## Result

**The pinned implementation substantially reduces overlap and low-TTC exposure, increases hard braking and delay, and is NOT equivalent to GAPACC `gs`.** All three offline arms completed on gpuR, CPU only: **48 train-role states × 2 CRN seeds × 120 s = 288 arm rollouts**. No training or GPU operation was launched.

Emergency versus off: overlap pair-steps/run **−445.771 [−709.141, −228.433]**, TTC<1.5 **−0.398 [−0.527, −0.280] percentage points**, hard-brake share **+0.554 [+0.453, +0.649] points**, total delay proxy **+50.194 [+30.482, +72.547] veh-s**. Adding implemented gs reduces post-insertion braking but increases the maximum low-speed ramp-wait statistic by **6.677 [2.594, 10.715] s** and lowers merge-zone occupancy by **0.224 [0.125, 0.342] HDVs** versus emergency alone.

## Provenance and design

- Runtime/model source: **`8c7ff0a19cfae59a50eebcd3146f7bcce51c5b37`**, branch `c4/v10r13e`. The deployed `src` Git tree is verified as **`04f5160ee221475ea48b14fb710161f7f0bcd199`**. Model/physics/gs implementation was not changed for this study. Diagnostic tooling adds four-worker execution, same-panel one-step sampling, progress journaling and occupancy/speed metrics; its executed SHA256 is `f62b561657748444ecf066a42619cc7cc1068a0d2a7c65ac0724000a6f8a1cc9`.
- Archive: minimal allowlisted `git archive 8c7ff0a1`, SHA256 `251fe058eefd31387e0ce94d0b569576957c7f481fcd0d855e1305ed7f041780`, extracted to `/root/autodl-tmp/c4/v5/repo_r13e_8c7ff0a1`.
- Checkpoint: `/root/autodl-tmp/c4/v5/runs/c4_wm_v10_round12c/wm_s.pt`, SHA256 **`c76c338dea6355f85cfaee63cdb9077fb129ed25db0ab582d77eda2961822091`**. Strict key/tensor-byte loading and post-inference identity checks passed. State-dict digest: `fd16166d6955d4a4b7ddbdcdbed79ff754fb9a7c7978f881b565af7285c449da`.
- All arms retain `temporal_accel_off=residual_accel_off=follower_courtesy_off=true`. Only `physics.emergency_braking` and `physics.secure_gap_veto` vary: off `(false,false)`; emergency `(true,false)`; emergency+gs `(true,true)`.
- **Panel correction, fixed before full-arm inference:** all 48 old `states192.json.gz[::4]` snapshots are source-internal **validation-role**, not train-role. They were audited without relabeling and excluded. The replacement uses 24 genuine train-role worlds from each of `data/v5_native` and `data/genrand_plain`, selected at uniform midpoints of sorted eligible episodes; t0=450 s, 60 s history, 120 s recorded future. Its SHA256 is **`32354ca9fc223a0a713c50d37556b7c2ac99bdfa922d7bd434f48fe47a0b80bd`**. [Panel receipt](docs/r13e_offline/panel_receipt.json) records the exact episodes, split roles, array-slice hashes and teacher checks. These estimates are **not paired reproductions of the older GAPACC/R12CROLL panel**.
- dt=0.5 s; t0 roster only; entries off; recorded CAV motion exogenous; HDV recorded exit/junction overrides disabled; HDV lane decisions free. Each source's recorded arm supplies its CAV teacher, including generic-random actions.
- Seeds `130013 + 2*state + replicate`, reset identically across arms for Python/NumPy/Torch and lane draws. **1,000-draw 95% percentile snapshot-cluster bootstrap**, retaining the two seeds together. Paired differences are formed within state/seed before resampling. Rollout tables report equal-run means; percentages are means of per-run shares, not pooled ratios. Differences between percentage rows are percentage points. Empty event denominators are missing, not zero; fewer than five finite runs or two contributing states gives no CI. BASE uses finite-row means with window-cluster bootstrap; there is one window per selected state.
- Main inference used `/root/miniconda3/bin/python`, nice 19, four one-thread workers; an exact-owned-child `/proc` audit confirmed these limits. Main successful process: PID 561734, start ticks 4149806622. The 144 paired one-step arm windows and 288 rollout arms completed; rollout completion was at 1296.816 s from the output journal's start.

## Offline arms

Mean [95% state-cluster CI], 96 runs per arm except event-share rows with missing denominators.

| Metric | off | emergency | emergency+gs |
|---|---|---|---|
| Overlap pair-steps/run | 543.562 [317.110, 804.961] | 97.792 [71.732, 127.136] | 92.656 [68.283, 120.750] |
| TTC<1.5 rows/run | 65.417 [45.384, 87.857] | 21.240 [14.625, 28.355] | 20.406 [14.007, 28.001] |
| TTC<1.5 (%) | 0.590 [0.431, 0.761] | 0.192 [0.143, 0.245] | 0.184 [0.138, 0.237] |
| Hard-brake rows/run | 5.073 [2.811, 7.625] | 60.354 [47.458, 73.278] | 57.344 [45.124, 70.211] |
| Hard-brake (%) | 0.044 [0.025, 0.065] | 0.598 [0.491, 0.703] | 0.571 [0.470, 0.675] |
| HDV inserter post-HB rows/run | 0.260 [0.021, 0.656] | 1.125 [0.740, 1.552] | 0.698 [0.438, 1.000] |
| HDV inserter post-HB rows (%) | 0.492 [0.028, 1.262] | 1.395 [0.898, 1.929] | 0.918 [0.539, 1.356] |
| HDV inserter post-HB events (%) | 1.821 [0.206, 4.222] | 12.354 [8.742, 16.188] | 8.550 [5.238, 12.292] |
| HDV follower post-HB rows/run | 0.115 [0.000, 0.344] | 1.104 [0.666, 1.583] | 0.406 [0.198, 0.688] |
| HDV follower post-HB rows (%) | 0.079 [0.000, 0.243] | 1.454 [0.898, 2.057] | 0.881 [0.330, 1.535] |
| HDV follower post-HB events (%) | 0.086 [0.000, 0.265] | 9.487 [6.340, 13.089] | 4.736 [1.888, 8.387] |
| Stuck vehicle-steps/run | 10.354 [3.260, 18.682] | 7.719 [2.780, 14.138] | 11.844 [4.072, 22.784] |
| Final stuck vehicles/run | 0.010 [0.000, 0.031] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Maximum low-speed ramp wait (s) | 29.661 [22.686, 37.521] | 28.714 [22.590, 35.151] | 35.391 [28.499, 42.615] |
| Ramp merges/run | 4.385 [3.542, 5.250] | 4.375 [3.531, 5.250] | 4.354 [3.521, 5.208] |
| Upstream delay proxy (veh-s) | 940.570 [736.534, 1187.730] | 983.987 [769.859, 1237.983] | 974.518 [764.027, 1228.771] |
| Merge delay proxy (veh-s) | 350.835 [281.297, 425.063] | 348.059 [279.896, 414.243] | 366.039 [296.762, 436.915] |
| Downstream delay proxy (veh-s) | 365.686 [325.372, 405.585] | 375.240 [335.331, 415.321] | 370.301 [330.813, 410.330] |
| Total delay proxy (veh-s) | 1657.092 [1391.824, 1950.283] | 1707.285 [1437.931, 2008.392] | 1710.858 [1453.426, 2010.008] |
| Merge-zone occupancy (HDVs) | 3.047 [2.562, 3.537] | 3.065 [2.585, 3.523] | 2.841 [2.423, 3.239] |
| Merge-zone speed (m/s) | 12.671 [10.576, 14.737] | 12.395 [10.289, 14.491] | 12.936 [10.887, 14.978] |
| Dynamics-failure runs (%) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### Paired contrasts

Later minus earlier, mean [95% CI]. The machine-readable report includes all metrics and all three contrasts.

| Metric | emergency − off | emergency+gs − off | emergency+gs − emergency |
|---|---|---|---|
| Overlap pair-steps/run | −445.771 [−709.141, −228.433] | −450.906 [−713.508, −233.151] | −5.135 [−8.803, −1.791] |
| TTC<1.5 (percentage points) | −0.398 [−0.527, −0.280] | −0.406 [−0.542, −0.285] | −0.008 [−0.022, 0.003] |
| Hard-brake rows/run | 55.281 [43.478, 67.190] | 52.271 [41.050, 63.782] | −3.010 [−5.604, −0.969] |
| Hard-brake (percentage points) | 0.554 [0.453, 0.649] | 0.527 [0.433, 0.622] | −0.027 [−0.049, −0.008] |
| HDV inserter post-HB rows/run | 0.865 [0.583, 1.156] | 0.438 [0.188, 0.677] | −0.427 [−0.604, −0.250] |
| HDV inserter post-HB events (points) | 10.532 [7.313, 14.313] | 6.729 [2.570, 10.977] | −3.803 [−6.678, −1.657] |
| HDV follower post-HB rows/run | 0.990 [0.573, 1.448] | 0.292 [−0.031, 0.646] | −0.698 [−1.146, −0.333] |
| HDV follower post-HB events (points) | 9.507 [6.397, 13.149] | 4.701 [1.874, 8.303] | −5.192 [−8.241, −2.333] |
| Stuck vehicle-steps/run | −2.635 [−8.459, 2.229] | 1.490 [−3.792, 7.199] | 4.125 [−0.563, 9.866] |
| Maximum low-speed ramp wait (s) | −0.948 [−5.282, 3.297] | 5.729 [0.571, 11.599] | 6.677 [2.594, 10.715] |
| Ramp merges/run | −0.010 [−0.052, 0.021] | −0.031 [−0.073, 0.000] | −0.021 [−0.062, 0.000] |
| Upstream delay proxy (veh-s) | 43.416 [25.162, 62.947] | 33.947 [20.479, 46.477] | −9.469 [−22.518, −1.104] |
| Merge delay proxy (veh-s) | −2.776 [−25.955, 16.815] | 15.204 [−7.399, 40.388] | 17.981 [0.702, 40.446] |
| Downstream delay proxy (veh-s) | 9.554 [−0.978, 21.102] | 4.615 [−7.729, 16.636] | −4.939 [−11.250, 1.359] |
| Total delay proxy (veh-s) | 50.194 [30.482, 72.547] | 53.767 [30.206, 77.876] | 3.573 [−13.872, 27.531] |
| Merge-zone occupancy (HDVs) | 0.018 [−0.131, 0.170] | −0.205 [−0.415, −0.032] | −0.224 [−0.342, −0.125] |
| Merge-zone speed (m/s) | −0.277 [−0.542, −0.024] | 0.265 [−0.131, 0.698] | 0.541 [0.273, 0.909] |

### Definitions and denominators

- **Overlap:** unordered same-lane present pairs with at least one HDV, strictly negative bumper gap, counted each post-step. A pair persisting for ten steps contributes ten; this is not an onset count or a collision percentage. Pooled totals are **52,182 / 9,388 / 8,895 pair-steps**.
- **TTC:** nearest same-lane leader, bumper gap / positive closing speed <1.5 s, including negative gaps. Denominator includes finite HDVs without a closing leader. Pooled low-TTC rows: **6,280 / 2,039 / 1,959**; HDV rows: **887,025 / 894,588 / 895,357**. Pooled percentages differ from the equal-run table.
- **Hard braking:** continuously present HDV transitions, realized Δv/0.5 <−3 m/s², all lanes. Totals **487 / 5,794 / 5,505**. Recorded CAVs remain exogenous and can create contacts; zero overlaps is not established.
- **Post-insertion:** any continuous-present HDV lane change into lane 0 or 1, including mainline changes, not only ramp merges. Excludes the insertion transition itself; measures eligible transitions ending 0.5…6 s later. Report both the inserter's own braking and its fixed nearest-behind HDV follower's braking. Overlapping event windows can count a transition more than once; eligibility is checked per transition and can resume after a lane return. Truncated windows are excluded. Full definitions are embedded in [report.json](docs/r13e_offline/report.json).
- Inserter full-window events with eligible rows: **859 / 819 / 792**; with a hard brake: **7 / 80 / 52**. Follower events: **558 / 539 / 438**; with a hard brake: **1 / 56 / 17**. These pooled event ratios are not the equal-run event shares above. Inserter event-share coverage: 95 runs / 48 states for each arm; follower coverage: **89/46, 90/46, 86/44**. Paired event-share contrasts use the finite intersection, so they need not equal subtraction of the marginal means.
- **Stuck:** consecutive post-t0 lane-2 time within 0…75.44 m below 1 m/s, strictly >60 s. This differs from the historical “never merged by horizon” statistic; clock resets outside the condition. No pre-t0 wait is inferred.
- **Delay:** HDV speed-deficit proxy, sum 0.5·max(0,1−v/31.29), not travel-time delay. All lanes: upstream s<0, merge 0≤s<75.44, downstream 75.44≤s<500 m. Exits end exposure; this t0-roster study excludes later arrivals/source waiting.
- **Zone:** valid post-step HDVs in lanes 0/1, 0≤s<75.44 m. Occupancy is time-mean count; speed is row-weighted within each run. Lane-2 ramp vehicles are excluded from zone occupancy/speed.
- **Dynamics failure:** invalid/unphysical state or |a|>100 as defined by the runner. All arms have zero such runs, invalid HDV rows and bad-acceleration rows. This is **not** the older trajectory-error “blow-up” label and says nothing about trajectory matching.

## Round-12 BASE checks: flags off versus on

**All three S-R12C arms: REJECT under BASE-1; PASS BASE-2, BASE-3 and BASE-5.** BASE-4 is a read, not a pass criterion. These are the unchanged legacy BASE rules on this panel, with a separately strict-loaded **as-trained round-9** reference, SHA256 `af9bc6435aef3008d73af1bd8bfba1b3f2a035ca703760f38b962b8a23f9585a`. They do not reproduce the historical panel or a round-9 emission-zero counterfactual.

48 identical filled-presence windows, **423,329 all-mainline HDV rows per arm**; lane-1 class counts below. Bias = final predicted minus recorded next-row acceleration, m/s²; mean [95% window-cluster CI]. All reported classes have at least 18 contributing windows. The emergency and emergency+gs **full one-step arrays are byte-for-byte identical**, not merely equal after rounding, as expected for a replay-only veto.

| BASE class | round 9 as-trained | off | emergency | emergency+gs |
|---|---|---|---|---|
| Slow start (n=2,027) | 1.065 [0.947, 1.179] | 0.571 [0.524, 0.616] | 0.571 [0.524, 0.616] | 0.571 [0.524, 0.616] |
| Far-upstream slow (n=49) | 0.540 [0.361, 0.685] | 0.157 [0.024, 0.245] | 0.157 [0.024, 0.245] | 0.157 [0.024, 0.245] |
| Upstream slow (n=99) | 0.944 [0.185, 1.513] | 0.381 [−0.040, 0.715] | 0.381 [−0.040, 0.715] | 0.381 [−0.040, 0.715] |
| Free road (n=13,034) | −0.403 [−0.585, −0.213] | 0.204 [0.140, 0.274] | 0.204 [0.140, 0.274] | 0.204 [0.140, 0.274] |
| Last-3-s target follower (n=789) | 1.138 [0.959, 1.279] | 0.652 [0.562, 0.729] | 0.583 [0.457, 0.684] | 0.583 [0.457, 0.684] |
| Far partner (n=2,147) | 0.449 [0.293, 0.581] | 0.404 [0.335, 0.472] | 0.400 [0.332, 0.466] | 0.400 [0.332, 0.466] |
| Partner d_end>40 (n=4,733) | 0.154 [−0.101, 0.395] | −0.169 [−0.362, 0.032] | −0.291 [−0.500, −0.078] | −0.291 [−0.500, −0.078] |
| Alongside (n=3,844) | −0.158 [−0.386, 0.104] | −0.342 [−0.523, −0.143] | −0.605 [−0.792, −0.403] | −0.605 [−0.792, −0.403] |
| Recorded braking (n=12,939) | 0.590 [0.491, 0.714] | 0.602 [0.518, 0.708] | 0.509 [0.414, 0.625] | 0.509 [0.414, 0.625] |
| After merge zone (n=41,710) | −0.614 [−0.702, −0.517] | −0.062 [−0.090, −0.031] | −0.062 [−0.090, −0.031] | −0.062 [−0.090, −0.031] |
| All lane 1 (n=211,899) | −0.041 [−0.078, −0.006] | −0.098 [−0.169, −0.024] | −0.206 [−0.301, −0.105] | −0.206 [−0.301, −0.105] |

- **BASE-1 REJECT:** upstream slow-start |bias| 0.381 exceeds the literal 0.3 threshold, although all-position slow-start improves from round 9's 1.065 to 0.571. The CI crossing zero does not override this point-threshold rule.
- **BASE-2 PASS:** free-road 0.204 ≤0.503; far-upstream slow-start 0.157 ≤0.640, using max(0.3, round-9 magnitude+0.1).
- **BASE-3 PASS:** braking magnitude versus round 9 is +0.012 off and −0.082 with emergency, both below the rejection margin +0.1.
- **BASE-4 READ:** target-follower bias 0.652 off →0.583 on; neither is within ±0.3. Courtesy emission is off, so the corresponding head-zero read is identical.
- **BASE-5 PASS:** encoded accel/decel near-bound shares are both zero in all S-R12C arms. Per-arm percentiles are retained in [base_checks.json](docs/r13e_offline/base_checks.json).

| Paired emergency − off, m/s² | Bias change [95% CI] | MAE change [95% CI] |
|---|---|---|
| Recorded braking | −0.093 [−0.116, −0.072] | **+0.086 [0.065, 0.108]** |
| Last-3-s target follower | −0.069 [−0.132, −0.028] | +0.064 [0.023, 0.126] |
| All lane 1 | −0.108 [−0.137, −0.081] | **+0.108 [0.080, 0.136]** |

Lane-1 MAE is 0.411 off versus 0.518 on; recorded-braking MAE is 0.760 versus 0.845. Reduced braking **bias** is not improved MAE. In particular, the flags alone add >0.1 to lane-1 MAE on this panel, empirically supporting the review's requirement that any future mid-run comparator use matching enabled flags. gs−emergency is exactly zero in replay. All class MAEs and all paired class contrasts are in the JSON.

## gs equivalence: NOT EQUIVALENT

The actual GAPACC `scripts/factworker.py::install_gate(use_T=True)` wrapper was evaluated against R13E on **the same 256 recorded ramp insertion events**, with identical S-R12C encoder-inferred per-slot parameters from the exact pre-t0 history. Of these, **204 are HDV insertions**. Both float32 and float64 give:

| Event set | Same gate / final-boundary decisions | GAPACC allows, R13E vetoes | R13E allows, GAPACC vetoes |
|---|---|---|---|
| All recorded insertions | 136 / 256 | **120 / 256** | 0 / 256 |
| HDV insertions | 107 / 204 | **97 / 204** | 0 / 204 |

GAPACC uses `T_back*v_back + relu(v_back²/(2*b_back) − v_front²/(2*b_front))`. **R13E adds learned `s0_back`** on each side. It is a stricter rule, not a reproduction of tested `gs`. There were no log-versus-direct comparison discrepancies in these real events. Synthetic boundary cases additionally expose differing invalid-input handling and one-ULP log-rounding behavior: 4,096 heterogeneous cases per dtype give 935 differing boundaries; the synthetic counts are not empirical frequencies.

The hard veto has no learned multiplier; its implicit multiplier is one. Existing learned `tau` controls latent dispersion/soft surrogate, not the physical threshold. Source fingerprints, examples, decision counts and the method are in [gs_equivalence.json](docs/r13e_offline/gs_equivalence.json); separate synthetic evidence is in [gs_synthetic.json](docs/r13e_offline/gs_synthetic.json).

## Preregistration and configuration receipt

Appended, after re-reading, to `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md`, section **“2026-10-25 S-R13E preregistration — pinned-code offline study and prospective two-stage budget”**, plus the pre-outcome same-panel round-9/gs measurement addendum. Existing entries were preserved. The entry records purpose, exact runtime SHA, config hashes, init, two-stage budget, panel correction and mid-run stop rule.

- Stage-A config SHA256: `41aade0a1316a6aed42b254edd83fa072661b9c6f92b40abea70ef6c18aac575`.
- Stage-B config SHA256: `652a5cdccb786de3400f5391187b14c35c7cb2bd6f583cd750bd0d59eda2b587`.
- S-R12C train-config SHA256: `276f76fe378556daa18ecc3881fd10749d71db817b42b059d872009b32d4be1a`; physics SHA256: `86473e84c8f4338f4416b4307f3b63190d7860932f6ad33eb861746ff0129f85`.
- Prospective A: final S-R12C as-is, only core head/centre trained, emissions off, 30×120 s, batch4, val4, max2.5 h. Prospective B: A FINAL, core frozen, temporal off, 13C residual/repaired-courtesy recipe, 40×120 s +60×300 s, batches4/2, val16, max8 h.
- Registered checkpoint read: lane-1 realized hard-brake share >2×S-R12C or one-step lane-1 MAE >S-R12C+0.1; errors fail closed. **Owner amendment requires SAME-FLAGS S-R12C reference for future training.** Pinned gate still needs that repair, final-acceleration envelope enforcement, and resolution of the gs definition. These measurements do not declare the unrepaired gate launch-ready or supersede the later owner's emergency-only/weaker-yield decision.

## Operational notes

The first full-repository archive extraction was too broad and exhausted remote disk space. Only this invocation's copied archive and newly created partial deployment were removed; the minimal runtime allowlist was then deployed. The successful study uses only explicitly guarded TRAIN arrays, checkpoint sidecars and its own work directory. A pre-inference file-descriptor failure was preserved; the corrected own child inherited nofile=65536. Three initial one-step timing probes did not contribute outcomes. A legacy round-9 collector incompatibility was corrected by supplying descriptive pre-acceleration/zero-courtesy fields without changing its final acceleration or checkpoint tensors. Failed-attempt logs/receipts remain under `diag_emerg`; they were not substituted into results. Only an exact-identity pidfd for this task's idle postprocessing coordinator was signaled when refreshing that coordinator; no training or foreign process was signaled.

## Reproduction and evidence

From the prepared pinned runtime plus committed diagnostic overlay on gpuR:

```bash
L=/root/autodl-tmp/c4/v5
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
nice -n 19 /root/miniconda3/bin/python -B -u \
  "$L/repo_r13e_8c7ff0a1/tools/r13e_checks/offline.py" \
  --checkpoint "$L/runs/c4_wm_v10_round12c/wm_s.pt" \
  --train-root "$L/data/v5_native" --train-root "$L/data/genrand_plain" \
  --states "$L/diag_emerg/train_states48.json.gz" \
  --out "$L/diag_emerg/NEW_OUTPUT_DIRECTORY" --workers 4 --one-step-panel
```

Raise the child file-descriptor soft limit to 65536 first. `prepare_train_panel.py` runs against the original pinned offline loader **before** the overlay is installed. `remote_ops.py` transports workspace script sources over SSH without shell-quoting their Python bodies. `postprocess_remote.py` runs gs differential and same-panel round-9 BASE after the study Pools exit. `summarize_evidence.py` renders the rounded tables from the copied JSON.

Remote raw one-step NPZ tables and progress remain at `/root/autodl-tmp/c4/v5/diag_emerg/offline_train48_workers4/`. [Full empirical evidence](docs/r13e_offline/report.json), [panel receipt](docs/r13e_offline/panel_receipt.json) and [owned launch receipt](docs/r13e_offline/offline_owned_process.json) are committed alongside this report.

## Verification

- Windows CPU: `python -B -m pytest -q -p no:cacheprovider --noconftest tests/test_r13e_offline.py tests/test_r13e_secure_gap.py` — **96 passed, 4 Linux-only cases skipped**.
- gpuR CPU: `/root/miniconda3/bin/python -B -m pytest -q -p no:cacheprovider --noconftest tests/test_r13e_offline.py` — **70 passed**, including the real Linux fork cases; [byte-exact captured log packaged as JSON](docs/r13e_offline/linux_tests.json).
- `python -B tools/r13e_checks/verify_evidence.py` — **passed**: all **240** stored rollout estimates/CIs independently reconstructed from per-run rows, SHA256 validation (local text LF-canonicalized for Windows checkouts), 48 state/window matches, 288 arms, exact CRN seeds, panel-role checks and same-event gs counts.
- Final remote audit — **passed**: pinned `src` tree and checkpoint unchanged; round-9 replay geometry exactly matches S-R12C; all four one-step tables have 423,329 rows; emergency and emergency+gs NPZ SHA256s are identical; CUDA uninitialized. [Audit and file hashes](docs/r13e_offline/final_audit.json); [completion receipt](docs/r13e_offline/POSTPROCESS_COMPLETE.json).
- `python -B -m ruff check tools/r13e_checks tests/test_r13e_offline.py` and `git diff --check` — **passed**.

This completes the offline measurement/registration task. The scientific results do not qualify the unrepaired two-flag training recipe; carry the measured trade-offs, BASE-1 failure and gs mismatch into the separately assigned launch fixes/owner decision.
