# S-COOPPLAN consumer / execution handoff

## Implemented API
- `tools/coopplan/predictor.py`: `Predictor.predict(ObservableState, actions, horizon, K)`.
- `SumoOraclePredictor(run_full_rerun)`: K1 exact deterministic paired outcomes.
- `WMPhysPredictor(rollout, draw_factory)`: sample complete futures once, reuse them for every action.
- `CompleteDraw(draw_id, payload)` supports identity-bound tensor-backed WM future objects;
  plain JSON-compatible draw mappings are also accepted. Payload copies isolate action mutation.
- Bind RB-INT's immutable source/config/checkpoint and RB-UNC full hidden-driver/arrival
  builder at the adapter boundary. `rollout(state, action, H, draw)` returns `team_delay`,
  `breakdown` (bool/None), `member_delay` and `safety` dictionaries.
- Same `Planner` handles both backends; set K=1 for SUMO, K as registered for WM.
- `wmdecide_check(prediction)` invokes the actual S-RB-DECIDE Future/decide API.
  Parity with the planner was verified in all three frozen host environments.
- Future WM live integration must populate public 60s motion/history and observed request counters
  in `ObservableState`; the current demonstration uses only the SUMO replay adapter.
- A WM empirical win probability is a model forecast, not proof of calibration.

## Frozen demonstration
- Original design at a0934600 (local file now adds an extension pointer):63lines, SHA256
  `53b0edf4ea8e6fc72165691727e923505ed6a805af9c0294efd63f2ab3389a5f`.
- Registered fresh development worlds983100–983111, never fitting; exact upper704760sim-s.
- MAIN approval is present in shared REGISTRATION.md; own preregistration appended before physics.
- Runtime source is immutable a0934600 + staged peer modules/AST-extracted unchanged LRDIAG BIC.
- COOPFREE learned checkpoint absent at freeze; no outcome-dependent late addition to this panel.
- Host roots: cpuQ `/home/ubuntu/coopplan20261101_v1`;
  gpuA/gpuR `/root/autodl-tmp/coopplan20261101_v1`.
- Own production supervisors: cpuQ458354, gpuA327915, gpuR744203. Do not relaunch.
- All hosts passed16 CPU tests/0skips and real S-RB-DECIDE parity with SUMO execution blocked.
- Registered2040sim-s mechanics PASS; CPUQ's first production baseline additionally must
  match the complete1200s mechanics baseline bitwise before its planner starts.
- `python -B -m tools.coopplan.ops status --host <host>` reads public status only.
- `python -B -m tools.coopplan.report` validates completed paired worlds and reports progress.
- `python -B -m tools.coopplan.report --collect` produces immutable final receipts, CSVs,
  exact sample credits, unchanged detector evidence and figures ONLY after all12 pairs complete.
- Never retry an existing attempt, clear STOP, change frozen source, refund charges or signal peers.
- Post-dispatch local WM-payload/test/report additions do not mutate the frozen oracle source.
- Original preparatory local contract suite18/18 PASS, including half-open BIC prehistory bins.
- Future COOPFREE adapter passed a real frozen-host CPU ABI check with synthetic test weights,
  a three-CAV input pruned to the frozen nearest-two group, and process/SUMO creation blocked.
  This is an adapter test, not a learned-policy or rollout-efficacy result.

## Result interpretation
- Short horizon is120s, decisions60s at300..960; MAPPO-only tail1020..1200.
- J600/J900 exclude native300s warmup. Full-team delay includes source waiting.
- Sample first executed group per world, H120: positive `C_G` is superadditive delay saving.
- `C_G>0` and behaviour counts are separate from actual long-horizon performance improvement.
- A result report must retain negative/null effects, unsampled worlds and all safety metrics.

## COMPLETE — original panel / first-action-only public results
- All12 pairs complete; actual=charged220980sim-s; no STOP/failure/retry/incomplete attempt.
- Paper report: `docs/paper_results/COOPPLAN.md`; immutable artifacts `docs/paper_results/coopplan/final/`.
- J600 saving396.280veh-s/world,95%CI[0.894,1128.773]; J9001195.148,CI[-568.178,3724.114].
- 19 executed group actions,6 first-executed samples; C_G positive2/6 but mean-38.379,CI[-94.759,-1.022].
- Positive examples are small:983108/t600 +.198734 and983111/t300 +.014360veh-s.
- One paired-policy-mean-supported yield +18 gap events; no executed O2 wall option.
- Collisions/teleports0 both arms; braking220→222veh-s, TTC exposure7004.5→6623veh-s.
- Verdict: functioning predictor/planner + positive short-score result; robust superadditive cooperation NOT established.
- Frozen dev panel must remain never-fitting; do not select/tune candidates or WM settings from these outcomes.
- Full final local CPU/analysis suite20/20PASS; rendered figures visually checked.

## MAIN01:28 extension — COMPLETE, all36 paired worlds verified
- Preserve the complete original panel and roots above. Resume reconciliation commit97a8b34b.
- Extension source/design/preregistration commitc2646eb1; separate immutable `*_v2` roots.
- Exact cap579020sim-s plus original220980=800000. Fresh DEV983200–983223, never fitting.
- Design `COOPPLAN_EXTENSION_DESIGN.md` SHA25617356a3ee726ffea189da8d1f0e10cb376e4eae1293cfdaf1014c68a8a826918.
- All executed extension groups received two solos; backfilled13 missing original group credits
  (22560sim-s) using original states/O0/joint outcomes. No original episode is rerun.
- Every pair and solo obligation reserved before lookaheads; zero quota-pruned decisions.
- COOPFREE no complete learned update at shared freeze; no untrained substitute/late switch.
- Three-host safe-name/registration audits PASS. All three immutable hosts27CPU tests/0skips
  plus real WMDECIDE parity PASS, SUMO process creation blocked; local29CPU/analysis PASS.
- CPU/binding/freeze/audit manifests: `docs/paper_results/coopplan/extension/`.
- Read-only: `python -B -m tools.coopplan.extension status --host <cpuQ|gpuA|gpuR>`.
- Dispatched once: cpuQ565742, gpuA335613, gpuR756955; immutable `*_v2` roots.
  All COMPLETE, no STOP/incomplete attempts; pools drained and zero outstanding reservations.
  NEVER relaunch these supervisors or the original ones; dispatch receipts are committed.
- Final report `docs/paper_results/COOPPLAN.md`; immutable result/CSV/figures/QA under
  `docs/paper_results/coopplan/extension/final/`. All result bytes SHA-bound/Git-preserved.
- Extension24: J600 saving+774.471 [199.815,1480.502], J900+1778.158 [714.789,3076.597].
- Pooled36: J600 saving+648.407 [209.699,1173.616], J900+1583.821 [588.246,2765.414].
- All52 executed groups have104 member-alone twins;24/52 C_G positive. Pooled mean
  C_G+21.771 [-42.074,90.467] remains uncertain; extension+7.757 [-86.168,112.091].
- Material cooperation case983223/t960/O4: joint D_G+568.333707, solos-302.399975/-18.034632,
  hence C_G+888.768314; both alone harm but the joint action helps. All negative cases retained.
- Pooled causal COOPBEHAV:7supported yield/112gap/2metering/345wave events; O2wall0executed.
- Pooled safety MAPPO/planner: collisions0/0,teleports0/0,TTC21662/20280.5,braking765/733;
  safety clamped1446/3557 requests. All603full-prefix checks pass.
- Actual=charged565140sim-s total (original220980 +extension344160),234860unused under800000.
  Extension:57600paired +22560original backfill +53520new twins +210480lookaheads.
  Hosts actual/charged:cpuQ78480,gpuA101520,gpuR164160, zero outstanding; no retries/refunds.
- Backfill13/13 COMPLETE: all19 original actions now have twins; event-weighted mean C_G
  +46.111598, whole-world-cluster95%CI[3.664586,109.339097],10/19positive in6/12worlds.
  New immutable receipts: `coopplan/extension/original_all_groups/`; original six-sample
  artifacts retained. The complete-action estimand differs from the first-action sample.
- CPUQ resource notice in shared `COOPPLAN_RESOURCE_NOTICE.md` and REGISTRATION:
  disk briefly fell below immutable2GiB guard; recovered to2621468672bytes at
  Unix1793469380 and existing workers resumed automatically; all24 pairs now complete.
- Reproduce: `python -B -m tools.coopplan.extension_report --verify` (idempotent/read-only
  after QA exists), `--examples` for exact extrema; local35CPU/analysis tests passed.
  Never recollect into an existing immutable result directory; no live source was changed.
- Final interpretation: robust dev-panel delay benefit and substantial exact cooperation
  cases; robust positive mean superadditivity is not established on extension/pooled panels.
  WM-PHYS transfer/calibration remains a future backend qualification, not this result.

## MAIN02:21 H300 continuation — DISPATCHED ONCE
- Resume37380f46, prospective design/source03256981, all-host CPU bindings3cc7128a.
- All prior H120 results/roots preserved; no prior rollout repeated.
- Fresh confirmation18DEV983300–983317 + separate wall6DEV983318–983323;
  allH300/cadence60, original strict0.8 gate/options/safety/noise, ALL-group twins.
- Exact continuation cap234720:78240per host/9780per world, global upper799860.
  Design `COOPPLAN_H300_DESIGN.md`, receipts `coopplan/h300/`;140unallocated.
- Corrected HOLD COOPFREE learned checkpoint absent at shared predispatch freeze;
  no legacy/zero-update replacement or outcome-dependent checkpoint addition.
- New immutable `_v3` roots: cpuQ693091, gpuA348726, gpuR768580.
  All three30CPU/0skips and actual WMDECIDE parity PASS before any dispatch.
- NEVER relaunch these supervisors. `python -B -m tools.coopplan.h300 status --host HOST`
  is read-only. Budget pruning is expected; availability logged before admission.
- Next: immutable result collection, separate confirmation18/wall6 J600/J900,
  O2 available/evaluated/pruned/chosen and D_G/C_G; whole-world bootstrap and safety.
