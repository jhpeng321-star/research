# S-WMPLAN prospective live design — 2026-11-01

## Question and frozen choice
Does online WM-PHYS cooperative look-ahead improve source-inclusive full-team
delay over MAPPO on S-DISTILL's fresh development panel? Primary is expected
saving; strict empirical P(win)>=0.8 is secondary. Neither is a trained student.
No result justifies the proposed headline before this test closes.

Before ANY live outcome, reuse completed S-WMTEACH exact-quality predictions on
the same fixed 12 EXAM states. At H300 select FULL-v5 source b2df7458/recipe
023fa74a4d3059115ec0a5f28d39030a76137d871db5798eb1d23a13fc1e32c5 iff BOTH
helpful-state exact argmax agreement and mean exact chosen saving on all12
states are >= FULL-v4 22cc32f6. Also report helpful chosen saving. Otherwise v4. Complete matching states,
available options and exact targets are required. Save comparison, hashes and
selected backend before dispatch; never select from live outcomes. Do not repeat
the already-running peer quality forecasts or control any peer process.

## Predictor and execution
- Exact S-WMTEACH teacher-quality function `runtime.fit_futures(rt,None,r,k)`;
  eight complete futures per state, paired across every available O0..O4 action,
  H300. No learned/output correction, action pruning before prediction, or
  substitution of SUMO predictions. Missing/failed futures stop the experiment.
- Use exactly the validated roster-only input path: observed CAV telemetry/RSU
  detections, frozen FIT driver population, sampled uniform-32FIT source priors,
  cold motion/controller memory. Queue/history are NOT claimed observed. No
  scenario demand, actual hidden driver/type or realised arrivals enter forecasts.
  State/world identity only addresses deterministic RNG and cache identity.
- Identical observable states at the same clock/world share one immutable forecast
  across arms. Distinct states receive their own paired K8 forecasts.
- Same inherited group selection, rotating setting, O0=MAPPO, executors, safety,
  and cadence60 at t300..960; MAPPO tail through1200. H300 predictions may extend
  beyond the scoring endpoint, as in the inherited planner.
- Expected: minimum mean team delay among alternatives iff strictly below O0;
  tie by option ID. Strict: inherited minimum mean among P(win)>=0.8 eligible
  alternatives. Empirical K8 confidence is not asserted calibrated.

## Panel, reuse and exact budget
- Same24 worlds984000..984023/two per12cells; canonical panel SHA256
  72776ff122956f021d1caf2db9622674b0cde2a74af8bd9f20c3de5ba0a383a6.
- Reuse ALL S-DISTILL completed bitwise MAPPO baseline episodes. New MAPPO
  episodes0, new mechanics0. Preserve the original mechanics receipt and bind
  each reused baseline's bytes, runtime and checkpoint. Own planner warmup must
  match reused baseline bitwise; all-O0 episodes must match final prefix bitwise.
- Exact upper199200<=200000 sim-s: 48planner episodes*1200=57600 plus
  48independent world-arm group-credit allowances*2950=141600; 800unallocated.
  No outcome-adaptive transfer, refund, scientific retry or top-up.
- Each proposed executed two-member group first reserves its COMPLETE exact
  H120 quartet: reference O0, joint, member0 alone, member1 alone, each from the
  full executed-history prefix. Cost4*(t+120) includes every replay/warmup step.
  Each branch checks BITWISE physical/controller prefix and observable roster.
- If its own world-arm allowance cannot cover the COMPLETE quartet, execute O0
  and explicitly log proposed action/resource refusal. Forecast all options first.
  This is a budget-limited planner estimand: at most one early group per world-arm
  can fit, and no new group aftert600 fits. Report this important restriction,
  both raw policy proposals and executions, and every quota refusal. It is not a
  behavioural decision threshold or evidence about an unconstrained planner.
- All selected group moves have complete C_G=(J0-JG)-sum_i(J0-Ji). Complete the
  quartet before physical execution, never use its outcomes to choose actions.

## Runtime, checks and analysis
Immutable gpuA root /root/autodl-tmp/wmplan20261101_v1; SUMO on host CPUs,
WM inference on gpuA GPU; no gpuR output. <=2nice19/single-thread CPU episode
workers, <=2isolated creditworkers, <=2GPU services each strict8GiB. Resource
admission load<=20, available RAM>=8GiB, disk>=2GiB, GPUfree>12GiB before load.
Freeze source/model/artifact/runtime/checkpoint/baseline/design hashes; run CPU
contract tests and BOTH guards for all24worlds with SUMO creation blocked.
Durable registry/design/model freeze must precede one-use dispatch. Exclusive
claims/charges; any own fault writes durable STOP, preserves partial evidence,
and prevents automatic retry. All created PIDs bind boot/startticks/command.

Paired J600(t300..900)/J900(t300..1200): MAPPO-minus-planner veh-s/world,
10000whole-world paired bootstrap resamples/RNG20261101, primary and secondary
reported separately. Report all24pairs, source/on-road decomposition, moves by
type and world, quota refusals, all-group C_G and D_G, positive-beneficial C_G,
collisions, teleports, TTC<3 exposure and hard braking. No efficacy early stop,
tuning, optional continuation or model switch. Compare S-DISTILL on identical
worlds; include WM student only if frozen complete same-panel results exist.
Existing SUMO-oracle COOPPLAN panels differ: descriptive unpaired context only,
no new oracle simulation and no matched-panel oracle superiority claim.

Never read forbidden paths/truth/type assignment files or forbidden seed blocks;
no peer PID action, deletion, new environment, reset/clean/stash-drop or push.
Commit each completed preparation/dispatch/analysis step. Deliver WMPLAN.md with
plain language first and append the shared registration with exact plan/results.
