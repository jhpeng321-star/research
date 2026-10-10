# Externally released RL damping: graph and provenance audit

**Decision: a reproducible candidate is available for a bounded transfer check;
no accepted damping controller or exact paper-checkpoint identity is established.**
The owner has clarified that escalation is the principal contribution. Reusing
an existing trained baseline is preferable to inventing another damping
reference. The failed v1, v2 and v2.1 evidence remains preserved.

This task performed source, graph and CPU inference inspection only. A separate
coordinating task owns the bounded IDM/5--30% penetration check in
`runs/probe/published_idm_fit_check_2026-09-06/`. Do not launch overlapping
simulations or extend that check from this audit. No training, acceptance data
or wall generation was used here.

## Reproducible source and limits of attribution

Four ONNX candidates were retrieved from the MIT-licensed
[CIRCLES algos-stack repository](https://github.com/CIRCLES-consortium/algos-stack/tree/3b3ee3b91afdae93075c8b90334d65748cd9ff74),
pinned at `3b3ee3b91afdae93075c8b90334d65748cd9ff74`. Every downloaded file was
verified against its Git blob SHA-1. The raw URLs, SHA-256 values and sizes are
in [download_manifest.json](../runs/probe/published_damping_reuse_audit_2026-09-06/download_manifest.json).

The four filenames are `09_04_rl_finaldecn3p0_maxacc0p25`,
`09_04_rl_ttc4p0scale2p0_maxacc0p25`,
`09_04_rll_ttc7p0scale1p8_maxacc0p25`, and
`09_05_rll_ttc6p5scale1p8_maxacc0p75`, each containing `super_resolution.onnx`.
They were added as binary-only commits by Eugene Vinitsky on August 4--5, 2021:
[first](https://github.com/CIRCLES-consortium/algos-stack/commit/b7d699fd4274ff84bc583447f230cc3e65034117),
[second](https://github.com/CIRCLES-consortium/algos-stack/commit/bd859cd73b1c7db9fd3a8da813c82f6ab7114d85),
[third](https://github.com/CIRCLES-consortium/algos-stack/commit/d58018525e6d44eb8a4f8b9af4276083313e680a),
[fourth](https://github.com/CIRCLES-consortium/algos-stack/commit/3b3ee3b91afdae93075c8b90334d65748cd9ff74).
Neither their apparent folder dates nor parameter-like filename suffixes are
reliable specifications. The commit descriptions also disagree with some
filename values. The graph is the executable authority.

[Lichtle et al., ICRA 2022](https://bayen.berkeley.edu/sites/default/files/deploying_traffic_smoothing_cruise_controllers_learned_from_trajectory_data.pdf)
links a [training release](https://github.com/nathanlct/trajectory-training-icra).
That repository documents PPO, four 64-unit tanh layers, 0.1 s training physics,
one AV followed by five IDM vehicles, and 100 s trajectory clips. Its current
tree has no trained model binaries, releases or tags supplying the missing
checkpoint; the README states that the energy model was removed. Its
[export script](https://github.com/nathanlct/trajectory-training-icra/blob/eebf116a57d90e2134babf58e656b4e7955cfb66/export_to_tf.py#L23)
normalizes `[v, v_leader, gap]` by `[40,40,100]`, clips acceleration to
`[-3,1.5]`, and exports **v + 0.05 a**. The four audited binaries instead
return **a** after additional rules. Matching architecture and normalization
support a plausible lineage, not identity of the trained weights.

[Bhadani et al. 2023, section 4.2](https://fangyuwu.com/data/bhadani_2023.pdf)
describes a modified Lichtle controller and PyTorch-to-ONNX deployment and links
`algos-stack/onnx2ros`. It does not map these filenames/hashes to its evaluated
controller. Therefore use the description **"frozen, externally released
CIRCLES RL controller candidate, evaluated in our merge setting"** until an
exact paper-to-checkpoint bridge is obtained. Do not borrow the paper's effect
sizes for our controller or imply the merge transfer is already validated.

The `cra` README calls its controller a Time-Headway Follower-Stopper, but its
rename history traces the file to an RL-acceleration-labelled asset. Labels
alone establish neither a purely analytical implementation nor a particular
published trained checkpoint. It was not substituted into this check.

## Verified model interface

All four graphs have **exactly the same neural weights**, including every layer
and bias. The combined named-weight digest is
`58c8ae422bd873744c5cbcc12354b073cbde097605bcdb8722b420d7e0c700ca`.
These are four rule wrappers around one neural controller, not four independent
training seeds or four independently trained policies.

- Input: one float32 row `[ego speed in m/s, leader speed in m/s, bumper gap in m]`.
  Divide by nothing in the adapter: the graph already divides by `[40,40,100]`.
- Neural core: four 64-unit tanh layers, linear scalar head, acceleration clip
  `[-3,1.5] m/s2`; no recurrent state or online counterfactual input.
- Final output: one absolute acceleration in m/s2 after rule mixing and another
  `[-3,1.5]` clip. There is **no addition of ego speed, no integration timestep
  and no approximate-IDM residual addition** in any of these four graphs.
- The upstream ROS reader passes raw state but names its output topic `v_des`.
  That topic name is inconsistent with these specific graphs' acceleration
  output and cannot establish their intended actuator contract.
- The ROS prompt reader publishes at 20 Hz. This is documented deployment
  cadence, not verified training cadence for these exact weights. The related
  ICRA 0.1 s training and 0.05 s export integration are different quantities.

The coordinating task selected the **first listed candidate** before its
physical outcomes; this audit does not choose the best-performing wrapper.
Its whole-file SHA-256 is
`762fde0763c693c0e88f41b98b22fc0a11135a312b78c4c567e6211a3035ccee`.

## Exact singleton wrapper interpretation

Let A be the clipped neural acceleration and S(x)=1/(1+exp(-x)). The graph
extracts approximate column values with maxima after dividing other columns
by one million, rather than using direct indexing. Its exact proxies are:

`V=max(v, v_leader/1e6, gap/1e6)`;
`VL=max(v/1e6, v_leader, gap/1e6)`;
`G=max(v/1e6, v_leader/1e6, gap)`.

For ordinary positive road states these equal the intended columns. At zero
and extreme values they can differ; zero-speed handling must follow the graph,
not a presumed ideal formula. Define Q=S(0.1*(G-120)), the large-gap blend.

The first candidate uses **time headway**, H=G/V, not TTC:

`P=S(5*(2-H))`;
`B=(1-P)*A + P*(0.9375*H-2.375)`;
`output=clip((1-Q)*B + Q*0.5, -3, 1.5)`.

The second candidate uses `T=G/max(0.1,V-VL)` and `P=S(4-T)`:

`B=(1-P)*A - 3*P`;
`output=clip((1-Q)*B + Q*0.5, -3, 1.5)`.

The third uses `P=S(1.8*(7-T))`, first blends `F=(1-Q)*A+Q*0.5`, then
returns `clip((1-P)*F-3*P,-3,1.5)`. The fourth instead uses
`P=S(1.5*(6.5-T))` and large-gap acceleration 0.75. These ordering differences
matter: the last two apply TTC braking after large-gap mixing.

The artifact stores exact float32 constants. A separate scalar reconstruction
agreed with CPU ONNX inference within **2.17e-7 m/s2** across the finite
engineering probes. This verifies the interpretation, not physical efficacy.
For the first model, `[5,20,30]` returns **+1.499877**, `[20,5,30]` returns
**-1.048372**, and `[15,15,30]` returns **-0.353354 m/s2**. Thus its filename
does not impose a +0.25 m/s2 acceleration ceiling, and equal leader/ego speed
does not imply zero acceleration.

## Batch and boundary defects that an adapter must handle

All four graphs execute `[N,3]` input as an **N by N output**, because the
one-dimensional rule vectors broadcast against an N by 1 network output.
With the first model, the diagonal happens to match separate singleton calls.
The other three also reduce maximum closing speed over the batch, so even
the diagonal differs from singleton results by as much as 4.50 m/s2 in the
four-row probe. **Use singleton inference only; taking a diagonal is not a
general repair.** Upstream ROS itself forces one row.

The first model returns **NaN for `[0,0,0]`** through 0/0 headway. Missing
leader detection must not be encoded as this triple or invented ground-truth
state. Validate observations and output; an explicit native fallback must
release any owned command. The bounded adapter records its invalid-output
fallback separately. Finite outputs at the other probes are not a safety
guarantee. These wrappers contain no verified emergency-stop guarantee.

## What the transfer check can establish

The local inputs are CAV-acquirable under explicit leader sensing assumptions.
This supports implementability; it says nothing by itself about damping
benefit, reliability under occlusion, or network-wide service. An RSU may
later combine CAV reports for escalation, but no native outcome or online
counterfactual belongs in this controller's observation.

The coordinating task's adapter uses 1 s commands, 0.5 s SUMO physics,
same-lane 100 m sensing and native IDM background vehicles. Its workspace-bound
arm additionally projects acceleration/jerk; a separately labelled direct
`[-3,1.5]` arm diagnoses that projection difference. These are explicit transfer
assumptions and interventions on execution, not faithful reproduction of the
original field stack. Report requested versus executed actions and clipping
alongside braking, effort, service, source waiting and safety. Compare paired
native outcomes on exposed development cases; do not relabel them acceptance.

The current check can reject an unsuitable baseline or justify a subsequent
prospectively specified evaluation. It cannot prove that the original published
method fails, that 5--30% penetration is generally sufficient, or that the
controller prevents collapse. Frozen source/model hashes and the exact adapter
must accompany any eventual result. Acceptance data remain sealed and wall
training remains paused until a baseline is accepted.

Reproduce this audit with `tools/audit_published_damping_onnx.py`. It uses an
isolated ONNX parser under the audit directory and CPU ONNX Runtime; no SUMO
or learning is invoked. Graph dumps, exact constants, per-weight hashes, all
singleton/batch probes and boundary outputs are in
[graph_inference_audit.json](../runs/probe/published_damping_reuse_audit_2026-09-06/graph_inference_audit.json).
