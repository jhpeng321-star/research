# S-GROUPGRAD-1 — frozen group pathway confirmed

**Root cause:** native-init deliberately trains only a 137-parameter linear residual
head over frozen random features. This prevents learned group representation,
but cooperation credit itself is not globally disabled.

## Checkpoints

Reconstructed cycle0 init with seed965140; all15 non-residual actor tensors and
all26 group tensors exactly match c00. Against init, c00 `1b9949cc`, c01-ON
`64f70af4`, c01-OFF `e1fa4881`, c02-best `44f3eb66` change **only two actor tensors**.
Maximum residual-head deltas: **0.001078/0.005763/0.005184/0.011308**.
Group encoder and other actor features never change; lane heads are absent.
All30 critic parameter tensors change, with maximum deltas
0.01964/0.04088/0.04088/0.08927 versus constructor init, before critic warmups;
the scale buffer stays fixed. Cycle2's critic wrapper prefix is normalized.

## Loss and gradients

`policy.py:107` freezes existing actor networks; `dreamer.py:658-660` freezes
and excludes the group encoder from Adam. Independently,
`policy.py:184` uses `residual_head(features.detach())`.
The group signal is used numerically: zeroing it changes ON residuals by
mean0.00427m/s², but no group gradient passes.

`dreamer.py:1331` minimizes negative normalized weighted counterfactual returns.
Final effective ON individual/group/coop/team/all weights are
**8.9/9.9/8.9/10.3/1**; real-credit multipliers are1. OFF is0/0/0/0/1.
Groups exist in142/150 canonical cycle1 updates; team credit requires triples.
Policy14, longcritic and paircrit do not reconnect the encoder; native lanes,
zero entropy gradients and report-only WM safety add no actor training signal.

CPU tiny-config **actual WM/actor/trainer** update: group gradient0, residual
gradient4.8873. Four weighted credit terms combined contribute4.7250 versus
total4.8873; pure synergy contributes only2.15e-8. Six production cycle1
unit-weight synergy/all gradient ratios range7.53e-12–3.10e-5.
Near-native effects are approximately additive; credit mainly rescales a shared
gradient. The tiny test has cosine≈1 and first-step Adam ON/OFF delta difference
3.17e-9, explaining why larger credit weights need not separate policies.

## Residual size

Across all26 cycle1 TRAIN starts/410 CAV observations, requested mean|residual|
is **0.0105/0.0464/0.0434/0.1026m/s²** across those checkpoints;
WM-native head mean|command| is0.2578. ON p95 is0.0681; ON–OFF mean difference
is0.00300. Thus0.004 describes the arm difference, not the whole residual.
Decode is **1.2×tanh(latent)**, not a milliscale clamp. Zero initialization,
Adam lr5e-5, frozen features and similar gradients yield small, unsaturated
outputs; safety/execution envelopes can further restrict realized changes.
The exact E5 trajectory tie is not causally isolated by these offline tests.

## Minimal fix and confirmation

Propose default-OFF `native_residual_train_group`: enable/include group parameters
and detach only own/context features, preserving differentiable group signal.
Retain zero residual initialization and native execution. An isolated in-memory
prototype passes: update1 group gradient0 is mathematically expected with zero
output weights; **update2 gradient0.001169 changes all26 group tensors**.
Before tuning lr/schedule, log per-module weighted gradients and physical
residual growth; no behavioral thresholds are needed.

Evidence and reproducible assertions: `tools/groupgrad/`. CPU-only, ZERO SUMO;
no production/loop changes or push.
