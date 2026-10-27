# Source and execution evidence

## Freeze and forward

- `src/wms/control/policy.py:103-110`: native-init freezes existing policy
  parameters, then creates a zero-initialized trainable `residual_head`.
- `src/wms/control/dreamer.py:655-660`: native-init also freezes the group
  encoder and filters it out of Adam. Saved optimizers have exactly two tensors,
  137 scalar parameters, lr `5e-5`, steps 30/180/180/380. No prefix-based
  selective unfreezing overrides these decisions.
- `policy.py:168-192`: the group signal is concatenated with own/context features,
  so it is numerically used; `features.detach()` independently stops its gradient.
  Merely adding the encoder to the optimizer cannot fix this.
- `policy.py:443-521,542-592` and `nativeinit.py:48-53,95-118`: deployment uses
  the same residual head and native-relative command semantics. Zeroing the group
  signal on all 26 cycle1 TRAIN starts changes ON residuals by mean absolute
  `0.0042703 m/s2`: frozen features are used, rather than bypassed.

## Actor objective and other components

- `dreamer.py:1307-1358,1554-1613`: minimize negative weighted row-return
  differences divided by a detached return-range scale, plus entropy/constraint.
  Native counterfactual baselines are detached; policy rollout returns remain
  differentiable. Critic parameters are frozen in the actor bootstrap, but its
  state derivative remains available. Critic regression uses detached states/targets.
- `counterfactual.py:360-385`: ALL gets `w_all`; each solo gets `w_individual`
  minus `w_coop` for its group; each group gets `w_group+w_coop` plus one `w_team`
  per flip; flips get `-w_team`. Team credit is omitted for pairs. With final ON
  weights, solos cancel, a triple group gets 49.7, each flip -10.3, ALL 1.
  `n_individual=0` does not eliminate solos: `cover_group_members=true` adds them.
- `policy14.py:54-77`: real-credit multipliers apply before this objective; saved
  cycle1 multipliers are all 1. Marginal score-function lane terms require lanes;
  native-init has no lane head. `coopw.py:460-527` provides unit-weight term audits,
  not an additional loss. Logged effective ON weights reach individual/group/coop/
  team/all = 8.9/9.9/8.9/10.3/1; OFF has 0/0/0/0/1.
- `dreamer.py:790-799`: first-update curriculum anchoring persists on resume.
  `logs.json` preserves separate historical/replayed files; canonical cycle1 uses
  c01 updates31-170 plus c01.a3 updates171-180, exactly150 updates. Group terms
  have rows in142 updates; team triples in97. Six pure-synergy/all unit-gradient
  ratios range7.53e-12 to3.10e-5 (median1.40e-8).
- `longcritic.py:109-141,180-222` fits a separate native reference/critic loss;
  `paircrit.py:244-264` wraps the critic. Risk-conditioned actor replacement is
  conditional and absent in these checkpoints (unchanged26 group tensors/two
  optimizer tensors). Neither enables the frozen encoder. Paircrit's `base.`
  prefix is removed only for the numeric critic comparison in the census.
- Native-init log probabilities are identically zero, hence entropy gradients zero;
  WM safety is report-only in main. Multiplicative exploration leaves a zero head
  exactly zero. Production actor gradients nevertheless are nonzero. Small residuals
  and nearly additive returns make synergy weak; positive rescaling of a gradient
  is largely normalized away by Adam. This is mechanistic evidence, not a proof
  that these factors alone explain the exact realised E5 tie.

## Reproduce the CPU diagnosis

Own CPU root: `/home/ubuntu/groupgrad`; no detached job or GPU execution.
`operate.py` implements read-only gpuA input transport, SHA verification, and staging.
The native-init reconstruction uses seed965140 and verifies all15 non-residual
actor tensors and all26 group tensors against c00. Critic init is constructor
init, before its real-return/longcritic warmups; its30 parameter tensors change
in every checkpoint, whereas its value-scale buffer stays constant.

Run `checkpoints.py` then `diagnose.py` with CPU torch and the isolated `runtime`
source. Set `CUDA_VISIBLE_DEVICES=`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
and `nice -n 19`. The source receipt verifies deployed/unit AST equivalence for
policy, dreamer, policy14, counterfactual, coopw, longcritic and bridge. Paircrit
is separately traced, not enabled in the tiny test.

The tiny test uses the real WM, actor, group encoder, reward and trainer, six
synthetic vehicles, a three-CAV group and three seconds of imagination. It asserts
nonzero residual gradients, zero original group gradients, weighted gradient
decomposition, and nonzero group gradients/changed tensors on candidate update2.
The candidate is an **in-memory, default-OFF proof**, not a production patch:
unfreeze/include the group encoder and pass `signal` without detaching while
keeping own/context features detached. Both zero heads remain native at update0;
group gradient zero on update1 follows mathematically from zero output weights.

`diagnosis.json` includes all26 converted cycle1 starts/410 CAV observations for
physical residual measurements, before safety/caps. Native comparison is the exact
learned WM-native head, not a claim about final SUMO junction/stop execution.
