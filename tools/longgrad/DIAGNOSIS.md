# Initial CPU/code diagnosis (2026-10-30)

The supplied s8/s9 update-191 receipts give pre-clip actor norms 230145616 /
0.291780204 = 788763642x. Residual head 229677709; own encoder 6784934;
report encoder 3197711; attention 9126615; group encoder 8694330. The head
accounts for 99.59% of squared norm. Long/dense scalar-credit ratio is 82.73,
and return scales are essentially identical (2418.276 / 2417.776).

Only 1 of 5 nonzero-coefficient rows passes the detached gain-only gate. Which
row/future/step caused the norm is not present in these receipts. The signed
joint q10 selector uses only two futures per start (interpolation weights .3/.7).
It multiplies by K=8, canceled by the subsequent mean over K; this is correct.
Row coefficients already average over starts. Long sums the entire horizon;
dense averages finite-horizon suffix returns. Neither an extra row-count nor
horizon division is missing under the registered objective.

Replay sums the disjoint active-future VJPs without an extra K factor. Mode-2
reward VJP detaches each input occurrence and differentiates the original call;
pair-bias checkpoints the same pure MLP. No mis-scaling found in code. Existing
LONGMEM300.md also records GPU dense-90 equality with both flags ON vs OFF;
that is evidence against a CUDA-only recipe bug, not a proof at 300 s.
follow_speed already includes 184542ab's inf/tiny-gap frozen branch. TTC divides
by floored closing speed, not gap; it is report-only in candidate A and not part
of the long reward. Zero cotangents can nevertheless expose nonfinite internal
Jacobians; finite receipts cannot exclude every state-specific dynamics bug.

`PYTHONPATH=src;. .venv/Scripts/python tools/longgrad/diagnose.py --horizons
10 30 60 90 120` runs an untrained real WmSModel with its learned encoder, four
synthetic slots (two HDVs), no entries, no production assets, no decay. Actor
norms of summed reward are 12.9452, 128.596, 401.302, 674.009, 946.716. Mode 0
and mode 2 values/norms match exactly at every horizon. Growth is approximately
linear after 30 s, NOT the catastrophic production explosion. This scene does
not reproduce the congested/entry/merge trained-WM trajectory. Do not claim
that it establishes a Lyapunov growth rate or a particular faulty step.

Most defensible fix: objective-preserving, default-off optimizer preconditioning:
separate dense and long actor gradients and cap the latter's norm at the former.
Keep the registered scalar loss and all gate/tail decisions. Add exact term,
module, future, row-selector and state-cotangent diagnostics to localize the
production sensitivity without extra GPU experiments. Defer state-adjoint
clipping until those diagnostics establish a problematic state path; it changes
the pathwise derivative and has no demonstrated benefit on this CPU scene.

Adam sees the post-clip gradient (norm <=100), not 2.3e8. Relative to dense .29,
its squared-moment impulse can still be ~1.2e5 larger, decaying on beta2=.999's
~1000-update time constant. The raw clip factor (~4.35e-7) suppresses dense credit.
