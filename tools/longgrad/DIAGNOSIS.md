# Initial CPU/code diagnosis (2026-10-30)

The supplied s8/s9 update-191 receipts give pre-clip actor norms 230145616 /
0.291780204 = 788763642x. Residual head 229677709; own encoder 6784934;
report encoder 3197711; attention 9126615; group encoder 8694330. The head
accounts for 99.59% of squared norm. Long/dense scalar-credit ratio is 82.73,
and return scales are essentially identical (2418.276 / 2417.776).

Only 1 of 5 nonzero-coefficient rows passes the detached gain-only gate. Which
exact row index/future/step caused the norm is not present in these receipts. The signed
joint q10 selector uses only two futures per start (interpolation weights .3/.7).
One further deduction: gain-only accepts positive paired improvements, yet
the logged weighted long credit is negative. With exactly one ON row, it must
be a negative-coefficient flip-one row (-10.3), not ALL or the group row. Its
exact index and the relative contributions of the two futures remain unknown.
The current candidate-A config has since retired flip rows (n_flip=0, w_team=0),
so the stopped-actor receipt is not a measurement of the current production rows.
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

`$env:PYTHONPATH='src;.'; .\.venv\Scripts\python.exe tools/longgrad/diagnose.py
--horizons 10 30 60 90 120` runs an untrained hybrid WmSModel with its learned encoder, four
synthetic slots (two HDVs), no entries, no production assets, no decay. Actor
norms of synthetic summed reward are 12.9447, 128.594, 401.297, 674.002, 946.707. Mode 0
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
