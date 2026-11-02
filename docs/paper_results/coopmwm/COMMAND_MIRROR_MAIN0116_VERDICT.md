# MAIN01:16 — command-mirror qualification verdict

The completed source-bound gate is **FAIL**, so `mirror_active=false` remains
mandatory. This is the existing `COMMAND_MIRROR_GPU_GATE_1651.json`, adopted
without another WM call or SUMO run. Its current source hashes were verified in
`COMMAND_MIRROR_ADOPTED_1793637654.json`.

- CPU/CUDA command-callback maximum trace difference: **0.0**.
- TRAIN986051: absolute member-motion mismatch **0.8599243164 m** and
  **0.3810596466 m/s**.
- TRAIN986070: **0.0787963867 m** and **0.0550899506 m/s**.
- The original and post-safety prototype paths have the same absolute errors.
  Plan-vs-policy callback trajectories are identical on both null fixtures.
- These fixtures contain one-second observations. Half-step backgrounds were
  reconstructed from endpoint kinematics, with unresolved roster/lane changes.
  They cannot establish exact full-trajectory SUMO parity. A new identical probe
  would neither repair this evidence gap nor qualify activation.

The mirror is therefore **not qualified or activated**. No tolerance is widened,
no failed gate is re-labelled, and no unsupported statement that clipping caused
all observed bank nulls is made. Current bank null flags describe equality of
exact branch delay, not the causal mechanism. Collection/grading continue under
the existing authorized path. The next authorized performance change is
`it2_K2`: two screen iterations, K2/H300, full population, final K8/H_final.
