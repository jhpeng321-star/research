# S-RAMPFIX-PREP — mainline residual + lane_v5 long head (build only, nothing launched)

Branch `c4/rampfix` (from `c4/longloss@c106f3e4`). Flag-gated, default OFF, bit-identical when
off (proven by tests). CPU-only; nothing launched.

## Design
1. **`model.mainline_interaction_residual`** (oracle priority 1; `wms/model/mainline_residual.py`):
one learned acceleration offset for mainline HDVs (lanes 0/1, `s <= 75.44 m` = upstream + merge;
ramp/downstream slots masked to exact zero). 24 smooth asinh/scaled inputs: leader/follower gaps,
Δv, presence (same + adjacent lane), nearest ramp car ahead, density and mean speed ahead (150 m),
distance to merge-zone start/end, ramp queue state. Output `1.5·tanh(MLP)`, zero-init read-out
= exact no-op; added to the combined `accel_delta` BEFORE the existing
gate/soft-bound/cap/plant clamp (gated and bounded exactly like the existing residual); CAVs
excluded by the `learned` mask; inputs keep autograd, so the loop fine-tune and long-run paired
loss train it through weights and state. `enable_mainline_residual()` extends a loaded WM1;
legacy state dicts auto-fill. Group prefix `mainline_residual`.
2. **`optim.lane_v5_long`** (+ `lane_v5_long_lr_scale=0.05`): unfreezes the EXISTING lane_v5
network (no architecture change): removed from `freeze_prefixes`, `requires_grad=True`, own
optimizer group at 0.05× stage lr; flag-off optimizer report byte-equal to before.
3. **`joint_ramp`**: DEFERRED (oracle rank 3; shared acceptance/courtesy/yield/ramp-speed net,
identity at init, inside the physics envelope — not built).

## Tests (`tests/unit/test_rampfix.py`, 11 pass, CPU)
Flag-off key-set parity; ON-vs-OFF forward bit-identical at init (also via enable-hook and legacy
state-dict load); ramp/downstream mask zero with non-zero weights; gradients flow (weights +
state); `g(0)=0` kills the applied offset; lane_v5_long OFF freezes exactly as before; ON gives
own group `lr=0.05×`, merge_core frozen; flat-config roundtrip. Existing suites re-run: 100 pass;
4 failures are pre-existing sparse-checkout environment gaps.

## gpuR launch (ready; DO NOT launch without owner)
Reuses S-LONGLOSSTEST verbatim (prepare → train ≤3 h/≤400 updates, longwm 300 s graph, 600-tick
truncation, batch 2, coupled paired randomness → horizon_gate 48 groups → accept.py →
world-cluster bootstrap). Prerequisite: longloss OUT on gpuR. Run:

```
python tools/rampfix/operate.py state      # disk / longloss artifacts present
python tools/rampfix/operate.py stage      # stages c4/rampfix HEAD (src, config, tools, doc)
python tools/rampfix/operate.py sync
python tools/rampfix/operate.py preflight  # compile, variant configs, zero-init head, bootstrap identity
python tools/rampfix/operate.py launch     # detached chain.py; DONE.json; mirror to gpuA
python tools/rampfix/operate.py verify
```

Variants: `mres` (WM1 + head, group lr 1.0), `lane5` (lane_v5 @0.05), `both`. Verdict **helps**
iff 300 s paired-bias reduction > 0, world-bootstrap CI low > 0, 10/30/60 s sign agreement
≥ .868421 (unchanged); acceptance separate. WM1 SHA `5257a64c…` asserted at prepare. Guard
refuses sealed/holdout/…/vtype, ZERO SUMO; allocator 0.85; disk stop 3.25 GiB.
