# POLICY14 preparation — 2026-10-25

## Status and changes versus v8/v9entry

Prepared on `c4/policy14`, forked from `c4/v10r13c` at `eb75010c`.
This is a bounded learning/integration preparation, not a qualified policy result.
The original worktree and running round-13C job were untouched.

1. **Authoritative objective:** reuse v9entry demand, initial backlog, coupled
   arrivals and entry-slot machinery. Primary reward is exactly minus total
   on-road speed-deficit delay plus source waiting, in vehicle-seconds. No
   admitted/completed-count divisor, throughput bonus or fairness double-counting.
   Fixed requested-population denominators are used only for reporting. Accrued
   terminal waiting remains charged; a critic continuation is labelled estimated.
2. **Real-return critic:** strict TRAIN-only numeric NPZ provider, 32 critic-only
   warm-start updates, then configurable real/imagined regression. Critic output
   and residual scales are 10,000 vehicle-seconds; values remain raw -J. Backlog-
   only road-empty states now have learnable value. Twelve existing native
   600 s summary-ledger targets were exported without opening scenario/type/truth
   files. They are diagnostic, off-policy p40/2400 warm-starts.
3. **Optimisation audit:** reward/constraint/entropy/lane/critic gradient norms,
   saturation, physical-action entropy, and independent critic-free J rescoring.
   Acceleration uses team pathwise gradients; sampled leave-one-out credit adds
   averaged per-CAV categorical lane score-function advantages. Detached reference
   subtraction is not claimed as extra continuous-action credit. Synergy is off.
4. **Gradient paths:** undamped imagination derivatives; paired finite differences
   through safety projection and the WM. Repair-A also needed to retain gradients
   through previous-acceleration history when `follower_courtesy_state_grad=true`;
   this new fix leaves forward values unchanged.
5. **Entropy:** broad scratch actor support in [-3,1.5] m/s²; entropy anneals only
   on independently rescored unshaped improvement, with a registered floor.
6. **Bounded multi-seed handoff:** three sequential scratch seeds, early update-10
   optimisation gate, then independent imagination/native comparisons before any
   separately authorized SUMO phase. No GPU training or SUMO was launched.

The existing shared CAV-attention/RSU encoder and privileged critic are reused.
No controller selector, demonstrations, actor initialization or named wall option.
The actor remains the existing nonrecurrent architecture; persistent/recurrent
group-belief enhancements are future work, not claimed implemented here.

## WM interchangeability and lane safety

`policy14_cli` requires paired `--wm-ckpt` / `--wm-config` lists; one member is
allowed. TOML inheritance and resolved/flat JSON are supported, including explicit
false flags that must override old checkpoint flags. Model/physics semantics and
hashes are recorded. Start-data configuration is separate from WM configuration.

The default-off 13E emergency-braking and secure-gap runtime was ported from
`39b2e866`, preserving R13C temporal variants and checkpoint tensors. A pluggable
lane veto augments the existing mask. The additional 13E inferred T,b,min-gap tier
is now explicitly opt-in for policy CAVs even with an all-13E ensemble; CAVCUTIN
uses the learned follower response instead. Forced ramp merges are masked too;
reference commands share caps and projection. Actual SUMO command parity still
needs its future probe.

## CPU smoke evidence

Reproduce from this worktree with `.venv/Scripts/python.exe`, `PYTHONPATH=src`:

```text
python -m wms.control.policy14_cli --config configs/c4_policy14.toml --wm-ckpt runs/wm_s.pt --wm-config runs/train_config.json --real-returns runs/policy14_native_returns.npz --device cpu --seed 965140 --max-updates 2 --output-dir runs/policy14_repeat --smoke --compact
```

S-R12C SHA-256: `c76c338dea6355f85cfaee63cdb9077fb129ed25db0ab582d77eda2961822091`.
CPU, two threads, two 2-second updates, entries/backlog enabled. Actor losses
5.32247/6.18132; critic losses 4.30477/3.45094; all gradient norms finite.
Critic warm-start scaled MSE decreased 38.6285→14.3479.
Independently scored J was 61.423374→61.425308 veh-s, including 47.5 source veh-s:
**slightly worse, not a learning win**. The expensive budget is guarded at update 10.

Exact-checkpoint dJ/da = -0.05592841 versus FD -0.05593300. S-R12C deliberately
disables an exactly-zero courtesy output head. A separately labelled disposable
copy with seeded nonzero output weights and repaired pivot intent gives dJ/da
-0.05558668 versus FD -0.05558729 (h=.04). Isolated courtesy contribution:
0.00034173 versus FD 0.00034571; h=.02 also passes. Training WM remains unchanged.
The separate scratch free-road actor learned -0.4635→+1.3553 m/s² and reduced J.
300 tests pass; the SUMO integration test was deselected. Bash syntax/dry-run pass.
Compact retained evidence: `POLICY14_SMOKE.json`; full logs/checkpoints: ignored `runs/`.

## Launch and remaining risks

### Guard follow-up — 2026-10-25 (partial; not Oct 26 launch-ready)

- **CAVCUTIN ON:** `lanes.consequence_mask=true` adds a veto to the existing
  geometry/gap/dwell mask. Compute the nearest new follower's **unclipped** IDM
  acceleration after insertion using its WM driver-encoder settings and require
  `a_F >= -b_F`; reject leader/follower body overlap. Missing/invalid inferred
  settings fail closed. WM callbacks export those settings independently of the
  WM's 13E flag, without changing checkpoint tensors or emergency-braking physics.
  Recheck simultaneous proposals, propagate cancelled leavers, and apply the
  same guard to forced ramp merges. The optional 13E hook defaults off for CAVs.
- **Available ROBUST pieces ON:** the supplied TRAIN acceleration endpoints are
  implemented through the existing actor/executor bounds `[-3.0,+1.2] m/s²`,
  with the native counterfactual ceiling also `+1.2`. These endpoints come from
  the engineering brief, **not a new quantile derivation from local data**.
  `trainer.low_penetration_horizon_cap=true` caps imagination at 60 s when
  `10 * observed_present_CAV_count <= observed_present_vehicle_count` at t0.
  Empty roads count as zero share; padding is ignored; a mixed batch shares its
  shortest horizon to preserve paired counterfactuals. Config/world labels and
  imagined arrivals do not determine penetration. Entries use the same horizon.
  `trainer.wm_safety_report_only=true` logs WM TTC costs/excess but removes their
  actor gradient and dual update. SUMO safety qualification remains outstanding.
  Launch preflight requires these switches and matching acceleration ceilings.
- **Blocked ROBUST remainder:** no local ROBUST.md, exact 10 s action-feature
  implementation, TRAIN reference windows/normalisation/kNN definition, or
  empirical lane-rate/hold-length maxima are available. Do not claim an exact
  `relu(D/1.76-1)` penalty, a `D>2.55` untrusted treatment, or data-derived lane
  limits: none is implemented. The brief's named outside-workspace authorization
  conflicts with its later explicit no-outside-read execution rule. This session
  stayed inside the worktree; external registry append is also pending. Resolve
  that scope conflict or stage the exact evidence here before completing ROBUST.
- **Verified partial build:** 287 CPU tests pass (all four `test_policy14_*`
  modules plus `test_c4_dreamer.py` and `test_dreamer_loop.py`); Ruff, Bash syntax
  and launcher smoke dry-run pass. Re-run the CPU CLI command above with output
  `runs/policy14_guard_final_smoke`: two updates pass with CAVCUTIN, the supplied
  acceleration bounds, the horizon-cap flag and report-only WM safety enabled.
  Actor losses 5.29027/6.18075; critic losses 4.30478/3.45093; gradients finite.
  Independent accrued J 61.483139→61.483597 veh-s: no efficacy claim.
  Compact evidence: `POLICY14_GUARD_SMOKE.json`. No GPU training or SUMO ran.

`scripts/policy14_launch.sh WM_CKPT WM_CONFIG REAL_RETURNS --qualification FILE`
targets gpuR's new `policy14_stage` directory, earliest Oct 26. Inspect with
`--dry-run`. Batch 1, H=10 s, checkpointing, one counterfactual start, two torch
threads; 600 updates/seed ceiling. Registration draft contains budgets/comparators.

Open risks: WM action-response qualification; native critic bootstrap bias and
finite-horizon/time-conditioning; weak single-stratum real corpus; discontinuous
lane/insertion credit; follower braking/fairness; 24 GB production memory unmeasured.
One WM provides no epistemic-disagreement evidence. No calibration is supplied:
deployment must abstain to safety-projected hold-speed until independently
calibrated probability of outperforming is at least 0.8. Native/MAPPO comparisons
must match sensors, commands and total simulation budgets.
