# S-INTEGRATE candidate-A gpuA commands (DO NOT RUN — MAIN executes after the WM choice)

Preconditions (all four, in order):
1. Morning WM choice recorded (registered rule); WM checkpoint + config + CCT-2 file for that
   WM's sha256 exist (`authority/CCT2_<wm_sha256>.json`; WM2 `b8797242…` and WM3 `41e18f09…`
   are committed; any other WM needs S-CCT2-INC output first).
2. DONE (S-INTEGRATE2): the PASSING mode-2 execution parity receipt for the C1 actor under
   the always-commanded law is committed at
   `tools/execparity/alwayscmd3_execution_parity.json` (pooled executed RMSE 0.146538 <=
   native-head band 0.160312, re-confirmed bitwise with `C4_SUMO_LEADER_RULE=1` on this
   tree). The c4/execparity3 leader-rule slot is BOUND in `tools/candA/authority.py`
   (source hashes + config `[execution] leader_rule = true` + the receipt); the launch env
   below MUST keep `C4_SUMO_LEADER_RULE=1` — the receipt is only valid with the rule ON in
   the imagination mirror (deployment executor path unchanged).
3. This branch (c4/candA-int) staged on gpuA as an immutable source tree
   `/root/autodl-tmp/c4/candA_<commit12>` (SOURCE.json records the commit; `pip install
   '.[parallel-futures]'` for cloudpickle in the runtime env).
4. Auditfix-authority BOUNDARY.json + measured horizon-gate file built for the chosen WM
   (`python -m wms.control.auditfix_authority …`; inherited-90s template is NOT valid for a
   new WM), then the candA authority built and verified.

## Step A — build + verify the launch authority (CPU, on the staged runtime)

```sh
cd /root/autodl-tmp/c4/candA_<commit12>
export C4_EXECUTION_PARITY=2 C4_EXECUTION_DRIVER=/root/autodl-tmp/c4/execparity/fit_cav_idm_v1.json
L=/root/autodl-tmp/c4/restart20261026/arm/loop
/root/miniconda3/bin/python -B tools/candA/authority.py build \
  --wm-ckpt $L/wm/<CHOSEN_WM>/wm_s.pt --wm-config configs/c4_wm_loop14_r16_ft.toml \
  --cct2 authority/CCT2_<wm_sha256>.json --out $L/c4prep/CANDA_AUTHORITY.json \
  --actor $L/policy/c01.a3/policy_latest.pt \
  --parity-receipt tools/execparity/alwayscmd3_execution_parity.json --allow-dirty=false \
  --dense-phase1
/root/miniconda3/bin/python -B tools/candA/authority.py verify \
  --authority $L/c4prep/CANDA_AUTHORITY.json
```

`--dense-phase1` (S-LAUNCHA 18:25 fallback, registered 23:25 rule) declares the
phase-1 credit mode: dense 90 s only, the CCT-2 file staying bound as the
REGISTERED SECOND-PHASE authority (requires `long_credit.enabled = false` in the
config; `gain_only` builds require `= true` — fail-closed both ways).

`verify` refuses on ANY mismatch (source commit, runtime manifest incl. the four leader-rule
sources, config chain incl. `[execution] leader_rule = true`, C1 actor 64f70af4…, WM
sha/config, CCT-2 H_G, learned-mix binding, leader-rule slot, parity receipt). No verify ->
no launch.

## Step B — exact 2-update GPU qualification (gpuA, ~12 min at 350.8 s/update)

Registered dry recipe: FGA flags + N4 parallel futures, NINE-row exact path, update 181
fixture equivalence already proved bitwise (158/158 leaves, tools/polspeed2/evidence/REPORT.json).

```sh
cd /root/autodl-tmp/c4/candA_<commit12>
export PYTHONPATH=$PWD/src:$PWD C4_ALLOWED_ROOTS=/root/autodl-tmp/c4 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  C4_PROBES_DEVICE=cpu C4_GPU_MEM_FRACTION=0.45 \
  WMS_DATA_V5_NATIVE_Q=/root/autodl-tmp/c4/restart20261026_inputs/data/v5_native_q \
  WMS_DATA_GENRAND_PLAIN_Q=/root/autodl-tmp/c4/restart20261026_inputs/data/genrand_plain_q \
  C4_EXECUTION_PARITY=2 C4_EXECUTION_DRIVER=/root/autodl-tmp/c4/execparity/fit_cav_idm_v1.json \
  C4_SUMO_LEADER_RULE=1 \
  WMS_SPEED_FUTURE_WORKERS=4 WMS_SPEED_PARALLEL_FORWARD=1 WMS_SPEED_FUTURE_VJP_SLOTS=4 \
  WMS_SPEED_FUTURE_MEMORY_FRACTION=.10 WMS_SPEED_FORWARD_MEMORY_FRACTION=.04 \
  WMS_SPEED_REPLAY_FORWARD_MEMORY_FRACTION=.022 WMS_SPEED_GRAPH_CVAR=1 WMS_SPEED_ASYNC_SPILL=1 \
  WMS_WMSPEED_DEDUPE_MOBIL=1 WMS_WMSPEED_MOBIL_LAZY_SAFE=1 WMS_SPEED_PIVOT_DIAG_OFF=1 \
  WMS_SPEED_FAST_MERGE_LANE=1 WMS_SPEED_FAST_CAV_DRIVE=1 WMS_SPEED_FAST_LONG_HISTORY=1 \
  WMS_SPEED_GRAD_AUDIT_EVERY=25
L=/root/autodl-tmp/c4/restart20261026/arm/loop
nohup setsid /root/miniconda3/bin/python -B -m wms.control.policy14_cli \
  --config configs/c4_policy14_candA.toml \
  --wm-ckpt $L/wm/<CHOSEN_WM>/wm_s.pt --wm-config configs/c4_wm_loop14_r16_ft.toml \
  --real-returns $L/returns/returns_c03.a1.npz \
  --longcritic-native-returns $L/returns/native_returns_c03.a1.npz \
  --paircrit-receipt $L/returns/pairing_c03.a1.json \
  --paircrit-policy-sha256 0978aa7288cf9e36a0b61949b1e236b3fe48a0d220dada64f67a4797f969dc5e \
  --gate-panel /root/autodl-tmp/c4/restart20261026/arm/gate_panel.json \
  --horizon-gate <CHOSEN_WM measured gate.json> \
  --looplong-boundary <CHOSEN_WM BOUNDARY.json> \
  --real-credit $L/sumocredit/REAL_CREDIT_c03.a1.json --critic-warmup-updates 0 \
  --native-init --phase-relative-signal-gate \
  --resume $L/policy/c01.a3/policy_latest.pt --new-phase \
  --device cuda --seed 965140 --max-updates 150 --prep-dry-run-updates 2 \
  --output-dir /root/autodl-tmp/c4/candA_dry_<commit12> --compact \
  > /root/autodl-tmp/c4/candA_dry_<commit12>.log 2>&1 < /dev/null &
```

PASS = two UNIQUE finite 90 s/K8 updates, nonzero module gradients, effective 8.9/8.9,
monitors on, direct credit disabled+flagged, long-credit receipt (H_G of the chosen WM,
the declared dense-only phase-1 reason `dense_90s_phase1`, or another explicit dense-only
reason), memory within cap. Commit the receipt BEFORE Step C.

## Step C — production launch (gpuA, 150 updates, ~15.9 h projected)

Identical to Step B (keep `C4_SUMO_LEADER_RULE=1` in the env — the parity receipt is only
valid with the imagination leader rule ON) except: drop `--prep-dry-run-updates`, write to
the phase output dir:

```sh
  … --max-updates 150 --output-dir /root/autodl-tmp/c4/candA_phase_<commit12> --compact \
  > /root/autodl-tmp/c4/candA_phase_<commit12>.log 2>&1 < /dev/null &
```

Use the run's own STOP file (never signals) to stop; `checkpoint_every=1` writes
policy_<NNNNNN>.pt per update (update-50 checkpoint is the A-LONGX fork point). C4A_RUNNING
may only be claimed after >= 10 UNIQUE finite updates + gradient/credit/memory evidence and
the unchanged phase-relative signal gate PASS, exactly as S-C4LAUNCH registered.
