# R3_STAGE — gpuA staging preparation for WM-S-R3-20261010

Prepared 2026-10-09/10 in this worktree (`wms/r3-stage`), local only: gpuA was
OFF, no ssh/scp was attempted, nothing was launched or killed. Round-3 recipe
source: `E:/research-dense/docs/WM_CHANGE_LOG.md` ("Round 3"); prior staging
reference: `E:/research-dense/runs/active/WM-S-MAIN2-20261009/REGISTRATION.md`
and its gpuA `main2_env.sh`. Registration draft:
`E:/research-dense/runs/active/WM-S-R3-20261010/REGISTRATION.md`.

## What is here

| file | role |
|---|---|
| `r3_box/r3_env.sh` | box env template (LF). CODE_ROOT=/root/autodl-tmp/wms_r3, real pack/sidecar locations from MAIN2, new lpwo_v3 + sidecars_packv3_v2 paths, W=/root/autodl-tmp/wms_train_r3/r3_main_a1, MAX_HOURS=14, OMP=2, SMOKE_CONTENTION_GUARD=0, RSS_GUARD_BYTES=60000000000, braking losses without cap, R3 flags in `FLAGS` |
| `r3_box/stage_gpuA.sh` | local: builds `wms_r3_code_<HEAD>.tar` via `git archive HEAD` + the ~21.6 MB data tar (lpwo_v3 lists + sidecars_packv3_v2 payload), writes sha256, uploads once with scp (alias `gpuA`, `unset MSYS_NO_PATHCONV`), verifies hashes on the box, refuses an existing `/root/autodl-tmp/wms_r3`, unpacks, prints the exact single `nohup setsid ... &` launch command. Never launches, never loops on ssh. `STAGE_DRY_RUN=1` = local build + verification only |
| `r3_box/fetch_and_shutdown.sh` | local: refuses while a `wms.cli train` process runs, stream-tars the finished run dir to `E:/research-dense/runs/active/WM-S-R3-20261010/box/gpuA/` (uncompressed so the re-hash is deterministic), compares local vs fresh box-side stream sha256, then only PRINTS the shutdown command |

## Box layout after staging

```
/root/autodl-tmp/wms_r3/                  # CODE_ROOT (fresh; script refuses if it exists)
  _out/train_main_chain.sh                # integrated r3 chain (from git archive HEAD)
  _out/merge_params.json, _out/r3_box/r3_env.sh
  lpwo_v3/                                # v3 lists (receipt-hashed)
  sidecars_packv3_v2/{identities,outcomes}/
  output/wd01/wm_n1/...                   # wms code
/root/autodl-tmp/wms_train_r3/r3_main_a1  # W (created by the run itself)
```

Real packs (`wmn1_m1_45c9032_20261003_A1/work/pack`, `wmn1_a9_data/pack_dspeed`,
`wmn1_a10_data/pack_damping`, `wms_run2_20261006/work/sidecars`,
`wmn1_switching_train`) stay untouched from MAIN2; the packv3fix loader
resolves the canonical `pack_v3` list name to the staged base-pack basename.

## Prerequisites before running stage_gpuA.sh

1. The integrated r3 branch must be committed in this worktree (the script
   refuses on tracked-dirty trees and archives exactly HEAD). The r3 chain
   must implement the C1 recipe defaults (no hard cap, start-shift 0.75, no
   traffic/stability/conservation) + the R3 flags + the MAIN2 box patches as
   env gates (`SMOKE_CONTENTION_GUARD`, `RSS_GUARD_BYTES`); its help gate
   refuses unknown `FLAGS` tokens before training.
2. The verified pack_v3 sidecar payload must be placed under
   `_out/sidecars_packv3_v2/` (`identities/` + `outcomes/`, ~21.1 MB; built in
   the packv3fix worktree, sha256 pinned by the tracked
   `_out/sidecars_packv3_v2/BUILD_RECEIPT.json`). The staging script verifies
   all 8 payload files and all 5 list files against those pinned hashes and
   refuses on any mismatch or absence — fail-closed, so no wrong bytes can
   reach the box.

## Launch (only when the owner turns gpuA on)

```bash
STAGE_DRY_RUN=1 bash _out/r3_box/stage_gpuA.sh   # local check
bash _out/r3_box/stage_gpuA.sh                   # upload once; then follow the printed steps
```

The printed launch sequence is: interactive `ssh gpuA`, source
`_out/r3_box/r3_env.sh`, `bash _out/train_main_chain.sh preflight`, and — only
if preflight passes — exactly one `nohup setsid bash _out/train_main_chain.sh
run > /root/autodl-tmp/wms_train_r3/launch_r3_main_a1.log 2>&1 &`. If ssh
drops, never relaunch: reconnect and read `W/chain.log`. After the run:
`bash _out/r3_box/fetch_and_shutdown.sh`, then power the box down manually with
the printed command.

## PROVISIONAL r3_env.sh values (confirm at integration)

| flag | value | basis |
|---|---|---|
| `--residual-soft-bound-mps2` | 2.0 | C1 raw residual mean 1.8 m/s²; soft bound just above working range (bisect evidence) |
| `--residual-mag-weight` | 0.01 | same order as MAIN2 aux losses (0.01) |
| `--lane-rate-weight` | 0.01 | idem |
| `--lane-rate-target` | 0.0123 | recorded 1.23 %/HDV-step (change #6 evidence) |
| `--history-s` / `--history-stride-s` | 60 / 15 | R3-5 30–60 s window; stride is a guess pending the integrated CLI |
| `MAX_HOURS` | 14 | ~9–10 h budget + margin; smoke-gate projection must be below |

## Verification performed (local only, no simulation, no ssh)

- `bash -n` on all three scripts (Git Bash): pass.
- No CR bytes in any staged file (LF check): pass.
- `STAGE_DRY_RUN=1 bash stage_gpuA.sh`: lpwo_v3 receipt-hash block executes
  for real (all 5 lists match); script then refuses at the absent payload —
  the intended fail-closed behavior until the payload is copied in.
- Tamper test: synthetic payload with wrong bytes → refused by the pinned
  payload hash check; tar/sha256/unpack command patterns exercised in a temp
  sandbox roundtrip.
- lpwo_v3 list hashes independently re-verified against `BUILD_RECEIPT.json`
  with PowerShell SHA256: match.

## Constraints honored

No ssh/scp to any box; no process killed; writes confined to this worktree and
`E:/research-dense/runs/active/WM-S-R3-20261010/`; no CAL/HOLDOUT/sealed/
chunk-5/J4B/INC1 data touched (lists are metadata; receipts record sealed
tokens as never-read); no credentials handled (ssh alias `gpuA` is the user's
own config).
