# LOOP14 — Dreamer-style policy/WM loop on 13E (2026-10-25)

Branch `c4/loop14` = `c4/policy14e` (95ee61bf) + merge of `c4/v10r14` (83ce0ec1). `c4/v10r15` did not exist at build time; merge it later as a separate commit.

Owner decision (10-25): no separate round 14 if 13E qualifies. The policy trains on 13E from Oct 26; R14 and the paired loss enter the first loop WM update.

## Cycle k (`scripts/c4/loop14.py`, one command per phase, fail-closed)

| phase | host | what |
|---|---|---|
| `policy` | gpuR | `policy14_cli`, N updates, WM frozen. Later cycles use `--resume` on the **accepted** WM, with the previous cycle's current-policy returns. |
| `collect` | cpuN | `collect --policy-pairs` (`traffic_shepherd/c4/policy_pairs.py`). Native base per fresh TRAIN world, then **POLICY** and native **B** arms from each `saveState`. The lane mask's learned settings come from the accepted WM's encoder on SUMO kinematic history. |
| `convert` | cpuN to gpuR | `wms.data.c4_episodes` + `wms.control.loop14_returns` (real −J from the env ledger). Only converted pairs and returns move to gpuR. |
| `wm_ft` | gpuR | `configs/c4_wm_loop14_ft.toml`, initialised from the accepted WM. |
| `check` | gpuR CPU | `tools/loop14_checks/accept.py`. ACCEPT replaces the accepted WM; REJECT keeps it. |

State lives in `LOOP14_STATE.json` (atomic, resumable). Ledgers are `cycle_ledger.jsonl` and `seed_ledger.jsonl` (allocations are written before SUMO starts).

Guards: pinned, acknowledged registration; budgets; GPU-busy (any compute process gives exit 5; nothing is signalled); disk floor; restricted tokens and evaluation seeds; a new `.aN` directory per `retry`.

## WM fine-tune

- **Frozen:** driver encoder and updater, gates, lane nets, merge core and pivot, temporal, and emergency/gs physics.
- **Trained:** residual (non-lane), courtesy (not re-zeroed), learned yield.
- **D1:** on. **S-TTC `physics.emergency_onset = "driver_b"`:** on. The envelope plans with the ego's and leader's learned comfortable b, brakes harder only when the plant plan demands it, and keeps the −9 floor. Default `plant` is bit-identical. On the 24 TTC.md states (13E-A, gpuR CPU, `LOOP14_TTC_ONSET.json`):
  - gate TTC ratio 2.01 → 0.27 [0.14, 0.44], below the 0.28 band;
  - strict 5.17 → 0.67;
  - 1 s hard-brake share 0.70% → 0.08% (SUMO 0.03%);
  - delay unchanged (1.00× SUMO);
  - merges 10.4 → 9.9 (SUMO 11.3).
- **Data:** the R14 D3 mix (low-share packs at ×1.69), each share ×0.85, plus the collected pairs at 0.15.
- **Schedule:** 120 s, 40 steps, lr 3e-5.
- **R15 hook:** `pairs300` stays the single paired source for now. When R15 lands, set `loop14_pairs.paired=true` and `pairs300.paired=false` through `wm.extra_set` before init.

Fix: `WmSModel._apply_cav_overrides` raised a KeyError on timeline-command windows with gs on. It now exports gs inputs only when learned settings exist.

## Acceptance rule

Both WMs run under the candidate config on a fixed TRAIN panel. REJECT if any of:
- lane-1 HDV a<−3 share (60 s rollouts) is more than 2× the reference's;
- all-lane-1 one-step MAE is more than the reference's + 0.1 m/s² (the 13E rule, `round13c_waiter.stop_reasons`);
- there are more blowups or overlap pair-steps than the reference;
- TTC<1.5 s share is more than 2× the reference's;
- there is a measurement error.

Paired-difference error and BASE/MIDRUN are recorded but do not gate.

## Stop rules

Stop on:
- 2 consecutive rejections;
- `max_cycles`;
- SUMO budget exhausted;
- any branch collision or teleport;
- a failed phase (the operator decides on `retry`);
- the policy14 update-10 gate.

Review if SUMO's mean policy−native J is above 0 for 2 cycles. The deployment rule (calibrated ≥ 0.8) is unchanged.

## Budgets per cycle

| phase | budget |
|---|---|
| policy | 150 updates, ≤ 3 GPU-h |
| collect | 16 worlds × (661 + 2×2×300) = 29,776 sim-s |
| convert | about 0.4 GB on gpuR |
| wm_ft | 40 steps, ≤ 1.5 GPU-h |
| check | 24 windows, CPU, ≤ 2 h |

Over 4 cycles: ≤ 120,000 sim-s and 64 seeds.

## Registration template

```json
{"id": "C4-LOOP14-20261026", "status": "registered", "split": "TRAIN", "acknowledged_by": "<main, time>",
 "seed_block": [968000, 968999], "exclude": [],
 "sources": {"initial_wm_sha256": "<13E B FINAL>", "policy0_sha256": "<adopted>", "code_sha": "<commit>"},
 "comparators": "previous accepted WM (same flags/panel); SUMO native arm per pair",
 "budget": {"max_cycles": 4, "policy_updates_per_cycle": 150, "sim_s_per_cycle": 30000, "sim_s_total": 120000,
            "gpu_h_policy_per_cycle": 3, "gpu_h_wm_per_cycle": 1.5, "wm_steps_per_cycle": 40},
 "stop": {"consecutive_wm_rejects": 2, "max_branch_incidents_per_cycle": 0}}
```

968000–968999 lies in the registered loop TRAIN block (966000–969999), clear of v10loop's 966xxx and the used 969xxx. Grep the remote seed ledgers before acknowledging; add hits to `exclude`.

## Operator commands (Oct 26)

```bash
V5=/root/autodl-tmp/c4/v5; P=/root/miniconda3/bin/python; C=configs/c4_loop14_gpur.json
# stage code at the loop14 commit on gpuR + cpuN; fill REPO_SHA / cpuN target in $C
bash scripts/policy14_launch.sh $V5/runs/c4_wm_v10_round13e_b/wm_s.pt \
  configs/c4_wm_v10round13e_stage_b.toml $V5/policy14_stage/runs/policy14_native_returns.npz \
  --qualification <13E qualification JSON>      # seed 965140 = loop policy (no best-of-3)
$P scripts/c4/loop14.py init --config $C --adopt-policy $V5/runs/c4_policy14_20261026/seed_965140/policy_latest.pt
$P scripts/c4/loop14.py plan --config $C        # dry run
$P scripts/c4/loop14.py next --config $C        # collect, convert, wm_ft, check, policy, ...
$P scripts/c4/loop14.py status --config $C      # exit 5 = GPU busy (rerun); 4 = WM rejected
```


## Evidence (laptop, mechanics only)

- 22 new tests (`test_loop14`, `test_loop14_emergency_onset`); 633 related tests pass.
- `tools/loop14_checks/cpu_cycle_smoke.py` ran one full cycle (S-R12C stand-in, seeds 4107–4108, SUMO 1.27.1; `LOOP14_SMOKE.json`).

## Open items

- The low-share packs (2.4 GB) are not on gpuR, which has about 3.6 GB free.
- SUMO lane-command parity has not been probed.
- The settings encoder uses kinematic history, not the RSU belief.
- The returns file is single-source.
- `driver_b` undershoots the gate TTC band. The C9 worlds and fidelity C2 were not run.
