# WM-N1 Idea A — P10 multibox + P11 pre-flight (owner runbook)

All commands are run **from `output/wd01/wm_n1`** (the only directory where
`bundle`, `preflight`, `ctrl`, `hdv`, `rollout2`, `train`, `corpus`, `evalh`
and `model` import as top-level packages).  CPU only, zero SUMO.

```
cd output/wd01/wm_n1
```

---

## P10 — one config per GPU box, boxes in parallel

`bundle.multibox` splits the work into **8 configs**, one per box:

| box | config | what it is |
| --- | --- | --- |
| `box-member-1` | `member-1` | ensemble member 1, trained FIRST |
| `box-member-2` … `box-member-5` | `member-2` … `member-5` | members 2-5 (wait for member 1) |
| `box-heldout-i-j4b` | `heldout-i-j4b` | (i) train J4 (law v2.1) → eval J4B (law v2.2) |
| `box-heldout-ii-leave-pi` | `heldout-ii-leave-pi` | (ii) train without the PI arms → eval CAL PI |
| `box-heldout-iii-leave-highest-wall-speed` | `heldout-iii-leave-highest-wall-speed` | (iii) train without the highest J4 wall-speed factor |

Each held-out config is a 1-member model.  A box **refuses any config not
assigned to it**; every member-k (k>1) refuses to start until `member-1` has
completed.  State is per box (`boxes/<box>/state.json`) plus a fleet ledger
(`fleet_state.json`), so a re-run resumes.

### Exact per-box commands

Print the plan:

```
python -m bundle.multibox plan
```

Run one box (member 1 first).  The default executor entry point is
`bundle.e2e.run_config_job(box_id, config, work_dir)`; override with
`--executor module.func`:

```
python -m bundle.multibox run-box --box box-member-1 --work-dir $WORK
python -m bundle.multibox run-box --box box-member-2 --work-dir $WORK
python -m bundle.multibox run-box --box box-heldout-i-j4b --work-dir $WORK
python -m bundle.multibox run-box --box box-heldout-ii-leave-pi --work-dir $WORK
python -m bundle.multibox run-box --box box-heldout-iii-leave-highest-wall-speed --work-dir $WORK
```

Run one config with explicit approval/gate files:

```
python -m bundle.multibox run-config --box box-member-1 --config member-1 \
    --work-dir $WORK \
    --approval $WORK/approvals/member-1/APPROVAL.json \
    --timing-gate $WORK/approvals/member-1/TIMING_GATE.json
```

Run the whole fleet (member-1's box first, then the rest in parallel):

```
python -m bundle.multibox run-fleet --work-dir $WORK --parallel --max-workers 8
```

### Per-config owner `APPROVAL.json`

Each box writes its own `TIMING_GATE.json`; the owner then approves that config
with `APPROVAL.json` (schema `wm_n1_multibox_approval/1`), binding the gate's
raw-bytes sha256 **and all four frozen records** (LF-normalised, CRLF-safe):

```json
{
  "schema": "wm_n1_multibox_approval/1",
  "approved": true,
  "owner": "<owner>",
  "config": "member-1",
  "timing_gate_sha256": "<sha256 of this box's TIMING_GATE.json>",
  "registration_lf_sha256": "666233aa1d1e466750d6d944b9102c84a2f32850ebeb4a97c9f3c9b56dd93246",
  "amendment_1_lf_sha256": "decb96562ebfb549b56406d980be9c78bf4a377bd50c5eadeea08aac16e16c57",
  "amendment_2_lf_sha256": "56fa99a04fdd38e37a518dfc65972bc5e2e6d227210372a75258bf28249c83ed",
  "amendment_3_lf_sha256": "a597d9861dcc859cbcb9c10b65069a81031732516ec12a0296fd68aff604eef6"
}
```

A wrong/truncated declared sha, or a tampered record file, is refused.  A CRLF
checkout of the records hashes identically (LF-normalised).

---

## P11 — owner pre-flight gates G1–G6

```
python -m preflight <zero-residual|ctrl-parity|label-leak|overfit|e2e-smoke|heldout-dryrun|all>
```

Each gate prints `PASS`/`FAIL` and writes `preflight/reports/<gate>.json`;
`all` also writes `preflight/reports/PREFLIGHT_SUMMARY.json`.  The process exits
0 only when every requested gate passes.  No SUMO, CPU only, TRAIN worlds only.

| gate | entry point(s) it calls |
| --- | --- |
| G1 `zero-residual` | `rollout2.equivalence.zero_residual_report` (P3) |
| G2 `ctrl-parity` | `ctrl.parity.strategy_parity_report` (P1) |
| G3 `label-leak` | `hdv.input_schema.main_config_schema`, `hdv.input_schema.perturbation_check` (P2) |
| G4 `overfit` | `train.idea_a.overfit_report` (P7) |
| G5 `e2e-smoke` | `bundle.e2e.run_e2e_smoke` + torch>=2.7,<2.13 scan/version gate |
| G6 `heldout-dryrun` | `train.idea_a.heldout_split_report` (P7) |

List every gate and the entry points it needs without running anything:

```
python -m preflight all --dry
```

Run them one at a time, or all at once:

```
python -m preflight zero-residual
python -m preflight ctrl-parity
python -m preflight label-leak
python -m preflight overfit
python -m preflight e2e-smoke
python -m preflight heldout-dryrun
python -m preflight all
```

A gate whose piece is not built yet reports `FAIL <gate>: piece missing: ...`
rather than crashing, so the pre-flight is runnable during parallel
development.

### Torch >=2.7,<2.13 (cu128)

G5 runs a static API scan (forbids `torch.compile`, deprecated `torch.cuda.amp`
(use `torch.amp`), removed `torch._six`, `scaled_dot_product_attention`,
`flex_attention`, `torch.export`) and a `torch.__version__`-gated op test that
RUNS inside the supported `>=2.7,<2.13` window (2.12.1+cpu locally).  To run it
on the real GPU-box environment:

```
python -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7,<2.13"
python -m preflight e2e-smoke
python -m bundle.envcheck          # CUDA/sm_120 + real-member forward/backward
```

---

## Tests

```
WD_FORBID_D=1 python -m pytest bundle/tests preflight/tests -q
```
