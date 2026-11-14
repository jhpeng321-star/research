# WM-N1 GPU-box bundle (`bundle/`)

Owner of the dedicated WM-N1 GPU-box queue
(`REGISTRATION_WM_N1_AMENDMENT_1.md` A3) and of INTERFACES.md **Mismatch 2**
(the timing gate must measure the REAL `model.net.WmN1Net`, not the evalh
`Toy`).

The queue is strictly sequential and **stops after the timing gate**. No
member trains until the owner `APPROVAL.json` binds the written
`TIMING_GATE.json` sha256 and both frozen registration LF-shas.

## Recon-input path (C4L2) is FUTURE WORK

**WM-N1 trains and evaluates on the TRUE state** (`WM_N1_DESIGN.md` section 2:
"Training uses the TRUE state"). The recon-input path, i.e. evaluating on
reconstructed / observed state inputs (a C4L2-like look), is **FUTURE WORK**:
it is a later, separately registered step and is NOT part of this bundle or of
the current GPU queue. Do not feed recon samples into the timing gate,
training or evaluation here.

The multibox GPU queue (`bundle/multibox.py`, `bundle/GPU_PLAN.md`) additionally
carries the variant **N** comparator (the original v1 label-conditioned pure
neural model, trained with `python -m train --member k`) alongside the Idea A
members and the held-out-strategy tests; it also trains/evaluates on the TRUE
state. Each config keeps its own owner approval and each family obeys
member-1-first. Regenerate the 3-box / 4-box planner tables with:

```
python -m bundle.multibox plan-fleet --boxes 3
python -m bundle.multibox plan-fleet --boxes 4 --json
```

```
verify-upload -> convert+verify -> timing-gate (HARD STOP) -> approval
             -> member-1 (alone) -> members-2-5
```

## 0. Environment

GPU box: Python 3.10/3.11 + `torch>=2.7,<2.13` from the cu128 wheel index
(Blackwell sm_120; see [`requirements-wmn1.txt`](requirements-wmn1.txt)):

```
python -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7,<2.13"
python -m pip install -r output/wd01/wm_n1/bundle/requirements-wmn1.txt
python -m bundle.envcheck          # box-side self-check (torch/CUDA/sm_120/F+B)
```

Run every command **from `output/wd01/wm_n1`** so `corpus`, `train`, `evalh`,
`model` and `bundle` import (they are top-level packages only there). Use
`$WORK` (or `%WORK%`) for the box-local run directory — no path is hardcoded:

```
# Linux
cd output/wd01/wm_n1          # from the repository root
export WORK=/data/wm_n1/run1
# Windows PowerShell (from the repository root)
# Set-Location output/wd01/wm_n1
# $WORK = "E:\wm_n1\run1"   # (or D:; the code never hardcodes a root)
```

## 1. Upload manifest + upload tree

`python -m corpus manifest` produces the upload manifest and REFUSES any
HOLDOUT / excluded / chunk-5 run (sealed). Put the unpacked upload tree at
`$WORK/upload` (the queue verifies and converts it in place).

```
python -m corpus manifest --pack PACK_DIR --out $WORK/upload_manifest.json
```

## 2. Queue — step by step (exact commands)

The stage names are exactly `STAGE_ORDER = ("verify-upload",
"convert+verify", "timing-gate", "approval", "member-1", "members-2-5")`.
`--stop-after` selects the last stage to run; `--stop-after timing-gate` is the
default hard STOP.

**Step 1 — verify upload (manifest shas; refuses chunk-5/HOLDOUT):**

```
python -m bundle --manifest $WORK/upload_manifest.json \
    --work-dir $WORK --approval $WORK/APPROVAL.json \
    --stop-after verify-upload --device cuda
```

**Step 2 — convert + verify (calls `corpus.pack.convert` then
`corpus.verify.verify_pack`):**

```
python -m bundle --manifest $WORK/upload_manifest.json \
    --work-dir $WORK --approval $WORK/APPROVAL.json \
    --stop-after convert+verify --device cuda
```

**Step 3 — timing gate (REAL member; HARD STOP):** writes
`$WORK/TIMING_GATE.json` and `$WORK/logs/timing-gate.log`. Read the verdict
before approving. For a CPU smoke of the same wrapper add `--device cpu
--tiny`.

```
python -m bundle --manifest $WORK/upload_manifest.json \
    --work-dir $WORK --approval $WORK/APPROVAL.json \
    --stop-after timing-gate --device cuda
```

**Step 4 — owner `APPROVAL.json`.** Compute the two now-known values:

```
python - <<'PY'
import hashlib, json, os, pathlib
w = pathlib.Path(os.environ["WORK"])
gate = hashlib.sha256((w/"TIMING_GATE.json").read_bytes()).hexdigest()
def lf(p):
    b = pathlib.Path(p).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(b).hexdigest()
reg = lf("REGISTRATION_WM_N1.md")
amd = lf("REGISTRATION_WM_N1_AMENDMENT_1.md")
print(json.dumps({"timing_gate_sha256": gate,
                  "registration_lf_sha256": reg,
                  "amendment_lf_sha256": amd}, indent=1))
PY
```

Write `$WORK/APPROVAL.json` (exact format):

```json
{
  "schema": "wm_n1_approval/1",
  "approved": true,
  "owner": "<owner name>",
  "approved_utc": "2026-09-30T00:00:00Z",
  "stage": "member-1",
  "timing_gate_sha256": "<64 hex from above>",
  "registration_lf_sha256": "666233aa1d1e466750d6d944b9102c84a2f32850ebeb4a97c9f3c9b56dd93246",
  "amendment_lf_sha256": "decb96562ebfb549b56406d980be9c78bf4a377bd50c5eadeea08aac16e16c57"
}
```

The frozen `REGISTRATION_WM_N1_AMENDMENT_1.md` (commit `79e686e`) must be
present next to this bundle; the gate re-hashes it (LF-normalised, CRLF-safe).

**Step 5 — member 1 alone (A1):** runs the approval stage then member 1;
writes `$WORK/members/member_1.pt`, `SEEDS.json`, `TRAIN_RECEIPT.json`.

```
python -m bundle --manifest $WORK/upload_manifest.json \
    --work-dir $WORK --approval $WORK/APPROVAL.json \
    --stop-after member-1 --device cuda
```

**Step 6 — members 2-5** (refuses unless member 1 completed):

```
python -m bundle --manifest $WORK/upload_manifest.json \
    --work-dir $WORK --approval $WORK/APPROVAL.json \
    --stop-after members-2-5 --device cuda
```

Every step logs to `$WORK/logs/<stage>.log`, records a receipt in the
idempotent/resumable `$WORK/state.json`, and skips stages already complete.
Re-run any command to resume; a failure raises and stops.

## 3. Underlying CLIs (equivalent, called by the queue)

```
python -m corpus pack    --data-root $WORK/upload --out $WORK/pack --workers 4
python -m corpus verify  --pack $WORK/pack
python -m train --member 1 --seed 101 --corpus $WORK/pack --out $WORK/members --device cuda
```

## 4. Timing gate JSON

`TIMING_GATE.json` uses schema `wm_n1_timing_gate/1` and adds
`measured_model: "model.net.WmN1Net"`, `proxy: false`, `parameter_count` and
`architecture`. Thresholds and the extrapolation formula are reused verbatim
from `evalh.timing_gate` (`BUDGET_HOURS_MAX=50`, `GATE_LIMIT_HOURS=1`,
`ROLLOUT_FRACTION=0.30`, `EVAL_HOURS=2`).

## 5. Tests (CPU, no SUMO, no real training)

```
WD_FORBID_D=1 python -m pytest output/wd01/wm_n1/bundle/tests -q
```
