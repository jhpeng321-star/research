# WM-N1 torch >=2.7,<2.13 (cu128) compatibility

**Branch:** `zcode/torch27` (at `zcode/wd01`).
**Goal:** make the WM-N1 GPU bundle runnable on the rented Blackwell-class
sm_120 vGPU-48GB boxes (driver CUDA <= 13.2).  The bundle previously pinned
`torch==2.5.1+cu121`; the code was developed/tested on `torch 2.12.1+cpu` and
statically scanned for 2.5 APIs.  This document records the static audit, the
pin change, the box-side self-check and the local verification.  **No package
was downloaded or installed here** (static + local 2.12 CPU checks only; the box
runs `python -m bundle.envcheck`).

## 1. Target environment + exact box install command

- Python **3.10 / 3.11** (`>=3.10,<3.12`).
- torch **`>=2.7,<2.13`** from the CUDA 12.8 wheel index (Blackwell sm_120).
- numpy `>=1.24`, pandas `>=2.0`, pyarrow `>=14.0` (PyPI).

Exact install on the GPU box:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7,<2.13"
# then the rest of the bundle deps (the file also carries the cu128 extra-index)
python -m pip install -r output/wd01/wm_n1/bundle/requirements-wmn1.txt
```

`bundle/requirements-wmn1.txt` now contains
`--extra-index-url https://download.pytorch.org/whl/cu128` and
`torch>=2.7,<2.13`, so the `-r` install resolves the same cu128 torch wheel.

## 2. Static API audit (torch 2.7-2.12), findings with file:line

Method: full-tree grep over `output/wd01/wm_n1/**/*.py` for `torch.load`,
`torch.cuda.amp`, `torch.amp`/`autocast`, `GradScaler`, TF32 flags
(`torch.backends.cuda.matmul.allow_tf32` / `cudnn.allow_tf32`),
`torch._six`/`torch._C`, `torch.compile`, `scaled_dot_product_attention`,
`flex_attention`, `torch.export`, `half()`/`bfloat16`, and torch device calls.
The G5 scan (`preflight/torch_compat.py`) also enforces this list in CI.

| # | Finding | file:line | Verdict |
| --- | --- | --- | --- |
| 1 | `torch.load` **always** passes explicit `weights_only=True` (no silent pickle; neutral to the 2.6+ default flip) | `adapter/checkpoint.py:131,147,158,161`; `adapter/predictor.py:425`; `train/idea_a.py:1151`; `train/trainer.py:193` | PASS |
| 2 | Deprecated `torch.cuda.amp` (-> `torch.amp`) | none in active code | PASS |
| 3 | Removed `torch._six` / private `torch._C` | none in active code | PASS |
| 4 | `torch.amp` / `autocast` / `GradScaler` (AMP) | none used | PASS |
| 5 | TF32 flags (`torch.backends.cuda.matmul.allow_tf32`, `cudnn.allow_tf32`) | none set; compute stays fp32 | PASS |
| 6 | `torch.compile`, `scaled_dot_product_attention`, `flex_attention`, `torch.export` | none in active code (only `preflight/torch_compat.py` forbidden-list + `preflight/tests/fakes.py` fixture) | PASS |
| 7 | `torch.cuda` device calls | `bundle/timing.py:90-91` (`is_available`), `179-180,186-187,196-197` (`synchronize`) — stable API in 2.7-2.12 | PASS |
| 8 | `torch.set_num_threads(1)` / `torch.Generator().manual_seed` | `bundle/timing.py:89,96,164,216` | PASS |
| 9 | `Tensor.scatter_reduce`, `nonzero`, `where`, `clamp` | `ctrl/*` (see `ctrl/PARITY.md`) | PASS (2.5-2.12) |
| 10 | `resource` import (POSIX-only) | `bundle/timing.py:146-155` guarded by `try/except` with `tracemalloc` fallback | PASS on Windows/CPU |
| 11 | Precision: stored corpus arrays are `np.float16`; model compute is fp32. No `half()`/`autocast` | `features/featurizer.py`, `train/dataset.py` (storage only) | PASS; bf16/AMP not needed. If added later use `torch.amp.autocast("cuda", dtype=torch.bfloat16)` |
| 12 | Residual prose still says "torch 2.5.1" | `train/trainer.py:9`, `train/idea_a.py:1101`, `rollout2/smoke.py:6`, `WM_N1_IDEA_A_DESIGN.md:222`, `REVIEW_*.md` | Informational only (not API commitments); left unchanged (outside the edited scope) |

No 2.5-only behaviour was found. Nothing depends on a removed API.

## 3. Changes made

- **Pin:** `bundle/requirements-wmn1.txt` -> `torch>=2.7,<2.13` + cu128 extra-index.
- **All box configs:** `bundle/box_configs/**` (2box a/b/c, 3box, 4box — 13
  `box*.json`) `gpu_env` updated (python 3.10/3.11, torch range, cu128 note,
  envcheck pointer).
- **Planner:** `bundle/multibox.py` `_GPU_ENV` updated to match.
- **Version gate + scan:** `preflight/torch_compat.py` now targets
  `>=2.7,<2.13` (runs the op test on 2.12.1+cpu locally; skips outside the
  window) and forbids deprecated `torch.cuda.amp` / `torch._six`.
- **Gate wording:** `preflight/gates/g5_e2e_smoke.py` title/messages.
- **New:** `bundle/envcheck.py` (+ `bundle/tests/test_envcheck.py`).
- **Docs:** `CODE_READY.md`, `bundle/README.md`, `bundle/box_configs/README.md`,
  `preflight/README.md`, `adapter/checkpoint.py`, `adapter/README.md`,
  `INTERFACES.md`, `INTERFACES_V2.md`, `ctrl/PARITY.md`, `train/README.md`.

## 4. Box-side self-check — `python -m bundle.envcheck`

Run from `output/wd01/wm_n1` on the box, before any queue stage:

```bash
python -m bundle.envcheck                 # auto: cuda if available, else cpu
python -m bundle.envcheck --device cuda   # require CUDA
python -m bundle.envcheck --device cpu    # local CPU fallback (parity box)
python -m bundle.envcheck --json          # machine-readable report
```

It prints torch/CUDA versions, `torch.cuda.get_arch_list()`, device name and
compute capability, then runs a tiny **forward + backward of the REAL
`model.net.WmN1Net` member** (the exact module the timing gate measures) via
`bundle.timing`, and checks the loss/gradients are finite.

Refuses (exit code 2) when:

- `torch < 2.7` (Blackwell cu128 minimum);
- a Blackwell GPU (compute capability major 12) is present but the arch list
  lacks `sm_120` / `compute_120`;
- `--device cuda` is requested while `torch.cuda.is_available()` is `False`.

Exit code 0 = PASS.  On the local CPU-only box the same command runs the CPU
fallback and passes (torch 2.12.1+cpu).

## 5. Local verification (this workspace, no GPU, no downloads)

```
python -m bundle.envcheck --device cpu
python -m pytest -q                         # full WM-N1 suite, one process
python -m preflight all --data-root E:/research-dense
```

Results are recorded in the final execution report (pytest count, PREFLIGHT
6/6).  `WD_FORBID_D=1` was set for every command; zero SUMO; no
HOLDOUT/CAL/chunk-5 reads.
