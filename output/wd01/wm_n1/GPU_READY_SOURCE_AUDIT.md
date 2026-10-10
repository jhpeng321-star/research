# WM-N1 m1 code handoff: source audit and exact archive list

Source-only revision, 2026-10-02. No Python import, test, training, simulation, pack, sidecar, CAL content, credential, SSH, upload, or archive command was run for this revision. The accepted recovery source inspected is `E:/rd-wt/wmn1recovery` at `8bf2f47`; this report is authored on the separate `gpurunbook` report branch. The shipping source HEAD can still change during guard compatibility repair. Recheck the final HEAD and put it in a shipping copy of the manifest before packaging. The owner has authorized uploading the reviewed training package code, checks, and newly built TRAIN sidecars to AutoDL File Storage before GPU power; this source audit does not claim any file has been uploaded or that a GPU is ready.

## Current source finding

The accepted recovery branch contains `bcc9b43` and the CLI seed fix `8d33a08`, both ancestors of `8bf2f47`. The optional `WM_N1_CANONICAL_PACK` alias check exists in `train/idea_a.py`. The timing gate calls the trainer's canonical `prepare_training_source`; in the accepted source that method binds the eligible contrast pair pool, identity receipt, and TRAIN-fitted outcome normalization before measuring batches. The older audit's "unmerged" and "alias absent" blockers were stale. Both the CLI and `bundle.e2e.run_config_job` now seed Torch immediately before constructing Idea A bundle weights. The integrating agent reports production seed tests (10 passed), loader tests (7 passed), launcher tests (12 passed), and pure guard tests (17 passed). Those results do not establish native PyArrow startup, real metadata planning, sidecar building, or CPU40 training; native guard compatibility repair is ongoing.

The supported m1 command is `bundle.multibox run-box-config` with `bundle/box_configs/2box/c_m1only/boxA.json`. Its executor, `bundle.e2e.run_config_job`, accepts only an Idea A member, requires CUDA and a real v3/K192 pack, preserves the full registered 15 × 2000 schedule and seed 101, refuses an existing member output, and stops after the checkpoint. It does not execute the config's dormant CAL/evaluation command strings. `boxB.json` is included only as the exact c_m1only plan record; its held-out task remains blocked. The legacy `bundle` queue's `member-1` stage still calls the v1 trainer and must not launch this m1 run. `preflight all` opens real TRAIN physics at G1 and is not a synthetic-only box check.

Both `WM_N1_TRAIN_OUTCOMES` and `WM_N1_TRAIN_IDENTITIES` are required for the real mixed-split pack, including timing. The $WORK/pack symlink may be used only after validating the canonical pack target and setting `WM_N1_CANONICAL_PACK` to that exact target. Missing or mismatched identity evidence refuses the run.

## Exact code archive

[GPU_CODE_FILE_MANIFEST.json](GPU_CODE_FILE_MANIFEST.json) names **104 individual files**. It enumerates the production `.py` files in `train`, `model`, `evalh`, `bundle`, `arrival`, `corr`, `ctrl`, `hdv`, `corpus`, and `features`, plus exact requirements, three c_m1only box config files, seven frozen registration Markdown files, the opaque CAL split record, `wdpaths.py`, `forbid_d_hook.py`, the three `partA_v2g` engine/parameter files, two exact `partA_ctrl` controller source files, and the opaque SHA-pinned CIRCLES `model.onnx`. It contains no tests, results, evidence, inventory, raw TRAIN data, checkpoints, other box plans, credentials, runtime binaries, or recursive directory pathspec. Some packaged production modules contain dormant build or evaluation functions; the m1 path does not execute them.

The source closure requires `features/idea_a.py`: `train.idea_a` imports its kinematics view and CAV scales, and `hdv.input_schema` imports the same view. `corpus.pack` imports `features.constants` and `features.writer`; the corpus package eagerly imports its production modules. `bundle.envcheck` optionally loads `bundle.timing`. The timing queue eagerly brings in `corpus`, `train.cli`, `model`, and `evalh.timing_gate`. `bundle.multibox` resolves `bundle.e2e.run_config_job` dynamically, which imports `train.idea_a` and builds `arrival`, `corr`, `ctrl`, and `hdv`; `hdv.__init__` also imports `hdv.input_schema`. `features.idea_a` has a deferred `partA_v2g.engine` import with frozen `params_nominal.json`; the extra path helper files cover deferred corpus/path guards. For damping or PI slots, `ctrl.mover.CavMover` lazily imports `partA_ctrl.emulators`, which imports the local `partA_ctrl.onnx2torch` reader and loads the frozen `baselines/circles/model.onnx`; all three are now exact archive members. The reader uses the already-declared NumPy/Torch requirements, without an external `onnx2torch` package. The excluded `partB_v0data` and `rollout2` dependencies belong to dormant build/evaluation paths and must not be invoked by m1. Any later plan to execute those paths needs a new closure audit.

Seven registration files and `CAL_SPLIT_A4.json` are mandatory frozen identities. `bundle.multibox.require_config_approval` checks the declared LF hashes and computes LF-normalized hashes over the actual eight files' bytes; missing CAL bytes refuse approval. The allowed m1 path imports `evalh.cal_split` for constants but does not call its CAL JSON parser. Treat CAL as opaque: include and hash the file without opening or printing its contents. The per-run `TIMING_GATE.json` is a ninth, separately produced raw-byte hash binding for the approval.

The committed manifest records `8bf2f47` as the inspected source basis and intentionally has no file hashes. All 104 listed paths were checked as unique, existing, tracked regular files at that source basis; no tracked production `.py` file in the ten listed package directories was omitted. Static path inspection found no sealed outcome file, dataset array/parquet, test, credential, or runtime binary. The opaque non-source payloads are the explicitly SHA-pinned ONNX model and the frozen CAL split identity JSON that must remain unread. This is a static closure check for the CPU producer, timing gate, and full m1 code paths, including inert modules reached by eager imports. It is not itself an archive member, so a committed self-reference to the final commit hash is unnecessary. After the final source commit, the integrating owner must review the integrated file list, write the final HEAD into a preserved shipping copy of the manifest, compute raw-byte hashes from the resulting archive members, and retain those hashes with the shipping receipt. Do not infer archive identity from working-tree CRLF bytes.

## Archive recipe for the integrating owner (not executed here)

Use a fresh private GUID directory, preserve the archive, and compare the actual tar members with the manifest's file list. This is a recipe for the final reviewed HEAD only; the code archive is authorized for handoff. It contains no data sidecars. Never pass the entire `output/wd01/wm_n1` directory as a Git pathspec.

```powershell
$repo = 'E:/rd-wt/wmn1recovery'
$manifestPath = Join-Path $repo 'output/wd01/wm_n1/GPU_CODE_FILE_MANIFEST.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$finalHead = (git -C $repo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve final source HEAD' }
# Review the integrated source closure and exact paths at $finalHead before continuing.
$paths = @($manifest.files | ForEach-Object { $_.path })
if ($paths.Count -ne $manifest.file_count -or ($paths | Select-Object -Unique).Count -ne $paths.Count) {
    throw 'Manifest count or uniqueness mismatch'
}
$shipDir = Join-Path $env:LOCALAPPDATA ('wmn1-ship-' + [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $shipDir) { throw 'Shipping directory already exists' }
$null = New-Item -ItemType Directory -Path $shipDir -ErrorAction Stop
$manifest.source_head = $finalHead
$manifest.source_head_status = 'final integrated shipping source'
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $shipDir 'GPU_CODE_FILE_MANIFEST.json') -Encoding utf8
$archive = Join-Path $shipDir 'wmn1-code.tar'
if (Test-Path -LiteralPath $archive) { throw 'Archive already exists' }
git -C $repo archive --format=tar --output=$archive $finalHead -- $paths
if ($LASTEXITCODE -ne 0) { throw 'git archive failed' }
$actual = @(tar -tf $archive | Where-Object { -not $_.EndsWith('/') })
if ($LASTEXITCODE -ne 0) { throw 'tar member listing failed' }
$delta = Compare-Object ($paths | Sort-Object) ($actual | Sort-Object)
if ($delta) { throw 'Archive members differ from exact manifest paths' }
Get-FileHash -LiteralPath $archive -Algorithm SHA256
```

The local archive file is created inside a fresh directory; no PowerShell binary pipe is used. The tar member comparison filters directory entries and rejects missing or extra files. Extract into a second fresh directory under `$shipDir` and hash each extracted member's raw bytes before shipping. Retain the archive, manifest, member hashes, final commit ID, and transport receipt. Use a fresh destination on each box and verify its archive hash and exact members before any environment or training command. No local or remote cleanup is scripted.

## Remaining gates

The native guard compatibility check, guarded real metadata plan, TRAIN sidecar build and verification, and registered real-pack CPU40 producer run are still pending in fresh Windows outputs. These stages have **not** executed as of this revision. The reviewed code archive can be staged once its final HEAD and member hashes are frozen; add each verified check receipt and TRAIN sidecar as it is produced, using fresh names and retaining upload receipts. CPU40 must pass before GPU power, not before an earlier code-only upload. The owner has already authorized uploading the exact reviewed code/checks package and the newly produced TRAIN-only identity and outcome sidecars to AutoDL File Storage before turning on either GPU box; no further owner approval is needed for that specific transfer. Preserve the seven frozen registration documents and opaque `CAL_SPLIT_A4.json` in the code archive: hash and transport CAL bytes without reading its outcomes or using it in a preflight. The 49 GB pack is already on both boxes; do not duplicate it in the code archive or sidecar upload. Keep both GPU boxes off during File Storage staging. The AutoDL browser session was unavailable in this source review, so File Storage region, capacity, and upload method remain unverified and no upload has occurred.

At final shipping HEAD, create a fresh exact-file code archive and separately enumerate the concrete check receipts and new TRAIN sidecar filenames, then hash every raw-byte archive member and sidecar before upload. Preserve upload receipts and verify the staged bytes before GPU power. On the boxes, verify the existing pack's prescribed hashes and canonical path before setting the alias environment variable. Then run `bundle.envcheck` and the real-pack timing gate with both TRAIN sidecar envs. GPU50 is a separate bounded CUDA smoke, with all five registered loss terms and a verified device/checkpoint receipt. Only after those checks does the root author the approval binding the frozen eight records and timing gate. gpuA may then run boxA m1; gpuB stops at timing. CAL tuning, held-out runs, members 2–5, and final tests remain outside this handoff.

Source inspection alone cannot certify the eventual archive bytes, real-pack timing, CPU40/GPU50 outcomes, or GPU readiness. Those are explicit acceptance checks at shipping and execution time.
