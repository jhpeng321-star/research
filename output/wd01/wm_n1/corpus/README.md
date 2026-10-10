# CORPUS (WM-N1 piece)

Turns `inventory/inventory.parquet` into per-run `features/` shards, packs them
into the train read-side memmap layout (`train/dataset.py`), verifies the pack,
and builds an upload manifest that refuses HOLDOUT / excluded / chunk-5 runs.
Also ships the real `evalh` `EvalSource` (`corpus.source.PackedEvalSource`,
INTERFACES Mismatch 5).

## FEAT1 — one-command v1 + Idea A over the whole TRAIN+CAL corpus

The P9 blockers are fixed (branch `zcode/feat1prep`): the J4
`member_left_network` wall runs featurize (terminal-tick departed-member
tolerance, `corpus/j4_wall_fix.py`, frozen v1 files untouched),
`featurize-all` writes and packs the Idea A arrays in the same pass
(documented 2-line `_run_one` patch), the branch-arm counter history
resolves through the plan's `bases_root` (`runs/campaign_stage1_chunk4`),
`lower_priority_below_normal` works on 64-bit Python, and the owner-approved
D: output override is available. Interpreter:
`C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe` (cwd
`output/wd01/wm_n1`).

### FEAT1 shard-key fix (branch `zcode/feat1fix`) — shards_v2

The first full run crashed because shards were keyed by the plan-level
POLICY arm (`shards/chunk1/w8410m2700p02t480/fixed`), so replicated arms
(`__fixed1`/`__fixed2`, `__switching1/2/3`) collapsed into one shard and the
writer refused. Since `zcode/feat1fix` every shard is keyed by the **exact
run-dir arm suffix** under the **batch** first component:

    <out>/<batch>/<world>/<run-dir arm suffix>     e.g. shards_v2/chunk1/w8410m2700p02t480/fixed1

The policy label (`fixed`/`switching`/…) and the chunk label are metadata
only (`PLAN_MAP.json`, `meta.json`, `DONE.json`, `IDEA_A_DONE.json`,
pack items, index records). The batch component also disambiguates the
cross-batch WM-72 runs that share world + run name (`chunk2`/`chunk3`
native+pi vs the `truearms_chunk2`/`truearms_chunk3` genuine fleet arms,
different `root_rel`). Audit over the refreshed
`inventory/inventory.parquet` (2476 TRAIN+CAL rows, 2463 readable):
**980 shard-path collisions before -> 0 after** (readable subset
971 -> 0); all 1687 inventory contrast pairs resolve to distinct shards
(`corpus/tests/test_inventory_injectivity.py`). Pack item keys
`(batch, world, arm)` are unique and `corpus verify` now fails on duplicate
items; MemmapCorpus pair keys stay `world@t0=<t0>` with `policy` read from
item metadata, so each replicate pairs with the single native reference.

```powershell
$env:WD_FORBID_D = "1"
$py = "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe"

# 1) v1 + Idea A shards, whole TRAIN+CAL corpus (resumable, <=6 workers).
#    Writes ONLY under D:/rd-staging/wmn1_feat on D:; every other D: path
#    (in particular D:/research-dense) stays blocked by the forbid-D hook.
#    shard key = <batch>/<world>/<run-dir arm suffix> (injective, see above)
& $py -m corpus featurize-all `
    --inventory output/wd01/wm_n1/inventory/inventory.parquet `
    --data-root E:/research-dense `
    --out D:/rd-staging/wmn1_feat/shards_v2 `
    --workers 6 --allow-d-output-root D:/rd-staging/wmn1_feat

# 2) pack (TRAIN_ARRAYS now includes the five Idea A arrays)
& $py -m corpus pack `
    --data-root D:/rd-staging/wmn1_feat/shards_v2 `
    --out D:/rd-staging/wmn1_feat/pack_v2 `
    --workers 6 --allow-d-output-root D:/rd-staging/wmn1_feat

# 3) verify the packed corpus (arrays, index, manifest, SHA256SUMS, sources)
& $py -m corpus verify --pack D:/rd-staging/wmn1_feat/pack_v2

# 4) Idea A receipt verify over the shard tree (sha tamper detection)
& $py -m corpus.idea_a verify --data-root D:/rd-staging/wmn1_feat/shards_v2

# 5) upload manifest; refuses HOLDOUT / holdout_exclusions.json / chunk-5
& $py -m corpus manifest `
    --pack D:/rd-staging/wmn1_feat/pack_v2 `
    --out D:/rd-staging/wmn1_feat/upload_manifest.json `
    --allow-d-output-root D:/rd-staging/wmn1_feat
```

Notes: the crashed `D:/rd-staging/wmn1_feat/shards` (policy-keyed, partial)
is abandoned — do NOT resume into it; `shards_v2` starts fresh (the old
tree is another consideration: never delete it here, and never write into
it). Resume a killed v2 run by re-issuing command 1 (skip-by-sha on both
the v1 and Idea A stages); `corpus.idea_a backfill` remains as the repair
tool for pre-patch shard trees and is a no-op after command 1 (it resolves
the same batch + arm-suffix keys). D: needs >= ~40 GB free for shards +
pack + headroom (estimate below). The featurize memory gate (>= 10 GB free
virtual/RAM) still applies; keep `--workers <= 6`.

## Exact CLI lines (E: roots)

```powershell
# inventory -> per-run shards (resumable, skip-by-sha; <=6 workers)
python -m corpus featurize-all --inventory output/wd01/wm_n1/inventory/inventory.parquet --data-root E:/research-dense --out runs/wm_n1_shards --workers 4

# shards -> packed memmap (manifest.json + INDEX.json + SHA256SUMS)
python -m corpus pack --data-root runs/wm_n1_shards --out runs/wm_n1_pack --workers 4

# re-pack / recast an existing pack (no raw physics)
python -m corpus convert --in-dir runs/wm_n1_pack --out runs/wm_n1_pack_f32

# verify the packed corpus (arrays, index, manifest, SHA256SUMS, source shas)
python -m corpus verify --pack runs/wm_n1_pack

# upload manifest; refuses HOLDOUT / holdout_exclusions.json / chunk-5
python -m corpus manifest --pack runs/wm_n1_pack --out runs/wm_n1_upload_manifest.json
```

`featurize-all` sets Windows BELOW_NORMAL priority, enforces `--workers` <= 6,
and gates on **>= 10 GB** free virtual memory before start and between batches
(pausing while low); it writes `ALLOCATION.json`, `PLAN_MAP.json` and
`opened_paths.json` receipts. The full corpus was **not** featurized.

## Idea A channels (P9, `corpus/idea_a.py` + `features/idea_a.py`)

Five extra per-run arrays (`veh_kin` [N,128,7] f16, `hdv_base` [N,128,6] f16,
`cav_plan_ch` [N,128,8] f16, `event_tok` [N,8,8] f16 all-zero,
`event_mask` [N,8] u8 all-zero) written next to the v1 arrays with their own
`IDEA_A_DONE.json` receipt (sha256 per array, `arrays_bytes`,
`amendment2_road_end_relabel` + `road_end_relabelled_n` on J4 wall arms).
The frozen v1 `DONE.json`/`meta.json` are never touched. The HDV baseline
channels replay the frozen `partA_v2g` nominal rules
(`params_nominal.json`, LF-sha256 `079a5576…35f7`, provenance-checked;
every channel uses recorded data `<= t` only).

**Ownership (INTERFACES_V2 file table):** P9 owns only `features/idea_a.py`
and `corpus/idea_a.py` (+ their tests). `corpus/featurize_all.py` and
`corpus/cli.py` belong to the landed CORPUS piece, so the single-command
integration is a documented INTEGRATOR patch, not code here. The exact
2-line patch in `corpus/featurize_all.py::_run_one`, after
`rec = W.write_shard(shard, job["data_root"], force=...)`:

```python
from .idea_a import write_idea_a_channels
write_idea_a_channels(shard, W.shard_dir(job["data_root"],
                        job["chunk"], job["world"], job["arm"]))
```

**FEAT1 status: this patch has LANDED** (branch `zcode/feat1prep`), and
`corpus/pack.py` carries the extension:

```python
TRAIN_ARRAYS = TRAIN_ARRAYS + ("veh_kin", "hdv_base", "cav_plan_ch",
                               "event_tok", "event_mask")
```

(`corpus/verify.py` imports `TRAIN_ARRAYS` from `corpus.pack`, so it picks
the extension up; the packed-array roundtrip incl. the new arrays is proven
by `corpus/tests/test_idea_a_corpus_channels.py` under exactly this
monkeypatch.) Until that patch lands, the two-command form is:

```powershell
# v1 shards (registered driver, split-before-open, sealed-refusing)
python -m corpus featurize-all --inventory output/wd01/wm_n1/inventory/inventory.parquet --data-root E:/research-dense --out runs/wm_n1_shards --workers 6
# + the P9 Idea A channels (resumable, idempotent; no-op after the patch)
python -m corpus.idea_a backfill --data-root runs/wm_n1_shards --raw-data-root E:/research-dense --workers 1
python -m corpus.idea_a verify --data-root runs/wm_n1_shards
```

`python -m corpus.idea_a backfill` rebuilds each v1 shard in memory and adds
the Idea A channels; with the patch in place it becomes a skip-only no-op.
`corpus.idea_a.verify_idea_a` re-checks every receipt sha (tamper-detecting).

### P9 micro-smoke (TRAIN only, tmp dir, 1 worker, BelowNormal)

3 runs through the REAL `featurize_all` driver over a 3-run inventory copy
(chunk-4 TRAIN `w8623m2600p02t540` base+native, J4 TRAIN wall branch
`w8659m2700p10t420__W-lo-early`), then the P9 backfill:

```text
featurize-all: runs=3 written=3 skipped=0 workers=1
  [written] chunk4/w8623m2600p02t540__base n_ticks=2281 bytes=13300267
  [written] chunk4/w8623m2600p02t540__native n_ticks=1201 bytes=7388218
  [written] chunk4/w8659m2700p10t420__W-lo-early n_ticks=1201 bytes=7359316
idea-a backfill: v1_shards=3 runs=3 written=3 skipped=0 workers=1 wall=115.1s
  chunk4/w8623m2600p02t540__base      n_ticks=2281 v1=13300267 idea_a=12573512 relabel=False(0) v1_B/tick=5831 idea_B/tick=5512
  chunk4/w8623m2600p02t540__native    n_ticks=1201 v1=7388218  idea_a=6620552  relabel=False(0) v1_B/tick=6152 idea_B/tick=5513
  chunk4/w8659m2700p10t420__W-lo-early n_ticks=1201 v1=7359316 idea_a=6620552  relabel=True(0)  v1_B/tick=6128 idea_B/tick=5513
```

Resume: second `featurize-all` + `backfill` -> `written=0 skipped=3` (both
stages, skip-by-sha). Verifiers (fresh processes, `WD_FORBID_D=1`):
`python -m features verify` -> `{"ok": true, "n_runs": 3, "errors": []}`;
`python -m corpus.idea_a verify` -> `{"ok": true, "n_runs": 3, "errors": []}`.
The amendment-2 flag is set on the J4 wall arm (no in-hold abort in this run
-> count 0; the clean release has 0 aborted tokens).

### Updated full-corpus size + wall-time estimate

Plan-only `features estimate` totals (v1 arrays): chunk1 4.174 GB (504
plan arms), chunk1_fix27 0.873 GB, chunk2 4.172 GB, chunk3 4.166 GB,
truearms_chunk2/3 2.014 GB each, chunk4 **5.337 GB / 648 runs**, J4 walls
pilot 1.294 GB (176 runs). The registered INVENTORY selection is 2476
TRAIN+CAL runs (inventory-driven, so e.g. chunk1/2/3 contribute 448 runs
each and the truearms 112 each), i.e. **v1 corpus ~19.8 GB**. The Idea A
arrays add a measured **5512 B/tick + 640 B/run** (0.90-0.95x the v1 bytes;
smoke ratio 0.920), so **v1+Idea A ~38 GB uncompressed**. Wall-time: the
smoke measured 5.2 s/run (v1) + ~38 s/run (P9 scene pass, single worker,
this laptop) -> **~5 h at 6 workers for the full corpus** (fused into
featurize-all by the patch above; the separate backfill pass would add the
~5 s/run v1 rebuild). Pack grows by the same Idea A arrays. FEAT1 sizing
for the D: root `D:/rd-staging/wmn1_feat`: shards ~19.8 GB (v1) + ~18 GB
(Idea A) + pack of the same magnitude -> provision >= ~40 GB free on D:
and keep the 10 GB virtual-memory gate active. J4 wall runs add the
pre-restore counter history (blocker-3 fix), so their shard bytes grow
slightly versus the P9 smoke; non-J4 shards are byte-identical (verified
below).

### Blockers found by the P9 smoke — ALL RESOLVED by FEAT1 (zcode/feat1prep)

1. **J4 `member_left_network` runs failed the frozen v1 featurizer.** At the
   safety-abort tick the departed members are gone from the physics row's
   `vehicles` while `wall_events.jsonl` still names them, so
   `wall_tokens_v04.WallEpisodeSet.walls_at` raised `WallTokenError:
   non-CAV member(s)` via `features/featurizer.py:278` (roster =
   per-tick present CAVs). All 55 TRAIN member_left_network wall runs were
   affected (mechanism probed on `w8661m2600p10t240__w-mid-early`,
   `w8660m2700p10t300__w-lo-late`, both TRAIN). **Resolution:** the
   corpus-owned wrapper `corpus/j4_wall_fix.py` (installed by
   `featurize_all._run_one` and `corpus.idea_a`) tolerates the departed
   members at the wall's terminal tick ONLY — the episode ends at the
   members' last present tick; active ticks keep the loud contract error;
   `features/featurizer.py` and `partB_v0data/wall_tokens_v04.py` are
   byte-unchanged. The abort tick now lands in the wall tensor with the
   `aborted` phase, and `IDEA_A_DONE.json` records
   `amendment2_road_end_relabel=true` + `road_end_relabelled_n` (amendment
   2: `member_left_network -> road_end`; run receipts keep their labels).
2. **`corpus._util.lower_priority_below_normal()` always returned False on
   64-bit Python** (classic `GetCurrentProcess` pseudo-handle truncation:
   `ctypes` default `restype=int`, `GetLastError: 6`). **Resolution:**
   proper `c_void_p` restype + `SetPriorityClass` argtypes (psutil preferred
   when importable); covered by `corpus/tests/test_priority_fix.py`.
3. Branch-arm counter pre-history: the walls-pilot base runs live under
   `runs/campaign_stage1_chunk4` (the plan's `bases_root`), while the
   driver resolved `base_history_run_dir` inside the walls-pilot root, so
   J4 branch counters carried no pre-restore history. **Resolution:**
   `featurize_all.BASES_ROOT_REL` maps `walls_pilot_chunk4` ->
   `runs/campaign_stage1_chunk4` (with a plan-row `bases_root` fallback
   anchored under the data root), and the J4 counter channel now carries
   the 300 s pre-restore history (asserted in the J4 tests).
4. **D: output override (owner-approved):** `--allow-d-output-root
   D:/rd-staging/wmn1_feat` permits writing ONLY under that exact root;
   `WD_FORBID_D=1` keeps blocking every other D: path, and
   `D:/research-dense` (+ `D:/research-dense-backups`) are never
   overridable (`forbid_d_hook.PROTECTED_D_ROOTS`). Covered by
   `corpus/tests/test_d_override.py`.

## Micro-smoke (2 chunk-4 TRAIN runs, 1 worker)

```text
  [written] chunk4/w8623m2600p02t540__base n_ticks=2281 bytes=14741859
  [written] chunk4/w8623m2600p02t540__native n_ticks=1201 bytes=8147250
featurize-all: runs=2 written=2 skipped=0 workers=1
```

Second (resume) invocation:

```text
  [skipped] chunk4/w8623m2600p02t540__base n_ticks=2281 bytes=14741859
  [skipped] chunk4/w8623m2600p02t540__native n_ticks=1201 bytes=8147250
featurize-all: runs=2 written=0 skipped=2 workers=1
```

Pack + verify over those shards:

```text
corpus pack: n_runs=2 n_ticks=3482 n_decision=40 bytes=19249683
corpus verify: {"ok": true, "n_runs": 2, "n_items": 2, "n_arrays": 10, "errors": []}
```

## Tests

```powershell
$env:WD_FORBID_D="1"; python -m pytest output/wd01/wm_n1/corpus/tests
```

Covers split-before-open + opened-path audit, resume/skip-by-sha, pack->verify
roundtrip + tamper detection, packed-layout parity with `train/dataset.py`
(MemmapCorpus-equivalent reader over the pack output; packsynth fixture copied
read-only from `zcode/wmn1train`), manifest sealed-world refusals, `EvalSource`
rows in `evalh.protocol` shape, and the mocked memory gate.
