# FEAT1 corpus pack_v2c — completion report (WM-N1, branch `zcode/feat1v2c`)

Date: 2026-10-01. Base: `zcode/feat1j4fix` @ `e100fa6` (reviewed J4 wall-token
fix). Executor: ZCode GLM session, python
`C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`, `WD_FORBID_D=1` with
the single owner-approved D: root `D:/rd-staging/wmn1_feat`. Zero SUMO; no
`--force`; nothing overwritten or deleted; `pack_v2`, `upload_v2`, `shards`
(the old tree) and `D:/research-dense` untouched.

## 1. Corpus counts and the count assertion

```
2476 inventory rows - 13 non-featurized = 2463 expected
     (select_runs readable TRAIN/CAL = 2463); verified shard dirs = 2463;
     INDEX.json runs = 2463
```

- `shards_v2` before this task: INDEX 2445 runs (18 J4 wall runs absent,
  asserted per-run directory absence before any write; PLAN_MAP 2463 keys,
  snapshot `logs_v2c/pre_v2c_PLAN_MAP.json`).
- Featurize of EXACTLY the 18 runs listed in `J4_WALLTOKEN_FAILED_RUNS.txt`
  (subset parquet `logs_v2c/sub_j4_18.parquet`, 18 rows):
  `featurize-all ... --workers 6 --idea-fast --min-free-ram-gb 8
  --min-free-virtual-gb 10` → `runs=18 written=18 skipped=0 failed=0`,
  started 06:37:14Z finished 06:44:17Z, memory gate at start: RAM 16.0 GB.
  PLAN_MAP.json restored to the full 2463-entry map after verifying the
  subset-run's 18 entries are identical to the snapshot's.
- Full-tree verification (frozen quarantine-scan verifier: every v1
  `DONE.json` output sha256 + every Idea A receipt sha/bytes, read-only):
  **2463/2463 ok** (`logs_v2c/assertion_v2c.json`).
- `python -m corpus.idea_a verify --data-root .../shards_v2` →
  **`ok: true, n_runs: 2463, errors: []`**.

## 2. pack_v2c (NEW directory)

`python -m corpus pack --data-root D:/rd-staging/wmn1_feat/shards_v2 --out
D:/rd-staging/wmn1_feat/pack_v2c --workers 6`:

- schema `wm_n1_corpus_pack/2`, **n_runs 2463**, **n_ticks 3,190,623**,
  **n_decision 49,260**, **k_max 128**, **totals.bytes 35,309,431,491**
  (35.31 GiB on disk), created 2026-10-01T07:07:47Z.
- `python -m corpus verify --pack pack_v2c` → **`ok: true, n_runs: 2463,
  n_items: 2463, n_arrays: 15, errors: []`** (re-hashes every packed array,
  validates the TRAIN read-side manifest and every SHA256SUMS entry, and
  re-verifies all 2463 source shard DONE hashes).

## 3. Upload manifest

`python -m corpus manifest --pack pack_v2c --out
D:/rd-staging/wmn1_feat/upload_manifest_v2c.json` (default workspace
holdout exclusions; chunk-5/HOLDOUT refused by the pack/manifest code):

- files 19, n_runs 2463, splits 2153 TRAIN / 310 CALIBRATION, excluded 0.
- On-disk sha256: `b2efda9109de399d9db678ce8dbf4b48e2d80a2cf58c6207219bf357929ec841`
  (this is the sha recorded in `PARTS.json`/`UNPACK.md`, same
  hash-the-on-disk-bytes convention as upload_v2's `a1a5a871…`).
  Note: the CLI's stdout `sha256` (`aa36eca1…`) is the sha of the LF body
  before Windows newline translation — a pre-existing `write_manifest`
  quirk, not a v2c defect; the anchor that matters (actual file bytes)
  matches in both v2 and v2c.

## 4. upload_v2c (NEW directory) and local verification

Built exactly like upload_v2: deterministic GNU tar (18 members =
pack dir minus `SHA256SUMS`, sorted by name, mtime=0, uid=gid=0, mode 0644,
flat names) streamed through zstd CLI `v1.5.7,` level 6 `-T6`, split into
fixed 2,147,483,648-byte parts:

| part | bytes | sha256 |
|---|---|---|
| pack_v2c.tar.zst.part000 | 2147483648 | `2cd214ebe5f4ab8bb8a34a44754418317e61d523a4b7fbb53adb91d255aeabaf` |
| pack_v2c.tar.zst.part001 | 2147483648 | `b7f583a556535259ccbaaffbe537926a449c1c856764e94b3a2c164f9d617a19` |
| pack_v2c.tar.zst.part002 | 2147483648 | `a744961c913cff37c3207bc99975cd9ec73ae5150cf9f90cb65473a1c7c76eaa` |
| pack_v2c.tar.zst.part003 | 469987741 | `abc1eb84e9d0b4e4979ec0789e2c16f64fc22e5cc29cef7277edcdd9adf4ff78` |

- compressed stream 6,912,438,685 bytes; member bytes 35,311,442,370
  (= pack 17 payload files 35,309,431,491 + INDEX.json 2,010,879).
- `SHA256SUMS` + `PARTS.json` (schema `wm_n1_upload_parts/1`) +
  `UNPACK.md` written; anchors: pack `SHA256SUMS` sha256
  `53c27b371070a5d56167f2e289734ff5769e01c9624c5b3fba47a77ebd1da1d2`,
  upload_manifest sha256 `b2efda91…` (above).
- Local verification (`feat1v2c_tools/upload_verify_v2c.py`): every part
  re-hashed and matched to SHA256SUMS+PARTS.json (sizes + shas), anchors
  re-checked, then streaming verification without writing the tar to disk:
  parts concatenated in order → `zstd -dc` → tar stream listing →
  **`ok: true, parts: 4, members: 18, member_bytes_total: 35,311,442,370,
  errors: []`** — and every member payload sha256 matched the pack's
  `INDEX.json` `files` entry (byte-identity of archive vs pack).
  (upload_v2 has a stray 0-byte `part004` not listed in its SHA256SUMS or
  PARTS.json; v2c omits stray empty parts.)

## 5. The J4 fix that re-includes the 18 walls_pilot_chunk4 runs

Mechanisms (full detail: `J4_WALLTOKEN_DIAGNOSIS.md` + `_RAW.json`,
commit `e100fa6`):

- **A — road-end detection latency (15 runs):** a member reaches the
  amendment-2 road end 0.5–1.5 s BEFORE the recorded `member_frame_missing`
  `safety_abort`; the old wrapper tolerated the terminal tick only.
- **B — slot-pool drop (3 runs + released terminals):** dense worlds exceed
  the frozen 128-slot pool (`n_dropped_concurrent` 18282 / 7287 / 6458 for
  `w8669m2700p10t360__w-mid-early` / `__w-mid-late` /
  `w8704m2700p40t480__w-mid-late`), so members are present+CAV-reported but
  unrostered at active ticks.

`corpus/j4_wall_fix.py` keeps the unconditional terminal-tick tolerance and
adds evidence-gated tolerance at other visible ticks via one lazy per-run
members-only physics pass (absent-from-network at t = road-end case per
amendment 2; present-but-ever-CAV-reported = report/slot gap, last reported
state carried; present + never-reported = data bug, still loud; no physics
evidence = old behaviour). Event timeline and relabel untouched; frozen v1
files untouched; byte-identity on finished runs 2/2 (38/42 files sha-equal;
4 receipt JSONs differ only in `shard_dir`/`written_utc`).

The 18 now included (all `walls_pilot_chunk4`, written 18/18 failed=0):

| run | split | mechanism |
|---|---|---|
| w8658m2600p10t360__w-lo-early | CALIBRATION | A |
| w8664m2600p10t360__w-lo-early | TRAIN | A |
| w8667m2700p10t420__w-lo-early | TRAIN | A |
| w8669m2700p10t360__w-mid-early | TRAIN | B (slot-pool drop) |
| w8669m2700p10t360__w-mid-late | TRAIN | B (slot-pool drop) |
| w8670m2700p10t300__w-lo-late | TRAIN | A |
| w8682m2400p20t240__w-lo-late | TRAIN | A |
| w8684m2700p20t360__w-lo-early | TRAIN | A |
| w8686m2600p20t240__w-mid-late | TRAIN | A |
| w8692m2600p40t300__w-lo-late | CALIBRATION | A |
| w8694m2600p40t240__w-lo-late | TRAIN | A |
| w8697m2700p40t240__w-lo-late | TRAIN | A |
| w8698m2600p40t300__w-lo-late | TRAIN | A |
| w8700m2600p40t420__w-mid-late | TRAIN | A |
| w8703m2600p40t540__w-lo-early | TRAIN | A |
| w8704m2700p40t480__w-mid-late | TRAIN | B (slot-pool drop) |
| w8705m2700p40t300__w-lo-early | TRAIN | A |
| w8705m2700p40t300__w-lo-late | TRAIN | A |

## 6. Slot-pool census (read-only; counts only)

Full tables: `SLOT_POOL_CENSUS.md` / `SLOT_POOL_CENSUS.json` (metric
definitions inside). Headline: across all 2463 runs the featurizer dropped
**21,368,334 vehicle-ticks** because >128 vehicles were concurrent
(5.9998% of 356,151,464 rostered vehicle-ticks); 1,200,454 ticks
(37.62%) ran with the 128-slot pool full (upper-bound indicator set of
drop-affected ticks). By loading: m2000 = 0 dropped vehicle-ticks,
m2400 = 1.89%, m2600 = 5.25%, m2700 = 7.75% — slot-pool pressure is a
dense-traffic effect, consistent with mechanism B. Worst single runs:
`walls_pilot_chunk4/w8659m2700p10t420__w-mid-early` 73,914 and
`w8662m2600p10t300__w-mid-early` 64,989 dropped vehicle-ticks.

## 7. Disclosure chain (attempts, crashes, stall)

1. **v2 (bcf1ddc):** 12-worker live featurize relaunch after stop+quarantine
   evidence; n_done 420/2463 at T+23.5 min.
2. **v3 (573a02a):** 09:28 `BrokenProcessPool` — parent could not unpickle
   worker exceptions (`wall_tokens_v04` not on the parent path); driver now
   puts wm_n1 on the parent path; quarantine v3 scan 1825 dirs (1 moved);
   byte-identity 3/3; relaunch 10 workers.
3. **v4 (ae7550c):** crash #2 — same `ModuleNotFoundError wall_tokens_v04`
   (partB root also missing on parent) → `features.actions_adapter` import
   added; per-run exclusion-and-continue; 1-worker repro isolated
   `w8658m2600p10t360__w-lo-early` WallTokenError → 18 wall runs explicitly
   excluded; 4 partial dirs quarantined (1834 scanned, 1830 ok, 4 moved).
4. **J4 fix (e100fa6):** all 18 diagnosed (mechanisms A/B) and re-featurized
   into `E:` `_tmp_j4fix_refeat` (--idea-fast, workers=1, failed=0);
   byte-identity 2/2 old-wrapper TRAIN runs vs live shards_v2 read-only;
   D: never written during diagnosis.
5. **v5 (concurrent session, logs_v5):** 6 disjoint subset relaunches
   finished the remaining 613 runs; consolidation 2445 verified;
   pack_v2 (35,071,202,460 bytes, 18 files) and upload_v2 (4 parts +
   SHA256SUMS + PARTS.json + UNPACK.md, built UTC 06:08:50Z = 14:08 local)
   produced from the 2445-run corpus.
6. **Observed concurrent interference (this task, untouched):** at
   ~14:19–14:21 local another process rebuilt `upload_v2` from scratch and
   was interrupted — only a truncated `pack_v2.tar.zst.part000`
   (796,917,760 bytes) remains there. Not touched (constraint), and v2c
   work is fully separate (`pack_v2c`, `upload_v2c`, `upload_manifest_v2c.json`).
7. **v2c (this task):** subset featurize 18/18 → full-tree verify 2463/2463
   → idea_a verify ok → pack_v2c → corpus verify ok → manifest →
   upload_v2c build → local verify ok → census → this report.

## 8. Constraint compliance

- `WD_FORBID_D=1` active; the only D: writes are under
  `D:/rd-staging/wmn1_feat` (forbid-D hook installed with that single
  allowed root for every writing step).
- Memory gate before heavy steps: free RAM ≥ 8 GB (measured 15.0–16.1 GB)
  and free virtual ≥ 10 GB; featurize driver re-enforced both
  (`--min-free-ram-gb 8 --min-free-virtual-gb 10`).
- CPU: featurize/pack drivers run `BELOW_NORMAL`; compression `-T6`.
- Zero SUMO; HOLDOUT/chunk-5 never selected (`select_runs` + split
  refusals; manifest excluded=0); `dispatcher_rows`/`insertions`/
  `evaluate_summary`/`suffix_summary` never read.
- No `--force`, no overwrites, no deletions; git: no reset/clean/
  checkout --/stash drop used.

## 9. Ledger status (honest disclosure)

No new ledger entries were added by this bounded task (the integrating
agent owns the shared registry; another agent was concurrently active on
`D:/rd-staging/wmn1_feat`). Read-only checks: `validate` reports 1560
pre-existing hard errors (MIG-04-era missing `output/...` paths) and
`verify-sources` 11 pre-existing source-missing failures — both unchanged
by, and unrelated to, the v2c corpus work; flagged for the integrating
agent's close-out.
