# Independent review — one-folder reorganisation V3

## VERDICT: FAIL

**V3 is not safe to execute as delivered.** Splitting out the main-tree rename is
an improvement over V2, but its executable guards do not establish a zero-writer
window. The default move selection also includes cross-volume phase 5, deletion
targets include the active review and its verification evidence, and cloud backup
is confused with permission to remove still-referenced local data.

Review date: 2026-10-09. Scope: supplied `_in/` evidence only. No live source paths
were inspected; snapshot facts are not fresh filesystem/Drive verification.
No reorganisation script was executed, and no data was moved, deleted or uploaded.
Credential files and restricted datasets were not opened. Python verification
uses only small text/JSON metadata, sequentially; no arrays or runs are loaded.

## Numbered findings and exact fixes

1. **BLOCKER — default phase-2 execution includes phase 5; cross-volume handling
   is unsafe.** `_in/scripts/phase2_moves.ps1:240-242` selects every move except
   `2b`, including all 12 phase-5 imports. `:217-223` checks only file counts,
   without size/hash or free-space checks. It leaves the original directory in
   place and then calls `New-CompatJunction`, which rejects that directory
   (`:140-147,227`). The `copy+verify` file method falls through to `Move-Item`
   (`:224-227`), removes the source without verification, then attempts a directory
   junction to a file. With 6.30 GiB free this can create partial copies and fail.
   **Fix:** default and allowlist phase `2a` only; remove phases `2b`/`5` from this
   executor. Implement cross-volume migration separately: measured per-item
   capacity plus working reserve, copy without removing source, per-member
   count/size/checksum verification, then a separately authorized quiescent
   cutover. `fixed/phase2_moves.ps1` restricts the execution surface to `2a`.

2. **BLOCKER — cleanup destroys the current review and evidence used to justify
   later deletion.** `_in/scripts/DELETE_AFTER_VERIFY.ps1:45,81` includes
   `tmp/drive_verify_20261006` and the entire `E:/rd-wt/reorg3` parent. Inventory
   records four active OpenCode processes under `reorg3` while marking it
   `live:false` (`_in/inventory_E.json:6715-6748`). The former directory is the
   cited preservation record for D: archives (`_in/INVENTORY_D.md:17,63-75,114`).
   Phase 1 would delete it before phase 3/4 (`_in/REORG_PLAN_V3.md:273-278`).
   **Fix:** remove both from every delete/reclaim list; retain the review, source
   inputs and recovery manifests durably. Never infer disposability from a
   `DELETE-CANDIDATE` label that contradicts process/reference evidence. Protect
   script, journal and receipt paths and all their ancestors. The fixed deleter
   disables destructive execution until the eligibility design is repaired.

3. **BLOCKER — Drive verification does not make live packs/evidence removable.**
   `_in/REORG_PLAN_V3.md:160,197-198,238-251,262-265` allows deleting packs before
   the later move or using Drive-only storage instead. The live
   `E:/rd-wt/ws-cl/_ext/pack_v3` dependency supplied in the brief and the recorded
   D: junctions would break. `_in/INVENTORY_D.md:75-81,92-97` identifies continuing
   consumers of `pack_v3`, `shards_v3` and CPU payload. `shards_v3` and CPU payload
   have no other copy recorded; the pack's powered-off GPU copies verify only
   18/18 hashes against 19 inventoried files (`:31,81`), not a full current
   recoverable tree. Five shard archives are truncated (`:70-73`).
   **Fix:** KEEP these local sources and all four label sidecars. Backups may be
   created during a stable snapshot window, but remove nothing until each local
   consumer has a verified accessible replacement and all historical paths still
   resolve. Do not delete first and then try copying a now-missing source. For a
   partial `wmn1_feat` migration, use individual child junctions; do not replace
   the whole parent while other children/manifests remain on D:
   (`_in/MOVE_MAP.json:934-996` has conflicting parent/child alias instructions).
   Frozen evidence offload additionally needs an explicit itemized owner decision
   and a local resolution/restore contract; a Drive URL is not a Windows junction
   target. `fixed/drive_offload.sh` emits backup-only receipts, never deletion
   authorization. Packs are absent from the original executable deletion list;
   the dangerous instruction is currently in the plan, not that list.

4. **BLOCKER — recursive deletion has no child/ancestor junction barrier and the
   protected-path guard is bypassable.** `_in/scripts/DELETE_AFTER_VERIFY.ps1:173-182`
   probes recursively *before* identifying a root link, inspects only the target
   root's ReparsePoint attribute, and never enumerates child links or reparse
   ancestors. After moving `rd-release`, its old path is an ancestor junction for
   the listed virtualenvs. A future nested link can expose an unrelated target;
   provider/version-specific recursion must not be the safety boundary.
   `-AllowProtectedSealed` actually disables protection for CAL/HOLDOUT/chunk5/
   J4B/INC1 as well (`:25,31,158`), allowing the two listed chunk5 failed attempts
   and J4B staging. Root junction removal does not reclaim the relocated target.
   **Fix:** remove the bypass; canonicalize and compare real-path boundaries;
   enumerate metadata iteratively without entering links, including ancestors;
   refuse recursive deletion of any linked/protected/unknown tree. Link removal
   must be an explicit link-only owner operation, with exact target recorded and
   no recursive flags. The fixed deleter contains no removal operation.

5. **BLOCKER — only-known-copy and research records are classified as disposable
   without proof.** Smoke/failed runs, the failed-branches tar and v1 collision
   shards are local-only deletion entries (`_in/scripts/DELETE_AFTER_VERIFY.ps1:65-69,99`).
   The v1 shards were explicitly kept “to be disclosed”
   (`_in/INVENTORY_D.md:36`); failure receipts/logs must be preserved
   (`_in/AGENTS.md:124-127`). Metadata of the already-deleted v2c pack and
   logs/quarantine also have no preserved evidence copy identified
   (`_in/inventory_D.json:34-44,114-115`). `plexp1d` redundancy is an assertion,
   not a file-hash comparison, and V2 had already recorded the proposed copy as
   absent (`_in/REORG2_REVIEW.md:33`). C: agent temp includes session/diff data
   (`_in/INVENTORY_CG.md:53`), not demonstrably reproducible cache.
   **Fix:** preserve these candidates until a per-file retain/delete manifest
   proves redundancy or the owner explicitly disposes of the recorded evidence.
   Keep failed-run receipts, original reports and reconstruction metadata. Do not
   rerun experiments to justify deletion. Treat missing backups as unknown, not
   safe. Global uniqueness cannot be proved from the supplied snapshots. The
   original deleter does use literal, wildcard-rejected paths; however the map's
   descriptive `wmn1_feat/{logs,quarantine_v2,*}` and other wildcard/braced delete
   entries (`_in/MOVE_MAP.json:1184-1191,1256-1274`) are not an executable manifest.
   Never expand them into a broad deletion of the active staging parent.

6. **MAJOR — status-only Drive receipts are neither usable nor sufficient.**
   The offloader writes a header without `local` (`_in/scripts/drive_offload.sh:40`);
   the deleter accesses `$r.local` under StrictMode (`DELETE_AFTER_VERIFY.ps1:27,145-146`),
   which fails on that header. If corrected, eligibility still trusts only
   `status='verified'`, ignores remote identity/check return codes/member hashes,
   and accepts an arbitrarily old source snapshot (`:165-170`). Source mutation
   after verification can destroy the only new version. Plain `rclone check`
   compares size and common hashes, but may fall back to size if hashes are
   unavailable; no check log/coverage requirement prevents this.
   **Fix:** define a versioned receipt schema binding exact canonical source,
   remote/object identity, full relative member set, bytes/hash algorithm/hashes,
   check/copy exits, checked timestamp and stable snapshot; reject unknown headers
   and malformed/incomplete receipts. Revalidate the same source and remote just
   before owner deletion while keeping writers stopped. Require actual common
   hashes for every source member, not `--size-only`; keep `--one-way`. The fixed
   backup copy checks one-way size/checksum plus complete MD5 listings and a second
   source hash listing; its `backup_verified` cannot unlock the original deleter.
   This is backup verification, not a deletion-grade immutable snapshot receipt.

7. **MAJOR — archive layout and upload behavior do not match the evidence.**
   Inventory verifies `.tar.zst` archives (`_in/INVENTORY_D.md:65-73`), whereas
   `_in/scripts/drive_offload.sh:58-69` copies raw directories to different remote
   paths. It cannot recheck the old archive members this way. `--immutable`
   (`:94`) refuses replacing a differing existing object, preventing repair of a
   truncated object in that namespace. The chunk1 subset path (`:56`) is invented,
   unlike the described source (`_in/MOVE_MAP.json:1113`); worktree archives are
   absent from the script. Missing sources are skipped and a failed/missing run
   still returns success (`:78-80,108-110`). Directory `stat -c %s` is inode size,
   not recursive content bytes (`:82-85`); expected D: bytes disagree with the map
   and were inferred from rounded GiB, not measured.
   **Fix:** choose either explicit verified archive objects with full member
   manifests, or a separately identified raw-tree backup. Use a new namespace to
   preserve old archives, permit retries of checksum-different incomplete objects,
   and fail nonzero on missing/copy/check failures. Measure directory members;
   keep historical size estimates out of verification. Do not create a 100-GiB
   intermediate archive on a 6-GiB-free disk. Fixed copy selects one explicit
   raw-tree item, excludes restricted/undefined subsets, and uses streaming rclone
   hashing/copying without a local archive.

8. **MAJOR — zero-writer/live/handle guards are incomplete.**
   `-AllowHighRisk` is the only extra executable gate for the main-tree rename
   (`_in/scripts/phase2_moves.ps1:162-166`); no window is confirmed or maintained,
   and no main-tree dirty-state check occurs. Only five worktree descendants are
   protected (`:70-79`), not ancestors, journal paths, KEEP-ACTIVE siblings or
   inbound junction targets. `E:/tmp` and `research-github` are KEEP-ACTIVE but
   scheduled for `2a` (`_in/INVENTORY_E.md:23,26`; `_in/MOVE_MAP.json:75-112`).
   Process scans suppress errors, match raw slash spelling, and cannot see cwd or
   future writers. File probes read only 2,000 files for deletion without a
   completeness check (`DELETE_AFTER_VERIFY.ps1:126-136`); empty returned arrays
   collapse to `$null` and `.Count` can fail under StrictMode (both callers).
   Git failures are unchecked during cleanliness/list/repair and worktree
   registration is a substring with separator mismatch (`phase2_moves.ps1:183-188,215`).
   **Fix:** hard-disable `2b` tonight; maintain an owner-established idle window
   for any later cutover. Protect normalized ancestors/descendants, all KEEP paths
   and inbound consumers; refuse inconclusive process/handle checks, never use
   file-content probes as proof. Wrap result calls with `@(...)`; compare exact
   normalized registered paths and check every Git exit. Fixed `2a` copy implements
   conservative refusals and requires `-OwnerConfirmedIdle`; that switch is an
   owner assertion, not an enforced writer lock.

9. **MAJOR — “idle archive worktree” is not an unreferenced disposable data tree.**
   `_in/inventory_E.json:130-137,165-172` records `ws-p2`/`ws-p2data` links into
   `damping-snap/_out` and `dspeed-data/_out`; another link targets `a9/_out`
   (`:145-147`). These sources are in the move/archive pool. The first two trees
   contain 18,664,357,644 bytes combined. Removing them breaks consumers even if
   Git is clean. `_in/path_refs.json:1471-1477` mentions remote `pack_dspeed` and
   `pack_damping`; they lack separate sized/dispositioned local inventory entries.
   **Fix:** build a reverse link/reference graph before selecting worktree
   removals, exclude every referenced target, and identify every named pack and
   its recoverable copies individually. Same-volume moves with verified aliases
   can preserve these references; Drive plus `git worktree remove` cannot. Git
   cleanliness does not account for ignored unique outputs. Do not assume either
   named pack is a verified D: duplicate from the remote path mentions alone.

10. **MAJOR — journal guarantees and recovery claims are overstated.**
    Phase-2 journal directory creation precedes any record (`:47-49`), junction
    creation has only an after-record (`:148-151`), and append lacks disk flush
    (`:58`). Deletion's “per-item manifest before any deletion” is actually saved
    only after all removals (`DELETE_AFTER_VERIFY.ps1:190`). Offload has no durable
    before-copy/check intent. Existing destinations fail, missing originals skip,
    and a crash after rename but before alias creation has no automatic recovery
    (`phase2_moves.ps1:171-178`); an existing link is accepted without target
    comparison (`:140-145`). Errors do not stop subsequent moves or set failure
    exit (`:258-267`). Repo `worktree repair`, PATH_MAP and MIG-06 are promised but
    not implemented. Rollback includes deleting a partial destination without a
    link inventory (`_in/REORG_PLAN_V3.md:335-340`).
    **Fix:** durably flush intent before each namespace/Git mutation and completion
    after it; persist the candidate manifest first; fail fast/nonzero. Recovery
    must distinguish source-only, renamed/no-alias, completed/validated alias and
    ambiguous collision, using journal plus exact link target and Git registration.
    Persist PATH_MAP and repair all affected worktree metadata in the eventual
    main-tree cutover. Fixed move copy journals and stops safely but deliberately
    refuses partial/completed reruns: **it is not an idempotent/resumable executor**.
    Recovery/rollback remains a design task; no removal script is supplied.

11. **MAJOR — E: reclaim is not additive or execution-complete.**
    `_in/REORG_PLAN_V3.md:214-235` treats 2.15 GiB of unhashed name/size duplicates
    and all 36.59 GiB of archive worktrees as independent reclaim. Duplicate
    examples include a 305,887,744-byte virtualenv DLL already counted with venvs
    and 688,277,914 bytes of state files inside archive worktrees
    (`_in/inventory_E.json:9639-9677`). Worktree total is whole checkouts, not just
    unique ignored output; archive-only upload is not enough to verify all deleted
    members. No delivered offload/delete workflow handles the 60 worktrees or
    `partB_bundles`/c-migrated deletion. Some listed run candidates are hard-blocked.
    **Fix:** use a disjoint exact-file set with retained-copy hashes, exclude
    referenced/evidence/protected data, and count only owner-authorized physical
    deletions. Same-volume renames free **zero** E: bytes. Remove 47.07 GiB as a
    tonight promise; see the conditional calculation below.

12. **MAJOR — capacity shortfall uses logical folder sums, not occupied space.**
    `_in/REORG_PLAN_V3.md:254-265` gives 91.35 GiB shortfall. E: top-level logical
    bytes sum to 501,313,308,539, versus occupied 504,683,868,160
    (`_in/inventory_E.json:4-7,9-1100`): a 3.139078-GiB difference. Plan's rounded
    sibling subtotal is also not a complete occupied-space model. The map's known
    phase-5 imports total 108,794,445,033 bytes (101.322723 GiB); even using those
    estimated bytes, the later 6.30-GiB baseline requires **95.022723 GiB** reclaim
    before reserve. Two KEEP label variants add about 0.020 GiB; `ray_results` and
    named additional packs remain unbounded. D: deleting 153 GiB does not increase
    E: capacity, and its active 101-GiB component cannot simply disappear.
    **Fix:** use fresh per-volume occupied/free space and measured allocation
    sizes, unknown-size blockers, non-overlapping reclaim and explicit working
    reserve. Budget ~95.04 GiB plus reserve for these known imports, not 91.35;
    do not promise a complete physical consolidation until omissions are measured.

13. **MAJOR — one-folder completion is not achieved tonight or by the proposed
    deletion shortcut.** Main repo stays outside the umbrella tonight; real
    `E:/rd-wt` and active data on D: remain exceptions (`_in/REORG_PLAN_V3.md:78-104`).
    Permanent top-level aliases also remain visible in Explorer. The design gives
    an umbrella, not literally one top-level directory. Drive-only evidence does
    not meet the “frozen evidence paths keep resolving” rule
    (`_in/AGENTS.md:27,70-81`). **Fix:** state tonight's milestone as *partial
    namespace consolidation*. Owner should distinguish one navigational entry
    point from one physical root. Keep active/frozen data local; a future physical
    cutover needs real quiescence and adequate capacity. An umbrella alias to the
    unmoved repo could provide navigation if desired, but is not a physical move.

14. **MINOR — portability and delivered-file claims need correction.**
    All three originals hard-code `reorg3/architect` defaults, not the supplied
    `_in/` layout (`phase2_moves.ps1:27-28`, deleter `:24`, offloader `:27`).
    `tools/build_move_map.py` and the described rollback/helper scripts are not in
    the provided package (`_in/REORG_PLAN_V3.md:338-362`). `$PSScriptRoot` is not
    used in original parameter defaults, so that earlier pitfall is absent;
    fixed copy resolves it **after** parameter binding. Original deleter does not
    use `-Confirm`, so no per-file prompting bug is currently demonstrated.
    Original and fixed Bash both use LF (static byte check), so the earlier CRLF
    pitfall is absent; fixed Bash passes `bash -n`. **Fix:** resolve defaults
    relative to packaged scripts after `param`,
    ship every required helper, keep `.sh` LF, and use a single owner authorization
    plus exact literal removals in any eventual owner-only deletion tool. Do not
    add per-file confirmation prompts as a substitute for verification.

15. **MINOR — snapshot measurements and precision are inconsistent.**
    E: snapshot free is 6.605736 GiB, D: scout says 6.68 GiB, later plan says
    6.30 GiB; these are different observation times, not contradictory current
    guarantees. D: `size_bytes` values are conversions of rounded GiB estimates.
    The scout 1.44-GiB quick-win headline lacks a sized CPU superseded subset:
    its explicitly quantified pieces sum to about 0.244 GiB
    (`_in/INVENTORY_D.md:36,42,44,46-47,117`). C/G inventory also labels binary
    quantities GB/MB. **Fix:** label GiB/MiB, retain exact measured bytes where
    available and identify estimates; remeasure only at owner execution time.

## Recomputed space and feasible ordering

All following sums are **GiB**, and reclaim is conditional, not authorization.
`verify_review.py` reproduces the principal exact metadata totals.

- Same-volume `2a` renames/junctions: approximately zero net byte change; small
  filesystem/journal overhead only. They can fit with 6.30 GiB free, provided the
  individual source is truly idle and the reference/link preflight is satisfied.
  The conservative fixed script will reject KEEP/linked sources in the original
  map; it does not approve running that map wholesale.
- Supplied tmp candidate subset: 404,401,113 bytes = 0.376628 GiB, including
  2,178,100 bytes of verification evidence that must stay. Thirteen scout entries:
  96,560,755 bytes = 0.089929 GiB, including active `reorg3` (108,366 snapshot bytes).
  Run candidates total 134,307,083 bytes = 0.125083 GiB, including protected chunk5
  failed attempts (1,681,439 bytes); none are blanket-approved for deletion.
- An **optimistic local-only ceiling**, assuming every remaining scratch/scout
  item and the estimated 0.916-GiB venvs prove disposable, is
  `6.30 + 0.916 + (404401113 - 2178100 + 96560755 - 108366)/2^30`
  = **7.680428 GiB**. Actual permission/space can be less. Guaranteed reclaim from
  this review is zero. Unhashed duplicates and failed-run evidence add no approved
  space to this ceiling.
- All 60 archive checkout bytes: 39,288,490,760 = 36.590258 GiB. Adding those and
  `partB_bundles` (563,055,757 = 0.524387 GiB) to the optimistic ceiling gives
  **44.795073 GiB**, a counterfactual upper bound before excluding live references
  and overlapping/retained records, **not tonight availability**. The original
  47.07-GiB estimate is unsupported.
- Historical D: verified subset: `6.391 + 6.438 + 4.553 + 13.5 = 30.882 GiB`.
  Only after a fresh archive-member/checksum/source check and owner action would
  D: free rise from 36.33 to about **67.212 GiB**. E: remains unchanged. Another
  ~21 GiB awaits re-upload; active packs/shards/payload remain local.
- CPU payload plus all four label sidecars plus the listed small C: artifacts:
  approximately `1.118 + 0.040 + 0.200723 = 1.358723 GiB`. This is physically
  within the current baseline; it does not require 102 GiB free as the blanket
  phase-5 gate claims. Any later copy/cutover still needs individual stability,
  hashes, reference preservation and working reserve. Unknown `ray_results` is
  excluded and remains ACL-blocked. The bulk ~101-GiB transfer cannot fit now.
- Offloading all 108.46 GiB of the proposed gated E: evidence could nominally
  supply sufficient bytes, but there are no execution waivers, exact subset
  manifests, working-path replacements or upload-rate evidence here. No claim
  that this can finish tonight is justified; restricted data is outside this review.

## Owner-facing tonight summary

1. **Preserve the live repo, five named live worktrees, all KEEP-ACTIVE trees,
   `reorg3`, verification records and local packs.** Stop the blanket cleanup path.
   Snapshot Git registrations and the no-follow link/reference graph in the owner
   environment before considering any migration.
2. **Organize only proven-idle E: siblings and approved idle worktrees** under the
   umbrella, keeping exact compatibility aliases. Eligible examples are release,
   backups, migration logs and unused bundle/launcher folders after fresh checks.
   This improves layout but frees no space; do not move live scratch or rename
   the main repo tonight. Plan per-item batches, not the unfiltered original map.
3. **Back up an explicitly selected stable source to Drive** using a working
   rclone environment; verify all bytes/members/checksums. Keep local originals
   and recovery receipts. Repair the five truncated archives in a documented
   namespace, without confusing raw trees with archived objects. Upload duration
   is unknown. Frozen/restricted datasets need a separate owner decision.
4. **Owner deletion is a later, separate step** after the receipt and eligibility
   fixes, revalidation and reference preservation. Small proven cache cleanup may
   recover about 1.38 GiB on E:, while the historically verified ~30.88 GiB frees
   D: only. Do not spend hypothetical reclaim before owner deletion actually
   completes. Small ~1.36-GiB imports can be planned individually; large packs
   await sufficient real E: headroom or stay on D: behind the existing junctions.

Result tonight: a tidier **partial** umbrella and possibly verified backups,
without disrupting training/evaluation. Full physical one-folder consolidation
requires later quiescence, measured capacity and a path-preserving evidence design.

## Fixed copies and verification scope

- `fixed/phase2_moves.ps1`: conservative `2a` only, packaged defaults, no dry-run
  writes, normalized protections, no-follow metadata checks, fail-closed handle/
  Git checks, durable before/after mutation journal and fail-fast errors. Refuses
  links/KEEP sources and existing destinations. No automatic resume/rollback;
  writer shutdown remains an owner responsibility.
- `fixed/drive_offload.sh` and `fixed/backup_receipt.py`: one explicit ordinary
  source, no restricted/invented subsets, true side-effect-free dry-run, durable
  upload/check intent, separate raw-tree namespace, one-way size/hash check plus
  complete MD5 member comparison and repeat source hashes. No data-loading Python.
  Backup retries are possible; remote-object stability and crash-safe source
  snapshots are not established. Receipts never authorize deletion.
- `fixed/DELETE_AFTER_VERIFY.ps1`: report-only, `-Execute` hard refusal and zero
  eligible deletions. This is an intentional fail-closed mitigation, **not** a
  claim that the underlying deletion policy is mechanically repaired.
- Static verification: JSON parse/map cardinality/destination uniqueness, exact
  arithmetic and evidence assertions, Python AST parsing, PowerShell parser for
  originals and fixed copies, Bash `-n` for the LF copy, line-ending byte checks
  and `git diff --check`. No original/fixed executor or helper runtime was run;
  no cloud checks, locks, data hashes or recovery behavior were exercised.

The input package remains unchanged and untracked; the commit contains review
artifacts only. The review is complete; **plan execution remains blocked by the
findings above**, and syntax checks do not change the FAIL verdict.
