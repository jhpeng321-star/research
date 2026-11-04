# MAIN10:12 remote slim migration complete and independently verified

Completed at **2026-11-04 10:45:22 +08** (unix1793760322.22):
- **447 rejected branches /894 marker+payload files /14 SHA-verified batches**.
- **936,070,977bytes** of full originals retained on gpuA at
  `/root/cpuq_archive_20261103/slim_migration1`, using original home-relative paths.
- **4,209,419bytes** of sealed local SLIM summaries retained.
- **931,861,558bytes net** recovered (archived payload/markers minus local SLIM).
- Zero planning errors, failed/partial moves or new simulation charges.

The owner2600683 exited normally. **Do not redispatch the completed migration.**
Both original rolling movers remained running; the GPU's lowest verified free
space across all batches was18,795,958,272bytes, above its8GiB floor.

## Independent completion verification

At unix1793760423.44, a separate read-only verification passed:
1. All staged source hashes match the registered immutable authority.
2. All14manifest digests and remote per-file SHA proofs match, with894unique files.
3. Every moved branch remains eligible under the reviewed, sealed rejected-only
   selection. No admitted/reference/solo/first-seed chosen branch was selected.
4. Full source marker/payload pairs are absent only for the447migrated branches;
   their local SLIM seals, original result/payload identities and window totals pass.
5. Gross bytes minus actual local SLIM sizes exactly equals the completion net.

Certificate: `SLIM_MIGRATION_MAIN1012_V4_VERIFIED_COMPLETE.json`.
The original code/transport tests were23LinuxPASS and6localPASS; qualification,
registration and dispatch remain in the preceding immutable receipts.

## Remaining capacity and live run

Verification cpuQ free was**7,756,210,176bytes (7.22GiB)**, leaving
**2,981,208,064bytes (2.78GiB) short of10GiB**. Concurrent writers change free
space; the reported recovered bytes come from certificates, not a disk delta.
The9GiB backfill guard and MAIN's collector STOP are retained. The earlier
legacy-tail archive separately completed476,840,470bytes/240files/4batches.

The owner10:40 release was independently observed in canonical registration.
The delegated operator's gate_coop8 is **PASS** for src7/a7ae6d9a and runtime
06e88ecd (six offline checks true, no reasons). Resume r2 starts B/R at u4 from
their durable u3 checkpoints. At unix1793760464.83: A10/B3/R3 completed; B u4
SUMO episodes run; supervisor2636865 and B/R runners2636963/2636968 are live;
new statistical learning monitor4 PID2635161 is live. No competing dispatch here.
B-u6 lesson/isolated-synergy evidence is not yet available.
