# WD-01-C2L1: Amendment 1 (disclosed, made BEFORE any look)

**Date:** 2026-09-29. Written by the world model session.

**Authority:** owner decision (direct), relayed by main discussion: "C2L1 =
option (B)".

**Amends:** REGISTRATION_NATIVE_PARTA_CHUNK2.md. Its original sha256
(LF-normalised) is
28167500e2419e9bcbc2570e958b138b1738a7a4a6fa7367a31441516070e5da and is
kept. That file is not edited.

**Status at amendment time:**

- C2L1 has NOT been run.
- No chunk-2 CALIBRATION world has been opened by any Part A code.
- The amendment follows the owner's approval of the structural lane-change
  and merge redesign (DESIGN_PARTA_STRUCTURAL.md). It does not follow any
  result on these worlds.

## Changes

1. **Two models, one pass.** The single evaluation pass scores two models
   side by side on the same 7 chunk-2 CALIBRATION worlds, the same windows
   and the same horizons:
   - **v1:** native Part A v1, `idm_merge`, frozen p3o parameters and spread
     factors. Hashes 458890d9… and c7ffa754…, exactly as registered.
   - **v2:** native Part A v2, engine kind `idm_lc2013`. This is the
     structural redesign: LC2013-like speed-gain discretionary lane changes
     and strategic late merging with speed matching. The end-force and
     merge-probability hacks are removed. v2 is fitted only by TRAIN-internal
     k-fold on chunk-1-era native TRAIN roots and/or chunk-2 TRAIN worlds,
     with no validation look.
2. **v2 is the candidate under test.**
   - The verdict rule of §3 is unchanged (gate v2 with amendments 1 and 2;
     PASS if and only if (a)–(d) all pass at H = 30 s). It is applied to v2.
   - "Native Part A gate v2 met" may be claimed only if **v2** passes.
   - v1 is scored with the same rule and reported alongside, but only as the
     comparison of record. A v1 PASS with a v2 FAIL does NOT meet the gate.
     Choosing the better model after seeing results is not allowed.
3. **Freeze before the look.** Before the pass, v2's frozen artefacts are
   recorded in `C2L1_V2_FREEZE.json` with their sha256 (LF-normalised for
   text), the freeze UTC time and the code commit:
   - engine code;
   - parameter file;
   - spread/calibration factors;
   - fold assignment.

   The sha256 of `C2L1_V2_FREEZE.json` is sent to main discussion before the
   run. The evaluation refuses to start unless the registration, this
   amendment and `C2L1_V2_FREEZE.json` all match their recorded hashes.
4. **Unchanged:**
   - worlds, windows and horizons;
   - access control (the loader refuses the CAL worlds without the
     registration hash, now also checking this amendment's hash);
   - one pass, and raw rows saved before aggregation;
   - the crash and restart rule;
   - FAIL means TRAIN-only diagnosis with no second look;
   - the look is logged as chunk-2 look #1 in WD-01-DEV-2.
5. **After a v2 PASS:**
   - the DEV-1 holdout second use becomes eligible, but only after main
     discussion relays the owner's confirmation, under its own frozen
     registration;
   - the result is reported to main discussion first.
