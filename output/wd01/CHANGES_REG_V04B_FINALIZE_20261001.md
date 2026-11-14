# CHANGES: Part B v0.4b registration FINALIZED (2026-10-01)

`REGISTRATION_PARTB_V04B.md` finalizes the draft
(`REGISTRATION_PARTB_V04B_DRAFT.md`, zcode/pbv04b 5092c4c, LF-sha256
`9f7876485367b63812fe0e2466ef6155769956ff1f1b6c0a3726f405f27ce171`)
under the owner's finalize instruction. The draft file is untouched.
Four fold-ins; everything else is draft text unchanged (verified: body
diff vs draft is exactly header/status, §11 annotations, §12, §13).

1. **§12 — owner action-descriptor requirement, VERBATIM**
   (`ACTION_DESCRIPTOR_REQUIREMENT_20261001.md`, zcode/wd01 27a724c,
   LF-sha256 `ac6e6f046fc3dce3e0f6d8d59da45ef23080df0ba009c5619aaaf14206848f3b`;
   only the heading level demoted, disclosed in §12): the action input is
   the PLANNED SCHEDULE (19 × 30 s segments per CAV/group + wall
   hold/speed/cap); "damp 60 s then native" ≠ "damp 570 s"; results
   reported stratified by duration/onset (≤ 60 s, 60–240 s, whole
   window) × tau.
2. **§12.2 — a-priori representation counts** (from zcode/wmn1p8s
   `output/wd01/wm_n1/evalh/reports/ACTION_REPRESENTATION.md`, LF-sha256
   `4754c65ac1aaecf311194ed45f93ce4618fb00822ffbbdf21bf91422fbf78fdf`):
   4758/4952 = 96.1 % whole-window TRAIN pairs, 5 ≤ 60 s, 189 at
   60–240 s, 2 differing only in the first 60 s; short commitments come
   only from the J4 wall arms. Registered as the whole-window-dominance
   disclosure.
3. **§13 — J-PULSE** (`REGISTRATION_JPULSE.md`, zcode/wd01 36766cd,
   FROZEN, LF-sha256
   `a4d1c5f9beb40dc117c2f53ef64ab176bab434f3dcab8481ad9cd279e3621ea7`):
   PULSE pairs (44 worlds × damping/PI × 60/120/240 s then native vs
   `__native`) are FUTURE TRAIN data; v0.4b is fitted first on J4
   (+ chosen auxiliary) data, then a DISCLOSED retrain including PULSE
   when it returns, with the registered PULSE out-of-fold diagnostic at
   τ ∈ {120, 300, 570} (world-grouped 5-fold CV; diagnostic only, not a
   selection input).
4. **§11 — open questions annotated with owner rulings:** Q1 (aux) and
   Q5 (sequencing/wall-v2.2 dependency) RULED (finalize instruction;
   wall-v2.2 ack granted and frozen, LF `e7fd3536…`); Q2 (gating
   contrast), Q3 (A1 with 5 CAL worlds), Q4 (safety heads) remain
   OWNER QUESTIONS.

No fitting, no data reads beyond registrations/plans; zero SUMO.
