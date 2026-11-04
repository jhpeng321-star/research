# Deployed practice source adopted locally — Nov4 14:13 UTC+8

Merged the reviewed/deployed `c4/v2v-resume` commit `b918fcdb` into this
worktree, preserving existing archive work and every unrelated dirty file.
This is local source adoption; no remote deployment, gate, training, monitor,
physics or inference job was dispatched.

`PRACTICE_SOURCE_ADOPTION_1400_1793772784828720000.json` independently
verifies all12changed Python files: merged index bytes == b918fcdb blobs ==
deployed src8 SHA256. The actual immutable Linux receipt is PASS313/313,
zero skips/zero SUMO,58.260s; receipt SHA
27c73a30b38eac73a1db340f4458c9f43594c906d76525e30d0f7cb2b48aaad4.
The actual gate_coop11 report is PASS, no reasons, commitb918fcdb,
runtime1d67a685. Existing qualification was read, not rerun.

Local additional test attempts were NOT a PASS: system Python3.14 lacks
torch and failed collection; the existing rl_driving Python invocation
produced no output before the120s command timeout. No environment was
created/modified. Do not substitute these for the verified Linux receipt
or launch another qualification job simply because Windows timed out.
Read-only source-proof SSH attempts hit connection reset/closed errors;
the third bounded connection-level attempt succeeded. No job restarted.

Canonical registration now records practice switch DONE at u10 for both
arms: r3supervisor3104282/B3104306/R3104318, A2049830adopted. B10first
practice update completed:6practice/18normal,6matched A references, composite
safetyPASS,5full imitation passes/KL~0.0440. These are operator-published
numbers pending the next independent full progress read. R matches B's
realised6practice episodes. Owner-approved effective-command imitation and
CPU-S/variety collection remain separately delegated; no activation here.

Earlier independent R9 receipt:5passes/KL0.0447897/noimitationfailure,
action MAE0.454724->0.453737,3cooperativeexamples, pass1loss CI
[-0.009141,0.001705] (not significantly harmful). Composite safetyPASS,
24worlds/zero collisions+teleports/criticalR-A+8.0CI[-24.229,40.0].
`PUBLIC_PROGRESS_1793771726201903600.json` gives charged13,928,720/cap
17,471,840, no negative phases; archive extension3,647,233,349bytes certified.
Disk remains the active resource concern, not a reason to relax any floor.
