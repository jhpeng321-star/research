# Registered learning-mechanics pause — R u3

At Nov4 08:51:38 +08, R u3 finished and the current-runtime-bound monitor3
detected `pass1_effective=false`. At08:52 the registered shared B/R mechanics
pause was written. Both replacement workers exited through `HealthPause` at
08:52:07/08. A continues. Durable checkpoints are **A6/B3/R3** at the latest
probe. No u4 update, competing restart, health release or criterion change here.

## Diagnosis from immutable TRAIN receipts

R used102trainable bank pairs from three cooperative examples, completed five
full passes with zero backtracks and all step scales1.0. First-pass full-bank
AWR loss rose **19.347578048706055 ->19.349313735961914**, so its reported
drop−0.001735687255859375 and ineffective flag are internally consistent.
Each later pass decreased the loss, ending at19.3341064453125. Overall L1 action
error fell0.45307767184108105->0.4529062353540212. KL0.00015507939422283978
is within0.045. No nonfinite value or missing coverage is reported.

The code accepts finite full-pass attempts when their KL fits, then measures
first-pass AWR progress after acceptance. It does not backtrack on a first-pass
AWR increase. The receipt guard therefore applied the registered rule correctly.
Adam/minibatch optimization need not decrease the full-bank objective at every
individual pass, but the precise cause of this increase is not established from
these records. No parser/wiring/physics/accounting fault has been demonstrated.

R3 composite safety passes on24paired TRAIN worlds: collisions/teleports zero;
critical-total delta−5.875 CI[−22.4375,+10.3542]; hard-brake delta+0.416667
CI[−1.52083,+2.16667]. B3 also passes safety and its five-pass learning mechanics.
This is a learning-mechanics pause, not a safety, intent-trend or cooperation
result. Bank intent loss0.487287->0.488759 remains report-only.

## Disposition

Preserve the pause, checkpoint, charges and all receipts. MAIN was notified in
`_sol/COOPMWM_LEARNING_PAUSE_R3_0900_MAIN.md`. Without a demonstrated engineering
fault, the registered procedure requires reporting to MAIN before resuming.
No unapproved optimizer change, threshold adjustment, first-pass redefinition or
optional learner activation was made. B-u6 lesson/synergy remains pending.

Evidence:
- `R3_MECHANICS_DIAGNOSIS_1793753887293557800.json`
- `V2V_LEARNING_WATCH_20261104_V3_STATUS_1793753828470235700.json`
- `RESUME_LIVE_0815_1793753815011438500.json`

Storage movers and monitor3 remain live. Latest free space7022444544bytes
(6.54GiB), below the9GiB backfill-resume threshold. New closed diagnostic archive
is complete with289878263bytes/255files/3certified batches retained remotely.
