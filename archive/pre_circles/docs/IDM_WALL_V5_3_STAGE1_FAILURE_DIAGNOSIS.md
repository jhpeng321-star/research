# IDM wall v5.3 Stage-1 failure diagnosis

Status: **completed read-only diagnosis on existing artifacts; no SUMO branch,
threshold, treatment parameter, or outcome label changed.**

Date: 2026-08-31

## Question

The frozen v5.3 Stage-1 replay failed because 4312/660 contained one more
absolute hard-brake vehicle-step than its matched v5.2 branch and because the
prescribed seven-case cohort did not retain two of three prior
safe-beneficial starts. This diagnosis determines whether those failures show
a direct v5.3 mechanism defect or a protocol/cohort problem. It does not
retroactively pass v5.3.

## 4312/660 event-level result

The v5.2 treatment contains 27 hard-brake vehicle-steps and v5.3 contains 28.
The branches share 22 event starts. Relative to v5.2, v5.3 removes two starts
and adds three, for a net increase of one:

- removed: `mainline.604` at step 903 and `mainline.610` at step 912;
- added: `mainline.593` at step 883, `mainline.594` at step 885, and
  `mainline.612` at step 913.

All five differing events occur on mainline edge `1191554636` during the
post-treatment `damping` phase. The v5.3 compactness-loss taper completed near
the original treatment window; the added starts occur more than 200
simulation seconds after the decision. None is a wall member, an immediate
wall follower, or under an active wall command. The event change is therefore
a delayed metastable trajectory divergence, not evidence that the taper
directly forced a hard brake.

The v5.3 branch remains inside the frozen absolute safety budget relative to
its damping control: hard-brake excess is -6 vehicle-steps, low-TTC excess is
-1, and there are no excess collisions or teleports. Its Gate-B label changes
from safe-beneficial under v5.2 to safe-nonbeneficial under v5.3 because delay
benefit falls from 414.49 s to 178.00 s and discharge benefit falls from +2 to
-1. It also remains a realized-dose failure: the compactness-loss path delivers
18/20 override steps, 90% lane coverage, 61.1% HOLDING, and 61.1% within-span
geometry.

Conclusion: the one-step non-worsening failure is real under exact replay, but
it is not a direct terminal-mechanics safety failure and does not cross the
frozen absolute safety constraint. The case should remain a negative Gate-F
dose example and a nonbeneficial Gate-B outcome; it must not be deleted or
relabelled.

## Safe-beneficial cohort audit

The three dose-qualified, safe, service-noninferior and meaningfully beneficial
v5.2 starts are exactly:

- 4302/700;
- 4307/660;
- 4311/660.

The frozen seven-case v5.3 list instead contains 4302/620, 4307/680 and
4311/700. The latter two were beneficial but unsafe under v5.2; 4302/620 was
unsafe and nonbeneficial. Therefore the Stage-1 requirement to retain two of
the three prior safe-beneficial cases was not evaluable from the declared
seven starts. No replacement cases may be inserted retrospectively.

## Scientific decision

The original Stage-1 result remains a recorded no-go under its own frozen
protocol. This diagnosis does not weaken that result. It supports a separately
versioned prospective support protocol under the already adopted two-stage
estimand:

- keep the v5.3 treatment implementation and hashes unchanged;
- use Gate F to refuse predecision-infeasible and predicted dose-unreliable
  cases;
- retain every attempted dose failure as a failure;
- use full-horizon safety and benefit as Gate-B labels;
- require the eventual gate-selected actions, not all raw attempts, to meet
  the 90% prospective safety target;
- freeze new seeds and criteria before any new outcome is opened.

The next run is therefore not a retroactive v5.3 pass or a taper efficacy
test. It is a prospective test of whether the unchanged v5.3 treatment
provides enough physical and outcome diversity to support the selective-gate
estimand.

## Evidence

- `runs/probe/idm_wall_v5_3_stage1_replay/cases/seed4312_step660.json`
- `runs/probe/idm_wall_v5_3_stage1_replay/summary.json`
- `runs/eval/idm_wall_v5_2_prospective_p30_s4301_4312/seeds/seed4312.json`
- `runs/eval/idm_wall_v5_2_prospective_p30_s4301_4312/summary.json`

