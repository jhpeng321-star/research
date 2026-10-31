# COOPMEAS2 — causal cooperation

S-COOPMEAS2-1. D is saved full-population road + entrance delay (veh-s); positive is beneficial.
C_G = D_GROUP − sum D_ONLY:i; positive C_G is beneficial interaction.
Frozen native-state groups; same-state, identity-keyed exogenous requests and saved RNG.
Paired 95% stratified world-bootstrap CIs; 5000 draws, seed20261028.
Tiny-gain groups are retained. Missing groups are reported, never replaced by zero.

Coverage: {"actual_sim_s": 150000, "charged_sim_s": 151200, "controller_worlds": 56, "entrant_controlled_ticks": 302690, "missing_group": 0, "native_repeat_passed": 56, "uninformative_rows": 7, "worlds": 28}

## All-world interaction estimates
- 60s ours: -2.981614453451974 [-6.131293008157185, -0.19636651022093696] veh-s; n=28.
- 60s mappo: -1.6807728665215627 [-9.028809294956071, 4.846119809576443] veh-s; n=28.
- 60s ours_minus_mappo: -1.3008415869304115 [-7.662013443878719, 5.278381184169221] veh-s; n=28.
- 120s ours: -35.99608337879181 [-103.40326075213295, 32.08668987197533] veh-s; n=28.
- 120s mappo: -5.715474316052028 [-120.23796088014329, 116.04721090282709] veh-s; n=28.
- 120s ours_minus_mappo: -30.280609062739778 [-110.0203647664615, 46.898354568651726] veh-s; n=28.
- 300s ours: 137.99150262560133 [-438.54408461323834, 727.5112956583494] veh-s; n=28.
- 300s mappo: -188.99825746672505 [-1028.4482420078348, 637.9928792404269] veh-s; n=28.
- 300s ours_minus_mappo: 326.9897600923265 [-395.11446186474416, 1035.268527429444] veh-s; n=28.

## Safety and interpretation
All native/GROUP/ONLY/fleet safety totals are in COOPMEAS2_branches.csv (existing hard-braking exposure, TTC<3s, collisions, teleports, dropped steps).
Coverage, identity lists, group sizes, null flags and gain magnitudes: COOPMEAS2_worlds.csv.
All clocks/cells and high-share estimates: COOPMEAS2.csv. No behavior counts establish causality.
Cycle1 and MAPPO only. Future cycle4 candidates must all be reported on these same worlds/groups.
A positive fleet gain does not establish group cooperation; C_G and its uncertainty are the interaction evidence.
- GROUP hard_braking_vs ours−MAPPO: -1.25 [-2.3571428571428568, -0.1250000000000001]; lower is safer.
- GROUP ttc_below_3_vs ours−MAPPO: 15.642857142857142 [-3.0897321428571427, 33.69642857142857]; lower is safer.
- GROUP collisions ours−MAPPO: 0.0 [0.0, 0.0]; lower is safer.
- GROUP teleports ours−MAPPO: 0.0 [0.0, 0.0]; lower is safer.
- GROUP dropped_vehicle_steps ours−MAPPO: 0.0 [0.0, 0.0]; lower is safer.
- ALL hard_braking_vs ours−MAPPO: 2.4107142857142856 [0.9285714285714285, 3.9821428571428568]; lower is safer.
- ALL ttc_below_3_vs ours−MAPPO: 2.1607142857142856 [-15.947767857142857, 20.911607142857132]; lower is safer.
- ALL collisions ours−MAPPO: 0.0 [0.0, 0.0]; lower is safer.
- ALL teleports ours−MAPPO: 0.0 [0.0, 0.0]; lower is safer.
- ALL dropped_vehicle_steps ours−MAPPO: 0.0 [0.0, 0.0]; lower is safer.

## Final interpretation and design context

The primary 300 s interaction evidence is **inconclusive**: ours C_G +138
[−439, +728], MAPPO −189 [−1028, +638], and paired ours−MAPPO +327
[−395, +1035] veh-s (95% CIs). Neither beneficial cooperation by ours nor an
interaction advantage over MAPPO is established. The achieved paired CI
half-width is 715.2 veh-s, versus the preregistered approximate 9.84 veh-s
projection at 300 s. The E4-based planning approximation substantially
understated observed uncertainty. The fixed sample was not enlarged.

Fleet-arm saved delay at 300 s is +736 [315, 1154] veh-s for ours and +627
[237, 1054] for MAPPO; paired ours−MAPPO +110 [−306, 508]. Both improve on
native in these native-state counterfactuals, while fleet superiority remains
inconclusive. These fleet results do not establish cooperation or whole-episode
efficacy. The 60/120 s clocks, 24-world high-share cohort, and six cell results
remain secondary; all are exported without outcome-based selection.

The fixed allocation is six worlds in each 20/40% CAV × 2400/2700 veh/h cell
plus two in each 10% cell. Fresh seeds981200–981227; native t0=300 s.
Groups contain the nearest up-to-four observable non-junction upstream CAVs
(s<0), ordered geometrically then by identity, before intervention. Actual
groups have three or four members; all28 worlds have a group. B, B_repeat,
each ONLY:i, GROUP and entrant-controlling ALL branch from the same state.
MAPPO aecd7d70 and ours cycle1 64f70af4 use the same worlds/groups.

For gain magnitude max(|D_GROUP|, max_i |D_ONLY:i|), the min/median/max in
veh-s at60/120/300 s are:
- Ours: 0/5.25/128.70; 0/163.58/482.22; 0/959.54/3919.34.
- MAPPO: 0.006/17.34/196.05; 0.006/177.01/753.05; 0.006/1624.73/3365.76.
Numerically-null groups use the frozen native-repeat resolution (1e-6 veh-s
here): ours3/2/2 and MAPPO0/0/0 at60/120/300 s. All remain in the estimates;
the seven null controller-clock rows are not seven missing worlds.

All scored branches have zero collisions, teleports and dropped vehicle steps.
Safety is mixed: at300 s the fleet arm has +2.41 [0.93, 3.98] veh-s more
hard-braking exposure for ours than MAPPO; GROUP has −1.25 [−2.36, −0.13].
TTC<3 s differences are uncertain for both fleet and GROUP. These safety
comparisons are secondary, without multiplicity-adjusted confirmatory claims.

Production charged151200/151200 sim-s, actual150000; paired mechanics smoke
charged/actual1200/1500 cap. Combined charged152400/152700, actual151200.
Reservations are retained with no refund or retry. Both E3 bulk shards
completed before smoke;56/56 native repeats and paired initial-state/group/CRN
checks passed. Independent exported-CSV checks reconstruct168 controller-world-
clock contrasts and reconcile1332 branch-clock rows. Worktree delivery receipt:
`evidence_coopmeas2/DELIVERY_VERIFIED.json`. Cycle4 candidates remain pending
prospective binding and separate budget; no candidate has been selected here.

This interpretation was appended after the fixed complete-panel analysis;
the registered estimates, bootstrap draws and CSVs are unchanged.
