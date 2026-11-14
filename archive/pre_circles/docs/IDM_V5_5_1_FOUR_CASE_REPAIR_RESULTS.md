# IDM v5.5.1 four-case adaptive hold/release repair results

Status: **completed development-only mechanism gate; failed.** No expanded
screen, predictor dataset, predictor fit, validation seed, or final-paper seed
was opened.

## Execution and integrity

The frozen v5.5.1 contract (`d28c91c2...`) ran eight paired SUMO branches on
the four previously inspected 2700/400, 30%-CAV development cases. Each pair
used the exact frozen analytical IDM P30 damping arm, absolute decision time
880 s, maximum 20 s policy window, compact full-occupied-lane wall, and common
1080 s endpoint. All cases had full active-lane coverage and non-negligible
authority (26--40 controlled CAV-seconds). The complete repository test suite
passed **934/934** before the run.

The adaptive law was a risk-gated immediate-follower taper: a 20 m / 6 s rear
risk gate plus same-follower compression trend, existing -3 m/s2 braking
guard, existing 3-follower/100 m hard layer, and a five-second monotone taper
to exact damping. It is an engineering repair, not a fitted predictor feature.

## Per-seed result

| Seed | Lifecycle / dose | Safety & service | Physical result | Gate status |
|---:|---|---|---|---|
| 4401 | Planned 14 s hold + 5 s taper; no soft trigger; 40 CAV-s | Safety fails: +3 hard-brake steps/onsets; service passes | No primary benefit | Fail |
| 4402 | Soft release; 12 s hold + 5 s taper; 36 CAV-s | Safety fails: +4 hard-brake steps/onsets; service passes | Prevents sustained congestion: 91 s reduction | Fail |
| 4404 | Soft release began but a controlled CAV reached hard braking during taper; 11 s hold + 1 s taper; 26 CAV-s | Outcome safety/service guardrail passes, but treatment lifecycle has a hard abort and is invalid | No primary benefit | Fail |
| 4405 | Planned 14 s hold + 5 s taper; no soft trigger; 40 CAV-s | Safe and service-admissible; hard braking improves by one step/onset | Materially shortens sustained congestion by 24 s | Pass individually |

The v5.5.1 all-four gate fails because the all-integrity and all-safety gates
are false. Service is noninferior in all four cases, both 4402 and 4405 retain
their physical benefit, and no case has negligible authority. Those partial
positives cannot offset the two continuing safety failures or the 4404 hard
abort.

## Mechanism diagnosis

The repair did not solve the causal limitation identified in v5.4.

- **4401:** the immediate follower never entered the risk-gated soft-release
  state (minimum sampled rear TTC 14.0 s), yet the treatment retained the same
  three excess hard-braking steps. The local follower criterion does not see
  this downstream response pathway.
- **4402:** the soft criterion was reached and the policy completed its
  taper, but four excess hard-braking steps/onsets remained despite the 91 s
  congestion prevention. The discharge/safety trade-off persists after the
  shortened hold.
- **4404:** the criterion caught a 5.36 s rear-TTC / rising-closing-speed
  state, but the controlled wall member hard-braked during taper. This is an
  unsafe release/authority interaction, not an admissible completed treatment.
- **4405:** remains the only safe beneficial case; it never entered the soft
  state (minimum sampled rear TTC 12.6 s).

Thus the v5.5.1 evidence supports neither a safe general moving-wall treatment
nor a benefit label suitable for predictor training. The failure is not a
replay artifact: predecision/action-path identity and exact damping return pass
whenever the lifecycle completes. It is a treatment-concept limitation under
the current compact geometry and late 2700 veh/h overload cell.

## Frozen decision

The protocol requires a stop here. Do not add the fresh 4407--4430 seeds or
fit a feasibility/safety/benefit predictor. The recommended paper route is the
**decentralized-authority-boundary / conditional escalation** framing: report
the safe 4405 benefit and the unsafe 4402 prevention as mechanism evidence,
but retain the safety boundary and do not claim a verified safe escalation
policy. Any v5.5.2 geometry or state-responsive-duration proposal is review
only and requires a new user-approved scientific treatment definition.

## Artifacts

- runner: `experiments/idm/run_idm_v5_5_1_four_case_repair.py`;
- contract: `experiments/idm/wall_v5_5_1_contract.py`;
- machine records: `runs/probe/idm_v5_5_1_four_case_repair_p30_s4401_4402_4404_4405/`.
