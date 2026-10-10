# IDM v5.6 treatment tournament results

Status: **complete development-only tournament; no winner**

Date: 2026-09-01

## Decision

All six routes were independently audited before selection.  Only Candidate A
cleared its entry gate and it completed the required four-case matched screen.
Candidate A failed that screen.  Candidates B--F were stopped individually by
their own treatment-blind feasibility or focused-safety no-headroom gate; none
of those stops terminated the independent A screen.

There is no treatment winner, no fresh-seed protocol, no fresh seed opened,
and no predictor-scale generation or predictor/RL training authorization.

## Audit ledger

| Candidate | Audit conclusion | Screen status | Tournament consequence |
| --- | --- | --- | --- |
| A: low authority | 1.0 m/s2 cap materially active; 0.3 m/s2 is a real treatment change. | Four matched branches completed. | Failed integrity/authority and 4405-benefit gates. |
| B: upstream support | Valid non-reserve support cohort cannot be established from treatment-blind geometry under the current band/road length. | Not eligible. | Stopped before outcomes. |
| C: discharge release | Cannot trigger before the known unsafe paths in 4401/4402 diverge. | Not eligible. | Focused-safety no-headroom. |
| D: corridor duration | Can affect 4404 but cannot affect unsafe 4401/4402 in time. | Not eligible. | Focused-safety no-headroom. |
| E: geometry | No separately projected alternate geometry is justified; `static_all` is not a geometry-only repair. | Not eligible. | Stopped before outcomes. |
| F: multiwall | One-wall controller and stored telemetry cannot establish two safe, noninteracting cohorts. | Not eligible. | Stopped before outcomes. |

The protocol and individual evidence are in
`IDM_V5_6_TREATMENT_TOURNAMENT_PROTOCOL.md` and the five companion audit
documents.

## Candidate A four-case screen

Candidate A changed both active adaptive-servo caps from 1.0 to 0.3 m/s2 and
nothing else.  It used four pairs (eight SUMO branches) with the frozen
analytical-damping control, t=880 activation, 20-second window, and t=1080
endpoint.  The output is
`runs/probe/idm_v5_6_tournament_a_low_authority_p30_s4401_4402_4404_4405/`.

| Seed | Lifecycle / dose | Safety and service | Physical effect | Result |
| --- | --- | --- | --- | --- |
| 4401 | complete; 40 controlled CAV-s | admissible; 0 excess hard-brake onsets | no primary congestion benefit | no screen benefit |
| 4402 | complete; 40 controlled CAV-s | admissible; −1 hard-brake onset | 29 s material shortening | retained benefit |
| 4404 | complete; 38 controlled CAV-s | admissible; +1 hard-brake onset | no primary benefit | no screen benefit |
| 4405 | formation veto; 0 controlled CAV-s | no treatment applied | no shortening | integrity/action-headroom failure |

For 4405, the candidate-specific trajectory projection produced a 59.50 m
maximum span against the frozen 50 m cap.  The treatment therefore did not
form.  This is the decisive failure: a low-authority wall cannot both preserve
the current compactness contract and retain the known safe-beneficial 4405
case.  It must not be recoded as a neutral no-op.

Compared with v5.4/v5.5.1, Candidate A reduced excess hard-braking onsets in
4401 from +3/+3 to 0 and in 4402 from +4/+4 to −1, while 4404 was +1 versus
+2/+1.  This evidence shows that reducing authority can reduce the direct
braking signature, but it does not establish a viable wall treatment because
the treatment loses required feasibility/benefit in 4405.  The all-case gate
therefore fails: `all_integrity=false`, `no_negligible_authority=false`, and
`4405_retains_material_benefit=false`.

## Verification and supported claims

- Focused v5.6/v5.5.1 tests: 11 passed.
- Full repository suite: 937 passed.
- New SUMO branches: 8 (four damping-versus-Candidate-A pairs).
- Fresh, validation, and final branches: 0.

Supported: the v5.4/v5.5.1 compact wall has authority, lower authority changes
that action materially, and under this frozen geometry/time horizon the
low-authority variant loses a necessary treatment dose in a known
safe-beneficial state.

Unsupported: a safe escalation treatment, collapse prevention across seeds,
a winning moving-wall repair, a predictor, calibrated probabilities, IDM
generalization, or any policy/RL efficacy claim.
