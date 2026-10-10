# IDM v5.6.2 phase-split-authority four-case results

Status: **completed development-only four-case gate — no-go**

## Frozen execution

- Contract: `idm_compact_moving_wall/5.6.2-phase-split-authority-settling`
- Fingerprint: `a5f3f2bbe3dc42ed7e3a7e1258e4dd34447f537bfaecbdba286de13ed26bfe02`
- Control: exact frozen analytical IDM disturbance-P30 damping.
- Treatment: one compact occupied-lane wall at t=880, 20-second lifecycle,
  five-second taper, and t=1080 horizon.
- Development seeds only: 4401, 4402, 4404, 4405.
- Artifact: `runs/probe/idm_v5_6_2_phase_split_authority_p30_s4401_4402_4404_4405/`.

Focused lifecycle tests passed 41/41. The repository suite passed **948/948**
before execution. Four paired branches completed; no fresh, validation, final,
predictor, or RL data were opened.

## Authority and dose audit

Every completed treatment had replay/prefix identity, one one-second
1.0 m/s²-cap assembly command, full lane coverage, no collision or teleport,
exact damping handoff, a 20-step maximum window, and 0.3 m/s²-cap HOLD/taper.
The realised post-assembly member accelerations were already within the
0.3 m/s² low-authority envelope (about -0.07 to +0.00 m/s²). The
actual-command rule therefore entered HOLD directly and recorded zero SETTLING
commands. This is not a bypass: no realised high-deceleration response needed
to be relaxed.

| Seed | Dose (assembly/HOLD/taper) | Safety/service | Maximum span (m) | Physical result |
| --- | ---: | --- | ---: | --- |
| 4401 | 1 / 14 / 5 | admissible | 40.07 | no harm; no primary benefit |
| 4402 | 1 / 14 / 5 | admissible | 27.23 | 5 s reduction; not material |
| 4404 | 1 / 13 / 5 | admissible | 153.92 | 50 m span failure; no benefit |
| 4405 | 1 / 14 / 5 | admissible | 90.93 | 50 m span failure; no benefit |

4404 exceeded 50 m in HOLD and reached 153.92 m during taper. 4405 stayed
compact in HOLD but reached 90.93 m during taper. Both retained full occupied
lane coverage; this is relative-speed/compactness failure, not lane loss.

## Gate decision

Passed: replay, predecision identity, formation, coverage, authority caps,
settling jerk/tau checks, common horizon, no hard abort, delayed safety, all
four safety admissions, all four service admissions, and non-negligible dose.

Failed:

1. all-case compact-span integrity (4404 and 4405);
2. retained 4402 material benefit: v5.4 91 s prevention versus v5.6.2 5 s;
3. retained 4405 material benefit: v5.4 24 s shortening versus v5.6.2 6 s,
   below the 10 s minimum.

The four-case gate is **false**. The conditional 24-seed screen was not
authorized and was not run.

Supported: the phase-split lifecycle preserves the declared authority boundary
and immediate safety/service in these four development cases. Unsupported:
compactness through release, retained benefit, a working escalation treatment,
collapse prevention, or predictor labels.
