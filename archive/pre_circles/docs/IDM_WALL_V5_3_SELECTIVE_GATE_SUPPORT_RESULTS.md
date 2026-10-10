# IDM wall v5.3 selective-gate support results

Status: **completed no-go at the frozen integrity gate.**

Date: 2026-08-31

The treatment-blind scan evaluated 20 prospective seeds at 10--30% CAV
penetration. The lowest level satisfying the frozen availability rule was 25%:
24 admitted windows across 16 seeds (20% had 19 windows across 13 seeds).

The paired p25 pilot then evaluated 80 seed/windows, opening treatment only for
the 24 admitted starts. Exact damping and treatment repeats passed. Twenty-two
attempts (91.7%) met the compact composite dose contract across 15 seeds. Those
qualified cases supplied five joint safe/service-noninferior/beneficial cases
across five seeds and 17 negative cases across 13 seeds; safe and unsafe labels
each exceeded the three-seed support minimum. No excess collision or teleport
occurred.

The pilot nevertheless fails and does not authorize predictor generation.
Three late hard-brake safety aborts occurred after the nominal taper had begun.
The existing v5.3 runtime correctly removed wall authority immediately, but its
audit sequence ended at taper blend 0.8 rather than recording the physical
zero-authority handoff to identical damping. This violates the prospectively
frozen requirement that the taper reach zero. A separate admitted case never
reached HOLD and remains an ordinary dose failure.

The failure is narrow: prefix identity, first-request base actions,
out-of-band damping, wall-member exclusivity, post-release damping, compactness,
full-lane HOLD, replay and treatment-effect-above-noise checks all pass. Historical
artifacts are unchanged. v5.3 is retained as negative integrity evidence.

The next permitted treatment is separately versioned v5.4. It does not weaken
the hard safety abort: an abort still removes wall authority immediately and
returns the CAV to its already-computed damping action. It adds an explicit,
machine-auditable zero-authority endpoint when a hard abort interrupts the
nominal taper. v5.4 must pass on a fresh prospective cohort before larger data
generation.
