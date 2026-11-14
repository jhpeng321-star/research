"""4-slide deck. python build_short.py -> WM_short_20261005.pptx + OUTLINE_short.md (reuses build_deck helpers)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import build_deck as bd
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN

bd.FOOT = "WM-N1 world model | 2026-10-05 | seconds; lower is better for errors and delay"
prs = bd.prs; S = bd.new_slide
tb, table, takeaway = bd.tb, bd.table, bd.takeaway

# 1
s = S("WM-N1 world model — progress (2026-10-05)", "Title page.", ["Title", "Subtitle"])
tb(s, 0.5, 2.6, 12.3, 1.2, "What the model achieves now, and what we fix next", size=34, color="0E7C86", bold=True)
tb(s, 0.5, 6.0, 12.3, 0.6, "Jinghan Peng | 2026-10-05", size=18)

# 1b how the world model works
s = S("How our world model works now",
      "Physics-informed rollout every 0.5 s: CAVs run the real controller code; human drivers get IDM car-following acceleration plus a learned neural correction (and MOBIL lane changes plus a learned adjustment); kinematic update with no overlaps. A separate effect head predicts the change in total delay vs native at 30 s to 9.5 min with uncertainty, reading the snapshot plus a 120 s CAV-motion summary.",
      ["Snapshot -> per-step loop -> physics update", "a = a_IDM + correction", "Effect head output"])
C = dict(color="1F3A5F", bold=True, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
tb(s, 0.3, 1.5, 2.6, 2.6, ["RSU snapshot", "CAVs + the cars they detect", "(position, speed, lane)"], size=17, fill="EEF3F7", line="1F3A5F", **C)
tb(s, 2.95, 2.45, 0.5, 0.6, "→", size=34, **C)
tb(s, 3.5, 1.35, 5.9, 1.25, ["CAVs: the REAL controller code", "(native / PI / damping / wall) → exact acceleration"], size=17, fill="FFF4D6", line="C77700", **C)
tb(s, 3.5, 2.8, 5.9, 1.45, ["Human drivers:", "a = a_IDM(speed, gap, closing speed) + Δa (neural correction)", "lane change = MOBIL + learned adjustment"], size=17, fill="D5EEF0", line="0E7C86", **C)
tb(s, 9.45, 2.45, 0.5, 0.6, "→", size=34, **C)
tb(s, 10.0, 1.5, 3.0, 2.6, ["Physics update", "speed += a·0.5 s", "position += speed·0.5 s", "no overlaps"], size=17, fill="EEF3F7", line="1F3A5F", **C)
tb(s, 3.5, 4.3, 5.9, 0.5, "repeat every 0.5 s → imagined future (e.g. 120 s)", size=16, color="666666", anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
tb(s, 0.3, 4.95, 12.7, 1.1, ["The network only learns the CORRECTION to IDM: how real drivers deviate from the textbook rule.",
   "Delay output: a separate effect head predicts the change in total delay vs native (30 s to 9.5 min, with uncertainty)."], size=18, color="1F3A5F")
takeaway(s, "IDM gives the physics; the neural network learns how human drivers deviate from it", y=6.2, h=0.75)

# 2
s = S("What our world model achieves now",
      "Prediction vs reality for the PI-vs-native delay effect on 30 fresh worlds (post-hoc, source scorer-repl ACCURACY.md). Values A13 (A11). Direction accuracy among pairs with |true| >= 1 s; coverage of the model 80% range, ideal 80%; Spearman rank correlation. Message: the model knows whether PI helps with honest uncertainty, not yet how much.",
      ["Table 30 s / 300 s"])
tb(s, 0.4, 1.35, 12.5, 0.5, "Prediction vs reality: PI effect, 30 fresh worlds", size=22, color="1F3A5F", bold=True)
rows = [["Horizon", "Error A13 (A11), s", "True effect size, s", "Direction right", "80%-range coverage", "Rank corr."],
        ["30 s", "16.1 (21.0)", "18.3", "85% (81%)", "87% (63%)", "0.45 (0.29)"],
        ["300 s", "963 (957)", "1057", "77% (73%)", "83% (80%)", "0.44 (0.26)"]]
table(s, 0.4, 2.0, 12.5, rows, col_w=[1.6, 2.4, 2.2, 2.0, 2.3, 2.0], size=22, row_h=1.1)
tb(s, 0.4, 5.6, 12.5, 0.8, "Delay = total over all vehicles (vehicle-seconds), PI minus native. Coverage ideal: 80%. Numbers in brackets: A11.", size=18, color="666666")

# 3
s = S("Next fix: blend physics and learning",
      "Today the delay change comes only from the learned head reading a coarse CAV summary; the simulated driver reactions are unused. Blend: physics-measured delay change from the open-road rollout (all cars, IDM+correction, real CAV controllers) for short horizons, learned head for long horizons; weight w(tau) fitted on training data with leave-one-world-out CV. Status: closed-road physics nearly blind (predicted spread 0.7 s vs true 25 s); open-road version being tested now; needs realistic queues/speeds.",
      ["Today vs blend", "Formula", "How (4 steps)", "Status"])
tb(s, 0.4, 1.3, 12.5, 0.9, ["Today: the delay change comes only from the learned head, which reads a coarse CAV summary.",
   "The simulated reactions of each human driver are NOT used for the delay."], size=19, color="1F3A5F")
tb(s, 0.9, 2.25, 11.5, 0.9, "Δdelay(τ) = w(τ) · Δdelay_physics(τ) + (1 − w(τ)) · Δdelay_learned(τ)", size=22, color="0E7C86", bold=True, fill="EEF3F7", line="0E7C86", anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
tb(s, 0.4, 3.25, 12.5, 2.95, [
   "1. For each candidate plan, simulate ALL cars on an open road (IDM + correction; CAVs run their real controllers)",
   "2. Count the total delay of every simulated car for the plan and for native → physics effect at 30 / 60 / 120 s",
   "3. The learned head gives its effect (with uncertainty) at all horizons",
   "4. Weight w(τ): high at short horizons (physics reliable), low at long horizons (simulation drifts), fitted on training data",
   "Status: closed-road physics saw almost no effect (spread 0.7 s vs true 25 s); open-road version being tested now"], size=17)
takeaway(s, "Use simulated driver reactions directly where physics is reliable; learning where it is not", y=6.2, h=0.75)

prs.save(os.path.join(HERE, "WM_short_20261005.pptx"))
out = ["# WM-N1 short deck - outline and speaker notes (generated by build_short.py)\n"]
for n, t, ls, nt in bd.OUTLINE:
    out.append(f"## Slide {n}: {t}\n"); out += [f"- {l}" for l in ls]; out.append(f"\n**Notes:** {nt}\n")
open(os.path.join(HERE, "OUTLINE_short.md"), "w", encoding="utf-8").write("\n".join(out))
print(len(prs.slides), "slides"); print("\n".join(bd.WARN) or "no overflow warnings")
