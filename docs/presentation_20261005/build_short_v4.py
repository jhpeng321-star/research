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
      "Snapshot -> step-by-step simulation (CAVs run real controller code; human drivers = IDM physics + learned correction) for 120 s -> summary of CAV motion -> effect head -> change in total delay vs native at 30 s to 9.5 min with uncertainty. Delay is predicted by the head, not counted from the simulated cars.",
      ["5 boxes: snapshot, simulate, summary, effect head, output"])
B = dict(color="1F3A5F", bold=True, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
tb(s, 0.3, 1.7, 2.2, 2.2, ["Snapshot", "what the RSU sees"], size=22, fill="EEF3F7", line="1F3A5F", **B)
tb(s, 2.5, 2.45, 0.45, 0.7, "→", size=36, **B)
tb(s, 2.95, 1.4, 3.6, 2.8, ["Simulate 120 s", "CAVs: real controller", "Humans: IDM + learned correction"], size=21, fill="D5EEF0", line="0E7C86", **B)
tb(s, 6.55, 2.45, 0.45, 0.7, "→", size=36, **B)
tb(s, 7.0, 1.7, 2.0, 2.2, ["CAV motion", "summary"], size=22, fill="EEF3F7", line="1F3A5F", **B)
tb(s, 9.0, 2.45, 0.45, 0.7, "→", size=36, **B)
tb(s, 9.45, 1.4, 3.55, 2.8, ["Effect head", "→ Δ total delay vs native", "30 s … 9.5 min"], size=21, fill="FFF4D6", line="C77700", **B)
tb(s, 0.3, 4.6, 12.7, 1.0, "Delay is predicted by the effect head — not counted from the simulated cars", size=22, color="C77700", bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
takeaway(s, "Physics (IDM) + learned correction simulates traffic; a learned head predicts the delay change", y=6.0, h=0.85)

# 2
s = S("What our world model achieves now",
      "Prediction vs reality for the PI-vs-native delay effect on 30 fresh worlds (post-hoc, source scorer-repl ACCURACY.md). Values A13 (A11). Direction accuracy among pairs with |true| >= 1 s; Spearman rank correlation. Message: the model knows whether PI helps with honest uncertainty, not yet how much.",
      ["Table 30 s / 300 s"])
tb(s, 0.4, 1.35, 12.5, 0.5, "Prediction vs reality: PI effect, 30 fresh worlds", size=22, color="1F3A5F", bold=True)
rows = [["Horizon", "Error A13 (A11), s", "True effect size, s", "Direction right", "Rank corr."],
        ["30 s", "16.1 (21.0)", "18.3", "85% (81%)", "0.45 (0.29)"],
        ["300 s", "963 (957)", "1057", "77% (73%)", "0.44 (0.26)"]]
table(s, 0.4, 2.0, 12.5, rows, col_w=[1.9, 2.9, 2.7, 2.5, 2.5], size=22, row_h=1.1)
tb(s, 0.4, 5.6, 12.5, 0.8, "Delay = total over all vehicles (vehicle-seconds), PI minus native. Numbers in brackets: A11.", size=18, color="666666")

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
   "Status: physics not accurate enough yet (fitted w ≈ 0). Root cause found: merge rule never fired (coordinate bug); fix cuts merge-speed error 13.5 → 3.9 m/s; next: calibrate merging (now over-queues)"], size=17)
takeaway(s, "Use simulated driver reactions directly where physics is reliable; learning where it is not", y=6.2, h=0.75)

prs.save(os.path.join(HERE, "WM_short_20261005.pptx"))
out = ["# WM-N1 short deck - outline and speaker notes (generated by build_short.py)\n"]
for n, t, ls, nt in bd.OUTLINE:
    out.append(f"## Slide {n}: {t}\n"); out += [f"- {l}" for l in ls]; out.append(f"\n**Notes:** {nt}\n")
open(os.path.join(HERE, "OUTLINE_short.md"), "w", encoding="utf-8").write("\n".join(out))
print(len(prs.slides), "slides"); print("\n".join(bd.WARN) or "no overflow warnings")
