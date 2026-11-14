"""Progress-update deck. python build_progress.py -> WM_progress_20261005.pptx + OUTLINE_progress.md (reads numbers.json, reuses build_deck helpers)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import build_deck as bd
import matplotlib.pyplot as plt
N = json.load(open(os.path.join(HERE, "numbers.json"), encoding="utf-8"))
bd.FOOT = "WM-N1 progress update | 2026-10-05 | lower is better for all delay and error numbers (seconds)"
TAUS = N["taus"]; TL = [f"{t} s" for t in TAUS]
col = {"A11": "#6BAED6", "A12": "#B03A48", "A13": "#0E7C86"}


def save(fig, n):
    p = os.path.join(bd.FIG, n); fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig); return p


def fig_repl():
    names = ["native", "A11", "A9", "A10", "A12", "A13"]; v = [1117, 1073, 1136, 1138, 1139, 1160]
    fig, ax = plt.subplots(figsize=(8.6, 5))
    ax.bar(names, v, color=["#1F3A5F", "#0E7C86", "#9E9AC8", "#6BAED6", "#B03A48", "#444444"])
    for i, x in enumerate(v): ax.text(i, x + 15, str(x), ha="center", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1400); ax.set_ylabel("regret at 300 s, seconds of total delay\n(lower is better)")
    ax.set_title("30 fresh worlds: every model ties native", fontsize=14)
    return save(fig, "prog_repl.png")


def fig_cl():
    names = ["always-PI", "fleet-A11", "native", "planner+A11", "planner+walls", "random"]; v = [40750, 41943, 41972, 42003, 42003, 42409]
    fig, ax = plt.subplots(figsize=(8.6, 5))
    ax.bar(names, [x - 40000 for x in v], color=["#0E7C86", "#6BAED6", "#1F3A5F", "#6BAED6", "#9E9AC8", "#E08A5B"])
    for i, x in enumerate(v): ax.text(i, x - 40000 + 25, str(x), ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(range(6)); ax.set_xticklabels(names, rotation=20, fontsize=12)
    ax.set_ylim(0, 2800); ax.set_yticks([0, 1000, 2000]); ax.set_yticklabels(["40000", "41000", "42000"])
    ax.set_ylabel("mean total delay over 600 s, seconds\n(lower is better; axis cut at 40000)")
    ax.set_title("Closed loop, 30 fresh worlds", fontsize=14)
    return save(fig, "prog_cl.png")


prs = bd.prs
S = bd.new_slide
tb, table, takeaway, bullets, pic = bd.tb, bd.table, bd.takeaway, bd.bullets, bd.pic

# 1
s = S("WM-N1 world model: progress update (2026-10-05)", "Progress update; the audience knows the project. Three status points.", ["3 status bullets"])
bullets(s, ["Short-horizon error halved (A13), calibrated ranges", "Closed-loop imagination bottleneck found and cut 4x", "Pre-registered decision test: not yet beating native"], y=2.0, h=3.2, size=30)
tb(s, 0.5, 6.0, 12.3, 0.6, "Jinghan Peng | 2026-10-05", size=18)

# 2
s = S("This round at a glance: what was done", "Everything below ran this round. Counts are SUMO runs or models as stated.", ["6 work items"])
rows = [["Work item", "Scale"],
        ["Models A8 to A13 trained and scored", "6 models"],
        ["Evaluation on all controllers", "CAL-tune + evaluation slice"],
        ["Pre-registered replication", "30 fresh worlds, 390 SUMO runs"],
        ["Closed-loop test, 6 policies (incl. per-CAV to group planner, walls)", "30 worlds, 180 runs"],
        ["Open-road rollout fix", "exits + entries added"],
        ["Physics-vs-learning check", "short-horizon rollout diagnostic"]]
table(s, 0.5, 1.5, 12.3, rows, col_w=[7.0, 5.3], size=20, row_h=0.62)

# 3
s = S("Achievement 1: prediction vs reality (PI effect, A11 to A13)", "Source E:/rd-wt/scorer-repl/_out/accuracy/ACCURACY.md. PI-vs-native delay effect; post-hoc descriptive, no tuning. Error = median absolute error of the predicted median vs the true effect. Direction accuracy among pairs with |true| >= 1 s. Coverage = true effect inside the model 80% range (ideal 80%). Rank correlation = Spearman. CAL-tune 11 situations; fresh worlds 30 situations (post-hoc). A13 halves the 30 s error on CAL (5.4 vs 12.4) and fresh (16.1 vs 21.0); ranges are calibrated; but 300 s error (963 vs true 1057 on fresh) is close to the effect size: situation discrimination is the open problem.",
      ["Table A13 vs A11 at 30 s and 300 s, CAL and fresh", "Scatter A13/A11 fresh 300 s"])
rows = [["Set, horizon", "Error A13 / A11, s", "True effect, s", "Direction right", "80%-range coverage", "Rank corr."],
        ["CAL (11), 30 s", "5.4 / 12.4", "5.9", "0.71 / 0.71", "1.00 / 0.82", "0.61 / 0.66"],
        ["CAL (11), 300 s", "585 / 509", "768", "0.82 / 0.82", "0.91 / 1.00", "0.58 / 0.37"],
        ["Fresh (30), 30 s", "16.1 / 21.0", "18.3", "0.85 / 0.81", "0.87 / 0.63", "0.45 / 0.29"],
        ["Fresh (30), 300 s", "963 / 957", "1057", "0.77 / 0.73", "0.83 / 0.80", "0.44 / 0.26"]]
table(s, 0.4, 1.35, 12.5, rows, col_w=[2.6, 2.4, 1.7, 1.9, 2.2, 1.7], size=17, row_h=0.5)
pic(s, os.path.join(bd.FIG, "scatter_A13_repl_300s.png"), 3.2, 3.95, h=2.3)
pic(s, os.path.join(bd.FIG, "scatter_A11_repl_300s.png"), 6.6, 3.95, h=2.3)
tb(s, 0.4, 4.4, 2.8, 1.4, "Fresh worlds, 300 s: predicted vs true effect (A13 left, A11 right)", size=14)
takeaway(s, "A13 halves the 30 s error, ranges well calibrated (~80-90%), direction right ~77-85%; 300 s error (963 vs 1057 s) still near effect size", y=6.3, h=0.7)

# 4
s = S("Achievement 2: closed-loop bottleneck found, error cut 4x", "The imagined future was a closed road (no exits or entries), giving unrealistic CAV-motion summaries and inflated predictions. 8 TRAIN states, PI effect at 300 s. Example: true +1123, recorded plan +1355, closed road +3732. Source E:/rd-wt/rollout-entry/_out/ENTRY_FIX.md.",
      ["Table of 4 futures", "Example true +1123 / recorded +1355 / closed +3732", "n_cav plot"])
rows = [["Imagined future (8 TRAIN states)", "Mean abs error, s", "Corr"], ["Closed road", "3564", "0.45"], ["Exits only", "961", "0.35"], ["Open road", "947", "0.37"], ["Real-recorded future", "761", "0.52"]]
table(s, 0.4, 1.5, 7.0, rows, col_w=[3.6, 2.2, 1.2], size=19, row_h=0.6)
tb(s, 0.4, 4.7, 7.0, 1.4, "Example: true +1123 s; recorded plan +1355 s; closed road +3732 s", size=19, color="1F3A5F", bold=True)
pic(s, os.path.join(bd.FIG, "wm_ncav.png"), 7.6, 1.5, w=5.4)
takeaway(s, "Closed road inflated CAV counts; open road tracks the real future (error 3564 to 947 s)")

# 5
s = S("Achievement 3: a rigorous decision test, honest result", "Evaluation slice: A11 972 vs native 1556 at 300 s (CI [-1339,-3]). Pre-registered replication on 30 fresh worlds: A11 1073 vs native 1117 (diff CI [-225,+113]) not replicated; A9 1136, A10 1138, A12 1139, A13 1160 all tie native. The replication caught a selection effect; the model does not yet separate situations.",
      ["Slice 972 vs 1556", "Fresh 30 worlds 1073 vs 1117 CI [-225,+113]", "All models tie native"])
pic(s, fig_repl(), 0.4, 1.4, w=7.2)
bullets(s, ["Evaluation slice: A11 972 vs native 1556 at 300 s (CI [-1339, -3])", "Fresh worlds: A11 1073 vs native 1117 (CI [-225, +113]), not replicated", "A9 1136, A10 1138, A12 1139, A13 1160: all tie native"], x=7.8, y=1.5, w=5.2, h=4.6, size=20)
takeaway(s, "Replication caught a selection effect: no model separates situations yet")

# 6
s = S("Closed loop: safe, but the planner does not yet capture gains", "30 fresh worlds, mean total delay over 600 s: always-native 41972; planner+A11 42003 (diff +31 [-62,+154]); planner+walls 42003 (no wall assembled); fleet-A11 41943 (-29 [-198,+123]); always-PI 40750 (-1221 [-3168,+480]); random 42409. 0 collisions/teleports in 180 runs; planner non-native only 0.4% of CAV-decisions. Headroom about 1100 s/decision at 300 s.",
      ["Bar chart", "0 collisions in 180 runs", "planner non-native 0.4%"])
pic(s, fig_cl(), 0.3, 1.4, w=7.4)
bullets(s, ["planner+A11: +31 vs native [-62, +154]", "planner+walls: same 42003 (no wall assembled)", "fleet-A11: -29 [-198, +123]", "always-PI: -1221 [-3168, +480]", "0 collisions / teleports in 180 runs", "Planner non-native in only 0.4% of CAV-decisions"], x=7.8, y=1.5, w=5.2, h=4.7, size=20)
takeaway(s, "Gains exist (always-PI, ~1100 s/decision headroom at 300 s); the model does not capture them yet")

# 7
s = S("What did not work", "A12: per-horizon gate penalised only the gate, so the correction compensated; short-horizon error not fixed. Physics rollout at short horizon: closed-road rollout error 13.3 s at 30 s vs true effect sd 25 s, predicted effects nearly zero (sd 0.7 s); needs a realistic rollout. Arrival head degenerate (predicts 0), so entries use observed counters.",
      ["A12", "physics rollout short horizon", "arrival head"])
rows = [["Attempt", "What went wrong"],
        ["A12 per-horizon gate", "Penalty on gate only; correction compensated; short-horizon error not fixed"],
        ["Physics rollout, short horizon", "Closed-road rollout error 13.3 s at 30 s vs true effect sd 25 s; predicted effects nearly zero (sd 0.7 s); needs realistic rollout"],
        ["Arrival head", "Degenerate (predicts 0); entries use observed counters instead"]]
table(s, 0.5, 1.6, 12.3, rows, col_w=[3.6, 8.7], size=22, row_h=1.15)

# 8
s = S("Still waiting to be fixed", "Open issues with status. Situation discrimination is the main open problem.", ["6 open issues"])
rows = [["Issue", "Why it matters", "Status"],
        ["Situation discrimination (corr ~0.4-0.5)", "Main open problem; captures ~4% of headroom", "Open"],
        ["Per-CAV plan input", "Pooled summary is blind to which CAV / where", "Code in progress (F3), training next"],
        ["Rollout queues / speeds (22-27 vs 16-23 m/s, no queues)", "Imagined traffic not realistic enough", "Diagnosis running (F2)"],
        ["Physics-measured short-horizon delay + blend", "Short-horizon fidelity", "Running on open road (F1)"],
        ["Keep 300 s sharpness with A13's 30 s fix", "300 s error still ~90% of effect size", "Tune"],
        ["Unseen-controller test (leave-PI-out)", "Generalisation to new controllers", "Registered, ~2 h GPU"],
        ["Seed variance", "Are differences real?", "1 extra run"]]
table(s, 0.4, 1.4, 12.5, rows, col_w=[4.4, 4.4, 3.7], size=17, row_h=0.66)

# 9
s = S("Next steps", "Order: F2 fidelity fix then open-road diagnostic; train per-CAV model A14 on gpuA; leave-PI-out and seed replicate; then re-test decisions on fresh worlds and closed loop with open-road rollout.", ["4 ordered steps", "closing line"])
bullets(s, ["1. F2 fidelity fix, then rerun the open-road diagnostic", "2. Train per-CAV model (A14) on gpuA", "3. Leave-PI-out test + seed replicate", "4. Re-test decisions on fresh worlds and in closed loop with open-road rollout"], y=1.6, h=3.8, size=26)
takeaway(s, "Controllers help (always-PI -1221 s; ~1100 s/decision headroom); the path is realistic, per-CAV imagined futures", y=5.6, h=1.1)

# 10 appendix
s = S("Appendix: A13 on fresh worlds", "A13 300 s regret 1160 vs native 1117, diff CI [-198,+287]: ties native.", ["A13 fresh"])
bullets(s, ["300 s regret: A13 1160 vs native 1117", "Difference CI [-198, +287]: ties native"], y=2.0, h=2.5, size=30)

prs.save(os.path.join(HERE, "WM_progress_20261005.pptx"))
out = ["# WM-N1 progress deck - outline and speaker notes (generated by build_progress.py)\n"]
for n, t, ls, nt in bd.OUTLINE:
    out.append(f"## Slide {n}: {t}\n"); out += [f"- {l}" for l in ls]; out.append(f"\n**Notes:** {nt}\n")
open(os.path.join(HERE, "OUTLINE_progress.md"), "w", encoding="utf-8").write("\n".join(out))
print(len(prs.slides), "slides"); print("\n".join(bd.WARN) or "no overflow warnings")
