"""Regenerate WM-N1_update_20261005.pptx, OUTLINE.md and figures/*.png from numbers.json.

Usage (rl_driving env):  python build_deck.py
Edit numbers.json only: every null value in repl / closed_loop / a12 renders as a
'PENDING (result expected HH:MM)' placeholder; fill it and rebuild.

numbers.json keys
  taus                       list of horizons (s)
  models[]                   name/change/ckpt rows for the progression table
  slice_regret.{A11,A10,A9,A8,always_native,random}   regret (s) per tau; a11_vs_native_300/570 {a11,native,ci90}
  pi_native_crps.{A11,A10,A9,A8,M1FIX1,climatology}   CRPS (s) per tau; a11_300_ci
  noise.{ratio,ci_lo,ci_hi,n1_sd,pi_sd}               PI-native sd / one-CAV-nudge sd per tau
  t4g.{clim_crps,hgb_crps,clim_sign,hgb_sign}         predictability from t0 state
  dspeed.*                                            direct-speed campaign counts
  repl.{expected, a11_regret_300, native_regret_300, ci90_diff_300 ("[lo, hi]" string or list),
        a10_regret_300, a9_regret_300, other_taus{30,60,120,570}, verdict}   PENDING until filled
  closed_loop.{expected, pl01_a11, fleet_wm, always_native, always_pi, random, pl01_a11_walls,
        diff_pl01_vs_native_ci90, walls_cutoff_note}                          PENDING until filled
        (set pl01_a11_walls to the string "NOT DONE" if the 17:00 cutoff is missed)
  a12.{expected, crps_30, crps_300, ref{...}}                                  PENDING until filled
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "numbers.json"), encoding="utf-8"))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
TAUS = N["taus"]

NAVY, TEAL, GREY, LIGHT = "1F3A5F", "0E7C86", "444444", "EEF3F7"
AMBER_BG, AMBER = "FFF3CD", "C77700"
C = {"A11": "#0E7C86", "A10": "#6BAED6", "A9": "#9E9AC8", "A8": "#BDBDBD", "native": "#1F3A5F",
     "random": "#E08A5B", "clim": "#C77700", "M1": "#B03A48"}
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})
TAU_LABELS = [f"{t} s" for t in TAUS]


def rgb(h): return RGBColor.from_string(h)
def pending(v): return v is None or (isinstance(v, str) and v.strip().upper() == "PENDING")
def ptxt(exp): return f"PENDING (result expected {exp})"


def show(v, exp, fmt="{}"):
    if pending(v): return ptxt(exp)
    if isinstance(v, (list, tuple)): return "[" + ", ".join(str(x) for x in v) + "]"
    if isinstance(v, (int, float)): return fmt.format(v)
    return str(v)


# ----------------------------------------------------------------- figures
def save(fig, name):
    p = os.path.join(FIG, name); fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig); return p


def fig_noise():
    n = N["noise"]; r = n["ratio"]
    lo = [a - b for a, b in zip(r, n["ci_lo"])]; hi = [b - a for a, b in zip(r, n["ci_hi"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(TAU_LABELS, r, color=C["A11"], yerr=[lo, hi], capsize=6, ecolor="#333")
    ax.axhline(1, color="#B03A48", ls="--", lw=2, label="noise floor (ratio 1)"); ax.legend(loc="upper right", fontsize=12)
    ax.set_yscale("log"); ax.set_ylim(0.8, 3000)
    for i, v in enumerate(r): ax.text(i + 0.12, v * 1.05, f"{v:g}x", ha="left", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel("sd(PI-native) / sd(one-CAV nudge)\n(log scale, 90% CI)"); ax.set_xlabel("prediction horizon")
    ax.set_title("PI effect spread vs null-perturbation noise floor (delay, s)", fontsize=14)
    return save(fig, "noise_floor.png")


def fig_regret():
    s = N["slice_regret"]
    ser = [("A11", s["A11"], C["A11"]), ("A10", s["A10"], C["A10"]), ("A9", s["A9"], C["A9"]), ("A8", s["A8"], C["A8"]),
           ("always-native", s["always_native"], C["native"]), ("random", s["random"], C["random"])]
    fig, ax = plt.subplots(figsize=(11, 5.2)); w = 0.13; x = range(len(TAUS))
    for k, (lab, v, col) in enumerate(ser):
        ax.bar([i + (k - 2.5) * w for i in x], v, w, label=lab, color=col, edgecolor="white")
    ax.set_yscale("log"); ax.set_xticks(list(x)); ax.set_xticklabels(TAU_LABELS)
    ax.set_ylabel("decision regret, s of total delay\n(lower is better, log scale)"); ax.set_xlabel("decision horizon")
    ax.legend(ncol=3, fontsize=11, loc="upper left")
    i = TAUS.index(300)
    ax.annotate(f"A11 {s['A11'][i]:g} s\nvs native {s['always_native'][i]:g} s", xy=(i - 2.5 * w, s["A11"][i]), xytext=(1.1, 4500),
                fontsize=12, fontweight="bold", color=C["A11"], arrowprops=dict(arrowstyle="->", color=C["A11"]))
    ax.set_ylim(0.3, 3e4)
    return save(fig, "regret_slice.png")


def fig_crps():
    d = N["pi_native_crps"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, col, ls, lw in [("M1FIX1", C["M1"], ":", 2), ("A8", C["A8"], "-", 2), ("A9", C["A9"], "-", 2),
                           ("A10", C["A10"], "-", 2), ("A11", C["A11"], "-", 3.5), ("climatology", C["clim"], "--", 3)]:
        ax.plot(TAU_LABELS, d[k], ls, color=col, lw=lw, marker="o", label=k if k != "climatology" else "TRAIN climatology")
    ax.set_yscale("log"); ax.set_ylabel("CRPS, seconds of paired delay\n(lower is better, log scale)"); ax.set_xlabel("prediction horizon")
    ax.legend(fontsize=11, ncol=2); ax.set_title("PI-vs-native paired effect, CAL-tune, 11 decisions", fontsize=14)
    return save(fig, "crps_models.png")


def fig_t4g():
    t = N["t4g"]
    fig, ax = plt.subplots(figsize=(9, 5)); w = 0.38; x = range(len(TAUS))
    ax.bar([i - w / 2 for i in x], t["clim_crps"], w, color=C["clim"], label="climatology (marginal)")
    ax.bar([i + w / 2 for i in x], t["hgb_crps"], w, color=C["A10"], label="HGB on 50 t0-state features")
    ax.set_yscale("log"); ax.set_xticks(list(x)); ax.set_xticklabels(TAU_LABELS)
    ax.set_ylabel("CRPS, s of delay (lower is better, log)"); ax.set_xlabel("prediction horizon"); ax.legend(fontsize=12)
    ax.set_title("World-grouped 5-fold CV, 161 PI-native pairs (TRAIN)", fontsize=14)
    return save(fig, "t4g_predictability.png")


def box(ax, x, y, w, h, text, fc="#EEF3F7", ec="#1F3A5F", fs=12, bold=False, tc="#1F3A5F"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, lw=2, color="#444"))


def fig_model():
    fig, ax = plt.subplots(figsize=(13, 4.8)); ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")
    box(ax, 0.1, 3.0, 2.3, 1.4, "RSU-observable\nstate at t0\n(CAV self state +\ndetected neighbours)", fs=11)
    box(ax, 0.1, 0.3, 2.3, 1.8, "Candidate plan\n(controller / wall /\ndirect-speed;\nplan summary\n120 s)", fc="#FFF3CD", ec="#C77700", tc="#7A4B00", fs=11)
    box(ax, 3.2, 2.9, 3.4, 1.7, "Rollout simulator\nCAVs: real controller code\nHDVs: IDM/MOBIL +\nneural residual", fs=12, bold=True)
    box(ax, 3.2, 0.6, 3.4, 1.4, "Predicted CAV/HDV\ntrajectories", fs=12)
    box(ax, 7.4, 1.3, 3.0, 2.4, "Plan-aware effect head\nclimatology anchor\n+ gated plan-aware\nresidual\n(per-tau gate in A12)", fc="#D5EEF0", ec="#0E7C86", tc="#0A5A62", fs=11, bold=True)
    box(ax, 11.0, 1.3, 1.9, 2.4, "Quantiles of\ndelay change\nvs native at\n30-570 s", fs=12)
    arrow(ax, 2.4, 3.7, 3.2, 3.7); arrow(ax, 2.4, 1.3, 3.2, 3.3); arrow(ax, 4.9, 2.9, 4.9, 2.0)
    arrow(ax, 6.6, 1.3, 7.4, 2.0); arrow(ax, 6.6, 3.7, 7.4, 3.0); arrow(ax, 10.4, 2.5, 11.0, 2.5)
    ax.text(8.9, 0.35, "trained with ranking loss within each decision\n(rank loss only at tau >= 60 s in A12)", ha="center", fontsize=11, color="#555", style="italic")
    return save(fig, "model_diagram.png")


def fig_planner():
    fig, ax = plt.subplots(figsize=(13, 3.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 3.6); ax.axis("off")
    steps = ["1. Individual check\nper CAV: native,\ndamping, PI\n(others held)",
             "2. Group CAVs by\nmeasured interaction\n(step-1 predicted\ntrajectories)",
             "3. Group joint\ncheck (incl. wall\ncandidates)",
             "4. Switch rule\nvs current plan",
             "5. Combine +\nsafety fallback"]
    w, g = 2.25, 0.4
    for i, s in enumerate(steps):
        x = 0.1 + i * (w + g)
        hi = i in (0, 2)
        box(ax, x, 0.8, w, 2.2, s, fs=11, fc="#D5EEF0" if hi else "#EEF3F7", ec=C["A11"] if hi else "#1F3A5F")
        if i < 4: arrow(ax, x + w, 1.9, x + w + g, 1.9)
    ax.text(6.5, 0.25, "re-decision every 60 s;  predictor = world model (A11) behind the planner Predictor interface", ha="center", fontsize=12, color="#555", style="italic")
    return save(fig, "planner_diagram.png")


def fig_setting():
    fig, ax = plt.subplots(figsize=(8.5, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5); ax.axis("off")
    for y in (3.2, 2.6): ax.add_patch(Rectangle((0, y), 10, 0.55, fc="#D9D9D9", ec="#888"))
    ax.plot([0, 3.5, 5.5], [1.3, 1.3, 2.6], color="#888", lw=22, solid_capstyle="butt", alpha=0.5)
    ax.text(1.2, 0.7, "on-ramp", fontsize=12, color="#555"); ax.text(8.2, 3.95, "mainline", fontsize=12, color="#555")
    for x, y, c in [(0.8, 3.35, "h"), (2.0, 3.35, "c"), (3.3, 3.35, "h"), (4.6, 3.35, "h"), (6.2, 3.35, "c"), (7.5, 3.35, "h"),
                    (1.1, 2.75, "h"), (2.8, 2.75, "c"), (4.0, 2.75, "h"), (6.9, 2.75, "h"), (8.4, 2.75, "c"), (2.0, 1.45, "c")]:
        ax.add_patch(Rectangle((x, y - 0.1), 0.55, 0.3, fc="#0E7C86" if c == "c" else "#9E9E9E", ec="#222"))
    ax.plot([5.5, 5.5], [3.8, 4.7], color="#C77700", lw=3); ax.plot(5.5, 4.75, "^", color="#C77700", ms=16)
    ax.text(5.5, 5.1, "RSU: timestamped CAV state + detected neighbours\n(no true penetration, no hidden driver type)", ha="center", fontsize=11, color="#7A4B00")
    ax.add_patch(Rectangle((0.2, 0.1), 0.4, 0.22, fc="#0E7C86", ec="#222")); ax.text(0.7, 0.12, "CAV", fontsize=12)
    ax.add_patch(Rectangle((1.6, 0.1), 0.4, 0.22, fc="#9E9E9E", ec="#222")); ax.text(2.1, 0.12, "HDV (IDM, never commanded)", fontsize=12)
    return save(fig, "setting.png")


def fig_closed():
    c = N["closed_loop"]
    items = [("PL01+A11", c["pl01_a11"]), ("fleet-WM", c["fleet_wm"]), ("always-native", c["always_native"]),
             ("always-PI", c["always_pi"]), ("random", c["random"]), ("PL01+A11+walls", c["pl01_a11_walls"])]
    items = [(k, v) for k, v in items if isinstance(v, (int, float))]
    p = os.path.join(FIG, "closed_loop.png")
    if len(items) < 2:
        if os.path.exists(p): os.remove(p)
        return None
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar([k for k, _ in items], [v for _, v in items], color=[C["A11"] if k.startswith("PL01") else C["native"] for k, _ in items])
    ax.set_ylabel("total delay, s (lower is better)"); plt.setp(ax.get_xticklabels(), rotation=20)
    return save(fig, "closed_loop.png")


def build_figs():
    return {k: f() for k, f in [("noise", fig_noise), ("regret", fig_regret), ("crps", fig_crps), ("t4g", fig_t4g),
                                ("model", fig_model), ("planner", fig_planner), ("setting", fig_setting), ("closed", fig_closed)]}


# ----------------------------------------------------------------- deck helpers
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
OUTLINE = []; WARN = []


def est_lines(text, width_in, pt):
    cpl = max(1, int(width_in * 72 / (pt * 0.50)))
    return sum(max(1, -(-len(p) // cpl)) for p in text.split("\n"))


def tb(s, x, y, w, h, paras, size=22, color=GREY, bold=False, align=None, anchor=MSO_ANCHOR.TOP, fill=None, line=None, name=""):
    shp = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if fill: shp.fill.solid(); shp.fill.fore_color.rgb = rgb(fill)
    if line: shp.line.color.rgb = rgb(line); shp.line.width = Pt(2)
    tf = shp.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.12)
    if isinstance(paras, str): paras = [paras]
    total = 0
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        sz, b, col = size, bold, color
        if isinstance(p, tuple): p, opts = p; sz = opts.get("size", sz); b = opts.get("bold", b); col = opts.get("color", col)
        r = para.add_run(); r.text = p; r.font.size = Pt(sz); r.font.bold = b; r.font.color.rgb = rgb(col); r.font.name = "Calibri"
        if align: para.alignment = align
        para.space_after = Pt(6)
        total += est_lines(p, w - 0.3, sz) * sz * 1.2 + 6
    if total / 72 > h + 0.05:
        WARN.append(f"possible overflow slide {len(prs.slides)} box '{name or str(paras[0])[:30]}': need {total/72:.2f}in have {h}in")
    return shp


def new_slide(title, notes, outline_lines):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12)); bar.fill.solid(); bar.fill.fore_color.rgb = rgb(TEAL); bar.line.fill.background()
    tb(s, 0.5, 0.3, 12.3, 1.1, title, size=30 if len(title) < 70 else 26, color=NAVY, bold=True, anchor=MSO_ANCHOR.MIDDLE, name="title")
    tb(s, 0.5, 7.05, 9, 0.35, "WM-N1 research update | 2026-10-05 | lower is better for all delay/regret/CRPS numbers (seconds)", size=11, color="888888")
    tb(s, 12.2, 7.05, 0.8, 0.35, str(len(prs.slides)), size=11, color="888888", align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    OUTLINE.append((len(prs.slides), title, outline_lines, notes))
    return s


def pic(s, path, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return s.shapes.add_picture(path, Inches(x), Inches(y), **kw)


def table(s, x, y, w, rows, col_w=None, size=15, header_fill=NAVY, row_h=0.42):
    nr, nc = len(rows), len(rows[0])
    gt = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr)).table
    if col_w:
        for i, cw in enumerate(col_w): gt.columns[i].width = Inches(cw)
    for r in range(nr):
        gt.rows[r].height = Inches(row_h)
        for c in range(nc):
            cell = gt.cell(r, c); v = str(rows[r][c]); cell.text = v
            cell.margin_left = cell.margin_right = Inches(0.06); cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]; f = p.runs[0].font; f.size = Pt(size); f.name = "Calibri"
            is_p = v.startswith("PENDING")
            cell.fill.solid()
            if r == 0: cell.fill.fore_color.rgb = rgb(header_fill); f.color.rgb = rgb("FFFFFF"); f.bold = True
            elif is_p: cell.fill.fore_color.rgb = rgb(AMBER_BG); f.color.rgb = rgb("7A4B00"); f.bold = True
            else: cell.fill.fore_color.rgb = rgb("FFFFFF" if r % 2 else LIGHT); f.color.rgb = rgb("222222")
            if c > 0 and r > 0 and not is_p and len(v) < 28: p.alignment = PP_ALIGN.CENTER
            cw = (col_w[c] if col_w else w / nc) - 0.15
            if est_lines(v, cw, size) * size * 1.2 / 72 > row_h + 0.05:
                WARN.append(f"possible table cell overflow slide {len(prs.slides)} r{r}c{c}: '{v[:30]}'")
    return gt


def pending_box(s, x, y, w, h, exp, what):
    tb(s, x, y, w, h, [(ptxt(exp), {"bold": True, "size": 24, "color": "7A4B00"}), (what, {"size": 16, "color": "7A4B00"})],
       fill=AMBER_BG, line=AMBER, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER, name="pending")


def takeaway(s, text, y=6.3, h=0.7):
    tb(s, 0.5, y, 12.3, h, text, size=20, color="FFFFFF", bold=True, fill=TEAL, anchor=MSO_ANCHOR.MIDDLE, name="takeaway")


def bullets(s, items, x=0.5, y=1.5, w=12.3, h=4.6, size=22):
    return tb(s, x, y, w, h, ["• " + i if not i.startswith(" ") else "    – " + i.strip() for i in items], size=size, name="bullets")


# ----------------------------------------------------------------- slides
def build_deck(F):
    sr, ds = N["slice_regret"], N["dspeed"]
    rp, cl, a12 = N["repl"], N["closed_loop"], N["a12"]

    # 1 title
    s = prs.slides.add_slide(BLANK)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height); bg.fill.solid(); bg.fill.fore_color.rgb = rgb(NAVY); bg.line.fill.background()
    tb(s, 0.7, 0.6, 12, 0.8, "WM-N1 research update | 2026-10-05", size=22, color="9FD3D8", bold=True)
    head = ("A world model that predicts how human drivers react to different CAV controllers lets an RSU pick the best controller "
            "for each situation, with reliable short-horizon predictions and calibrated long-horizon uncertainty")
    tb(s, 0.7, 1.6, 12, 3.4, head, size=34, color="FFFFFF", bold=True, name="headline")
    tb(s, 0.7, 5.3, 12, 1.4, ["Headline claim under test; status claim by claim on slide 14.", "Numbers come from verified reports only; amber boxes mark results still pending today."], size=20, color="CFE3EA")
    s.notes_slide.notes_text_frame.text = ("Open with the headline as a hypothesis, not a result. Four sub-claims: predicts reaction, picks best controller, reliable short horizon, calibrated long horizon. "
        "The deck walks through evidence for each and ends with an honest scorecard on slide 14.")
    OUTLINE.append((1, "Title + headline", [head], s.notes_slide.notes_text_frame.text))

    # 2 setting
    s = new_slide("Setting: a highway merge, mixed CAV/HDV traffic, an RSU that only sees what CAVs report",
        "Network is the preserved highway merge. HDVs keep IDM and are never commanded. The RSU gets timestamped CAV self state and locally detected neighbour reports, with receipt age and missingness. No true penetration or hidden driver type is used online. Candidate controllers: per-CAV native, PI, damping; group moving walls (gated group action, assembly rarely succeeds); direct-speed plans (WM-N1-DSPEED generator).",
        ["Merge scenario; CAVs + HDVs; RSU sees only CAV-reported state", "Per-CAV: native, PI, damping", "Group: moving walls", "Direct-speed plans"])
    pic(s, F["setting"], 0.4, 1.5, w=6.4)
    bullets(s, ["Candidate controllers", " per-CAV: native, PI, damping", " group: moving walls (gated; assembly rarely succeeds)",
                " direct-speed plans (per-CAV target-speed profiles, 120 s action window, then native)",
                "RSU input: timestamped CAV state + detected neighbours only",
                "CAVs are the only actuators; HDVs keep IDM"], x=7.0, y=1.5, w=6.0, h=4.7, size=20)
    takeaway(s, "Question: which controller should the RSU run for this traffic situation?")

    # 3 method
    s = new_slide("Method: roll the real controllers forward, let a learned head predict the delay change",
        "CAVs in the rollout run the real controller code. HDVs are IDM/MOBIL plus a neural residual. The effect head is anchored on the TRAIN climatology of the paired delay change (so an untrained head equals climatology) and adds a gated plan-aware residual. Plan summary: 10 s CAV plan in A9, 120 s in A11. A11 adds a within-decision ranking loss. Output: quantiles of the delay change versus native at 30/60/120/300/570 s.",
        ["Diagram: RSU state + candidate plan -> rollout (real CAV code, HDV IDM/MOBIL + residual) -> trajectories -> climatology-anchored plan-aware effect head -> delay-change quantiles", "Plan summary 120 s; ranking loss"])
    pic(s, F["model"], 1.4, 1.5, w=10.5)
    bullets(s, ["Climatology anchor: an untrained head reproduces the TRAIN marginal exactly (safe default)",
                "Plan summary (120 s) tells the head what the CAVs actually do; ranking loss trains the within-decision ordering"], y=5.5, h=1.4, size=18)

    # 4 planner
    s = new_slide("Planner PL01: check CAVs one at a time, then in interacting groups, then act",
        "PL01 (traffic_shepherd/planner/loop.py). Step 1 individually checks native/damping/PI per CAV with the others held. Step 2 groups neighbouring CAVs by measured interaction taken from step-1 predicted trajectories. Step 3 does a joint check for each group, including wall candidates. Step 4 applies the switch rule against the current plan. Step 5 combines decisions and applies the safety fallback. Re-decision every 60 s. A11 sits behind the planner Predictor interface.",
        ["5-step diagram: individual check -> grouping by measured interaction -> group joint check (incl. walls) -> switch rule -> combine + safety fallback", "60 s cadence; A11 as predictor"])
    pic(s, F["planner"], 0.35, 1.5, w=12.6)
    bullets(s, ["Owner requirement: evaluate a single CAV first, then check with its surrounding CAVs as a group",
                "Walls enter only at the group stage and only when feasible (assemble / hold / release lifecycle, safety abort)"], y=5.3, h=1.6, size=20)

    # 5 broken/fixed
    s = new_slide("What was broken, and what was fixed overnight",
        "Previous model M1FIX1 lost to climatology at every tau by up to 10x. Diagnosis: invalid comonotone paired-dV construction (same-level quantile difference), 1,320 of 2,153 TRAIN items had no outcome labels and only 161 PI-native pairs supervised the contrast. Fixes across A8-A11 as listed. A8 gave 6.6 vs 52.4 CRPS at 30 s (reported as 8x better).",
        ["Table: problem -> fix (labels, scorer, labelled pairs, loss scale, batching)"])
    rows = [["Problem found", "Fix"],
            ["Scorer: paired dV from same-level quantile difference (comonotone, invalid intervals)", "Independent inverse-CDF sampling (attempt 6)"],
            ["1,320 / 2,153 TRAIN items had no outcome labels", "Label coverage overlay: 1,308 labelled pairs (A8)"],
            ["Only 161 PI-native pairs supervised the contrast", "Direct-speed (+644 runs) and re-aligned damping (+263 runs): 2,113 pairs"],
            ["~1 pair per batch; absolute head biased ~30-40% high", "Pair-dense K=4 batching, climatology-anchored head, per-tau loss scale"],
            ["Eval slice scored only PI-vs-native", "All-controller slice: 8-9 candidates per decision"]]
    table(s, 0.5, 1.5, 12.3, rows, col_w=[6.6, 5.7], size=16, row_h=0.7)
    takeaway(s, "Result: A8 CRPS 6.6 s vs M1FIX1 52.4 s at 30 s (PI-native, 11 decisions)", y=6.0, h=0.7)

    # 6 noise floor
    s = new_slide("Is the effect real? PI's effect is 5-27x larger than a one-CAV nudge",
        "90 runs, 30 TRAIN worlds x (R0 true native, N1 one-CAV nudge, S1 seed+1). R0 reproduces native bit-exactly. S1 is bit-identical to R0 (RNG restored with the snapshot) so the seed floor is exactly zero. sd(PI-native)/sd(N1-native): 27.17 [16.32, 44.71] at 30 s, 9.42, 6.40, 20.82, 4.96 [3.00, 1724.41] at 570 s. Caveat: N1 spread is itself heavy-tailed at 570 s (sd 656 s) and frac>0 for PI falls to 0.467 at 570 s: PI effect is a larger spread, not uniformly positive.",
        ["Chart: sd ratio per tau with 90% CI, log scale", "All lower CI bounds > 1 (16.3, 5.1, 4.0, 12.2, 3.0)", "Seed change = exactly 0; simulator bit-exact"])
    pic(s, F["noise"], 0.4, 1.4, w=7.6)
    bullets(s, ["All 90% CIs lie above 1 at every horizon (lower bounds 16.3, 5.1, 4.0, 12.2, 3.0)",
                "Seed change after snapshot restore: exactly zero",
                "Simulator reproduces native runs bit-exactly",
                "Caveat: 570 s is heavy-tailed even for the nudge"], x=8.2, y=1.5, w=4.8, h=4.6, size=19)
    takeaway(s, "Long-horizon differences are real action effects, not amplified noise")

    # 7 predictability
    s = new_slide("Predictability from the t0 state: only a small short-horizon signal",
        "T4g, zero-simulation, TRAIN only: 161 PI-native pairs in 161 worlds, 50 observable t0 features, world-grouped 5-fold CV. Only HGB at 30 s beats climatology (dCRPS -1.54 [-2.40, -0.71]); at 300/570 s learned models are significantly worse than climatology; sign accuracy never beats climatology. Implication: the bar for the world model is per-tau climatology, and most achievable skill is the marginal shape.",
        ["Chart: CRPS climatology vs HGB per tau", "Only 30 s shows significant gain: -1.54 [-2.40, -0.71]", "Sign accuracy never beats climatology"])
    pic(s, F["t4g"], 0.4, 1.4, w=7.6)
    bullets(s, ["Only HGB at 30 s beats climatology: dCRPS -1.54 [-2.40, -0.71]",
                "At 300 / 570 s the learned model is significantly worse",
                "Sign accuracy never beats climatology (~0.6 marginal rate)",
                "Bar for the world model = per-tau climatology, not zero"], x=8.2, y=1.5, w=4.8, h=4.6, size=19)
    takeaway(s, "The t0 state alone says little; skill must come from modelling the plan's effect")

    # 8 progression
    s = new_slide("Model progression A8 to A11: ties climatology, best at 300 s, worst at 30 s",
        "PI-vs-native only, CAL-tune, 11 decisions. All models tie climatology at 60-570 s; A11 is best at 300 s (point 431.7 vs 528.2, CI [-266, 52]) but worst at 30 s (11.0 vs 5.3, significantly worse than climatology). Calibration (10-90 coverage) 0.82-1.00 for A10/A11. M1FIX1 was far worse at every tau.",
        ["Table: A8..A11 changes + checkpoints", "Chart: CRPS vs climatology per tau (log)", "A11 300 s: 431.7 vs 528.2, CI [-266, 52]"])
    rows = [["Model", "What changed", "Ckpt"]] + [[m["name"], m["change"], m["ckpt"]] for m in N["models"]]
    table(s, 0.4, 1.4, 6.3, rows, col_w=[0.8, 4.6, 0.9], size=12, row_h=0.8)
    pic(s, F["crps"], 6.8, 1.35, w=6.3)
    bullets(s, ["A11 at 300 s: 431.7 vs climatology 528.2 (CI [-266, 52]); at 30 s: 11.0 vs 5.3 (significantly worse)"], x=0.4, y=5.4, w=6.3, h=0.9, size=16)
    takeaway(s, "No model clearly beats climatology on accuracy; coverage 0.82-1.00", y=6.4, h=0.6)

    # 9 decision quality
    s = new_slide("Decision quality: A11 beats always-native at 300 s on the all-controller slice",
        "CAL-tune sealed slice, 11 worlds x 5 taus = 55 decision points; 8-9 candidates per decision (native, PI, 4 direct-speed plans, fixed mix, switching mix, walls). Regret = seconds of total delay vs the best controller per decision. A11 never does worse than always-native (always-native is the TRAIN fixed-best) and beats it at 300 s: 972 vs 1556 s, 90% CI of difference [-1339, -3]; and at 570 s 5409 vs 6505, CI [-3288, 0]. Caveat: 11 worlds per tau; 20 model x tau cells were tested, so one significant cell needs replication before it is a claim. A11 was selected among A8-A11 on this slice.",
        ["Chart: regret per tau for A11/A10/A9/A8/always-native/random (log)", "300 s: A11 972.3 vs native 1555.8, 90% CI [-1339, -3]", "570 s: 5408.9 vs 6504.8, CI [-3288, 0]", "Caveat: 11 worlds, 20 model x tau cells tested -> needs replication"])
    pic(s, F["regret"], 0.3, 1.3, w=8.4)
    d3, d5 = sr["a11_vs_native_300"], sr["a11_vs_native_570"]
    bullets(s, [f"300 s: A11 {d3['a11']:g} vs native {d3['native']:g} s, 90% CI [{d3['ci90'][0]}, {d3['ci90'][1]}] (excludes 0)",
                f"570 s: {d5['a11']:g} vs {d5['native']:g} s, CI [{d5['ci90'][0]}, {d5['ci90'][1]}]",
                "30-120 s: A11 picks native (equal to always-native)",
                "Random choice costs most: most direct-speed plans add delay"], x=8.8, y=1.4, w=4.3, h=3.5, size=17)
    tb(s, 8.8, 4.9, 4.3, 1.3, ["Caveat: 11 worlds per tau; 20 model x tau cells tested. A11 was chosen on this slice: replication needed (slide 11)."],
       size=15, color="7A4B00", fill=AMBER_BG, line=AMBER, name="caveat")
    takeaway(s, "First decision win over the fixed-best baseline with CI excluding 0 - promising, not yet a claim")

    # 10 direct speed
    sm = ds["smoke"]
    s = new_slide("Direct-speed campaign: CAVs given arbitrary speed profiles, to teach the model HDV reactions",
        "WM-N1-DSPEED: per-CAV seeded piecewise target-speed profiles (random_target, slowdown_ramp, speedup_ramp, merge_aimed, gap_opening) over a 120 s action window, then every CAV returns to native. CommandOwnership is the sole writer; targets bounded to accel [-3, +1.5] m/s^2, jerk 1.5 m/s^3. Campaign: 644 TRAIN runs + 44 sealed CAL-tune runs, 0 failures. Documented smoke (one TRAIN world): 25 CAVs, 8 selected, 0 collisions, 0 teleports. Note: the morning summary says 0 collisions in smoke tests (plural); PART_A documents one smoke.",
        [f"{ds['train_runs']} TRAIN runs + {ds['cal_tune_runs']} sealed CAL-tune runs, {ds['failures']} failures", "Smoke: 25 CAVs, 8 selected, 0 collisions, 0 teleports", "A9 gain: +644 runs -> 1,952 pairs"])
    bullets(s, [f"{ds['train_runs']} TRAIN runs + {ds['cal_tune_runs']} sealed CAL-tune runs, {ds['failures']} failures",
                "Five profile types: random target, slowdown ramp, speed-up ramp, merge-aimed, gap opening; 120 s window, then native",
                "Bounded: accel [-3, +1.5] m/s^2, jerk 1.5 m/s^3; CommandOwnership is the sole writer",
                f"Smoke (1 TRAIN world, {sm['horizon_s']} s): {sm['cavs']} CAVs, {sm['selected']} selected, {sm['collisions']} collisions, {sm['teleports']} teleports; {sm['unit_tests']} unit tests pass",
                "Effect on the model: A9 = A8 + 644 runs (1,952 pairs); on the slice most plans add delay (random is costly)"], y=1.5, h=4.7, size=21)
    takeaway(s, "A wide action space: the planner can choose among plans, not just native vs PI")

    # 11 replication
    exp = rp["expected"]
    s = new_slide("TODAY: fresh-world replication of the A11 300 s win",
        f"Registration WM-N1-REPL-CL-20261005 Part R: 30 fresh worlds (seeds 930001-930030), 13 arms per world, 390 runs. Frozen primary hypothesis: A11 decision regret < always-native (TRAIN fixed-best) at tau=300 s on delay_s, world-bootstrap 90% CI of the difference below 0. Secondary: all taus; A10/A9. Result expected about {exp}. Fill numbers.json repl.* when available.",
        ["Frozen primary: A11 regret < always-native at 300 s, 90% CI below 0", f"Table with PENDING cells (expected {exp})"])
    sec = (ptxt(exp) if pending(rp["a10_regret_300"]) or pending(rp["a9_regret_300"])
           else f"{rp['a10_regret_300']:g} / {rp['a9_regret_300']:g}")
    rows = [["Quantity (s of total delay regret)", "Value"],
            ["A11 regret at 300 s (primary)", show(rp["a11_regret_300"], exp, "{:g}")],
            ["always-native regret at 300 s", show(rp["native_regret_300"], exp, "{:g}")],
            ["A11 minus always-native, 90% CI", show(rp["ci90_diff_300"], exp)],
            ["A10 / A9 regret at 300 s (secondary)", sec]]
    for t in ("30", "60", "120", "570"):
        rows.append([f"A11 vs always-native at {t} s (secondary)", show(rp["other_taus"][t], exp)])
    table(s, 0.6, 1.4, 12.1, rows, col_w=[6.6, 5.5], size=15, row_h=0.42)
    tb(s, 0.6, 5.3, 12.1, 0.6, "30 fresh worlds, new seed block, outcomes sealed until the registered scoring pass; no model choice on these worlds.", size=16)
    if pending(rp["verdict"]): pending_box(s, 0.6, 5.95, 12.1, 0.95, exp, "verdict: replicated / not replicated")
    else: takeaway(s, str(rp["verdict"]), y=5.95, h=0.95)

    # 12 closed loop
    exp = cl["expected"]
    s = new_slide("TODAY: closed-loop test in SUMO - does the planner reduce total delay?",
        f"Part C of WM-N1-REPL-CL-20261005 (+ Addenda 1 and 2). Same 30 fresh worlds, 600 s horizon. Policies: PL01+A11 (primary), fleet-WM (secondary), always-native, always-PI, random = 150 runs; plus PL01+A11 with wall-v2.2 group candidates (30 runs; hard cutoff: if the wall policy has not passed its smoke by 17:00 +0800 it is NOT run and reported as not done). Primary hypothesis (frozen): PL01+A11 total delay < always-native, world-bootstrap 90% CI of the difference below 0. Expected about {exp}.",
        [f"Table: total delay (s) per policy over [t0, t0+600], PENDING (expected {exp})", "Primary: PL01+A11 < always-native, CI below 0", "Walls: only if smoke passes by 17:00"])
    pol = [("PL01 + A11 (primary)", cl["pl01_a11"]), ("fleet-level WM (secondary)", cl["fleet_wm"]), ("always-native", cl["always_native"]),
           ("always-PI", cl["always_pi"]), ("random among candidates", cl["random"]), ("PL01 + A11 + walls (secondary)", cl["pl01_a11_walls"])]
    rows = [["Policy (30 fresh worlds, 600 s)", "Mean total delay, s"]] + [[k, show(v, exp, "{:g}")] for k, v in pol]
    rows.append(["PL01+A11 minus always-native, 90% CI", show(cl["diff_pl01_vs_native_ci90"], exp)])
    table(s, 0.5, 1.4, 8.2, rows, col_w=[4.8, 3.4], size=15, row_h=0.55)
    if F.get("closed"): pic(s, F["closed"], 8.9, 1.4, w=4.2)
    else: pending_box(s, 8.9, 1.4, 4.1, 2.2, exp, "chart appears when values are filled")
    tb(s, 8.9, 3.9, 4.1, 1.9, [cl["walls_cutoff_note"]], size=15, color="7A4B00", fill=AMBER_BG, line=AMBER, name="walls note")
    takeaway(s, "This is the result the headline needs: picking the controller must pay off in closed loop", y=6.1, h=0.8)

    # 13 A12
    exp = a12["expected"]; r = a12["ref"]
    s = new_slide("A12: fixing the 30 s regression without losing the 300 s gain",
        f"A12 = A11 + per-tau gate (one scalar gate moved all taus together), gate L2 0.1, rank loss only at tau >= 60 s. Same data, seed 101, 15x2000 steps. Registration WM-N1-A12-20261005. Code commit 82b75f8: 44 targeted tests pass, full-suite failure set identical to the A11 baseline (43 = 43), CPU dry run passed. GPU training status is not in the reports read; scores are expected about {exp}.",
        ["Changes: per-tau gate, gate L2 0.1, rank loss only >= 60 s", f"A11 reference: 30 s CRPS {r['a11_crps_30']} vs climatology {r['clim_crps_30']}; 300 s {r['a11_crps_300']} vs {r['clim_crps_300']}", f"A12 CRPS at 30 s and 300 s: PENDING (expected {exp})"])
    bullets(s, ["Per-tau gate: one scalar gate moved every horizon together; now one gate per tau (untrained head == climatology exactly)",
                "Gate L2 (0.1) shrinks gates toward climatology",
                "Ranking loss only at tau >= 60 s"], y=1.4, h=2.4, size=19)
    rows = [["CRPS, s (lower better)", "30 s", "300 s"],
            ["TRAIN climatology", f"{r['clim_crps_30']}", f"{r['clim_crps_300']}"],
            ["A11", f"{r['a11_crps_30']}", f"{r['a11_crps_300']}"],
            ["A12", show(a12["crps_30"], exp, "{:g}"), show(a12["crps_300"], exp, "{:g}")]]
    table(s, 0.6, 3.9, 12.1, rows, col_w=[3.3, 4.4, 4.4], size=18, row_h=0.55)
    takeaway(s, "Success = A12 near climatology (5.3) at 30 s, still near 431.7 at 300 s", y=6.3, h=0.65)

    # 14 honest status
    s = new_slide("Honest status against the headline, claim by claim",
        "Predicts reaction: partly - effect is real (noise floor 5-27x) but accuracy only ties climatology at 60-570 s and A11 is worse at 30 s. Picks best: promising - 300 s win with CI [-1339, -3] on 11 worlds, 20 cells tested, replication pending today. Short-horizon reliable: not yet - A11 worst at 30 s (11.0 vs 5.3); A12 pending. Calibrated long horizon: yes - 10-90 coverage 0.82-1.00 for A10/A11.",
        ["Predicts reaction = PARTLY", "Picks best = PROMISING, pending replication", "Short-horizon reliable = NOT YET (A12 pending)", "Calibrated long-horizon uncertainty = YES"])
    rows = [["Headline claim", "Status", "Evidence"],
            ["Predicts how HDVs react to controllers", "PARTLY", "Effect real (5-27x noise floor); accuracy ties climatology at 60-570 s"],
            ["Lets the RSU pick the best controller", "PROMISING", "300 s regret 972 vs 1556 s, CI [-1339, -3]; replication pending"],
            ["Reliable short-horizon predictions", "NOT YET", "A11 worst at 30 s (CRPS 11.0 vs 5.3); A12 pending"],
            ["Calibrated long-horizon uncertainty", "YES", "10-90 coverage 0.82-1.00 (A10/A11)"]]
    gt = table(s, 0.5, 1.5, 12.3, rows, col_w=[4.1, 1.9, 6.3], size=17, row_h=0.9)
    colmap = {"PARTLY": "FCE5B0", "PROMISING": "CFE8D8", "NOT YET": "F6C9C9", "YES": "A9DCB8"}
    for r_ in range(1, 5):
        c_ = gt.cell(r_, 1); c_.fill.solid(); c_.fill.fore_color.rgb = rgb(colmap[rows[r_][1]]); c_.text_frame.paragraphs[0].runs[0].font.bold = True
    takeaway(s, "One claim supported, one partly; two hinge on results arriving this afternoon", y=6.2, h=0.7)

    # 15 next steps
    s = new_slide("Next steps",
        "From the morning summary and the registration: replicate (running), closed loop (running), A12 short-horizon fix (running), add fixed/switching/trajectory contrasts to the slice CRPS table (scorer gap). Walls only if feasibility and benefit evidence supports them.",
        ["Replicate on fresh worlds", "Closed-loop planner test", "A12 30 s fix", "Add fixed/switching/trajectory contrasts to slice CRPS table"])
    bullets(s, ["Replicate the A11 300 s win on fresh worlds (today, frozen primary test)",
                "Closed-loop test: PL01 + A11 vs always-native / always-PI / random / fleet-WM (today)",
                "Fix the 30 s accuracy regression: A12 per-tau gates; score on CAL-tune, slice and fresh worlds",
                "Close the scorer gap: add fixed / switching / trajectory contrasts to the slice CRPS table",
                "Walls: report as secondary; evaluate only if the smoke passes (cutoff 17:00)",
                "Decide next training round only after replication and closed-loop results"], y=1.5, h=4.7, size=23)
    takeaway(s, "Decision point tonight: does the planner win in closed loop on fresh worlds?")


def write_outline():
    out = ["# WM-N1 update deck - outline and speaker notes (generated by build_deck.py)\n"]
    for n, title, lines, notes in OUTLINE:
        out.append(f"## Slide {n}: {title}\n")
        out += [f"- {l}" for l in lines]
        out.append(f"\n**Notes:** {notes}\n")
    open(os.path.join(HERE, "OUTLINE.md"), "w", encoding="utf-8").write("\n".join(out))


if __name__ == "__main__":
    F = build_figs(); build_deck(F)
    prs.save(os.path.join(HERE, "WM-N1_update_20261005.pptx")); write_outline()
    print("slides:", len(prs.slides)); print("\n".join(WARN) or "no overflow warnings")
