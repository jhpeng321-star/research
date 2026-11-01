# Paper B — owner-review Overleaf draft

The latest owner clarification is implemented: **all main prose, including Results,
matches the confident owner-review Doc export `DOC_CONFIDENT_v2.md` (revision 11);
graphs alone are empty**. The two figure boxes
describe the intended panels. The switching-v2 comparison has an empty outcome
table. Red PENDING text marks the unavailable complete picker comparison and
independently qualified successor-v5 results. Author/affiliation/funding fields
need the owner's information.

## Upload and compile

1. Upload `paperB_overleaf.zip` using Overleaf **New Project → Upload Project**.
2. Set `main.tex` as the main document and select **pdfLaTeX**.
3. Recompile; Overleaf handles BibTeX. The project uses the actual `IEEEtran`
   conference class and IEEE numbered bibliography style.

Local compile (run inside this folder, with MiKTeX on PATH):

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Equivalent explicit sequence, used if latexmk/Perl is unavailable:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The delivery build records pagination, warnings and file hashes in
`BUILD_REPORT.json`. The owner-review PDF is also copied to
`C:/Users/jpeng/Desktop/PaperB_draft_v2.pdf`; the v1 Desktop PDF is preserved.
`DRAFT_TEXT.md` is generated from the
same manuscript, including its results, captions, pending table and reference
text; adjacent TeX source comments become Markdown HTML comments.

The submission target is **IEEE IV 2027, six pages including references** (owner
brief). The standard IEEE conference structure is present. Exact conference CFP,
paper-size and anonymity requirements still require owner confirmation; no
conference-specific template metadata has been invented. Layout dimensions and
equation constants are typesetting/mathematical definitions, not measurements.

## Figures and references

`figures/` carries copies of the existing architecture, wall, direction,
closed-loop, regret and scenario-opportunity artwork as optional replacement
assets. They are **not inserted in the PDF**, following the latest owner request.
Do not insert them without preserving checkpoint/protocol/cost labels. Current
rebuild results are described in prose; older graphics are not substitutes for
the final rebuild panel. No figures were regenerated and no experiment was run.

`refs.bib` contains only the cited entries copied from the verified pool in
`../paperB_c1_worldmodel/refs.bib`. Their verification and permitted comparison
scope are in `../paperB_c1_worldmodel/RELATED_WORK.md`. Existing verified preprints
remain preprints. The numerical years, volumes, pages, DOI/arXiv identifiers and
model names in the references are bibliographic/identity metadata, not empirical
claims. The Markdown bibliography is the actual compiled IEEE reference text.

## Number-source CHECKLIST

Check each item against the named report before submitting. Intervals are
descriptive source intervals, not multiplicity-adjusted. Empirical numerical
sentences in `main.tex` have an adjacent `% Source:` comment, including repetitions
in the Abstract, contributions, captions and Conclusion. The owner's requested
rounding is retained; exact source precision and derivations are identified below.
Positive improvements negate source error contrasts **and reverse CI endpoints**.
No CI is inferred from marginal medians or independent interval subtraction.

- [ ] **Wall** — Abstract, IV-A, V-A: 60 s; 67 starts/29 worlds; errors
  8.71/25.41/40.07 m; paired WM-minus-BB-traj -9.98 [-16.78,-5.09] m and
  WM-minus-BB-id -31.78 [-45.37,-18.61] m; 95% intervals.
  `E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md:4,8,27`.
  Historical teacher forcing, entries OFF, seen-world other-arm training.
- [ ] **Wall ratio summary** — I, V-A, VII: 3–5 times is the owner's rounded
  whole-number summary of the marginal ratios 25.41/8.71 = 2.917 and
  40.07/8.71 = 4.600. Same wall source:8,27. It is not a paired effect ratio.
- [ ] **Later wall** — Abstract, I, V-A: 120 s; improvement
  49.21 [7.31,110.05] m = negated paired -49.21 [-110.05,-7.31] m.
  `E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md:38`.
  Different surviving starts; not an autonomous wall result.
- [ ] **PI shifted genuine history** — Abstract, III-C, V-A: 60 s history;
  60 s forecast; improvement 45.05 [25.25,52.73] m = negated
  -45.05 [-52.73,-25.25] m. Abstract rounds 45.05 to 45 m. Within-identical-
  shifted-start improvement 13.47 [1.49,26.92] m negates
  full-history-minus-snapshot -13.47 [-26.92,-1.49] m.
  `E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md:25-37`.
  Conditional presence and CAV teacher, entries OFF; shift changes the cohort.
- [ ] **Completed feedback setup** — IV-A: 40 development worlds, 60 s histories.
  `E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED.md:2-6`.
- [ ] **Completed feedback endpoints/delay** — V-A: 60 s PI -5.17 m;
  MAPPO -3.45 m (source intervals [-18,7.09] / [-13.4,4.58] include zero).
  MAPPO 60 s J absolute-error difference -116 [-375,-11] veh-s rounds the
  source Markdown -116 [-375,-11.1] to whole veh-s.
  `E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED.md:19,27`.
- [ ] **Frozen direction panel** — Abstract, V-B, Figure 2: 90 s horizon;
  10–60 s accuracy 0.92–0.95 rounds the 10/30/60 s values
  0.9211/0.9474/0.9526. At 90 s, 0.89 [0.84,0.93] rounds
  0.8895 [0.8424,0.9341]; at 120 s, 0.75 [0.68,0.82] rounds
  0.7526 [0.6824,0.8187]. Tested PASS 10/30/60/90 s, FAIL 120/180 s.
  `E:/research-dense/docs/paper_results/E2_direction_horizon.md:5-10`.
  Exported verdicts; exact original checkpoint/rule binding incomplete.
- [ ] **Observable-start SUMO** — V-B: at 120 s, inferred delay CRPS 647 versus
  climatology 1433 veh-s; at 600 s, 16728 versus 9131 veh-s. The 5–10 min
  range converts 300/600 s (source 300 s scores 4635 versus 4375).
  `E:/agent-work/c4-sumocore/docs/paper_results/SUMOCORE.md:12`.
  This separate diagnostic retains 18/144 initialization-collision futures
  (same report:19,23); it does not prove a clean best-achievable ceiling.
- [ ] **Configured RNG ensemble** — V-B: eight different seeds, 600 s;
  zero spread = maximum within-state/arm delay standard deviation 0 veh-s,
  with identical physical-tape hashes.
  `E:/agent-work/c4-ensprobe/docs/paper_results/ENSPROBE.md:4,7,17`.
  Fixed driver population/requests and restored starts; not real-world determinism.
- [ ] **Rebuild sample** — IV-A: 20-world campaign, quoted reserved 8-world
  comparison, 5 arms, 8 futures/arm.
  `E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:18-25`.
- [ ] **Rebuild CRPS** — Abstract, I, V-C, VII: at 2/5/10 min (120/300/600 s),
  WM2 1499/6817/24752 versus FULL-v4 1105/3777/13105 veh-s. Percent changes
  100*(new/old−1) = −26.284/−44.594/−47.055%, rounded to −26/−45/−47%.
  Abstract/contributions quote 26–47%; Conclusion quotes up to 47%.
  `E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:24-25,41`.
  Independently exported paired differences are −393/−3040/−11647 veh-s;
  rounding explains why the displayed means do not always subtract to them.
- [ ] **Rebuild endpoint/adoption** — V-C: at 60 s, improvement 15.26 m negates
  the source contrast −15.26 m. At 5 min (300 s), breakdown-rate-distance
  worsening 5.6 percentage points rounds +5.63 pp; final adoption rule FAIL.
  `E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:42-47`.
  All registered full-world CIs NA; earlier interim numerical CIs retracted.
- [ ] **Rebuild controller effects** — V-C: at 10 min (600 s), predicted/recorded
  reductions MAPPO 1246/2014, PI 1674/1131 veh-s negate their negative
  controller-minus-native means; CIRCLES increases are +4583/+3746 veh-s.
  `E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:30-32`.
  Point-mean signs only, not calibrated action choice.
- [ ] **Action overconfidence source (qualitative in v2)** — selection probability >=0.8;
  120/300/600 s observed 52.9/55.6/70.0%, forecast 92.6/92.4/93.1%.
  `E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:50-52`.
  Retrospective hypothetical-incumbent comparisons, not live switching.
- [ ] **Lane precision source (qualitative in v2)** — default 3.006% precision, 37.942% recall.
  `E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:34`.
  Separate conditional diagnostic, not the reserved adoption cohort.
- [ ] **Scenario grid** — IV-A: demand 1900–2700 veh/h, penetration 5–40%.
  `E:/agent-work/c4-c1switch2/docs/paper_results/C1SWITCH2_DESIGN.md:22-23`
  registers (1900,2400,2700) × (0.05,0.10,0.20,0.40); recorded-panel cell
  examples corroborate the bounds in
  `E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH_WINNERS.md:12`.
- [ ] **Scenario winners** — IV-A, V-D: 90 worlds, 10 min (600 s); native 22, CTRL0 17,
  MAPPO 19, PI 29, CIRCLES 3.
  `E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH_WINNERS.md:5-8`.
  Fixed episodes; not same-moment switching performance.
- [ ] **Strict-history selection** — V-D: 60 s, 32 states/21 worlds;
  improvement 8.24 [0.08,18.99] veh-s negates and rounds the source
  WM-selected-minus-always-MAPPO −8.241 [−18.99,−0.08363] veh-s.
  `E:/research-dense/docs/paper_results/E1_C1_fair.md:197`.
- [ ] **Paired offline regret** — V-D: improvement 8.41 [0.31,19.23] veh-s
  negates WM-minus-BB-traj −8.41 [−19.23,−0.31] veh-s.
  `E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md:29`.
  Exploratory exported team objective, exact historical plant/cost binding incomplete.
- [ ] **PARTIAL switching source (qualitative in VI)** — 5 low-penetration worlds, 15 measured
  decisions/model; 60 s WM-BB +5.102 [-9.389,+16.746]; WM-best-fixed
  +8.232 [+1.692,+18.477] veh-s; best-fixed identity reselected per bootstrap draw.
  `E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH.md:41-51`.
  Scheduler-selected complete intersection, not full-panel efficacy or null.
- [ ] **Pending outcomes/protocol** — red pending prose and Table I; only [--]
  outcome cells. Switching every 20 s among six controllers including wall is
  prospective, from
  `E:/agent-work/c4-c1switch2/docs/paper_results/C1SWITCH2_DESIGN.md:8-10,20-26`.
  Successor-v5 numbers and complete learned-picker comparisons remain pending;
  no training progress is substituted for a performance result.
- [ ] **Identity/definition tokens** — CTRL0, WM2, WM-13E, FULL-v4, v5, LC2013 and GAIA-1
  are version/model names, not measured values; their identities follow the source
  reports/verified bibliography. Subscripts, equation numbering and CRPS's one-half
  are mathematical definitions. Figure box height is a layout choice.
- [ ] **Submission metadata** — replace author, affiliation and acknowledgment;
  confirm IV 2027 CFP, anonymity, paper size and six-page inclusive limit with owner.
  Final page count is a local build fact in `BUILD_REPORT.json`, not an experiment.

**Source reconciliation:** no numerical quantity in the approved Doc is unsourced.
The confident text is the owner's wording; it does not revise the underlying
protocol receipts. Historical residual-only WM-S did not wire inferred settings
into physics, whereas later hybrid/rebuild variants do. Exact component attribution
needs matched ablations. Later model corpora differ from the frozen black boxes,
and copied initialization ancestry remains incompletely audited. The E2 horizon
does not automatically transfer to every checkpoint. Observable-start SUMO is an
informative diagnostic, not a proven information ceiling. Rebuild full-world CIs
are NA for singleton strata and its conjunctive adoption rule FAILS; switching
therefore retains WM2. The favorable historical selection contrast is exploratory
offline evidence, distinct from the separate 90-world scenario-winner panel and
from PARTIAL reference switching. These qualifications are preserved here and in
source comments for the owner's next scientific revision, without replacing the
approved main paragraphs. See `V2_RECEIPT.md`, `CLAIMS.md` and `THREATS.md` in the
source package for their exact provenance.

## Rebuild the delivery artifacts

Run `python build_delivery.py prepare` before compilation to copy only cited
verified bibliography entries and the explicitly named optional existing artwork.
After compilation run `python build_delivery.py deliver` to generate the matching
Markdown, inspect the final PDF/log, copy the Desktop PDF, and build the ZIP.
The script reads only this project, the named bibliography/figure assets and the
Desktop destination. It starts no scientific job and removes no files.
