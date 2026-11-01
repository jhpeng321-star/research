# Paper B — owner-review Overleaf draft

The latest owner clarification is implemented: **all main prose, including Results,
is written from existing evidence; graphs alone are empty**. The two figure boxes
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
`C:/Users/jpeng/Desktop/PaperB_draft_v1.pdf`. `DRAFT_TEXT.md` is generated from the
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
descriptive source intervals, not multiplicity-adjusted. All empirical numerical
sentences in `main.tex` have an adjacent `% Source:` comment. Repeated numbers in
the Abstract and Setup share the same source as their Results occurrence. Minus
contrasts consistently mean WM minus comparator unless an effect is explicitly
controller minus native. Decimal precision is retained as printed in the named
source; no inferred paired CI is used.

- [ ] **Wall** — Abstract, IV-A, V-A: 60 s; 67 starts/29 worlds; errors
  8.71/25.41/40.07 m; paired -9.98 [-16.78,-5.09] m.
  `E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md:8,27`.
  Historical teacher forcing, entries OFF, seen-world other-arm training.
- [ ] **Later wall** — V-A: 120 s; paired -49.21 [-110.05,-7.31] m.
  `E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md:38`.
  Different surviving starts; not an autonomous wall result.
- [ ] **PI adverse original starts** — V-A: 60 s; +17.85 [3.24,31.27] m.
  `E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md:14`.
- [ ] **PI shifted genuine history** — V-A: 60 s history; 60 s forecast;
  -45.05 [-52.73,-25.25] m; within-identical-shifted-start history contrast
  -13.47 [-26.92,-1.49] m.
  `E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md:25-37`.
  Conditional presence and CAV teacher, entries OFF; shift changes the cohort.
- [ ] **Completed feedback setup** — IV-A: 40 development worlds, 60 s histories.
  `E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED.md:2-6`.
- [ ] **Completed feedback endpoints** — V-A: 60 s PI -5.17 [-18,7.09] m;
  MAPPO -3.45 [-13.4,4.58] m; no common endpoint cohort at 300 s.
  `E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED.md:19,23`.
  Source Markdown precision retained; both contrasts include zero.
- [ ] **Frozen direction panel** — V-B: PASS 10/30/60/90 s; FAIL 120/180 s;
  accuracy 90 s 0.8895 [0.8424,0.9341], 120 s 0.7526 [0.6824,0.8187].
  `E:/research-dense/docs/paper_results/E2_direction_horizon.md:5-10`.
  Exported verdicts; exact original checkpoint/rule binding incomplete.
- [ ] **Observable-start SUMO** — V-B: 120/300/600 s; inferred delay CRPS
  647/4635/16728, climatology 1433/4375/9131 veh-s; 18/144 futures with
  initialization collisions.
  `E:/agent-work/c4-sumocore/docs/paper_results/SUMOCORE.md:12,19,23`.
  Separate diagnostic, not a clean information ceiling or full-state oracle.
- [ ] **Configured RNG ensemble** — V-B: 600 s; maximum within-state/arm delay
  standard deviation 0 veh-s.
  `E:/agent-work/c4-ensprobe/docs/paper_results/ENSPROBE.md:7,17`.
  Fixed driver population/requests and restored starts; not real-world determinism.
- [ ] **Rebuild sample** — IV-A: 20-world campaign, quoted reserved 8-world
  comparison, 5 arms, 8 futures/arm.
  `E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:18-25`.
- [ ] **Rebuild CRPS** — V-C: 120/300/600 s; WM2 1499/6817/24752;
  FULL-v4 1105/3777/13105 veh-s; independently exported paired differences
  -393/-3040/-11647 veh-s. Rounded means need not subtract to rounded contrasts.
  `E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:24-25,41`.
- [ ] **Rebuild endpoint/adoption** — V-C: 60 s -15.26 m; 300 s
  breakdown-rate-distance change +5.63 percentage points; final rule FAIL.
  `E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:42-47`.
  All registered full-world CIs NA; earlier interim numerical CIs retracted.
- [ ] **Rebuild controller effects** — V-C: 600 s, predicted/recorded
  MAPPO-native -1246/-2014, PI-native -1674/-1131,
  CIRCLES-native +4583/+3746 veh-s; mean delay bias +14648 veh-s.
  `E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:26,30-32`.
  Point-mean signs only, not calibrated action choice.
- [ ] **Action overconfidence** — V-C: selection probability >=0.8;
  120/300/600 s observed 52.9/55.6/70.0%, forecast 92.6/92.4/93.1%.
  `E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:50-52`.
  Retrospective hypothetical-incumbent comparisons, not live switching.
- [ ] **Lane precision** — V-C: default 3.006% precision, 37.942% recall.
  `E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:34`.
  Separate conditional diagnostic, not the reserved adoption cohort.
- [ ] **Scenario winners** — V-D: 90 worlds, 600 s; native 22, CTRL0 17,
  MAPPO 19, PI 29, CIRCLES 3.
  `E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH_WINNERS.md:5-8`.
  Fixed episodes; not same-moment switching performance.
- [ ] **Strict-history selection** — V-D: 60 s, 32 starts/21 worlds;
  WM-selected minus always-MAPPO -8.241 [-18.99,-0.08363] veh-s.
  `E:/research-dense/docs/paper_results/E1_C1_fair.md:197`.
- [ ] **Paired offline regret** — V-D: -8.41 [-19.23,-0.31] veh-s.
  `E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md:29`.
  Exploratory exported team objective, exact historical plant/cost binding incomplete.
- [ ] **PARTIAL switching** — V-D: 5 low-penetration worlds, 15 measured
  decisions/model; 60 s WM-BB +5.102 [-9.389,+16.746]; WM-best-fixed
  +8.232 [+1.692,+18.477] veh-s; best-fixed identity reselected per bootstrap draw.
  `E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH.md:41-51`.
  Scheduler-selected complete intersection, not full-panel efficacy or null.
- [ ] **Pending outcomes** — Table I contains only [--]; no invented switching-v2
  or successor-v5 performance. Protocol sources:
  `E:/agent-work/c4-paperb/docs/paper/paperB_c1_worldmodel/CLAIMS.md:C1-SWITCH2,WM-PHYS`.
- [ ] **Identity/definition tokens** — WM2, WM-13E, FULL-v4, LC2013 and GAIA-1
  are version/model names, not measured values; their identities follow the source
  reports/verified bibliography. Subscripts, equation numbering and CRPS's one-half
  are mathematical definitions. Figure box height is a layout choice.
- [ ] **Submission metadata** — replace author, affiliation and acknowledgment;
  confirm IV 2027 CFP, anonymity, paper size and six-page inclusive limit with owner.
  Final page count is a local build fact in `BUILD_REPORT.json`, not an experiment.

## Rebuild the delivery artifacts

Run `python build_delivery.py prepare` before compilation to copy only cited
verified bibliography entries and the explicitly named optional existing artwork.
After compilation run `python build_delivery.py deliver` to generate the matching
Markdown, inspect the final PDF/log, copy the Desktop PDF, and build the ZIP.
The script reads only this project, the named bibliography/figure assets and the
Desktop destination. It starts no scientific job and removes no files.
