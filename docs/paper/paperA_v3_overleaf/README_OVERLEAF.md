# Paper A v3 — Overleaf package

**Status (2026-11-02):** compiles cleanly with pdfLaTeX + BibTeX (MiKTeX, `latexmk -pdf`); **6 pages incl. references** (page 6 about half full). No undefined citations or references; no BibTeX warnings.

## Upload
1. Overleaf -> New Project -> Upload Project -> select `../paperA_v3_overleaf.zip` (contains `main.tex`, `refs.bib`, `figures/`).
2. Menu -> Settings: Compiler = **pdfLaTeX**, Main document = `main.tex`. Overleaf runs BibTeX automatically (`IEEEtran.bst`).
3. Recompile. Local build: `latexmk -pdf main.tex`. Fig. 2 rebuild: `python figures/build_fig2.py` (rl_driving env; reads the read-only coopplan/coopmeasure result files listed in the script header).

## Red `\tbf` items still [to be filled in] (14 uses)
- Abstract (l.34): guided-policy result on the sealed 96-scenario set.
- Contributions C3/C4 (l.57-58): main comparison result; discovered group behaviors.
- Training (l.162): number of updates for R.
- E2 (l.189): predicted vs realized advantage of executed plans; advantage over own actions; replay-credit vs team-return variance.
- E3 (l.192) text and Table I (l.203-206, all 20 cells: B-MAPPO, B-A, B-R, A-MAPPO).
- E4 (l.219), E5 (l.229).
- Discussion (l.234) and Conclusion (l.243) result sentences.
- Fig. 3 (learning curves + trust) and Fig. 4 (space-time diagram) are still `\fbox` placeholders.
- Acknowledgment: "To be added."

## Blue `\tbc` values to confirm against registered design / network files
- Return time constant "about 400 s" and AR(1) exploration noise (l.121).
- Screen size K=4 and 3 kept plans (l.134); k=4 (l.138, CEM/MPPI refinement step).
- "About 3.2 km of controlled road; lane counts to confirm" and 100 m sensing (l.150).

## References
- 16 entries added 2026-11-02; each verified from Crossref DOI metadata, PMLR, NeurIPS, IFAAMAS, ICLR or arXiv pages (source URL in a comment above each entry). **No UNVERIFIED entries.**
- `toghi_social2022` is the arXiv 2211.11963 paper whose first author is Valiente (Toghi second); key kept as cited.
- `zhang_marie2024` cited as arXiv (arXiv comment says accepted by TMLR; journal version not inspected).
- `anthony_exit2017` has no page numbers (NeurIPS official BibTeX gives none). AAMAS entries carry IFAAMAS URLs, not ACM 10.5555 pseudo-DOIs.
- Pre-existing note in `refs.bib`: Dreamer and a dedicated platoon reference remain CITE NEEDED (not cited in the current text).

## Known layout issue
- Table I is ~80 pt too wide only because each cell holds the red placeholder text; it fits once numbers are filled in.
