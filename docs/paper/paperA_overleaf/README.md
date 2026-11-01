# Paper A — professor-review / Overleaf outline v2

Open `main.tex` as the root document in Overleaf (pdfLaTeX), or read
`OUTLINE_IEEEIV.md` for the paragraph-level writing plan and actual source paths.
`refs.bib` is a verified subset of the inherited Paper A bibliography.

## Build and check

From this directory in PowerShell:

```powershell
python -B deliver_v2.py --preserve-v1
& "C:/Users/jpeng/AppData/Local/Programs/MiKTeX/miktex/bin/x64/latexmk.exe" -pdf -interaction=nonstopmode -halt-on-error main.tex
python -B validate_outline.py
python -B deliver_v2.py --deliver
```

The QA helper requires the already-installed PyMuPDF and Pillow packages. It
reads only explicitly named producer reports, this package and the inherited
bibliographic verification receipt. `BUILD_QA.json` records source/artifact
hashes, citation identity checks, page bounds, placeholders and compilation QA.
`qa_render/` contains local page images and a contact sheet for visual review.
Compilation intermediates and renderings stay local.

## Deliverables and evidence boundary

- `main.pdf` is the compiled IEEEtran conference outline. `BUILD_QA.json` records
  the actual review page count; the paragraph outline allocates **six
  pages including references** for the completed paper. The remaining space is
  available for pending results and final visual panels. These are editorial
  counts, checked by the PDF QA receipt, not experimental results.
- Desktop delivery: `C:/Users/jpeng/Desktop/PaperA_outline_v2.pdf`.
  `PaperA_outline_v1.pdf` is preserved byte-for-byte; `V2_DELIVERY.json` records
  both hashes and verifies that v2 matches `main.pdf`.
- Red `[PENDING: ...]` markers identify missing evidence or unresolved method
  details. Red `[CITE NEEDED: ...]` markers retain unverified Dreamer and
  dedicated platoon reference slots. COMA and Wolpert–Tumer are already verified.
- Empirical numbers have adjacent source comments in `main.tex` and source paths
  in `OUTLINE_IEEEIV.md`. The exact planner's positive delay result is distinct
  from its inconclusive pooled mean synergy and from pending learned-policy
  outcomes. S-BDRANK includes option-delay concordance at both requested
  horizons and the separate controller breakdown-probability tie count.
- The final-test registration metadata, primary-first order and two later
  replications are included. Only public protocol metadata is consumed.
- Open implementation slots include the actual proposal-mixture likelihood
  (including deterministic atoms), adaptive trust rule and coalition-alone
  diagnostic exports. The inspected producer design still describes the older
  reward-credit arm; it is historical authority for shared architecture/panels,
  not evidence that the redesigned method has run.

The working title is **World-Model-Guided Exploration for Learning Multi-CAV
Cooperation**. The latest owner redesign (MAIN17:12/17:14) controls: the WM
proposes likely-winning moves, SUMO supplies all reward/advantage, and trust
decays adaptively as the policy catches up. Importance-corrected PPO and
trust-weighted imitation are explicit. E2 replay fidelity/noise remains a
diagnostic; the reward-credit ramp and horizon gate are superseded. Results
must distinguish teacher-assisted performance from cooperation persisting
after trust reaches zero. Red pending slots carry all missing outcomes.
