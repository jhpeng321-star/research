# Paper A — professor-review / Overleaf outline

Open `main.tex` as the root document in Overleaf (pdfLaTeX), or read
`OUTLINE_IEEEIV.md` for the paragraph-level writing plan and actual source paths.
`refs.bib` is a verified subset of the inherited Paper A bibliography.

## Build and check

From this directory in PowerShell:

```powershell
& "C:/Users/jpeng/AppData/Local/Programs/MiKTeX/miktex/bin/x64/latexmk.exe" -pdf -interaction=nonstopmode -halt-on-error main.tex
python -B validate_outline.py
```

The QA helper requires the already-installed PyMuPDF and Pillow packages. It
reads only explicitly named producer reports, this package and the inherited
bibliographic verification receipt. `BUILD_QA.json` records source/artifact
hashes, citation identity checks, page bounds, placeholders and compilation QA.
`qa_render/` contains local page images and a contact sheet for visual review.
Compilation intermediates and renderings stay local.

## Deliverables and evidence boundary

- `main.pdf` is the compiled IEEEtran conference outline. The review build has
  **five pages including references**; the paragraph outline allocates **six
  pages including references** for the completed paper. The remaining space is
  available for pending results and final visual panels. These are editorial
  counts, checked by the PDF QA receipt, not experimental results.
- Desktop delivery: `C:/Users/jpeng/Desktop/PaperA_outline_v1.pdf`.
- Red `[PENDING: ...]` markers identify missing evidence or unresolved method
  details. Red `[CITE NEEDED: ...]` markers retain unverified Dreamer and
  dedicated platoon reference slots. COMA and Wolpert–Tumer are already verified.
- Empirical numbers have adjacent source comments in `main.tex` and source paths
  in `OUTLINE_IEEEIV.md`. The exact planner's positive delay result is distinct
  from its inconclusive pooled mean synergy and from pending learned-policy
  outcomes. Cached S-BDRANK action rankings arrived during drafting and are
  included with their limited scope.
- Two contract issues remain explicit: coalition-alone synergy versus the older
  producer's member-revert marginal credits, and the latest owner-approved
  multi-horizon rule versus the earlier learner horizon. The package does not
  silently label the earlier design as the finalized method.

The working title and main story follow the owner's latest outline brief.
The inherited teacher–student outline is historical context; the new manuscript
centres cooperation emerging from MAPPO through world-model replay credit.
