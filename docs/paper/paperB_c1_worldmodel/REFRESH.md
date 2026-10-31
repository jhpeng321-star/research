# Paper B — source refresh and end-of-night handoff

## Current package cut

First-package source refresh is recorded at **2026-10-31 16:15:03 +08:00**; see `SOURCE_REFRESH.json` for the exact paths/hashes and later refresh timestamps. This is an afternoon source cut, not a claim to have inspected future evening results. <!-- Source: docs/paper/paperB_c1_worldmodel/SOURCE_REFRESH.json, checked_at. -->

### Updated from PENDING / stale summaries

- **Recovered wall evidence:** original report and recovery receipt now support paired uncertainty; the older main checklist's “secondary source only / no CI” restriction is superseded within historical teacher-forced scope. No new inference was run by S-PAPERB.
- **Paired short-clock regret:** the existing current E1 CSV/Markdown includes an explicit strict-history team WM-minus-BB regret contrast. The package now quotes that contrast and plots it separately from marginal regret bars; its value was not estimated by subtracting CI endpoints. See C1-MAPPO and `data/regret_paired.csv`.
- **Exposure and independent reconstruction landed:** `E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md` is the actual delivered path, while the task-named root file is absent. Recovery, audited-corpus algorithm exclusion and independent paired reconstruction are now supported. The report documents a remaining initialization-lineage gap and final recommendations in progress. The ledger also adds its inconclusive endpoint/long-clock regret contrasts and correction to historical wall-regret interpretation. A byte-preserved audit snapshot is in `data/c1gap.source.md`.
- **Repair successor:** the data-only successor report exists and is awaiting resources. Its readiness does not change the failed pilot and smoothing verdicts. It remains PENDING efficacy.
- **Snapshot consistency:** all numerical figure inputs match the current external files at this cut; hashes are in `SOURCE_REFRESH.json` and `data/SOURCE_MANIFEST.json`.

### Still pending, with exact owner/agent dependency

- `[[PENDING: completed feedback report E:/agent-work/c4-c1closed/C1CLOSED.md or docs/paper_results/C1CLOSED.md, existing CSV exports and checkpoint/controller/plant/entrant bindings / S-C1CLOSED]]`.
- `[[PENDING: S-C1GAP's original selection team-cost binding, exact direction-panel binding, final recommendations and unbroken initialization lineage / S-C1GAP]]`. Independent paired reconstruction and the world-bootstrap recipe are now available.
- `[[PENDING: completed E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md, full grouped-CV selection and independent transfer results / S-WMCALIB]]`. Handoff/resume files explicitly say old small-panel outputs are smoke artifacts.
- `[[PENDING: data-only response-head successor's complete accuracy verdict / S-WMFIXMERGE]]`. Retain failed attempts even if a successor improves.
- `[[PENDING: owner prose, authors, conference template/CFP confirmation and compiled page count / owner]]`.

## End-of-night refresh procedure

This package includes a bounded source inventory command for the requested later check. Run from `E:/agent-work/c4-paperb`:

```powershell
python -B "docs/paper/paperB_c1_worldmodel/refresh_sources.py"
```

Then semantically reread newly present/changed reports using their exact paths. The inventory reads the task's fixed safe allowlist; it never starts agents, training, GPU inference, simulator work, monitors or remote jobs, and never automatically promotes a claim.

- Inspect the C1 gap report's latest conclusion, not only its presence; preserve any lineage or comparability caveats. A completed numerical reanalysis can resolve a numerical PENDING field without completing controller exposure proof.
- Read the feedback report before filling the outline's headline paragraph/table. Require that originals and entrants actually use the claimed feedback/plant/lane contract. Historical recorded-motion evidence must remain labeled historical.
- Read completed calibration only after its completeness checks and frozen selection; distinguish raw-WM and arm-mean comparators, independent transfer, reference arms and cost definitions. Do not use old smoke results.
- Recheck the model plan/change log, E1/E2 exports, corrected qualification and entry retarget reports. Never silently transfer the frozen direction envelope to a different model or population.
- If numerical figure inputs changed, review their provenance and metrics first, then run the explicit `build_figures.py --refresh-inputs` command. Otherwise reuse frozen local inputs. New result CSVs need an explicit allowlisted rendering change, not a directory crawl.
- Update `CLAIMS.md`, affected outline paragraphs/captions, and this handoff; retain failed and missing outcomes. Every numerical insertion needs an adjacent HTML source comment. Mark each remaining missing result `[[PENDING: what / agent]]`.
- Run `python -B "docs/paper/paperB_c1_worldmodel/validate_package.py"`, inspect any changed PNGs, inspect status/diff/recent commits, and commit only package files. Source snapshots are byte-preserved by local `.gitattributes`; the data directory is intentionally tracked despite the repository's broad local exclusion.
- The requested first-package registration is append-once; do not append a duplicate completion line during an ordinary refresh. A materially changed scientific verdict should get its own clearly labeled update by the integrating owner.

## First-package verification scope

`VERIFICATION.json` records artifact checks: exact E1/E2 selection and paired contrast, source byte hashes, transcribed report cells, separate source clocks, per-row provenance, PNG/PDF integrity, bibliography keys/fields and numerical source comments. All PNGs were visually inspected, including the revised paired-regret plot and repaired schematic.

This verifies a drafting package. It does not independently reproduce the source experiments, compile an IEEE manuscript, validate formal admissibility, or complete missing agent results. No prohibited input was opened and no experimental run was started by the paper agent.

## Registration

The main requested shared `REGISTRATION.md` is under an ignored `runs/` directory and is not tracked by the main Git index. Append the completion line there without importing the existing shared log into this branch. The same line is preserved in `REGISTRATION_RECEIPT.md` and committed with the package, so the paper task's completion is versioned without absorbing unrelated agents' history.
