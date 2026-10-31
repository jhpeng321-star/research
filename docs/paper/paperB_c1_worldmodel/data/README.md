# Figure data and provenance

The builder preserves the complete allowed E1/E2 source CSVs and extracts exact named rows into plot CSVs. Regional, entry and historical-wall cells are extracted from existing Markdown reports, without new model execution or statistical inference. `SOURCE_MANIFEST.json` binds source bytes by SHA256 and stores source paths. Every numerical plot CSV includes a source path and selector. Summary-report extraction does not recover raw observations or create unavailable uncertainty.

The evening builder adds `EVENING_SOURCE_MANIFEST.json`, byte-preserved diagnosis/calibration/switching/pulse sources, and source-selected `pi_diagnosis.csv`, `calibration.csv`, `switch_headroom.csv`. Winner counts are descriptive summaries of the existing exported candidate costs, with ties separate; no new experiment or uncertainty calculation. `floor.source.csv` preserves unavailable endpoint cells and original status fields. Consult `../FIGURES.md` for captions and distinct panel contracts.

The overnight builder adds `OVERNIGHT_SOURCE_MANIFEST.json`, exact completed
feedback/distribution CSVs and report snapshots, and selected `closed_loop.csv`,
`distribution.csv`, `value_tail.csv`. Existing source estimates/intervals are
selected or transcribed without resampling. Feedback paired contrasts are not
differences of marginal medians; sparse/missing endpoint clocks retain status and
denominator. Distribution absolute/effect panels use explicitly different baseline
columns. Value-tail A remains unavailable and is never manufactured. The partial
reference switching report is retained separately from the pending main study.
