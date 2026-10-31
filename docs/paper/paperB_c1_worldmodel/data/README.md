# Figure data and provenance

The builder preserves the complete allowed E1/E2 source CSVs and extracts exact named rows into plot CSVs. Regional, entry and historical-wall cells are extracted from existing Markdown reports, without new model execution or statistical inference. `SOURCE_MANIFEST.json` binds source bytes by SHA256 and stores source paths. Every numerical plot CSV includes a source path and selector. Summary-report extraction does not recover raw observations or create unavailable uncertainty.
