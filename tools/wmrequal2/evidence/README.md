# S-WMREQUAL2 evidence

Existing validation rows remain immutable in `tools/wmrequal/evidence/validation`.
This directory retains new CPU-only provenance and per-arm analyses.

`RUNTIME_TEXT.json` supplements the immutable first runtime audit with LF-normalized
source hashes. Original staging preserved mixed CRLF/LF Python bytes, while Git
normalizes fresh checkouts; canonical source text and all original raw receipts
are independently verified. The fast-adapter digest also reproduces all 36 batch
source receipts from SOURCE + grouped-feedback + score-fast hashes.
