# Corrected publication commit-only recovery — 2026-11-01 ~17:00 +08

CI recovery completed successfully: old collector9660 and publisher24320 exited
their exact expected source guards before successors launched. CollectorV3 is
LIVE25240/create_filetime134379970282244431, root runs/rb2int_finish_v3. Its first
poll16:57:08 collected12/216 complete world jobs (no failures on remote status).

PublisherV2 completed corrected analysis at16:57:09, wrote the corrected report,
then its commit encountered the collector's simultaneous initial staging. Its
explicit `foreign staging present; preserve it` STOP retained all completed
analysis and report bytes. Collector committed normally at a5bbb30b. No scientific
task or analysis failure; no forecast/scoring rerun required.

Commit the existing corrected16:57snapshot and report once. PublisherV3 will reuse
that exact completed snapshot, binding its analysis SHA and the numerical-CI
repair source hashes from recovery COMPLETE. It checks publisherV2 is actually
exited and its STOP is exactly commit-only contention. Its next timed refresh is
~19:57; complete reserved8 still publishes immediately when ready.

Publication commits now wait for a quiet window30..1000s after a successful sole-
collector poll, and for its named COLLECT staging to clear. Foreign non-collector
staging still stops safely. This avoids the deterministic startup commit collision;
no inference/collection dispatch, no staged-file removal and no analysis retry.
Original live remote supervisors473053/924719 remain untouched.
