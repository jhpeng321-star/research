# COOPBANDIT world-model EXAM v1

states.jsonl contains numeric observable features and the observed vehicle roster.
labels.csv is long format: state_id, option O0..O4, horizon_s 120/300/600;
D=J(O0)-J(option), positive saves fleet plus source delay (vehicle-seconds).
Unavailable options are explicit exact no-ops. J, D_road, D_source, safety and
override shares are included. group_credit.csv has C_G=D_G-sum(D_i).
world_id is grouping metadata, never a predictor; cell is descriptive only.
Group every split by world_id. Dev worlds 981080..981091 are excluded.
The roster uses only RSU detections and CAV telemetry, no driver truth.
provenance.json and MANIFEST.json bind the sources and design with SHA256.
Partial exports are exploratory, completion-order subsets, not final panel estimates.
