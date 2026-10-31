# LC2013 FIT lane diagnostic

`validate.py` reads **only** `episode.json`, `time_s.npy`, `veh_kin.npy`,
`veh_mask.npy`, `is_cav.npy`, `vid.npy` from WMDATA observable views. Exact
allowlist: worlds 969400–969431, native/ctrl0/mappo/pi/circles, 160 episodes.
No global manifest/split is opened. Each input file has a SHA256 receipt.

Frozen diagnostic: causal 240–300s wish warmup; score 300–360s. Both models see
the same measured post-motion positions/speeds/accelerations with pre-change
lanes. Same-ID HDVs on normal lanes in both adjacent frames are scored;
connectors, births, exits and CAV decisions are excluded. All visible normal
road bodies, including CAVs, constrain safety. Parameters: existing SUMO-fidelity
population defaults, no hidden driver files, no fitting or selection.

Reports exact directed-event recall/precision, ±2s one-to-one event matches,
matched timing/position MAE, per-transition and per-arm results. One-to-one
matching maximizes cardinality then minimizes total timing difference. Exact
position error is necessarily zero under longitudinal teacher forcing and is
not a trajectory prediction score. The ±2s tolerance is evaluation-only.

Positive *steps* can repeat while a model wants to change but the teacher still
stays. Thus both positive-step counts/precision and compressed positive-run
counts/window matches are reported. Wishes reset on **observed** lane changes,
never on a hypothetical predicted success. This diagnostic does not measure
free-running lane-change rate, discharge or breakdown fidelity.

Historical MOBIL 1.4% is from another panel/contract. The meaningful paired
comparison reruns current MOBIL on exactly these snapshots and settings.

CPU execution: `operate.py smoke` stages one immutable source bundle on cpuQ;
`operate.py panel` stages one full panel. Existing dispatch receipts block
duplicates. One nice19 worker, torch/OMP/MKL/OpenBLAS single-thread, CUDA hidden,
load admission <20. No signals or process-name killing. `operate.py status
--kind panel` checks identity-bound PID/startticks. `operate.py collect --kind
panel` requires completed output and exited worker. Old results retained.

The completed 969400/native smoke uses the original greedy window matcher;
the full panel uses the CPU-tested maximum-cardinality scorer. Model physics
and settings are unchanged after smoke. No tape outcomes were used to tune
parameters. A junction-zero-onehot bug was corrected before the smoke.
