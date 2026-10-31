# CPU-only cooperation behaviours

Requires the existing Python environment with NumPy and Matplotlib; tests use
pytest. No model, TraCI, SUMO, hidden driver labels or assignment files are read.

```
python -m pytest tests/unit/test_coopbehav.py tests/unit/test_coopbehav_discovery.py tests/unit/test_coopbehav_report.py -q
python -m tools.coopbehav.report --views /root/wmdata2/views --label WMDATA2 --out /root/coopbehav20261031_analysis/WMDATA2
python -m tools.coopbehav.report --views /home/ubuntu/wmdata20261031/views --label WMDATA --out /home/ubuntu/coopbehav20261031_analysis/WMDATA
python -m tools.coopbehav.report --pairs COOPBEHAV_INDEX.json --out paired_report
python -m tools.coopbehav.poll --config tools/coopbehav/handoffs.json --out runs/coopbehav_poll --watch
```

## Outputs

`summary.json`: group event counts, member event counts, CAV exposure, episode
exposure, controller-specific rates and paired-support coverage. `members.csv`
includes zero-event CAVs; `events.jsonl.gz` contains exact group/member identities,
event boundaries, geometry, censoring and evidence status. `episodes.json` and
`inputs.json` retain per-episode coverage and SHA256s of all arrays actually read.
Four PNGs per arm show examples (or an explicit no-event panel). Selection is the
largest gap opening, otherwise longest episode, not a representative sample.
`signatures.json` and `discovery.json` contain measured-gain ranking, physical
features, deterministic cluster method, centroids and unexplained candidates.
`COMPLETE.json` is written only after report generation succeeds.

E3DATA commands are joined on `control_time_s`, held until the next command,
and never carried across identity recycling. There is no interpolation or
extrapolation of unknown command ledgers. Native/no-command status must be
explicit for release support. Junction lane one-hots are unknown, not lane 0.

## Paired-source producer handoff

Publish a JSON index atomically with `ready: true`, `source`, and `pairs`.
Each pair has `id`, `seed`, `arm`, `kind: "native_member_release"`, `members`
(identity strings), `factual`, `reference`, `branch: {t0_s, horizon_s}` and,
when available, reconciled `C_G`, `D_G`, `sum_D_i` for that exact group/outcome.
For discovery also provide `intervention_window_s` and name `credit_outcome`.
Signatures use only the intervention window, not the later outcome horizon.
Publish separate indices/reports for different registered credit outcomes.
Paths are relative to the index or absolute. Files may be E3DATA directories or
normalized `.jsonl[.gz]` trajectories; each line is:

```
{"time_s":300.0,"vehicles":[{"vid":"main.1","s_m":-20.0,"v_mps":12.0,"length_m":5.0,"lane":1,"is_cav":true,"controller":0,"issued_acceleration_mps2":null}]}
```

`controller: 0` or `"native"` and a null issued command explicitly represent
released native members. The factual and reference initial states and clocks
must match; missing/unknown release status fails closed. Multiple released
members support a **group intervention**, not isolated attribution to one CAV.
Policy-mean baselines from SUMOCF use `kind: "policy_mean_intervention"`.
The producer's published alias `policy_mean_reference` is also supported,
including shared top-level scope/precision metadata and bound trace hashes.
Their reference member frames must carry `controller: "policy_mean"` and an
`actions` entry whose `vid`, `policy_mean` and recorded issued command agree.
They are explicitly labelled sampled-window-versus-policy-mean support, never
native-release support. Native frames without commands remain valid.

Include the **same-lane leader/follower identities**, lengths, CAV flags,
controller/no-command status and immediate mainline neighbours, as well as ramp
vehicles. A members-plus-ramp-only crop cannot identify true receiving slots or
follower chains. Declare `trace_scope` and `trace_limitations`; a full observable
fleet export is preferred. For JSONL rounded to a source dtype, provide
`position_dtype_epsilon`. No default vehicle length or hidden-type inference.

The configured local handoff paths are in `handoffs.json`; update those explicit
paths if producers publish elsewhere. Polling records the design-document SHAs
and trace-related lines, and automatically reports ready indices. Remote indices
can be configured as `remote_indices: [{host: "gpuR", path: "/absolute/path/COOPBEHAV_INDEX.json"}]`,
or published verbatim as `gpuR:/absolute/path/COOPBEHAV_INDEX.json` (or `cpuQ:`)
in either watched design document. The poller launches only its installed CPU
reporter under the analysis root and retains PID/start-tick receipts, once per
index SHA. It cannot reconstruct trajectories from SHA256 digests or discover
unspecified remote roots.
`sources.py` now also polls the published gpuR SUMOCF job-result/trace root,
uses the saved subset roster (never fixed branch numbers), independently
reconstructs group/solo credit and generates ready indices under our own root.
It waits for closed `result.json` files. The original COOPAUDIT crop is explicitly
blocked for missing geometry/neighbour/action fields; no default lengths or
guessed controller commands are substituted.
Place `STOP` in the poller's output directory for cooperative exit. Remote
reporting uses `remote.py` and stores exact owned PID/start-tick receipts; it
never signals a process or imports a simulator.
