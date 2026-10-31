# S-SUMOCORE

Completed observable-start SUMO 1.27.1 diagnostic. Scientific report:
`docs/paper_results/SUMOCORE.md`; frozen design and every amendment:
`docs/paper_results/SUMOCORE_DESIGN.md`.

## Existing artifacts

- cpuQ `/home/ubuntu/sumocore20261031`: immutable plans,150precharged attempt
  identities, all original failures/STOPs,144effective completed futures,
  own-PID dispatch receipts and versioned source directories.
- Mirrored compact receipts,10s observables, scores and hashes:
  `docs/paper_results/sumocore/`. Full0.5s trajectories remain on cpuQ.
- All supervisors exited. `python tools/sumocore/status.py` only inspects these
  owned receipts/processes; no inference or simulation is launched.

## Scientific adapter

`prepare_bundle.py` reads observable `start.pt` and `start_history.npz` only,
parses the network and creates its OWN future-demand/types XML. It never calls
the ordinary scenario reset or imports original scenario assignments/state.
`model_env.py` reuses frozen CAV execution, force-inserts all observed vehicles
before the first physics tick and audits their placement/speed. `inference.py`
documents the IDM parameter map; `demand.py` preserves paired future streams.
No shared WM model file or default behavior was edited.

`supervisor.py`, `admission.py`, and `dispatch.py` enforce exclusive ownership,
precharge, explicit registered amendments and completed-run reuse. Production
is complete; these are one-shot campaign commands, not periodic jobs.

## Verification and scoring

- Local motion-demand/mapping/accounting contracts:
  `python tools/sumocore/test_contracts.py` (4PASS).
- Immutable scorer stage cpuQ `code/1f061a300604/` contains `score.py`,
  `verify.py`, `test_scoring.py`, S-RB-EVAL `core.py` snapshot, registered BIC
  extraction, the S-WMBREAKDOWN factual receipt and WM2 K8 draw receipt.
- Three proper-score/clustering tests PASS. The completed independent audit
  re-integrates all144road/source paths, checks pairing and preserves all
  collision flags; `scores/verification.json` separates data/accounting PASS
  from scientific physical-validity FAIL.
- Score files are exclusive/immutable. `stage.py` snapshots allowlisted sources
  into NEW SHA-named remote directories; it never replaces a live runtime.
- K3/threearms cannot be imported as RBEVAL's fixed K8/fivearm qualification
  panel. We reuse its exact detector, region reduction and CRPS primitives,
  while reporting this registered small panel with its own explicit identity.

The shared obsstate helper541f109f cross-checks649vehicles/routes/lengths;
the targets helper independently cross-checks54development clock values.
No helper's optional future-demand or validation-exit inputs drive forecasts.
