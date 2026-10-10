# Local research inputs (MIG-01)

The owner requested removing the project's dependency on `D:/research`, retaining
required dependencies only. Active execution now uses
`assets/research/highway_merge_shepherd_long/osm.net.xml.gz`.
The source directory was **not deleted or modified**.

## Preserved inputs

The [copy manifest](../assets/research/migration_manifest.json) records original
path, local path, size and SHA-256 for every preserved file. It contains 568 files,
9,064,282 bytes, under `assets/research/`:

- Inputs and build support from `highway_merge_shepherd`, `_long`, `_long2`,
  `_long3` and `_mid`, including `_light` scenarios and referenced auxiliary files.
- 82 `all_hdv_*.json` inputs explicitly used by archived checkpoint-comparison
  tools, under `traffic_shepherd_rl/runs/eval`. These were copied and hashed as
  opaque bytes, not parsed, analyzed, or added to development/training data.

The migration does not import the old RL project or its checkpoints into the
active controller. Unreferenced old simulation outputs, old RL source/checkpoints,
the separate lane-drop project and unrelated documents remain in `D:/research`.
Their preservation/deletion is a separate owner decision; this migration is not a
claim that those files lack historical value.

Four source `osm.sumocfg` examples in `_long`, `_long2`, `_long3` and `_mid` already
referenced a missing `osm.passenger.medium.trips.xml`. These gaps are recorded in
the manifest, not silently repaired. They do not affect the active route template
or network. All existing dependencies of the selected SUMO configurations were
copied; no new missing input was introduced.

## Verification and historical paths

Run `python tools/research_dependencies.py verify` in the existing `rl_driving`
environment. It verifies local hashes without reading the source directory.

The active SUMO launcher resolves old `D:/research/...` input references to the
verified copies using CLI overrides; saved `.sumocfg` files remain unchanged.
Historical scenario-identity calculation likewise uses identical copied network
bytes while retaining the original config bytes. There is no junction or fallback
to the retired directory.

For a raw historical SUMO config, prepare a new derived config without launching:

```powershell
python tools/research_dependencies.py prepare runs/active/OLD_JOB/scenario.sumocfg --output output/NEW_REPLAY_PREPARATION
```

Replace the example source with the actual config and choose a new output directory.
The helper resolves config input paths and records hashes in `relocation.json`.
It does not launch SUMO or grant a run budget. Do not execute a frozen `.sumocfg`
directly with its old absolute path after removing `D:/research`.

Archived Python code keeps its original defaults. For an explicitly authorized
historical rerun, pass the corresponding local scenario root/reference directory
through that tool's existing override, or use the prepared config. For example,
the archived `select_checkpoint.py --hdv-dir` input is now
`assets/research/traffic_shepherd_rl/runs/eval`. Resolve a legacy pathname with:

```powershell
python tools/research_dependencies.py resolve "D:/research/highway_merge_shepherd"
```

These are input relocation mechanisms, not a guarantee that every old launcher
still matches its historical source identity. Existing replay and budget checks
remain in force; archived source and manifests are not rewritten.

## Identity and research boundaries

Only path handling changed. Network bytes, ONNX bytes, control timing, requests,
physical controller laws, observation boundaries and criteria were not changed.
Adapter/source file hashes did change, so a new registration must record current
identity. Do not relabel an old registration as current or weaken equality checks
to make a previous campaign resume. The migration receipt documents the path-only
config change and preserved physical assets for explicit reuse assessment.

No simulation, training, new seed exposure or scientific comparison was performed.
See the [verification receipt](../output/research_dependency_migration_v1/VERIFICATION.json)
and [migration report](../output/research_dependency_migration_v1/REPORT.md).
The migration tests and local asset checks pass. The historical adoption checker
still flags existing PPO `torch` imports; that unrelated import-rule discrepancy
is recorded in the report and was not changed by this migration.


<!-- MIG02-status -->
**MIG-02 (2026-09-17):** [Source-isolation verification](../output/research_removal_v1/REPORT.md) passes: 568 local asset hashes, 1020 tests, exact saved replay and archive path/size checks. No further dependencies needed copying. Automatic approval review blocked deletion; all D:/research contents restored, old folder NOT deleted. Scientific route unchanged.
