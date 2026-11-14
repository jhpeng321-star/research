# IDM v5.5 t=880 control-identity reconciliation protocol

Status: frozen diagnostic amendment, 2026-09-01. No treatment and no new seed.

## Reason for this bounded audit

The completed matched pilot reproduces the density mapper's anonymous 1 Hz
telemetry, terminal state, flow and population exactly for all six development
seeds, but the omnibus `ReplayTrace.trace_digest` differs. Source inspection
shows that the digest includes supervisor observation index 10, the normalized
episode-progress feature. The density mapper and pilot use different episode
caps while ending at the same absolute time, so that nonphysical feature must
differ even when traffic is identical.

This audit does not inspect or change treatment outcomes. It reopens only seeds
4401--4406 as command-off analytical-damping controls and requests no wall.

## Frozen identity layers

For each seed, rerun the exact density-mapper and pilot command-off wrappers and
require all of the following:

1. both rerun raw hashes reproduce their already-completed source artifacts;
   both complete controller-audit hashes and reset records must also reproduce
   those artifacts, whose file hashes and authorized schema/scenario/control
   contracts are pinned in every resumable audit record;
2. snapshot count and absolute/relative clocks are exact;
3. full sorted microscopic vehicle rows are exact, including vehicle ID, edge,
   lane, position, speed, acceleration, time loss, route index, type and tau;
4. active-CAV sets, collision IDs and teleport IDs are exact;
5. dynamic analytical-damping outputs are exact: base actions, direct-control
   IDs, base diagnostics, applied environment actions and out-of-band IDs;
6. no wall is active and no wall override ID exists;
7. supervisor observations are exact after excluding only index 10;
8. every raw snapshot difference is confined to supervisor index 10.

Static dormant-overlay configuration is not a dynamic action and is excluded.
No other supervisor feature is normalized or removed.

The audit passes only if every layer passes for all six seeds. This correction
can reconcile physical execution integrity, but it cannot change safety,
service, physical-benefit or diversity outcomes. The matched pilot remains a
no-expansion result unless its independently frozen scientific gates pass.
