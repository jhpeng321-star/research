# WMFIX shard staging receipt — 2026-10-31 07:2x

gpuR /root/autodl-tmp/c4/wmfixmerge20261031 verified ready for a conditional
zero-head panel shard (addendum 07:20), pending the candB pilot vacating the GPU.

Hash-verified identical to gpuA (and to running PID 100400 LAUNCH.json):
- All 8 BOUND_FILES of zero_head_panel.py, plus zero_head_panel.py itself
  (f9ab3f05...) and tools/wmfixmerge/data.py (220595aa...).
- reference b8797242... / base_config f95f7c4c... / validation_real d754ac73...
- e3data manifests: C4_EPISODES_MANIFEST.json a1887d93..., SPLIT.json e1b2b9c5...
- verdict_plan.json synced to gpuA bytes f47a8f4c... (gpuR staged copy was stale
  a0947b4b...; overwritten with the canonical file, no deletions).
- Copied in (were MISSING): controllers DESIGN.json b658c6b5..., REAL.json
  d754ac73... relayed gpuA -> local -> gpuR (boxes cannot ssh each other).

Shard NOT launched: gpuR still occupied by candB WMFAST2 pilot (PID 481173 +
supervisor 481170 with auto-relauncher). ~12.5 GiB free < 12,098 MiB panel
tensors + context. Launch condition: pilot process gone >= 5 min AND
>= 13.5 GiB free; command (groups 26-35, never colliding with gpuA cursor,
which reaches 26 no earlier than ~09:40):

  cd /root/autodl-tmp/c4/wmfixmerge20261031 && nohup /root/miniconda3/bin/python -B \
    -m tools.wmfixmerge.zero_head_panel --plan tools/wmfixmerge/verdict_plan.json \
    --out validation_zero_head_shardR --groups 26-35 --memory-fraction 0.6 \
    --code-sha 1303d4a2_shardR > zero_head_shardR.log 2>&1 &

Batch files to be copied into gpuA validation_zero_head/ as they appear
(cp -n, then verify row counts) and merged before zero_head_compare.py.
Cross-device note: shard rows come from gpuR hardware; medians-over-8-draws and
the 2*SE decision rule are insensitive to reduction-order noise; shard device
recorded in its LAUNCH.json.
