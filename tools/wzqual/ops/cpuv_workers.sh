#!/usr/bin/env bash
# Launch WM-D shard-loop workers on cpuV:  bash cpuv_workers.sh RUN CKPT_DIR TOML NWORK THREADS MAXROWS "SHARDS..."
# Worker j gets every NWORK-th shard of the SHARDS list starting at position j.
set -u
RUN=${1:?}; CK=${2:?}; TOML=${3:?}; NW=${4:?}; THR=${5:?}; MR=${6:?}; SHARDS=${7:?}
C=/home/ubuntu/wmqual_20261106/wmcode_45547b832307
mkdir -p "$RUN"
read -r -a ALL <<< "$SHARDS"
for j in $(seq 0 $((NW-1))); do
  L=""
  for ((i=j; i<${#ALL[@]}; i+=NW)); do L="$L ${ALL[$i]}"; done
  [ -z "$L" ] && continue
  WZQ_NICE=19 WZQ_PYTHON=/home/ubuntu/miniconda3/bin/python /usr/bin/setsid nohup bash $C/tools/wzqual/ops/shard_loop.sh \
    w$j "$L" 80 $C "/home/ubuntu/wmqual_20261106/runs/*/s*/prefix/TAPE.npz" /home/ubuntu/wmqual_20261106/qdata \
    "$RUN" "$CK/wm_s.pt" "$CK/TRAIN_WORLDS.json" cpu "$THR" "$TOML" \
    --identity /home/ubuntu/wmqual_20261106/IDENTITY.json --samples 4 --start-stride 240 --batch-starts 1 \
    --batch-moments 1 --max-rows "$MR" > "$RUN/w$j.log" 2>&1 < /dev/null &
done
sleep 5
echo "workers: $(ls $RUN/w*.loop.pid 2>/dev/null | wc -l)"
uptime
