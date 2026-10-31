# S-WMFIXMERGE PILOT v2 conditional launcher (MAIN 13:50 TRACK 1).
# Waits for the v2-bound smoothed FD feasibility receipt; launches pilot_v2 ONLY if
# status PASS, launch_ready and finite_difference_ok are all true (pilot_v2 re-checks
# the same binding fail-closed). Otherwise writes v2/PILOT_V2_NOT_LAUNCHED.json.
cd /root/autodl-tmp/c4/wmfixmerge20261031
F=v2/FEASIBILITY_V2c.json
while [ ! -f $F ]; do
  if ! kill -0 $(cat v2/FEASV2c_PID.txt) 2>/dev/null && [ ! -f $F ]; then sleep 5; [ -f $F ] || { echo '{"launched": false, "reason": "feasibility process exited without receipt"}' > v2/PILOT_V2_NOT_LAUNCHED.json; exit 1; }; fi
  sleep 20
done
OK=$(/root/miniconda3/bin/python -c "import json;d=json.load(open('$F'));print(int(d.get('status')=='PASS' and d.get('launch_ready') is True and d.get('finite_difference_ok') is True))")
if [ "$OK" != "1" ]; then
  /root/miniconda3/bin/python -c "import json;d=json.load(open('$F'));json.dump(dict(launched=False,reason='FD feasibility not PASS',status=d.get('status'),finite_difference_ok=d.get('finite_difference_ok'),launch_ready=d.get('launch_ready'),blocker=d.get('blocker')),open('v2/PILOT_V2_NOT_LAUNCHED.json','x'),indent=1)"
  exit 2
fi
mkdir v2/pilot
export PYTHONPATH=src:.:/root/autodl-tmp/c4/wmfixmerge20261031/_in/ortdeps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
timeout --signal=INT --kill-after=30s 21570s /root/miniconda3/bin/python -m tools.wmfixmerge.pilot_v2 \
  --prepared v2/TRAIN_PREPARED_V2.json \
  --config /root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml \
  --checkpoint /root/autodl-tmp/c4/restart20261026/arm/loop/wm/c02/wm_s.pt \
  --controllers /root/autodl-tmp/c4/overnight20261028/W5_COMBINED/DESIGN.json \
  --feasibility $F --out v2/pilot --device cuda --memory-fraction 0.35 --smooth-events \
  > v2/pilot_v2.log 2>&1
echo "pilot_v2 exit $?" >> v2/pilot_v2.log
