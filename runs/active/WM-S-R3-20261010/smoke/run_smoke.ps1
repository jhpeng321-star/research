$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'E:/rd-wt/ws-r3-int/output/wd01/wm_n1'
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:TMP = 'E:/rd-wt/ws-r3-int/_out/r3_smoke/tmp'
$env:TEMP = $env:TMP
$env:TMPDIR = $env:TMP
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:WM_N1_TRAIN_IDENTITIES = 'E:/rd-wt/ws-packv3fix/_out/sidecars_packv3_v2/identities'
$env:WM_N1_TRAIN_OUTCOMES = 'E:/rd-wt/ws-packv3fix/_out/sidecars_packv3_v2/outcomes'
& 'C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe' -m wms.cli train `
  --pack 'D:/rd-staging/wmn1_feat/pack_v3' `
  --identities 'E:/rd-wt/ws-packv3fix/_out/sidecars_packv3_v2/identities' `
  --outcomes 'E:/rd-wt/ws-packv3fix/_out/sidecars_packv3_v2/outcomes' `
  --pack 'E:/rd-wt/dspeed-data/_out/pack_dspeed' `
  --identities 'E:/rd-wt/a9/_out/identities_dspeed' `
  --overlay 'E:/rd-wt/dspeed-data/_out/labels_overlay_dspeed' `
  --pack 'E:/rd-wt/damping-snap/_out/pack_damping' `
  --identities 'E:/rd-wt/damping-snap/_out/identities_damping' `
  --overlay 'E:/rd-wt/damping-snap/_out/labels_overlay_damping' `
  --window-list '_out/lpwo_v3/windows_train_noPI_noWall.json' `
  --val-window-list '_out/lpwo_v3/windows_val_noPI_noWall.json' `
  --exclude-window-list '_out/lpwo_v3/excluded_runs_PI_wall_v3.json' `
  --merge-cal --merge-cal-params '_out/merge_params.json' `
  --cav-drive teacher --device cpu --threads 1 --num-workers 0 `
  --batch-size 1 --sumo-fidelity --start-shift-prob 0.75 --truncation-steps 20 `
  --rollout-mode residual --lane-head off `
  --history-s 60 --history-stride-s 1 `
  --residual-soft-bound-mps2 4 --residual-mag-weight 0.01 `
  --lane-rate-weight 1.0 --lane-rate-target 0.0123 `
  --select-by-long-horizon --val-every 200 `
  --asym-speed-loss 2 --decel-weight 5 --decel-threshold 3 `
  --propagation-loss 1 --prop-window 3 `
  --loss-share-logging `
  --unroll-s 10 --unroll-s 30 --unroll-s 60 --unroll-s 120 `
  --stage-steps 2,2,2,2 `
  --out '_out/r3_smoke/smoke'
exit $LASTEXITCODE
