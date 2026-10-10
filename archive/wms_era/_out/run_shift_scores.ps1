$ErrorActionPreference = "Stop"
$py = 'C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe'
$root = 'E:/rd-wt/ws-evalshift/output/wd01/wm_n1'
$b = 'E:/rd-wt/ws-evalshift/_out/c1_states_v2shift'
$lpo = 'E:/rd-wt/ws-p5/_out/lpo/windows_train_noPI.json'
$out = 'E:/rd-wt/ws-evalshift/_out/shift_scores'

$common = @(
  '--sets', "u_same=$b/u_same.json",
  '--sets', "u_new=$b/u_new.json",
  '--sets', "seen=$b/seen.json",
  '--train-windows', $lpo,
  '--horizon', '120'
)

$mainArgs = @('-m', 'wms_eval.c1v2',
  '--models', 'keep_speed=', '--models', 'pure_physics=',
  '--models', 'wms=_out/shift_ckpt/struct/wm_s.pt',
  '--models', 'blackbox_id=_out/shift_ckpt/bb_id/wm_s.pt',
  '--models', 'blackbox_traj=_out/shift_ckpt/bb_traj/wm_s.pt',
  '--out', "$out/_run_main", '--workers', '3') + $common

$oldArgs = @('-m', 'wms_eval.c1v2',
  '--models', 'wms=_out/shift_ckpt/oldhist/wm_s.pt',
  '--out', "$out/_run_oldhist", '--workers', '1') + $common

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'

$p1 = Start-Process -FilePath $py -ArgumentList $mainArgs -WorkingDirectory $root -RedirectStandardOutput "$out/_run_main.out" -RedirectStandardError "$out/_run_main.err" -PassThru
$p2 = Start-Process -FilePath $py -ArgumentList $oldArgs -WorkingDirectory $root -RedirectStandardOutput "$out/_run_oldhist.out" -RedirectStandardError "$out/_run_oldhist.err" -PassThru
"$($p1.Id) $($p2.Id)" | Set-Content "$out/_pids.txt"
Write-Output "started main=$($p1.Id) oldhist=$($p2.Id)"
