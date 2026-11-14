$env:WD_FORBID_D = "1"
$py = "C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe"
$wd = "E:\rd-wt\wd01v04a\output\wd01"
$root = "E:/research-dense/output/wd01_partB_v0data_v04a"
$res = "E:/research-dense/output/partB_results"
$logd = "E:\rd-wt\wd01v04a\output\wd01\partB_v0data\v04a_results"
function Launch($name, $feat, $extra) {
  $args = @("-m","partB_v0","train","--value-model","gbm","--gbm-variant","auto","--features",$feat,"--data",$root,"--out","$res/$name","--no-eval","--train-only","--gbm-max-iter","300","--gbm-cal-folds","2") + $extra
  $p = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $wd -RedirectStandardOutput "$logd\fit_$name.log" -RedirectStandardError "$logd\fit_$name.err" -WindowStyle Hidden -PassThru
  $p.PriorityClass = 'BelowNormal'
  "launched $name pid $($p.Id)"
}
Launch "v0.4a_strict" "deployable_strict" @()
Launch "v0.4a_detector" "deployable_detector" @()
$bk = Start-Process -FilePath $py -ArgumentList @("partB_v0data/finalize_backup_v04a.py","backup","--root",$root,"--dst","E:/research-dense-backups/wd01/partB_v0data_v04a") -WorkingDirectory $wd -RedirectStandardOutput "$logd\backup.log" -RedirectStandardError "$logd\backup.err" -WindowStyle Hidden -PassThru
$bk.PriorityClass = 'BelowNormal'
"launched backup pid $($bk.Id)"
