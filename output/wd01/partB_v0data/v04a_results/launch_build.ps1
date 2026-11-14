$env:WD_FORBID_D = "1"
$py = "C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe"
$wd = "E:\rd-wt\wd01v04a\output\wd01\partB_v0data"
$p = Start-Process -FilePath $py -ArgumentList @("build_v04a.py","--data-root","E:/research-dense","--workers","4","--arm-verification","E:/rd-wt/wd01v04a/output/wd01/partB_v0data/v04a_results/arm_verification.json") -WorkingDirectory $wd -RedirectStandardOutput "$wd\v04a_results\build.log" -RedirectStandardError "$wd\v04a_results\build.err" -WindowStyle Hidden -PassThru
$p.PriorityClass = 'BelowNormal'
"launched pid $($p.Id)"
