# WD-01-PB-V04A single-job launcher with the 12 GB free-RAM gate.
# ONE heavy job at a time (registration s.7 / task rules). Refuses to
# launch when free RAM < 12 GB (exit 3). Logs the RAM reading + PID.
param(
  [Parameter(Mandatory=$true)][string]$JobName,
  [Parameter(Mandatory=$true)][string]$LogName,
  [Parameter(Mandatory=$true)][string[]]$PyArgs,
  [string]$WorkDir = "E:\rd-wt\wd01v04a\output\wd01"
)
$ErrorActionPreference = "Stop"
$env:WD_FORBID_D = "1"
$py = "C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe"
$logd = "E:\rd-wt\wd01v04a\output\wd01\partB_v0data\v04a_results"
$os = Get-CimInstance Win32_OperatingSystem
$freeGB = [math]::Round($os.FreePhysicalMemory/1MB, 2)
$stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$line = "[$stamp] job=$JobName free_ram_gb=$freeGB priority=BelowNormal wd_forbid_d=1 threads<=2(fit)/1(predict) gate_min_gb=12"
Add-Content -Path "$logd\$LogName.ram.log" -Value $line -Encoding utf8
if ($freeGB -lt 12.0) {
  Add-Content -Path "$logd\$LogName.ram.log" -Value "GATE-REFUSE: free RAM $freeGB GB < 12 GB; NOT launched" -Encoding utf8
  Write-Output "GATE-REFUSE free_ram_gb=$freeGB"
  exit 3
}
$p = Start-Process -FilePath $py -ArgumentList $PyArgs -WorkingDirectory $WorkDir `
  -RedirectStandardOutput "$logd\$LogName.log" -RedirectStandardError "$logd\$LogName.err" `
  -WindowStyle Hidden -PassThru
$p.PriorityClass = 'BelowNormal'
Add-Content -Path "$logd\$LogName.ram.log" -Value "[$stamp] launched pid=$($p.Id)" -Encoding utf8
Write-Output "LAUNCHED job=$JobName pid=$($p.Id) free_ram_gb=$freeGB"
