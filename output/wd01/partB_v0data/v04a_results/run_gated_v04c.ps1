# WD-01-PB-V04A resume (worktree wd01v04c): single-job launcher.
# Gates BEFORE launch: free RAM >= 12 GB AND FreeVirtualMemory >= 8 GB
# (registration s.7 / task rules). If lower: poll every 5 min, max 60 min.
# ONE heavy job at a time. Logs the RAM readings + PID. WD_FORBID_D=1.
param(
  [Parameter(Mandatory=$true)][string]$JobName,
  [Parameter(Mandatory=$true)][string]$LogName,
  [Parameter(Mandatory=$true)][string[]]$PyArgs,
  [string]$WorkDir = "E:\rd-wt\wd01v04c\output\wd01"
)
$ErrorActionPreference = "Stop"
$env:WD_FORBID_D = "1"
$py = "C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe"
$logd = "E:\rd-wt\wd01v04c\output\wd01\partB_v0data\v04a_results"

function Read-Gate {
  $os = Get-CimInstance Win32_OperatingSystem
  [pscustomobject]@{
    ram = [math]::Round($os.FreePhysicalMemory/1MB, 2)
    vm  = [math]::Round($os.FreeVirtualMemory/1MB, 2)
  }
}

$g = Read-Gate
$tries = 0
while (($g.ram -lt 12.0 -or $g.vm -lt 8.0) -and $tries -lt 12) {
  $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  Add-Content -Path "$logd\$LogName.ram.log" `
    -Value "[$stamp] job=$JobName WAIT try=$tries free_ram_gb=$($g.ram) free_vm_gb=$($g.vm) (need ram>=12 vm>=8)" `
    -Encoding utf8
  Start-Sleep -Seconds 300
  $g = Read-Gate
  $tries += 1
}
$stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$line = "[$stamp] job=$JobName free_ram_gb=$($g.ram) free_vm_gb=$($g.vm) priority=BelowNormal wd_forbid_d=1 gate_min_ram_gb=12 gate_min_vm_gb=8"
Add-Content -Path "$logd\$LogName.ram.log" -Value $line -Encoding utf8
if ($g.ram -lt 12.0 -or $g.vm -lt 8.0) {
  Add-Content -Path "$logd\$LogName.ram.log" `
    -Value "GATE-REFUSE: free RAM $($g.ram) GB / VM $($g.vm) GB below gates after 60 min; NOT launched" `
    -Encoding utf8
  Write-Output "GATE-REFUSE free_ram_gb=$($g.ram) free_vm_gb=$($g.vm)"
  exit 3
}
$p = Start-Process -FilePath $py -ArgumentList $PyArgs -WorkingDirectory $WorkDir `
  -RedirectStandardOutput "$logd\$LogName.log" -RedirectStandardError "$logd\$LogName.err" `
  -WindowStyle Hidden -PassThru
$p.PriorityClass = 'BelowNormal'
Add-Content -Path "$logd\$LogName.ram.log" -Value "[$stamp] launched pid=$($p.Id)" -Encoding utf8
Write-Output "LAUNCHED job=$JobName pid=$($p.Id) free_ram_gb=$($g.ram) free_vm_gb=$($g.vm)"
