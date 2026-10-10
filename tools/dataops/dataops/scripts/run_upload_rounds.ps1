# run_upload_rounds.ps1 -- run the parallel upload driver in time-boxed rounds,
# re-launching after each round until the queue is drained. Never runs two
# drivers: it refuses to start if any upload_driver.py process is already alive.
param(
  [int]$Workers = 4,
  [int]$RoundSeconds = 10800,
  [int]$MaxRounds = 0
)
$ErrorActionPreference = 'Continue'
$root  = 'E:\rd-wt\reorg3\dataops'
$py    = 'C:\Python314\python.exe'
$queue = "$root\_out\queue.json"
$driver= "$root\scripts\upload_driver.py"
$pendc = "$root\scripts\pending_count.py"
$rlog  = "$root\logs\upload.rounds.log"

function Get-Drivers {
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like '*upload_driver.py*' }
}

$alive = Get-Drivers
if ($alive) {
  "[$(Get-Date -Format o)] REFUSE start: driver already running pid=$($alive.ProcessId -join ',')" |
    Out-File -LiteralPath $rlog -Append -Encoding utf8
  exit 3
}

for ($r = 1; ; $r++) {
  if ($MaxRounds -gt 0 -and $r -gt $MaxRounds) { break }
  $drv = Get-Drivers
  if ($drv) {
    "[$(Get-Date -Format o)] round $r waiting for existing driver pid=$($drv.ProcessId -join ',')" |
      Out-File -LiteralPath $rlog -Append -Encoding utf8
    while (Get-Drivers) { Start-Sleep -Seconds 30 }
  }
  "[$(Get-Date -Format o)] round $r START workers=$Workers max=$RoundSeconds" |
    Out-File -LiteralPath $rlog -Append -Encoding utf8
  & $py $driver --queue $queue --workers $Workers --max-seconds $RoundSeconds *>> $rlog
  $rc = $LASTEXITCODE
  "[$(Get-Date -Format o)] round $r END rc=$rc" | Out-File -LiteralPath $rlog -Append -Encoding utf8
  $pend = & $py $pendc $queue
  "[$(Get-Date -Format o)] round $r pending=$pend" | Out-File -LiteralPath $rlog -Append -Encoding utf8
  if ([int]$pend -le 0) { break }
}
"[$(Get-Date -Format o)] rounds complete" | Out-File -LiteralPath $rlog -Append -Encoding utf8
