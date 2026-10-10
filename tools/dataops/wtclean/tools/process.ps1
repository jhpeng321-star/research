param(
  [string[]]$Names = @(),
  [long]$MaxBytes = [long]::MaxValue,
  [long]$MinBytes = 0,
  [switch]$Force
)

$ErrorActionPreference = "Continue"
$Base = "E:\rd-wt\reorg3\wtclean"
$Root = "E:\rd-wt"
$Rclone = "C:\Users\jpeng\miniconda3\bin\rclone.exe"
$InvFile = Join-Path $Base "inventory\inventory.json"
$ReceiptDir = Join-Path $Base "receipts"
$ListDir = Join-Path $Base "lists"
$LogDir = Join-Path $Base "inventory\logs"
$MainRepo = "E:\research-dense"
$ProgressLog = Join-Path $Base "wt_progress.log"
$CommitMsg = "wip: preserve before worktree retirement"

foreach ($p in @($ReceiptDir, $ListDir, $LogDir)) { if (-not (Test-Path -LiteralPath $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null } }

function Log([string]$msg) {
  $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ss"), $msg
  Add-Content -LiteralPath $ProgressLog -Value $line -Encoding UTF8
  Write-Host $line
}

function Invoke-Rclone {
  param([string[]]$RArgs, [string]$LogFile)
  & $Rclone @RArgs *> $LogFile
  return $LASTEXITCODE
}

$inv = Get-Content -LiteralPath $InvFile -Raw | ConvertFrom-Json
$Names = @($Names | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$targets = @($inv | Where-Object { -not $_.active })
if ($Names.Count -gt 0) { $targets = @($targets | Where-Object { $Names -contains $_.name }) }
$targets = @($targets | Where-Object { $_.uploadBytes -ge $MinBytes -and $_.uploadBytes -le $MaxBytes -and ($_.uploadCount -ge 0) })
$targets = @($targets | Sort-Object uploadBytes)

Log "=== PROC start names=$($Names.Count) range=[$MinBytes,$MaxBytes] targets=$($targets.Count) ==="

foreach ($t in $targets) {
  $name = $t.name
  $receiptPath = Join-Path $ReceiptDir ($name + ".json")
  if ((Test-Path -LiteralPath $receiptPath) -and -not $Force) {
    try { $existing = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json } catch { $existing = $null }
    if ($existing -and $existing.status -eq 'ok') { continue }
  }

  $dir = $t.path
  $rec = [ordered]@{
    name = $name
    path = $dir
    class = $t.class
    manual = ($t.junctions.Count -gt 0)
    branch = $t.branch
    headBefore = $t.head
    headType = $null
    dirtyBefore = $t.trackedDirtyCount
    committed = $false
    commitSha = $null
    preserveRef = $null
    dest = $null
    listFile = $t.listFile
    uploadCount = $t.uploadCount
    uploadBytes = $t.uploadBytes
    upload = [ordered]@{ ran = $false; exit = $null; log = $null }
    check = [ordered]@{ ran = $false; exit = $null; result = $null; log = $null }
    junctions = @($t.junctions)
    skippedSensitiveCount = $t.skippedSensitive.Count
    skippedSensitiveSample = @($t.skippedSensitive | Select-Object -First 25)
    skippedJunctionCount = $t.skippedJunctionPaths.Count
    status = "ok"
    error = $null
    timestamp = (Get-Date).ToString("o")
  }

  try {
    # ---- (b) commit tracked changes ----
    if ($t.class -in @('worktree','gitdir') -and $t.trackedDirtyCount -gt 0) {
      & git -C $dir add -u 2>&1 | Out-Null
      $addExit = $LASTEXITCODE
      $cout = & git -C $dir -c user.name="worktree-archive" -c user.email="worktree-archive@local" commit -m $CommitMsg 2>&1
      if ($LASTEXITCODE -eq 0) {
        $rec.committed = $true
        $rec.commitSha = (& git -C $dir rev-parse HEAD 2>$null | Select-Object -First 1)
        $br = (& git -C $dir rev-parse --abbrev-ref HEAD 2>$null | Select-Object -First 1)
        $rec.branch = $br
        if ($br -eq 'HEAD') {
          # detached: keep commit reachable via a preservation ref in the main repo
          $ref = "refs/heads/wip-preserve/$name"
          & git -C $MainRepo update-ref $ref $rec.commitSha 2>&1 | Out-Null
          if ($LASTEXITCODE -eq 0) { $rec.preserveRef = $ref }
          $rec.headType = 'detached'
        } else {
          $rec.headType = 'branch'
        }
        Log "  $name committed $($rec.commitSha) -> branch=$($rec.branch) ref=$($rec.preserveRef)"
      } else {
        $rec.status = "commit_failed"
        $rec.error = ("commit failed: " + (($cout | Select-Object -Last 3) -join ' '))
        Log "  $name COMMIT FAILED: $($rec.error)"
      }
    }

    # ---- (c) upload untracked+ignored ----
    if ($t.uploadCount -gt 0 -and $t.listFile -and (Test-Path -LiteralPath $t.listFile)) {
      $dest = "gdrive:research-archive/worktrees/$name"
      $rec.dest = $dest
      $upLog = Join-Path $LogDir ($name + ".upload.log")
      $rec.upload.ran = $true
      $rec.upload.log = $upLog
      $uArgs = @('copy', $dir, $dest, '--files-from', $t.listFile, '--transfers', '32', '--checkers', '32',
        '--drive-pacer-min-sleep', '0s', '--drive-pacer-burst', '3000',
        '--retries', '5', '--low-level-retries', '10', '--stats-one-line', '--stats', '10', '--log-level', 'INFO')
      $ue = Invoke-Rclone -RArgs $uArgs -LogFile $upLog
      $rec.upload.exit = $ue
      if ($ue -ne 0) {
        $rec.status = "upload_failed"
        $rec.error = "rclone copy exit $ue"
        Log "  $name UPLOAD FAILED exit=$ue"
      } else {
        $chkLog = Join-Path $LogDir ($name + ".check.log")
        $rec.check.ran = $true
        $rec.check.log = $chkLog
        $cArgs = @('check', $dir, $dest, '--files-from', $t.listFile, '--one-way', '--log-level', 'INFO')
        $ce = Invoke-Rclone -RArgs $cArgs -LogFile $chkLog
        $rec.check.exit = $ce
        $cl = @(Get-Content -LiteralPath $chkLog -ErrorAction SilentlyContinue)
        $diffLine = ($cl | Where-Object { $_ -match 'differences found|matching files' } | Select-Object -Last 2) -join ' | '
        $rec.check.result = $diffLine
        if ($ce -ne 0) {
          $rec.status = "check_failed"
          $rec.error = "rclone check exit $ce : $diffLine"
          Log "  $name CHECK FAILED exit=$ce $diffLine"
        } else {
          Log "  $name ok upload+check files=$($t.uploadCount) bytes=$($t.uploadBytes) $diffLine"
        }
      }
    } else {
      Log "  $name no untracked content (committed=$($rec.committed))"
    }
  } catch {
    $rec.status = "error"
    $rec.error = $_.Exception.Message
    Log "  $name EXCEPTION: $($rec.error)"
  }

  $rec | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
  Log "  $name RECEIPT status=$($rec.status)"
}

Log "=== PROC done ==="
