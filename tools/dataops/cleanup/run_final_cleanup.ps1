<#
.SYNOPSIS
  One-shot final clean-up (owner runs it once). Chains every step of docs/records/final_cleanup_runbook_20261010.md.

.DESCRIPTION
  Asks for ONE typed confirmation (CLEAN), then runs, in order, logging everything to
  data_ops\cleanup_out\final_run_<stamp>.log:
    1. MOVE_TO_SEALED -Execute            (sensitive-named files -> sealed\, never opened)   [stops the run on any failure]
    2. clean-up Stage 2 -RemoveOld        (old pack copies; each re-verified on Drive)
    3. clean-up Stage 1                   (all worktrees, D:\rd-staging, Drive-only raw data; each re-verified on Drive)
    4. git worktree prune
    5. E:\rd-wt\reorg3                    (archived whole to Drive as a .tar, MD5-checked, then removed)
    6. E:\rd-wt$w, E:\tmp                 (junctions unlinked link-only, never followed)
    7. E:\rd-wt if empty; D:\research-dense, D:\rd-staging\_dataops_tmp, _wtclean_tmp
    8. C:  Codex worktrees of this repo (only if `git status` is empty), pytest/pip temp, an old Claude temp folder
  Not touched: this Claude session's temp folder and %TEMP%\opencode (in use), E:\research-dense-backups,
  unknown files in %TEMP% (ro.bin, ro2.bin, undo.bin, wd01_full.tar), anything non-research.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File E:\research-dense\tools\dataops\cleanup\run_final_cleanup.ps1
#>
[CmdletBinding()]
param([switch]$Yes)
$ErrorActionPreference = 'Continue'
$Repo = 'E:\research-dense'
$PY = 'C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe'
$RCLONE = 'C:\Users\jpeng\miniconda3\bin\rclone.exe'
$Out = Join-Path $Repo 'data_ops\cleanup_out'
$CL = Join-Path $Out 'MORNING_CLEANUP.ps1'
$Move = Join-Path $Repo 'tmp\consolidate\MOVE_TO_SEALED.ps1'
New-Item -ItemType Directory -Path $Out -Force | Out-Null
$Log = Join-Path $Out ('final_run_{0}.log' -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
Set-Location $Repo

function Say([string]$m, [string]$c = 'Cyan') { Write-Host $m -ForegroundColor $c; Add-Content -LiteralPath $Log -Value $m }
function Run([string]$title, [scriptblock]$body) {
    Say ''; Say ("===== {0}  ({1:HH:mm:ss}) =====" -f $title, (Get-Date))
    $text = & $body 2>&1 | Tee-Object -FilePath $Log -Append | Out-String
    return $text
}
function Remove-NoFollow([string]$p) {
    # delete a tree without ever following junctions/symlinks (links are removed as links)
    Get-ChildItem -LiteralPath $p -Force -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            cmd /c rmdir "$($_.FullName)" 2>$null
            if (Test-Path -LiteralPath $_.FullName) { Remove-Item -LiteralPath $_.FullName -Force }
            Say ("  unlinked (link only): {0}" -f $_.FullName) 'DarkGray'
        } elseif ($_.PSIsContainer) { Remove-NoFollow $_.FullName; Remove-Item -LiteralPath $_.FullName -Force }
        else { Remove-Item -LiteralPath $_.FullName -Force }
    }
}

Say "FINAL CLEAN-UP  log: $Log" 'Yellow'
Say 'Runs: sealed move -> Stage 2 -> Stage 1 -> worktree prune -> reorg3 -> E:\tmp, rd-wt$w -> D: leftovers -> C: leftovers.' 'Yellow'
Say 'Every removal of research data is re-verified against its Google Drive copy right before it happens.' 'Yellow'
if (-not $Yes) { $a = Read-Host 'Type CLEAN to start'; if ($a -cne 'CLEAN') { Say 'aborted - nothing changed.' 'Yellow'; return } }

# 0. regenerate + self-test
$t = Run 'Step 0: regenerate the clean-up script and self-test' { & $PY -I tools\dataops\cleanup\gen_cleanup.py; powershell -NoProfile -ExecutionPolicy Bypass -File $CL -SelfTest }
if ($t -notmatch 'SELF TEST: \d+ passed, 0 failed') { Say 'STOP: self-test did not pass. Nothing was removed. Send me the log.' 'Red'; return }

# 1. sealed move
$t = Run 'Step 1: move sensitive-named files into sealed\' { powershell -NoProfile -ExecutionPolicy Bypass -File $Move -Execute -Yes }
if ($t -notmatch 'failed 0' -and $t -notmatch 'to move 0') { Say 'STOP: the sealed move reported failures. Nothing else was run. Send me the log.' 'Red'; return }

# 2. old pack copies
Run 'Step 2: Stage 2 - remove old pack copies' { powershell -NoProfile -ExecutionPolicy Bypass -File $CL -Stage 2 -RemoveOld -Execute -Yes } | Out-Null

# 3. items + worktrees
Run 'Step 3: Stage 1 - worktrees, D:\rd-staging, Drive-only raw data' { powershell -NoProfile -ExecutionPolicy Bypass -File $CL -Stage 1 -Execute -RelocateSensitive -AllowNestedGit -Yes } | Out-Null

# 4. prune
Run 'Step 4: git worktree prune' { git -C $Repo worktree prune; git -C $Repo worktree list } | Out-Null

# 5. reorg3: whole-folder archive to Drive, verify, remove
if (Test-Path -LiteralPath 'E:\rd-wt\reorg3') {
    Run 'Step 5: archive E:\rd-wt\reorg3 to Drive, verify, remove' {
        $tarDir = 'E:\reorg3_archive'; New-Item -ItemType Directory -Path $tarDir -Force | Out-Null
        $tar = Join-Path $tarDir 'reorg3_full_20261011.tar'
        tar -cf $tar -C 'E:\rd-wt' reorg3
        if ($LASTEXITCODE -ne 0) { "tar failed ($LASTEXITCODE) - reorg3 kept"; return }
        & $RCLONE copy $tarDir 'gdrive:research-archive/repo-evidence/reorg3_full' 2>&1
        & $RCLONE check $tarDir 'gdrive:research-archive/repo-evidence/reorg3_full' --one-way 2>&1
        if ($LASTEXITCODE -ne 0) { 'Drive check failed - reorg3 kept'; return }
        Remove-NoFollow 'E:\rd-wt\reorg3'; Remove-Item -LiteralPath 'E:\rd-wt\reorg3' -Force
        Remove-Item -LiteralPath $tarDir -Recurse -Force
        'reorg3 archived on Drive (gdrive:research-archive/repo-evidence/reorg3_full) and removed'
    } | Out-Null
}

# 6. stale junction folder and E:\tmp
Run 'Step 6: E:\rd-wt$w and E:\tmp (links removed as links)' {
    if (Test-Path -LiteralPath 'E:\rd-wt$w') { Remove-NoFollow 'E:\rd-wt$w'; Remove-Item -LiteralPath 'E:\rd-wt$w' -Force; 'removed E:\rd-wt$w' }
    if (Test-Path -LiteralPath 'E:\tmp') { Remove-NoFollow 'E:\tmp'; Remove-Item -LiteralPath 'E:\tmp' -Force; 'removed E:\tmp' }
} | Out-Null

# 7. E:\rd-wt if empty; D: leftovers
Run 'Step 7: E:\rd-wt (if empty) and D: leftovers' {
    if (Test-Path -LiteralPath 'E:\rd-wt') {
        $left = @(Get-ChildItem -LiteralPath 'E:\rd-wt' -Force)
        if ($left.Count -eq 0) { Remove-Item -LiteralPath 'E:\rd-wt' -Force; 'removed empty E:\rd-wt' }
        else { 'E:\rd-wt still holds (kept): ' + (($left | ForEach-Object Name) -join ', ') }
    }
    foreach ($p in 'D:\research-dense', 'D:\rd-staging\_dataops_tmp', 'D:\rd-staging\_wtclean_tmp') {
        if (Test-Path -LiteralPath $p) { Remove-NoFollow $p; Remove-Item -LiteralPath $p -Force; "removed $p" }
    }
} | Out-Null

# 8. C: leftovers
Run 'Step 8: C: leftovers (Codex worktrees if clean, temp folders)' {
    foreach ($w in 'wmn1-eval-bridge', 'wmn1-j4-identity', 'wmn1-label-time-audit') {
        $p = "C:\Users\jpeng\.codex\worktrees\$w"
        if (-not (Test-Path -LiteralPath $p)) { continue }
        $st = git -C $p status --short 2>&1
        if ([string]::IsNullOrWhiteSpace(($st | Out-String))) { git -C $Repo worktree remove $p; "removed Codex worktree $w" }
        else { "KEPT $w (uncommitted changes):"; $st }
    }
    git -C $Repo worktree prune
    foreach ($p in "$env:TEMP\pytest-of-jpeng", "$env:TEMP\claude\E--research-dense\77f8fe92-adec-40ea-b584-eb82afb017f3") {
        if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue; "removed $p" }
    }
    Get-ChildItem $env:TEMP -Directory -Filter 'pip-*' -ErrorAction SilentlyContinue | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue; "removed $($_.FullName)" }
} | Out-Null

Say ''
Say '===== DONE =====' 'Green'
foreach ($d in 'C', 'D', 'E') { $f = (Get-PSDrive $d).Free; Say ("{0}: {1:N1} GiB free" -f $d, ($f / 1GB)) 'Green' }
Say "Send the log to Claude: $Log" 'Yellow'
Say 'Later, after training and with this Claude session closed, also delete %TEMP%\claude\E--research-dense\8110729b-4744-41ed-a71d-1be9209d4cb7 and %TEMP%\opencode (~16 GB).' 'Yellow'
