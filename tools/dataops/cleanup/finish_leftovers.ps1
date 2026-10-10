<#
.SYNOPSIS
  Retire the last E:\rd-wt worktrees the main clean-up held (j4bfix, pl57, pl58, pl59, ws-packv3fix). Owner runs it once.

.DESCRIPTION
  For every folder: (1) remove junctions/symlinks inside it as LINKS (never followed);
  (2) move sensitive-named files (holdout, heldout, cal_split, chunk5, chunk-5, j4b, inc1, sealed, token "cal";
      matched on the path INSIDE the folder) unopened into sealed\relocated_20261010\E\rd-wt\<name>\..., SHA-256 checked;
  (3) archive the rest whole as a .tar, upload to gdrive:research-archive/worktrees_final/<name>/, verify with rclone check;
  (4) only then delete the folder; finally `git worktree prune`, and remove E:\rd-wt if it is empty.
  Any failed upload/check keeps that folder untouched. Log: data_ops\cleanup_out\finish_leftovers_<stamp>.log
#>
[CmdletBinding()]
param([switch]$Yes)
$ErrorActionPreference = 'Continue'   # native tools (rclone, git) write notices to stderr; failures are checked explicitly
$Repo = 'E:\research-dense'
$RCLONE = 'C:\Users\jpeng\miniconda3\bin\rclone.exe'
$Sealed = 'E:\research-dense\sealed\relocated_20261010\E\rd-wt'
$Stage = 'E:\leftover_archive'
$Log = Join-Path $Repo ('data_ops\cleanup_out\finish_leftovers_{0}.log' -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$Names = 'ws-packv3fix', 'pl57', 'pl58', 'pl59', 'j4bfix'
$Sens = '(?i)(holdout|heldout|cal_split|chunk5|chunk-5|j4b|inc1|sealed)|(^|[\\/_.-])cal([\\/_.-]|$)'

function Say([string]$m, [string]$c = 'Cyan') { Write-Host $m -ForegroundColor $c; [IO.File]::AppendAllText($Log, $m + "`r`n") }
function Sha([string]$p) { (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash }
function Unlink-Links([string]$dir) {
    $n = 0
    foreach ($e in (New-Object IO.DirectoryInfo $dir).EnumerateFileSystemInfos()) {
        if ($e.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            if ($e -is [IO.DirectoryInfo]) { cmd /c rmdir "$($e.FullName)" | Out-Null } else { [IO.File]::Delete($e.FullName) }
            if (Test-Path -LiteralPath $e.FullName) { throw "could not unlink $($e.FullName)" }
            Say ("  unlinked (link only): {0}" -f $e.FullName) 'DarkGray'; $n++
        } elseif ($e -is [IO.DirectoryInfo]) { $n += Unlink-Links $e.FullName }
    }
    return $n
}
function Files([string]$dir) {
    $out = New-Object Collections.ArrayList
    foreach ($e in (New-Object IO.DirectoryInfo $dir).EnumerateFileSystemInfos()) {
        if ($e.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
        if ($e -is [IO.DirectoryInfo]) { foreach ($x in (Files $e.FullName)) { [void]$out.Add($x) } } else { [void]$out.Add($e.FullName) }
    }
    return $out
}

$todo = @($Names | Where-Object { Test-Path -LiteralPath "E:\rd-wt\$_" })
Say ("FINISH LEFTOVERS: {0}   log: {1}" -f ($todo -join ', '), $Log) 'Yellow'
if ($todo.Count -eq 0) { Say 'no E:d-wt folders left' }
if (-not $Yes) { $a = Read-Host 'Type FINISH to start'; if ($a -cne 'FINISH') { Say 'aborted - nothing changed.'; return } }
New-Item -ItemType Directory -Path $Stage -Force | Out-Null

foreach ($name in $todo) {
    $dir = "E:\rd-wt\$name"
    Say ''; Say "== $name"
    try {
        $links = Unlink-Links $dir
        # sensitive-named files -> sealed (opaque move, hash-checked)
        $moved = 0
        foreach ($f in (Files $dir)) {
            $rel = $f.Substring($dir.Length).TrimStart('\')
            if ($rel -notmatch $Sens) { continue }
            $to = Join-Path (Join-Path $Sealed $name) $rel
            if (Test-Path -LiteralPath $to) { throw "sealed destination exists: $to" }
            New-Item -ItemType Directory -Path (Split-Path -Parent $to) -Force -ErrorAction Stop | Out-Null
            $h = Sha $f; [IO.File]::Move($f, $to)
            if ((Sha $to) -ne $h) { throw "hash changed after move: $to" }
            $moved++
        }
        Say ("  links unlinked: {0}; sensitive-named files moved to sealed: {1}" -f $links, $moved)
        # archive the rest, upload, verify
        $tarDir = Join-Path $Stage $name; New-Item -ItemType Directory -Path $tarDir -Force -ErrorAction Stop | Out-Null
        $tar = Join-Path $tarDir "$name.tar"
        tar -cf $tar -C 'E:\rd-wt' $name
        if ($LASTEXITCODE -ne 0) { throw "tar failed ($LASTEXITCODE)" }
        $remote = "gdrive:research-archive/worktrees_final/$name"
        cmd /c "`"$RCLONE`" copy `"$tarDir`" $remote 2>nul"
        if ($LASTEXITCODE -ne 0) { throw "upload failed ($LASTEXITCODE)" }
        cmd /c "`"$RCLONE`" check `"$tarDir`" $remote --one-way 2>nul"
        if ($LASTEXITCODE -ne 0) { throw "Drive check failed ($LASTEXITCODE)" }
        Say ("  archived: {0} ({1:N1} MB) -> {2} (verified)" -f "$name.tar", ((Get-Item $tar).Length / 1MB), $remote)
        # delete: registered worktrees via git, broken ones directly (links are already gone)
        $g = git -C $Repo worktree remove --force $dir 2>&1
        if (Test-Path -LiteralPath $dir) { Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction Stop }
        if (Test-Path -LiteralPath $dir) { throw "could not remove $dir" }
        Remove-Item -LiteralPath $tarDir -Recurse -Force
        Say "  removed $dir" 'Green'
    } catch {
        Say ("  KEPT {0}: {1}" -f $name, $_.Exception.Message) 'Red'
    }
}
# C: Codex worktrees of this repo (real checkout is <name>\research-dense); removed only if git status is empty
foreach ($w in 'wmn1-eval-bridge', 'wmn1-j4-identity', 'wmn1-label-time-audit') {
    $outer = "C:\Users\jpeng\.codex\worktrees\$w"; $p = Join-Path $outer 'research-dense'
    if (-not (Test-Path -LiteralPath $p)) { continue }
    $st = (git -C $p status --short 2>&1 | Out-String).Trim()
    if ($st) { Say ("  KEPT Codex worktree {0} (uncommitted changes):`n{1}" -f $w, $st) 'Yellow'; continue }
    git -C $Repo worktree remove $p 2>&1 | Out-Null
    if (-not (Test-Path -LiteralPath $p)) {
        if (-not (Get-ChildItem -LiteralPath $outer -Force)) { Remove-Item -LiteralPath $outer -Force }
        Say "  removed Codex worktree $w" 'Green'
    } else { Say "  KEPT Codex worktree ${w}: git worktree remove failed" 'Yellow' }
}
git -C $Repo worktree prune
if ((Test-Path -LiteralPath 'E:\rd-wt') -and -not (Get-ChildItem -LiteralPath 'E:\rd-wt' -Force)) { Remove-Item -LiteralPath 'E:\rd-wt' -Force; Say 'removed empty E:\rd-wt' 'Green' }
if ((Test-Path -LiteralPath $Stage) -and -not (Get-ChildItem -LiteralPath $Stage -Force)) { Remove-Item -LiteralPath $Stage -Force }
Say ''; Say ("DONE. git worktree list:`n" + ((git -C $Repo worktree list) -join "`n")) 'Green'
