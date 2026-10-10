<#
.SYNOPSIS
  Retire the last leftovers the main clean-up held. Owner runs it once (re-runnable).

.DESCRIPTION
  Targets: E:\rd-wt\{ws-packv3fix, pl57, pl58, pl59, j4bfix}, the rest of D:\rd-staging (label sidecars + shards_v3
  index files, ~50 MB), D:\research-dense (a pytest cache), the 3 Codex worktrees of this repo on C: (only if
  `git status` is empty) and the empty C:\Users\jpeng\.codex\worktrees\wm16-state-review.
  For every archived folder: (1) junctions/symlinks inside are removed as LINKS (never followed);
  (2) sensitive-named files (holdout, heldout, cal_split, chunk5, chunk-5, j4b, inc1, sealed, token "cal"; matched on the
      path INSIDE the folder) are moved unopened into sealed\relocated_20261010\<drive>\<folder>\..., SHA-256 checked;
  (3) the rest is archived whole as a .tar, uploaded to gdrive:research-archive/worktrees_final/<name>/ and verified with
      rclone check; (4) only then the folder is deleted. A failed step keeps that folder untouched.
  Log: data_ops\cleanup_out\finish_leftovers_<stamp>.log
#>
[CmdletBinding()]
param([switch]$Yes)
$ErrorActionPreference = 'Continue'   # native tools (rclone, git) print notices on stderr; every failure is checked explicitly
$Repo = 'E:\research-dense'
$RCLONE = 'C:\Users\jpeng\miniconda3\bin\rclone.exe'
$SealedRoot = 'E:\research-dense\sealed\relocated_20261010'
$Stage = 'E:\leftover_archive'
$Log = Join-Path $Repo ('data_ops\cleanup_out\finish_leftovers_{0}.log' -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$Sens = '(?i)(holdout|heldout|cal_split|chunk5|chunk-5|j4b|inc1|sealed)|(^|[\\/_.-])cal([\\/_.-]|$)'
$targets = @(
    @{ name = 'ws-packv3fix'; dir = 'E:\rd-wt\ws-packv3fix'; root = 'E:\rd-wt'; sealed = 'E\rd-wt\ws-packv3fix' },
    @{ name = 'pl57'; dir = 'E:\rd-wt\pl57'; root = 'E:\rd-wt'; sealed = 'E\rd-wt\pl57' },
    @{ name = 'pl58'; dir = 'E:\rd-wt\pl58'; root = 'E:\rd-wt'; sealed = 'E\rd-wt\pl58' },
    @{ name = 'pl59'; dir = 'E:\rd-wt\pl59'; root = 'E:\rd-wt'; sealed = 'E\rd-wt\pl59' },
    @{ name = 'j4bfix'; dir = 'E:\rd-wt\j4bfix'; root = 'E:\rd-wt'; sealed = 'E\rd-wt\j4bfix' },
    @{ name = 'rd-staging'; dir = 'D:\rd-staging'; root = 'D:\'; sealed = 'D\rd-staging' }
)

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

$todo = @($targets | Where-Object { Test-Path -LiteralPath $_.dir })
Say ("FINISH LEFTOVERS: {0}   log: {1}" -f (($todo | ForEach-Object { $_.dir }) -join ', '), $Log) 'Yellow'
if (-not $Yes) { $a = Read-Host 'Type FINISH to start'; if ($a -cne 'FINISH') { Say 'aborted - nothing changed.'; return } }
New-Item -ItemType Directory -Path $Stage -Force | Out-Null

foreach ($t in $todo) {
    $name = $t.name; $dir = $t.dir
    Say ''; Say "== $dir"
    try {
        $links = Unlink-Links $dir
        $moved = 0
        foreach ($f in (Files $dir)) {
            $rel = $f.Substring($dir.Length).TrimStart('\')
            if ($rel -notmatch $Sens) { continue }
            $to = Join-Path (Join-Path $SealedRoot $t.sealed) $rel
            if (Test-Path -LiteralPath $to) { throw "sealed destination exists: $to" }
            New-Item -ItemType Directory -Path (Split-Path -Parent $to) -Force -ErrorAction Stop | Out-Null
            $h = Sha $f
            if ($f.Substring(0, 2) -ieq $to.Substring(0, 2)) { [IO.File]::Move($f, $to) }
            else { [IO.File]::Copy($f, $to); if ((Sha $to) -ne $h) { throw "copy hash differs: $to" }; [IO.File]::Delete($f) }
            if ((Sha $to) -ne $h) { throw "hash changed after move: $to" }
            $moved++
        }
        Say ("  links unlinked: {0}; sensitive-named files moved to sealed: {1}" -f $links, $moved)
        $tarDir = Join-Path $Stage $name
        New-Item -ItemType Directory -Path $tarDir -Force -ErrorAction Stop | Out-Null
        $tar = Join-Path $tarDir "$name.tar"
        tar -cf $tar -C $t.root $name
        if ($LASTEXITCODE -ne 0) { throw "tar failed ($LASTEXITCODE)" }
        $remote = "gdrive:research-archive/worktrees_final/$name"
        cmd /c "`"$RCLONE`" copy `"$tarDir`" $remote 2>nul"
        if ($LASTEXITCODE -ne 0) { throw "upload failed ($LASTEXITCODE)" }
        cmd /c "`"$RCLONE`" check `"$tarDir`" $remote --one-way 2>nul"
        if ($LASTEXITCODE -ne 0) { throw "Drive check failed ($LASTEXITCODE)" }
        Say ("  archived {0}.tar ({1:N1} MB) -> {2} (verified)" -f $name, ((Get-Item $tar).Length / 1MB), $remote)
        if ($dir -like 'E:\rd-wt\*') { git -C $Repo worktree remove --force $dir 2>&1 | Out-Null }
        if (Test-Path -LiteralPath $dir) { Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction Stop }
        if (Test-Path -LiteralPath $dir) { throw "could not remove $dir" }
        Remove-Item -LiteralPath $tarDir -Recurse -Force
        Say "  removed $dir" 'Green'
    } catch {
        Say ("  KEPT {0}: {1}" -f $dir, $_.Exception.Message) 'Red'
    }
}

# small leftovers
if (Test-Path -LiteralPath 'D:\research-dense') { Remove-Item -LiteralPath 'D:\research-dense' -Recurse -Force; Say 'removed D:\research-dense (pytest cache)' 'Green' }
$wm16 = 'C:\Users\jpeng\.codex\worktrees\wm16-state-review'
if ((Test-Path -LiteralPath $wm16) -and -not (Get-ChildItem -LiteralPath $wm16 -Force)) { Remove-Item -LiteralPath $wm16 -Force; Say "removed empty $wm16" 'Green' }
foreach ($w in 'wmn1-eval-bridge', 'wmn1-j4-identity', 'wmn1-label-time-audit') {
    $outer = "C:\Users\jpeng\.codex\worktrees\$w"; $p = Join-Path $outer 'research-dense'
    if (-not (Test-Path -LiteralPath $p)) { continue }
    $st = (git -C $p status --short 2>&1 | Out-String).Trim()
    if ($st) { Say ("  KEPT Codex worktree {0} (uncommitted changes):`n{1}" -f $w, $st) 'Yellow'; continue }
    git -C $Repo worktree remove $p 2>&1 | Out-Null
    if (-not (Test-Path -LiteralPath $p)) {
        if (-not (Get-ChildItem -LiteralPath $outer -Force)) { Remove-Item -LiteralPath $outer -Force }
        Say "removed Codex worktree $w" 'Green'
    } else { Say "  KEPT Codex worktree ${w}: git worktree remove failed" 'Yellow' }
}
git -C $Repo worktree prune
if ((Test-Path -LiteralPath 'E:\rd-wt') -and -not (Get-ChildItem -LiteralPath 'E:\rd-wt' -Force)) { Remove-Item -LiteralPath 'E:\rd-wt' -Force; Say 'removed empty E:\rd-wt' 'Green' }
if ((Test-Path -LiteralPath $Stage) -and -not (Get-ChildItem -LiteralPath $Stage -Force)) { Remove-Item -LiteralPath $Stage -Force }
Say ''; Say ("DONE. git worktree list:`n" + ((git -C $Repo worktree list) -join "`n")) 'Green'
foreach ($d in 'C', 'D', 'E') { Say ("{0}: {1:N1} GiB free" -f $d, ((Get-PSDrive $d).Free / 1GB)) 'Green' }
