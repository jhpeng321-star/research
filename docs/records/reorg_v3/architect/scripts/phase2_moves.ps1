#Requires -Version 5.1
<#
  phase2_moves.ps1  --  REORG_PLAN_V3 phase-2 move executor.

  SAFETY MODEL
    * DRY-RUN BY DEFAULT. Nothing is moved unless -Execute is passed.
    * A journal record is written (and flushed) BEFORE every mutation, and a
      matching 'after' record is written afterwards. Journal:
        <JournalDir>\phase2_<yyyyMMddTHHmmssZ>.jsonl
    * Every path comes from MOVE_MAP.json (explicit enumeration). Paths that
      contain wildcard characters are rejected.
    * Refuses to move anything under a LIVE worktree (hard-coded list).
    * Refuses if a process appears to hold files open under a source
      (handle.exe if available, else a process-cmdline scan + exclusive-open
      probe). Refuses dirty / unregistered worktrees.
    * E:/rd-wt is kept as a REAL directory; per-worktree junctions are created
      at the old paths. The main git tree rename (phase 2b) additionally
      requires -AllowHighRisk and a confirmed zero-writer window.

  THIS SCRIPT IS DELIVERED UNEXECUTED (planning only). The owner runs it.
#>
[CmdletBinding()]
param(
    [switch]$Execute,
    [string]$OnlyPhase = "",
    [switch]$AllowHighRisk,
    [string]$JournalDir = "E:\rd-wt\reorg3\architect\journal",
    [string]$MoveMap = "E:\rd-wt\reorg3\architect\MOVE_MAP.json",
    [int]$ProbeFileLimit = 4000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Hard-coded safety lists -----------------------------------------------------
$LiveWorktrees = @(
    'E:\rd-wt\ws-main', 'E:\rd-wt\ws-main2', 'E:\rd-wt\ws-brakefix',
    'E:\rd-wt\ws-trainsetup', 'E:\rd-wt\ws-wallc1'
)
$RepoRoot   = 'E:\research-dense'
$AllowedRoots = @('E:\', 'E:/', 'D:\', 'D:/', 'C:\', 'C:/')
$WildcardChars = @('*', '?', '[', ']')

function Get-NormPath { param([string]$Path) return ($Path -replace '/', '\').TrimEnd('\') }

$Timestamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
if (-not (Test-Path -LiteralPath $JournalDir)) {
    New-Item -ItemType Directory -Path $JournalDir -Force | Out-Null
}
$JournalPath = Join-Path $JournalDir ("phase2_{0}.jsonl" -f $Timestamp)
$ManifestPath = Join-Path $JournalDir ("manifest_{0}.json" -f $Timestamp)

function Write-Journal {
    param([hashtable]$Record)
    $Record['ts'] = (Get-Date).ToUniversalTime().ToString('o')
    $Record['dry_run'] = (-not $Execute)
    $line = ($Record | ConvertTo-Json -Compress -Depth 6)
    [System.IO.File]::AppendAllText($JournalPath, $line + [Environment]::NewLine)
}

function Assert-SafePath {
    param([string]$Path, [string]$Label)
    if ([string]::IsNullOrWhiteSpace($Path)) { throw "empty $Label" }
    foreach ($c in $WildcardChars) { if ($Path.Contains($c)) { throw "wildcard '$c' in ${Label}: $Path" } }
    $ok = $false
    foreach ($r in $AllowedRoots) { if ($Path.StartsWith($r, [StringComparison]::OrdinalIgnoreCase)) { $ok = $true; break } }
    if (-not $ok) { throw "$Label outside allowed roots: $Path" }
}

function Test-UnderLiveWorktree {
    param([string]$Path)
    $np = Get-NormPath $Path
    foreach ($lw in $LiveWorktrees) {
        $nlw = Get-NormPath $lw
        if ($np -ieq $nlw) { return $true }
        if ($np.StartsWith($nlw + '\', [StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    return $false
}

function Get-HandleExe {
    $c = Get-Command handle.exe -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    $p = 'C:\SysinternalsSuite\handle.exe'
    if (Test-Path -LiteralPath $p) { return $p }
    return $null
}

function Test-SourceBusy {
    param([string]$Path, [int]$Limit)
    $reasons = New-Object System.Collections.Generic.List[string]

    # (1) process command line / image path scan
    try {
        $norm = $Path.TrimEnd('\')
        $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
                 Where-Object { $_.CommandLine -and $_.CommandLine -like "*$norm*" }
        foreach ($p in $procs) {
            if ($p.ProcessId -ne $PID) { $reasons.Add("process pid=$($p.ProcessId) name=$($p.Name) cmdline references source") }
        }
    } catch { $reasons.Add("process scan unavailable: $($_.Exception.Message)") }

    # (2) handle.exe (authoritative when present)
    $handle = Get-HandleExe
    if ($handle) {
        try {
            $out = & $handle -accepteula -nobanner $Path 2>$null
            if ($out) {
                foreach ($l in $out) {
                    if ($l -match 'pid:\s*(\d+)') { $reasons.Add("handle.exe: $l") }
                }
            }
        } catch { $reasons.Add("handle.exe failed: $($_.Exception.Message)") }
    }
    else {
        # (3) fallback exclusive-open probe (bounded)
        $probed = 0
        try {
            Get-ChildItem -LiteralPath $Path -Recurse -File -Force -ErrorAction SilentlyContinue |
              Select-Object -First $Limit | ForEach-Object {
                $probed++
                try {
                    $fs = [System.IO.File]::Open($_.FullName, [System.IO.FileMode]::Open,
                            [System.IO.FileAccess]::Read, [System.IO.FileShare]::None)
                    $fs.Close(); $fs.Dispose()
                } catch {
                    $reasons.Add("exclusive-open failed: $($_.FullName)")
                }
            }
        } catch { $reasons.Add("probe failed: $($_.Exception.Message)") }
        if ($probed -ge $Limit) {
            $reasons.Add("probe hit limit ($Limit files); treat as inconclusive -> BUSY")
        }
    }
    return $reasons
}

function New-CompatJunction {
    param([string]$Old, [string]$New)
    if (Test-Path -LiteralPath $Old) {
        $item = Get-Item -LiteralPath $Old -Force
        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            Write-Journal @{ event='junction_exists'; old=$Old; new=$New }
            return
        }
        throw "compat junction target already exists and is not a reparse point: $Old"
    }
    if ($Execute) {
        New-Item -ItemType Junction -Path $Old -Target $New | Out-Null
    }
    Write-Journal @{ event='junction_create'; old=$Old; new=$New; executed=$Execute }
}

function Invoke-Move {
    param($M)
    $old = [string]$M.old; $new = [string]$M.new; $method = [string]$M.method
    $size = $M.size_bytes

    Assert-SafePath -Path $old -Label 'old'
    Assert-SafePath -Path $new -Label 'new'

    if ($M.phase -eq '2b' -and -not $AllowHighRisk) {
        Write-Journal @{ event='gated'; order=$M.order; old=$old; reason='phase 2b requires -AllowHighRisk' }
        Write-Host "[GATED] phase2b $old -> $new (pass -AllowHighRisk after a confirmed zero-writer window)" -ForegroundColor Yellow
        return
    }
    if (Test-UnderLiveWorktree -Path $old) {
        Write-Journal @{ event='refused_live_worktree'; order=$M.order; old=$old }
        throw "refusing to move LIVE worktree path: $old"
    }
    if (-not (Test-Path -LiteralPath $old)) {
        Write-Journal @{ event='skipped_missing_source'; order=$M.order; old=$old }
        Write-Host "[SKIP] source missing: $old" -ForegroundColor DarkYellow
        return
    }
    if (Test-Path -LiteralPath $new) {
        throw "destination already exists: $new"
    }

    # git worktree guards
    if ($method -like 'git worktree move*') {
        $name = Split-Path -Leaf $old
        $porcelain = & git -C $RepoRoot worktree list --porcelain 2>$null
        if (-not ($porcelain -match [regex]::Escape($old))) {
            throw "not a registered worktree: $old"
        }
        $st = & git -C $old status --porcelain 2>$null
        if ($st) { throw "worktree is dirty, refusing: $old" }
    }

    # open-handle refusal
    $busy = Test-SourceBusy -Path $old -Limit $ProbeFileLimit
    if ($busy.Count -gt 0 -and -not ($method -like 'rename (same volume)')) {
        Write-Journal @{ event='refused_busy'; order=$M.order; old=$old; reasons=@($busy) }
        throw "refusing: files open under $old`n$($busy -join "`n")"
    }
    if ($busy.Count -gt 0) {
        Write-Journal @{ event='busy_warning'; order=$M.order; old=$old; reasons=@($busy) }
        Write-Host "[WARN] process evidence under $old; review before -Execute" -ForegroundColor Yellow
        if ($Execute) { throw "refusing: files open under $old`n$($busy -join "`n")" }
    }

    # JOURNAL BEFORE MUTATION
    Write-Journal @{ event='before_move'; order=$M.order; phase=$M.phase; old=$old; new=$new;
                     method=$method; size_bytes=$size; class=$M.class; executed=$Execute }

    if ($Execute) {
        $parent = Split-Path -Parent $new
        if ($parent -and -not (Test-Path -LiteralPath $parent)) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        if ($method -like 'git worktree move*') {
            & git -C $RepoRoot worktree move $old $new
            if ($LASTEXITCODE -ne 0) { throw "git worktree move failed: $old" }
            & git -C $new worktree repair
        }
        elseif ($method -like 'robocopy*') {
            & robocopy $old $new /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /NFL /NDL /NP | Out-Null
            if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE): $old" }
            $srcCount = (Get-ChildItem -LiteralPath $old -Recurse -File -Force).Count
            $dstCount = (Get-ChildItem -LiteralPath $new -Recurse -File -Force).Count
            if ($srcCount -ne $dstCount) { throw "copy file-count mismatch $srcCount != $dstCount for $old" }
        }
        else {
            Move-Item -LiteralPath $old -Destination $new
        }
        New-CompatJunction -Old $old -New $new
    }
    else {
        Write-Host ("[DRY] {0} -> {1}  ({2}, {3:N2} GiB)" -f $old, $new, $method,
                    ([double]$size / 1GB)) -ForegroundColor Cyan
    }

    # JOURNAL AFTER MUTATION
    Write-Journal @{ event='after_move'; order=$M.order; phase=$M.phase; old=$old; new=$new;
                     method=$method; executed=$Execute; status='ok' }
}

# ---- main -------------------------------------------------------------------
$map = Get-Content -LiteralPath $MoveMap -Raw -Encoding UTF8 | ConvertFrom-Json
if ($OnlyPhase) { $planned = @($map.moves | Where-Object { $_.phase -eq $OnlyPhase }) }
else            { $planned = @($map.moves | Where-Object { $_.phase -ne '2b' }) }  # 2b never implicit

Write-Host ("phase2_moves: {0} planned item(s); Execute={1}; phase-filter='{2}'" -f `
            $planned.Count, $Execute, $OnlyPhase) -ForegroundColor Green
Write-Journal @{ event='run_start'; execute=$Execute; only_phase=$OnlyPhase; count=$planned.Count;
                 move_map=$MoveMap }

# per-item manifest BEFORE any mutation
$manifest = $planned | ForEach-Object {
    [pscustomobject]@{ order=$_.order; phase=$_.phase; old=$_.old; new=$_.new;
                       size_bytes=$_.size_bytes; method=$_.method; class=$_.class }
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8
Write-Host "manifest: $ManifestPath" -ForegroundColor Green

$failures = 0
foreach ($m in ($planned | Sort-Object order)) {
    try { Invoke-Move -M $m }
    catch {
        $failures++
        Write-Journal @{ event='error'; order=$m.order; old=$m.old; message=$_.Exception.Message }
        Write-Host "[ERROR] $($m.old): $($_.Exception.Message)" -ForegroundColor Red
    }
}
Write-Journal @{ event='run_end'; failures=$failures; executed=$Execute }
Write-Host ("done; failures={0}; journal={1}" -f $failures, $JournalPath) -ForegroundColor Green
