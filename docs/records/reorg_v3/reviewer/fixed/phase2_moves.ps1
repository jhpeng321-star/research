#Requires -Version 5.1
<#
  phase2_moves.ps1 -- conservative REORG_PLAN_V3 phase-2a copy ONLY.

  This copy does NOT solve zero-writer/executor shutdown or crash recovery.
  It still requires an owner-verified idle window for the entire operation;
  -Execute requires -OwnerConfirmedIdle. Process/handle scans are snapshots,
  not a writer lock. Phase 2b and phase 5 remain unsafe and are not executable.

  DRY-RUN BY DEFAULT: no journal, manifest or directory writes in dry-run.
  Execute appends durable before/after records using FileStream.Flush(true).
  If JournalDir is absent, a bootstrap journal in its nearest existing parent
  records each directory creation before the main journal can be opened.
  Keep BOTH journals. Child metadata is enumerated without following links;
  any source link, reparse ancestor, busy or inconclusive scan is refused.
  Supplied inventory KEEP paths are protected, even if the move map disagrees.

  There is NO atomic resume: an existing destination is a hard failure.
  A junction failure is logged, stops the run, and requires owner repair from
  the journal. No rollback, overwrite, force move, copy or deletion is provided.
  THIS COPY IS DELIVERED UNEXECUTED. The owner runs it after verifying idle.
#>
[CmdletBinding()]
param(
    [switch]$Execute,
    [ValidateSet('2a')]
    [string]$OnlyPhase = '2a',
    [switch]$OwnerConfirmedIdle,
    [string]$JournalDir = '',
    [string]$MoveMap = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($MoveMap)) {
    $MoveMap = Join-Path $PSScriptRoot '..\_in\MOVE_MAP.json'
}
if ([string]::IsNullOrWhiteSpace($JournalDir)) {
    $JournalDir = Join-Path $PSScriptRoot 'journal'
}
$InventoryPath = Join-Path $PSScriptRoot '..\_in\inventory_E.json'

# Hard-coded protections supplement the supplied inventory --------------------
$LiveWorktrees = @(
    'E:\rd-wt\ws-main', 'E:\rd-wt\ws-main2', 'E:\rd-wt\ws-brakefix',
    'E:\rd-wt\ws-trainsetup', 'E:\rd-wt\ws-wallc1'
)
$ProtectedPaths = $LiveWorktrees + @(
    'E:\research-dense', 'E:\research\repo', 'E:\rd-wt\reorg3',
    'E:\rd-wt\gpu1', 'E:\rd-wt\scorer-a9full', 'E:\rd-wt\feat1j4fix',
    'E:\rd-wt\bundle3', 'E:\rd-wt\ws-eval', 'E:\rd-wt\ws-evalshift',
    'E:\rd-wt\ws-p5', 'E:\rd-wt\ws-run3', 'E:\rd-wt\ws-swgen',
    'E:\rd-wt\ws-swpack', 'E:\rd-wt\ws-wall', 'E:\tmp', 'E:\research-github',
    'D:\rd-staging', 'G:\', 'C:\Users\jpeng\miniconda3\envs\rl_driving',
    'C:\Users\jpeng\Documents\Codex\tools\AgentControlPlane',
    'C:\Users\jpeng\.claude', 'C:\Users\jpeng\.codex\worktrees'
)
$RepoRoot = 'E:\research-dense'
$WorktreeRoot = 'E:\rd-wt'
$ResearchRoot = 'E:\research'
$Timestamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffffffZ')
$RunId = '{0}_{1}' -f $Timestamp, ([guid]::NewGuid().ToString('N'))
$JournalPath = $null
$BootstrapPath = $null

function Get-NormPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { throw 'empty path' }
    $p = $Path.Replace('/', '\')
    if ($p -notmatch '^[A-Za-z]:\\' -or $p -match '[*?\[\]<>"|\x00-\x1f]' -or
        $p.Substring(2).Contains(':')) { throw "invalid absolute literal path: $Path" }
    foreach ($part in $p.Substring(3).TrimEnd('\').Split('\')) {
        if ($part -eq '.' -or $part -eq '..' -or $part -match '[. ]$') {
            throw "dot segment or trailing dot/space in path: $Path"
        }
    }
    $full = [System.IO.Path]::GetFullPath($p)
    if ($full.Length -gt 3) { $full = $full.TrimEnd('\') }
    return $full
}

function Test-PathWithin {
    param([string]$Path, [string]$Root)
    return ($Path.Equals($Root, [StringComparison]::OrdinalIgnoreCase) -or
        $Path.StartsWith($Root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase))
}

function Test-PathOverlap {
    param([string]$A, [string]$B)
    return ((Test-PathWithin $A $B) -or (Test-PathWithin $B $A))
}

function Assert-SafePath {
    param([string]$Path, [ValidateSet('old', 'new')][string]$Label)
    $np = Get-NormPath $Path
    if ($np -notmatch '^E:\\.' -or $np.Length -le 3) {
        throw "$Label must be a nonroot E: path: $Path"
    }
    if ($Label -eq 'new' -and
        (-not (Test-PathWithin $np $ResearchRoot) -or $np -ieq $ResearchRoot)) {
        throw "new must be a strict E:\research descendant: $Path"
    }
    # rd-wt stays real, but individual idle ARCHIVE children may be moved.
    if (Test-PathWithin $WorktreeRoot $np) { throw "protecting worktree root/ancestor: $Path" }
    foreach ($keep in $ProtectedPaths) {
        if (Test-PathOverlap $np (Get-NormPath $keep)) {
            throw "protecting LIVE/KEEP path or ancestor: $Path overlaps $keep"
        }
    }
}

function Get-PathItem {
    param([string]$Path)
    try { return Get-Item -LiteralPath $Path -Force -ErrorAction Stop }
    catch [System.Management.Automation.ItemNotFoundException] { return $null }
}

function Write-Journal {
    param([hashtable]$Record)
    if (-not $Execute) { return }
    if (-not $JournalPath) { throw 'journal is not initialized' }
    $Record['ts'] = (Get-Date).ToUniversalTime().ToString('o')
    $Record['dry_run'] = $false
    $line = ($Record | ConvertTo-Json -Compress -Depth 8) + [Environment]::NewLine
    $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($line)
    $fs = [System.IO.FileStream]::new($JournalPath, [System.IO.FileMode]::Append,
        [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
    try { $fs.Write($bytes, 0, $bytes.Length); $fs.Flush($true) }
    finally { $fs.Dispose() }
}

function Assert-NoReparseAncestors {
    param([string]$Path)
    $parts = [System.Collections.Generic.Stack[string]]::new()
    $p = Get-NormPath $Path
    while ($p) {
        $parts.Push($p)
        $p = [System.IO.Path]::GetDirectoryName($p)
    }
    while ($parts.Count -gt 0) {
        $p = $parts.Pop()
        $item = Get-PathItem $p
        if ($null -ne $item) {
            if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                Write-Journal @{ event='refused_reparse_ancestor'; path=$p; requested=$Path }
                throw "reparse path/root/ancestor refused: $p"
            }
            if ($p -ine (Get-NormPath $Path) -and -not $item.PSIsContainer) {
                throw "ancestor is not a directory: $p"
            }
        }
    }
}

function New-JournaledDirectory {
    param([string]$Path)
    $pending = [System.Collections.Generic.Stack[string]]::new()
    $p = Get-NormPath $Path
    Assert-NoReparseAncestors $p
    while ($null -eq (Get-PathItem $p)) {
        $pending.Push($p)
        $p = [System.IO.Path]::GetDirectoryName($p)
        if (-not $p) { throw "no existing directory ancestor: $Path" }
    }
    if (-not (Get-PathItem $p).PSIsContainer) { throw "not a directory: $p" }
    while ($pending.Count -gt 0) {
        $p = $pending.Pop()
        Assert-NoReparseAncestors $p
        Write-Journal @{ event='before_directory_create'; path=$p }
        New-Item -ItemType Directory -Path $p -ErrorAction Stop | Out-Null
        Write-Journal @{ event='after_directory_create'; path=$p; status='ok' }
    }
}

function Get-SourceLinks {
    param([string]$Path)
    # Metadata only. Never recurse via Get-ChildItem -Recurse or enter a link.
    $pending = [System.Collections.Generic.Stack[string]]::new()
    $pending.Push($Path)
    while ($pending.Count -gt 0) {
        $directory = $pending.Pop()
        Assert-NoReparseAncestors $directory
        foreach ($item in (Get-ChildItem -LiteralPath $directory -Force -ErrorAction Stop)) {
            $reparse = [bool]($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)
            $linkType = $null
            if ($item.PSObject.Properties['LinkType']) { $linkType = $item.LinkType }
            $link = $reparse -or -not [string]::IsNullOrEmpty([string]$linkType)
            $record = @{ event='source_metadata'; source=$Path; path=$item.FullName;
                attributes=[string]$item.Attributes; directory=[bool]$item.PSIsContainer;
                reparse=$reparse; link=$link; last_write_utc=$item.LastWriteTimeUtc.ToString('o') }
            if (-not $item.PSIsContainer -and -not $link) { $record['length'] = $item.Length }
            if ($link) {
                foreach ($property in @('LinkType', 'Target')) {
                    if ($item.PSObject.Properties[$property]) { $record[$property] = $item.$property }
                }
            }
            Write-Journal $record
            if ($link) {
                Write-Host "[REFUSE LINK] $($item.FullName)" -ForegroundColor Yellow
                $item.FullName
            }
            elseif ($item.PSIsContainer) { $pending.Push($item.FullName) }
        }
    }
}

function Get-HandleExe {
    $commands = @(Get-Command handle.exe -CommandType Application -ErrorAction Stop)
    if ($commands.Count -ne 1) { throw 'handle.exe resolution is inconclusive' }
    return $commands[0].Source
}

function Test-SourceBusy {
    param([string]$Path)
    $reasons = [System.Collections.Generic.List[string]]::new()
    $norm = (Get-NormPath $Path).Replace('/', '\')
    try {
        $procs = @(Get-CimInstance Win32_Process -ErrorAction Stop)
        foreach ($p in $procs) {
            if ($p.ProcessId -eq $PID) { continue }
            foreach ($text in @($p.CommandLine, $p.ExecutablePath)) {
                if ($text -and $text.Replace('/', '\').IndexOf($norm,
                        [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                    $reasons.Add("process pid=$($p.ProcessId) name=$($p.Name) references source")
                    break
                }
            }
        }
    } catch { $reasons.Add("process scan inconclusive: $($_.Exception.Message)") }

    # No file-content/exclusive-open fallback. EULA must already be accepted.
    try {
        $handle = Get-HandleExe
        $out = @(& $handle -nobanner $norm 2>&1)
        $code = $LASTEXITCODE
        $text = ($out | ForEach-Object { [string]$_ }) -join "`n"
        if ($code -ne 0) { $reasons.Add("handle.exe exit=$code; inconclusive") }
        if ($text -match '(?im)access denied|error|failed|unable|could not|eula') {
            $reasons.Add('handle.exe reported an error; inconclusive')
        }
        $handleMatches = @($out | Where-Object { [string]$_ -match 'pid:\s*\d+' })
        foreach ($line in $handleMatches) { $reasons.Add("handle.exe: $line") }
        if ($handleMatches.Count -eq 0 -and $text -notmatch '(?im)^\s*No matching handles found\.?\s*$') {
            $reasons.Add('handle.exe did not explicitly report no matching handles; inconclusive')
        }
    } catch { $reasons.Add("handle.exe unavailable/failed; inconclusive: $($_.Exception.Message)") }
    return $reasons.ToArray()
}

function Invoke-GitChecked {
    param([string[]]$GitArgs)
    # Read-only status must not opportunistically refresh/write the index.
    $out = @(& git --no-optional-locks @GitArgs 2>&1)
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "git exit=${code}: $($GitArgs -join ' ')`n$($out -join "`n")" }
    return $out
}

function New-CompatJunction {
    param([string]$Old, [string]$New)
    Assert-NoReparseAncestors $Old
    Assert-NoReparseAncestors $New
    if ($null -ne (Get-PathItem $Old)) { throw "compat junction path already exists: $Old" }
    Write-Journal @{ event='before_junction_create'; old=$Old; new=$New }
    try {
        New-Item -ItemType Junction -Path $Old -Target $New -ErrorAction Stop | Out-Null
    } catch {
        Write-Journal @{ event='junction_create_failed'; old=$Old; new=$New;
            message=$_.Exception.Message; owner_action='repair from journal; do not rerun/resume' }
        throw "junction creation failed; owner must repair from journal: $Old -> $New. $($_.Exception.Message)"
    }
    Write-Journal @{ event='after_junction_create'; old=$Old; new=$New; status='ok' }
}

function Invoke-Move {
    param($M)
    $old = Get-NormPath ([string]$M.old)
    $new = Get-NormPath ([string]$M.new)
    $method = [string]$M.method
    if ($M.phase -ne '2a') { throw 'only conservative phase 2a is supported' }
    if ($method -notin @('rename (same volume)', 'git worktree move + git worktree repair')) {
        throw "unsupported phase-2a method: $method"
    }
    Assert-SafePath $old 'old'
    Assert-SafePath $new 'new'
    if ((Test-PathWithin $old $WorktreeRoot) -and
        $method -ne 'git worktree move + git worktree repair') {
        throw "worktrees cannot bypass git guards via a rename method: $old"
    }
    if (Test-PathOverlap $old $new) { throw "source/destination overlap: $old -> $new" }
    Assert-NoReparseAncestors $old
    Assert-NoReparseAncestors $new
    if ($null -ne (Get-PathItem $new)) { throw "destination already exists; no resume: $new" }
    $source = Get-PathItem $old
    if ($null -eq $source) { throw "source missing; refusing partial/resumed plan: $old" }
    if (-not $source.PSIsContainer) { throw "source must be a real directory: $old" }
    Write-Journal @{ event='source_root_metadata'; path=$old; attributes=[string]$source.Attributes }
    $links = @(Get-SourceLinks $old)
    if ($links.Count -gt 0) { throw "source contains links; refusing: $old" }

    if ($method -eq 'git worktree move + git worktree repair') {
        $porcelain = @(Invoke-GitChecked @('-C', $RepoRoot, 'worktree', 'list', '--porcelain'))
        $registered = @($porcelain | Where-Object { ([string]$_).StartsWith('worktree ', [StringComparison]::Ordinal) } |
            ForEach-Object { Get-NormPath (([string]$_).Substring(9)) })
        if ($registered -notcontains $old) { throw "not an exactly registered worktree: $old" }
        $st = @(Invoke-GitChecked @('-C', $old, 'status', '--porcelain=v1', '--untracked-files=all'))
        if ($st.Count -gt 0) { throw "worktree is dirty (including untracked), refusing: $old" }
    }
    $busy = @(Test-SourceBusy -Path $old)
    if ($busy.Count -gt 0) {
        Write-Journal @{ event='refused_busy'; order=$M.order; old=$old; reasons=$busy }
        throw "busy/inconclusive source scan refused: $old`n$($busy -join "`n")"
    }
    if (-not $Execute) {
        Write-Host ("[DRY] {0} -> {1} ({2}, {3:N2} GiB)" -f $old, $new, $method,
            ([double]$M.size_bytes / 1GB)) -ForegroundColor Cyan
        return
    }

    New-JournaledDirectory ([System.IO.Path]::GetDirectoryName($new))
    Assert-NoReparseAncestors $old
    Assert-NoReparseAncestors $new
    if ($null -ne (Get-PathItem $new)) { throw "destination appeared; refusing: $new" }
    Write-Journal @{ event='before_move'; order=$M.order; phase=$M.phase; old=$old; new=$new;
        method=$method; size_bytes=$M.size_bytes; class=$M.class }
    if ($method -eq 'git worktree move + git worktree repair') {
        Invoke-GitChecked @('-C', $RepoRoot, 'worktree', 'move', $old, $new) | Out-Null
        Write-Journal @{ event='after_worktree_move'; old=$old; new=$new; status='ok' }
        Write-Journal @{ event='before_worktree_repair'; old=$old; new=$new }
        Invoke-GitChecked @('-C', $new, 'worktree', 'repair') | Out-Null
        Write-Journal @{ event='after_worktree_repair'; old=$old; new=$new; status='ok' }
    }
    else {
        # Directory.Move also refuses an existing destination at mutation time;
        # Move-Item could instead nest the source inside a destination directory.
        [System.IO.Directory]::Move($old, $new)
        Write-Journal @{ event='after_rename'; old=$old; new=$new; status='ok' }
    }
    New-CompatJunction -Old $old -New $new
    Write-Journal @{ event='after_move'; order=$M.order; phase=$M.phase; old=$old; new=$new;
        method=$method; status='ok' }
}

# ---- main -------------------------------------------------------------------
try {
    if ($Execute -and -not $OwnerConfirmedIdle) {
        throw 'phase 2a Execute requires -OwnerConfirmedIdle and an owner-verified idle window'
    }
    $map = Get-Content -LiteralPath $MoveMap -Raw -Encoding UTF8 -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
    $inventory = Get-Content -LiteralPath $InventoryPath -Raw -Encoding UTF8 -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
    # Protect every supplied KEEP worktree, not merely the five LIVE names.
    foreach ($item in @($inventory.top_level) + @($inventory.worktrees)) {
        if (($item.proposed_class -like 'KEEP*') -and $item.path -ine 'E:/rd-wt') {
            $ProtectedPaths += Get-NormPath ([string]$item.path)
        }
    }
    $planned = @($map.moves | Where-Object { $_.phase -eq $OnlyPhase } |
        Sort-Object @{ Expression={ if ($_.method -eq 'rename (same volume)') { 0 } else { 1 } } },
            @{ Expression={ [int]$_.order } })
    $JournalDir = [System.IO.Path]::GetFullPath($JournalDir)
    Assert-NoReparseAncestors $JournalDir
    # Never place a journal in a tree that this run will move.
    foreach ($m in $planned) {
        if ((Test-PathOverlap (Get-NormPath $JournalDir) (Get-NormPath $m.old)) -or
            (Test-PathOverlap (Get-NormPath $JournalDir) (Get-NormPath $m.new))) {
            throw 'JournalDir overlaps a planned source/destination'
        }
    }
    if ($Execute) {
        $anchor = $JournalDir
        while ($null -eq (Get-PathItem $anchor)) {
            $anchor = [System.IO.Path]::GetDirectoryName($anchor)
            if (-not $anchor) { throw 'no existing parent for bootstrap journal' }
        }
        if (-not (Get-PathItem $anchor).PSIsContainer) { throw "journal parent is not a directory: $anchor" }
        if ($anchor -ine $JournalDir) {
            $BootstrapPath = Join-Path $anchor ("phase2_{0}.bootstrap.jsonl" -f $RunId)
            $JournalPath = $BootstrapPath
            Write-Journal @{ event='bootstrap_start'; journal_dir=$JournalDir }
            New-JournaledDirectory $JournalDir
        }
        $JournalPath = Join-Path $JournalDir ("phase2_{0}.jsonl" -f $RunId)
    }
    Write-Host ("phase2_moves: {0} planned item(s); Execute={1}; phase='{2}'" -f
        $planned.Count, $Execute, $OnlyPhase) -ForegroundColor Green
    Write-Journal @{ event='run_start'; execute=[bool]$Execute; only_phase=$OnlyPhase;
        owner_confirmed_idle=[bool]$OwnerConfirmedIdle; count=$planned.Count;
        move_map=$MoveMap; keep_inventory=$InventoryPath; bootstrap_journal=$BootstrapPath }
    # The per-item manifest lives in the durable journal, not a separate file.
    foreach ($m in $planned) {
        Write-Journal @{ event='manifest_item'; order=$m.order; phase=$m.phase; old=$m.old;
            new=$m.new; size_bytes=$m.size_bytes; method=$m.method; class=$m.class }
    }
    foreach ($m in $planned) { Invoke-Move -M $m }
    Write-Journal @{ event='run_end'; failures=0; executed=[bool]$Execute }
    Write-Host ("done; failures=0; journal={0}; bootstrap={1}" -f $JournalPath, $BootstrapPath) -ForegroundColor Green
}
catch {
    $message = $_.Exception.Message
    if ($Execute -and $JournalPath) {
        try { Write-Journal @{ event='error'; message=$message; status='stopped'; owner_action='inspect journals; repair manually before any further run' } }
        catch { Write-Host "[ERROR] could not append failure journal: $($_.Exception.Message)" -ForegroundColor Red }
    }
    Write-Host "[ERROR] stopped: $message" -ForegroundColor Red
    exit 1
}
