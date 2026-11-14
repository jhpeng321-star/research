#Requires -Version 5.1
<#
  DELETE_VERIFIED_LOCAL.ps1                      (OWNER-RUN)

  Removes a LOCAL item only when a passing Drive backup receipt proves that a
  verified copy already exists on `gdrive:`. REPORT-ONLY by default: nothing is
  removed unless -Execute is supplied.

  Eligibility (all must hold)
    * receipt is receipts\<name>.json with schema driveops_receipt/1,
      check = "passed" and class in DELETE-CANDIDATE / DRIVE-ONLY;
      KEEP-ACTIVE-backup-only and ARCHIVE items are never listed.
    * path is a literal path (no wildcards), not excluded, and not referenced by
      active code/config found under E:\research-dense and E:\rd-wt\ws-r3-*.
    * a no-follow scan finds no junction/symlink inside the item and no reparse
      ancestor (otherwise the item is reported "needs manual link handling").

  Exclusions
    E:\rd-wt\reorg3 (workspace, receipts, scripts, journal), E:\research-dense,
    E:\rd-wt\ws-* (live worktrees), the live packs pack_v3 / pack_dspeed /
    pack_damping, and any path component named sealed / CAL / HOLDOUT / chunk-5 /
    chunk5 / INC1. This script never deletes remote Drive objects.

  Re-verification immediately before each delete
    * local tree still exists, its total size equals the receipt, and its file
      set + size + MD5 still match receipts\md5\<name>.md5 (detects mutation);
    * the current local tree has no reparse points and no reparse ancestor;
    * the Drive copy still exists at the receipt's path with matching size+MD5
      via `rclone check --one-way --checksum` (dirs) / `rclone md5sum`+size
      (loose files).

  Audit
    Every intent and completion is appended and flushed to
    journal\delete_<utcstamp>.jsonl; a JSON result manifest is written next to it.
    The final console line reports bytes ACTUALLY removed (0 in report-only mode).

  Usage
    powershell -ExecutionPolicy Bypass -File DELETE_VERIFIED_LOCAL.ps1
    powershell -ExecutionPolicy Bypass -File DELETE_VERIFIED_LOCAL.ps1 -Execute
#>
[CmdletBinding()]
param(
    [switch]$Execute,
    [string]$WorkspaceRoot = '',
    [string]$ReceiptsDir = '',
    [string]$JournalDir = '',
    [string]$RcloneExe = 'C:\Users\jpeng\miniconda3\bin\rclone.exe',
    [string[]]$Only = @(),
    [string[]]$ReferenceRoots = @('E:\research-dense'),
    [string]$S4RefsJson = 'E:\rd-wt\reorg3\s4_refs\path_refs.json',
    [switch]$SkipReferenceScan,
    [int]$RcloneTimeoutSec = 7200
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$EligibleClasses = @('DELETE-CANDIDATE', 'DRIVE-ONLY')
$WildcardChars = @('*', '?', '[', ']')

$scriptRoot = $PSScriptRoot
if (-not $scriptRoot) { $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $WorkspaceRoot) { $WorkspaceRoot = $scriptRoot }
if (-not $ReceiptsDir) { $ReceiptsDir = Join-Path $WorkspaceRoot 'receipts' }
if (-not $JournalDir) { $JournalDir = Join-Path $WorkspaceRoot 'journal' }

$ProtectedRoots = @(
    $WorkspaceRoot,
    'E:\rd-wt\reorg3'
)
$ProtectedTokens = @('sealed', 'CAL', 'HOLDOUT', 'chunk-5', 'chunk5', 'INC1')
$LivePackNames = @('pack_v3', 'pack_dspeed', 'pack_damping')

function Write-Log {
    param([string]$Message, [string]$Color = 'Gray')
    Write-Host $Message -ForegroundColor $Color
}

function Get-Norm {
    param([string]$Path)
    return (($Path -replace '/', '\').TrimEnd('\'))
}

function Get-RepoFiles {
    param([string]$Root)
    $rootNorm = Get-Norm $Root
    $stack = New-Object System.Collections.Stack
    $stack.Push($rootNorm)
    $out = New-Object System.Collections.ArrayList
    while ($stack.Count -gt 0) {
        $cur = [string]$stack.Pop()
        foreach ($e in [System.IO.Directory]::EnumerateFileSystemEntries($cur)) {
            $attr = [System.IO.File]::GetAttributes($e)
            if ($attr -band [System.IO.FileAttributes]::ReparsePoint) {
                throw "reparse point inside item: $e"
            }
            if ($attr -band [System.IO.FileAttributes]::Directory) { $stack.Push($e) }
            else { [void]$out.Add($e) }
        }
    }
    Write-Output -NoEnumerate $out
}

function Get-ReparseAncestor {
    param([string]$Path, [string]$StopAt)
    if (-not $Path) { return $null }
    $cur = Get-Norm $Path
    $stop = if ($StopAt) { (Get-Norm $StopAt) } else { $null }
    while ($cur) {
        if ($cur -match '^[A-Za-z]:$') { break }
        if (-not (Test-Path -LiteralPath $cur)) { break }
        $attr = [System.IO.File]::GetAttributes($cur)
        if ($attr -band [System.IO.FileAttributes]::ReparsePoint) { return $cur }
        if ($stop -and $cur -ieq $stop) { break }
        $parent = Split-Path -Parent $cur
        if (-not $parent -or $parent -eq $cur) { break }
        $cur = $parent
    }
    return $null
}

function Test-PathUnder {
    param([string]$Path, [string]$Root)
    $p = Get-Norm $Path; $r = Get-Norm $Root
    return (($p -ieq $r) -or ($p.StartsWith($r + '\', [System.StringComparison]::OrdinalIgnoreCase)))
}

function Test-Excluded {
    param([string]$Path)
    $p = Get-Norm $Path
    foreach ($r in $ProtectedRoots) { if ($r -and (Test-PathUnder $p $r)) { return "protected workspace root: $r" } }
    if (Test-PathUnder $p 'E:\research-dense') { return 'repo E:\research-dense is excluded' }
    $comps = $p -split '\\'
    foreach ($c in $comps) {
        if ($ProtectedTokens -contains $c) { return "protected token component: $c" }
        if ($LivePackNames -contains $c) { return "live pack: $c" }
        if ($c -like 'ws-*' -and (Test-PathUnder $p 'E:\rd-wt')) { return "live worktree: $c" }
    }
    return $null
}

function Get-ReferenceRoots {
    $roots = New-Object System.Collections.ArrayList
    foreach ($r in $ReferenceRoots) { if (Test-Path -LiteralPath $r) { [void]$roots.Add((Get-Norm $r)) } }
    $rdwt = 'E:\rd-wt'
    if (Test-Path -LiteralPath $rdwt) {
        foreach ($d in Get-ChildItem -LiteralPath $rdwt -Directory -Force -ErrorAction SilentlyContinue) {
            if ($d.Name -like 'ws-r3-*') { [void]$roots.Add((Get-Norm $d.FullName)) }
        }
    }
    Write-Output -NoEnumerate $roots
}

function Test-Referenced {
    param([string]$Path, [string[]]$ScanRoots)
    $fwd = [regex]::Escape((Get-Norm $Path).Replace('\', '/'))
    $bwd = [regex]::Escape((Get-Norm $Path))
    $boundary = '\b'
    $patterns = @(($fwd + $boundary), ($bwd + $boundary))
    $globs = @('*.py', '*.ps1', '*.sh', '*.json', '*.toml', '*.yaml', '*.yml', '*.cfg', '*.ini', '*.bat', '*.cmd')
    $exts = @($globs | ForEach-Object { $_.Substring(1) })
    $rg = Get-Command rg -ErrorAction SilentlyContinue
    $hits = New-Object System.Collections.ArrayList
    if ($rg) {
        $rgArgs = @('-i', '-l', '--no-ignore', '--max-filesize', '10M')
        foreach ($g in $globs) { $rgArgs += @('--glob', $g) }
        $rgArgs += @('--glob', '!.git/**')
        foreach ($pt in $patterns) { $rgArgs += @('-e', $pt) }
        $rgArgs += $ScanRoots
        $out = & $rg.Source @rgArgs 2>$null
        if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq 1) {
            foreach ($line in @($out)) { if ($line) { [void]$hits.Add($line) } }
        }
    } else {
        foreach ($root in $ScanRoots) {
            foreach ($f in Get-ChildItem -LiteralPath $root -Recurse -File -Force -ErrorAction SilentlyContinue) {
                if ($f.FullName -match '\\\.git\\') { continue }
                if ($exts -notcontains $f.Extension) { continue }
                try {
                    if (Select-String -LiteralPath $f.FullName -Pattern $patterns -List -ErrorAction SilentlyContinue) { [void]$hits.Add($f.FullName) }
                } catch { }
            }
        }
    }
    Write-Output -NoEnumerate $hits
}

function Test-ReferencedInS4Refs {
    param([string]$Path, [string]$RefsJsonPath)
    $hits = New-Object System.Collections.ArrayList
    if (-not $RefsJsonPath -or -not (Test-Path -LiteralPath $RefsJsonPath)) { Write-Output -NoEnumerate $hits; return }
    try { $d = Get-Content -LiteralPath $RefsJsonPath -Raw | ConvertFrom-Json } catch { Write-Output -NoEnumerate $hits; return }
    $needleB = (Get-Norm $Path).ToLower()
    $needleF = $needleB.Replace('\', '/')
    foreach ($e in @($d.active_code_files)) {
        if (-not $e.literals) { continue }
        foreach ($lit in @($e.literals)) {
            $l = ([string]$lit).ToLower()
            if ($l.Contains($needleB) -or $l.Contains($needleF)) { [void]$hits.Add("s4_refs:$($e.tree):$($e.file)"); break }
        }
    }
    Write-Output -NoEnumerate $hits
}

function Read-Md5Manifest {
    param([string]$ManifestPath)
    $map = @{}
    if (-not (Test-Path -LiteralPath $ManifestPath)) { return $null }
    foreach ($line in Get-Content -LiteralPath $ManifestPath) {
        if (-not $line.Trim()) { continue }
        $m = [regex]::Match($line, '^([0-9a-fA-F]{32})\s+(\d+)\s+(.*)$')
        if (-not $m.Success) { return $null }
        $rel = $m.Groups[3].Value.Trim().Replace('\', '/')
        $map[$rel] = @{ md5 = $m.Groups[1].Value.ToLower(); size = [long]$m.Groups[2].Value }
    }
    return $map
}

function Get-LocalMd5 {
    param([string]$File)
    $h = [System.Security.Cryptography.MD5]::Create()
    try {
        $fs = [System.IO.File]::Open($File, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
        try { return ([BitConverter]::ToString($h.ComputeHash($fs)) -replace '-', '').ToLower() } finally { $fs.Dispose() }
    } finally { $h.Dispose() }
}

function Invoke-Rclone {
    param([string[]]$RcloneArgs)
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $RcloneExe
    $quoted = foreach ($a in $RcloneArgs) {
        if ($a -match '[\s"]') { '"' + ($a -replace '"', '\"') + '"' } else { $a }
    }
    $psi.Arguments = ($quoted -join ' ')
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    $p = [System.Diagnostics.Process]::Start($psi)
    $outTask = $p.StandardOutput.ReadToEndAsync()
    $errTask = $p.StandardError.ReadToEndAsync()
    if (-not $p.WaitForExit($RcloneTimeoutSec * 1000)) {
        try { $p.Kill() } catch { }
        return @{ rc = 124; out = ''; err = "rclone timeout after $RcloneTimeoutSec s" }
    }
    return @{ rc = $p.ExitCode; out = $outTask.Result; err = $errTask.Result }
}

function Test-RemoteDir {
    param([string]$LocalDir, [string]$RemoteDir)
    $r = Invoke-Rclone @('check', '--one-way', '--checksum', '--checkers', '4', $LocalDir, $RemoteDir)
    if ($r.rc -ne 0) { return @{ ok = $false; detail = (($r.err + ' ' + $r.out).Trim() -replace '\s+', ' ').Substring(0, [Math]::Min(400, (($r.err + ' ' + $r.out).Trim() -replace '\s+', ' ').Length)) } }
    if (($r.out + $r.err) -match 'differences found') {
        $mm = [regex]::Match(($r.out + $r.err), '(\d+) differences? found')
        if ($mm.Success -and [int]$mm.Groups[1].Value -ne 0) { return @{ ok = $false; detail = "remote differs: $($mm.Groups[1].Value) differences" } }
    }
    return @{ ok = $true; detail = '' }
}

function Test-RemoteFile {
    param([string]$LocalFile, [string]$RemoteFile)
    $ls = Invoke-Rclone @('lsf', '--format', 's', $RemoteFile)
    if ($ls.rc -ne 0 -or -not $ls.out.Trim()) { return @{ ok = $false; detail = 'remote file missing' } }
    $remoteSize = [long]($ls.out.Trim().Split("`n")[0].Trim())
    $localSize = (Get-Item -LiteralPath $LocalFile -Force).Length
    if ($remoteSize -ne $localSize) { return @{ ok = $false; detail = "size differs remote=$remoteSize local=$localSize" } }
    $ms = Invoke-Rclone @('md5sum', $RemoteFile)
    if ($ms.rc -ne 0 -or -not $ms.out.Trim()) { return @{ ok = $false; detail = 'remote md5sum unavailable' } }
    $remoteMd5 = ($ms.out.Trim().Split("`n")[0].Trim() -split '\s+')[0].ToLower()
    $localMd5 = Get-LocalMd5 $LocalFile
    if ($remoteMd5 -ne $localMd5) { return @{ ok = $false; detail = "md5 differs remote=$remoteMd5 local=$localMd5" } }
    return @{ ok = $true; detail = '' }
}

function Remove-TreeNoFollow {
    param([string]$Root)
    $files = Get-RepoFiles $Root
    $dirs = New-Object System.Collections.ArrayList
    $stack = New-Object System.Collections.Stack
    $stack.Push((Get-Norm $Root))
    while ($stack.Count -gt 0) {
        $cur = [string]$stack.Pop()
        [void]$dirs.Add($cur)
        foreach ($e in [System.IO.Directory]::EnumerateFileSystemEntries($cur)) {
            $attr = [System.IO.File]::GetAttributes($e)
            if ($attr -band [System.IO.FileAttributes]::ReparsePoint) { throw "refusing: reparse point appeared at $e" }
            if ($attr -band [System.IO.FileAttributes]::Directory) { $stack.Push($e) }
        }
    }
    $bytes = [long]0
    foreach ($f in $files) {
        $bytes += [long](Get-Item -LiteralPath $f -Force).Length
        [System.IO.File]::Delete($f)
    }
    foreach ($d in ($dirs | Sort-Object { $_.Length } -Descending)) {
        if (Test-Path -LiteralPath $d) { [System.IO.Directory]::Delete($d, $false) }
    }
    if (Test-Path -LiteralPath $Root) { [System.IO.Directory]::Delete((Get-Norm $Root), $false) }
    return $bytes
}

# ---------------------------------------------------------------------------

if (-not (Test-Path -LiteralPath $RcloneExe)) { throw "rclone not found: $RcloneExe" }
if (-not (Test-Path -LiteralPath $ReceiptsDir)) { throw "receipts dir not found: $ReceiptsDir" }
if (-not (Test-Path -LiteralPath $JournalDir)) { New-Item -ItemType Directory -Path $JournalDir -Force | Out-Null }

$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$JournalPath = Join-Path $JournalDir ("delete_{0}.jsonl" -f $stamp)
$ResultPath = Join-Path $JournalDir ("delete_result_{0}.json" -f $stamp)

function Write-Journal {
    param([hashtable]$Record)
    $Record['ts'] = (Get-Date).ToUniversalTime().ToString('o')
    $Record['execute'] = [bool]$Execute
    $line = ($Record | ConvertTo-Json -Compress -Depth 8)
    $fs = [System.IO.File]::Open($JournalPath, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
    try {
        $sw = New-Object System.IO.StreamWriter($fs, (New-Object System.Text.UTF8Encoding($false)))
        try { $sw.WriteLine($line); $sw.Flush(); $fs.Flush($true) } finally { $sw.Dispose() }
    } finally { $fs.Dispose() }
}

$scanRoots = @()
if (-not $SkipReferenceScan) { $scanRoots = Get-ReferenceRoots }

$receiptFiles = Get-ChildItem -LiteralPath $ReceiptsDir -Filter '*.json' -File | Sort-Object Name
$results = New-Object System.Collections.ArrayList
$bytesDeleted = [long]0

Write-Log ("DELETE_VERIFIED_LOCAL  mode={0}" -f $(if ($Execute) { 'EXECUTE' } else { 'REPORT-ONLY' })) 'Cyan'
Write-Log ("journal: {0}" -f $JournalPath)
Write-Journal @{ event = 'run_start'; mode = $(if ($Execute) { 'execute' } else { 'report' }); scan_roots = $scanRoots }

foreach ($rf in $receiptFiles) {
    $rec = $null
    try { $rec = Get-Content -LiteralPath $rf.FullName -Raw | ConvertFrom-Json } catch { continue }
    if (-not $rec -or $rec.schema -ne 'driveops_receipt/1') { continue }
    $name = [string]$rec.name
    if ($Only.Count -gt 0 -and ($Only -notcontains $name)) { continue }
    if ([string]$rec.check -ne 'passed') { continue }
    if ($EligibleClasses -notcontains [string]$rec.class) { continue }

    $locals = @()
    if ($rec.local -is [System.Array]) { $locals = @($rec.local) } else { $locals = @([string]$rec.local) }

    foreach ($localPath in $locals) {
        $p = [string]$localPath
        $p = Get-Norm $p
        $item = [ordered]@{ name = $name; class = [string]$rec.class; path = $p; drive = [string]$rec.drive; action = ''; reason = ''; bytes = 0 }
        try {
            $bad = $false
            foreach ($c in $WildcardChars) { if ($p.Contains($c)) { $item.action = 'rejected'; $item.reason = 'wildcard in literal path'; $bad = $true; break } }
            if ($bad) { [void]$results.Add($item); Write-Log ("REJECT  {0}  {1}" -f $p, $item.reason) 'Red'; continue }

            if (-not (Test-Path -LiteralPath $p)) { $item.action = 'missing'; $item.reason = 'not present'; [void]$results.Add($item); Write-Log ("MISSING {0}" -f $p) 'DarkGray'; continue }

            $excl = Test-Excluded -Path $p
            if ($excl) { $item.action = 'excluded'; $item.reason = $excl; [void]$results.Add($item); Write-Log ("EXCLUDE {0}  ({1})" -f $p, $excl) 'Yellow'; continue }

            $selfAttr = (Get-Item -LiteralPath $p -Force).Attributes
            if ($selfAttr -band [System.IO.FileAttributes]::ReparsePoint) {
                $item.action = 'needs_manual_link_handling'; $item.reason = 'item itself is a reparse point'
                [void]$results.Add($item); Write-Log ("LINK    {0}  {1}" -f $p, $item.reason) 'Yellow'; continue
            }
            $anc = Get-ReparseAncestor -Path (Split-Path -Parent $p)
            if ($anc) { $item.action = 'needs_manual_link_handling'; $item.reason = "reparse ancestor: $anc"; [void]$results.Add($item); Write-Log ("LINK    {0}  ancestor {1}" -f $p, $anc) 'Yellow'; continue }

            $isDir = [bool]($selfAttr -band [System.IO.FileAttributes]::Directory)
            if ($isDir) {
                try { $null = Get-RepoFiles $p } catch { $item.action = 'needs_manual_link_handling'; $item.reason = $_.Exception.Message; [void]$results.Add($item); Write-Log ("LINK    {0}  {1}" -f $p, $item.reason) 'Yellow'; continue }
            }

            if (-not $SkipReferenceScan) {
                $hits = Test-Referenced -Path $p -ScanRoots $scanRoots
                $refHits = Test-ReferencedInS4Refs -Path $p -RefsJsonPath $S4RefsJson
                foreach ($rh in $refHits) { [void]$hits.Add($rh) }
                if ($hits.Count -gt 0) {
                    $item.action = 'referenced_active'
                    $item.reason = ("active code/config refs: " + ($hits -join '; '))
                    [void]$results.Add($item); Write-Log ("REFER   {0}  ({1} file(s))" -f $p, $hits.Count) 'Yellow'; continue
                }
            }

            # ---- local integrity vs receipt manifest ----
            $manifestRel = [string]$rec.md5_manifest
            $manifestPath = Join-Path $WorkspaceRoot ($manifestRel -replace '/', '\')
            $man = Read-Md5Manifest -ManifestPath $manifestPath
            if ($null -eq $man) { $item.action = 'receipt_invalid'; $item.reason = "missing/unreadable md5 manifest: $manifestRel"; [void]$results.Add($item); Write-Log ("BADREC  {0}" -f $p) 'Red'; continue }

            if ($isDir) {
                $localFiles = Get-RepoFiles $p
                $localRel = @{}
                $prefix = ''
                if ([string]$rec.kind -eq 'dirs') { $prefix = (Split-Path -Leaf (Get-Norm $p)) + '/' }
                foreach ($lf in $localFiles) {
                    $rel = (Get-Norm $lf).Substring((Get-Norm $p).Length).TrimStart('\').Replace('\', '/')
                    $localRel[$prefix + $rel] = $lf
                }
                $expected = @($man.Keys | Where-Object { $_.StartsWith($prefix) })
                if ($localRel.Count -ne $expected.Count) {
                    $item.action = 'local_changed'; $item.reason = "file count local=$($localRel.Count) expected=$($expected.Count)"; [void]$results.Add($item); Write-Log ("CHANGED {0}  {1}" -f $p, $item.reason) 'Red'; continue
                }
                $mismatch = $null
                foreach ($rel in $expected) {
                    if (-not $localRel.ContainsKey($rel)) { $mismatch = "missing local: $rel"; break }
                    $fi = Get-Item -LiteralPath $localRel[$rel] -Force
                    if ([long]$fi.Length -ne [long]$man[$rel].size) { $mismatch = "size changed: $rel"; break }
                    if ((Get-LocalMd5 $localRel[$rel]) -ne $man[$rel].md5) { $mismatch = "md5 changed: $rel"; break }
                }
                if ($mismatch) { $item.action = 'local_changed'; $item.reason = $mismatch; [void]$results.Add($item); Write-Log ("CHANGED {0}  {1}" -f $p, $mismatch) 'Red'; continue }
                $remoteDir = ([string]$rec.drive).TrimEnd('/')
                if ($prefix) { $remoteDir += '/' + $prefix.TrimEnd('/') }
                $rv = Test-RemoteDir -LocalDir $p -RemoteDir $remoteDir
            } else {
                $rel = Split-Path -Leaf $p
                if (-not $man.ContainsKey($rel)) { $item.action = 'receipt_invalid'; $item.reason = "file not in manifest: $rel"; [void]$results.Add($item); Write-Log ("BADREC  {0}" -f $p) 'Red'; continue }
                $fi = Get-Item -LiteralPath $p -Force
                if ([long]$fi.Length -ne [long]$man[$rel].size -or (Get-LocalMd5 $p) -ne $man[$rel].md5) { $item.action = 'local_changed'; $item.reason = 'size/md5 differs from receipt'; [void]$results.Add($item); Write-Log ("CHANGED {0}" -f $p) 'Red'; continue }
                $rv = Test-RemoteFile -LocalFile $p -RemoteFile (([string]$rec.drive).TrimEnd('/') + '/' + $rel)
            }
            if (-not $rv.ok) { $item.action = 'remote_unverified'; $item.reason = $rv.detail; [void]$results.Add($item); Write-Log ("REMOTE  {0}  {1}" -f $p, $rv.detail) 'Red'; continue }

            # ---- eligible ----
            $size = [long]0
            if ($isDir) { $size = [long](($expected | ForEach-Object { [long]$man[$_].size } | Measure-Object -Sum).Sum) } else { $size = [long](Get-Item -LiteralPath $p -Force).Length }
            $item.bytes = $size
            Write-Journal @{ event = 'before_delete'; path = $p; kind = $(if ($isDir) { 'dir' } else { 'file' }); bytes = $size }
            if ($Execute) {
                if ($isDir) { $actual = Remove-TreeNoFollow -Root $p } else { $actual = [long](Get-Item -LiteralPath $p -Force).Length; [System.IO.File]::Delete($p) }
                $bytesDeleted += $actual
                $item.action = 'deleted'; $item.reason = ''
                [void]$results.Add($item)
                Write-Journal @{ event = 'after_delete'; path = $p; status = 'ok'; bytes_deleted = $actual }
                Write-Log ("DELETED {0}  ({1} bytes)" -f $p, $actual) 'Green'
            } else {
                $item.action = 'would_delete'; $item.reason = ''
                [void]$results.Add($item)
                Write-Log ("WOULD   {0}  ({1} bytes)" -f $p, $size) 'Green'
            }
        } catch {
            $item.action = 'error'; $item.reason = $_.Exception.Message
            [void]$results.Add($item)
            Write-Log ("ERROR   {0}  {1}" -f $p, $_.Exception.Message) 'Red'
        }
    }
}

$resultArr = @($results.ToArray())
$resultArr | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ResultPath -Encoding UTF8
Write-Journal @{ event = 'run_end'; bytes_deleted = $bytesDeleted; entries = $resultArr.Count }

$wouldDelete = @($resultArr | Where-Object { $_.action -eq 'would_delete' }).Count
$deleted = @($resultArr | Where-Object { $_.action -eq 'deleted' }).Count
Write-Log ("result manifest: {0}" -f $ResultPath) 'Cyan'
Write-Log ("summary: would_delete={0} deleted={1} total_entries={2}" -f $wouldDelete, $deleted, $resultArr.Count) 'Cyan'
Write-Log ("BYTES DELETED: {0} ({1:N3} GiB) - measured, not estimated" -f $bytesDeleted, ($bytesDeleted / 1073741824)) 'Magenta'
