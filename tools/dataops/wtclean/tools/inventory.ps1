param(
  [string]$Root = "E:\rd-wt",
  [string]$OutDir = "E:\rd-wt\reorg3\wtclean\inventory",
  [string]$ListDir = "E:\rd-wt\reorg3\wtclean\lists"
)

$ErrorActionPreference = "Continue"
$active = @(
  'ws-r3-int','ws-r3-base','ws-r3base','ws-r3-resid','ws-r3-hist','ws-r3-jam',
  'ws-r3-stage','ws-r3-score','ws-r3-evalstates','ws-wallc1','ws-main2','ws-cl',
  'ws-p2','reorg3'
)
# sensitive token boundary regex (skips contents, recorded, never uploaded)
$sensitiveRegex = '(?i)(^|[\\/_.\-\s])(sealed|holdout|chunk-?5|inc1|j4b|cal)([\\/_.\-\s]|$)'

function Get-ReparsePaths {
  param([string]$Start)
  $res = New-Object System.Collections.Generic.List[string]
  $stack = New-Object System.Collections.Stack
  $stack.Push($Start)
  while ($stack.Count -gt 0) {
    $cur = $stack.Pop()
    try { $entries = [System.IO.Directory]::GetFileSystemEntries($cur) } catch { continue }
    foreach ($e in $entries) {
      try { $a = [System.IO.File]::GetAttributes($e) } catch { continue }
      if ($a -band [IO.FileAttributes]::ReparsePoint) {
        $res.Add($e)  # do NOT descend into reparse target
      } elseif ($a -band [IO.FileAttributes]::Directory) {
        $stack.Push($e)
      }
    }
  }
  return $res
}

function RelPath([string]$base, [string]$full) {
  $b = $base.TrimEnd('\','/') + '\'
  if ($full.StartsWith($b, [StringComparison]::OrdinalIgnoreCase)) { return $full.Substring($b.Length) }
  return $full
}

if (-not (Test-Path -LiteralPath $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }
if (-not (Test-Path -LiteralPath $ListDir)) { New-Item -ItemType Directory -Force -Path $ListDir | Out-Null }

$dirs = Get-ChildItem -LiteralPath $Root -Force -Directory | Sort-Object Name
$records = New-Object System.Collections.Generic.List[object]
$i = 0
foreach ($d in $dirs) {
  $i++
  $name = $d.Name
  $rec = [ordered]@{
    name = $name
    path = $d.FullName
    active = $false
    class = $null
    topReparse = $false
    topTarget = $null
    gitKind = $null
    branch = $null
    head = $null
    trackedDirtyCount = 0
    untrackedCount = 0
    ignoredCount = 0
    uploadCount = 0
    uploadBytes = 0
    listFile = $null
    junctions = @()
    skippedSensitive = @()
    skippedJunctionPaths = @()
    notes = @()
  }

  if ($active -contains $name -or $name -eq 'reorg3') {
    $rec.active = $true
    $rec.class = 'active'
    $records.Add([pscustomobject]$rec)
    Write-Host "[$i/$($dirs.Count)] ACTIVE   $name"
    continue
  }

  $di = Get-Item -LiteralPath $d.FullName -Force
  if ($di.Attributes -band [IO.FileAttributes]::ReparsePoint) {
    $rec.topReparse = $true
    $rec.class = 'junction'
    $rec.topTarget = [string]($di.Target)
    $records.Add([pscustomobject]$rec)
    Write-Host "[$i/$($dirs.Count)] JUNCTION $name -> $($rec.topTarget)"
    continue
  }

  # scan junctions inside (non-following)
  $reps = Get-ReparsePaths -Start $d.FullName
  $rec.junctions = @($reps | ForEach-Object { RelPath $d.FullName $_ })
  $junctionPrefixes = @($rec.junctions | ForEach-Object { ($_ -replace '/','\').TrimEnd('\') })

  $hasGit = Test-Path -LiteralPath (Join-Path $d.FullName '.git')
  $uploadFiles = New-Object System.Collections.Generic.List[string]
  $skippedSens = New-Object System.Collections.Generic.List[string]
  $skippedJun = New-Object System.Collections.Generic.List[string]

  function Test-UnderJunction([string]$rel) {
    $r = ($rel -replace '/','\')
    foreach ($jp in $junctionPrefixes) {
      if ($r.Equals($jp, [StringComparison]::OrdinalIgnoreCase)) { return $true }
      if ($r.StartsWith($jp + '\', [StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    return $false
  }

  if ($hasGit) {
    $gi = Get-Item -LiteralPath (Join-Path $d.FullName '.git') -Force
    if ($gi.PSIsContainer) { $rec.gitKind = 'gitdir' } else { $rec.gitKind = 'worktree' }
    $rec.branch = (git -C $d.FullName rev-parse --abbrev-ref HEAD 2>$null | Select-Object -First 1)
    $rec.head = (git -C $d.FullName rev-parse HEAD 2>$null | Select-Object -First 1)
    $porc = @(git -C $d.FullName status --porcelain 2>$null)
    $rec.trackedDirtyCount = @($porc | Where-Object { $_ -and -not $_.StartsWith('??') }).Count
    $untracked = @(git -C $d.FullName ls-files --others --exclude-standard 2>$null)
    $ignored = @(git -C $d.FullName ls-files --others --ignored --exclude-standard 2>$null)
    $rec.untrackedCount = $untracked.Count
    $rec.ignoredCount = $ignored.Count
    $cand = @($untracked + $ignored) | Sort-Object -Unique
    $rec.class = if ($rec.gitKind -eq 'worktree') { 'worktree' } else { 'gitdir' }
  } else {
    $rec.gitKind = 'none'
    $rec.class = 'plaindir'
    $cand = New-Object System.Collections.Generic.List[string]
    $pstack = New-Object System.Collections.Stack
    $pstack.Push($d.FullName)
    while ($pstack.Count -gt 0) {
      $pcur = $pstack.Pop()
      try { $pent = [System.IO.Directory]::GetFileSystemEntries($pcur) } catch { continue }
      foreach ($pe in $pent) {
        try { $pa = [System.IO.File]::GetAttributes($pe) } catch { continue }
        if ($pa -band [IO.FileAttributes]::ReparsePoint) { continue }
        if ($pa -band [IO.FileAttributes]::Directory) { $pstack.Push($pe) }
        else { $cand.Add((RelPath $d.FullName $pe)) }
      }
    }
  }

  foreach ($c in $cand) {
    if (-not $c) { continue }
    $rel = ($c -replace '\\','/')
    if ($rel -match '(^|/)\.git(/|$)') { continue }
    if (Test-UnderJunction $rel) { $skippedJun.Add($rel); continue }
    if ($rel -match $sensitiveRegex) { $skippedSens.Add($rel); continue }
    $full = Join-Path $d.FullName ($rel -replace '/','\')
    try { $it = Get-Item -LiteralPath $full -Force -ErrorAction Stop } catch { continue }
    if ($it.Attributes -band [IO.FileAttributes]::ReparsePoint) { $skippedJun.Add($rel); continue }
    if ($it.PSIsContainer) { continue }
    $uploadFiles.Add($rel)
    $rec.uploadCount++
    $rec.uploadBytes += $it.Length
  }

  if ($uploadFiles.Count -gt 0) {
    $lf = Join-Path $ListDir ($name + ".txt")
    [System.IO.File]::WriteAllLines($lf, $uploadFiles)
    $rec.listFile = $lf
  }
  $rec.skippedSensitive = @($skippedSens | Select-Object -First 500)
  $rec.skippedJunctionPaths = @($skippedJun | Select-Object -First 500)

  $records.Add([pscustomobject]$rec)
  Write-Host ("[{0}/{1}] {2,-9} {3,-16} files={4} bytes={5} junc={6} dirty={7}" -f $i,$dirs.Count,$rec.class,$name,$rec.uploadCount,$rec.uploadBytes,$rec.junctions.Count,$rec.trackedDirtyCount)
}

$outFile = Join-Path $OutDir "inventory.json"
$records | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $outFile -Encoding UTF8
Write-Host "WROTE $outFile  records=$($records.Count)"
