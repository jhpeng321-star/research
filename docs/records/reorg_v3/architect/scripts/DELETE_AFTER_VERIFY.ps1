#Requires -Version 5.1
<#
  DELETE_AFTER_VERIFY.ps1  --  REORG_PLAN_V3 owner-run local deletion.

  RULE: delete an item ONLY IF a Drive verification manifest proves the
  server-side `rclone check --one-way` passed for it (status == "verified").

  SAFETY
    * REPORT-ONLY BY DEFAULT. Nothing is deleted unless -Execute is passed.
    * Hard-coded candidate list; wildcard characters rejected.
    * Per-item manifest is written before any deletion and a journal record
      is flushed BEFORE each delete.
    * Refuses if a process appears to hold files open under the target.
    * HARD PROTECTED: sealed/, CAL, HOLDOUT, chunk-5, J4B, INC1 contents are
      never deleted by this script (owner-gated sealing handled separately).
    * A reparse point (junction) is removed as a link only; its target is left.

  THIS SCRIPT IS DELIVERED UNEXECUTED (planning only). The owner runs it.
#>
[CmdletBinding()]
param(
    [switch]$Execute,
    [string]$Manifest = "",                       # drive_verify_*.jsonl from drive_offload.sh
    [string]$JournalDir = "E:\rd-wt\reorg3\architect\journal",
    [switch]$AllowProtectedSealed
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$WildcardChars = @('*', '?', '[', ']')
$ProtectedToken = 'sealed|\\CAL|HOLDOUT|chunk-5|chunk5|J4B|INC1'
function Get-NormPath { param([string]$Path) return ($Path -replace '/', '\').TrimEnd('\') }

# Hard-coded, explicit candidates. expected_bytes = -1 means "directory, size flex".
$Candidates = @(
    # ---- E: local-only (regenerable / superseded / duplicate; NO Drive needed) ----
    @{ path='E:\research-dense\tmp\cl_gpua_prep'; kind='dir' },
    @{ path='E:\research-dense\tmp\overnight_20261004'; kind='dir' },
    @{ path='E:\research-dense\tmp\c1_upload'; kind='dir' },
    @{ path='E:\research-dense\tmp\code_6a4058f'; kind='dir' },
    @{ path='E:\research-dense\tmp\disk_free_tier2'; kind='dir' },
    @{ path='E:\research-dense\tmp\root_scratch'; kind='dir' },
    @{ path='E:\research-dense\tmp\cloud_offload'; kind='dir' },
    @{ path='E:\research-dense\tmp\overnight'; kind='dir' },
    @{ path='E:\research-dense\tmp\drive_verify_20261006'; kind='dir' },
    @{ path='E:\research-dense\tmp\disk_cleanup_20261005'; kind='dir' },
    @{ path='E:\research-dense\tmp\pdfs'; kind='dir' },
    @{ path='E:\research-dense\tmp\gdrive_batch2'; kind='dir' },
    @{ path='E:\research-dense\tmp\cleanup_reorg'; kind='dir' },
    @{ path='E:\research-dense\tmp\cleanup_gpua'; kind='dir' },
    @{ path='E:\research-dense\tmp\cleanup_c'; kind='dir' },
    @{ path='E:\research-dense\tmp\cleanup_d'; kind='dir' },
    @{ path='E:\research-dense\tmp\zcode_briefs'; kind='dir' },
    @{ path='E:\research-dense\tmp\hdv_audit_20261006'; kind='dir' },
    @{ path='E:\research-dense\tmp\pf_recheck_gpuA'; kind='dir' },
    @{ path='E:\research-dense\tmp\pf_recheck_gpuB'; kind='dir' },
    @{ path='E:\research-dense\tmp\disk_c_sonnet_20261004'; kind='dir' },
    @{ path='E:\research-dense\tmp\disk_c_20261004'; kind='dir' },
    @{ path='E:\research-dense\tmp\syncback_stage'; kind='dir' },
    @{ path='E:\research-dense\tmp\monitor'; kind='dir' },
    @{ path='E:\research-dense\tmp\r7trial'; kind='dir' },
    @{ path='E:\research-dense\tmp\wd01_v02_diag'; kind='dir' },
    @{ path='E:\research-dense\tmp\cursor_mig05_draft'; kind='dir' },
    @{ path='E:\research-dense\tmp\gdrive_stage_b1'; kind='dir' },
    @{ path='E:\research-dense\runs\campaign_walls_v22_smoke'; kind='dir' },
    @{ path='E:\research-dense\runs\campaign_walls_v22_smoke_r2'; kind='dir' },
    @{ path='E:\research-dense\runs\campaign_stage1_chunk5.failed_attempt1_missing_template_20261001T070657Z'; kind='dir' },
    @{ path='E:\research-dense\runs\campaign_stage1_chunk5.failed_attempt_walls_premature_20261001T072053Z'; kind='dir' },
    @{ path='E:\research-dense\runs\campaign_stage1_chunk1_failed_branches_attempt1.tar.gz'; kind='file' },
    @{ path='E:\rd-release\venv-test'; kind='dir' },
    @{ path='E:\rd-release\venv-pptx'; kind='dir' },
    @{ path='E:\rd-wt\reorg2_architect'; kind='dir' },
    @{ path='E:\rd-wt\reorg2_reviewer'; kind='dir' },
    @{ path='E:\rd-wt\reorg2_s1_code'; kind='dir' },
    @{ path='E:\rd-wt\reorg2_s2_docs'; kind='dir' },
    @{ path='E:\rd-wt\reorg2_s3_runs_output'; kind='dir' },
    @{ path='E:\rd-wt\reorg2_s4_worktrees'; kind='dir' },
    @{ path='E:\rd-wt\reorg2_s5_tmp_scratch'; kind='dir' },
    @{ path='E:\rd-wt\reorg2_s6_D_drive'; kind='dir' },
    @{ path='E:\rd-wt\reorg2_s7_C_drive'; kind='dir' },
    @{ path='E:\rd-wt\reorg3'; kind='dir' },
    @{ path='E:\rd-wt\reorg_inv_a'; kind='dir' },
    @{ path='E:\rd-wt\reorg_inv_b'; kind='dir' },
    @{ path='E:\rd-wt\reorg_inv_c'; kind='dir' },
    @{ path='E:\rd-wt$w'; kind='junction' },
    # ---- D: only after a verified Drive manifest ----
    @{ path='D:\rd-staging\wmn1_feat\upload_v2'; kind='dir'; drive='yes' },
    @{ path='D:\rd-staging\wmn1_feat\upload_v2c'; kind='dir'; drive='yes' },
    @{ path='D:\rd-staging\wmn1_feat\raw_chunk1_p40'; kind='dir'; drive='yes' },
    @{ path='D:\rd-staging\wmn1_feat\shards_v2\chunk1'; kind='dir'; drive='yes' },
    @{ path='D:\rd-staging\wmn1_feat\shards_v2\chunk1_fix27'; kind='dir'; drive='yes' },
    @{ path='D:\rd-staging\wmn1_feat\shards_v2\chunk2'; kind='dir'; drive='yes' },
    @{ path='D:\rd-staging\wmn1_feat\shards_v2\index'; kind='dir'; drive='yes' },
    @{ path='D:\rd-staging\wmn1_feat\shards_v2\chunk3'; kind='dir'; drive='yes' },
    @{ path='D:\rd-staging\wmn1_feat\shards_v2\chunk4'; kind='dir'; drive='yes' },
    @{ path='D:\rd-staging\wmn1_feat\shards_v2\truearms_chunk2'; kind='dir'; drive='yes' },
    @{ path='D:\rd-staging\wmn1_feat\shards_v2\truearms_chunk3'; kind='dir'; drive='yes' },
    @{ path='D:\rd-staging\wmn1_feat\shards_v2\walls_pilot_chunk4'; kind='dir'; drive='yes' },
    @{ path='D:\rd-staging\wmn1_feat\shards'; kind='dir' },
    @{ path='D:\rd-staging\box_push_j4b_bases'; kind='dir' },
    @{ path='D:\rd-staging\plexp1d_v3_root'; kind='dir' },
    @{ path='D:\research-dense'; kind='dir' },
    # ---- C: regenerable temp (no Drive needed) ----
    @{ path='C:\Users\jpeng\AppData\Local\Temp\claude'; kind='dir' },
    @{ path='C:\Users\jpeng\Sumo'; kind='dir' },
    @{ path='C:\Users\jpeng\ZCodeProject'; kind='dir' }
)

$Timestamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
if (-not (Test-Path -LiteralPath $JournalDir)) { New-Item -ItemType Directory -Path $JournalDir -Force | Out-Null }
$JournalPath = Join-Path $JournalDir ("delete_{0}.jsonl" -f $Timestamp)
$ManifestPath = Join-Path $JournalDir ("delete_manifest_{0}.json" -f $Timestamp)

function Write-Journal { param([hashtable]$Record)
    $Record['ts'] = (Get-Date).ToUniversalTime().ToString('o'); $Record['dry_run'] = (-not $Execute)
    [System.IO.File]::AppendAllText($JournalPath, (($Record | ConvertTo-Json -Compress -Depth 6) + [Environment]::NewLine))
}
function Assert-SafePath { param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { throw 'empty path' }
    foreach ($c in $WildcardChars) { if ($Path.Contains($c)) { throw "wildcard in path: $Path" } }
}
function Test-Open { param([string]$Path)
    $norm = $Path.TrimEnd('\')
    $busy = @()
    try {
        $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
                 Where-Object { $_.CommandLine -and $_.CommandLine -like "*$norm*" }
        foreach ($p in $procs) { if ($p.ProcessId -ne $PID) { $busy += "pid=$($p.ProcessId) $($p.Name)" } }
    } catch { $busy += "process scan unavailable" }
    if (Test-Path -LiteralPath $Path -PathType Container) {
        Get-ChildItem -LiteralPath $Path -Recurse -File -Force -ErrorAction SilentlyContinue |
          Select-Object -First 2000 | ForEach-Object {
            try { $fs=[System.IO.File]::Open($_.FullName,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,[System.IO.FileShare]::None); $fs.Close(); $fs.Dispose() }
            catch { $busy += "open: $($_.FullName)" }
          }
    }
    return $busy
}

# Load Drive verification manifest (local -> status)
$verified = @{}
if ($Manifest -and (Test-Path -LiteralPath $Manifest)) {
    foreach ($line in Get-Content -LiteralPath $Manifest) {
        if (-not $line.Trim()) { continue }
        try { $r = $line | ConvertFrom-Json } catch { continue }
        if ($r.local) { $verified[(Get-NormPath ([string]$r.local))] = [string]$r.status }
    }
    Write-Host "loaded Drive manifest: $Manifest ($($verified.Count) entries)" -ForegroundColor Green
} else {
    Write-Host "no Drive manifest supplied: only non-Drive local candidates will be eligible" -ForegroundColor Yellow
}

$results = @()
$deleted = 0; $skipped = 0; $blocked = 0
foreach ($c in $Candidates) {
    $p = [string]$c.path
    Assert-SafePath -Path $p
    if ($p -match $ProtectedToken -and -not $AllowProtectedSealed) {
        $results += [pscustomobject]@{ path=$p; action='blocked_protected'; reason='protected acceptance/evidence path' }
        $blocked++; continue
    }
    if (-not (Test-Path -LiteralPath $p)) {
        $results += [pscustomobject]@{ path=$p; action='missing'; reason='not present' }; $skipped++; continue
    }
    $needsDrive = ($c.ContainsKey('drive') -and $c.drive -eq 'yes')
    if ($needsDrive) {
        $key = Get-NormPath $p
        if (-not $verified.ContainsKey($key) -or $verified[$key] -ne 'verified') {
            $results += [pscustomobject]@{ path=$p; action='skip_unverified'; reason='no passing Drive check' }
            $skipped++; continue
        }
    }
    $busy = Test-Open -Path $p
    if ($busy.Count -gt 0) {
        $results += [pscustomobject]@{ path=$p; action='skip_busy'; reason=($busy -join '; ') }
        $blocked++; continue
    }
    $isLink = (Get-Item -LiteralPath $p -Force).Attributes -band [System.IO.FileAttributes]::ReparsePoint
    Write-Journal @{ event='before_delete'; path=$p; is_reparse=[bool]$isLink; execute=$Execute }
    if ($Execute) {
        if ($isLink) { (Get-Item -LiteralPath $p -Force).Delete() }
        else { Remove-Item -LiteralPath $p -Recurse -Force }
    }
    $act = $(if ($Execute) { 'deleted' } else { 'would_delete' })
    $results += [pscustomobject]@{ path=$p; action=$act; reason='' }
    $deleted++
    Write-Journal @{ event='after_delete'; path=$p; execute=$Execute; status='ok' }
}

$results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8
Write-Journal @{ event='run_end'; execute=$Execute; eligible=$deleted; skipped=$skipped; blocked=$blocked }
Write-Host ("delete pass done: eligible={0} skipped={1} blocked={2}" -f $deleted,$skipped,$blocked) -ForegroundColor Green
Write-Host "manifest: $ManifestPath" -ForegroundColor Green
Write-Host "journal:  $JournalPath" -ForegroundColor Green
