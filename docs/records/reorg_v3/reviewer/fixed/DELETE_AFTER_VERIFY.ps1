#Requires -Version 5.1
<#
  Conservative review copy: destructive execution is disabled.
  V3 receipts do not bind a current source snapshot, archive membership, owner
  authorization, retained references and a no-writer window. Mechanical fixes
  alone cannot establish those facts. See ../REVIEW_V3.md findings 2-6.
  The original candidate list is intentionally not treated as authorization.
  No input is read and no report/journal directory is created by this copy.
#>
[CmdletBinding()]
param(
    [switch]$Execute,
    [string]$Manifest = '',
    [string]$JournalDir = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($Execute) {
    throw 'Deletion disabled: owner must resolve REVIEW_V3.md blockers and obtain snapshot-bound checksum receipts first.'
}
[pscustomobject]@{
    action = 'blocked_pending_review_fixes'
    eligible_deletions = 0
    reason = 'V3 candidate list and status-only manifests are insufficient deletion evidence.'
    review = (Join-Path $PSScriptRoot '..\REVIEW_V3.md')
}
